"""Focused tests for economic TD3 reward and feasible action mapping."""
import unittest
from unittest.mock import patch
import gymnasium as gym
import numpy as np
from .economic_action_wrappers import FeasibleIntervalActionWrapper
from .economic_ems import EconomicEMSTD3Reward, REWARD_COMPONENT_FIELDS, load_economic_ems_config, make_economic_env

class FakePhysicalEnv(gym.Env):
    metadata = {"render_modes": []}
    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32)
        self.episode = type("Episode", (), {"load_kw": np.array([400.0])})()
        self.index, self.terminated = 0, False
        self.physical = type("Physical", (), {"soc": 0.5})()
        self.reward_identity = {"reward_version": "test_physical_v1"}
    def _physically_feasible_fc_range(self, load_kw): return 250.0, 500.0
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed); self.index = 0; self.terminated = False
        return np.array([0.0], dtype=np.float32), {}
    def step(self, action):
        power = 300.0 * (float(action[0]) + 1.0)
        self.index += 1; self.terminated = True
        info = {
            "requested_fc_kw": power, "fc_actual_kw": power, "battery_actual_kw": 400.0-power,
            "load_kw": 400.0, "soc": 0.5, "step_executed": True,
            "terminal_reason": "completed", "reward_is_failure_override": False,
            "h2_cost_cny": 0.10, "fuel_cell_transient_cost_cny": 0.01,
            "fuel_cell_low_load_cost_cny": 0.02, "fuel_cell_high_load_cost_cny": 0.0,
            "fuel_cell_start_stop_cost_cny": 0.03, "c_bat_eq_cny": 0.04,
            "battery_degradation_cost_cny": 0.05, "total_cost_cny": 0.25,
            "vessel_episode": {"completed": True, "sample_id": "fake", "split": "train",
                               "soc_initial": 0.5, "soc_delta": 0.0},
        }
        return np.array([0.0], dtype=np.float32), 0.0, True, False, info

class EconomicEMSTests(unittest.TestCase):
    def test_config(self):
        config = load_economic_ems_config()
        self.assertEqual(config["reward_scale"], 0.001)
        self.assertEqual(config["soc_reference"], 0.5)
        self.assertEqual(config["soc_terminal_weight_cny"], 20000.0)
    def test_feasible_interval_mapping(self):
        env = FeasibleIntervalActionWrapper(FakePhysicalEnv()); env.reset(seed=42)
        for action, expected in ((-1.0, 250.0), (0.0, 375.0), (1.0, 500.0)):
            inner, meta = env._map(np.array([action], dtype=np.float32))
            actual = 300.0 * (float(inner[0]) + 1.0)
            self.assertAlmostEqual(actual, expected, places=4)
            self.assertAlmostEqual(meta["action_mapped_fc_kw"], expected, places=6)
    def test_economic_reward_and_start_stop_scope(self):
        env = EconomicEMSTD3Reward(FakePhysicalEnv()); env.reset(seed=42)
        _, reward, terminated, truncated, info = env.step(np.array([0.0], dtype=np.float32))
        self.assertTrue(terminated); self.assertFalse(truncated)
        self.assertAlmostEqual(info["reward_economic_cost_cny"], 0.22)
        self.assertAlmostEqual(info["reward_total_cost_cny"], 0.22)
        self.assertAlmostEqual(reward, -0.00022)
        self.assertAlmostEqual(info["total_cost_cny"], 0.25)
        self.assertEqual(info["reward_terminal_soc_penalty_cny"], 0.0)
        self.assertEqual(len(REWARD_COMPONENT_FIELDS), 7)

    def test_train_factory_never_randomizes_voyage(self):
        base = FakePhysicalEnv()
        base.episodes = [type("Voyage", (), {
            "sample_id": "parent_005_seg01", "load_kw": np.zeros(9258),
        })()]
        with patch(f"{__package__}.economic_ems.AggregateDDPGEnv", return_value=base) as factory:
            env = make_economic_env("data", split="train")
            factory.assert_called_once_with(data_root="data", split="train", fixed_train_id="parent_005_seg01")
            with self.assertRaisesRegex(ValueError, "parent_005_seg01"):
                make_economic_env("data", split="train", fixed_train_id="parent_001_seg01")
        self.assertIn("parent_005_seg01", env.reward_identity["training_selection"])

    def test_terminal_soc_uses_signed_cost_and_penalizes_only_success(self):
        base = FakePhysicalEnv()
        env = EconomicEMSTD3Reward(base)
        env.reset(seed=42)
        _, _, _, _, info = base.step(np.array([0.0], dtype=np.float32))
        info.update(soc=0.2, c_bat_eq_cny=-0.04)
        base.physical.soc = 0.2
        with patch.object(base, "step", return_value=(np.zeros(1), 0.0, True, False, info)):
            _, reward, _, _, result = env.step(np.zeros(1))
        self.assertAlmostEqual(result["reward_battery_equivalent_hydrogen_cost_cny"], -0.04)
        self.assertAlmostEqual(result["reward_soc_tracking_penalty_cny"], 0.045)
        self.assertAlmostEqual(result["reward_terminal_soc_penalty_cny"], 1800.0)
        self.assertAlmostEqual(reward, -1.800185)
        self.assertEqual(result["total_cost_cny"], info["total_cost_cny"])

        env.reset(seed=42)
        info.update(step_executed=False, reward_is_failure_override=True,
                    terminal_reason="no_feasible_physical_power")
        with patch.object(base, "step", return_value=(np.zeros(1), 0.0, True, False, info)):
            _, reward, _, _, result = env.step(np.zeros(1))
        self.assertEqual(reward, -20.0)
        self.assertEqual(result["reward_terminal_soc_penalty_cny"], 0.0)
        self.assertEqual(result["reward_soc_tracking_penalty_cny"], 0.0)

if __name__ == "__main__": unittest.main()
