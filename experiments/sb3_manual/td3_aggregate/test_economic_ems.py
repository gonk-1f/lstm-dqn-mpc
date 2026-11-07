"""Focused tests for economic TD3 reward and feasible action mapping."""
import unittest
import gymnasium as gym
import numpy as np
from .economic_action_wrappers import FeasibleIntervalActionWrapper
from .economic_ems import EconomicEMSTD3Reward, REWARD_COMPONENT_FIELDS, load_economic_ems_config

class FakePhysicalEnv(gym.Env):
    metadata = {"render_modes": []}
    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
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

if __name__ == "__main__": unittest.main()