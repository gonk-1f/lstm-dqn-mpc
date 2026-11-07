"""Focused tests for economic TD3 reward and feasible action mapping."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch
import gymnasium as gym
import numpy as np
from .economic_action_wrappers import FeasibleIntervalActionWrapper
from .economic_ems import EconomicEMSTD3Reward, REWARD_COMPONENT_FIELDS, load_economic_ems_config, make_economic_env
from .economic_ems_callbacks import EconomicEMSTrainingCallback, TD3TrainingCallback

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
        self.assertEqual(config["soc_step_weight_cny"], 5.0)
        self.assertNotIn("soc_terminal_weight_cny", config)
        self.assertNotIn("battery_equivalent_hydrogen_weight", config)
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
        self.assertAlmostEqual(info["reward_economic_cost_cny"], 0.18)
        self.assertAlmostEqual(info["reward_total_cost_cny"], 0.18)
        self.assertAlmostEqual(reward, -0.00018)
        self.assertAlmostEqual(info["total_cost_cny"], 0.25)
        self.assertAlmostEqual(info["c_bat_eq_cny"], 0.04)
        self.assertAlmostEqual(info["fuel_cell_start_stop_cost_cny"], 0.03)
        self.assertNotIn("reward_battery_equivalent_hydrogen_cost_cny", info)
        self.assertNotIn("reward_terminal_soc_penalty_cny", info)
        self.assertEqual(len(REWARD_COMPONENT_FIELDS), 5)

    def test_train_factory_defaults_to_random_and_preserves_the_fixed_diagnostic(self):
        base = FakePhysicalEnv()
        base.episodes = [type("Voyage", (), {
            "sample_id": "parent_005_seg01", "load_kw": np.zeros(9258),
        })()]
        with patch(f"{__package__}.economic_ems.AggregateDDPGEnv", return_value=base) as factory:
            env = make_economic_env("data", split="train")
            factory.assert_called_once_with(data_root="data", split="train", fixed_train_id=None)
            fixed = make_economic_env("data", split="train", fixed_train_id="parent_005_seg01")
            factory.assert_called_with(data_root="data", split="train", fixed_train_id="parent_005_seg01")
            with self.assertRaisesRegex(ValueError, "parent_005_seg01"):
                make_economic_env("data", split="train", fixed_train_id="parent_001_seg01")
        self.assertIn("Train35", env.reward_identity["training_selection"])
        self.assertEqual(env.reward_identity, fixed.reward_identity)

    def test_callback_checks_the_sample_specific_length(self):
        callback = EconomicEMSTrainingCallback()
        self.assertFalse(callback.require_fixed_train)
        callback._train_steps = {"parent_003_seg01": 21480, "parent_005_seg01": 9258}
        summary = {
            "sample_id": "parent_003_seg01", "total_steps": 21480, "executed_steps": 21480,
            "soc_initial": 0.6, "terminal_reason": "completed", "completed": True,
        }
        callback._episode_counts["raw_reward"] = 21480
        with patch.object(TD3TrainingCallback, "_record_episode") as record:
            callback._record_episode(summary)
            record.assert_called_once_with(summary)
            with self.assertRaisesRegex(ValueError, "21480"):
                callback._record_episode({**summary, "total_steps": 9258})

    def test_final_step_keeps_signed_equivalent_hydrogen_only_in_the_ledger(self):
        base = FakePhysicalEnv()
        env = EconomicEMSTD3Reward(base)
        env.reset(seed=42)
        _, _, _, _, info = base.step(np.array([0.0], dtype=np.float32))
        info.update(soc=0.2, c_bat_eq_cny=-0.04, total_cost_cny=0.17)
        base.physical.soc = 0.2
        with patch.object(base, "step", return_value=(np.zeros(1), 0.0, True, False, info)):
            _, reward, _, _, result = env.step(np.zeros(1))
        self.assertAlmostEqual(result["c_bat_eq_cny"], -0.04)
        self.assertNotIn("reward_battery_equivalent_hydrogen_cost_cny", result)
        self.assertAlmostEqual(result["reward_soc_tracking_penalty_cny"], 0.45)
        self.assertNotIn("reward_terminal_soc_penalty_cny", result)
        self.assertAlmostEqual(reward, -0.00063)
        self.assertEqual(result["total_cost_cny"], info["total_cost_cny"])

        env.reset(seed=42)
        info.update(step_executed=False, reward_is_failure_override=True,
                    terminal_reason="no_feasible_physical_power")
        with patch.object(base, "step", return_value=(np.zeros(1), 0.0, True, False, info)):
            _, reward, _, _, result = env.step(np.zeros(1))
        self.assertEqual(reward, -20.0)
        self.assertNotIn("reward_terminal_soc_penalty_cny", result)
        self.assertEqual(result["reward_soc_tracking_penalty_cny"], 0.0)

    def test_pause_request_stops_only_after_storing_the_ending_transition(self):
        callback = EconomicEMSTrainingCallback()
        callback.model = Mock(_episode_num=0)
        callback.completed_episodes = 1
        callback.locals = dict.fromkeys(("infos", "replay_buffer", "buffer_actions", "new_obs", "rewards"))
        with TemporaryDirectory() as folder, patch.object(TD3TrainingCallback, "_on_step", return_value=True):
            callback.pause_request = Path(folder) / "pause_after_episode.request"
            callback.pause_request.touch()
            callback.locals["dones"] = [False]
            self.assertTrue(callback._on_step())
            callback.model._store_transition.assert_not_called()
            callback.locals["dones"] = [True]
            self.assertFalse(callback._on_step())
        callback.model._store_transition.assert_called_once()
        self.assertEqual(callback.model._episode_num, 1)

if __name__ == "__main__": unittest.main()
