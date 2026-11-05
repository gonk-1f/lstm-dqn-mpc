"""Test4 data adapter checks; no model prediction or physical action is executed."""
from __future__ import annotations

import importlib
import importlib.util
import unittest

import numpy as np

PACKAGE = "experiments.sb3_manual.ddpg_aggregate"
EXPECTED = {
    "parent_001_seg01": 7037,
    "parent_004_seg01": 7288,
    "parent_013_seg01": 10308,
    "parent_032_seg01": 9841,
}


class Test4AdapterTests(unittest.TestCase):
    def adapter_module(self):
        name = PACKAGE + ".test4_env"
        self.assertIsNotNone(importlib.util.find_spec(name), "Independent Test4 adapter is missing")
        return importlib.import_module(name)

    def test_fixed_test4_order_and_complete_one_second_grid(self):
        module = self.adapter_module()
        env = module.AggregateDDPGTestEnv(module.DEFAULT_DATA_ROOT)
        try:
            self.assertEqual(env.split, "test")
            self.assertEqual(env.opened_test_payloads, 4)
            self.assertEqual([episode.sample_id for episode in env.episodes], list(EXPECTED))
            for episode in env.episodes:
                duration = EXPECTED[episode.sample_id]
                self.assertEqual(episode.split, "test")
                self.assertEqual(episode.step_count, duration)
                self.assertTrue(np.array_equal(episode.time_s, np.arange(duration)))
                self.assertTrue(np.array_equal(episode.dt_s, np.ones(duration)))
                self.assertEqual(episode.terminal_boundary_time_s, duration)
                self.assertEqual(len(episode.timestamp), duration)
                self.assertEqual((episode.terminal_boundary_timestamp - episode.timestamp[-1]).total_seconds(), 1)
            for sample_id in list(EXPECTED) + [next(iter(EXPECTED))]:
                _, info = env.reset(seed=42)
                self.assertEqual((info["sample_id"], info["split"]), (sample_id, "test"))
                self.assertTrue(info["physical_fc_on"])
                self.assertEqual(env.index, 0)
                self.assertEqual(env.physical.soc, 0.6)
            payloads = env.test_data_identity["payloads"]
            self.assertEqual([item["sample_id"] for item in payloads], list(EXPECTED))
            self.assertTrue(all(len(item["sha256"]) == 64 for item in payloads))
        finally:
            env.close()

    def test_observation_action_and_physics_methods_are_inherited(self):
        module = self.adapter_module()
        base = module.AggregateDDPGEnv
        adapter = module.AggregateDDPGTestEnv
        for name in ("reset", "step", "_observation", "_physically_feasible_fc_range", "_episode_summary"):
            self.assertIs(getattr(adapter, name), getattr(base, name), name)

    def test_original_environment_still_rejects_test_split(self):
        module = self.adapter_module()
        with self.assertRaisesRegex(ValueError, "Test is not accessible"):
            module.AggregateDDPGEnv(module.DEFAULT_DATA_ROOT, split="test")


if __name__ == "__main__":
    unittest.main()
