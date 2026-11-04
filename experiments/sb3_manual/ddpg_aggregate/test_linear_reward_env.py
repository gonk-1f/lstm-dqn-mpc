"""Real one-step reward/termination checks; no model learning or full voyage."""
from __future__ import annotations

from pathlib import Path
import math
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from .env import AggregateDDPGEnv, ECONOMIC_SUM_FIELDS
from v3.control import AccountState, SOC_MIN


class NegativeEconomicRewardEnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = AggregateDDPGEnv(ROOT / "data/processed/operating_dataset_v6_1s",
                                   fixed_train_id="parent_005_seg01")
        cls.addClassCleanup(cls.env.close)

    def setUp(self):
        self.env.reset(seed=42)

    def check_normal_reward(self, reward, info):
        cost = math.fsum(info[key] for key in ("h2_cost_cny", "fuel_cell_degradation_cost_cny",
                                              "battery_degradation_cost_cny", "c_bat_eq_cny"))
        self.assertAlmostEqual(info["total_cost_cny"], cost)
        self.assertNotIn("c_soc_cny", info)
        self.assertNotIn("soc_penalty", info)
        reward_cost = math.fsum(info[key] for key in (
            "h2_cost_cny", "c_bat_eq_cny", "battery_degradation_cost_cny",
            "fuel_cell_transient_cost_cny", "fuel_cell_low_load_cost_cny",
            "fuel_cell_high_load_cost_cny",
        ))
        self.assertAlmostEqual(info["reward_total_cost_cny"], reward_cost)
        self.assertAlmostEqual(cost, reward_cost + info["fuel_cell_start_stop_cost_cny"])
        self.assertEqual(reward, -info["reward_total_cost_cny"])

    def test_normal_step_excludes_start_stop_from_reward_but_not_ledger(self):
        _, reward, terminated, truncated, info = self.env.step([0.0])
        self.assertTrue(info["step_executed"])
        self.assertEqual(info["fc_actual_kw"], 300.0)
        self.assertEqual(info["delta_fc_kw"], 300.0)
        self.assertIsNone(self.env.reward_identity["fc_ramp_limit_kw_per_s"])
        self.assertFalse(terminated or truncated)
        self.assertGreater(info["fuel_cell_start_stop_cost_cny"], 0.0)
        self.check_normal_reward(reward, info)
        self.assertEqual(self.env.index, 1)
        self.assertEqual(self.env.economic_totals["total_cost_cny"], info["total_cost_cny"])

    def test_idle_at_initial_soc_has_zero_cost_and_reward_zero(self):
        # The first fixed-voyage load is the recorded zero.
        _, reward, terminated, truncated, info = self.env.step([-1.0])
        self.assertEqual(info["load_kw"], 0.0)
        self.assertEqual(info["soc"], 0.6)
        self.assertEqual(info["total_cost_cny"], 0.0)
        self.assertEqual(reward, 0.0)
        self.assertFalse(terminated or truncated)
        self.check_normal_reward(reward, info)

    def test_normal_terminal_step_has_no_extra_terminal_reward(self):
        self.env.index = len(self.env.episode.load_kw) - 1
        self.env.physical = AccountState(soc=0.6, previous_fc_kw=300.0)
        _, reward, terminated, truncated, info = self.env.step([0.0])
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["vessel_episode"]["completed"])
        self.check_normal_reward(reward, info)

    def test_infeasible_step_is_minus_2000_without_physics_or_compensation(self):
        self.env.index = int(np.flatnonzero(self.env.episode.load_kw > self.env.plant.fuel_cell_rated_total_kw)[0])
        self.env.physical = AccountState(soc=SOC_MIN, previous_fc_kw=0.0)
        index = self.env.index
        state = self.env.physical
        with patch.object(self.env.physics, "interval", side_effect=AssertionError("Unexecuted physics")), \
             patch("experiments.sb3_manual.ddpg_aggregate.env.economic_reward",
                   side_effect=AssertionError("No failed-step economic reward or compensation")):
            _, reward, terminated, truncated, info = self.env.step([0.0])
        self.assertEqual(reward, -2000.0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertFalse(info["step_executed"])
        self.assertEqual(info["terminal_reason"], "no_feasible_physical_power")
        self.assertNotIn("battery_actual_kw", info)
        self.assertIs(self.env.physical, state)
        self.assertEqual(self.env.index, index)
        self.assertTrue(all(self.env.economic_totals[key] == 0 for key in ECONOMIC_SUM_FIELDS))
        self.assertEqual(info["vessel_episode"]["reward_sum"], -2000.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
