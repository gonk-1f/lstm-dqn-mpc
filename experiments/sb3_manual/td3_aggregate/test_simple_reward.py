"""Bounded acceptance checks for the independent quadratic FC/SOC reward."""
from __future__ import annotations

import importlib
import importlib.util
from types import SimpleNamespace
import unittest

import numpy as np

from .test_td3_integration import CachedTrainCase, DATA, FIXED_ID, PACKAGE


class SimpleRewardTests(CachedTrainCase):
    def module(self):
        name = PACKAGE + ".simple_reward"
        self.assertIsNotNone(importlib.util.find_spec(name), "Independent simple reward is missing")
        return importlib.import_module(name)

    def pair(self):
        from experiments.sb3_manual.ddpg_aggregate.env import AggregateDDPGEnv
        raw = AggregateDDPGEnv(DATA, fixed_train_id=FIXED_ID)
        simple = self.module().make_simple_env(DATA, fixed_train_id=FIXED_ID)
        self.addCleanup(raw.close)
        self.addCleanup(simple.close)
        return raw, simple

    def test_historical_coefficients_and_independent_identity(self):
        module = self.module()
        config = module.load_simple_reward_config()
        self.assertEqual(config["lambda_fc"], 0.0025)
        self.assertEqual(config["lambda_soc"], 10.0)
        self.assertEqual(config["soc_reference"], 0.5)
        self.assertEqual(config["fc_variation_power"], 2)
        self.assertEqual(config["historical_source"]["commit"],
                         "49082cdbb45fcbd964420b36e6d0263682783370")
        raw, simple = self.pair()
        identity = simple.reward_identity
        self.assertEqual(identity["reward_version"], module.SIMPLE_REWARD_VERSION)
        self.assertEqual(identity["env_id"], "AggregateShipTD3Simple-v0")
        self.assertEqual(identity["reward_scale"], 0.0001)
        self.assertEqual(identity["physical_accounting_identity"], raw.reward_identity)
        self.assertEqual(simple.unwrapped.reward_identity, raw.reward_identity)

    def test_normal_reward_has_only_hydrogen_quadratic_ramp_and_soc(self):
        _, simple = self.pair()
        simple.reset(seed=42)
        _, reward, terminated, truncated, info = simple.step([0.0])
        self.assertFalse(terminated or truncated)
        self.assertEqual(info["delta_fc_kw"], 300.0)
        self.assertEqual(info["reward_fc_variation_penalty"], 225.0)
        soc_cost = 10.0 * (info["soc"] - 0.5) ** 2
        expected = -(info["h2_cost_cny"] + 225.0 + soc_cost)
        self.assertAlmostEqual(info["raw_reward"], expected)
        self.assertAlmostEqual(reward, expected / 10000.0)
        self.assertEqual(info["reward_h2_cost_cny"], info["h2_cost_cny"])
        self.assertEqual(info["reward_economic_cost_cny"], info["h2_cost_cny"])
        self.assertEqual(info["reward_soc_penalty"], soc_cost)
        self.assertEqual(info["soc_deviation_penalty_cny"], soc_cost)
        self.assertEqual(info["fc_ramp_penalty"], 225.0)
        self.assertEqual(info["failure_reward_total"], 0.0)
        self.assertGreater(info["fuel_cell_degradation_cost_cny"], 0.0)
        self.assertGreater(info["battery_degradation_cost_cny"], 0.0)
        self.assertLess(info["c_bat_eq_cny"], 0.0)

    def test_identical_actions_preserve_real_physics_and_complete_economic_ledger(self):
        raw, simple = self.pair()
        raw_obs, raw_info = raw.reset(seed=42)
        obs, info = simple.reset(seed=42)
        np.testing.assert_array_equal(obs, raw_obs)
        self.assertEqual(info["sample_id"], raw_info["sample_id"])
        ledger_fields = (
            "h2_cost_cny", "hydrogen_mass_kg", "c_bat_eq_cny",
            "fuel_cell_degradation_cost_cny", "battery_degradation_cost_cny",
            "total_cost_cny", "physical_economic_cost_cny", "fc_actual_kw",
            "battery_actual_kw", "battery_current_a", "projected",
        )
        for action in (-1.0, 0.0, 1.0, -0.3, 0.5):
            with self.subTest(action=action):
                raw_obs, _, raw_done, raw_truncated, raw_info = raw.step([action])
                obs, _, done, truncated, info = simple.step([action])
                np.testing.assert_array_equal(obs, raw_obs)
                self.assertEqual((done, truncated), (raw_done, raw_truncated))
                for name in ledger_fields:
                    self.assertEqual(info[name], raw_info[name], name)
                self.assertEqual(simple.unwrapped.physical, raw.physical)
                self.assertEqual(simple.unwrapped.economic_totals, raw.economic_totals)

    def test_real_failure_keeps_fixed_reward_without_fabricated_step_costs(self):
        from v3.control import AccountState, SOC_MIN
        raw, simple = self.pair()
        for env in (raw, simple):
            env.reset(seed=42)
            base = env.unwrapped
            base.index = int(np.flatnonzero(base.episode.load_kw > 600.0)[0])
            base.physical = AccountState(soc=SOC_MIN, previous_fc_kw=300.0)
        _, _, _, _, raw_info = raw.step([0.0])
        _, reward, terminated, truncated, info = simple.step([0.0])
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertFalse(info["step_executed"])
        self.assertEqual((info["raw_reward"], reward), (-20000.0, -2.0))
        for name in ("reward_h2_cost_cny", "reward_fc_variation_penalty",
                     "reward_soc_penalty", "reward_total_cost_cny"):
            self.assertEqual(info[name], 0.0, name)
        self.assertEqual(info["failure_reward_total"], -20000.0)
        summary = info["vessel_episode"]
        self.assertEqual(summary["reward_sum"], -20000.0)
        self.assertEqual(summary["algorithm_reward_sum"], -2.0)
        self.assertEqual(summary["failure_reward_total"], -20000.0)
        self.assertEqual(info["total_cost_cny"], raw_info["total_cost_cny"])
        self.assertGreater(info["fuel_cell_start_stop_cost_cny"], 0.0)

    def test_terminal_shutdown_does_not_add_another_ramp_or_soc_reward(self):
        from v3.control import AccountState
        _, simple = self.pair()
        simple.reset(seed=42)
        rewards = []
        for action in (-1.0, 0.25):
            _, _, _, _, info = simple.step([action])
            rewards.append(info["raw_reward"])
        vessel = simple.unwrapped
        vessel.index = len(vessel.episode.load_kw) - 1
        vessel.physical = AccountState(soc=0.6, previous_fc_kw=300.0)
        _, reward, terminated, truncated, info = simple.step([0.0])
        rewards.append(info["raw_reward"])
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["delta_fc_kw"], 0.0)
        self.assertEqual(info["reward_fc_variation_penalty"], 0.0)
        self.assertTrue(info["shutdown_executed"])
        self.assertEqual(info["shutdown_delta_fc_kw"], -300.0)
        self.assertGreater(info["shutdown_fc_transient_cost_cny"], 0.0)
        self.assertEqual(info["post_shutdown_fc_kw"], 0.0)
        self.assertEqual(info["post_shutdown_battery_kw"], 0.0)
        summary = info["vessel_episode"]
        self.assertAlmostEqual(summary["reward_sum"], sum(rewards))
        self.assertAlmostEqual(summary["algorithm_reward_sum"], sum(rewards) / 10000.0)
        self.assertAlmostEqual(
            summary["reward_total_cost_cny"],
            summary["reward_h2_cost_cny"] + summary["reward_fc_variation_penalty"]
            + summary["reward_soc_penalty"],
        )
        self.assertEqual(summary["reward_version"], self.module().SIMPLE_REWARD_VERSION)
        simple.reset(seed=42)
        _, reward, _, _, info = simple.step([0.0])
        self.assertEqual(simple.raw_reward_sum, info["raw_reward"])
        self.assertEqual(simple.algorithm_reward_sum, reward)
        self.assertEqual(simple.component_totals["reward_fc_variation_penalty"], 225.0)

    def test_complex_and_simple_model_identities_cannot_be_mixed(self):
        from .reward_wrapper import assert_model_identity, make_env
        _, simple = self.pair()
        complex_env = make_env(DATA, fixed_train_id=FIXED_ID)
        self.addCleanup(complex_env.close)
        simple_model = SimpleNamespace(_td3_reward_identity=simple.reward_identity, reward_scale=0.0001)
        complex_model = SimpleNamespace(_td3_reward_identity=complex_env.reward_identity, reward_scale=0.0001)
        assert_model_identity(simple_model, simple.reward_identity)
        with self.assertRaisesRegex(ValueError, "identity"):
            assert_model_identity(simple_model, complex_env.reward_identity)
        with self.assertRaisesRegex(ValueError, "identity"):
            assert_model_identity(complex_model, simple.reward_identity)

    def test_control_reward_excludes_hydrogen_without_erasing_real_costs(self):
        control = self.module().make_control_env(DATA, fixed_train_id=FIXED_ID)
        self.addCleanup(control.close)
        control.reset(seed=42)
        _, reward, terminated, truncated, info = control.step([0.0])
        self.assertFalse(terminated or truncated)
        self.assertGreater(info["h2_cost_cny"], 0.0)
        self.assertEqual(info["reward_h2_cost_cny"], 0.0)
        self.assertEqual(info["reward_economic_cost_cny"], 0.0)
        expected = -(0.0025 * info["delta_fc_kw"] ** 2
                     + 10.0 * (info["soc"] - 0.5) ** 2)
        self.assertAlmostEqual(info["raw_reward"], expected)
        self.assertAlmostEqual(reward, expected / 10000.0)
        self.assertEqual(info["reward_version"], self.module().CONTROL_REWARD_VERSION)

    def test_control_and_hydrogen_rewards_execute_identical_physics(self):
        module = self.module()
        simple = module.make_simple_env(DATA, fixed_train_id=FIXED_ID)
        control = module.make_control_env(DATA, fixed_train_id=FIXED_ID)
        self.addCleanup(simple.close)
        self.addCleanup(control.close)
        simple_obs, _ = simple.reset(seed=42)
        control_obs, _ = control.reset(seed=42)
        np.testing.assert_array_equal(control_obs, simple_obs)
        for action in (-1.0, 0.0, 1.0, -0.3, 0.5):
            simple_obs, _, simple_done, simple_truncated, simple_info = simple.step([action])
            control_obs, _, done, truncated, info = control.step([action])
            np.testing.assert_array_equal(control_obs, simple_obs)
            self.assertEqual((done, truncated), (simple_done, simple_truncated))
            for name in ("h2_cost_cny", "hydrogen_mass_kg", "total_cost_cny",
                         "physical_economic_cost_cny", "c_bat_eq_cny",
                         "fuel_cell_degradation_cost_cny", "battery_degradation_cost_cny",
                         "fc_actual_kw", "battery_actual_kw", "battery_current_a"):
                self.assertEqual(info[name], simple_info[name], name)
            self.assertEqual(control.unwrapped.physical, simple.unwrapped.physical)
            self.assertEqual(control.unwrapped.economic_totals, simple.unwrapped.economic_totals)
            self.assertAlmostEqual(info["raw_reward"] - simple_info["raw_reward"], info["h2_cost_cny"])

    def test_control_identity_is_distinct_and_legacy_default_is_preserved(self):
        from .reward_wrapper import assert_model_identity
        module = self.module()
        simple = module.make_simple_env(DATA, fixed_train_id=FIXED_ID)
        control = module.make_control_env(DATA, fixed_train_id=FIXED_ID)
        self.addCleanup(simple.close)
        self.addCleanup(control.close)
        self.assertEqual(simple.reward_identity["reward_version"], module.SIMPLE_REWARD_VERSION)
        self.assertNotIn("include_hydrogen", simple.reward_identity)
        self.assertEqual(control.reward_identity["reward_version"], module.CONTROL_REWARD_VERSION)
        self.assertEqual(control.reward_identity["env_id"], "AggregateShipTD3Control-v0")
        self.assertIs(control.reward_identity["include_hydrogen"], False)
        self.assertNotIn("C_H2", control.reward_identity["formula"])
        three_model = SimpleNamespace(_td3_reward_identity=simple.reward_identity, reward_scale=0.0001)
        two_model = SimpleNamespace(_td3_reward_identity=control.reward_identity, reward_scale=0.0001)
        assert_model_identity(two_model, control.reward_identity)
        with self.assertRaisesRegex(ValueError, "identity"):
            assert_model_identity(three_model, control.reward_identity)
        with self.assertRaisesRegex(ValueError, "identity"):
            assert_model_identity(two_model, simple.reward_identity)


if __name__ == "__main__":
    unittest.main()
