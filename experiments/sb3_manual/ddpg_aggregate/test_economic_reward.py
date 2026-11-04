"""Single-interval checks for the aggregate economic reward; no datasets or learning."""
from __future__ import annotations

import importlib
import importlib.util
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
PACKAGE = "experiments.sb3_manual.ddpg_aggregate"


class EconomicRewardTests(unittest.TestCase):
    def module(self):
        name = PACKAGE + ".economic_reward"
        self.assertIsNotNone(importlib.util.find_spec(name), "Economic reward implementation is missing")
        return importlib.import_module(name)

    def physics(self):
        from v3.control import EconomicMPC
        return EconomicMPC(nominal_cost_cny=1.0)

    def test_peak_uses_positive_formal_curve_and_dominates_dense_grid(self):
        module = self.module()
        physics = self.physics()
        peak = module.peak_fc_efficiency(physics.fc_efficiency)
        self.assertEqual(peak, max(physics.fc_efficiency.efficiencies))
        dense = physics.fc_efficiency.eta(np.linspace(0.01, 600.0, 60000))
        self.assertLessEqual(float(dense.max()), peak + 1e-12)

    def test_on_discharge_uses_fixed_efficiency_and_original_ledger_once(self):
        module = self.module()
        from v3.control import AccountState
        physics = self.physics()
        before = AccountState(soc=0.6, previous_fc_kw=300.0)
        after, ledger, battery_kw = physics.interval(before, 300.0, 400.0, duration_s=1.0)
        values = module.interval_economic_components(physics, before, after, 300.0, ledger)
        from v2.economics import HYDROGEN_PRICE_CNY_PER_KG
        from v2.models.fuel_cell_efficiency import LHV_H2_KWH_PER_KG
        _, eta_dis = physics.efficiency.require_calibrated()
        eta_fc = physics.fc_efficiency.eta(300.0)
        expected = battery_kw / 3600.0 / eta_dis / (module.ETA_FC_AVG * LHV_H2_KWH_PER_KG)
        self.assertAlmostEqual(values["battery_equivalent_hydrogen_kg"], expected, places=12)
        self.assertEqual(values["equivalent_fc_efficiency"], module.ETA_FC_AVG)
        self.assertEqual(values["fc_efficiency_actual"], eta_fc)
        self.assertAlmostEqual(values["c_bat_eq_cny"], expected * HYDROGEN_PRICE_CNY_PER_KG, places=12)
        self.assertEqual(values["physical_economic_cost_cny"], ledger.total_cost_cny)
        self.assertAlmostEqual(values["total_cost_cny"], ledger.total_cost_cny + values["c_bat_eq_cny"], places=12)
        self.assertAlmostEqual(
            values["fuel_cell_degradation_cost_cny"],
            math.fsum(values[key] for key in (
                "fuel_cell_transient_cost_cny", "fuel_cell_low_load_cost_cny",
                "fuel_cell_high_load_cost_cny", "fuel_cell_start_stop_cost_cny",
            )), places=12,
        )
        self.assertAlmostEqual(
            values["reward_total_cost_cny"],
            values["total_cost_cny"] - values["fuel_cell_start_stop_cost_cny"], places=12,
        )
        for field in ("h2_cost_cny", "fuel_cell_degradation_cost_cny", "battery_degradation_cost_cny", "shore_cost_cny"):
            self.assertEqual(values[field], getattr(ledger, field))

    def test_charge_is_signed_without_overcredit_or_double_efficiency(self):
        module = self.module()
        from v3.control import AccountState
        from v2.economics import HYDROGEN_PRICE_CNY_PER_KG
        from v2.models.fuel_cell_efficiency import LHV_H2_KWH_PER_KG
        physics = self.physics()
        before = AccountState(soc=0.6, previous_fc_kw=300.0)
        after, ledger, battery_kw = physics.interval(before, 300.0, 100.0, duration_s=1.0)
        values = module.interval_economic_components(physics, before, after, 300.0, ledger)
        eta_chg, _ = physics.efficiency.require_calibrated()
        expected = battery_kw / 3600.0 * eta_chg / (
            module.ETA_FC_AVG * LHV_H2_KWH_PER_KG
        ) * HYDROGEN_PRICE_CNY_PER_KG
        self.assertLess(values["c_bat_eq_cny"], 0)
        self.assertAlmostEqual(values["c_bat_eq_cny"], expected, places=12)
        self.assertGreaterEqual(ledger.h2_cost_cny + values["c_bat_eq_cny"], 0)
        self.assertGreaterEqual(values["total_cost_cny"], 0)

    def test_fc_off_discharge_uses_same_fixed_efficiency(self):
        module = self.module()
        from v3.control import AccountState
        from v2.economics import HYDROGEN_PRICE_CNY_PER_KG
        from v2.models.fuel_cell_efficiency import LHV_H2_KWH_PER_KG
        physics = self.physics()
        before = AccountState(soc=0.6)
        after, ledger, battery_kw = physics.interval(before, 0.0, 100.0, duration_s=1.0)
        values = module.interval_economic_components(physics, before, after, 0.0, ledger)
        _, eta_dis = physics.efficiency.require_calibrated()
        eta_fc_avg = module.ETA_FC_AVG
        self.assertEqual(values["fc_efficiency_actual"], 0.0)
        self.assertEqual(values["equivalent_fc_efficiency"], eta_fc_avg)
        self.assertEqual(values["h2_cost_cny"], 0.0)
        expected = battery_kw / 3600.0 / eta_dis / (eta_fc_avg * LHV_H2_KWH_PER_KG) * HYDROGEN_PRICE_CNY_PER_KG
        self.assertAlmostEqual(values["c_bat_eq_cny"], expected, places=12)
        self.assertTrue(math.isfinite(values["total_cost_cny"]))

    def test_idle_zero_cost_has_reward_zero_without_independent_soc_penalty(self):
        module = self.module()
        from v3.control import AccountState
        physics = self.physics()
        before = AccountState(soc=0.5)
        after, ledger, _ = physics.interval(before, 0.0, 0.0, duration_s=1.0)
        values = module.interval_economic_components(physics, before, after, 0.0, ledger)
        self.assertEqual(values["total_cost_cny"], 0.0)
        self.assertEqual(module.economic_reward(0.0), 0.0)

    def test_finite_signed_reward_cost_is_negated_without_reference_scaling(self):
        module = self.module()
        for cost in (-2.0, -0.0001, 0.0, 0.0001, 1.0, 2.0, 358.65):
            with self.subTest(cost=cost):
                self.assertEqual(module.economic_reward(cost), -cost)
        self.assertGreater(module.economic_reward(1.0), module.economic_reward(2.0))

    def test_nonfinite_reward_cost_and_invalid_physical_ledger_are_rejected(self):
        module = self.module()
        for value in (float("nan"), float("inf"), -float("inf"), True):
            with self.subTest(cost=value), self.assertRaises(ValueError):
                module.economic_reward(value)
        from v3.control import AccountState
        from v2.economics import RawCnyIntervalLedger
        physics = self.physics()
        before = AccountState(soc=0.6, previous_fc_kw=300.0)
        after, _, _ = physics.interval(before, 300.0, 100.0, duration_s=1.0)
        with self.assertRaisesRegex(ValueError, "components|c_bat_eq_cny"):
            module.interval_economic_components(physics, before, after, 300.0, RawCnyIntervalLedger(0.0, 0.0, 0.0, 0.0))

    def test_fixed_config_and_identity_are_stable(self):
        module = self.module()
        default = module.load_reward_config()
        self.assertGreater(default["c_ref"], 0.0)
        self.assertNotIn("lambda_soc", default)
        self.assertNotIn("soc_reference", default)
        self.assertNotIn("c_typ", default)
        self.assertEqual(default["reward_version"], "ddpg_economic_negative_no_start_stop_v5")
        self.assertEqual(default["failure_reward"], -2000.0)
        self.assertEqual(getattr(module, "FAILURE_REWARD", None), -2000.0)
        config = {
            "reward_version": module.REWARD_VERSION, "c_ref": 0.02,
            "failure_reward": -2000.0,
            "eta_fc_avg": default["eta_fc_avg"],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "reward.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            original = path.read_bytes()
            loaded = module.load_reward_config(path)
            self.assertEqual(loaded, config)
            self.assertEqual(path.read_bytes(), original)
            first = module.reward_identity(loaded)
            self.assertEqual(first, module.reward_identity(loaded))
            self.assertEqual(first["c_ref"], 0.02)
            self.assertNotIn("lambda_soc", first)
            self.assertEqual(first["fuel_cell_eol_voltage_loss_uv"], 142500.0)
            self.assertEqual(first["battery_current_accounting"], "soc_delta_cell_current_v1")
            self.assertEqual(first["reward_version"], module.REWARD_VERSION)
            self.assertEqual(first["failure_reward"], -2000.0)
            self.assertEqual(
                first["formula"],
                "-(C_H2+C_bat_eq+C_LB_deg+C_FC_transient+C_FC_low+C_FC_high)",
            )
            encoded = json.dumps(
                loaded, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
            self.assertEqual(first["config_sha256"], hashlib.sha256(encoded).hexdigest())
            for changes in (
                {"c_ref": 0.0}, {"c_ref": -3.0}, {"c_ref": float("nan")}, {"failure_reward": -2.0},
                {"failure_reward": float("nan")}, {"reward_version": "ddpg_economic_tanh_v1"},
            ):
                with self.subTest(changes=changes):
                    path.write_text(json.dumps({**config, **changes}), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        module.load_reward_config(path)

    def test_evaluation_accepts_equal_current_reward_identity(self):
        module = self.module()
        identity = module.reward_identity(module.load_reward_config())
        metadata = module.evaluation_identity(SimpleNamespace(_ddpg_reward_identity=dict(identity)), identity)
        self.assertEqual(metadata["model_reward_identity"], identity)
        self.assertEqual(metadata["evaluation_reward_identity"], identity)
        self.assertFalse(metadata["legacy_policy_evaluated_with_current_reward"])

    def test_reference_retains_historical_train35_calibration_metadata(self):
        config = self.module().load_reward_config()
        calibration = config["calibration"]
        self.assertEqual(calibration["split"], "train")
        self.assertEqual(calibration["selected_voyages"], 35)
        self.assertEqual(len(set(calibration["train_ids"])), 35)
        self.assertEqual(config["c_ref"], calibration["c_ref"])
        self.assertEqual(config["c_ref"], calibration["economic_cost_quantiles_cny"]["0.5"])
        self.assertEqual(calibration["fc_eol_voltage_loss_uv"], 142500.0)
        self.assertEqual(calibration["battery_current_accounting_version"], "soc_delta_cell_current_v1")
        self.assertEqual(calibration["battery_lifetime_throughput_factor"], 15000.0)
        for name in ("negative_economic_steps", "nonfinite_economic_steps",
                     "validation_payloads_used", "test_payloads_used"):
            self.assertEqual(calibration[name], 0, name)
        for voyage in calibration["per_voyage"]:
            self.assertLessEqual(voyage["max_soc_replay_error"], 1e-12)
            self.assertLessEqual(voyage["max_battery_power_replay_error_kw"], 1e-9)
            self.assertLessEqual(voyage["max_h2_cost_replay_error_cny"], 1e-12)

    def test_evaluation_rejects_unversioned_or_old_model_identity(self):
        module = self.module()
        identity = module.reward_identity(module.load_reward_config())
        models = [SimpleNamespace()]
        models.extend(SimpleNamespace(_ddpg_reward_identity=value) for value in (
            None, {}, {**identity, "reward_version": "ddpg_economic_tanh_v1"},
            {**identity, "reward_version": "ddpg_economic_linear_v2"},
            {**identity, "reward_version": "ddpg_economic_normalized_tanh_v4"},
            {**identity, "reward_version": "unknown"}, {**identity, "c_ref": 9.0},
            {**identity, "fuel_cell_eol_voltage_loss_uv": 70000.0},
        ))
        for model in models:
            with self.subTest(model=model), self.assertRaises(ValueError):
                module.evaluation_identity(model, identity)

    def test_evaluation_rejects_equal_but_unsupported_reward_identities(self):
        module = self.module()
        identity = module.reward_identity(module.load_reward_config())
        for version in (None, "ddpg_economic_tanh_v1", "ddpg_economic_linear_v2", "unknown"):
            unsupported = {**identity, "reward_version": version}
            if version is None:
                unsupported.pop("reward_version")
            with self.subTest(version=version), self.assertRaises(ValueError):
                module.evaluation_identity(SimpleNamespace(_ddpg_reward_identity=unsupported), unsupported)


if __name__ == "__main__":
    unittest.main()
