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

    def test_on_discharge_uses_actual_efficiency_and_original_ledger_once(self):
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
        expected = battery_kw / 3600.0 / eta_dis / (eta_fc * LHV_H2_KWH_PER_KG)
        self.assertAlmostEqual(values["battery_equivalent_hydrogen_kg"], expected, places=12)
        self.assertEqual(values["equivalent_fc_efficiency"], eta_fc)
        self.assertEqual(values["fc_efficiency_actual"], eta_fc)
        self.assertAlmostEqual(values["c_bat_eq_cny"], expected * HYDROGEN_PRICE_CNY_PER_KG, places=12)
        self.assertEqual(values["physical_economic_cost_cny"], ledger.total_cost_cny)
        self.assertAlmostEqual(values["total_cost_cny"], ledger.total_cost_cny + values["c_bat_eq_cny"], places=12)
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
            physics.fc_efficiency.eta(300.0) * LHV_H2_KWH_PER_KG
        ) * HYDROGEN_PRICE_CNY_PER_KG
        self.assertLess(values["c_bat_eq_cny"], 0)
        self.assertAlmostEqual(values["c_bat_eq_cny"], expected, places=12)
        self.assertGreaterEqual(ledger.h2_cost_cny + values["c_bat_eq_cny"], 0)
        self.assertGreaterEqual(values["total_cost_cny"], 0)

    def test_fc_off_discharge_uses_computed_positive_curve_peak(self):
        module = self.module()
        from v3.control import AccountState
        from v2.economics import HYDROGEN_PRICE_CNY_PER_KG
        from v2.models.fuel_cell_efficiency import LHV_H2_KWH_PER_KG
        physics = self.physics()
        before = AccountState(soc=0.6)
        after, ledger, battery_kw = physics.interval(before, 0.0, 100.0, duration_s=1.0)
        values = module.interval_economic_components(physics, before, after, 0.0, ledger)
        _, eta_dis = physics.efficiency.require_calibrated()
        peak = module.peak_fc_efficiency(physics.fc_efficiency)
        self.assertEqual(values["fc_efficiency_actual"], 0.0)
        self.assertEqual(values["equivalent_fc_efficiency"], peak)
        self.assertEqual(values["h2_cost_cny"], 0.0)
        expected = battery_kw / 3600.0 / eta_dis / (peak * LHV_H2_KWH_PER_KG) * HYDROGEN_PRICE_CNY_PER_KG
        self.assertAlmostEqual(values["c_bat_eq_cny"], expected, places=12)
        self.assertTrue(math.isfinite(values["total_cost_cny"]))

    def test_idle_zero_cost_has_reward_zero_and_soc_penalty_is_plain_square(self):
        module = self.module()
        from v3.control import AccountState
        physics = self.physics()
        before = AccountState(soc=0.5)
        after, ledger, _ = physics.interval(before, 0.0, 0.0, duration_s=1.0)
        values = module.interval_economic_components(physics, before, after, 0.0, ledger)
        self.assertEqual(values["total_cost_cny"], 0.0)
        self.assertEqual(module.economic_reward(0.0, 0.5, 8.0), (0.0, 0.0, 0.0))
        reward, penalty, total = module.economic_reward(1.0, 0.6, 8.0)
        self.assertAlmostEqual(penalty, 0.08)
        self.assertAlmostEqual(total, 1.08)
        self.assertAlmostEqual(reward, -1.08)

    def test_positive_economic_cost_keeps_raw_cny_units_without_saturation(self):
        module = self.module()
        for cost in (0.0001, 1.0, 2.0, 358.65):
            with self.subTest(cost=cost):
                self.assertEqual(module.economic_reward(cost, 0.5, 8.0), (-cost, 0.0, cost))

    def test_negative_or_nonfinite_cost_is_never_abs_or_clipped(self):
        module = self.module()
        for value in (-1.0, float("nan"), float("inf"), -float("inf")):
            with self.subTest(cost=value), self.assertRaises(ValueError):
                module.economic_reward(value, 0.6, 1000000.0)
        for soc, weight in ((float("nan"), 1.0), (float("inf"), 1.0), (0.5, -1.0), (0.5, float("inf"))):
            with self.subTest(soc=soc, weight=weight), self.assertRaises(ValueError):
                module.economic_reward(1.0, soc, weight)
        from v3.control import AccountState
        from v2.economics import RawCnyIntervalLedger
        physics = self.physics()
        before = AccountState(soc=0.6, previous_fc_kw=300.0)
        after, _, _ = physics.interval(before, 300.0, 100.0, duration_s=1.0)
        with self.assertRaisesRegex(ValueError, "components|c_bat_eq_cny"):
            module.interval_economic_components(physics, before, after, 300.0, RawCnyIntervalLedger(0.0, 0.0, 0.0, 0.0))

    def test_fixed_config_is_loaded_only_on_request_and_identity_is_stable(self):
        module = self.module()
        default = module.load_reward_config()
        self.assertEqual(default["lambda_soc"], 8.0)
        self.assertEqual(default["reward_version"], "ddpg_economic_linear_v2")
        self.assertEqual(default["failure_reward"], -10000.0)
        self.assertEqual(getattr(module, "FAILURE_REWARD", None), -10000.0)
        config = {
            "reward_version": module.REWARD_VERSION, "c_typ": 0.02,
            "lambda_soc": 1.7, "failure_reward": -10000.0,
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
            self.assertEqual(first["lambda_soc"], 1.7)
            self.assertEqual(first["c_typ"], 0.02)
            self.assertNotEqual(first["lambda_soc"], 100.0 * first["c_typ"])
            self.assertEqual(first["reward_version"], module.REWARD_VERSION)
            self.assertEqual(first["failure_reward"], -10000.0)
            self.assertEqual(
                first["formula"],
                "-(C_H2+C_FC_deg+C_LB_deg+C_bat_eq+lambda_soc*(SOC_next-0.5)**2)",
            )
            encoded = json.dumps(
                loaded, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
            self.assertEqual(first["config_sha256"], hashlib.sha256(encoded).hexdigest())
            for changes in (
                {"lambda_soc": -3.0}, {"failure_reward": -1.0},
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

    def test_evaluation_rejects_unversioned_or_old_model_identity(self):
        module = self.module()
        identity = module.reward_identity(module.load_reward_config())
        models = [SimpleNamespace()]
        models.extend(SimpleNamespace(_ddpg_reward_identity=value) for value in (
            None, {}, {**identity, "reward_version": "ddpg_economic_tanh_v1"},
            {**identity, "reward_version": "unknown"}, {**identity, "lambda_soc": 9.0},
        ))
        for model in models:
            with self.subTest(model=model), self.assertRaises(ValueError):
                module.evaluation_identity(model, identity)

    def test_evaluation_rejects_equal_but_unsupported_reward_identities(self):
        module = self.module()
        identity = module.reward_identity(module.load_reward_config())
        for version in (None, "ddpg_economic_tanh_v1", "unknown"):
            unsupported = {**identity, "reward_version": version}
            if version is None:
                unsupported.pop("reward_version")
            with self.subTest(version=version), self.assertRaises(ValueError):
                module.evaluation_identity(SimpleNamespace(_ddpg_reward_identity=unsupported), unsupported)


if __name__ == "__main__":
    unittest.main()
