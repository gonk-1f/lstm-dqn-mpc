"""Single-interval checks for the aggregate economic reward; no datasets or learning."""
from __future__ import annotations

import importlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
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

    def test_idle_zero_cost_has_reward_one_and_soc_penalty_is_plain_square(self):
        module = self.module()
        from v3.control import AccountState
        physics = self.physics()
        before = AccountState(soc=0.5)
        after, ledger, _ = physics.interval(before, 0.0, 0.0, duration_s=1.0)
        values = module.interval_economic_components(physics, before, after, 0.0, ledger)
        self.assertEqual(values["total_cost_cny"], 0.0)
        self.assertEqual(module.economic_reward(0.0, 0.5, 100.0), (1.0, 0.0, 0.0))
        reward, penalty, total = module.economic_reward(1.0, 0.6, 100.0)
        self.assertAlmostEqual(penalty, 1.0)
        self.assertAlmostEqual(total, 2.0)
        self.assertAlmostEqual(reward, math.tanh(0.5))
        self.assertEqual(module.economic_reward(1.0, 0.5, 0.0), (math.tanh(1.0), 0.0, 1.0))

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
        self.assertEqual(module.load_reward_config()["lambda_soc"], 17.0)
        config = {"reward_version": module.REWARD_VERSION, "c_typ": 0.02, "lambda_soc": 1.7}
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
            path.write_text(json.dumps({**config, "lambda_soc": -3.0}), encoding="utf-8")
            with self.assertRaises(ValueError):
                module.load_reward_config(path)


if __name__ == "__main__":
    unittest.main()
