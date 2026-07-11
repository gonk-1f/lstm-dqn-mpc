from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
MAIN = SRC / "main"
for path in (SRC, MAIN):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_lstm_mpc_test import (  # noqa: E402
    DEFAULT_WEIGHT_SETS,
    build_mpc_load_ref,
    mpc_config_from_weights,
)
from mpc.solvers.casadi_solver import (  # noqa: E402
    CasadiMPCConfig,
    ShipCasadiMPC,
    casadi_available,
    compute_soc_stage_cost,
    resolve_soc_reference,
)


class TestMpcSocReferenceModes(unittest.TestCase):
    def test_initial_soc_reference_uses_voyage_start_value(self) -> None:
        cfg = CasadiMPCConfig(soc_reference_mode="initial_soc")

        self.assertAlmostEqual(
            resolve_soc_reference(cfg, current_soc=0.61, soc_reference_value=0.55),
            0.55,
        )

    def test_reserve_only_has_no_penalty_above_reserve(self) -> None:
        cfg = CasadiMPCConfig(soc_reference_mode="reserve_only", soc_reserve=0.55)

        self.assertAlmostEqual(compute_soc_stage_cost(cfg, soc=0.60, soc_reference_value=0.55), 0.0)
        self.assertGreater(compute_soc_stage_cost(cfg, soc=0.54, soc_reference_value=0.55), 0.0)

    def test_fixed_target_mode_remains_available_for_diagnosis(self) -> None:
        cfg = CasadiMPCConfig(soc_reference_mode="fixed_target", soc_target=0.65)

        self.assertAlmostEqual(resolve_soc_reference(cfg, current_soc=0.55), 0.65)

    def test_nextstep_reference_uses_lstm_h1_to_h6(self) -> None:
        pred = np.arange(1.0, 7.0)

        ref = build_mpc_load_ref(10.0, pred, pred_horizon=6, mpc_horizon=6)

        np.testing.assert_array_equal(ref, pred)

    def test_deprecated_rule_based_config_does_not_reach_mpc_config(self) -> None:
        cfg = mpc_config_from_weights(
            {
                **DEFAULT_WEIGHT_SETS["dp0_raw_h2_soc_batt_ramp_nextstep_v1"],
                "soc_recovery_power_limit": {
                    "enabled": True,
                    "max_charge_power_kw": 80.0,
                    "apply_when_soc_above_reserve": True,
                    "soc_reserve": 0.55,
                },
                "low_load_fc_suppression": {"enabled": True},
                "sustained_load_battery_discharge_limit": {"enabled": True},
                "fc_overproduction_limit": {"enabled": True},
            }
        )

        self.assertFalse(hasattr(cfg, "soc_recovery_power_limit_enabled"))
        self.assertFalse(hasattr(cfg, "low_load_fc_suppression_enabled"))
        self.assertFalse(hasattr(cfg, "sustained_load_battery_discharge_limit_enabled"))
        self.assertFalse(hasattr(cfg, "fc_overproduction_limit_enabled"))

    @unittest.skipUnless(casadi_available(), "CasADi is not installed")
    def test_solver_reports_only_physical_bounds_without_rule_flags(self) -> None:
        cfg = CasadiMPCConfig(
            prediction_horizon=2,
            dt_hours=30.0 / 3600.0,
            battery_capacity_kwh=1806.0,
            battery_charge_max_kw=350.0,
            battery_discharge_max_kw=350.0,
            fuel_cell_max_kw=560.0,
            fuel_cell_ramp_kw=560.0,
            soc_min=0.2,
            soc_max=0.8,
            soc_target=0.65,
            soc_reference_mode="fixed_target",
            soc_band=0.01,
            q_h2=0.0,
            q_soc=100.0,
            q_ramp=0.0,
            q_batt=0.0,
            q_terminal_soc=100.0,
            use_dimensionless_objective=True,
            enable_terminal_soc_soft_penalty=True,
            terminal_soc_band=0.01,
        )
        solver = ShipCasadiMPC(cfg)

        result = solver.solve(
            current_soc=0.55,
            prev_fc_kw=0.0,
            load_forecast_kw=np.zeros(2),
        )

        self.assertTrue(result.success)
        self.assertAlmostEqual(result.objective_info["P_fc_upper_bound"], cfg.fuel_cell_max_kw)
        self.assertAlmostEqual(result.objective_info["P_fc_lower_bound"], cfg.fuel_cell_min_kw)
        self.assertAlmostEqual(result.objective_info["battery_discharge_upper_bound"], cfg.battery_discharge_max_kw)
        self.assertAlmostEqual(result.objective_info["battery_charge_upper_bound"], cfg.battery_charge_max_kw)
        for key in result.objective_info:
            self.assertNotIn("low_load_fc_suppression", key)
            self.assertNotIn("soc_recovery_power_limit", key)
            self.assertNotIn("sustained_load_battery_discharge_limit", key)
            self.assertNotIn("fc_overproduction_limit", key)

    @unittest.skipUnless(casadi_available(), "CasADi is not installed")
    def test_raw_physical_solver_succeeds_near_zero_load(self) -> None:
        cfg = CasadiMPCConfig(
            prediction_horizon=2,
            dt_hours=30.0 / 3600.0,
            battery_capacity_kwh=1806.0,
            battery_charge_max_kw=350.0,
            battery_discharge_max_kw=350.0,
            fuel_cell_max_kw=560.0,
            fuel_cell_ramp_kw=48.0,
            soc_min=0.2,
            soc_max=0.8,
            soc_reference_mode="initial_soc",
            objective_mode="raw_physical",
            use_raw_objective=False,
            use_dimensionless_objective=False,
            use_h2_mass_cost=True,
            normalize_h2_cost=False,
            q_h2=1.0,
            q_soc=20.0,
            q_batt=0.1,
            q_ramp=0.0,
            q_terminal_soc=0.0,
            enable_terminal_soc_soft_penalty=False,
        )
        solver = ShipCasadiMPC(cfg)

        result = solver.solve(
            current_soc=0.55,
            prev_fc_kw=0.0,
            load_forecast_kw=np.array([0.0, 6.525]),
            soc_reference_value=0.55,
        )

        self.assertTrue(result.success, result.objective_info.get("solver_message", ""))
        self.assertEqual(result.objective_info["objective_scale_mode"], "raw_physical")
        self.assertGreaterEqual(result.objective_info["batt_throughput_kwh"], 0.0)


if __name__ == "__main__":
    unittest.main()
