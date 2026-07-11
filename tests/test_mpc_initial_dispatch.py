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
    TOTAL_LOAD_QSOC400_QBATT003_QRAMP2E5_WEIGHT_SET,
    build_mpc_load_ref,
    mpc_config_from_weights,
)
from mpc.solvers.casadi_solver import (  # noqa: E402
    CasadiMPCConfig,
    ShipCasadiMPC,
    casadi_available,
)


def _base_dispatch_config(**overrides: object) -> CasadiMPCConfig:
    kwargs = dict(
        prediction_horizon=2,
        dt_hours=30.0 / 3600.0,
        battery_capacity_kwh=1806.0,
        battery_charge_max_kw=350.0,
        battery_discharge_max_kw=350.0,
        fuel_cell_min_kw=0.0,
        fuel_cell_max_kw=560.0,
        fuel_cell_ramp_kw=560.0,
        soc_min=0.2,
        soc_max=0.8,
        soc_target=0.65,
        soc_reference_mode="initial_soc",
        soc_reserve=0.55,
        soc_band=0.005,
        terminal_soc_band=0.005,
        q_h2=0.0,
        q_soc=0.0,
        q_fc=0.0,
        q_batt=10.0,
        q_ramp=0.0,
        q_terminal_soc=0.0,
        use_raw_objective=False,
        use_dimensionless_objective=True,
        use_h2_mass_cost=True,
        normalize_h2_cost=True,
        enable_terminal_soc_soft_penalty=False,
        battery_throughput_penalty_enabled=True,
        battery_throughput_penalty_type="absolute_power",
        battery_throughput_normalization_kw=350.0,
    )
    kwargs.update(overrides)
    return CasadiMPCConfig(**kwargs)


class TestMpcInitialDispatch(unittest.TestCase):
    def test_nextstep_mpc_load_reference_uses_lstm_h1_to_h6(self) -> None:
        pred = np.array([21.0, 22.0, 23.0, 24.0, 25.0, 26.0])

        ref = build_mpc_load_ref(20.0, pred, pred_horizon=6, mpc_horizon=6)

        np.testing.assert_array_equal(ref, np.array([21.0, 22.0, 23.0, 24.0, 25.0, 26.0]))

    def test_active_weight_sets_do_not_contain_rule_based_blocks(self) -> None:
        forbidden = {
            "low_load_fc_suppression",
            "soc_recovery_power_limit",
            "sustained_load_battery_discharge_limit",
            "fc_overproduction_limit",
            "low_load_fc_suppression_enabled",
            "soc_recovery_power_limit_enabled",
            "sustained_load_battery_discharge_limit_enabled",
            "fc_overproduction_limit_enabled",
        }
        for name, weights in DEFAULT_WEIGHT_SETS.items():
            self.assertFalse(forbidden.intersection(weights.keys()), name)

    def test_rule_based_blocks_are_ignored_even_if_present_in_external_config(self) -> None:
        cfg = mpc_config_from_weights(
            {
                **DEFAULT_WEIGHT_SETS[TOTAL_LOAD_QSOC400_QBATT003_QRAMP2E5_WEIGHT_SET],
                "low_load_fc_suppression": {
                    "enabled": True,
                    "low_load_threshold_kw": 4.0,
                    "soc_allow_shutdown_threshold": 0.57,
                    "fc_idle_upper_kw": 1.5,
                },
                "soc_recovery_power_limit": {
                    "enabled": True,
                    "max_charge_power_kw": 80.0,
                },
                "sustained_load_battery_discharge_limit": {
                    "enabled": True,
                    "load_threshold_kw": 20.0,
                    "soc_margin_above_ref": 0.01,
                    "max_discharge_kw": 40.0,
                },
                "fc_overproduction_limit": {
                    "enabled": True,
                    "max_over_load_kw": 80.0,
                    "apply_when_soc_above_reserve": True,
                },
            }
        )

        self.assertFalse(hasattr(cfg, "low_load_fc_suppression_enabled"))
        self.assertFalse(hasattr(cfg, "soc_recovery_power_limit_enabled"))
        self.assertFalse(hasattr(cfg, "sustained_load_battery_discharge_limit_enabled"))
        self.assertFalse(hasattr(cfg, "fc_overproduction_limit_enabled"))

    @unittest.skipUnless(casadi_available(), "CasADi is not installed")
    def test_extreme_q_batt_makes_fuel_cell_supply_sustained_load(self) -> None:
        cfg = _base_dispatch_config(q_batt=10.0, battery_discharge_max_kw=350.0)
        solver = ShipCasadiMPC(cfg)

        result = solver.solve(current_soc=0.55, prev_fc_kw=0.0, load_forecast_kw=np.full(2, 80.0))

        self.assertTrue(result.success)
        self.assertLess(abs(result.battery_ref_kw), 5.0)
        self.assertGreater(result.fuel_cell_ref_kw, 75.0)

    @unittest.skipUnless(casadi_available(), "CasADi is not installed")
    def test_battery_discharge_max_kw_is_enforced(self) -> None:
        cfg = _base_dispatch_config(q_batt=0.0, battery_discharge_max_kw=20.0)
        solver = ShipCasadiMPC(cfg)

        result = solver.solve(current_soc=0.55, prev_fc_kw=0.0, load_forecast_kw=np.full(2, 80.0))

        self.assertTrue(result.success)
        self.assertLessEqual(max(result.battery_ref_traj_kw), 20.0 + 1e-5)

    @unittest.skipUnless(casadi_available(), "CasADi is not installed")
    def test_solver_does_not_apply_load_or_soc_based_bounds(self) -> None:
        cfg = _base_dispatch_config(q_batt=0.0, q_h2=1.0)
        solver = ShipCasadiMPC(cfg)

        result = solver.solve(current_soc=0.55, prev_fc_kw=0.0, load_forecast_kw=np.full(2, 80.0))

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


if __name__ == "__main__":
    unittest.main()
