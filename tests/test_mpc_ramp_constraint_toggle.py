from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (SRC,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mpc.solvers.casadi_solver import CasadiMPCConfig, ShipCasadiMPC, casadi_available  # noqa: E402


class TestMpcRampConstraintToggle(unittest.TestCase):
    @unittest.skipUnless(casadi_available(), "CasADi is not installed")
    def test_disabled_ramp_constraint_allows_step_larger_than_48kw(self) -> None:
        cfg = CasadiMPCConfig(
            prediction_horizon=2,
            dt_hours=30.0 / 3600.0,
            battery_capacity_kwh=277.2,
            battery_charge_max_kw=10.0,
            battery_discharge_max_kw=10.0,
            fuel_cell_min_kw=0.0,
            fuel_cell_max_kw=560.0,
            fuel_cell_ramp_kw=48.0,
            fuel_cell_ramp_constraint_enabled=False,
            soc_min=0.2,
            soc_max=0.8,
            soc_reference_mode="initial_soc",
            objective_mode="raw_physical",
            use_raw_objective=False,
            use_dimensionless_objective=False,
            use_h2_mass_cost=True,
            normalize_h2_cost=False,
            enable_terminal_soc_soft_penalty=False,
            q_h2=0.0,
            q_soc=0.0,
            q_batt=0.0,
            q_ramp=0.0,
            q_terminal_soc=0.0,
            ipopt_max_iter=80,
            ipopt_tol=1e-5,
        )
        solver = ShipCasadiMPC(cfg)

        result = solver.solve(
            current_soc=0.55,
            prev_fc_kw=0.0,
            load_forecast_kw=np.array([200.0, 200.0]),
            soc_reference_value=0.55,
        )

        self.assertTrue(result.success, result.objective_info.get("solver_message", ""))
        self.assertGreater(result.fuel_cell_ref_kw, cfg.fuel_cell_ramp_kw)
        self.assertFalse(result.objective_info["fuel_cell_ramp_constraint_enabled"])

    @unittest.skipUnless(casadi_available(), "CasADi is not installed")
    def test_enabled_ramp_constraint_limits_first_step(self) -> None:
        cfg = CasadiMPCConfig(
            prediction_horizon=2,
            dt_hours=30.0 / 3600.0,
            battery_capacity_kwh=277.2,
            battery_charge_max_kw=80.0,
            battery_discharge_max_kw=80.0,
            fuel_cell_min_kw=0.0,
            fuel_cell_max_kw=560.0,
            fuel_cell_ramp_kw=60.0,
            fuel_cell_ramp_constraint_enabled=True,
            soc_min=0.2,
            soc_max=0.8,
            soc_reference_mode="initial_soc",
            objective_mode="raw_physical",
            use_raw_objective=False,
            use_dimensionless_objective=False,
            use_h2_mass_cost=True,
            normalize_h2_cost=False,
            enable_terminal_soc_soft_penalty=False,
            q_h2=0.0,
            q_soc=0.0,
            q_batt=0.0,
            q_ramp=0.0,
            q_terminal_soc=0.0,
            ipopt_max_iter=80,
            ipopt_tol=1e-5,
        )
        solver = ShipCasadiMPC(cfg)

        result = solver.solve(
            current_soc=0.55,
            prev_fc_kw=0.0,
            load_forecast_kw=np.array([90.0, 90.0]),
            soc_reference_value=0.55,
        )

        self.assertTrue(result.success, result.objective_info.get("solver_message", ""))
        self.assertLessEqual(result.fuel_cell_ref_kw, cfg.fuel_cell_ramp_kw + 1e-5)
        self.assertTrue(result.objective_info["fuel_cell_ramp_constraint_enabled"])


if __name__ == "__main__":
    unittest.main()
