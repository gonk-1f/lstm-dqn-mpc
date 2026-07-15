from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
MAIN_ROOT = SRC / "main"
for path in (SRC, MAIN_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


class TestN6ExperimentContract(unittest.TestCase):
    def test_exact_candidate_set_and_horizon(self) -> None:
        from run_mpc_1s_n6_weight_selection import (
            CANDIDATES,
            DEFAULT_OUTPUT_ROOT,
            DEFAULT_SUMMARY_REPORT,
            DEFAULT_TABLE_REPORT,
            N6_HORIZON,
        )

        self.assertEqual(N6_HORIZON, 6)
        self.assertEqual(
            CANDIDATES,
            (
                {"candidate_id": "A", "q_h2": 0.5, "q_fc_var": 0.05, "q_batt": 0.05},
                {"candidate_id": "B", "q_h2": 0.5, "q_fc_var": 0.10, "q_batt": 0.05},
                {"candidate_id": "C", "q_h2": 0.5, "q_fc_var": 0.10, "q_batt": 0.10},
            ),
        )
        self.assertEqual(DEFAULT_OUTPUT_ROOT.name, "mpc_1s_n6_h2_fcvar_batt")
        self.assertEqual(DEFAULT_TABLE_REPORT.name, "mpc_1s_n6_h2_fcvar_batt_table.csv")
        self.assertEqual(DEFAULT_SUMMARY_REPORT.name, "mpc_1s_n6_h2_fcvar_batt_summary.md")

    def test_ideal_future_window_uses_t_plus_1_to_t_plus_6_without_crossing(self) -> None:
        from run_mpc_1s_n6_weight_selection import ideal_future_window

        loads = np.arange(8, dtype=float)
        np.testing.assert_array_equal(
            ideal_future_window(loads, decision_index=0),
            np.array([1, 2, 3, 4, 5, 6], dtype=float),
        )
        np.testing.assert_array_equal(
            ideal_future_window(loads, decision_index=6),
            np.array([7, 7, 7, 7, 7, 7], dtype=float),
        )
        with self.assertRaisesRegex(ValueError, "must have a future execution sample"):
            ideal_future_window(loads, decision_index=7)

    def test_only_first_planned_step_is_executed_and_actual_soc_uses_actual_battery(self) -> None:
        from mpc_solvers.mpc_qp_formulation import QpMpcConfig
        from run_mpc_1s_n6_weight_selection import extract_first_step

        config = QpMpcConfig(horizon=6, dt_seconds=1.0, battery_capacity_kwh=693.0)
        solution = np.zeros(19, dtype=float)
        solution[0] = 110.0
        solution[6] = -7.0
        solution[13] = 0.550003

        applied = extract_first_step(
            solution,
            config=config,
            load_actual_kw=100.0,
            current_soc=0.55,
        )

        self.assertEqual(applied["P_fc_plan_kw"], 110.0)
        self.assertEqual(applied["P_fc_actual_kw"], 110.0)
        self.assertEqual(applied["P_batt_plan_kw"], -7.0)
        self.assertEqual(applied["P_batt_actual_kw"], -10.0)
        self.assertEqual(applied["SOC_predicted"], 0.550003)
        self.assertAlmostEqual(applied["SOC_actual"], 0.55 + 10.0 / (3600.0 * 693.0))


class TestN6QpContract(unittest.TestCase):
    def test_candidate_config_has_n6_dimensions_and_ramp_structure(self) -> None:
        from mpc_solvers.mpc_qp_formulation import build_qp_problem
        from benchmark_mpc_qp_osqp_1s import _qp_bounds_for_step
        from run_mpc_1s_n6_weight_selection import candidate_config

        config = candidate_config("A")
        self.assertEqual(config.horizon, 6)
        self.assertEqual(config.dt_seconds, 1.0)
        self.assertEqual(config.battery_capacity_kwh, 693.0)
        self.assertEqual(config.battery_charge_max_kw, 346.5)
        self.assertEqual(config.battery_discharge_max_kw, 346.5)
        self.assertEqual(config.battery_power_ref_kw, 346.5)
        self.assertEqual(config.fuel_cell_max_kw, 560.0)
        self.assertEqual(config.soc_band, 0.05)
        self.assertEqual(config.objective_variant, "n6_h2_fc_variation_battery_v1")
        self.assertEqual(config.q_h2, 0.5)
        self.assertEqual(config.q_fc_var, 0.05)
        self.assertEqual(config.q_soc, 0.0)
        self.assertEqual(config.q_batt, 0.05)
        self.assertEqual(config.q_ramp, 0.0)
        self.assertEqual(config.q_terminal_soc, 0.0)

        problem = build_qp_problem(
            config,
            load_forecast_kw=np.full(6, 100.0),
            current_soc=0.55,
            prev_fc_kw=100.0,
            soc_reference=0.55,
        )
        self.assertEqual(problem.metadata["n_variables"], 19)
        self.assertEqual(problem.metadata["n_constraints"], 38)
        self.assertEqual(problem.metadata["variable_order"], "P_fc[0:N], P_batt[0:N], SOC[0:N+1]")

        lower, upper = _qp_bounds_for_step(
            config,
            load_forecast_kw=np.full(6, 100.0),
            current_soc=0.55,
            prev_fc_kw=100.0,
        )
        np.testing.assert_allclose(lower[-6:], np.array([52.0, -48.0, -48.0, -48.0, -48.0, -48.0]))
        np.testing.assert_allclose(upper[-6:], np.array([148.0, 48.0, 48.0, 48.0, 48.0, 48.0]))

        with self.assertRaisesRegex(ValueError, "exactly 6 points"):
            build_qp_problem(
                config,
                load_forecast_kw=np.full(5, 100.0),
                current_soc=0.55,
                prev_fc_kw=100.0,
                soc_reference=0.55,
            )

    def test_three_term_objective_has_fc_variation_and_no_soc_cost(self) -> None:
        from dataclasses import replace

        from mpc_solvers.mpc_qp_formulation import (
            build_qp_problem,
            h2_quadratic_kg_step_coefficients,
        )
        from run_mpc_1s_n6_weight_selection import candidate_config

        config = replace(
            candidate_config("B"),
            q_soc=999.0,
            q_ramp=777.0,
            q_terminal_soc=555.0,
        )
        previous_fc_kw = 120.0
        problem = build_qp_problem(
            config,
            load_forecast_kw=np.full(6, 200.0),
            current_soc=0.54,
            prev_fc_kw=previous_fc_kw,
            soc_reference=0.31,
        )
        same_problem_other_soc_reference = build_qp_problem(
            config,
            load_forecast_kw=np.full(6, 200.0),
            current_soc=0.54,
            prev_fc_kw=previous_fc_kw,
            soc_reference=0.79,
        )

        dense_p = problem.P.toarray()
        h2_quad, h2_linear, _, _ = h2_quadratic_kg_step_coefficients(config)
        h2_ref = h2_quad * 560.0**2 + h2_linear * 560.0
        fc_scale2 = 48.0**2
        h2_hessian = 2.0 * config.q_h2 * h2_quad / h2_ref
        fc_var_hessian = 2.0 * config.q_fc_var / fc_scale2

        self.assertEqual(
            problem.metadata["objective_terms"],
            ["H2_norm", "FC_var_norm", "Batt_norm"],
        )
        self.assertAlmostEqual(problem.metadata["h2_reference_kg_per_step"], 0.008839452966, places=12)
        self.assertAlmostEqual(dense_p[0, 0], h2_hessian + 2.0 * fc_var_hessian)
        self.assertAlmostEqual(dense_p[5, 5], h2_hessian + fc_var_hessian)
        self.assertAlmostEqual(dense_p[0, 1], -fc_var_hessian)
        self.assertAlmostEqual(
            problem.q[0],
            config.q_h2 * h2_linear / h2_ref
            - 2.0 * config.q_fc_var * previous_fc_kw / fc_scale2,
        )
        np.testing.assert_allclose(
            np.diag(dense_p)[6:12],
            np.full(6, 2.0 * config.q_batt / 346.5**2),
        )
        np.testing.assert_allclose(dense_p[12:, :], 0.0, atol=0.0)
        np.testing.assert_allclose(dense_p[:, 12:], 0.0, atol=0.0)
        np.testing.assert_allclose(problem.q[12:], 0.0, atol=0.0)
        np.testing.assert_allclose(problem.P.toarray(), same_problem_other_soc_reference.P.toarray())
        np.testing.assert_allclose(problem.q, same_problem_other_soc_reference.q)

        dense_a = problem.A.toarray()
        np.testing.assert_allclose(problem.l[12:19], 0.2)
        np.testing.assert_allclose(problem.u[12:19], 0.8)
        self.assertEqual(dense_a[20, 12], -1.0)
        self.assertEqual(dense_a[20, 13], 1.0)
        self.assertAlmostEqual(dense_a[20, 6], 1.0 / (3600.0 * 693.0))

    def test_osqp_scaling_is_an_exact_affine_reparameterization(self) -> None:
        from mpc_solvers.mpc_qp_formulation import build_qp_problem
        from run_mpc_1s_n6_weight_selection import candidate_config, scale_n6_qp_problem

        config = candidate_config("A")
        problem = build_qp_problem(
            config,
            load_forecast_kw=np.array([100.0, 110.0, 120.0, 130.0, 140.0, 150.0]),
            current_soc=0.55,
            prev_fc_kw=100.0,
            soc_reference=0.55,
        )
        scaled, transform = scale_n6_qp_problem(problem, config=config)
        physical = np.concatenate(
            [
                np.linspace(100.0, 150.0, 6),
                np.zeros(6),
                np.linspace(0.55, 0.5499, 7),
            ]
        )
        normalized = transform.to_normalized(physical)

        np.testing.assert_allclose(transform.to_physical(normalized), physical, atol=1.0e-12)
        np.testing.assert_allclose(
            scaled.A @ normalized - scaled.l,
            transform.row_scale * (problem.A @ physical - problem.l),
            atol=1.0e-12,
        )
        physical_objective = float(0.5 * physical @ problem.P @ physical + problem.q @ physical)
        normalized_objective = float(
            0.5 * normalized @ scaled.P @ normalized
            + scaled.q @ normalized
            + transform.objective_constant
        )
        self.assertAlmostEqual(physical_objective, normalized_objective, places=9)
        self.assertEqual(scaled.P.shape, problem.P.shape)
        self.assertEqual(scaled.A.shape, problem.A.shape)

    def test_scaled_linear_refresh_matches_fresh_problem_for_previous_fc(self) -> None:
        from mpc_solvers.mpc_qp_formulation import build_qp_problem
        from run_mpc_1s_n6_weight_selection import (
            candidate_config,
            scale_n6_qp_problem,
            scaled_linear_for_previous_fc,
        )

        config = candidate_config("B")
        loads = np.linspace(200.0, 250.0, 6)
        base_previous_fc = 180.0
        next_previous_fc = 213.0
        base_problem = build_qp_problem(
            config,
            load_forecast_kw=loads,
            current_soc=0.55,
            prev_fc_kw=base_previous_fc,
            soc_reference=0.55,
        )
        scaled_base, transform = scale_n6_qp_problem(base_problem, config=config)
        refreshed = scaled_linear_for_previous_fc(
            scaled_base.q,
            config=config,
            transform=transform,
            base_previous_fc_kw=base_previous_fc,
            previous_fc_kw=next_previous_fc,
        )

        fresh_problem = build_qp_problem(
            config,
            load_forecast_kw=loads,
            current_soc=0.55,
            prev_fc_kw=next_previous_fc,
            soc_reference=0.55,
        )
        scaled_fresh, _ = scale_n6_qp_problem(fresh_problem, config=config)
        np.testing.assert_allclose(refreshed, scaled_fresh.q, rtol=0.0, atol=1.0e-12)

    def test_n6_osqp_settings_are_tight_and_deterministic(self) -> None:
        from run_mpc_1s_n6_weight_selection import N6_OSQP_SETTINGS

        self.assertEqual(N6_OSQP_SETTINGS["eps_abs"], 1.0e-5)
        self.assertEqual(N6_OSQP_SETTINGS["eps_rel"], 1.0e-5)
        self.assertEqual(N6_OSQP_SETTINGS["max_iter"], 10000)
        self.assertEqual(N6_OSQP_SETTINGS["adaptive_rho_interval"], 25)
        self.assertTrue(N6_OSQP_SETTINGS["polishing"])
        self.assertTrue(N6_OSQP_SETTINGS["warm_starting"])


class TestN6RollingExecution(unittest.TestCase):
    def test_voyage_rolls_t_plus_1_horizon_and_executes_one_step(self) -> None:
        from run_mpc_1s_n6_weight_selection import run_voyage

        loads = np.array([100, 110, 120, 130, 140, 150, 160, 170], dtype=float)
        times = np.arange(len(loads), dtype=float)
        controls, solver = run_voyage(
            voyage_id="voyage_test",
            loads_kw=loads,
            times_s=times,
            candidate_id="A",
        )

        self.assertEqual(len(controls), 7)
        self.assertEqual(len(solver), 7)
        self.assertEqual(controls["decision_index"].tolist(), list(range(7)))
        self.assertEqual(controls["execution_index"].tolist(), list(range(1, 8)))
        np.testing.assert_allclose(controls["load_actual_kw"], loads[1:])
        self.assertEqual(controls.iloc[0]["load_h1_kw"], 110.0)
        self.assertEqual(controls.iloc[0]["load_h6_kw"], 160.0)
        self.assertEqual(controls.iloc[-1]["load_h1_kw"], 170.0)
        self.assertEqual(controls.iloc[-1]["load_h6_kw"], 170.0)
        self.assertTrue(solver["success"].all())
        np.testing.assert_allclose(
            controls["P_batt_actual_kw"],
            controls["load_actual_kw"] - controls["P_fc_actual_kw"],
            atol=1.0e-12,
        )
        np.testing.assert_allclose(controls["actual_balance_residual_kw"], 0.0, atol=1.0e-12)
        self.assertEqual(controls.iloc[0]["prev_fc_actual_kw"], 100.0)
        self.assertAlmostEqual(
            controls.iloc[1]["prev_fc_actual_kw"],
            controls.iloc[0]["P_fc_actual_kw"],
        )
        expected_first_soc = 0.55 - controls.iloc[0]["P_batt_actual_kw"] / (3600.0 * 693.0)
        self.assertAlmostEqual(controls.iloc[0]["SOC_actual"], expected_first_soc)
        self.assertTrue(
            {
                "P_batt_plan_kw",
                "P_batt_actual_kw",
                "SOC_predicted",
                "SOC_actual",
            }.issubset(controls.columns)
        )
        self.assertIsInstance(controls, pd.DataFrame)

    def test_voyage_updates_linear_cost_for_rolling_previous_fc(self) -> None:
        import run_mpc_1s_n6_weight_selection as module

        normalized_solution = np.zeros(19, dtype=float)
        normalized_solution[0] = 110.0 / 560.0
        solved_result = SimpleNamespace(
            x=normalized_solution,
            info=SimpleNamespace(
                status="solved",
                iter=25,
                prim_res=1.0e-8,
                dual_res=1.0e-8,
            ),
        )
        with (
            patch.object(module, "_try_import_osqp", return_value=(object(), None)),
            patch.object(module, "_setup_n6_osqp_solver", return_value="solver"),
            patch.object(
                module,
                "_solve_with_persistent_osqp",
                return_value=(solved_result, 0.1),
            ) as solve,
        ):
            module.run_voyage(
                voyage_id="v1",
                loads_kw=np.array([100.0, 120.0, 130.0]),
                times_s=np.array([0.0, 1.0, 2.0]),
                candidate_id="B",
            )

        first_linear = solve.call_args_list[0].kwargs["linear"]
        second_linear = solve.call_args_list[1].kwargs["linear"]
        self.assertEqual(len(first_linear), 19)
        self.assertFalse(np.isclose(first_linear[0], second_linear[0], rtol=0.0, atol=1.0e-12))
        np.testing.assert_allclose(first_linear[1:], second_linear[1:])

    def test_max_iter_is_cold_restarted_without_inventing_a_control_fallback(self) -> None:
        import run_mpc_1s_n6_weight_selection as module

        max_iter_result = SimpleNamespace(
            x=None,
            info=SimpleNamespace(
                status="maximum iterations reached",
                iter=4000,
                prim_res=0.1,
                dual_res=0.2,
            ),
        )
        solution = np.zeros(19, dtype=float)
        solution[0] = 110.0
        solution[6] = 0.0
        solution[13] = 0.55
        solved_result = SimpleNamespace(
            x=solution,
            info=SimpleNamespace(
                status="solved",
                iter=75,
                prim_res=1.0e-5,
                dual_res=2.0e-5,
            ),
        )

        with (
            patch.object(module, "_try_import_osqp", return_value=(object(), None)),
            patch.object(
                module,
                "_setup_n6_osqp_solver",
                side_effect=["warm_solver", "cold_solver"],
            ) as setup,
            patch.object(
                module,
                "_solve_with_persistent_osqp",
                side_effect=[(max_iter_result, 2.0), (solved_result, 0.5)],
            ),
        ):
            controls, solver = module.run_voyage(
                voyage_id="v1",
                loads_kw=np.array([100.0, 110.0]),
                times_s=np.array([0.0, 1.0]),
                candidate_id="A",
            )

        self.assertEqual(setup.call_count, 2)
        self.assertTrue(bool(controls.iloc[0]["success"]))
        self.assertEqual(solver.iloc[0]["status"], "solved")
        self.assertEqual(solver.iloc[0]["initial_status"], "maximum iterations reached")
        self.assertTrue(bool(solver.iloc[0]["max_iter_reached"]))
        self.assertTrue(bool(solver.iloc[0]["cold_restart_used"]))
        self.assertTrue(bool(solver.iloc[0]["cold_restart_succeeded"]))
        self.assertEqual(solver.iloc[0]["attempt_count"], 2)
        self.assertGreaterEqual(solver.iloc[0]["solve_ms"], 2.5)

    def test_voyage_stops_at_first_final_solver_failure(self) -> None:
        import run_mpc_1s_n6_weight_selection as module

        infeasible_result = SimpleNamespace(
            x=None,
            info=SimpleNamespace(
                status="primal infeasible",
                iter=100,
                prim_res=0.1,
                dual_res=0.2,
            ),
        )
        with (
            patch.object(module, "_try_import_osqp", return_value=(object(), None)),
            patch.object(module, "_setup_n6_osqp_solver", return_value="solver"),
            patch.object(
                module,
                "_solve_with_persistent_osqp",
                return_value=(infeasible_result, 1.0),
            ) as solve,
        ):
            controls, solver = module.run_voyage(
                voyage_id="v1",
                loads_kw=np.array([100.0, 110.0, 120.0, 130.0]),
                times_s=np.array([0.0, 1.0, 2.0, 3.0]),
                candidate_id="A",
            )

        self.assertEqual(solve.call_count, 1)
        self.assertEqual(len(controls), 1)
        self.assertEqual(len(solver), 1)
        self.assertFalse(bool(controls.iloc[0]["success"]))
        self.assertTrue(np.isnan(controls.iloc[0]["SOC_actual"]))
        self.assertEqual(controls.iloc[0]["voyage_expected_steps"], 3)
        self.assertEqual(solver.iloc[0]["voyage_expected_steps"], 3)


class TestN6Metrics(unittest.TestCase):
    def test_metrics_follow_physical_energy_and_solver_status_definitions(self) -> None:
        from run_mpc_1s_n6_weight_selection import (
            REQUIRED_N6_METRIC_KEYS,
            build_candidate_metrics,
            candidate_config,
        )

        controls = pd.DataFrame(
            {
                "voyage_id": ["v1", "v1", "v1"],
                "success": [True, True, False],
                "load_actual_kw": [100.0, 100.0, 100.0],
                "P_fc_actual_kw": [100.5, 90.0, 100.0],
                "P_batt_actual_kw": [-0.5, 10.0, 0.0],
                "fc_delta_actual_kw": [10.0, -20.0, 10.0],
                "SOC_before": [0.55, 0.550004, 0.55],
                "SOC_actual": [0.550004, 0.55, 0.55],
                "h2_kg_step": [1.0, 2.0, 3.0],
                "actual_balance_residual_kw": [0.0, 0.001, 0.0],
                "plan_balance_residual_kw": [0.002, 0.003, 0.004],
                "fc_bound_residual_kw": [0.0, 0.0, 0.0],
                "battery_bound_residual_kw": [0.0, 0.0, 0.0],
                "ramp_residual_kw": [0.0, 0.05, 0.0],
                "soc_bound_residual": [0.0, 0.0, 0.0],
                "soc_prediction_residual": [1.0e-8, 2.0e-8, 3.0e-8],
            }
        )
        solver = pd.DataFrame(
            {
                "voyage_id": ["v1", "v1", "v1"],
                "status": ["solved", "solved inaccurate", "maximum iterations reached"],
                "success": [True, True, False],
                "solved_inaccurate": [False, True, False],
                "max_iter_reached": [False, False, True],
                "solve_ms": [1.0, 2.0, 3.0],
                "iterations": [10, 20, 4000],
                "primal_residual": [1e-5, 2e-5, 1e-2],
                "dual_residual": [2e-5, 3e-5, 2e-2],
            }
        )

        summary, voyage_metrics, solver_statistics = build_candidate_metrics(
            controls,
            solver,
            config=candidate_config("A"),
        )

        self.assertEqual(summary["total_steps"], 3)
        self.assertAlmostEqual(summary["solver_success_rate"], 2.0 / 3.0)
        self.assertAlmostEqual(summary["solved_inaccurate_fraction"], 1.0 / 3.0)
        self.assertEqual(summary["max_iter_count"], 1)
        self.assertEqual(summary["final_max_iter_count"], 1)
        self.assertEqual(summary["primal_infeasible_count"], 0)
        self.assertAlmostEqual(summary["solve_time_ms_mean"], 2.0)
        self.assertAlmostEqual(summary["solve_time_ms_p95"], 2.9)
        self.assertAlmostEqual(summary["solve_time_ms_p99"], 2.98)
        self.assertAlmostEqual(summary["solve_time_ms_max"], 3.0)
        self.assertAlmostEqual(summary["hydrogen_total_kg"], 3.0)
        self.assertAlmostEqual(summary["load_energy_mwh"], 200.0 / 3600.0 / 1000.0)
        self.assertAlmostEqual(
            summary["hydrogen_intensity_kg_per_mwh"],
            3.0 / (200.0 / 3600.0 / 1000.0),
        )
        self.assertAlmostEqual(summary["battery_charge_energy_kwh"], 0.5 / 3600.0)
        self.assertAlmostEqual(summary["battery_discharge_energy_kwh"], 10.0 / 3600.0)
        self.assertAlmostEqual(summary["battery_throughput_kwh"], 10.5 / 3600.0)
        self.assertAlmostEqual(
            summary["fc_power_change_rms_kw"],
            np.sqrt((10.0**2 + 20.0**2) / 2.0),
        )
        self.assertEqual(summary["completion_rate"], 0.0)
        self.assertAlmostEqual(summary["fc_above_load_fraction"], 1.0 / 2.0)
        self.assertAlmostEqual(summary["fc_surplus_energy_kwh"], 0.5 / 3600.0)
        self.assertEqual(summary["fc_at_max_fraction"], 0.0)
        self.assertEqual(summary["battery_near_limit_fraction"], 0.0)
        self.assertAlmostEqual(summary["initial_soc"], 0.55)
        self.assertAlmostEqual(summary["final_soc"], 0.55)
        self.assertAlmostEqual(summary["soc_net_change"], 0.0)
        self.assertAlmostEqual(summary["soc_min"], 0.55)
        self.assertAlmostEqual(summary["soc_max"], 0.550004)
        self.assertEqual(summary["physical_infeasible_point_count"], 0)
        self.assertAlmostEqual(summary["max_soc_prediction_residual"], 3.0e-8)
        self.assertTrue(REQUIRED_N6_METRIC_KEYS.issubset(summary))
        self.assertEqual(len(voyage_metrics), 1)
        self.assertEqual(set(solver_statistics["scope"]), {"overall", "voyage"})

    def test_incomplete_voyage_metrics_are_prefix_only_and_not_comparable(self) -> None:
        from run_mpc_1s_n6_weight_selection import build_candidate_metrics, candidate_config

        controls = pd.DataFrame(
            {
                "voyage_id": ["v1"],
                "voyage_expected_steps": [3],
                "success": [False],
                "load_actual_kw": [100.0],
                "P_fc_actual_kw": [np.nan],
                "P_batt_actual_kw": [np.nan],
                "SOC_before": [0.55],
                "SOC_actual": [np.nan],
                "h2_kg_step": [np.nan],
                "actual_balance_residual_kw": [np.nan],
                "plan_balance_residual_kw": [np.nan],
                "fc_bound_residual_kw": [np.nan],
                "battery_bound_residual_kw": [np.nan],
                "ramp_residual_kw": [np.nan],
                "soc_bound_residual": [np.nan],
                "soc_prediction_residual": [np.nan],
            }
        )
        solver = pd.DataFrame(
            {
                "voyage_id": ["v1"],
                "voyage_expected_steps": [3],
                "status": ["primal infeasible"],
                "success": [False],
                "solved_inaccurate": [False],
                "max_iter_reached": [False],
                "solve_ms": [0.5],
                "iterations": [100],
                "primal_residual": [0.1],
                "dual_residual": [0.2],
            }
        )

        summary, _, _ = build_candidate_metrics(
            controls,
            solver,
            config=candidate_config("A"),
        )

        self.assertEqual(summary["total_steps"], 3)
        self.assertEqual(summary["attempted_steps"], 1)
        self.assertEqual(summary["applied_steps"], 0)
        self.assertEqual(summary["closed_loop_coverage_fraction"], 0.0)
        self.assertFalse(summary["closed_loop_complete"])
        self.assertFalse(summary["aggregate_metrics_comparable"])
        self.assertEqual(summary["solver_failure_count"], 1)
        self.assertEqual(summary["unattempted_after_failure_count"], 2)
        self.assertEqual(summary["primal_infeasible_count"], 1)
        self.assertEqual(summary["hydrogen_total_kg"], 0.0)
        self.assertEqual(summary["load_energy_mwh"], 0.0)

    def test_physical_hydrogen_is_independent_of_optimization_weight(self) -> None:
        from mpc_solvers.mpc_qp_formulation import QpMpcConfig
        from run_mpc_1s_n6_weight_selection import physical_h2_kg_step

        low_weight = QpMpcConfig(horizon=6, q_h2=0.5)
        high_weight = QpMpcConfig(horizon=6, q_h2=50.0)
        self.assertAlmostEqual(
            physical_h2_kg_step(low_weight, 280.0),
            physical_h2_kg_step(high_weight, 280.0),
        )


class TestN6Artifacts(unittest.TestCase):
    def test_core_table_contains_only_requested_summary_columns(self) -> None:
        from run_mpc_1s_n6_weight_selection import (
            CORE_TABLE_COLUMNS,
            write_core_summary_table,
        )

        summaries = []
        for candidate_id in "ABC":
            summaries.append(
                {
                    "candidate_id": candidate_id,
                    "completion_rate": 1.0,
                    "solver_failure_count": 0,
                    "worst_voyage_soc_net_change": -0.1,
                    "soc_min": 0.45,
                    "soc_max": 0.55,
                    "hydrogen_total_kg": 1.2,
                    "battery_charge_energy_kwh": 2.3,
                    "battery_discharge_energy_kwh": 4.5,
                    "battery_throughput_kwh": 6.8,
                    "fc_power_change_rms_kw": 7.8,
                    "fc_above_load_fraction": 0.1,
                    "solve_time_ms_mean": 0.2,
                    "solve_time_ms_p99": 0.4,
                }
            )

        with tempfile.TemporaryDirectory() as temporary_directory:
            table_path = write_core_summary_table(
                summaries,
                reports_dir=Path(temporary_directory),
            )
            table = pd.read_csv(table_path)

            self.assertEqual(tuple(table.columns), CORE_TABLE_COLUMNS)
            self.assertEqual(table["candidate_id"].tolist(), list("ABC"))
            self.assertEqual(table["global_soc_range"].tolist(), ["0.45..0.55"] * 3)
            self.assertEqual(
                [path.name for path in Path(temporary_directory).iterdir()],
                ["mpc_1s_n6_h2_fcvar_batt_table.csv"],
            )

    def test_core_plot_writer_retains_only_one_three_panel_png_per_voyage(self) -> None:
        from run_mpc_1s_n6_weight_selection import write_core_candidate_plots

        controls = pd.DataFrame(
            {
                "voyage_id": ["voyage_060", "voyage_060"],
                "time_s": [1.0, 2.0],
                "load_actual_kw": [100.0, 110.0],
                "P_fc_actual_kw": [100.0, 105.0],
                "P_batt_actual_kw": [0.0, 5.0],
                "SOC_actual": [0.55, 0.549998],
                "success": [True, True],
            }
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            candidate_dir = write_core_candidate_plots(
                controls,
                output_root=Path(temporary_directory),
                candidate_id="A",
            )

            self.assertEqual(candidate_dir.name, "candidate_A")
            self.assertEqual(
                [path.name for path in candidate_dir.iterdir()],
                ["voyage_060_power_soc.png"],
            )

    def test_core_experiment_runs_only_three_candidates_and_retains_core_outputs(self) -> None:
        import run_mpc_1s_n6_weight_selection as module

        controls = pd.DataFrame(
            {
                "voyage_id": ["voyage_060", "voyage_060"],
                "time_s": [1.0, 2.0],
                "load_actual_kw": [100.0, 110.0],
                "P_fc_actual_kw": [100.0, 105.0],
                "P_batt_actual_kw": [0.0, 5.0],
                "SOC_actual": [0.55, 0.549998],
                "success": [True, True],
            }
        )

        def fake_evaluate(candidate_id: str, **_: object) -> dict[str, object]:
            summary = {
                "candidate_id": candidate_id,
                "completion_rate": 1.0,
                "solver_failure_count": 0,
                "worst_voyage_soc_net_change": -0.01,
                "soc_min": 0.54,
                "soc_max": 0.55,
                "hydrogen_total_kg": 1.0,
                "battery_charge_energy_kwh": 0.1,
                "battery_discharge_energy_kwh": 0.2,
                "battery_throughput_kwh": 0.3,
                "fc_power_change_rms_kw": 4.0,
                "fc_above_load_fraction": 0.1,
                "solve_time_ms_mean": 0.2,
                "solve_time_ms_p99": 0.4,
            }
            return {"candidate_id": candidate_id, "summary": summary, "controls": controls}

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with patch.object(module, "_evaluate_candidate", side_effect=fake_evaluate) as evaluate:
                table = module.run_core_experiment(
                    input_path=root / "unused.parquet",
                    output_root=root / "outputs",
                    reports_dir=root / "reports",
                    expected_voyage_count=1,
                )

            self.assertEqual(
                [call.args[0] for call in evaluate.call_args_list],
                list("ABC"),
            )
            self.assertEqual(table["candidate_id"].tolist(), list("ABC"))
            self.assertEqual(
                sorted(path.relative_to(root / "outputs").as_posix() for path in (root / "outputs").rglob("*")),
                [
                    "candidate_A",
                    "candidate_A/voyage_060_power_soc.png",
                    "candidate_B",
                    "candidate_B/voyage_060_power_soc.png",
                    "candidate_C",
                    "candidate_C/voyage_060_power_soc.png",
                ],
            )
            self.assertEqual(
                [path.name for path in (root / "reports").iterdir()],
                ["mpc_1s_n6_h2_fcvar_batt_table.csv"],
            )

    def test_candidate_artifacts_are_compact_and_self_describing(self) -> None:
        from run_mpc_1s_n6_weight_selection import write_candidate_artifacts

        summary = {
            "candidate_id": "A",
            "total_steps": 2,
            "solver_success_rate": 1.0,
            "physical_infeasible_point_count": 0,
        }
        voyage_metrics = pd.DataFrame(
            [{"candidate_id": "A", "voyage_id": "v1", "total_steps": 2}]
        )
        solver_statistics = pd.DataFrame(
            [
                {
                    "candidate_id": "A",
                    "scope": "overall",
                    "voyage_id": "all",
                    "total_steps": 2,
                    "solver_success_rate": 1.0,
                }
            ]
        )
        controls = pd.DataFrame(
            {
                "voyage_id": ["v1", "v1"],
                "time_s": [1.0, 2.0],
                "load_actual_kw": [100.0, 110.0],
                "P_fc_actual_kw": [100.0, 105.0],
                "P_batt_actual_kw": [0.0, 5.0],
                "SOC_actual": [0.55, 0.549998],
            }
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            case_dir = write_candidate_artifacts(
                candidate_id="A",
                summary=summary,
                voyage_metrics=voyage_metrics,
                solver_statistics=solver_statistics,
                controls=controls,
                output_root=Path(temporary_directory),
                input_path=Path("outputs/mpc_solver_benchmark_1s/data/test_voyages_spline_1s.parquet"),
                make_plots=False,
            )

            self.assertEqual(case_dir.name, "candidate_A")
            required = {
                "config.json",
                "summary_metrics.json",
                "voyage_metrics.csv",
                "solver_statistics.csv",
                "constraint_audit.md",
            }
            self.assertTrue(required.issubset({path.name for path in case_dir.iterdir()}))
            self.assertFalse((case_dir / "control_timeseries.csv").exists())
            self.assertFalse((case_dir / "solver_step_log.csv").exists())

            config = json.loads((case_dir / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["candidate_id"], "A")
            self.assertEqual(config["timing"]["forecast_samples"], "t+1..t+6")
            self.assertEqual(config["timing"]["applied_action"], "first step only")
            self.assertEqual(config["forecast_source"], "offline ideal foresight")
            self.assertEqual(config["status"], "raw_candidate")
            self.assertEqual(config["tolerances"]["fc_above_load_kw"], 1.0e-6)

            audit = (case_dir / "constraint_audit.md").read_text(encoding="utf-8")
            self.assertIn("Numerical tolerance", audit)
            self.assertIn("physical_infeasible_point_count", audit)

    def test_candidate_run_validates_spline_input_and_writes_only_aggregates(self) -> None:
        from run_mpc_1s_n6_weight_selection import run_candidate

        rows: list[dict[str, object]] = []
        for voyage_id, offset in (("v1", 0.0), ("v2", 20.0)):
            for time_s in range(8):
                rows.append(
                    {
                        "voyage_id": voyage_id,
                        "split": "test",
                        "time_s": float(time_s),
                        "load_total_kw": 100.0 + offset + float(time_s),
                        "dataset_version": "cubic_spline_1s_natural_clipped",
                    }
                )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "test_spline.parquet"
            pd.DataFrame(rows).to_parquet(input_path, index=False)
            result = run_candidate(
                "A",
                input_path=input_path,
                output_root=root / "outputs",
                make_plots=False,
                max_steps_per_voyage=2,
            )

            self.assertEqual(result["summary"]["candidate_id"], "A")
            self.assertEqual(result["summary"]["total_steps"], 4)
            self.assertEqual(result["summary"]["voyage_count"], 2)
            self.assertEqual(len(result["voyage_metrics"]), 2)
            self.assertTrue((root / "outputs" / "candidate_A" / "summary_metrics.json").exists())
            self.assertFalse((root / "outputs" / "candidate_A" / "control_timeseries.csv").exists())

    def test_combined_report_is_manual_selection_aware(self) -> None:
        from run_mpc_1s_n6_weight_selection import write_combined_reports

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_root = root / "outputs"
            for candidate_id in "ABC":
                case_dir = output_root / f"candidate_{candidate_id}"
                case_dir.mkdir(parents=True)
                complete = candidate_id == "A"
                (case_dir / "summary_metrics.json").write_text(
                    json.dumps(
                        {
                            "candidate_id": candidate_id,
                            "voyage_count": 7,
                            "is_partial_debug_run": False,
                            "closed_loop_complete": complete,
                            "aggregate_metrics_comparable": complete,
                            "total_steps": 10,
                            "solver_success_rate": 1.0 if complete else 0.9,
                            "solver_failure_count": 0 if complete else 1,
                            "physical_infeasible_point_count": 0,
                            "soc_min": 0.50,
                            "soc_max": 0.56,
                            "final_soc": 0.55,
                            "worst_voyage_soc_net_change": -0.001,
                            "hydrogen_total_kg": 1.0,
                            "battery_throughput_kwh": 2.0,
                            "fc_above_load_fraction": 0.1,
                            "fc_surplus_energy_kwh": 0.2,
                            "solve_time_ms_p99": 0.3,
                            "solve_time_ms_max": 0.5,
                        }
                    ),
                    encoding="utf-8",
                )
            reports_dir = root / "reports"
            selection = {
                "status": "provisional",
                "selected_candidate": "A",
                "selection_method": "manual engineering review",
                "selection_reasons": ["physical feasibility first"],
                "candidate_decisions": {
                    "A": "retained as the provisional anchor",
                    "B": "rejected",
                    "C": "rejected",
                },
            }

            markdown_path, table_path = write_combined_reports(
                output_root=output_root,
                reports_dir=reports_dir,
                selection=selection,
            )

            report = markdown_path.read_text(encoding="utf-8")
            self.assertIn("natural-clipped cubic-spline", report)
            self.assertIn("future six true samples", report)
            self.assertIn("first action", report)
            self.assertIn("N=60", report)
            self.assertIn("provisional", report)
            self.assertIn("manual engineering review", report)
            self.assertNotIn("least-bad ranking", report)
            table = pd.read_csv(table_path)
            self.assertEqual(table["candidate_id"].tolist(), list("ABC"))

    def test_combined_report_rejects_selection_of_incomplete_candidate(self) -> None:
        from run_mpc_1s_n6_weight_selection import write_combined_reports

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for candidate_id in "ABC":
                case_dir = root / "outputs" / f"candidate_{candidate_id}"
                case_dir.mkdir(parents=True)
                (case_dir / "summary_metrics.json").write_text(
                    json.dumps(
                        {
                            "candidate_id": candidate_id,
                            "voyage_count": 7,
                            "is_partial_debug_run": False,
                            "closed_loop_complete": candidate_id != "A",
                            "physical_infeasible_point_count": 0,
                        }
                    ),
                    encoding="utf-8",
                )

            with self.assertRaisesRegex(ValueError, "closed-loop complete"):
                write_combined_reports(
                    output_root=root / "outputs",
                    reports_dir=root / "reports",
                    selection={
                        "status": "provisional",
                        "selected_candidate": "A",
                        "selection_method": "manual engineering review",
                        "selection_reasons": ["physical review"],
                        "candidate_decisions": {item: "reviewed" for item in "ABC"},
                    },
                )

    def test_accepted_selection_requires_recorded_verification(self) -> None:
        from run_mpc_1s_n6_weight_selection import write_combined_reports

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for candidate_id in "ABC":
                case_dir = root / "outputs" / f"candidate_{candidate_id}"
                case_dir.mkdir(parents=True)
                (case_dir / "summary_metrics.json").write_text(
                    json.dumps(
                        {
                            "candidate_id": candidate_id,
                            "voyage_count": 7,
                            "is_partial_debug_run": False,
                            "closed_loop_complete": candidate_id == "A",
                            "aggregate_metrics_comparable": candidate_id == "A",
                            "solver_failure_count": 0 if candidate_id == "A" else 1,
                            "physical_infeasible_point_count": 0,
                            "soc_min": 0.5,
                            "soc_max": 0.56,
                            "worst_voyage_soc_net_change": -0.001,
                            "solve_time_ms_max": 0.5,
                        }
                    ),
                    encoding="utf-8",
                )

            with self.assertRaisesRegex(ValueError, "verification"):
                write_combined_reports(
                    output_root=root / "outputs",
                    reports_dir=root / "reports",
                    selection={
                        "status": "accepted",
                        "selected_candidate": "A",
                        "selection_method": "manual engineering review",
                        "selection_reasons": ["all gates passed"],
                        "candidate_decisions": {item: "reviewed" for item in "ABC"},
                        "engineering_review_complete": True,
                    },
                )

    def test_combined_report_accepts_explicit_no_candidate_decision(self) -> None:
        from run_mpc_1s_n6_weight_selection import write_combined_reports

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for candidate_id in "ABC":
                case_dir = root / "outputs" / f"candidate_{candidate_id}"
                case_dir.mkdir(parents=True)
                (case_dir / "summary_metrics.json").write_text(
                    json.dumps(
                        {
                            "candidate_id": candidate_id,
                            "voyage_count": 7,
                            "is_partial_debug_run": False,
                            "closed_loop_complete": False,
                            "closed_loop_coverage_fraction": 0.7,
                            "physical_infeasible_point_count": 0,
                        }
                    ),
                    encoding="utf-8",
                )
            report_path, _ = write_combined_reports(
                output_root=root / "outputs",
                reports_dir=root / "reports",
                selection={
                    "status": "no_candidate_selected",
                    "selected_candidate": None,
                    "selection_method": "manual engineering review",
                    "selection_reasons": ["all candidates failed the physical gate"],
                    "candidate_decisions": {item: "rejected" for item in "ABC"},
                },
            )

            report = report_path.read_text(encoding="utf-8")
            self.assertIn("no_candidate_selected", report)
            self.assertIn("Selected candidate: none", report)

    def test_cli_runs_the_fixed_three_candidate_experiment(self) -> None:
        import run_mpc_1s_n6_weight_selection as module

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with patch.object(
                module,
                "run_core_experiment",
                return_value=pd.DataFrame({"candidate_id": list("ABC")}),
            ) as run_experiment:
                exit_code = module.main(
                    [
                        "--input",
                        str(root / "input.parquet"),
                        "--output-root",
                        str(root / "outputs"),
                        "--reports-dir",
                        str(root / "reports"),
                        "--no-plots",
                        "--max-steps-per-voyage",
                        "2",
                        "--expected-voyages",
                        "1",
                    ]
                )

            self.assertEqual(exit_code, 0)
            kwargs = run_experiment.call_args.kwargs
            self.assertFalse(kwargs["make_plots"])
            self.assertEqual(kwargs["max_steps_per_voyage"], 2)
            self.assertEqual(kwargs["expected_voyage_count"], 1)


if __name__ == "__main__":
    unittest.main()
