from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
MAIN_ROOT = SRC / "main"
for path in (SRC, MAIN_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


class TestMpcSolverBenchmark1s(unittest.TestCase):
    def test_ramp_rate_48kw_per_second_maps_to_48kw_per_1s_step(self) -> None:
        from mpc_solvers.mpc_qp_formulation import ramp_kw_per_step_from_rate

        self.assertAlmostEqual(ramp_kw_per_step_from_rate(48.0, dt_seconds=1.0), 48.0)
        self.assertAlmostEqual(ramp_kw_per_step_from_rate(48.0, dt_seconds=30.0), 1440.0)

    def test_qp_formulation_is_convex_and_records_1s_ramp_rate_source(self) -> None:
        from mpc_solvers.mpc_qp_formulation import QpMpcConfig, build_qp_problem, hessian_min_eigenvalue

        cfg = QpMpcConfig(
            horizon=3,
            dt_seconds=1.0,
            battery_capacity_kwh=1806.0,
            fuel_cell_max_kw=560.0,
            fuel_cell_ramp_rate_kw_per_s=48.0,
            fuel_cell_ramp_kw=None,
            q_h2=1.0,
            q_soc=400.0,
            q_batt=0.03,
            q_ramp=2e-5,
            q_terminal_soc=0.0,
        )
        problem = build_qp_problem(
            cfg,
            load_forecast_kw=np.array([80.0, 90.0, 85.0]),
            current_soc=0.55,
            prev_fc_kw=20.0,
            soc_reference=0.55,
        )

        self.assertGreaterEqual(hessian_min_eigenvalue(problem), -1e-10)
        self.assertTrue(problem.metadata["convex_qp"])
        self.assertEqual(problem.metadata["variable_order"], "P_fc[0:N], P_batt[0:N], SOC[0:N+1]")
        self.assertAlmostEqual(problem.metadata["fuel_cell_ramp_rate_kw_per_s"], 48.0)
        self.assertAlmostEqual(problem.metadata["fuel_cell_ramp_kw_per_step"], 48.0)

    def test_build_benchmark_dataset_uses_only_test_voyages_and_preserves_flags(self) -> None:
        from build_mpc_solver_benchmark_1s_data import build_benchmark_dataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "natural_clipped_by_voyage"
            out_dir = root / "benchmark"
            source_dir.mkdir()
            rows = [
                {
                    "dataset_version": "cubic_spline_1s_natural_clipped",
                    "voyage_id": "voyage_001",
                    "split": "train",
                    "time_original_or_reconstructed": "original_30s_point",
                    "timestamp": "2024-01-01 00:00:00",
                    "time_s": 0.0,
                    "load_total_kw": 10.0,
                    "is_original_30s_point": True,
                    "online_feasible": False,
                    "uses_future_endpoint": True,
                    "file_name": "train.xlsx",
                },
                {
                    "dataset_version": "cubic_spline_1s_natural_clipped",
                    "voyage_id": "voyage_066",
                    "split": "test",
                    "time_original_or_reconstructed": "original_30s_point",
                    "timestamp": "2024-01-02 00:00:00",
                    "time_s": 0.0,
                    "load_total_kw": 20.0,
                    "is_original_30s_point": True,
                    "online_feasible": False,
                    "uses_future_endpoint": True,
                    "file_name": "test.xlsx",
                },
                {
                    "dataset_version": "cubic_spline_1s_natural_clipped",
                    "voyage_id": "voyage_066",
                    "split": "test",
                    "time_original_or_reconstructed": "reconstructed_1s_point",
                    "timestamp": "2024-01-02 00:00:01",
                    "time_s": 1.0,
                    "load_total_kw": 21.0,
                    "is_original_30s_point": False,
                    "online_feasible": False,
                    "uses_future_endpoint": True,
                    "file_name": "test.xlsx",
                },
            ]
            pd.DataFrame([rows[0]]).to_csv(source_dir / "voyage_001__train.csv", index=False)
            pd.DataFrame(rows[1:]).to_csv(source_dir / "voyage_066__test.csv", index=False)

            result = build_benchmark_dataset(input_dir=source_dir, output_dir=out_dir)

            data = pd.read_parquet(result["parquet_path"])
            self.assertEqual(data["voyage_id"].unique().tolist(), ["voyage_066"])
            self.assertEqual(data["split"].unique().tolist(), ["test"])
            self.assertEqual(len(data), 2)
            self.assertFalse(data["online_feasible"].any())
            self.assertTrue(data["uses_future_endpoint"].all())
            self.assertGreaterEqual(float(data["load_total_kw"].min()), 0.0)

            split = json.loads(Path(result["split_json_path"]).read_text(encoding="utf-8"))
            self.assertEqual(split["test_voyages"], ["voyage_066"])
            self.assertEqual(split["sample_interval_seconds"], 1.0)
            self.assertFalse(split["online_feasible"])
            self.assertTrue(split["uses_future_endpoint"])

    def test_simplified_spec_config_uses_346p5kw_battery_limit_and_no_soft_ramp(self) -> None:
        from benchmark_mpc_qp_osqp_1s import default_config, json_safe_config

        cfg = default_config(horizon=60)
        payload = json_safe_config(cfg, osqp_available=True)

        self.assertAlmostEqual(cfg.battery_capacity_kwh, 693.0)
        self.assertAlmostEqual(cfg.battery_charge_max_kw, 346.5)
        self.assertAlmostEqual(cfg.battery_discharge_max_kw, 346.5)
        self.assertAlmostEqual(cfg.battery_power_ref_kw, 346.5)
        self.assertEqual(cfg.objective_variant, "simplified_normalized_literature_v1")
        self.assertAlmostEqual(payload["old_battery_capacity_kwh"], 1806.0)
        self.assertAlmostEqual(payload["battery_capacity_kwh"], 693.0)
        self.assertAlmostEqual(payload["battery_charge_max_kw"], 346.5)
        self.assertAlmostEqual(payload["battery_discharge_max_kw"], 346.5)
        self.assertAlmostEqual(payload["battery_power_ref_kw"], 346.5)
        self.assertAlmostEqual(payload["fuel_cell_ramp_rate_kw_per_s"], 48.0)
        self.assertAlmostEqual(payload["fuel_cell_ramp_kw_per_step"], 48.0)
        self.assertEqual(payload["objective_variant"], "simplified_normalized_literature_v1")
        self.assertAlmostEqual(payload["q_h2"], 1.0)
        self.assertAlmostEqual(payload["q_soc"], 1.0)
        self.assertAlmostEqual(payload["q_batt"], 0.05)
        self.assertAlmostEqual(payload["q_ramp"], 0.0)
        self.assertAlmostEqual(payload["q_terminal_soc"], 0.0)

    def test_simplified_normalized_qp_terms_are_scaled_and_convex(self) -> None:
        from mpc_solvers.mpc_qp_formulation import (
            QpMpcConfig,
            build_qp_problem,
            h2_quadratic_kg_step_coefficients,
            hessian_min_eigenvalue,
        )

        cfg = QpMpcConfig(
            horizon=2,
            dt_seconds=1.0,
            battery_capacity_kwh=693.0,
            battery_charge_max_kw=346.5,
            battery_discharge_max_kw=346.5,
            battery_power_ref_kw=346.5,
            fuel_cell_max_kw=560.0,
            fuel_cell_ramp_rate_kw_per_s=48.0,
            objective_variant="simplified_normalized_literature_v1",
            soc_band=0.05,
            q_h2=1.0,
            q_soc=2.0,
            q_batt=0.05,
            q_ramp=99.0,
            q_terminal_soc=77.0,
        )
        problem = build_qp_problem(
            cfg,
            load_forecast_kw=np.array([100.0, 101.0]),
            current_soc=0.55,
            prev_fc_kw=90.0,
            soc_reference=0.55,
        )
        h2_quad, h2_linear, _, _ = h2_quadratic_kg_step_coefficients(cfg)
        h2_ref = h2_quad * 560.0 * 560.0 + h2_linear * 560.0

        dense_p = problem.P.toarray()
        self.assertEqual(problem.metadata["objective_variant"], "simplified_normalized_literature_v1")
        self.assertTrue(problem.metadata["convex_qp"])
        self.assertGreaterEqual(hessian_min_eigenvalue(problem), -1e-10)
        self.assertAlmostEqual(problem.metadata["h2_reference_kg_per_step"], h2_ref)
        self.assertAlmostEqual(problem.metadata["battery_power_ref_kw"], 346.5)
        self.assertAlmostEqual(problem.metadata["soc_band"], 0.05)

        self.assertAlmostEqual(dense_p[0, 0], 2.0 * cfg.q_h2 * h2_quad / h2_ref)
        self.assertAlmostEqual(dense_p[2, 2], 2.0 * cfg.q_batt / (346.5**2))
        self.assertAlmostEqual(dense_p[5, 5], 2.0 * cfg.q_soc / (0.05**2))
        self.assertAlmostEqual(problem.q[0], cfg.q_h2 * h2_linear / h2_ref)
        self.assertAlmostEqual(problem.q[5], -2.0 * cfg.q_soc * 0.55 / (0.05**2))

        # Soft ramp and terminal penalties must not enter this formal variant.
        self.assertAlmostEqual(dense_p[0, 1], 0.0)
        self.assertNotIn("ramp", problem.metadata["objective_terms"])
        self.assertNotIn("terminal_soc", problem.metadata["objective_terms"])

    def test_qp_build_can_skip_expensive_diagnostics_for_rolling_benchmark(self) -> None:
        from mpc_solvers.mpc_qp_formulation import QpMpcConfig, build_qp_problem

        cfg = QpMpcConfig(horizon=2, dt_seconds=1.0)
        problem = build_qp_problem(
            cfg,
            load_forecast_kw=np.array([100.0, 101.0]),
            current_soc=0.55,
            prev_fc_kw=90.0,
            soc_reference=0.55,
            include_diagnostics=False,
        )

        self.assertFalse(problem.metadata["diagnostics_computed"])
        self.assertTrue(np.isnan(problem.metadata["hessian_min_eigenvalue"]))
        self.assertIsNone(problem.metadata["convex_qp"])

    def test_persistent_osqp_bound_update_matches_full_qp_build(self) -> None:
        from benchmark_mpc_qp_osqp_1s import _qp_bounds_for_step, default_config
        from mpc_solvers.mpc_qp_formulation import build_qp_problem

        cfg = default_config(horizon=3)
        load = np.array([100.0, 240.0, 906.5])
        full = build_qp_problem(
            cfg,
            load_forecast_kw=load,
            current_soc=0.537,
            prev_fc_kw=310.0,
            soc_reference=0.55,
            include_diagnostics=False,
        )
        lower, upper = _qp_bounds_for_step(
            cfg,
            load_forecast_kw=load,
            current_soc=0.537,
            prev_fc_kw=310.0,
        )

        np.testing.assert_allclose(lower, full.l)
        np.testing.assert_allclose(upper, full.u)

    def test_closed_loop_soc_update_uses_applied_battery_power(self) -> None:
        from benchmark_mpc_qp_osqp_1s import advance_soc_from_battery_power, default_config

        cfg = default_config(horizon=2)
        discharge_soc = advance_soc_from_battery_power(cfg, current_soc=0.55, battery_power_kw=346.5)
        charge_soc = advance_soc_from_battery_power(cfg, current_soc=0.55, battery_power_kw=-346.5)

        expected_step = 346.5 / (3600.0 * 693.0)
        self.assertAlmostEqual(discharge_soc, 0.55 - expected_step, places=12)
        self.assertAlmostEqual(charge_soc, 0.55 + expected_step, places=12)

    def test_osqp_polishing_is_enabled_for_fresh_and_persistent_solves(self) -> None:
        from types import SimpleNamespace

        from benchmark_mpc_qp_osqp_1s import _setup_persistent_osqp_solver, _solve_problem

        setup_calls: list[dict[str, object]] = []

        class FakeSolver:
            def setup(self, **kwargs: object) -> None:
                setup_calls.append(kwargs)

            def solve(self) -> object:
                return SimpleNamespace()

        fake_module = SimpleNamespace(OSQP=FakeSolver)
        problem = SimpleNamespace(P=None, q=None, A=None, l=None, u=None)

        _solve_problem(fake_module, problem)
        _setup_persistent_osqp_solver(fake_module, problem)

        self.assertEqual(len(setup_calls), 2)
        self.assertTrue(all(call["polish"] is True for call in setup_calls))

    def test_load_feasibility_check_marks_power_limit_exceedance(self) -> None:
        from benchmark_mpc_qp_osqp_1s import default_config, write_load_feasibility_check

        cfg = default_config(horizon=2)
        data = pd.DataFrame(
            [
                {"voyage_id": "voyage_001", "split": "test", "load_total_kw": 100.0},
                {"voyage_id": "voyage_001", "split": "test", "load_total_kw": 907.0},
                {"voyage_id": "voyage_002", "split": "test", "load_total_kw": 820.134823},
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            result = write_load_feasibility_check(data, cfg, output_dir=Path(tmp))
            check = pd.read_csv(result["csv_path"])
            text = Path(result["md_path"]).read_text(encoding="utf-8")

        self.assertAlmostEqual(float(check.loc[0, "P_available_max_kw"]), 906.5)
        self.assertEqual(int(check.loc[0, "num_steps_load_exceeds_available"]), 1)
        self.assertIn("voyage_001", str(check.loc[0, "voyages_with_exceedance"]))
        self.assertNotIn("voyage_002", str(check.loc[0, "voyages_with_exceedance"]))
        self.assertIn("P_batt_max = 346.5 kW", text)

    def test_control_performance_counts_load_above_power_limit_without_tolerance(self) -> None:
        from benchmark_mpc_qp_osqp_1s import control_performance_metrics, default_config

        cfg = default_config(horizon=2)
        control = pd.DataFrame(
            [
                {
                    "voyage_id": "voyage_001",
                    "step": 0,
                    "load_total_kw": 906.5,
                    "P_fc_kw": 560.0,
                    "P_batt_kw": 346.5,
                    "SOC": 0.55,
                    "fc_ramp_kw": 0.0,
                },
                {
                    "voyage_id": "voyage_001",
                    "step": 1,
                    "load_total_kw": 906.55,
                    "P_fc_kw": np.nan,
                    "P_batt_kw": np.nan,
                    "SOC": np.nan,
                    "fc_ramp_kw": np.nan,
                },
            ]
        )

        perf = control_performance_metrics(control, pd.DataFrame(), cfg)

        self.assertEqual(perf["load_exceeds_power_limit_count"], 1)
        self.assertAlmostEqual(perf["load_exceeds_power_limit_fraction"], 0.5)

    def test_summary_tables_and_validity_flags_are_generated_from_results(self) -> None:
        from benchmark_mpc_qp_osqp_1s import default_config, write_benchmark_artifacts

        cfg = default_config(horizon=2)
        control = pd.DataFrame(
            [
                {
                    "voyage_id": "voyage_001",
                    "step": 0,
                    "time_s": 0.0,
                    "load_total_kw": 100.0,
                    "P_fc_kw": 99.999,
                    "P_batt_kw": 0.001,
                    "SOC": 0.55,
                    "status": "solved",
                    "success": True,
                    "balance_violation_kw": 0.0,
                    "ramp_violation_kw": 0.0,
                    "soc_violation": 0.0,
                },
                {
                    "voyage_id": "voyage_001",
                    "step": 1,
                    "time_s": 1.0,
                    "load_total_kw": 101.0,
                    "P_fc_kw": 100.999,
                    "P_batt_kw": 0.001,
                    "SOC": 0.519,
                    "status": "solved",
                    "success": True,
                    "balance_violation_kw": 0.0,
                    "ramp_violation_kw": 0.0,
                    "soc_violation": 0.0,
                },
            ]
        )
        timing = pd.DataFrame(
            [
                {
                    "voyage_id": "voyage_001",
                    "step": 0,
                    "time_s": 0.0,
                    "status": "solved",
                    "success": True,
                    "build_ms": 1.0,
                    "setup_plus_solve_ms": 2.0,
                    "total_controller_ms": 3.0,
                    "iterations": 25,
                    "objective": 1.0,
                    "pri_res": 0.0,
                    "dua_res": 0.0,
                },
                {
                    "voyage_id": "voyage_001",
                    "step": 1,
                    "time_s": 1.0,
                    "status": "solved",
                    "success": True,
                    "build_ms": 1.0,
                    "setup_plus_solve_ms": 2.0,
                    "total_controller_ms": 4.0,
                    "iterations": 25,
                    "objective": 1.0,
                    "pri_res": 0.0,
                    "dua_res": 0.0,
                },
            ]
        )
        voyage_metrics = pd.DataFrame(
            [
                {
                    "voyage_id": "voyage_001",
                    "steps": 2,
                    "success_rate": 1.0,
                    "H2_total_kg": 0.1,
                    "SOC_initial": 0.55,
                    "SOC_min": 0.519,
                    "SOC_max": 0.55,
                    "SOC_final": 0.519,
                    "SOC_final_minus_initial": -0.031,
                    "SOC_mean_abs_deviation_from_ref": 0.0155,
                    "battery_throughput_kwh": 0.0,
                    "battery_discharge_energy_kwh": 0.0,
                    "battery_charge_energy_kwh": 0.0,
                    "P_batt_mean_abs": 0.001,
                    "P_batt_max": 0.001,
                    "P_batt_min": 0.001,
                    "P_fc_mean": 100.5,
                    "P_fc_max": 100.999,
                    "P_fc_min": 99.999,
                    "fc_ramp_max": 1.0,
                    "fc_ramp_violation_count": 0,
                    "power_balance_violation_max": 0.0,
                    "SOC_violation_count": 0,
                    "battery_power_violation_count": 0,
                    "fc_power_violation_count": 0,
                    "solver_success_rate": 1.0,
                    "mean_total_ms": 3.5,
                    "p95_total_ms": 3.95,
                    "p99_total_ms": 3.99,
                    "max_total_ms": 4.0,
                }
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = write_benchmark_artifacts(
                output_dir=Path(tmp),
                config=cfg,
                control_df=control,
                time_df=timing,
                metrics_df=voyage_metrics,
                input_path=Path("input.parquet"),
                max_steps_per_voyage=None,
                make_plots=False,
                report_filename="REPORT_EBATT277P2_WEIGHT_VALIDITY.md",
            )

            required = [
                "solver_benchmark_summary.csv",
                "solver_benchmark_by_voyage.csv",
                "solver_timing_distribution.csv",
                "solver_failure_cases.csv",
                "constraint_violation_summary.csv",
                "control_performance_summary.csv",
                "solver_config.json",
                "REPORT_EBATT277P2_WEIGHT_VALIDITY.md",
            ]
            for name in required:
                self.assertTrue((Path(tmp) / name).exists(), name)

            summary = pd.read_csv(Path(tmp) / "solver_benchmark_summary.csv")
            self.assertFalse(bool(summary.loc[0, "weights_valid"]))
            self.assertIn("SOC_sustain_failed", result["validity"]["failure_categories"])
            self.assertIn("over_conservative_battery_unused", result["validity"]["failure_categories"])

    def test_simplified_objective_check_reports_fixed_physical_denominators(self) -> None:
        from benchmark_mpc_qp_osqp_1s import (
            SIMPLIFIED_SPEC_NORM_OUTPUT_DIR,
            default_config,
            write_simplified_normalized_objective_check,
        )

        cfg = default_config(horizon=60)
        with tempfile.TemporaryDirectory() as tmp:
            result = write_simplified_normalized_objective_check(output_dir=Path(tmp), config=cfg)
            text = Path(result["md_path"]).read_text(encoding="utf-8")

            self.assertEqual(Path(result["md_path"]).name, "simplified_normalized_objective_check.md")
            self.assertIn("osqp_n60_Ebatt693_simplified_spec_norm", str(SIMPLIFIED_SPEC_NORM_OUTPUT_DIR))
            self.assertIn("P_batt_ref = 346.5 kW", text)
            self.assertIn("10 x 69.3 kWh = 693 kWh", text)
            self.assertIn("SOC_band = 0.05", text)
            self.assertIn("convex QP", text)
            self.assertIn("not test set max", text.lower())

    def test_objective_summary_uses_normalized_h2_soc_batt_terms_only(self) -> None:
        from benchmark_mpc_qp_osqp_1s import (
            classify_physical_baseline,
            default_config,
            objective_term_summary,
        )

        cfg = default_config(horizon=2, q_batt=0.05)
        control = pd.DataFrame(
            [
                {
                    "voyage_id": "voyage_001",
                    "step": 0,
                    "time_s": 0.0,
                    "load_total_kw": 100.0,
                    "P_fc_kw": 95.0,
                    "P_batt_kw": 5.0,
                    "SOC": 0.549,
                    "prev_fc_before_kw": 94.0,
                    "fc_delta_kw": 1.0,
                    "balance_violation_kw": 0.0,
                    "ramp_violation_kw": 0.0,
                    "soc_violation": 0.0,
                    "success": True,
                },
                {
                    "voyage_id": "voyage_001",
                    "step": 1,
                    "time_s": 1.0,
                    "load_total_kw": 102.0,
                    "P_fc_kw": 96.0,
                    "P_batt_kw": 6.0,
                    "SOC": 0.548,
                    "prev_fc_before_kw": 95.0,
                    "fc_delta_kw": 1.0,
                    "balance_violation_kw": 0.0,
                    "ramp_violation_kw": 0.0,
                    "soc_violation": 0.0,
                    "success": True,
                },
            ]
        )
        summary = objective_term_summary(control, cfg)

        self.assertTrue((summary["term_sum"] >= 0).all())
        self.assertEqual(set(summary["term_name"]), {"H2_norm", "SOC_norm", "Batt_norm"})
        self.assertNotIn("ramp", set(summary["term_name"]))
        self.assertNotIn("terminal_soc", set(summary["term_name"]))

        row = {
            "solver_success_rate": 1.0,
            "solve_time_ms_p99": 10.0,
            "power_balance_violation_max": 0.0,
            "SOC_min": 0.3,
            "SOC_max": 0.56,
            "fc_ramp_violation_count": 0,
            "battery_power_violation_count": 0,
            "fc_power_violation_count": 0,
            "battery_active_fraction_abs_gt_5kw": 0.30,
            "P_batt_near_zero_fraction_abs_le_1kw": 0.20,
            "battery_saturation_fraction_abs_gt_0p9Pmax": 0.0,
            "SOC_final_minus_initial_min": -0.02,
            "fc_energy_share": 0.90,
        }
        self.assertEqual(classify_physical_baseline(row), "PASS_PHYSICAL_BASELINE")

        row["battery_active_fraction_abs_gt_5kw"] = 0.0
        row["P_batt_near_zero_fraction_abs_le_1kw"] = 0.98
        self.assertEqual(classify_physical_baseline(row), "FAIL_BATTERY_UNUSED")

    def test_physical_label_distinguishes_power_limit_insufficient(self) -> None:
        from benchmark_mpc_qp_osqp_1s import classify_physical_baseline

        row = {
            "solver_success_rate": 0.5,
            "infeasible_count": 10,
            "load_exceeds_power_limit_count": 10,
        }
        self.assertEqual(classify_physical_baseline(row), "FAIL_POWER_LIMIT_INSUFFICIENT")

    def test_simplified_case_order_and_recommendation_prefers_physical_pass(self) -> None:
        from benchmark_mpc_qp_osqp_1s import (
            SIMPLIFIED_SPEC_NORM_CASES,
            select_recommended_simplified_baseline,
        )

        self.assertEqual(
            [case["case_name"] for case in SIMPLIFIED_SPEC_NORM_CASES],
            [
                "case_spec_norm_base",
                "case_spec_norm_more_batt",
                "case_spec_norm_batt_conservative",
                "case_spec_norm_soc_safe",
                "case_spec_norm_h2_low_fc_main",
                "case_spec_norm_h2_high_economy",
                "case_spec_norm_more_batt_soc_safe",
                "case_spec_norm_soc_strong",
            ],
        )
        summary = pd.DataFrame(
            [
                {
                    "case_name": "case_fail",
                    "physical_label": "FAIL_BATTERY_UNUSED",
                    "solver_success_rate": 1.0,
                    "battery_active_fraction_abs_gt_5kW": 0.0,
                    "SOC_drop_max_by_voyage": 0.0,
                    "fc_energy_share": 1.0,
                },
                {
                    "case_name": "case_pass",
                    "physical_label": "PASS_PHYSICAL_BASELINE",
                    "solver_success_rate": 1.0,
                    "battery_active_fraction_abs_gt_5kW": 0.20,
                    "SOC_drop_max_by_voyage": 0.01,
                    "fc_energy_share": 0.85,
                },
            ]
        )

        result = select_recommended_simplified_baseline(summary)

        self.assertTrue(result["accepted"])
        self.assertEqual(result["recommended_fixed_mpc_baseline_before_dqn"], "case_pass")

    def test_code_cleanup_report_records_legacy_entrypoint_scope(self) -> None:
        from benchmark_mpc_qp_osqp_1s import write_code_cleanup_report

        with tempfile.TemporaryDirectory() as tmp:
            path = write_code_cleanup_report(Path(tmp))
            text = Path(path).read_text(encoding="utf-8")

        self.assertIn("simplified_normalized_literature_v1", text)
        self.assertIn("raw_weight_retune", text)
        self.assertIn("30 s mainline: not modified", text)
        self.assertIn("CasADi/IPOPT baseline: not modified", text)
        self.assertIn("outputs: preserved", text)


if __name__ == "__main__":
    unittest.main()
