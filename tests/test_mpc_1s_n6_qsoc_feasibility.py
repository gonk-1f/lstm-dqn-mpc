from __future__ import annotations

import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
MAIN_ROOT = SRC / "main"
for path in (SRC, MAIN_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def passing_summary(candidate_id: str) -> dict[str, object]:
    from run_mpc_1s_n6_weight_selection import REQUIRED_N6_METRIC_KEYS

    summary: dict[str, object] = {key: 0.0 for key in REQUIRED_N6_METRIC_KEYS}
    summary.update({
        "candidate_id": candidate_id,
        "voyage_count": 7,
        "is_partial_debug_run": False,
        "closed_loop_complete": True,
        "solver_failure_count": 0,
        "physical_infeasible_point_count": 0,
        "aggregate_metrics_comparable": True,
        "soc_min": 0.50,
        "soc_max": 0.60,
        "worst_voyage_soc_net_change": -0.01,
        "closed_loop_coverage_fraction": 1.0,
        "solver_success_rate": 1.0,
        "final_soc": 0.54,
        "load_energy_mwh": 1.0,
        "hydrogen_intensity_kg_per_mwh": 1.0,
        "max_actual_power_balance_residual_kw": 0.0,
        "max_soc_bound_residual": 0.0,
        "max_soc_prediction_residual": 0.0,
        "solve_time_ms_mean": 1.0,
        "solve_time_ms_p95": 1.5,
        "solve_time_ms_p99": 1.8,
        "solve_time_ms_max": 2.0,
        "primal_residual_max_abs": 1.0e-7,
        "dual_residual_max_abs": 1.0e-7,
    })
    return summary


class TestQsocFeasibilityContract(unittest.TestCase):
    def test_exact_candidate_set(self) -> None:
        from run_mpc_1s_n6_qsoc_feasibility import QSOC_CANDIDATES

        self.assertEqual(
            QSOC_CANDIDATES,
            (
                {
                    "candidate_id": "QSOC_5",
                    "q_h2": 0.5,
                    "q_soc": 5.0,
                    "q_batt": 0.05,
                    "soc_band": 0.05,
                },
                {
                    "candidate_id": "QSOC_10",
                    "q_h2": 0.5,
                    "q_soc": 10.0,
                    "q_batt": 0.05,
                    "soc_band": 0.05,
                },
                {
                    "candidate_id": "QSOC_20",
                    "q_h2": 0.5,
                    "q_soc": 20.0,
                    "q_batt": 0.05,
                    "soc_band": 0.05,
                },
            ),
        )

    def test_candidate_configs_change_only_q_soc(self) -> None:
        from run_mpc_1s_n6_qsoc_feasibility import qsoc_candidate_config

        configs = [
            qsoc_candidate_config(candidate_id)
            for candidate_id in ("QSOC_5", "QSOC_10", "QSOC_20")
        ]
        self.assertEqual([config.q_soc for config in configs], [5.0, 10.0, 20.0])
        for config in configs:
            self.assertEqual(config.horizon, 6)
            self.assertEqual(config.dt_seconds, 1.0)
            self.assertEqual(config.battery_capacity_kwh, 693.0)
            self.assertEqual(config.battery_charge_max_kw, 346.5)
            self.assertEqual(config.battery_discharge_max_kw, 346.5)
            self.assertEqual(config.battery_power_ref_kw, 346.5)
            self.assertEqual(config.fuel_cell_max_kw, 560.0)
            self.assertEqual(config.fuel_cell_ramp_rate_kw_per_s, 48.0)
            self.assertEqual(config.soc_min, 0.2)
            self.assertEqual(config.soc_max, 0.8)
            self.assertEqual(config.q_h2, 0.5)
            self.assertEqual(config.q_batt, 0.05)
            self.assertEqual(config.soc_band, 0.05)
            self.assertEqual(config.q_ramp, 0.0)
            self.assertEqual(config.q_terminal_soc, 0.0)

        reference = asdict(configs[0])
        for config in configs[1:]:
            candidate = asdict(config)
            differing = {key for key in reference if reference[key] != candidate[key]}
            self.assertEqual(differing, {"q_soc"})

    def test_default_paths_are_isolated_from_old_experiment(self) -> None:
        from run_mpc_1s_n6_qsoc_feasibility import (
            DEFAULT_OUTPUT_ROOT,
            DEFAULT_SUMMARY_REPORT,
            DEFAULT_TABLE_REPORT,
        )

        self.assertEqual(DEFAULT_OUTPUT_ROOT.name, "mpc_1s_n6_qsoc_feasibility")
        self.assertEqual(DEFAULT_SUMMARY_REPORT.name, "mpc_1s_n6_qsoc_feasibility_summary.md")
        self.assertEqual(DEFAULT_TABLE_REPORT.name, "mpc_1s_n6_qsoc_feasibility_table.csv")
        self.assertNotIn("weight_selection", DEFAULT_OUTPUT_ROOT.as_posix())

    def test_gate_accepts_only_a_complete_physical_candidate(self) -> None:
        from run_mpc_1s_n6_qsoc_feasibility import evaluate_candidate_gate

        result = evaluate_candidate_gate(passing_summary("QSOC_5"))

        self.assertTrue(result["passed"])
        self.assertEqual(result["reasons"], [])

    def test_gate_rejects_each_failed_requirement_without_a_score(self) -> None:
        from run_mpc_1s_n6_qsoc_feasibility import evaluate_candidate_gate

        failures = {
            "closed_loop_complete": False,
            "solver_failure_count": 1,
            "physical_infeasible_point_count": 1,
            "aggregate_metrics_comparable": False,
            "soc_min": 0.19,
            "soc_max": 0.81,
            "worst_voyage_soc_net_change": -0.031,
            "solve_time_ms_max": 1000.0,
        }
        for key, value in failures.items():
            with self.subTest(key=key):
                summary = passing_summary("QSOC_5")
                summary[key] = value
                result = evaluate_candidate_gate(summary)
                self.assertFalse(result["passed"])
                self.assertTrue(result["reasons"])
                self.assertNotIn("score", result)

    def test_gate_rejects_nonfinite_or_inconsistent_required_metrics(self) -> None:
        from run_mpc_1s_n6_qsoc_feasibility import evaluate_candidate_gate

        failures = {
            "max_actual_power_balance_residual_kw": float("nan"),
            "max_plan_power_balance_residual_kw": 0.2,
            "max_fc_bound_residual_kw": 0.2,
            "max_battery_bound_residual_kw": 0.2,
            "max_ramp_residual_kw": 0.2,
            "max_soc_bound_residual": 2.0e-6,
            "max_soc_prediction_residual": 2.0e-5,
            "primal_residual_max_abs": float("nan"),
            "dual_residual_max_abs": float("nan"),
        }
        for key, value in failures.items():
            with self.subTest(key=key):
                summary = passing_summary("QSOC_5")
                summary[key] = value
                result = evaluate_candidate_gate(summary)
                self.assertFalse(result["passed"])
                self.assertTrue(result["reasons"])

    def test_decision_reports_all_witnesses_without_selecting_a_best(self) -> None:
        from run_mpc_1s_n6_qsoc_feasibility import build_diagnostic_decision

        summaries = [
            passing_summary(candidate_id)
            for candidate_id in ("QSOC_5", "QSOC_10", "QSOC_20")
        ]
        decision = build_diagnostic_decision(summaries)

        self.assertEqual(decision["status"], "weight_only_sufficient")
        self.assertEqual(
            decision["feasibility_witnesses"],
            ["QSOC_5", "QSOC_10", "QSOC_20"],
        )
        self.assertIsNone(decision["selected_candidate"])
        self.assertFalse(decision["provisional_config_created"])
        self.assertNotIn("score", decision)

    def test_decision_requires_three_formal_candidates_and_can_be_insufficient(self) -> None:
        from run_mpc_1s_n6_qsoc_feasibility import build_diagnostic_decision

        summaries = []
        for candidate_id in ("QSOC_5", "QSOC_10", "QSOC_20"):
            summary = passing_summary(candidate_id)
            summary["worst_voyage_soc_net_change"] = -0.31
            summaries.append(summary)

        decision = build_diagnostic_decision(summaries)
        self.assertEqual(decision["status"], "weight_only_insufficient_in_tested_range")
        self.assertEqual(decision["feasibility_witnesses"], [])
        self.assertIsNone(decision["selected_candidate"])

        invalid_sets = (
            summaries[:2],
            [{**item, "is_partial_debug_run": True} for item in summaries],
            [{**item, "voyage_count": 6} for item in summaries],
        )
        for invalid in invalid_sets:
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    build_diagnostic_decision(invalid)

    def test_old_candidates_remain_fixed_and_shared_runner_accepts_explicit_config(self) -> None:
        import run_mpc_1s_n6_weight_selection as base
        from mpc_solvers.mpc_qp_formulation import QpMpcConfig

        self.assertEqual(
            base.CANDIDATES,
            (
                {"candidate_id": "A", "q_h2": 0.5, "q_soc": 2.0, "q_batt": 0.05, "soc_band": 0.05},
                {"candidate_id": "B", "q_h2": 0.5, "q_soc": 1.5, "q_batt": 0.05, "soc_band": 0.05},
                {"candidate_id": "C", "q_h2": 0.5, "q_soc": 2.0, "q_batt": 0.05, "soc_band": 0.075},
                {"candidate_id": "D", "q_h2": 0.5, "q_soc": 2.0, "q_batt": 0.075, "soc_band": 0.05},
            ),
        )

        explicit = QpMpcConfig(horizon=6, q_soc=5.0)
        artifact_metadata = {"diagnostic_provenance": {"generation_id": "test-generation"}}
        data = pd.DataFrame(
            {
                "voyage_id": ["v1", "v1"],
                "time_s": [0.0, 1.0],
                "load_total_kw": [100.0, 101.0],
            }
        )
        controls = pd.DataFrame({"voyage_id": ["v1"]})
        solver = pd.DataFrame({"voyage_id": ["v1"]})
        voyage_metrics = pd.DataFrame({"voyage_id": ["v1"]})
        solver_statistics = pd.DataFrame({"scope": ["overall"], "voyage_id": ["all"]})
        with (
            patch.object(base, "candidate_config", side_effect=AssertionError("old lookup used")),
            patch.object(base, "load_spline_test_data", return_value=data),
            patch.object(base, "run_voyage", return_value=(controls, solver)) as run_voyage,
            patch.object(
                base,
                "build_candidate_metrics",
                return_value=({}, voyage_metrics, solver_statistics),
            ) as build_metrics,
            patch.object(base, "write_candidate_artifacts", return_value=Path("candidate_QSOC_5")) as write_artifacts,
        ):
            base.run_candidate(
                "QSOC_5",
                config=explicit,
                input_path=Path("input.parquet"),
                output_root=Path("outputs"),
                make_plots=False,
                artifact_metadata=artifact_metadata,
            )

        self.assertIs(run_voyage.call_args.kwargs["config"], explicit)
        self.assertIs(build_metrics.call_args.kwargs["config"], explicit)
        self.assertIs(write_artifacts.call_args.kwargs["config"], explicit)
        self.assertIs(
            write_artifacts.call_args.kwargs["artifact_metadata"],
            artifact_metadata,
        )

    def test_formal_config_fingerprint_rejects_weight_or_timing_drift(self) -> None:
        from run_mpc_1s_n6_qsoc_feasibility import (
            build_diagnostic_provenance,
            qsoc_candidate_config,
            validate_candidate_fingerprint,
        )
        from run_mpc_1s_n6_weight_selection import _candidate_metadata

        input_path = Path(
            "outputs/mpc_solver_benchmark_1s/data/test_voyages_spline_1s.parquet"
        )
        provenance = build_diagnostic_provenance(
            input_path,
            generation_id="fingerprint-test",
        )
        metadata = _candidate_metadata(
            "QSOC_5",
            input_path=input_path,
            config=qsoc_candidate_config("QSOC_5"),
            artifact_metadata={"diagnostic_provenance": provenance},
        )
        validate_candidate_fingerprint(
            "QSOC_5",
            metadata,
            expected_input_path=input_path,
        )

        bad_weight = {**metadata, "model": {**metadata["model"], "q_soc": 10.0}}
        bad_timing = {
            **metadata,
            "timing": {**metadata["timing"], "forecast_samples": "t..t+5"},
        }
        for invalid in (bad_weight, bad_timing):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    validate_candidate_fingerprint(
                        "QSOC_5",
                        invalid,
                        expected_input_path=input_path,
                    )

    def test_invalidation_removes_only_combined_diagnostic_artifacts(self) -> None:
        from run_mpc_1s_n6_qsoc_feasibility import invalidate_diagnostic_artifacts

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_root = root / "outputs"
            reports_dir = root / "reports"
            output_root.mkdir()
            reports_dir.mkdir()
            stale_paths = (
                output_root / "diagnostic_decision.json",
                reports_dir / "mpc_1s_n6_qsoc_feasibility_summary.md",
                reports_dir / "mpc_1s_n6_qsoc_feasibility_table.csv",
            )
            for path in stale_paths:
                path.write_text("stale", encoding="utf-8")
            unrelated = reports_dir / "keep.md"
            unrelated.write_text("keep", encoding="utf-8")

            invalidate_diagnostic_artifacts(output_root, reports_dir)

            self.assertTrue(all(not path.exists() for path in stale_paths))
            self.assertTrue(unrelated.exists())

    def test_provenance_hashes_input_and_rejects_cross_generation_cohorts(self) -> None:
        from run_mpc_1s_n6_qsoc_feasibility import (
            build_diagnostic_provenance,
            validate_provenance_cohort,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            input_path = Path(temporary_directory) / "input.parquet"
            input_path.write_bytes(b"stable-input")
            first = build_diagnostic_provenance(
                input_path,
                generation_id="generation-one",
            )
            second = build_diagnostic_provenance(
                input_path,
                generation_id="generation-one",
            )
            third = build_diagnostic_provenance(
                input_path,
                generation_id="generation-one",
            )

            self.assertEqual(first["generation_id"], "generation-one")
            self.assertEqual(len(first["input_sha256"]), 64)
            self.assertEqual(len(first["implementation_sha256"]), 64)
            self.assertTrue(first["git_head"])
            self.assertIn(
                "src/mpc/solvers/fc_dp0_curve.py",
                first["implementation_files"],
            )
            self.assertIn(
                "data/fuel_cell/FC_Dp0_curve_for_Python.csv",
                first["implementation_files"],
            )
            self.assertEqual(
                set(first["runtime_versions"]),
                {"python", "numpy", "pandas", "scipy", "osqp", "pyarrow", "matplotlib"},
            )
            validate_provenance_cohort([first, second, third])

            incompatible = {**third, "generation_id": "generation-two"}
            with self.assertRaises(ValueError):
                validate_provenance_cohort([first, second, incompatible])

    def test_fingerprint_rejects_input_or_implementation_hash_drift(self) -> None:
        from run_mpc_1s_n6_qsoc_feasibility import (
            build_diagnostic_provenance,
            qsoc_candidate_config,
            validate_candidate_fingerprint,
        )
        from run_mpc_1s_n6_weight_selection import _candidate_metadata

        with tempfile.TemporaryDirectory() as temporary_directory:
            input_path = Path(temporary_directory) / "input.parquet"
            input_path.write_bytes(b"formal-input")
            provenance = build_diagnostic_provenance(
                input_path,
                generation_id="formal-generation",
            )
            metadata = _candidate_metadata(
                "QSOC_5",
                input_path=input_path,
                config=qsoc_candidate_config("QSOC_5"),
                artifact_metadata={"diagnostic_provenance": provenance},
            )
            validate_candidate_fingerprint(
                "QSOC_5",
                metadata,
                expected_input_path=input_path,
            )

            for key in ("input_sha256", "implementation_sha256"):
                with self.subTest(key=key):
                    invalid = {
                        **metadata,
                        "diagnostic_provenance": {
                            **provenance,
                            key: "0" * 64,
                        },
                    }
                    with self.assertRaises(ValueError):
                        validate_candidate_fingerprint(
                            "QSOC_5",
                            invalid,
                            expected_input_path=input_path,
                        )

    def test_candidate_reset_removes_only_the_exact_new_diagnostic_directory(self) -> None:
        from run_mpc_1s_n6_qsoc_feasibility import reset_candidate_directory

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_root = root / "mpc_1s_n6_qsoc_feasibility"
            stale = output_root / "candidate_QSOC_5" / "plots" / "stale.png"
            sibling = output_root / "candidate_QSOC_10" / "keep.txt"
            historical = root / "mpc_1s_n6_weight_selection" / "keep.txt"
            for path in (stale, sibling, historical):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("keep", encoding="utf-8")

            reset_candidate_directory(output_root, "QSOC_5")

            self.assertFalse((output_root / "candidate_QSOC_5").exists())
            self.assertTrue(sibling.exists())
            self.assertTrue(historical.exists())


if __name__ == "__main__":
    unittest.main()
