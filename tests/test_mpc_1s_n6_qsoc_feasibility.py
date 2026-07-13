from __future__ import annotations

import sys
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
    return {
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
        "solve_time_ms_max": 2.0,
    }


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
            self.assertEqual(config.soc_ref, 0.55)
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
            )

        self.assertIs(run_voyage.call_args.kwargs["config"], explicit)
        self.assertIs(build_metrics.call_args.kwargs["config"], explicit)
        self.assertIs(write_artifacts.call_args.kwargs["config"], explicit)


if __name__ == "__main__":
    unittest.main()
