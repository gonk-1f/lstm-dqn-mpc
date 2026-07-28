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
    def test_persistent_osqp_solve_preserves_non_solved_result_objects(self) -> None:
        from mpc_solvers.osqp_runtime import _solve_with_persistent_osqp

        class FakeSolver:
            def __init__(self) -> None:
                self.solve_kwargs = None

            def update(self, **_: object) -> None:
                pass

            def solve(self, **kwargs: object) -> object:
                self.solve_kwargs = kwargs
                return object()

        solver = FakeSolver()
        result, solve_ms = _solve_with_persistent_osqp(
            solver,
            lower=np.asarray([0.0]),
            upper=np.asarray([1.0]),
        )

        self.assertIsNotNone(result)
        self.assertGreaterEqual(solve_ms, 0.0)
        self.assertEqual(
            solver.solve_kwargs,
            {"raise_error": False},
        )

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
                    "split": "test",
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
                    "split": "train",
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
                    "split": "train",
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
            active_split_path = root / "voyage_split_total_load_721.json"
            active_split_path.write_text(
                json.dumps(
                    {
                        "train_voyages": ["voyage_001"],
                        "validation_voyages": [],
                        "test_voyages": ["voyage_066"],
                        "excluded_voyages": [],
                    }
                ),
                encoding="utf-8",
            )

            result = build_benchmark_dataset(
                input_dir=source_dir,
                output_dir=out_dir,
                split_json_path=active_split_path,
            )

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
            self.assertEqual(split["active_split_json"], str(active_split_path))

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


if __name__ == "__main__":
    unittest.main()
