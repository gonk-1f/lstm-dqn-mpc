from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

MAIN_ROOT = Path(__file__).resolve().parents[1] / "src" / "main"
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

from build_spline_1s_diagnostics import (  # noqa: E402
    compute_physical_check,
    compute_predictability_audit,
    reconstruct_voyage_spline,
)


class TestSpline1sDiagnostics(unittest.TestCase):
    def test_reconstruct_voyage_marks_offline_future_dependent_rows(self) -> None:
        source = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2024-01-01 00:00:00", "2024-01-01 00:00:30", "2024-01-01 00:01:00"]
                ),
                "load_total_kw": [10.0, 40.0, 10.0],
                "speed_knots": [1.0, 2.0, 3.0],
            }
        )

        out = reconstruct_voyage_spline(
            source,
            voyage_id="voyage_test",
            split="test",
            dataset_version="cubic_spline_1s_natural",
            bc_type="natural",
        )

        self.assertEqual(len(out), 61)
        self.assertEqual(out["dataset_version"].unique().tolist(), ["cubic_spline_1s_natural"])
        self.assertEqual(out["voyage_id"].unique().tolist(), ["voyage_test"])
        self.assertEqual(out["split"].unique().tolist(), ["test"])
        self.assertTrue(out["speed_knots"].notna().all())
        self.assertFalse(out["online_feasible"].any())
        self.assertTrue(out["uses_future_endpoint"].all())
        self.assertTrue(bool(out.loc[out["time_s"].eq(30.0), "is_original_30s_point"].iloc[0]))
        self.assertFalse(bool(out.loc[out["time_s"].eq(15.0), "is_original_30s_point"].iloc[0]))
        row = out.loc[out["time_s"].eq(15.0)].iloc[0]
        self.assertEqual(row["source_interval_start_time"], "2024-01-01T00:00:00")
        self.assertEqual(row["source_interval_end_time"], "2024-01-01T00:00:30")

    def test_physical_check_counts_local_overshoot(self) -> None:
        frame = pd.DataFrame(
            {
                "dataset_version": ["demo"] * 4,
                "voyage_id": ["voyage_test"] * 4,
                "split": ["test"] * 4,
                "timestamp": pd.to_datetime(
                    [
                        "2024-01-01 00:00:00",
                        "2024-01-01 00:00:01",
                        "2024-01-01 00:00:02",
                        "2024-01-01 00:00:30",
                    ]
                ),
                "time_s": [0.0, 1.0, 2.0, 30.0],
                "load_total_kw": [10.0, 16.0, 8.0, 15.0],
                "source_interval_start_time": ["2024-01-01T00:00:00"] * 4,
                "source_interval_end_time": ["2024-01-01T00:00:30"] * 4,
                "is_original_30s_point": [True, False, False, True],
                "online_feasible": [False] * 4,
                "uses_future_endpoint": [True] * 4,
            }
        )

        check = compute_physical_check(frame)
        row = check.iloc[0]

        self.assertEqual(int(row["above_original_local_max_count"]), 1)
        self.assertEqual(int(row["below_original_local_min_count"]), 1)
        self.assertEqual(int(row["negative_load_count"]), 0)
        self.assertEqual(int(row["original_30s_negative_load_count"]), 0)
        self.assertEqual(int(row["reconstructed_1s_negative_load_count"]), 0)

    def test_physical_check_separates_original_and_reconstructed_negative_loads(self) -> None:
        frame = pd.DataFrame(
            {
                "dataset_version": ["demo"] * 3,
                "voyage_id": ["voyage_test"] * 3,
                "split": ["test"] * 3,
                "timestamp": pd.to_datetime(
                    ["2024-01-01 00:00:00", "2024-01-01 00:00:01", "2024-01-01 00:00:30"]
                ),
                "time_s": [0.0, 1.0, 30.0],
                "load_total_kw": [-1.0, -2.0, 4.0],
                "source_interval_start_time": ["2024-01-01T00:00:00"] * 3,
                "source_interval_end_time": ["2024-01-01T00:00:30"] * 3,
                "is_original_30s_point": [True, False, True],
                "online_feasible": [False] * 3,
                "uses_future_endpoint": [True] * 3,
            }
        )

        check = compute_physical_check(frame)
        row = check.iloc[0]

        self.assertEqual(int(row["negative_load_count"]), 2)
        self.assertEqual(int(row["original_30s_negative_load_count"]), 1)
        self.assertEqual(int(row["reconstructed_1s_negative_load_count"]), 1)

    def test_predictability_audit_outputs_required_horizons(self) -> None:
        rows = []
        for idx in range(120):
            rows.append(
                {
                    "dataset_version": "demo",
                    "voyage_id": "voyage_test",
                    "split": "test",
                    "time_s": float(idx),
                    "load_total_kw": 100.0 + 0.5 * idx,
                    "online_feasible": False,
                    "uses_future_endpoint": True,
                }
            )
        audit = compute_predictability_audit(pd.DataFrame(rows), horizons=(1, 6, 30, 60))
        row = audit.iloc[0]

        for horizon in (1, 6, 30, 60):
            self.assertIn(f"current_hold_h{horizon}_MAE", audit.columns)
            self.assertIn(f"last_slope_h{horizon}_MAE", audit.columns)
            self.assertIn(f"moving_average_h{horizon}_MAE", audit.columns)
        self.assertAlmostEqual(float(row["last_slope_h6_MAE"]), 0.0)
        self.assertFalse(bool(row["online_feasible"]))
        self.assertTrue(bool(row["uses_future_endpoint"]))

if __name__ == "__main__":
    unittest.main()
