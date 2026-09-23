from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from v2.data.zero_boundary_dataset import (  # noqa: E402
    FIXED_TEST_PARENTS,
    assign_parent_splits,
    reconstruct_one_second,
    segment_features,
    trim_to_zero_boundaries,
)


class ZeroBoundaryDatasetTests(unittest.TestCase):
    @staticmethod
    def frame(source_values: list[float]) -> pd.DataFrame:
        origin = datetime(2024, 5, 1, 8, 0, tzinfo=timezone.utc)
        source = np.asarray(source_values, dtype=float)
        battery_raw = np.full(len(source), 100.0)
        return pd.DataFrame(
            {
                "timestamp": [
                    origin + timedelta(seconds=30 * index)
                    for index in range(len(source))
                ],
                "fc_total_kw": source + battery_raw,
                "battery_raw_total_kw": battery_raw,
                "source_total_kw": source,
                "is_cubic_imputed": np.zeros(len(source), dtype=bool),
            }
        )

    def test_trims_leading_and_trailing_dwell_to_observed_zero_boundaries(self):
        frame = self.frame([0, 0, 0, 20, 30, 40, 35, 25, 10, 0, 0, -15])

        result = trim_to_zero_boundaries(frame)

        self.assertEqual(
            result.frame.source_total_kw.tolist(),
            [0, 20, 30, 40, 35, 25, 10, 0],
        )
        self.assertEqual(result.start.kind, "OBSERVED_DEADBAND")
        self.assertEqual(result.end.kind, "OBSERVED_DEADBAND")
        self.assertEqual(result.frame.iloc[0].source_total_kw, 0.0)
        self.assertEqual(result.frame.iloc[-1].source_total_kw, 0.0)

    def test_constructs_positive_to_negative_terminal_crossing_with_component_identity(
        self,
    ):
        frame = self.frame([0, 10, 20, 30, 25, 20, 10, -10, -20])

        result = trim_to_zero_boundaries(frame)

        self.assertEqual(result.end.kind, "CONSTRUCTED_CROSSING")
        self.assertAlmostEqual(result.end.crossing_fraction, 0.5)
        last = result.frame.iloc[-1]
        self.assertAlmostEqual(last.source_total_kw, 0.0, places=12)
        self.assertAlmostEqual(
            last.source_total_kw,
            last.fc_total_kw - last.battery_raw_total_kw,
            places=12,
        )

    def test_preserves_internal_stop_and_negative_interval(self):
        frame = self.frame([0, 15, 20, 25, 0, -5, 0, 18, 22, 26, 0, -20])

        result = trim_to_zero_boundaries(frame)

        self.assertIn(-5.0, result.frame.source_total_kw.tolist())
        self.assertEqual(result.frame.source_total_kw.tolist().count(0.0), 4)

    def test_rejects_missing_start_or_end_zero_bracket(self):
        with self.assertRaisesRegex(ValueError, "start boundary"):
            trim_to_zero_boundaries(self.frame([5, 10, 20, 30, 25, 10, 0]))
        with self.assertRaisesRegex(ValueError, "end boundary"):
            trim_to_zero_boundaries(self.frame([0, 10, 20, 30, 25, 10, 5]))

    def test_reconstructs_exact_one_second_axis_and_zero_endpoints(self):
        trimmed = trim_to_zero_boundaries(
            self.frame([0, 10, 20, 30, 20, 10, 0])
        )

        output, qa = reconstruct_one_second(trimmed.frame)

        np.testing.assert_array_equal(
            output.time_s,
            np.arange(len(output), dtype=float),
        )
        self.assertEqual(output.iloc[0].load_total_kw, 0.0)
        self.assertEqual(output.iloc[-1].load_total_kw, 0.0)
        self.assertEqual(qa["output_points"], len(output))

    @staticmethod
    def feature_frame() -> pd.DataFrame:
        parents = [f"parent_{index:03d}" for index in range(61)] + list(
            FIXED_TEST_PARENTS
        )
        origin = pd.Timestamp("2024-01-01T00:00:00Z")
        return pd.DataFrame(
            {
                "parent": parents,
                "chronological_timestamp": [
                    origin + pd.Timedelta(days=index)
                    for index in range(len(parents))
                ],
                "month": [(index % 6) + 1 for index in range(len(parents))],
                "duration_s": [1000.0 + 60.0 * index for index in range(len(parents))],
                "mean_load_kw": [50.0 + index for index in range(len(parents))],
                "p95_load_kw": [100.0 + 2.0 * index for index in range(len(parents))],
            }
        )

    def test_assigns_exact_deterministic_parent_splits_without_leakage(self):
        features = self.feature_frame()

        assignment = assign_parent_splits(features)
        shuffled = assign_parent_splits(
            features.sample(frac=1.0, random_state=7)
        )

        self.assertEqual(
            assignment.split.value_counts().to_dict(),
            {"train": 49, "validation": 12, "test": 5},
        )
        self.assertEqual(
            set(assignment.loc[assignment.split.eq("test"), "parent"]),
            set(FIXED_TEST_PARENTS),
        )
        self.assertEqual(assignment.groupby("parent").split.nunique().max(), 1)
        pd.testing.assert_frame_equal(assignment, shuffled)

    def test_non_test_quartiles_and_assignment_ignore_test_features(self):
        features = self.feature_frame()
        baseline = assign_parent_splits(features)
        changed = features.copy()
        test_mask = changed["parent"].isin(FIXED_TEST_PARENTS)
        changed.loc[test_mask, ["duration_s", "mean_load_kw", "p95_load_kw"]] = [
            1.0e9,
            -1.0e9,
            5.0e8,
        ]

        mutated = assign_parent_splits(changed)

        non_test = baseline.loc[~baseline["split"].eq("test")]
        for column in (
            "duration_quartile",
            "mean_load_quartile",
            "p95_load_quartile",
        ):
            self.assertEqual(set(non_test[column].astype(int)), {0, 1, 2, 3})
        self.assertEqual(
            baseline.loc[~baseline.split.eq("test"), ["parent", "split"]]
            .reset_index(drop=True)
            .to_dict("records"),
            mutated.loc[~mutated.split.eq("test"), ["parent", "split"]]
            .reset_index(drop=True)
            .to_dict("records"),
        )

    def test_extracts_split_features_from_trimmed_one_second_segment(self):
        trimmed = trim_to_zero_boundaries(
            self.frame([0, 10, 20, 30, 20, 10, 0])
        )
        one_second, _ = reconstruct_one_second(trimmed.frame)

        features = segment_features("parent", one_second)

        self.assertEqual(features["parent"], "parent")
        self.assertEqual(features["month"], 5)
        self.assertEqual(features["duration_s"], 180.0)
        self.assertAlmostEqual(
            features["mean_load_kw"],
            float(one_second.load_total_kw.mean()),
        )
        self.assertAlmostEqual(
            features["p95_load_kw"],
            float(np.percentile(one_second.load_total_kw, 95)),
        )


if __name__ == "__main__":
    unittest.main()
