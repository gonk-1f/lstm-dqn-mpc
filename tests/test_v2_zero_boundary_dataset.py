from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from v2.data.zero_boundary_dataset import (  # noqa: E402
    APPROVED_BOUNDARY_EXCLUSIONS,
    FIXED_TEST_PARENTS,
    assign_parent_splits,
    reconstruct_one_second,
    segment_features,
    trim_to_zero_boundaries,
)
from v2.data.segment_power_source import ParentPowerSeries  # noqa: E402
from main.build_zero_boundary_operating_dataset import build_dataset  # noqa: E402


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
        parents = [f"parent_{index:03d}" for index in range(48)] + list(
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
            {"train": 38, "validation": 10, "test": 5},
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

    @staticmethod
    def synthetic_parent_names() -> list[str]:
        names = []
        for index in range(48):
            month = index // 28 + 1
            day = index % 28 + 1
            names.append(f"合成{month}月{day}日00_00_{index:03d}")
        return names + list(FIXED_TEST_PARENTS) + list(
            APPROVED_BOUNDARY_EXCLUSIONS
        )

    @classmethod
    def synthetic_power_series(cls, parent: str) -> ParentPowerSeries:
        parents = cls.synthetic_parent_names()
        rank = parents.index(parent)
        scale = 1.0 + rank / 100.0
        exclusion_side = APPROVED_BOUNDARY_EXCLUSIONS.get(parent)
        if exclusion_side == "start":
            source = np.asarray(
                [20.0, 22.0, 24.0, 30.0, 28.0, 26.0, 20.0],
                dtype=float,
            )
        elif exclusion_side == "end":
            source = np.asarray(
                [0.0, 10.0, 20.0, 30.0, 28.0, 26.0, 20.0],
                dtype=float,
            )
        else:
            pause = [0.0] * (rank % 8)
            source = np.asarray(
                [0.0, 10.0, 20.0, 30.0, 0.0, -5.0]
                + pause
                + [15.0 * scale, 25.0 * scale, 35.0 * scale, 0.0, -20.0],
                dtype=float,
            )
        origin = datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(
            days=rank
        )
        timestamps = tuple(
            origin + timedelta(seconds=30 * index)
            for index in range(len(source))
        )
        battery_raw = np.full(len(source), 100.0)
        return ParentPowerSeries(
            parent=parent,
            timestamps=timestamps,
            fc_total_kw=source + battery_raw,
            battery_raw_total_kw=battery_raw,
            source_total_kw=source,
            ais_present=np.zeros(len(source), dtype=bool),
            duplicate_count=0,
            duplicate_conflict_count=0,
            channel_span_violation_count=0,
        )

    def test_builder_refuses_an_existing_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "already-exists"
            output.mkdir()

            with self.assertRaisesRegex(FileExistsError, "already exists"):
                build_dataset(
                    root,
                    output,
                    discover_parents=lambda _: self.synthetic_parent_names(),
                    load_parent=self.synthetic_power_series,
                )

    def test_builder_writes_complete_hashed_zero_boundary_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "dataset"
            parents = self.synthetic_parent_names()

            summary = build_dataset(
                root,
                output,
                discover_parents=lambda _: parents,
                load_parent=lambda _, parent: self.synthetic_power_series(parent),
            )

            manifest = pd.read_csv(output / "metadata" / "sample_manifest.csv")
            excluded = pd.read_csv(
                output / "metadata" / "excluded_parent_manifest.csv"
            )
            self.assertEqual(summary["raw_parent_count"], 66)
            self.assertEqual(summary["excluded_parent_count"], 13)
            self.assertEqual(summary["parent_count"], 53)
            self.assertEqual(len(manifest), 53)
            self.assertEqual(
                manifest["split"].value_counts().to_dict(),
                {"train": 38, "validation": 10, "test": 5},
            )
            self.assertEqual(
                set(manifest.loc[manifest.split.eq("test"), "parent"]),
                set(FIXED_TEST_PARENTS),
            )
            self.assertEqual(
                dict(zip(excluded.parent, excluded.missing_boundary)),
                APPROVED_BOUNDARY_EXCLUSIONS,
            )
            for relative, expected_hash in manifest[
                ["relative_path", "sha256"]
            ].itertuples(index=False, name=None):
                path = output / relative
                self.assertTrue(path.is_file())
                self.assertEqual(
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                    expected_hash,
                )
                frame = pd.read_csv(path)
                np.testing.assert_array_equal(
                    frame.time_s.to_numpy(dtype=float),
                    np.arange(len(frame), dtype=float),
                )
                self.assertEqual(frame.load_total_kw.iloc[0], 0.0)
                self.assertEqual(frame.load_total_kw.iloc[-1], 0.0)
            for name in (
                "sample_manifest.csv",
                "parent_split_manifest.csv",
                "excluded_parent_manifest.csv",
                "trim_boundary_audit.csv",
                "interpolation_audit.csv",
                "source_files.csv",
                "policy.json",
                "qa_summary.json",
            ):
                self.assertTrue((output / "metadata" / name).is_file(), name)
            qa = json.loads(
                (output / "metadata" / "qa_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(qa["formal_training_status"], "NO-GO")
            self.assertTrue(all(qa["acceptance_checks"].values()))


if __name__ == "__main__":
    unittest.main()
