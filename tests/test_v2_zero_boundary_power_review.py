from __future__ import annotations

import hashlib
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

from v2.data.zero_boundary_power_review import (  # noqa: E402
    axis_limits,
    load_review_entries,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


class ZeroBoundaryPowerReviewTests(unittest.TestCase):
    def make_dataset(self, root: Path) -> Path:
        dataset = root / "dataset"
        rows = []
        cases = (
            (
                "train",
                "zero_boundary_001",
                "parent_train",
                [0.0, 10.0, -2.0, 0.0],
            ),
            (
                "validation",
                "zero_boundary_002",
                "parent_validation",
                [0.0, 100.0, 50.0, 0.0],
            ),
            (
                "test",
                "zero_boundary_003",
                "parent_test",
                [0.0, 400.0, 200.0, 0.0],
            ),
        )
        for split, sample_id, parent, loads in cases:
            relative = Path(split) / f"{sample_id}.csv"
            path = dataset / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {
                    "timestamp": pd.date_range(
                        "2024-01-01", periods=4, freq="s", tz="Asia/Shanghai"
                    ),
                    "time_s": np.arange(4, dtype=float),
                    "load_total_kw": loads,
                }
            ).to_csv(path, index=False)
            rows.append(
                {
                    "parent": parent,
                    "sample_id": sample_id,
                    "relative_path": relative.as_posix(),
                    "split": split,
                    "point_count_1s": 4,
                    "start_timestamp": "2024-01-01T00:00:00+08:00",
                    "end_timestamp": "2024-01-01T00:00:03+08:00",
                    "duration_s": 3.0,
                    "sha256": sha256(path),
                }
            )
        metadata = dataset / "metadata"
        metadata.mkdir()
        pd.DataFrame(rows).to_csv(metadata / "sample_manifest.csv", index=False)
        return dataset

    def test_axis_limits_use_each_segments_own_range_and_include_zero(self):
        low = axis_limits(np.asarray([0.0, 10.0, -2.0, 0.0]))
        high = axis_limits(np.asarray([0.0, 400.0, 200.0, 0.0]))
        self.assertEqual(low, (-2.6, 10.6))
        self.assertEqual(high, (-20.0, 420.0))
        self.assertNotEqual(low, high)
        self.assertEqual(axis_limits(np.zeros(3)), (-1.0, 1.0))

    def test_load_entries_verifies_counts_hashes_and_retains_negative_values(self):
        with tempfile.TemporaryDirectory() as temp:
            dataset = self.make_dataset(Path(temp))
            entries = load_review_entries(
                dataset,
                expected_split_counts={"train": 1, "validation": 1, "test": 1},
            )
            self.assertEqual(
                [entry.split for entry in entries],
                ["train", "validation", "test"],
            )
            self.assertEqual(entries[0].minimum_kw, -2.0)
            self.assertEqual(entries[0].point_count, 4)
            self.assertEqual(entries[0].duration_s, 3.0)

            source = dataset / entries[0].source_relative_path
            source.write_text(
                source.read_text(encoding="utf-8") + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_review_entries(
                    dataset,
                    expected_split_counts={
                        "train": 1,
                        "validation": 1,
                        "test": 1,
                    },
                )


if __name__ == "__main__":
    unittest.main()
