from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_dataset_fixture(root: Path, *, train_relative_path: str) -> None:
    metadata = root / "metadata"
    train = root / "train"
    metadata.mkdir(parents=True)
    train.mkdir()
    train_path = train / "train_001.csv"
    pd.DataFrame(
        {
            "timestamp": ["2024-01-01T00:00:00+08:00"],
            "time_s": [0.0],
            "load_total_kw": [10.0],
        }
    ).to_csv(train_path, index=False)
    pd.DataFrame(
        [
            {
                "parent": "train_parent",
                "sample_id": "train_001",
                "relative_path": train_relative_path,
                "split": "train",
                "point_count_1s": 1,
                "start_timestamp": "2024-01-01T00:00:00+08:00",
                "end_timestamp": "2024-01-01T00:00:00+08:00",
                "duration_s": 0.0,
                "sha256": _sha256(train_path),
            },
            {
                "parent": "validation_parent",
                "sample_id": "validation_001",
                "relative_path": "validation/validation_001.csv",
                "split": "validation",
                "point_count_1s": 1,
                "start_timestamp": "2024-01-02T00:00:00+08:00",
                "end_timestamp": "2024-01-02T00:00:00+08:00",
                "duration_s": 0.0,
                "sha256": "held-out-file-must-not-be-opened",
            },
            {
                "parent": "test_parent",
                "sample_id": "test_001",
                "relative_path": "test/test_001.csv",
                "split": "test",
                "point_count_1s": 1,
                "start_timestamp": "2024-01-03T00:00:00+08:00",
                "end_timestamp": "2024-01-03T00:00:00+08:00",
                "duration_s": 0.0,
                "sha256": "held-out-file-must-not-be-opened",
            },
        ]
    ).to_csv(metadata / "sample_manifest.csv", index=False)
    (metadata / "qa_summary.json").write_text(
        json.dumps({"dataset_version": "operating_dataset_zero_boundary_v2"}),
        encoding="utf-8",
    )


class TrainStateAuditInputTests(unittest.TestCase):
    def test_load_train_segments_filters_held_out_without_opening_them(self) -> None:
        from v2.analysis.train_state_audit import load_train_segments

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            _write_dataset_fixture(root, train_relative_path="train/train_001.csv")

            segments = load_train_segments(root)

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].parent, "train_parent")
        self.assertEqual(segments[0].sample_id, "train_001")
        self.assertEqual(segments[0].relative_path, "train/train_001.csv")
        self.assertEqual(
            segments[0].dataset_version,
            "operating_dataset_zero_boundary_v2",
        )

    def test_load_train_segments_rejects_non_train_or_escaping_path(self) -> None:
        from v2.analysis.train_state_audit import load_train_segments

        for relative_path in ("validation/train_001.csv", "../train/train_001.csv"):
            with self.subTest(relative_path=relative_path):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary) / "dataset"
                    _write_dataset_fixture(root, train_relative_path=relative_path)
                    with self.assertRaisesRegex(ValueError, "Train path"):
                        load_train_segments(root)

    def test_load_train_segments_rejects_mutated_train_file(self) -> None:
        from v2.analysis.train_state_audit import load_train_segments

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            _write_dataset_fixture(root, train_relative_path="train/train_001.csv")
            (root / "train" / "train_001.csv").write_text(
                "mutated\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_train_segments(root)


if __name__ == "__main__":
    unittest.main()
