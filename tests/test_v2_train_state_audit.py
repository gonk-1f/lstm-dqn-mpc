from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

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


class TrainStateAuditCausalFeatureTests(unittest.TestCase):
    @staticmethod
    def _fixture(include_future: bool = False):
        from v2.analysis.train_state_audit import TrainSegment
        from v2.data.supervisory_rules import OperatingMode
        from v2.data.train_supervisory_audit import ParentSupervisoryState

        start = datetime(2024, 1, 1, tzinfo=ZoneInfo("Asia/Shanghai"))
        loads = [100.0, 120.0, 140.0, 160.0, 180.0, 200.0]
        fc = [80.0, 90.0, 100.0, 110.0, 120.0, 130.0]
        soc = [0.600, 0.599, 0.598, 0.597, 0.596, 0.595]
        if include_future:
            loads.append(999.0)
            fc.append(500.0)
            soc.append(0.594)
        states = []
        for index, (load_kw, fc_kw, soc_value) in enumerate(zip(loads, fc, soc)):
            timestamp = start + timedelta(seconds=30 * index)
            previous_fc = 70.0 if index == 0 else fc[index - 1]
            states.append(
                ParentSupervisoryState(
                    parent_id="train_parent",
                    timestamp=timestamp,
                    mode=OperatingMode.SAILING_ISLAND,
                    p_fc_total_kw=fc_kw,
                    p_batt_total_kw=load_kw - fc_kw,
                    p_load_kw=load_kw,
                    soc_system=soc_value,
                    previous_p_fc_total_kw=previous_fc,
                    channels_complete=True,
                    conflicting_duplicate=False,
                    long_gap_contaminated=False,
                    source_provenance=(),
                )
            )
        load_frame = pd.DataFrame(
            {
                "timestamp": [state.timestamp for state in states],
                "time_s": [float(index * 30) for index in range(len(states))],
                "load_total_kw": loads,
            }
        )
        segment = TrainSegment(
            parent="train_parent",
            sample_id="train_001",
            relative_path="train/train_001.csv",
            start_timestamp=start,
            end_timestamp=start + timedelta(seconds=150),
            sha256="synthetic",
            dataset_version="operating_dataset_zero_boundary_v2",
        )
        return segment, tuple(states), load_frame

    def test_builds_six_sample_causal_window_with_frozen_lpf(self) -> None:
        from v2.analysis.train_state_audit import build_causal_feature_rows

        segment, states, load_frame = self._fixture()

        rows = build_causal_feature_rows(segment, states, load_frame)

        self.assertEqual(len(rows), 1)
        row = rows[0]
        alpha = math.exp(-30.0 / 90.0)
        expected_base = 100.0
        for load in (120.0, 140.0, 160.0, 180.0, 200.0):
            expected_base = alpha * expected_base + (1.0 - alpha) * load
        self.assertAlmostEqual(row.base_load_kw, expected_base)
        self.assertAlmostEqual(row.delta_load_kw, 200.0 - expected_base)
        self.assertAlmostEqual(row.delta_fc_kw, 10.0)
        self.assertAlmostEqual(row.recent_delta_soc, -0.005)
        self.assertAlmostEqual(row.recent_load_mean_kw, 150.0)
        self.assertAlmostEqual(row.recent_load_trend_kw_per_s, 20.0 / 30.0)
        self.assertAlmostEqual(row.measured_power_balance_residual_kw, 0.0)
        self.assertEqual(row.history_sample_count, 6)

    def test_future_sample_cannot_change_current_feature_row(self) -> None:
        from v2.analysis.train_state_audit import build_causal_feature_rows

        segment, states, load_frame = self._fixture()
        expected = build_causal_feature_rows(segment, states, load_frame)
        future_segment, future_states, future_load_frame = self._fixture(
            include_future=True
        )

        actual = build_causal_feature_rows(
            future_segment,
            future_states,
            future_load_frame,
        )

        self.assertEqual(actual, expected)

    def test_history_before_segment_boundary_is_not_reused(self) -> None:
        from v2.analysis.train_state_audit import build_causal_feature_rows

        segment, states, load_frame = self._fixture()
        later_segment = type(segment)(
            parent=segment.parent,
            sample_id="train_002",
            relative_path="train/train_002.csv",
            start_timestamp=segment.start_timestamp + timedelta(seconds=30),
            end_timestamp=segment.end_timestamp,
            sha256="synthetic",
            dataset_version=segment.dataset_version,
        )

        rows = build_causal_feature_rows(later_segment, states, load_frame)

        self.assertEqual(rows, ())

    def test_train_row_builder_loads_only_whitelisted_parent_and_segment(self) -> None:
        from v2.analysis.train_state_audit import build_train_feature_rows

        segment, states, load_frame = self._fixture()
        calls: list[str] = []

        def state_loader(parent: str):
            calls.append(parent)
            return states

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / segment.relative_path
            path.parent.mkdir(parents=True)
            load_frame.to_csv(path, index=False)

            rows = build_train_feature_rows(root, (segment,), state_loader)

        self.assertEqual(calls, ["train_parent"])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].sample_id, "train_001")


if __name__ == "__main__":
    unittest.main()
