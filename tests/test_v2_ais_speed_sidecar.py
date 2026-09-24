from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import csv
import hashlib
import sys
import tempfile
import unittest
from zoneinfo import ZoneInfo

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from v2.data.ais_speed_sidecar import (  # noqa: E402
    AIS_GAP_PROVENANCE,
    AIS_RAW_PROVENANCE,
    align_supervisory_speed,
    build_ais_speed_sidecar,
)


TZ = ZoneInfo("Asia/Shanghai")


class TestAisSpeedSidecar(unittest.TestCase):
    def setUp(self) -> None:
        self.origin = datetime(2024, 1, 1, tzinfo=TZ)

    def times(self, *seconds: int) -> tuple[datetime, ...]:
        return tuple(self.origin + timedelta(seconds=value) for value in seconds)

    def test_uses_unique_nearest_record_within_ten_seconds_without_reuse(self) -> None:
        result = align_supervisory_speed(
            self.times(30, 60, 90),
            self.times(38, 52, 98),
            (1.0, 2.0, 3.0),
        )
        np.testing.assert_allclose(result.speed_kn, (1.0, 2.0, 3.0))
        self.assertEqual(result.provenance, (AIS_RAW_PROVENANCE,) * 3)

        no_reuse = align_supervisory_speed(
            self.times(30, 35),
            self.times(32, 70),
            (4.0, 8.0),
            fill_internal_gaps=False,
        )
        self.assertEqual(no_reuse.provenance, (AIS_RAW_PROVENANCE, "UNAVAILABLE"))
        self.assertTrue(np.isnan(no_reuse.speed_kn[1]))

    def test_pchip_fills_only_bracketed_internal_gap_and_is_bounded(self) -> None:
        result = align_supervisory_speed(
            self.times(30, 60, 90, 120, 150),
            self.times(25, 155),
            (2.0, 8.0),
        )
        self.assertEqual(result.provenance[0], AIS_RAW_PROVENANCE)
        self.assertEqual(result.provenance[-1], AIS_RAW_PROVENANCE)
        self.assertEqual(
            result.provenance[1:4],
            (AIS_GAP_PROVENANCE, AIS_GAP_PROVENANCE, AIS_GAP_PROVENANCE),
        )
        self.assertTrue(np.all(result.speed_kn[1:4] >= 2.0))
        self.assertTrue(np.all(result.speed_kn[1:4] <= 8.0))

    def test_never_extrapolates_or_accepts_negative_speed(self) -> None:
        result = align_supervisory_speed(
            self.times(0, 30, 60, 90, 120),
            self.times(25, 85),
            (1.0, 3.0),
        )
        self.assertTrue(np.isnan(result.speed_kn[0]))
        self.assertTrue(np.isnan(result.speed_kn[-1]))
        self.assertEqual(result.provenance[0], "UNAVAILABLE")
        self.assertEqual(result.provenance[-1], "UNAVAILABLE")

        with self.assertRaisesRegex(ValueError, "nonnegative"):
            align_supervisory_speed(
                self.times(30), self.times(25), (-0.1,)
            )

    def test_rejects_naive_or_unsorted_timestamps(self) -> None:
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            align_supervisory_speed(
                (datetime(2024, 1, 1),), self.times(0), (0.0,)
            )
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            align_supervisory_speed(
                self.times(60, 30), self.times(0), (0.0,)
            )

    def test_builder_preserves_split_and_authenticates_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            raw = root / "raw"
            output = root / "sidecar"
            (dataset / "metadata").mkdir(parents=True)
            (dataset / "train").mkdir()
            (raw / "p1" / "推进系统").mkdir(parents=True)
            segment = dataset / "train" / "s1.csv"
            segment.write_text(
                "timestamp,time_s,load_total_kw\n"
                "2024-01-01 00:00:00+08:00,0,0\n"
                "2024-01-01 00:00:30+08:00,30,10\n"
                "2024-01-01 00:01:00+08:00,60,20\n"
                "2024-01-01 00:01:30+08:00,90,10\n"
                "2024-01-01 00:02:00+08:00,120,0\n",
                encoding="utf-8",
            )
            segment_hash = hashlib.sha256(segment.read_bytes()).hexdigest()
            (dataset / "metadata" / "sample_manifest.csv").write_text(
                "parent,sample_id,relative_path,split,point_count_1s,start_timestamp,end_timestamp,duration_s,sha256\n"
                f"p1,s1,train/s1.csv,train,5,2024-01-01T00:00:00+08:00,2024-01-01T00:02:00+08:00,120,{segment_hash}\n",
                encoding="utf-8",
            )
            (raw / "p1" / "推进系统" / "AIS航速_p1.csv").write_text(
                "Time,航速(节)\n"
                "2024-01-01 00:00:00,0 kn\n"
                "2024-01-01 00:00:20,2 kn\n"
                "2024-01-01 00:01:20,6 kn\n"
                "2024-01-01 00:01:30,4 kn\n",
                encoding="utf-8-sig",
            )

            summary = build_ais_speed_sidecar(dataset, raw, output)
            self.assertEqual(summary["split_segment_counts"], {"train": 1})
            self.assertEqual(summary["supervisory_row_count"], 4)
            sidecar = output / "train" / "s1.csv"
            with sidecar.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([float(row["time_s"]) for row in rows], [0, 30, 60, 90])
            self.assertEqual(rows[0]["speed_provenance"], AIS_RAW_PROVENANCE)
            self.assertEqual(rows[2]["speed_provenance"], AIS_GAP_PROVENANCE)
            manifest = (output / "metadata" / "sample_manifest.csv").read_text(
                encoding="utf-8-sig"
            )
            self.assertIn(hashlib.sha256(sidecar.read_bytes()).hexdigest(), manifest)
            with self.assertRaises(FileExistsError):
                build_ais_speed_sidecar(dataset, raw, output)


if __name__ == "__main__":
    unittest.main()
