from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
MAIN = ROOT / "src" / "main"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(MAIN) not in sys.path:
    sys.path.insert(0, str(MAIN))

from utils.data_aligner import (  # noqa: E402
    FUEL_CELL_POWER_COLUMN,
    VoyageAlignmentError,
    _read_numeric_channel,
)


class TestTotalLoadDataset721(unittest.TestCase):
    def test_read_numeric_channel_repairs_only_known_voyage_054_timestamp_erratum(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target_dir = root / "7月9日08_00_7月9日16_00" / "燃料电池系统"
            target_dir.mkdir(parents=True)
            target_path = target_dir / "左氢燃料电池#1_fixture.csv"
            pd.DataFrame(
                {
                    "Time": [
                        "2024-07-09 09:56:19",
                        "2024-07-09 09:57:22",
                        "2024-07-09 09:57:22",
                        "2024-07-09 09:57:49",
                    ],
                    FUEL_CELL_POWER_COLUMN: [10.0, 20.0, 30.0, 40.0],
                }
            ).to_csv(target_path, index=False, encoding="utf-8-sig")
            source_bytes = target_path.read_bytes()

            corrected, metadata = _read_numeric_channel(
                target_path,
                [FUEL_CELL_POWER_COLUMN],
                "fuel_cell_left_1",
                {},
            )

            self.assertEqual(
                corrected["timestamp"].tolist(),
                list(
                    pd.to_datetime(
                        [
                            "2024-07-09 09:56:19",
                            "2024-07-09 09:56:49",
                            "2024-07-09 09:57:22",
                            "2024-07-09 09:57:49",
                        ]
                    )
                ),
            )
            self.assertEqual(
                corrected[FUEL_CELL_POWER_COLUMN].tolist(),
                [10.0, 20.0, 30.0, 40.0],
            )
            self.assertEqual(metadata["timestamp_correction_count"], 1)
            self.assertEqual(metadata["conflicting_duplicate_timestamp_count"], 0)
            correction = metadata["timestamp_corrections"][0]
            self.assertEqual(correction["original_timestamp"], "2024-07-09T09:57:22")
            self.assertEqual(correction["corrected_timestamp"], "2024-07-09T09:56:49")
            self.assertEqual(correction["csv_line_number"], 3)
            self.assertEqual(correction["raw_row_position_zero_based"], 1)
            self.assertIn("adjacent 30 s cadence", correction["reason"])
            self.assertIn("synchronized 20-channel row order", correction["reason"])
            self.assertEqual(target_path.read_bytes(), source_bytes)

            with self.assertRaises(VoyageAlignmentError) as raised:
                _read_numeric_channel(
                    target_path,
                    [FUEL_CELL_POWER_COLUMN],
                    "fuel_cell_right_1",
                    {},
                )
            self.assertEqual(raised.exception.code, "timestamp_errata_mismatch")

            ordinary_dir = root / "ordinary_voyage" / "燃料电池系统"
            ordinary_dir.mkdir(parents=True)
            ordinary_path = ordinary_dir / target_path.name
            ordinary_path.write_bytes(source_bytes)
            ordinary, ordinary_metadata = _read_numeric_channel(
                ordinary_path,
                [FUEL_CELL_POWER_COLUMN],
                "fuel_cell_left_1",
                {},
            )
            self.assertEqual(len(ordinary), 3)
            self.assertEqual(ordinary_metadata["timestamp_correction_count"], 0)
            self.assertEqual(ordinary_metadata["timestamp_corrections"], [])
            self.assertEqual(ordinary_metadata["conflicting_duplicate_timestamp_count"], 1)


if __name__ == "__main__":
    unittest.main()
