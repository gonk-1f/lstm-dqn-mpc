"""DEPRECATED: tests for invalid linear 30 s to 1 s reconstruction.

This file is retained only as an archived implementation check. It is not
collected by the default pytest pattern because the linear 1 s workflow is no
longer an active forecasting workflow.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

MAIN_DIR = Path(__file__).resolve().parents[1] / "src" / "main"
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

from _deprecated_build_total_load_1s_linear_interp_DO_NOT_USE import build_one_voyage_1s, resolve_ais_speed_file


class BuildTotalLoad1sExcelsTests(unittest.TestCase):
    def test_build_one_voyage_1s_interpolates_power_and_speed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            voyage = "voyage_001"
            ais_dir = root / "ais" / voyage / "推进系统"
            ais_dir.mkdir(parents=True)
            ais_path = ais_dir / "AIS航速_voyage_001.csv"
            pd.DataFrame(
                {
                    "Time": ["2024-01-01 00:00:00", "2024-01-01 00:00:20", "2024-01-01 00:00:40"],
                    "航速(节)": ["0 kn", "10 kn", "20 kn"],
                }
            ).to_csv(ais_path, index=False, encoding="utf-8-sig")

            source = pd.DataFrame(
                {
                    "timestamp": ["2024-01-01 00:00:00", "2024-01-01 00:00:30"],
                    "fuel_cell_total_kw": [0.0, 30.0],
                    "battery_total_kw": [10.0, 40.0],
                    "total_load_fc_plus_batt_kw": [10.0, 70.0],
                }
            )

            built, meta = build_one_voyage_1s(
                source,
                voyage_name=voyage,
                ais_root=root / "ais",
                source_file_name=f"{voyage}.xlsx",
            )

            self.assertEqual(len(built), 31)
            self.assertEqual(float(built.loc[0, "time_s"]), 0.0)
            self.assertEqual(float(built.loc[30, "time_s"]), 30.0)
            self.assertAlmostEqual(float(built.loc[15, "fuel_cell_total_kw"]), 15.0)
            self.assertAlmostEqual(float(built.loc[15, "battery_total_kw"]), 25.0)
            self.assertAlmostEqual(float(built.loc[15, "total_load_fc_plus_batt_kw"]), 40.0)
            self.assertAlmostEqual(float(built.loc[10, "speed_knots"]), 5.0)
            self.assertAlmostEqual(float(built.loc[30, "speed_knots"]), 15.0)
            identity = built["total_load_fc_plus_batt_kw"] - built["fuel_cell_total_kw"] - built["battery_total_kw"]
            self.assertLess(float(identity.abs().max()), 1e-9)
            self.assertEqual(meta["ais_source_file"], ais_path.name)
            self.assertEqual(meta["num_samples_1s"], 31)
            self.assertEqual(meta["timestamp_gap_count"], 0)

    def test_resolve_ais_speed_file_requires_matching_voyage_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(FileNotFoundError):
                resolve_ais_speed_file(root, "missing_voyage")


if __name__ == "__main__":
    unittest.main()
