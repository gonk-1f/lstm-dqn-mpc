from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from utils.rebuilt_operating_dataset import (  # noqa: E402
    collapse_battery_records,
    match_nearest_without_reuse,
)


class TestRebuiltOperatingDataset(unittest.TestCase):
    def test_battery_timestamp_collapsing_deduplicates_or_averages_power(self) -> None:
        timestamp = pd.Timestamp("2024-05-10 08:00:11")
        raw = pd.DataFrame(
            {
                "timestamp": [timestamp, timestamp, timestamp, timestamp + pd.Timedelta(seconds=30)],
                "voltage_v": [500.0, 500.0, 500.0, 500.0],
                "current_a": [-2.0, -2.0, -4.0, 2.0],
            }
        )

        collapsed, stats = collapse_battery_records(raw)

        self.assertEqual(len(collapsed), 2)
        self.assertAlmostEqual(float(collapsed.loc[0, "power_kw"]), 4.0 / 3.0)
        self.assertAlmostEqual(float(collapsed.loc[1, "power_kw"]), -1.0)
        self.assertEqual(stats["exact_duplicate_records_removed"], 0)
        self.assertEqual(stats["multi_value_timestamp_rows_aggregated"], 2)

    def test_nearest_alignment_never_uses_next_sample_to_fill_a_missing_slot(self) -> None:
        reference = pd.DataFrame(
            {"timestamp": pd.to_datetime(["2024-01-01 00:00:00", "2024-01-01 00:00:30"])}
        )
        source = pd.DataFrame(
            {"timestamp": pd.to_datetime(["2024-01-01 00:00:01"]), "power_kw": [10.0]}
        )

        matched = match_nearest_without_reuse(reference, source, tolerance_s=5.0)

        self.assertAlmostEqual(float(matched.loc[0, "power_kw"]), 10.0)
        self.assertTrue(np.isnan(matched.loc[1, "power_kw"]))
