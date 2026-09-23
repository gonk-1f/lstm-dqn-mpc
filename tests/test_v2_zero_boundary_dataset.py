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
    reconstruct_one_second,
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


if __name__ == "__main__":
    unittest.main()
