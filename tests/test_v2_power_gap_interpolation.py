from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from v2.data.power_gap_interpolation import interpolate_power_gaps


class PowerGapInterpolationTests(unittest.TestCase):
    def test_interpolates_only_nominal_interior_slots_and_preserves_identity(self):
        t0 = datetime(2024, 5, 11, 8, 20, tzinfo=timezone.utc)
        times = tuple(
            t0 + timedelta(seconds=seconds)
            for seconds in (0, 30, 60, 450, 480, 510)
        )
        fc = np.array([90.0, 100.0, 110.0, 180.0, 190.0, 200.0])
        batt = np.array([-10.0, -12.0, -14.0, -30.0, -32.0, -34.0])

        result = interpolate_power_gaps(times, fc, batt)

        expected = tuple(
            times[2] + timedelta(seconds=30 * number) for number in range(1, 13)
        )
        self.assertEqual(
            tuple(
                timestamp
                for timestamp, flag in zip(result.timestamps, result.is_interpolated)
                if flag
            ),
            expected,
        )
        np.testing.assert_allclose(
            result.source_kw, result.fc_kw - result.battery_raw_kw
        )
        self.assertEqual(result.timestamps[0], times[0])
        self.assertEqual(result.timestamps[-1], times[-1])
        self.assertEqual(result.interpolation_gap_count, 1)
        self.assertEqual(result.interpolated_point_count, 12)
        self.assertEqual(result.max_interpolated_gap_seconds, 390.0)

    def test_preserves_observed_values_exactly(self):
        t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
        times = tuple(
            t0 + timedelta(seconds=seconds)
            for seconds in (0, 30, 60, 120, 150, 180)
        )
        fc = np.array([1.0, 2.0, 3.0, 5.0, 6.0, 7.0])
        batt = np.array([-1.0, -2.0, -3.0, -5.0, -6.0, -7.0])

        result = interpolate_power_gaps(times, fc, batt)
        observed = ~result.is_interpolated

        np.testing.assert_array_equal(result.fc_kw[observed], fc)
        np.testing.assert_array_equal(result.battery_raw_kw[observed], batt)

    def test_rejects_gap_with_fewer_than_four_distinct_anchors(self):
        t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
        times = (t0, t0 + timedelta(seconds=390), t0 + timedelta(seconds=420))

        with self.assertRaisesRegex(ValueError, "fewer than four anchors"):
            interpolate_power_gaps(
                times, [10.0, 20.0, 30.0], [-1.0, -2.0, -3.0]
            )

    def test_reports_local_range_overshoot_without_clipping(self):
        t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
        times = tuple(
            t0 + timedelta(seconds=seconds)
            for seconds in (0, 30, 60, 450, 480, 510)
        )

        result = interpolate_power_gaps(
            times,
            [0.0, 200.0, 0.0, 200.0, 0.0, 200.0],
            [0.0, -100.0, 0.0, -100.0, 0.0, -100.0],
        )

        self.assertTrue(result.warning_messages)
        self.assertTrue(
            any("OVERSHOOT" in value for value in result.warning_messages)
        )


if __name__ == "__main__":
    unittest.main()
