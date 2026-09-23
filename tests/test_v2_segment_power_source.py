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

from v2.data.segment_power_source import (  # noqa: E402
    align_near_synchronous_cycles,
    assemble_parent_power_series,
    power_conventions,
)
from v2.data.train_supervisory_audit import (  # noqa: E402
    ParentRawChannels,
    RawChannel,
    RawRecord,
)


class SegmentPowerSourceTests(unittest.TestCase):
    origin = datetime(2024, 5, 1, 8, 0, tzinfo=timezone.utc)

    @classmethod
    def times(cls, *seconds: int) -> tuple[datetime, ...]:
        return tuple(cls.origin + timedelta(seconds=value) for value in seconds)

    @classmethod
    def channel(
        cls,
        channel_id: str,
        value: float,
        *,
        battery: bool = False,
        offsets: tuple[int, ...] = (0, 30, 60),
    ) -> RawChannel:
        values = (float(value), 0.5) if battery else (float(value),)
        return RawChannel(
            channel_id,
            tuple(
                RawRecord(
                    cls.origin + timedelta(seconds=offset),
                    values,
                    f"{channel_id}@{offset}",
                )
                for offset in offsets
            ),
        )

    def test_power_conventions_use_raw_battery_sign(self):
        battery_raw, source = power_conventions(
            [100.0, 120.0],
            [20.0, -10.0],
        )

        np.testing.assert_allclose(battery_raw, [-20.0, 10.0])
        np.testing.assert_allclose(source, [120.0, 110.0])
        np.testing.assert_allclose(
            source,
            np.asarray([100.0, 120.0]) - battery_raw,
        )

    def test_alignment_uses_each_channel_sample_at_most_once(self):
        result = align_near_synchronous_cycles(
            self.times(0, 30, 60),
            (self.times(1, 31, 61), self.times(2, 32, 62)),
            tolerance_seconds=10.0,
            max_channel_span_seconds=10.0,
        )

        self.assertEqual(result.valid_reference_positions, (0, 1, 2))
        self.assertEqual(result.channel_indices, ((0, 1, 2), (0, 1, 2)))
        self.assertEqual(result.snapshot_timestamps, self.times(2, 32, 62))

    def test_alignment_rejects_an_equidistant_ambiguous_match(self):
        result = align_near_synchronous_cycles(
            self.times(30),
            (self.times(25, 35),),
            tolerance_seconds=10.0,
            max_channel_span_seconds=10.0,
        )

        self.assertEqual(result.channel_indices, ((None,),))
        self.assertEqual(result.valid_reference_positions, ())

    def test_assembles_all_power_channels_without_requiring_ais(self):
        channels = ParentRawChannels(
            "parent",
            tuple(
                self.channel(f"fc-{index}", 10.0)
                for index in range(8)
            ),
            tuple(
                self.channel(f"battery-{index}", 2.0, battery=True)
                for index in range(12)
            ),
            RawChannel("ais-speed", ()),
        )

        result = assemble_parent_power_series(channels)

        self.assertEqual(result.parent, "parent")
        self.assertEqual(result.timestamps, self.times(0, 30, 60))
        np.testing.assert_allclose(result.fc_total_kw, [80.0, 80.0, 80.0])
        np.testing.assert_allclose(
            result.battery_raw_total_kw,
            [-24.0, -24.0, -24.0],
        )
        np.testing.assert_allclose(
            result.source_total_kw,
            [104.0, 104.0, 104.0],
        )
        np.testing.assert_array_equal(result.ais_present, [False, False, False])


if __name__ == "__main__":
    unittest.main()
