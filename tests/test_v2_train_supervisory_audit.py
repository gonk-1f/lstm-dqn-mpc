from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class TrainSupervisoryAuditBuilderTests(unittest.TestCase):
    def test_duplicate_resolution_drops_exact_rows_and_excludes_conflicts(self) -> None:
        from v2.data.train_supervisory_audit import RawRecord, resolve_duplicates

        t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
        t1 = t0 + timedelta(seconds=30)
        result = resolve_duplicates(
            (
                RawRecord(t0, (1.0,), "row-1"),
                RawRecord(t0, (1.0,), "row-2"),
                RawRecord(t1, (2.0,), "row-3"),
                RawRecord(t1, (3.0,), "row-4"),
            )
        )

        self.assertEqual(tuple(record.timestamp for record in result.records), (t0,))
        self.assertEqual(result.exact_duplicate_rows_removed, 1)
        self.assertEqual(result.conflicting_timestamps, (t1,))

    def test_causal_alignment_is_latest_fresh_and_without_source_reuse(self) -> None:
        from v2.data.train_supervisory_audit import causal_align_latest_without_reuse

        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        supervisory = tuple(start + timedelta(seconds=value) for value in (10, 20, 30))
        source = tuple(start + timedelta(seconds=value) for value in (0, 20, 31))

        self.assertEqual(
            causal_align_latest_without_reuse(supervisory, source),
            (0, 1, None),
        )

    @staticmethod
    def _parent_channels(*, stale_last_battery: bool = False):
        from v2.data.train_supervisory_audit import ParentRawChannels, RawChannel, RawRecord

        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        times = tuple(start + timedelta(seconds=30 * index) for index in range(3))

        def channel(channel_id: str, values: tuple[tuple[float, ...], ...]) -> RawChannel:
            return RawChannel(
                channel_id,
                tuple(
                    RawRecord(timestamp, value, f"train-parent/{channel_id}:{index}")
                    for index, (timestamp, value) in enumerate(zip(times, values))
                ),
            )

        fc = tuple(
            channel(f"fc-{index}", ((10.0,), (10.0,), (10.0,)))
            for index in range(8)
        )
        batteries = []
        for index in range(12):
            records = ((1.0, 0.5), (1.0, 0.5), (1.0, 0.5))
            current = channel(f"bms-{index}", records)
            if stale_last_battery and index == 11:
                current = RawChannel(current.channel_id, current.records[:-1])
            batteries.append(current)
        speed = channel("ais-speed", ((2.0,), (2.0,), (2.0,)))
        return ParentRawChannels("train-parent", fc, tuple(batteries), speed)

    def test_parent_builder_requires_all_channels_and_preserves_previous_fc(self) -> None:
        from v2.data.supervisory_rules import OperatingMode
        from v2.data.train_supervisory_audit import build_parent_supervisory_states

        complete = build_parent_supervisory_states(self._parent_channels())
        self.assertEqual(len(complete.states), 3)
        self.assertIs(complete.states[0].mode, OperatingMode.UNRESOLVED)
        self.assertIs(complete.states[1].mode, OperatingMode.ONBOARD)
        self.assertEqual(complete.states[1].p_fc_total_kw, 80.0)
        self.assertEqual(complete.states[1].p_batt_total_kw, 12.0)
        self.assertEqual(complete.states[1].p_load_kw, 92.0)
        self.assertEqual(complete.states[1].soc_system, 0.5)
        self.assertEqual(complete.states[1].previous_p_fc_total_kw, 80.0)
        self.assertEqual(len(complete.states[1].source_provenance), 21)

        stale = build_parent_supervisory_states(
            self._parent_channels(stale_last_battery=True)
        )
        self.assertIs(stale.states[-1].mode, OperatingMode.UNRESOLVED)
        self.assertIsNone(stale.states[-1].p_load_kw)
        self.assertFalse(stale.states[-1].channels_complete)


if __name__ == "__main__":
    unittest.main()
