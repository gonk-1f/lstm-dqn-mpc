from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class BatterySupervisoryAggregationTests(unittest.TestCase):
    @staticmethod
    def _samples(*, age_seconds: float = 0.0):
        from v2.data.supervisory_aggregation import AlignedBatteryClusterSample

        supervisory = datetime(2026, 1, 1, 0, 0, 30, tzinfo=timezone.utc)
        source = supervisory - timedelta(seconds=age_seconds)
        return tuple(
            AlignedBatteryClusterSample(
                cluster_id=f"bms-{index:02d}",
                supervisory_timestamp=supervisory,
                source_timestamp=source,
                soc=0.40 + index / 100.0,
                power_kw=float(index),
                provenance=f"train-parent/raw-bms-{index:02d}.csv:row-{index}",
            )
            for index in range(1, 13)
        )

    def test_equal_capacity_soc_mean_and_power_sum_require_all_twelve_clusters(self) -> None:
        from v2.data.supervisory_aggregation import aggregate_battery_clusters

        samples = self._samples(age_seconds=0.25)
        state = aggregate_battery_clusters(
            samples,
            expected_cluster_ids=tuple(f"bms-{index:02d}" for index in range(1, 13)),
            max_age_seconds=1.0,
        )

        self.assertAlmostEqual(state.soc_system, sum(sample.soc for sample in samples) / 12.0)
        self.assertEqual(state.p_batt_total_kw, sum(sample.power_kw for sample in samples))
        self.assertEqual(len(state.cluster_provenance), 12)

    def test_missing_duplicate_future_and_stale_cluster_samples_are_rejected(self) -> None:
        from v2.data.supervisory_aggregation import (
            AlignedBatteryClusterSample,
            aggregate_battery_clusters,
        )

        expected = tuple(f"bms-{index:02d}" for index in range(1, 13))
        samples = self._samples()
        with self.assertRaises(ValueError):
            AlignedBatteryClusterSample(
                cluster_id="bms-12",
                supervisory_timestamp=samples[-1].supervisory_timestamp,
                source_timestamp=samples[-1].supervisory_timestamp + timedelta(seconds=1),
                soc=samples[-1].soc,
                power_kw=samples[-1].power_kw,
                provenance=samples[-1].provenance,
            )

        attacks = (
            samples[:-1],
            samples + (samples[0],),
            self._samples(age_seconds=2.0),
        )
        for attacked in attacks:
            with self.subTest(size=len(attacked)), self.assertRaises(ValueError):
                aggregate_battery_clusters(
                    attacked,
                    expected_cluster_ids=expected,
                    max_age_seconds=1.0,
                )


if __name__ == "__main__":
    unittest.main()
