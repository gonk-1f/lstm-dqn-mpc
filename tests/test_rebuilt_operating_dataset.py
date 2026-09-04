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
MAIN = SRC / "main"
if str(MAIN) not in sys.path:
    sys.path.insert(0, str(MAIN))


from utils.rebuilt_operating_dataset import (  # noqa: E402
    align_ais_to_power,
    collapse_battery_records,
    collapse_scalar_records,
    find_contiguous_intervals,
    match_nearest_without_reuse,
    pchip_to_one_second,
    select_shore_intervals,
    chronological_parent_splits,
)
from build_rebuilt_operating_segment_dataset import (  # noqa: E402
    _classify_negative_intervals,
    _mark_abnormal,
    _nearest_offsets,
    _policies,
    _segments,
    _shore_policy,
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
        self.assertAlmostEqual(float(collapsed.loc[0, "power_kw"]), 1.5)
        self.assertAlmostEqual(float(collapsed.loc[1, "power_kw"]), -1.0)
        self.assertEqual(stats["exact_duplicate_records_removed"], 1)
        self.assertEqual(stats["multi_value_timestamp_rows_aggregated"], 1)

    def test_scalar_duplicate_records_are_deduplicated_before_timestamp_average(self) -> None:
        timestamp = pd.Timestamp("2024-01-01 00:00:00")
        collapsed, stats = collapse_scalar_records(
            pd.DataFrame({"timestamp": [timestamp, timestamp, timestamp], "value": [2.0, 2.0, 4.0]}),
            value_column="value", output_column="soc_pct",
        )
        self.assertEqual(len(collapsed), 1)
        self.assertAlmostEqual(float(collapsed.loc[0, "soc_pct"]), 3.0)
        self.assertEqual(stats["exact_duplicate_records_removed"], 1)

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

    def test_normal_29_and_31_second_spacing_stays_in_one_interval(self) -> None:
        frame = pd.DataFrame(
            {"timestamp": pd.to_datetime(["2024-01-01 00:00:00", "2024-01-01 00:00:29", "2024-01-01 00:01:00"])}
        )
        intervals = find_contiguous_intervals(frame, long_gap_threshold_s=120.0)
        self.assertEqual(intervals.tolist(), [0, 0, 0])

    def test_long_gap_starts_a_new_interval(self) -> None:
        frame = pd.DataFrame(
            {"timestamp": pd.to_datetime(["2024-01-01 00:00:00", "2024-01-01 00:00:30", "2024-01-01 00:06:30"])}
        )
        intervals = find_contiguous_intervals(frame, long_gap_threshold_s=120.0)
        self.assertEqual(intervals.tolist(), [0, 0, 1])

    def test_ais_alignment_reports_exact_linear_nearest_and_unavailable(self) -> None:
        power = pd.DataFrame(
            {"timestamp": pd.to_datetime(["2024-01-01 00:00:00", "2024-01-01 00:00:10", "2024-01-01 00:00:25", "2024-01-01 00:02:00"])}
        )
        ais = pd.DataFrame(
            {"timestamp": pd.to_datetime(["2024-01-01 00:00:00", "2024-01-01 00:00:20", "2024-01-01 00:00:40"]), "ais_speed_kn": [0.0, 2.0, 4.0]}
        )
        aligned = align_ais_to_power(power, ais, max_normal_gap_s=30.0, max_nearest_s=6.0)
        self.assertEqual(aligned["speed_source"].tolist(), ["exact", "linear", "linear", "unavailable"])
        self.assertAlmostEqual(float(aligned.loc[1, "speed_aligned_kn"]), 1.0)

    def test_ais_nearest_is_used_only_when_no_bounding_pair_exists(self) -> None:
        power = pd.DataFrame({"timestamp": pd.to_datetime(["2024-01-01 00:00:00"])})
        ais = pd.DataFrame({"timestamp": pd.to_datetime(["2024-01-01 00:00:04"]), "ais_speed_kn": [1.2]})
        aligned = align_ais_to_power(power, ais, max_normal_gap_s=30.0, max_nearest_s=5.0)
        self.assertEqual(aligned.loc[0, "speed_source"], "nearest")
        self.assertAlmostEqual(float(aligned.loc[0, "speed_aligned_kn"]), 1.2)

    def test_missing_ais_does_not_split_a_stationary_shore_candidate(self) -> None:
        frame = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=4, freq="30s"),
                "fc_total_kw": [0.0] * 4,
                "battery_total_kw": [-100.0] * 4,
                "speed_aligned_kn": [0.0, np.nan, np.nan, 0.0],
                "aligned": [True] * 4,
            }
        )
        marked, intervals = select_shore_intervals(
            frame, fc_idle_threshold_kw=0.1, speed_idle_threshold_kn=0.1,
            battery_charge_threshold_kw=5.0, minimum_shore_points=3, long_gap_threshold_s=120.0,
        )
        self.assertTrue(marked["is_shore"].all())
        self.assertEqual(len(intervals), 1)

    def test_stationary_battery_discharge_is_not_shore(self) -> None:
        frame = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=4, freq="30s"),
                "fc_total_kw": [0.0] * 4,
                "battery_total_kw": [20.0] * 4,
                "speed_aligned_kn": [0.0] * 4,
                "aligned": [True] * 4,
            }
        )
        marked, intervals = select_shore_intervals(
            frame, fc_idle_threshold_kw=0.1, speed_idle_threshold_kn=0.1,
            battery_charge_threshold_kw=5.0, minimum_shore_points=3, long_gap_threshold_s=120.0,
        )
        self.assertFalse(marked["is_shore"].any())
        self.assertTrue(intervals.empty)

    def test_zero_load_is_retained_and_pchip_is_per_segment(self) -> None:
        first = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=2, freq="30s"), "load_total_kw": [0.0, 10.0]})
        second = pd.DataFrame({"timestamp": pd.date_range("2024-01-01 01:00:00", periods=2, freq="30s"), "load_total_kw": [100.0, 100.0]})
        first_1s, _ = pchip_to_one_second(first)
        second_1s, _ = pchip_to_one_second(second)
        self.assertEqual(float(first_1s.loc[0, "load_total_kw"]), 0.0)
        self.assertEqual(float(first_1s.loc[first_1s.index[-1], "load_total_kw"]), 10.0)
        self.assertEqual(float(second_1s.loc[second_1s.index[0], "load_total_kw"]), 100.0)

    def test_parent_voyages_are_kept_together_in_chronological_split(self) -> None:
        parents = [f"voyage_{index:03d}" for index in range(1, 67)]
        splits = chronological_parent_splits(parents)
        self.assertEqual([len(splits[name]) for name in ("train", "validation", "test")], [46, 13, 7])
        self.assertFalse(set(splits["train"]) & set(splits["validation"]))
        self.assertFalse(set(splits["train"]) & set(splits["test"]))
        self.assertFalse(set(splits["validation"]) & set(splits["test"]))

    def test_timestamp_offset_audit_accepts_read_only_pandas_arrays(self) -> None:
        reference = pd.Series(pd.to_datetime(["2024-01-01 00:00:00", "2024-01-01 00:00:30"]))
        source = pd.Series(pd.to_datetime(["2024-01-01 00:00:01", "2024-01-01 00:00:31"]))
        self.assertTrue(np.allclose(_nearest_offsets(reference, source), [1.0, 1.0]))

    def test_policy_audit_tolerates_an_empty_ais_frame(self) -> None:
        channel = pd.DataFrame({"timestamp": pd.to_datetime(["2024-01-01 00:00:00", "2024-01-01 00:00:30"]), "power_kw": [1.0, 1.0], "soc_pct": [50.0, 50.0]})
        cache = {"voyage": ([channel] * 4, [channel] * 12, {})}
        policy = _policies(cache, [pd.DataFrame(columns=["timestamp", "ais_speed_kn"])])
        self.assertGreater(policy["ais_max_normal_gap_s"], 0.0)

    def test_abnormal_audit_tolerates_missing_battery_slots(self) -> None:
        frame = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=3, freq="30s"),
                "fc_total_kw": [0.0, 0.0, 0.0],
                "battery_total_kw": [None, None, None],
                "propulsion_inverter_kw": [10.0, 10.0, 10.0],
                "soc_mean_pct": [50.0, 50.0, 50.0],
            }
        )
        marked, intervals, status = _mark_abnormal(frame, {"fc_idle_threshold_kw": 0.1})
        self.assertFalse(marked["is_abnormal"].any())
        self.assertTrue(intervals.empty)
        self.assertEqual(status, "available")

    def test_shore_speed_threshold_ignores_one_positive_speed_noise_point(self) -> None:
        speeds = [0.0047619] + [0.1] * 100
        frame = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=len(speeds), freq="30s"),
                "parent_voyage": ["voyage"] * len(speeds),
                "aligned": [True] * len(speeds),
                "fc_total_kw": [0.0] * len(speeds),
                "battery_total_kw": [-50.0] * len(speeds),
                "speed_aligned_kn": speeds,
            }
        )
        policy = _shore_policy(frame, {"median_power_cadence_s": 30.0})
        self.assertAlmostEqual(float(policy["speed_idle_threshold_kn"]), 0.1)

    def test_shore_battery_threshold_uses_low_stationary_charge_quantile(self) -> None:
        magnitudes = [0.01] * 10 + [0.1] * 90
        frame = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=len(magnitudes), freq="30s"),
                "parent_voyage": ["voyage"] * len(magnitudes),
                "aligned": [True] * len(magnitudes),
                "fc_total_kw": [0.0] * len(magnitudes),
                "battery_total_kw": [-value for value in magnitudes],
                "speed_aligned_kn": [0.0] * len(magnitudes),
            }
        )
        policy = _shore_policy(frame, {"median_power_cadence_s": 30.0})
        self.assertAlmostEqual(float(policy["battery_charge_threshold_kw"]), 0.01)

    def test_sustained_negative_load_is_external_when_stationary(self) -> None:
        frame = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=3, freq="30s"),
                "fc_total_kw": [20.0] * 3,
                "battery_total_kw": [-30.0] * 3,
                "load_total_kw": [-10.0] * 3,
                "speed_aligned_kn": [0.0] * 3,
                "propulsion_inverter_kw": [0.0] * 3,
                "aligned": [True] * 3,
                "is_shore": [False] * 3,
            }
        )
        marked, external, physical = _classify_negative_intervals(frame, speed_idle_threshold_kn=0.1)
        self.assertTrue(marked["is_external_charging"].all())
        self.assertEqual(len(external), 1)
        self.assertTrue(physical.empty)

    def test_sustained_negative_load_is_physical_inconsistency_when_moving(self) -> None:
        frame = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=3, freq="30s"),
                "fc_total_kw": [20.0] * 3,
                "battery_total_kw": [-30.0] * 3,
                "load_total_kw": [-10.0] * 3,
                "speed_aligned_kn": [2.0] * 3,
                "propulsion_inverter_kw": [0.0] * 3,
                "aligned": [True] * 3,
                "is_shore": [False] * 3,
            }
        )
        marked, external, physical = _classify_negative_intervals(frame, speed_idle_threshold_kn=0.1)
        self.assertTrue(marked["is_physical_inconsistency"].all())
        self.assertTrue(external.empty)
        self.assertEqual(len(physical), 1)

    def test_short_subkilowatt_negative_is_retained_as_tolerance(self) -> None:
        frame = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=2, freq="30s"),
                "fc_total_kw": [0.0, 0.0],
                "battery_total_kw": [-0.2, -0.2],
                "load_total_kw": [-0.2, -0.2],
                "speed_aligned_kn": [0.0, 0.0],
                "propulsion_inverter_kw": [0.0, 0.0],
                "aligned": [True, True],
                "is_shore": [False, False],
            }
        )
        marked, external, physical = _classify_negative_intervals(frame, speed_idle_threshold_kn=0.1)
        self.assertFalse(marked["is_external_charging"].any())
        self.assertFalse(marked["is_physical_inconsistency"].any())
        self.assertTrue(external.empty)
        self.assertTrue(physical.empty)

    def test_classified_negative_rows_form_segment_boundaries(self) -> None:
        frame = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=4, freq="30s"),
                "aligned": [True] * 4,
                "is_shore": [False] * 4,
                "is_external_charging": [False, True, False, False],
                "is_abnormal": [False] * 4,
                "is_physical_inconsistency": [False] * 4,
            }
        )
        segments = _segments(frame)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["timestamp"].tolist(), frame.loc[[2, 3], "timestamp"].tolist())
