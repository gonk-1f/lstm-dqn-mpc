"""Review-only cubic-spline filling for aligned segment power gaps."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Sequence

import numpy as np
from scipy.interpolate import CubicSpline


@dataclass(frozen=True)
class InterpolatedPowerSeries:
    timestamps: tuple[datetime, ...]
    fc_kw: np.ndarray
    battery_raw_kw: np.ndarray
    source_kw: np.ndarray
    is_interpolated: np.ndarray
    gap_intervals: tuple[tuple[datetime, datetime], ...]
    interpolation_gap_count: int
    interpolated_point_count: int
    max_interpolated_gap_seconds: float
    warning_messages: tuple[str, ...]


def _nominal_missing_count(gap_seconds: float, step_seconds: float) -> int:
    return max(1, round(gap_seconds / step_seconds) - 1)


def interpolate_power_gaps(
    timestamps: Sequence[datetime],
    fc_kw: Sequence[float],
    battery_raw_kw: Sequence[float],
    *,
    gap_threshold_seconds: float = 45.0,
    step_seconds: float = 30.0,
    anchor_points_each_side: int = 3,
) -> InterpolatedPowerSeries:
    """Fill internal long gaps while retaining observed samples unchanged."""
    times = tuple(timestamps)
    fc = np.asarray(fc_kw, dtype=float)
    battery = np.asarray(battery_raw_kw, dtype=float)
    if len(times) != len(fc) or len(times) != len(battery):
        raise ValueError("timestamp and power lengths differ")
    if any(right <= left for left, right in zip(times, times[1:])):
        raise ValueError("timestamps must be strictly increasing")
    if not np.all(np.isfinite(fc)) or not np.all(np.isfinite(battery)):
        raise ValueError("observed power must be finite")

    rows: list[tuple[datetime, float, float, bool]] = []
    gaps: list[tuple[datetime, datetime]] = []
    warnings: list[str] = []
    max_gap = 0.0
    for left_index in range(len(times) - 1):
        rows.append((times[left_index], fc[left_index], battery[left_index], False))
        gap_seconds = (times[left_index + 1] - times[left_index]).total_seconds()
        if gap_seconds <= gap_threshold_seconds:
            continue

        anchor_start = max(0, left_index - anchor_points_each_side + 1)
        anchor_end = min(len(times), left_index + 1 + anchor_points_each_side)
        anchor_times = times[anchor_start:anchor_end]
        if len(anchor_times) < 4:
            raise ValueError(
                f"gap {times[left_index].isoformat()} has fewer than four anchors"
            )
        origin = anchor_times[0]
        x = np.array(
            [(timestamp - origin).total_seconds() for timestamp in anchor_times]
        )
        fc_anchor = fc[anchor_start:anchor_end]
        battery_anchor = battery[anchor_start:anchor_end]
        fc_spline = CubicSpline(x, fc_anchor, bc_type="natural")
        battery_spline = CubicSpline(x, battery_anchor, bc_type="natural")

        missing_count = _nominal_missing_count(gap_seconds, step_seconds)
        generated = tuple(
            times[left_index] + timedelta(seconds=step_seconds * number)
            for number in range(1, missing_count + 1)
        )
        generated = tuple(
            timestamp
            for timestamp in generated
            if timestamp < times[left_index + 1]
        )
        generated_x = np.array(
            [(timestamp - origin).total_seconds() for timestamp in generated]
        )
        generated_fc = np.asarray(fc_spline(generated_x), dtype=float)
        generated_battery = np.asarray(battery_spline(generated_x), dtype=float)
        if not np.all(np.isfinite(generated_fc)) or not np.all(
            np.isfinite(generated_battery)
        ):
            raise ValueError("cubic spline produced non-finite power")

        anchor_fc_min = float(np.min(fc_anchor))
        anchor_fc_max = float(np.max(fc_anchor))
        anchor_battery_min = float(np.min(battery_anchor))
        anchor_battery_max = float(np.max(battery_anchor))
        for timestamp, fc_value, battery_value in zip(
            generated, generated_fc, generated_battery
        ):
            if fc_value < 0.0 or not anchor_fc_min <= fc_value <= anchor_fc_max:
                warnings.append(f"{timestamp.isoformat()}:FC_OVERSHOOT")
            if not anchor_battery_min <= battery_value <= anchor_battery_max:
                warnings.append(f"{timestamp.isoformat()}:BATTERY_OVERSHOOT")
            rows.append(
                (timestamp, float(fc_value), float(battery_value), True)
            )
        gaps.append((times[left_index], times[left_index + 1]))
        max_gap = max(max_gap, gap_seconds)

    if times:
        rows.append((times[-1], fc[-1], battery[-1], False))
    rows.sort(key=lambda row: row[0])
    output_times = tuple(row[0] for row in rows)
    output_fc = np.array([row[1] for row in rows])
    output_battery = np.array([row[2] for row in rows])
    mask = np.array([row[3] for row in rows], dtype=bool)
    return InterpolatedPowerSeries(
        timestamps=output_times,
        fc_kw=output_fc,
        battery_raw_kw=output_battery,
        source_kw=output_fc - output_battery,
        is_interpolated=mask,
        gap_intervals=tuple(gaps),
        interpolation_gap_count=len(gaps),
        interpolated_point_count=int(np.count_nonzero(mask)),
        max_interpolated_gap_seconds=max_gap,
        warning_messages=tuple(warnings),
    )
