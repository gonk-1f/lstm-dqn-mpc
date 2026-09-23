"""Zero-boundary trimming and reconstruction for the v2 operating dataset."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from utils.rebuilt_operating_dataset import pchip_to_one_second


ZERO_DEADBAND_KW = 1.0
ACTIVE_THRESHOLD_KW = 1.0
SUSTAINED_POINTS = 3


@dataclass(frozen=True)
class BoundaryAudit:
    side: str
    kind: str
    left_timestamp: pd.Timestamp
    left_load_kw: float
    right_timestamp: pd.Timestamp
    right_load_kw: float
    original_boundary_load_kw: float | None
    canonical_boundary_load_kw: float
    crossing_fraction: float | None
    unrounded_boundary_timestamp: pd.Timestamp
    rounded_boundary_timestamp: pd.Timestamp
    rounding_adjustment_seconds: float
    removed_point_count: int
    removed_duration_s: float
    sustained_block_start_timestamp: pd.Timestamp
    sustained_block_end_timestamp: pd.Timestamp


@dataclass(frozen=True)
class TrimmedParent:
    frame: pd.DataFrame
    start: BoundaryAudit
    end: BoundaryAudit


def _sustained_starts(
    values: np.ndarray,
    threshold_kw: float,
    count: int,
) -> np.ndarray:
    positive = values > threshold_kw
    return np.flatnonzero(
        np.convolve(
            positive.astype(int),
            np.ones(count, dtype=int),
            mode="valid",
        )
        == count
    )


def _validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "timestamp",
        "fc_total_kw",
        "battery_raw_total_kw",
        "source_total_kw",
        "is_cubic_imputed",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"boundary source is missing columns: {sorted(missing)}")
    source = frame.copy().reset_index(drop=True)
    source["timestamp"] = pd.to_datetime(source["timestamp"], errors="coerce")
    if source["timestamp"].isna().any():
        raise ValueError("boundary source contains invalid timestamps")
    if (
        not source["timestamp"].is_monotonic_increasing
        or source["timestamp"].duplicated().any()
    ):
        raise ValueError("boundary source timestamps must be strictly increasing")
    for column in ("fc_total_kw", "battery_raw_total_kw", "source_total_kw"):
        source[column] = pd.to_numeric(source[column], errors="coerce")
        if not np.isfinite(source[column].to_numpy(dtype=float)).all():
            raise ValueError("boundary source power must be finite")
    return source


def _crossing_row(
    left: pd.Series,
    right: pd.Series,
    *,
    side: str,
    removed_point_count: int,
    removed_duration_s: float,
    sustained_start: pd.Timestamp,
    sustained_end: pd.Timestamp,
) -> tuple[pd.Series, BoundaryAudit]:
    y0 = float(left.source_total_kw)
    y1 = float(right.source_total_kw)
    if y0 == y1 or y0 * y1 > 0.0:
        raise ValueError(f"{side} boundary is not bracketed")
    fraction = -y0 / (y1 - y0)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"{side} boundary requires extrapolation")
    left_time = pd.Timestamp(left.timestamp)
    right_time = pd.Timestamp(right.timestamp)
    raw_time = left_time + fraction * (right_time - left_time)
    rounded_time = raw_time.round("s")
    row = left.copy()
    row["timestamp"] = rounded_time
    row["fc_total_kw"] = float(left.fc_total_kw) + fraction * (
        float(right.fc_total_kw) - float(left.fc_total_kw)
    )
    row["battery_raw_total_kw"] = float(left.battery_raw_total_kw) + fraction * (
        float(right.battery_raw_total_kw) - float(left.battery_raw_total_kw)
    )
    row["source_total_kw"] = float(
        row.fc_total_kw - row.battery_raw_total_kw
    )
    row["is_cubic_imputed"] = False
    if abs(float(row.source_total_kw)) > 1.0e-9:
        raise AssertionError("constructed boundary violates power identity")
    row["source_total_kw"] = 0.0
    audit = BoundaryAudit(
        side=side,
        kind="CONSTRUCTED_CROSSING",
        left_timestamp=left_time,
        left_load_kw=y0,
        right_timestamp=right_time,
        right_load_kw=y1,
        original_boundary_load_kw=None,
        canonical_boundary_load_kw=0.0,
        crossing_fraction=float(fraction),
        unrounded_boundary_timestamp=raw_time,
        rounded_boundary_timestamp=rounded_time,
        rounding_adjustment_seconds=float((rounded_time - raw_time).total_seconds()),
        removed_point_count=removed_point_count,
        removed_duration_s=removed_duration_s,
        sustained_block_start_timestamp=sustained_start,
        sustained_block_end_timestamp=sustained_end,
    )
    return row, audit


def _observed_boundary(
    source: pd.DataFrame,
    index: int,
    *,
    side: str,
    removed_point_count: int,
    removed_duration_s: float,
    sustained_start: pd.Timestamp,
    sustained_end: pd.Timestamp,
) -> tuple[pd.Series, BoundaryAudit]:
    row = source.iloc[index].copy()
    original = float(row.source_total_kw)
    row["source_total_kw"] = 0.0
    left = source.iloc[max(0, index - 1)]
    right = source.iloc[min(len(source) - 1, index + 1)]
    timestamp = pd.Timestamp(row.timestamp)
    audit = BoundaryAudit(
        side=side,
        kind="OBSERVED_DEADBAND",
        left_timestamp=pd.Timestamp(left.timestamp),
        left_load_kw=float(left.source_total_kw),
        right_timestamp=pd.Timestamp(right.timestamp),
        right_load_kw=float(right.source_total_kw),
        original_boundary_load_kw=original,
        canonical_boundary_load_kw=0.0,
        crossing_fraction=None,
        unrounded_boundary_timestamp=timestamp,
        rounded_boundary_timestamp=timestamp,
        rounding_adjustment_seconds=0.0,
        removed_point_count=removed_point_count,
        removed_duration_s=removed_duration_s,
        sustained_block_start_timestamp=sustained_start,
        sustained_block_end_timestamp=sustained_end,
    )
    return row, audit


def trim_to_zero_boundaries(
    frame: pd.DataFrame,
    *,
    zero_deadband_kw: float = ZERO_DEADBAND_KW,
    active_threshold_kw: float = ACTIVE_THRESHOLD_KW,
    sustained_points: int = SUSTAINED_POINTS,
) -> TrimmedParent:
    """Trim one parent to auditable zero boundaries without extrapolation."""
    if zero_deadband_kw < 0.0 or active_threshold_kw < 0.0:
        raise ValueError("power thresholds must be nonnegative")
    if sustained_points < 1:
        raise ValueError("sustained_points must be positive")
    source = _validate_frame(frame)
    values = source["source_total_kw"].to_numpy(dtype=float)
    starts = _sustained_starts(values, active_threshold_kw, sustained_points)
    if len(starts) == 0:
        raise ValueError("no sustained positive operation")
    first_active = int(starts[0])
    last_active = int(starts[-1] + sustained_points - 1)
    sustained_start = pd.Timestamp(source.iloc[first_active].timestamp)
    sustained_end = pd.Timestamp(source.iloc[last_active].timestamp)

    start_deadband = np.flatnonzero(
        np.abs(values[:first_active]) <= zero_deadband_kw
    )
    start_index: int | None = None
    start_crossing: tuple[int, int] | None = None
    if len(start_deadband):
        start_index = int(start_deadband[-1])
    else:
        for right in range(first_active, 0, -1):
            left = right - 1
            if values[left] < -zero_deadband_kw and values[right] > active_threshold_kw:
                start_crossing = (left, right)
                break
    if start_index is None and start_crossing is None:
        raise ValueError("start boundary is not bracketed")

    end_deadband_relative = np.flatnonzero(
        np.abs(values[last_active + 1 :]) <= zero_deadband_kw
    )
    end_index: int | None = None
    end_crossing: tuple[int, int] | None = None
    if len(end_deadband_relative):
        end_index = int(last_active + 1 + end_deadband_relative[0])
    else:
        for left in range(last_active, len(source) - 1):
            right = left + 1
            if values[left] > active_threshold_kw and values[right] < -zero_deadband_kw:
                end_crossing = (left, right)
                break
    if end_index is None and end_crossing is None:
        raise ValueError("end boundary is not bracketed")

    original_start = pd.Timestamp(source.iloc[0].timestamp)
    original_end = pd.Timestamp(source.iloc[-1].timestamp)
    if start_index is not None:
        start_row, start_audit = _observed_boundary(
            source,
            start_index,
            side="start",
            removed_point_count=start_index,
            removed_duration_s=float(
                (pd.Timestamp(source.iloc[start_index].timestamp) - original_start).total_seconds()
            ),
            sustained_start=sustained_start,
            sustained_end=sustained_end,
        )
        retained_start = start_index + 1
    else:
        assert start_crossing is not None
        left, right = start_crossing
        start_row, start_audit = _crossing_row(
            source.iloc[left],
            source.iloc[right],
            side="start",
            removed_point_count=right,
            removed_duration_s=float(
                (pd.Timestamp(source.iloc[right].timestamp) - original_start).total_seconds()
            ),
            sustained_start=sustained_start,
            sustained_end=sustained_end,
        )
        retained_start = right

    if end_index is not None:
        end_row, end_audit = _observed_boundary(
            source,
            end_index,
            side="end",
            removed_point_count=len(source) - end_index - 1,
            removed_duration_s=float(
                (original_end - pd.Timestamp(source.iloc[end_index].timestamp)).total_seconds()
            ),
            sustained_start=sustained_start,
            sustained_end=sustained_end,
        )
        retained_end = end_index
    else:
        assert end_crossing is not None
        left, right = end_crossing
        end_row, end_audit = _crossing_row(
            source.iloc[left],
            source.iloc[right],
            side="end",
            removed_point_count=len(source) - left - 1,
            removed_duration_s=float(
                (original_end - pd.Timestamp(source.iloc[left].timestamp)).total_seconds()
            ),
            sustained_start=sustained_start,
            sustained_end=sustained_end,
        )
        retained_end = left

    middle = source.iloc[retained_start : retained_end + 1].copy()
    rows = [start_row]
    if not middle.empty:
        rows.extend(row for _, row in middle.iterrows())
    rows.append(end_row)
    trimmed = pd.DataFrame(rows).reset_index(drop=True)
    trimmed = trimmed.drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    trimmed.iloc[0, trimmed.columns.get_loc("source_total_kw")] = 0.0
    trimmed.iloc[-1, trimmed.columns.get_loc("source_total_kw")] = 0.0
    if len(trimmed) < 2 or not trimmed["timestamp"].is_monotonic_increasing:
        raise ValueError("trimmed boundary timestamps are invalid")
    return TrimmedParent(trimmed, start_audit, end_audit)


def reconstruct_one_second(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int | float]]:
    """Reconstruct a trimmed parent on an exact one-second model time axis."""
    source = frame[["timestamp", "source_total_kw"]].rename(
        columns={"source_total_kw": "load_total_kw"}
    )
    output, qa = pchip_to_one_second(source)
    endpoint = output["load_total_kw"].iloc[[0, -1]].to_numpy(dtype=float)
    if not np.allclose(endpoint, 0.0, rtol=0.0, atol=1.0e-9):
        raise ValueError("one-second reconstruction changed a zero boundary")
    if not np.isfinite(output["load_total_kw"].to_numpy(dtype=float)).all():
        raise ValueError("one-second reconstruction produced non-finite load")
    output.loc[output.index[[0, -1]], "load_total_kw"] = 0.0
    return output, qa
