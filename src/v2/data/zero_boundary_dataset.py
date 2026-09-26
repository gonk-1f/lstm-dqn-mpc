"""Zero-boundary trimming and reconstruction for the v2 operating dataset."""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter

import numpy as np
import pandas as pd

from utils.rebuilt_operating_dataset import pchip_to_one_second


ZERO_DEADBAND_KW = 1.0
ACTIVE_THRESHOLD_KW = 1.0
SUSTAINED_POINTS = 3
ASSIGNMENT_TRAIN_COUNT = 38
ASSIGNMENT_VALIDATION_COUNT = 10
TRAIN_COUNT = 30
VALIDATION_COUNT = 8
TEST_COUNT = 5
FIXED_TEST_PARENTS = (
    "3月26日14_00_3月26日16_00",
    "3月29日08_00_3月29日15_00",
    "4月18日12_00_4月18日18_00",
    "5月8日08_00_5月8日17_00",
    "6月11日08_00_6月11日11_00",
)
APPROVED_BOUNDARY_EXCLUSIONS = {
    "3月25日14_00_3月25日17_00": "start",
    "3月28日08_00_3月28日11_00": "end",
    "4月7日08_00_4月7日12_00": "end",
    "4月21日08_00_4月21日16_00": "end",
    "4月23日13_00_4月23日18_00": "start",
    "4月24日07_00_4月24日17_00": "end",
    "4月29日08_00_4月29日18_00": "end",
    "6月6日10_00_6月6日21_00": "end",
    "6月13日08_00_6月13日14_00": "start",
    "7月9日08_00_7月9日16_00": "end",
    "7月19日07_00_7月19日09_00": "end",
    "7月22日08_00_7月22日11_00": "end",
    "7月24日14_00_7月24日17_00": "end",
}
UNEXPLAINED_NEGATIVE_POWER_EXCLUSIONS = {
    "zero_boundary_014": {
        "parent": "4月25日08_00_4月25日18_00",
        "split": "train",
    },
    "zero_boundary_023": {
        "parent": "5月14日12_00_5月14日19_00",
        "split": "train",
    },
    "zero_boundary_024": {
        "parent": "5月15日08_00_5月15日17_00",
        "split": "validation",
    },
    "zero_boundary_025": {
        "parent": "5月17日08_00_5月17日18_00",
        "split": "train",
    },
    "zero_boundary_026": {
        "parent": "5月20日08_00_5月20日18_00",
        "split": "train",
    },
    "zero_boundary_027": {
        "parent": "5月21日05_00_5月21日11_00",
        "split": "validation",
    },
    "zero_boundary_028": {
        "parent": "5月24日08_00_5月24日19_00",
        "split": "train",
    },
    "zero_boundary_030": {
        "parent": "5月29日14_00_5月29日17_00",
        "split": "train",
    },
    "zero_boundary_031": {
        "parent": "5月30日05_00_5月30日17_00",
        "split": "train",
    },
    "zero_boundary_039": {
        "parent": "6月20日08_00_6月20日16_00",
        "split": "train",
    },
}


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


def segment_features(parent: str, frame: pd.DataFrame) -> dict[str, object]:
    """Return the raw-only parent features used by frozen stratification."""
    required = {"timestamp", "time_s", "load_total_kw"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"segment feature source is missing columns: {sorted(missing)}")
    load = pd.to_numeric(frame["load_total_kw"], errors="coerce").to_numpy(
        dtype=float
    )
    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
    time_s = pd.to_numeric(frame["time_s"], errors="coerce").to_numpy(dtype=float)
    if (
        len(load) < 2
        or not np.isfinite(load).all()
        or timestamps.isna().any()
        or not np.isfinite(time_s).all()
    ):
        raise ValueError("segment features require a finite time series")
    return {
        "parent": str(parent),
        "chronological_timestamp": timestamps.iloc[0],
        "month": int(timestamps.iloc[0].month),
        "duration_s": float(time_s[-1]),
        "mean_load_kw": float(np.mean(load)),
        "p95_load_kw": float(np.percentile(load, 95)),
    }


def _quartile_labels(
    result: pd.DataFrame,
    non_test_index: pd.Index,
    column: str,
) -> pd.Series:
    labels, bins = pd.qcut(
        result.loc[non_test_index, column],
        q=4,
        labels=False,
        duplicates="drop",
        retbins=True,
    )
    if set(pd.Series(labels).dropna().astype(int)) != {0, 1, 2, 3}:
        raise ValueError(f"{column} does not form four non-Test quartiles")
    output = pd.Series(pd.NA, index=result.index, dtype="Int64")
    output.loc[non_test_index] = pd.Series(
        labels.to_numpy(dtype=int),
        index=non_test_index,
        dtype="Int64",
    )
    test_index = result.index.difference(non_test_index)
    extended = np.asarray(bins, dtype=float).copy()
    extended[0] = -np.inf
    extended[-1] = np.inf
    output.loc[test_index] = pd.cut(
        result.loc[test_index, column],
        bins=extended,
        labels=False,
        include_lowest=True,
    ).astype("Int64")
    return output


def assign_parent_splits(features: pd.DataFrame) -> pd.DataFrame:
    """Assign the fixed Test parents and deterministic 38/10 Train/Validation."""
    required = {
        "parent",
        "chronological_timestamp",
        "month",
        "duration_s",
        "mean_load_kw",
        "p95_load_kw",
    }
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f"parent features are missing columns: {sorted(missing)}")
    result = features.loc[:, sorted(required)].copy()
    result["parent"] = result["parent"].astype(str)
    result["chronological_timestamp"] = pd.to_datetime(
        result["chronological_timestamp"], errors="coerce"
    )
    expected_count = ASSIGNMENT_TRAIN_COUNT + ASSIGNMENT_VALIDATION_COUNT + TEST_COUNT
    if len(result) != expected_count:
        raise ValueError("expected exactly 53 eligible parent feature rows")
    if result["parent"].duplicated().any():
        raise ValueError("parent features contain duplicate identifiers")
    if result["chronological_timestamp"].isna().any():
        raise ValueError("parent features contain invalid timestamps")
    if set(FIXED_TEST_PARENTS).difference(result["parent"]):
        raise ValueError("one or more fixed Test parents are missing")
    numeric_columns = ("month", "duration_s", "mean_load_kw", "p95_load_kw")
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
        if not np.isfinite(result[column].to_numpy(dtype=float)).all():
            raise ValueError(f"parent feature {column} must be finite")
    result = result.sort_values(
        ["chronological_timestamp", "parent"], kind="stable"
    ).reset_index(drop=True)
    result["chronological_rank"] = np.arange(len(result), dtype=int)
    is_test = result["parent"].isin(FIXED_TEST_PARENTS)
    if int(is_test.sum()) != TEST_COUNT:
        raise ValueError("fixed Test parent count is not five")
    non_test_index = result.index[~is_test]
    for source_column, output_column in (
        ("duration_s", "duration_quartile"),
        ("mean_load_kw", "mean_load_quartile"),
        ("p95_load_kw", "p95_load_quartile"),
    ):
        result[output_column] = _quartile_labels(
            result,
            non_test_index,
            source_column,
        )

    label_columns = (
        "month",
        "duration_quartile",
        "mean_load_quartile",
        "p95_load_quartile",
    )

    def labels_for(row: pd.Series) -> tuple[str, ...]:
        return tuple(
            f"{column}={int(row[column])}" for column in label_columns
        )

    label_map = {
        str(result.loc[index, "parent"]): labels_for(result.loc[index])
        for index in non_test_index
    }
    available_counts = Counter(
        label for labels in label_map.values() for label in labels
    )
    targets = {
        label: 0.2 * count for label, count in available_counts.items()
    }
    selected: list[str] = []
    selected_counts: Counter[str] = Counter()
    candidates = [
        str(parent) for parent in result.loc[non_test_index, "parent"].tolist()
    ]
    chronological_rank = result.set_index("parent")["chronological_rank"].to_dict()
    while len(selected) < ASSIGNMENT_VALIDATION_COUNT:
        scored: list[tuple[float, int, str]] = []
        for parent in candidates:
            trial = selected_counts.copy()
            trial.update(label_map[parent])
            score = float(
                sum(
                    abs(float(trial[label]) - target)
                    for label, target in targets.items()
                )
            )
            scored.append((score, int(chronological_rank[parent]), parent))
        _, _, chosen = min(scored)
        selected.append(chosen)
        selected_counts.update(label_map[chosen])
        candidates.remove(chosen)

    validation = set(selected)
    result["split"] = np.where(
        is_test,
        "test",
        np.where(result["parent"].isin(validation), "validation", "train"),
    )
    if result["split"].value_counts().to_dict() != {
        "train": ASSIGNMENT_TRAIN_COUNT,
        "validation": ASSIGNMENT_VALIDATION_COUNT,
        "test": TEST_COUNT,
    }:
        raise AssertionError("parent split counts violate the frozen contract")
    return result


def exclude_unexplained_negative_power_segments(
    assignment: pd.DataFrame,
    exclusions: dict[str, dict[str, str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove the frozen moving-negative-power segments without renumbering."""

    selected = (
        UNEXPLAINED_NEGATIVE_POWER_EXCLUSIONS
        if exclusions is None
        else exclusions
    )
    required = {"parent", "split", "chronological_rank"}
    missing = required.difference(assignment.columns)
    if missing:
        raise ValueError(
            f"split assignment is missing exclusion columns: {sorted(missing)}"
        )
    source = assignment.copy()
    source["sample_id"] = source["chronological_rank"].map(
        lambda value: f"zero_boundary_{int(value) + 1:03d}"
    )
    if source["sample_id"].duplicated().any():
        raise ValueError("split assignment produces duplicate sample IDs")
    indexed = source.set_index("sample_id", drop=False)
    rows: list[dict[str, str]] = []
    for sample_id in sorted(selected):
        expected = selected[sample_id]
        if sample_id not in indexed.index:
            raise ValueError(f"unexplained-power exclusion is missing: {sample_id}")
        actual = indexed.loc[sample_id]
        if str(actual["parent"]) != str(expected["parent"]):
            raise ValueError(f"{sample_id}: exclusion parent identity differs")
        if str(actual["split"]) != str(expected["split"]):
            raise ValueError(f"{sample_id}: exclusion split identity differs")
        if str(actual["split"]) == "test":
            raise ValueError("unexplained-power exclusions cannot remove Test")
        rows.append(
            {
                "parent": str(actual["parent"]),
                "sample_id": sample_id,
                "split": str(actual["split"]),
                "reason": "UNEXPLAINED_MOVING_NEGATIVE_TOTAL_POWER",
            }
        )
    retained = source.loc[~source["sample_id"].isin(selected)].drop(
        columns="sample_id"
    )
    audit = pd.DataFrame(
        rows,
        columns=("parent", "sample_id", "split", "reason"),
    )
    return retained.reset_index(drop=True), audit
