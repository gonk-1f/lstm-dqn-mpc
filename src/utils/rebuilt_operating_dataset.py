"""Time-faithful helpers for the rebuilt operating-segment dataset."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator


FLOAT_NEGATIVE_EPSILON_KW = 1.0e-9


def _numeric(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.mask(values.eq(-9999.0))


def _collapse(
    frame: pd.DataFrame,
    *,
    value_columns: Iterable[str],
    power: pd.Series,
) -> tuple[pd.DataFrame, dict[str, int]]:
    required = {"timestamp", *value_columns}
    if not required.issubset(frame.columns):
        raise ValueError(f"channel is missing {sorted(required.difference(frame.columns))}")
    source = frame.copy()
    source["timestamp"] = pd.to_datetime(source["timestamp"], errors="coerce")
    source = source.loc[source["timestamp"].notna()].copy()
    source["power_kw"] = power.loc[source.index].to_numpy(dtype=float)
    source = source.sort_values("timestamp", kind="stable").reset_index(drop=True)
    for column in value_columns:
        source[column] = _numeric(source[column])
    retained = source.drop_duplicates(subset=["timestamp", *value_columns], keep="first")
    exact_removed = int(len(source) - len(retained))
    group_sizes = retained.groupby("timestamp", sort=True).size()
    aggregated = int((group_sizes - 1).clip(lower=0).sum())
    result = retained.groupby("timestamp", as_index=False, sort=True)["power_kw"].mean()
    return result, {
        "input_rows": int(len(source)),
        "output_rows": int(len(result)),
        "exact_duplicate_records_removed": exact_removed,
        "multi_value_timestamp_rows_aggregated": aggregated,
    }


def collapse_battery_records(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Collapse a BMS channel after calculating each raw-record power."""
    voltage = _numeric(frame["voltage_v"])
    current = _numeric(frame["current_a"])
    return _collapse(
        frame,
        value_columns=("voltage_v", "current_a"),
        power=-(voltage * current) / 1000.0,
    )


def collapse_fc_records(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Collapse duplicate FC telemetry at its original timestamp."""
    power = _numeric(frame["power_kw"])
    return _collapse(frame, value_columns=("power_kw",), power=power)


def collapse_scalar_records(
    frame: pd.DataFrame,
    *,
    value_column: str,
    output_column: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Collapse duplicate scalar telemetry without inventing a time axis."""
    if "timestamp" not in frame or value_column not in frame:
        raise ValueError(f"scalar channel requires timestamp and {value_column}")
    source = frame[["timestamp", value_column]].copy()
    source["timestamp"] = pd.to_datetime(source["timestamp"], errors="coerce")
    source[value_column] = _numeric(source[value_column])
    source = source.dropna(subset=["timestamp"]).sort_values("timestamp", kind="stable")
    retained = source.drop_duplicates(subset=["timestamp", value_column], keep="first")
    exact_removed = int(len(source) - len(retained))
    group_sizes = retained.groupby("timestamp", sort=True).size()
    aggregated = int((group_sizes - 1).clip(lower=0).sum())
    result = retained.groupby("timestamp", as_index=False, sort=True)[value_column].mean().rename(columns={value_column: output_column})
    return result, {
        "input_rows": int(len(source)),
        "output_rows": int(len(result)),
        "exact_duplicate_records_removed": exact_removed,
        "multi_value_timestamp_rows_aggregated": aggregated,
    }


def match_nearest_without_reuse(
    reference: pd.DataFrame,
    source: pd.DataFrame,
    *,
    tolerance_s: float,
) -> pd.DataFrame:
    """One-to-one nearest timestamp matching; a missing slot remains missing."""
    if tolerance_s <= 0.0:
        raise ValueError("tolerance_s must be positive")
    ref = reference[["timestamp"]].copy()
    src = source.copy()
    ref["timestamp"] = pd.to_datetime(ref["timestamp"], errors="coerce")
    src["timestamp"] = pd.to_datetime(src["timestamp"], errors="coerce")
    if ref["timestamp"].isna().any() or src["timestamp"].isna().any():
        raise ValueError("timestamp matching requires valid timestamps")
    if ref["timestamp"].duplicated().any() or src["timestamp"].duplicated().any():
        raise ValueError("timestamp matching requires collapsed unique timestamps")
    ref = ref.sort_values("timestamp", kind="stable").reset_index(drop=True)
    src = src.sort_values("timestamp", kind="stable").reset_index(drop=True)
    columns = [column for column in src.columns if column != "timestamp"]
    output = ref.copy()
    matched_values = {
        column: np.full(len(ref), np.nan, dtype=object)
        for column in columns
    }
    source_values = {
        column: src[column].to_numpy(copy=False)
        for column in columns
    }
    source_ns = src["timestamp"].map(lambda value: pd.Timestamp(value).value).to_numpy(dtype=np.int64)
    used = np.zeros(len(src), dtype=bool)
    tolerance_ns = int(tolerance_s * 1.0e9)
    for row, timestamp in enumerate(ref["timestamp"].map(lambda value: pd.Timestamp(value).value).to_numpy(dtype=np.int64)):
        insertion = int(np.searchsorted(source_ns, timestamp, side="left"))
        candidates = [index for index in (insertion - 1, insertion) if 0 <= index < len(src) and not used[index]]
        if not candidates:
            continue
        best = min(candidates, key=lambda index: (abs(int(source_ns[index]) - int(timestamp)), index))
        if abs(int(source_ns[best]) - int(timestamp)) <= tolerance_ns:
            for column in columns:
                matched_values[column][row] = source_values[column][best]
            used[best] = True
    for column in columns:
        output[column] = matched_values[column]
    return output


def find_contiguous_intervals(frame: pd.DataFrame, *, long_gap_threshold_s: float) -> pd.Series:
    """Label chronological intervals without treating normal cadence jitter as a break."""
    if long_gap_threshold_s <= 0.0:
        raise ValueError("long_gap_threshold_s must be positive")
    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
    if timestamps.isna().any():
        raise ValueError("interval construction requires valid timestamps")
    return timestamps.diff().dt.total_seconds().gt(long_gap_threshold_s).cumsum().astype(int)


def battery_state(values: pd.Series, *, deadband_kw: float) -> pd.Series:
    """Classify measured battery power without changing its measured value."""
    if deadband_kw < 0.0:
        raise ValueError("battery deadband must be non-negative")
    numeric = _numeric(values)
    return pd.Series(
        np.select(
            [numeric.lt(-deadband_kw), numeric.gt(deadband_kw)],
            ["charging", "discharging"],
            default="neutral",
        ),
        index=values.index,
        dtype="object",
    )


def align_ais_to_power(
    power: pd.DataFrame,
    ais: pd.DataFrame,
    *,
    max_normal_gap_s: float,
    max_nearest_s: float,
) -> pd.DataFrame:
    """Align sparse AIS to power timestamps without crossing an AIS long gap."""
    if max_normal_gap_s <= 0.0 or max_nearest_s <= 0.0:
        raise ValueError("AIS tolerances must be positive")
    targets = pd.DataFrame({"timestamp": pd.to_datetime(power["timestamp"], errors="coerce")})
    if targets["timestamp"].isna().any():
        raise ValueError("power timestamps must be valid")
    source = ais[["timestamp", "ais_speed_kn"]].copy()
    source["timestamp"] = pd.to_datetime(source["timestamp"], errors="coerce")
    source["ais_speed_kn"] = _numeric(source["ais_speed_kn"])
    source = source.dropna(subset=["timestamp", "ais_speed_kn"]).sort_values("timestamp", kind="stable")
    source = source.drop_duplicates("timestamp", keep="first").reset_index(drop=True)
    output = targets.copy()
    output["speed_aligned_kn"] = np.nan
    output["speed_source"] = "unavailable"
    if source.empty:
        return output

    source_ns = source["timestamp"].map(lambda value: pd.Timestamp(value).value).to_numpy(dtype=np.int64)
    speed = source["ais_speed_kn"].to_numpy(dtype=float)
    for index, timestamp in enumerate(targets["timestamp"].map(lambda value: pd.Timestamp(value).value).to_numpy(dtype=np.int64)):
        position = int(np.searchsorted(source_ns, timestamp, side="left"))
        if position < len(source_ns) and source_ns[position] == timestamp:
            output.loc[index, ["speed_aligned_kn", "speed_source"]] = [speed[position], "exact"]
            continue
        left, right = position - 1, position
        if left >= 0 and right < len(source_ns):
            gap_s = (source_ns[right] - source_ns[left]) / 1.0e9
            if gap_s <= max_normal_gap_s:
                fraction = (timestamp - source_ns[left]) / (source_ns[right] - source_ns[left])
                output.loc[index, ["speed_aligned_kn", "speed_source"]] = [
                    speed[left] + fraction * (speed[right] - speed[left]), "linear"
                ]
                continue
        choices = [candidate for candidate in (left, right) if 0 <= candidate < len(source_ns)]
        if choices:
            best = min(choices, key=lambda candidate: (abs(source_ns[candidate] - timestamp), candidate))
            if abs(source_ns[best] - timestamp) <= int(max_nearest_s * 1.0e9):
                output.loc[index, ["speed_aligned_kn", "speed_source"]] = [speed[best], "nearest"]
    return output


def select_shore_intervals(
    frame: pd.DataFrame,
    *,
    fc_idle_threshold_kw: float,
    speed_idle_threshold_kn: float,
    battery_charge_threshold_kw: float,
    minimum_shore_points: int,
    long_gap_threshold_s: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mark sustained stationary battery charging as external charging evidence."""
    if minimum_shore_points < 2:
        raise ValueError("minimum_shore_points must be at least two")
    result = frame.sort_values("timestamp", kind="stable").reset_index(drop=True).copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="coerce")
    candidate = (
        result["aligned"].astype(bool)
        & pd.to_numeric(result["battery_total_kw"], errors="coerce").lt(-battery_charge_threshold_kw)
    )
    interval = find_contiguous_intervals(result, long_gap_threshold_s=long_gap_threshold_s)
    start = candidate.ne(candidate.shift(fill_value=False)) | interval.ne(interval.shift(fill_value=interval.iloc[0]))
    candidate_run = start.cumsum()
    result["is_shore"] = False
    interval_rows: list[dict[str, object]] = []
    for _, run in result.loc[candidate].groupby(candidate_run[candidate], sort=False):
        known_speed = pd.to_numeric(run["speed_aligned_kn"], errors="coerce").dropna()
        stationary = not known_speed.empty and bool(known_speed.le(speed_idle_threshold_kn).all())
        if len(run) < minimum_shore_points or not stationary:
            continue
        result.loc[run.index, "is_shore"] = True
        interval_rows.append({
            "start_time": run["timestamp"].iloc[0].isoformat(),
            "end_time": run["timestamp"].iloc[-1].isoformat(),
            "raw_points": int(len(run)),
            "duration_s": float((run["timestamp"].iloc[-1] - run["timestamp"].iloc[0]).total_seconds()),
            "mean_fc_kw": float(run["fc_total_kw"].mean()),
            "mean_battery_kw": float(run["battery_total_kw"].mean()),
            "speed_min_kn": float(known_speed.min()),
            "speed_max_kn": float(known_speed.max()),
            "speed_unavailable_points": int(run["speed_aligned_kn"].isna().sum()),
        })
    return result, pd.DataFrame(interval_rows)


def chronological_parent_splits(parents: list[str]) -> dict[str, list[str]]:
    """Allocate chronological parent voyages approximately 70/20/10 without leakage."""
    total = len(parents)
    train_count = int(np.floor(total * 0.70))
    validation_count = int(np.floor(total * 0.20))
    return {
        "train": parents[:train_count],
        "validation": parents[train_count:train_count + validation_count],
        "test": parents[train_count + validation_count:],
    }


def pchip_to_one_second(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int | float]]:
    """Independently reconstruct one segment from its actual raw timestamps."""
    source = frame[["timestamp", "load_total_kw"]].copy().sort_values("timestamp", kind="stable")
    source["timestamp"] = pd.to_datetime(source["timestamp"], errors="coerce")
    source["load_total_kw"] = _numeric(source["load_total_kw"])
    source = source.dropna()
    if len(source) < 2:
        raise ValueError("PCHIP requires at least two finite source points")
    if source["timestamp"].duplicated().any():
        raise ValueError("PCHIP source timestamps must be unique")
    origin = source["timestamp"].iloc[0]
    x = (source["timestamp"] - origin).dt.total_seconds().to_numpy(dtype=float)
    grid = pd.date_range(source["timestamp"].iloc[0], source["timestamp"].iloc[-1], freq="1s")
    x_grid = (grid - origin).total_seconds().to_numpy(dtype=float)
    values = PchipInterpolator(x, source["load_total_kw"].to_numpy(dtype=float))(x_grid)
    raw_min = float(source["load_total_kw"].min())
    generated_negative = (values < -FLOAT_NEGATIVE_EPSILON_KW) & (raw_min >= -FLOAT_NEGATIVE_EPSILON_KW)
    if generated_negative.any():
        raise ValueError("PCHIP introduced substantive negative load from nonnegative source")
    tiny_negative = (values < 0.0) & (values >= -FLOAT_NEGATIVE_EPSILON_KW)
    values[tiny_negative] = 0.0
    return pd.DataFrame({"timestamp": grid, "time_s": np.arange(len(grid), dtype=float), "load_total_kw": values}), {
        "source_points": int(len(source)),
        "output_points": int(len(grid)),
        "raw_min_kw": raw_min,
        "pchip_min_kw": float(values.min()),
        "floating_negative_zeroed": int(tiny_negative.sum()),
    }
