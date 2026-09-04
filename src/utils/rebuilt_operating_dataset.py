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
    result: list[dict[str, float | pd.Timestamp]] = []
    exact_removed = 0
    aggregated = 0
    for timestamp, group in source.groupby("timestamp", sort=True):
        values = group[list(value_columns)].apply(_numeric)
        same_records = values.nunique(dropna=False).eq(1).all()
        if same_records:
            exact_removed += len(group) - 1
        else:
            aggregated += len(group) - 1
        result.append({"timestamp": timestamp, "power_kw": float(group["power_kw"].mean())})
    return pd.DataFrame(result), {
        "input_rows": int(len(source)),
        "output_rows": int(len(result)),
        "exact_duplicate_records_removed": int(exact_removed),
        "multi_value_timestamp_rows_aggregated": int(aggregated),
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
    for column in columns:
        output[column] = np.nan
    source_ns = src["timestamp"].astype("int64").to_numpy()
    used = np.zeros(len(src), dtype=bool)
    tolerance_ns = int(tolerance_s * 1.0e9)
    for row, timestamp in enumerate(ref["timestamp"].astype("int64").to_numpy()):
        insertion = int(np.searchsorted(source_ns, timestamp, side="left"))
        candidates = [index for index in (insertion - 1, insertion) if 0 <= index < len(src) and not used[index]]
        if not candidates:
            continue
        best = min(candidates, key=lambda index: (abs(int(source_ns[index]) - int(timestamp)), index))
        if abs(int(source_ns[best]) - int(timestamp)) <= tolerance_ns:
            output.loc[row, columns] = src.loc[best, columns].to_numpy()
            used[best] = True
    return output


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
