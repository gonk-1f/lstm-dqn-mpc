"""Build and audit offline cubic-spline 1 s reconstructed load sequences.

The outputs from this module are offline reconstructed data, not measured 1 s
data. Interior samples depend on future 30 s nodes and must not be used as
online forecasting/control evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

PROJ = Path(__file__).resolve().parents[2]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_total_load_dataset_721 import _prepare_voyage_frame  # noqa: E402


DEFAULT_INPUT_DIR = PROJ / "total_load_excels"
DEFAULT_SPLIT_JSON = PROJ / "outputs" / "config" / "voyage_split_total_load_721.json"
DEFAULT_OUTPUT_DIR = PROJ / "outputs" / "spline_1s_diagnostics"
DATASET_VERSIONS = {
    "cubic_spline_1s_natural": "natural",
    "cubic_spline_1s_not_a_knot": "not-a-knot",
}
HORIZONS = (1, 6, 30, 60)


def _iso(value: Any) -> str:
    return pd.Timestamp(value).isoformat()


def _split_by_voyage(split_json: Path) -> dict[str, str]:
    payload = json.loads(Path(split_json).read_text(encoding="utf-8"))
    mapping: dict[str, str] = {}
    for split_name, key in (("train", "train_voyages"), ("validation", "validation_voyages"), ("test", "test_voyages")):
        for voyage_id in payload.get(key, payload.get(split_name, [])):
            mapping[str(voyage_id)] = split_name
    return mapping


def _source_frame(source: pd.DataFrame) -> pd.DataFrame:
    frame = source.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame["load_total_kw"] = pd.to_numeric(frame["load_total_kw"], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "load_total_kw"])
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    if len(frame) < 2:
        raise ValueError("At least two timestamped load samples are required for spline reconstruction")
    return frame


def _strict_seconds(frame: pd.DataFrame) -> np.ndarray:
    t0 = pd.Timestamp(frame["timestamp"].iloc[0])
    seconds = (pd.to_datetime(frame["timestamp"]) - t0).dt.total_seconds().to_numpy(dtype=float)
    diffs = np.diff(seconds)
    if np.any(diffs <= 0):
        raise ValueError("Source timestamps must be strictly increasing after duplicate removal")
    return seconds


def _interpolate_optional_speed(frame: pd.DataFrame, seconds: np.ndarray, grid: np.ndarray, bc_type: str) -> np.ndarray | None:
    if "speed_knots" not in frame.columns:
        return None
    speed = pd.to_numeric(frame["speed_knots"], errors="coerce").interpolate().ffill().bfill()
    if speed.isna().any():
        return None
    return CubicSpline(seconds, speed.to_numpy(dtype=float), bc_type=bc_type)(grid)


def reconstruct_voyage_spline(
    source: pd.DataFrame,
    *,
    voyage_id: str,
    split: str,
    dataset_version: str,
    bc_type: str,
) -> pd.DataFrame:
    """Reconstruct one voyage to a 1 s grid using a cubic spline."""
    frame = _source_frame(source)
    seconds = _strict_seconds(frame)
    end_second = int(round(float(seconds[-1])))
    grid = np.arange(0, end_second + 1, dtype=float)
    load = CubicSpline(seconds, frame["load_total_kw"].to_numpy(dtype=float), bc_type=bc_type)(grid)
    timestamps = pd.Timestamp(frame["timestamp"].iloc[0]) + pd.to_timedelta(grid, unit="s")

    rounded_source = np.round(seconds, 6)
    rounded_grid = np.round(grid, 6)
    is_original = np.isin(rounded_grid, rounded_source)
    interval_idx = np.searchsorted(seconds, grid, side="right") - 1
    interval_idx = np.clip(interval_idx, 0, len(seconds) - 2)

    source_timestamps = pd.to_datetime(frame["timestamp"]).reset_index(drop=True)
    start_times: list[str] = []
    end_times: list[str] = []
    for idx, is_source in zip(interval_idx, is_original):
        if bool(is_source):
            nearest = int(np.argmin(np.abs(seconds - grid[len(start_times)])))
            source_time = _iso(source_timestamps.iloc[nearest])
            start_times.append(source_time)
            end_times.append(source_time)
        else:
            start_times.append(_iso(source_timestamps.iloc[int(idx)]))
            end_times.append(_iso(source_timestamps.iloc[int(idx) + 1]))

    out = pd.DataFrame(
        {
            "dataset_version": dataset_version,
            "voyage_id": voyage_id,
            "split": split,
            "time_original_or_reconstructed": np.where(is_original, "original_30s_point", "reconstructed_1s_point"),
            "timestamp": timestamps,
            "time_s": grid,
            "load_total_kw": load,
            "source_interval_start_time": start_times,
            "source_interval_end_time": end_times,
            "is_original_30s_point": is_original.astype(bool),
            "online_feasible": False,
            "uses_future_endpoint": True,
        }
    )
    speed = _interpolate_optional_speed(frame, seconds, grid, bc_type)
    if speed is not None:
        out["speed_knots"] = speed
    if "file_name" in frame.columns:
        out["file_name"] = str(frame["file_name"].iloc[0])
    return out


def _load_by_time(original_rows: pd.DataFrame) -> dict[str, float]:
    return {
        _iso(row["timestamp"]): float(row["load_total_kw"])
        for _, row in original_rows.iterrows()
    }


def compute_physical_check(spline_data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (dataset_version, voyage_id), group in spline_data.groupby(["dataset_version", "voyage_id"], sort=True):
        group = group.copy()
        original = group[group["is_original_30s_point"].astype(bool)]
        original_min = float(original["load_total_kw"].min())
        original_max = float(original["load_total_kw"].max())
        spline_min = float(group["load_total_kw"].min())
        spline_max = float(group["load_total_kw"].max())
        load_by_time = _load_by_time(original)

        reconstructed = group[~group["is_original_30s_point"].astype(bool)].copy()
        local_above = 0
        local_below = 0
        if not reconstructed.empty:
            start_load = reconstructed["source_interval_start_time"].map(load_by_time)
            end_load = reconstructed["source_interval_end_time"].map(load_by_time)
            local_max = pd.concat([start_load, end_load], axis=1).max(axis=1)
            local_min = pd.concat([start_load, end_load], axis=1).min(axis=1)
            local_above = int((reconstructed["load_total_kw"].to_numpy(dtype=float) > local_max.to_numpy(dtype=float) + 1e-9).sum())
            local_below = int((reconstructed["load_total_kw"].to_numpy(dtype=float) < local_min.to_numpy(dtype=float) - 1e-9).sum())
        local_den = max(1, int(len(reconstructed)))
        negative = int((group["load_total_kw"] < -1e-9).sum())
        original_negative = int((original["load_total_kw"] < -1e-9).sum())
        reconstructed_negative = int((reconstructed["load_total_kw"] < -1e-9).sum())
        global_overshoot = int(((group["load_total_kw"] > original_max + 1e-9) | (group["load_total_kw"] < original_min - 1e-9)).sum())

        comments = []
        if original_negative and reconstructed_negative:
            comments.append("negative load present in original 30s nodes and reconstructed 1s interior; clipped version required for nonnegative-only diagnostics")
        elif reconstructed_negative:
            comments.append("negative load introduced in reconstructed 1s interior; clipped version required")
        elif original_negative:
            comments.append("negative load inherited from original 30s nodes")
        if local_above or local_below:
            comments.append("local endpoint overshoot present")
        if global_overshoot:
            comments.append("global original-range overshoot present")
        rows.append(
            {
                "dataset_version": dataset_version,
                "voyage_id": voyage_id,
                "load_min_original_30s": original_min,
                "load_max_original_30s": original_max,
                "load_min_spline_1s": spline_min,
                "load_max_spline_1s": spline_max,
                "negative_load_count": negative,
                "negative_load_ratio": negative / max(1, int(len(group))),
                "original_30s_negative_load_count": original_negative,
                "original_30s_negative_load_ratio": original_negative / max(1, int(len(original))),
                "reconstructed_1s_negative_load_count": reconstructed_negative,
                "reconstructed_1s_negative_load_ratio": reconstructed_negative / max(1, int(len(reconstructed))),
                "above_original_local_max_count": local_above,
                "above_original_local_max_ratio": local_above / local_den,
                "below_original_local_min_count": local_below,
                "below_original_local_min_ratio": local_below / local_den,
                "global_overshoot_count": global_overshoot,
                "global_overshoot_ratio": global_overshoot / max(1, int(len(group))),
                "comment": "; ".join(comments) if comments else "no negative load or overshoot detected",
            }
        )
    return pd.DataFrame(rows)


def _mae(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan")
    return float(np.mean(np.abs(arr)))


def _current_hold_mae(grouped: Iterable[pd.DataFrame], horizon: int) -> float:
    errors: list[float] = []
    for group in grouped:
        y = group["load_total_kw"].to_numpy(dtype=float)
        if len(y) <= horizon:
            continue
        errors.extend((y[:-horizon] - y[horizon:]).tolist())
    return _mae(errors)


def _last_slope_mae(grouped: Iterable[pd.DataFrame], horizon: int) -> float:
    errors: list[float] = []
    for group in grouped:
        y = group["load_total_kw"].to_numpy(dtype=float)
        if len(y) <= horizon + 1:
            continue
        idx = np.arange(1, len(y) - horizon)
        pred = y[idx] + float(horizon) * (y[idx] - y[idx - 1])
        actual = y[idx + horizon]
        errors.extend((pred - actual).tolist())
    return _mae(errors)


def _moving_average_mae(grouped: Iterable[pd.DataFrame], horizon: int, window: int) -> float:
    errors: list[float] = []
    for group in grouped:
        y = group["load_total_kw"].to_numpy(dtype=float)
        if len(y) <= horizon + window:
            continue
        cumsum = np.concatenate([[0.0], np.cumsum(y)])
        idx = np.arange(window - 1, len(y) - horizon)
        pred = (cumsum[idx + 1] - cumsum[idx + 1 - window]) / float(window)
        actual = y[idx + horizon]
        errors.extend((pred - actual).tolist())
    return _mae(errors)


def compute_predictability_audit(
    spline_data: pd.DataFrame,
    *,
    horizons: tuple[int, ...] = HORIZONS,
    moving_average_window_s: int = 60,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset_version, version_frame in spline_data.groupby("dataset_version", sort=True):
        eval_frame = version_frame[version_frame["split"].eq("test")].copy()
        eval_scope = "test split"
        if eval_frame.empty:
            eval_frame = version_frame.copy()
            eval_scope = "all available rows"
        eval_frame = eval_frame.sort_values(["voyage_id", "time_s"]).reset_index(drop=True)
        voyage_groups = [g.sort_values("time_s") for _, g in eval_frame.groupby("voyage_id", sort=True)]

        second_diffs: list[np.ndarray] = []
        for group in voyage_groups:
            y = group["load_total_kw"].to_numpy(dtype=float)
            if len(y) >= 3:
                second_diffs.append(np.diff(y, n=2))
        diff = np.concatenate(second_diffs) if second_diffs else np.array([], dtype=float)
        abs_diff = np.abs(diff)
        row: dict[str, Any] = {
            "dataset_version": dataset_version,
            "online_feasible": bool(eval_frame["online_feasible"].all()) if "online_feasible" in eval_frame.columns else False,
            "uses_future_endpoint": bool(eval_frame["uses_future_endpoint"].any()) if "uses_future_endpoint" in eval_frame.columns else True,
            "second_diff_zero_ratio": float(np.mean(abs_diff <= 1e-12)) if len(abs_diff) else math.nan,
            "mean_abs_second_diff": float(np.mean(abs_diff)) if len(abs_diff) else math.nan,
            "median_abs_second_diff": float(np.median(abs_diff)) if len(abs_diff) else math.nan,
            "p95_abs_second_diff": float(np.percentile(abs_diff, 95)) if len(abs_diff) else math.nan,
            "p99_abs_second_diff": float(np.percentile(abs_diff, 99)) if len(abs_diff) else math.nan,
        }
        for horizon in horizons:
            row[f"current_hold_h{horizon}_MAE"] = _current_hold_mae(voyage_groups, int(horizon))
            row[f"last_slope_h{horizon}_MAE"] = _last_slope_mae(voyage_groups, int(horizon))
            row[f"moving_average_h{horizon}_MAE"] = _moving_average_mae(
                voyage_groups, int(horizon), int(moving_average_window_s)
            )
            row[f"moving_average_60_h{horizon}_MAE"] = _moving_average_mae(voyage_groups, int(horizon), 60)
            row[f"moving_average_180_h{horizon}_MAE"] = _moving_average_mae(voyage_groups, int(horizon), 180)
        row["comment"] = (
            f"{eval_scope}; baselines are causal, but labels are offline cubic-spline reconstructions using future nodes"
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _read_original_voyages(input_dir: Path, expected_count: int | None) -> list[tuple[str, pd.DataFrame]]:
    excel_paths = sorted(Path(input_dir).glob("*.xlsx"))
    if expected_count is not None and len(excel_paths) != int(expected_count):
        raise ValueError(f"Expected {expected_count} Excel files in {input_dir}, found {len(excel_paths)}")
    records = [_prepare_voyage_frame(path, include_existing_speed=True) for path in excel_paths]
    records.sort(key=lambda item: (item.start_time, item.path.name))
    return [(f"voyage_{idx:03d}", record.frame) for idx, record in enumerate(records, start=1)]


def build_spline_diagnostics(
    *,
    input_dir: Path = DEFAULT_INPUT_DIR,
    split_json: Path = DEFAULT_SPLIT_JSON,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    expected_count: int | None = 66,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    split_map = _split_by_voyage(Path(split_json))
    voyages = _read_original_voyages(Path(input_dir), expected_count)

    all_versions: list[pd.DataFrame] = []
    written_data: list[str] = []
    clipped_written: list[str] = []
    for dataset_version, bc_type in DATASET_VERSIONS.items():
        frames = [
            reconstruct_voyage_spline(
                frame,
                voyage_id=voyage_id,
                split=split_map.get(voyage_id, "unknown"),
                dataset_version=dataset_version,
                bc_type=bc_type,
            )
            for voyage_id, frame in voyages
        ]
        version_frame = pd.concat(frames, ignore_index=True)
        csv_path = data_dir / f"{dataset_version}.csv"
        version_frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
        written_data.append(str(csv_path.resolve()))
        all_versions.append(version_frame)
        if int((version_frame["load_total_kw"] < -1e-9).sum()) > 0:
            clipped = version_frame.copy()
            clipped["dataset_version"] = f"{dataset_version}_clipped"
            clipped["load_total_kw"] = clipped["load_total_kw"].clip(lower=0.0)
            clipped_path = data_dir / f"{dataset_version}_clipped.csv"
            clipped.to_csv(clipped_path, index=False, encoding="utf-8-sig")
            clipped_written.append(str(clipped_path.resolve()))
            all_versions.append(clipped)

    combined = pd.concat(all_versions, ignore_index=True)
    physical = compute_physical_check(combined)
    predictability = compute_predictability_audit(combined)
    physical_path = output_dir / "spline_physical_check.csv"
    predictability_path = output_dir / "spline_predictability_audit.csv"
    physical.to_csv(physical_path, index=False, encoding="utf-8-sig")
    predictability.to_csv(predictability_path, index=False, encoding="utf-8-sig")

    summary = {
        "input_dir": str(Path(input_dir).resolve()),
        "split_json": str(Path(split_json).resolve()),
        "output_dir": str(output_dir.resolve()),
        "dataset_csv": written_data,
        "clipped_dataset_csv": clipped_written,
        "physical_check_csv": str(physical_path.resolve()),
        "predictability_audit_csv": str(predictability_path.resolve()),
        "num_voyages": len(voyages),
        "online_feasible": False,
        "uses_future_endpoint": True,
        "note": "cubic spline 1 s outputs are offline reconstructions, not measured 1 s data",
    }
    (output_dir / "spline_build_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and audit offline cubic-spline 1 s reconstructed load data.")
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--split_json", type=Path, default=DEFAULT_SPLIT_JSON)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--expected_count", type=int, default=66)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = build_spline_diagnostics(
        input_dir=args.input_dir,
        split_json=args.split_json,
        output_dir=args.output_dir,
        expected_count=args.expected_count,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
