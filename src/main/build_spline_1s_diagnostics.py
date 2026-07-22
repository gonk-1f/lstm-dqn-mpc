"""Publish channel-reconstructed 1 s voyages to the retained formal path."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline


PROJ = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJ / "outputs" / "spline_1s_diagnostics"
DEFAULT_SPLIT_JSON = PROJ / "outputs" / "config" / "voyage_split_total_load_721.json"
FORMAL_SUBDIRECTORY = "natural_clipped_by_voyage"
FORMAL_DATASET_VERSION = "device_channel_natural_spline_1s"
CSV_FLOAT_FORMAT = "%.17g"
HORIZONS = (1, 6, 30, 60)


def _iso(value: Any) -> str:
    return pd.Timestamp(value).isoformat()


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
    if np.any(np.diff(seconds) <= 0):
        raise ValueError("Source timestamps must be strictly increasing after duplicate removal")
    return seconds


def _interpolate_optional_speed(
    frame: pd.DataFrame,
    seconds: np.ndarray,
    grid: np.ndarray,
    bc_type: str,
) -> np.ndarray | None:
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
    """Retained pure helper for legacy diagnostics; not used by the formal build."""

    frame = _source_frame(source)
    seconds = _strict_seconds(frame)
    end_second = int(round(float(seconds[-1])))
    grid = np.arange(0, end_second + 1, dtype=float)
    load = CubicSpline(
        seconds,
        frame["load_total_kw"].to_numpy(dtype=float),
        bc_type=bc_type,
    )(grid)
    timestamps = pd.Timestamp(frame["timestamp"].iloc[0]) + pd.to_timedelta(grid, unit="s")
    is_original = np.isin(np.round(grid, 6), np.round(seconds, 6))
    interval_idx = np.searchsorted(seconds, grid, side="right") - 1
    interval_idx = np.clip(interval_idx, 0, len(seconds) - 2)
    source_timestamps = pd.to_datetime(frame["timestamp"]).reset_index(drop=True)
    start_times: list[str] = []
    end_times: list[str] = []
    for grid_index, (idx, is_source) in enumerate(zip(interval_idx, is_original)):
        if bool(is_source):
            nearest = int(np.argmin(np.abs(seconds - grid[grid_index])))
            source_time = _iso(source_timestamps.iloc[nearest])
            start_times.append(source_time)
            end_times.append(source_time)
        else:
            start_times.append(_iso(source_timestamps.iloc[int(idx)]))
            end_times.append(_iso(source_timestamps.iloc[int(idx) + 1]))
    output = pd.DataFrame(
        {
            "dataset_version": dataset_version,
            "voyage_id": voyage_id,
            "split": split,
            "time_original_or_reconstructed": np.where(
                is_original,
                "original_30s_point",
                "reconstructed_1s_point",
            ),
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
        output["speed_knots"] = speed
    if "file_name" in frame.columns:
        output["file_name"] = str(frame["file_name"].iloc[0])
    return output


def _load_by_time(original_rows: pd.DataFrame) -> dict[str, float]:
    return {
        _iso(row["timestamp"]): float(row["load_total_kw"])
        for _, row in original_rows.iterrows()
    }


def compute_physical_check(spline_data: pd.DataFrame) -> pd.DataFrame:
    """Retained pure helper for existing diagnostics tests."""

    rows: list[dict[str, Any]] = []
    grouped = spline_data.groupby(["dataset_version", "voyage_id"], sort=True)
    for (dataset_version, voyage_id), group in grouped:
        group = group.copy()
        original = group[group["is_original_30s_point"].astype(bool)]
        reconstructed = group[~group["is_original_30s_point"].astype(bool)].copy()
        original_min = float(original["load_total_kw"].min())
        original_max = float(original["load_total_kw"].max())
        load_by_time = _load_by_time(original)
        local_above = 0
        local_below = 0
        if not reconstructed.empty:
            start_load = reconstructed["source_interval_start_time"].map(load_by_time)
            end_load = reconstructed["source_interval_end_time"].map(load_by_time)
            local_max = pd.concat([start_load, end_load], axis=1).max(axis=1)
            local_min = pd.concat([start_load, end_load], axis=1).min(axis=1)
            values = reconstructed["load_total_kw"].to_numpy(dtype=float)
            local_above = int((values > local_max.to_numpy(dtype=float) + 1e-9).sum())
            local_below = int((values < local_min.to_numpy(dtype=float) - 1e-9).sum())
        negative = int(group["load_total_kw"].lt(-1e-9).sum())
        original_negative = int(original["load_total_kw"].lt(-1e-9).sum())
        reconstructed_negative = int(reconstructed["load_total_kw"].lt(-1e-9).sum())
        global_overshoot = int(
            (
                group["load_total_kw"].gt(original_max + 1e-9)
                | group["load_total_kw"].lt(original_min - 1e-9)
            ).sum()
        )
        comments: list[str] = []
        if original_negative and reconstructed_negative:
            comments.append("negative load present in original and reconstructed rows")
        elif reconstructed_negative:
            comments.append("negative load introduced in reconstructed rows")
        elif original_negative:
            comments.append("negative load present in original rows")
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
                "load_min_spline_1s": float(group["load_total_kw"].min()),
                "load_max_spline_1s": float(group["load_total_kw"].max()),
                "negative_load_count": negative,
                "negative_load_ratio": negative / max(1, len(group)),
                "original_30s_negative_load_count": original_negative,
                "original_30s_negative_load_ratio": original_negative / max(1, len(original)),
                "reconstructed_1s_negative_load_count": reconstructed_negative,
                "reconstructed_1s_negative_load_ratio": reconstructed_negative
                / max(1, len(reconstructed)),
                "above_original_local_max_count": local_above,
                "above_original_local_max_ratio": local_above / max(1, len(reconstructed)),
                "below_original_local_min_count": local_below,
                "below_original_local_min_ratio": local_below / max(1, len(reconstructed)),
                "global_overshoot_count": global_overshoot,
                "global_overshoot_ratio": global_overshoot / max(1, len(group)),
                "comment": "; ".join(comments)
                if comments
                else "no negative load or overshoot detected",
            }
        )
    return pd.DataFrame(rows)


def _mae(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    return float(np.mean(np.abs(array))) if len(array) else float("nan")


def _current_hold_mae(groups: Iterable[pd.DataFrame], horizon: int) -> float:
    errors: list[float] = []
    for group in groups:
        values = group["load_total_kw"].to_numpy(dtype=float)
        if len(values) > horizon:
            errors.extend((values[:-horizon] - values[horizon:]).tolist())
    return _mae(errors)


def _last_slope_mae(groups: Iterable[pd.DataFrame], horizon: int) -> float:
    errors: list[float] = []
    for group in groups:
        values = group["load_total_kw"].to_numpy(dtype=float)
        if len(values) <= horizon + 1:
            continue
        idx = np.arange(1, len(values) - horizon)
        predicted = values[idx] + float(horizon) * (values[idx] - values[idx - 1])
        errors.extend((predicted - values[idx + horizon]).tolist())
    return _mae(errors)


def _moving_average_mae(
    groups: Iterable[pd.DataFrame],
    horizon: int,
    window: int,
) -> float:
    errors: list[float] = []
    for group in groups:
        values = group["load_total_kw"].to_numpy(dtype=float)
        if len(values) <= horizon + window:
            continue
        cumulative = np.concatenate([[0.0], np.cumsum(values)])
        idx = np.arange(window - 1, len(values) - horizon)
        predicted = (cumulative[idx + 1] - cumulative[idx + 1 - window]) / float(window)
        errors.extend((predicted - values[idx + horizon]).tolist())
    return _mae(errors)


def compute_predictability_audit(
    spline_data: pd.DataFrame,
    *,
    horizons: tuple[int, ...] = HORIZONS,
    moving_average_window_s: int = 60,
) -> pd.DataFrame:
    """Retained pure helper for existing diagnostics tests."""

    rows: list[dict[str, Any]] = []
    for dataset_version, version_frame in spline_data.groupby("dataset_version", sort=True):
        evaluation = version_frame[version_frame["split"].eq("test")].copy()
        scope = "test split"
        if evaluation.empty:
            evaluation = version_frame.copy()
            scope = "all available rows"
        evaluation = evaluation.sort_values(["voyage_id", "time_s"])
        groups = [group for _, group in evaluation.groupby("voyage_id", sort=True)]
        second_diffs = [
            np.diff(group["load_total_kw"].to_numpy(dtype=float), n=2)
            for group in groups
            if len(group) >= 3
        ]
        diff = np.concatenate(second_diffs) if second_diffs else np.array([], dtype=float)
        absolute_diff = np.abs(diff)
        row: dict[str, Any] = {
            "dataset_version": dataset_version,
            "online_feasible": bool(evaluation["online_feasible"].all())
            if "online_feasible" in evaluation
            else False,
            "uses_future_endpoint": bool(evaluation["uses_future_endpoint"].any())
            if "uses_future_endpoint" in evaluation
            else True,
            "second_diff_zero_ratio": float(np.mean(absolute_diff <= 1e-12))
            if len(absolute_diff)
            else math.nan,
            "mean_abs_second_diff": float(np.mean(absolute_diff))
            if len(absolute_diff)
            else math.nan,
            "median_abs_second_diff": float(np.median(absolute_diff))
            if len(absolute_diff)
            else math.nan,
            "p95_abs_second_diff": float(np.percentile(absolute_diff, 95))
            if len(absolute_diff)
            else math.nan,
            "p99_abs_second_diff": float(np.percentile(absolute_diff, 99))
            if len(absolute_diff)
            else math.nan,
        }
        for horizon in horizons:
            row[f"current_hold_h{horizon}_MAE"] = _current_hold_mae(groups, int(horizon))
            row[f"last_slope_h{horizon}_MAE"] = _last_slope_mae(groups, int(horizon))
            row[f"moving_average_h{horizon}_MAE"] = _moving_average_mae(
                groups,
                int(horizon),
                int(moving_average_window_s),
            )
            row[f"moving_average_60_h{horizon}_MAE"] = _moving_average_mae(
                groups, int(horizon), 60
            )
            row[f"moving_average_180_h{horizon}_MAE"] = _moving_average_mae(
                groups, int(horizon), 180
            )
        row["comment"] = (
            f"{scope}; baselines are causal, but labels are offline cubic-spline "
            "reconstructions using future nodes"
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_remove_directory(path: Path, root: Path) -> None:
    resolved_path = path.resolve()
    resolved_root = root.resolve()
    if resolved_path == resolved_root or not resolved_path.is_relative_to(resolved_root):
        raise ValueError(f"Refusing to remove directory outside output root: {resolved_path}")
    if path.exists():
        shutil.rmtree(path)


def _split_mapping(split_payload: dict[str, Any]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for split_name, key in (
        ("train", "train_voyages"),
        ("validation", "validation_voyages"),
        ("test", "test_voyages"),
    ):
        for voyage_id in split_payload.get(key, []):
            voyage_id = str(voyage_id)
            if voyage_id in mapping:
                raise ValueError(f"Split JSON repeats voyage {voyage_id}")
            mapping[voyage_id] = split_name
    return mapping


def publish_formal_voyages(
    *,
    staging_dir: Path,
    voyage_records: list[dict[str, Any]],
    split_json: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    retain_backup: bool = False,
) -> dict[str, Any]:
    """Finalize staged 1 s CSVs and atomically replace the formal directory.

    A coordinating caller may retain the previous directory until its other
    formal artifacts have committed, then remove or restore that backup.
    """

    staging_dir = Path(staging_dir)
    output_dir = Path(output_dir)
    data_dir = output_dir / "data"
    formal_dir = data_dir / FORMAL_SUBDIRECTORY
    if not staging_dir.is_dir():
        raise ValueError(f"Staging directory does not exist: {staging_dir}")
    if not staging_dir.resolve().is_relative_to(data_dir.resolve()):
        raise ValueError(f"Staging directory must be inside {data_dir}")

    split_json = Path(split_json)
    split_payload = json.loads(split_json.read_text(encoding="utf-8"))
    split_map = _split_mapping(split_payload)
    record_ids = [str(record["voyage_id"]) for record in voyage_records]
    expected_ids = set(record_ids)
    if len(record_ids) != len(expected_ids):
        raise ValueError("Voyage records contain duplicate voyage_id values")
    if set(split_map) != expected_ids:
        raise ValueError(
            f"Split coverage mismatch: split={sorted(split_map)}, staged={sorted(expected_ids)}"
        )
    split_sha256 = _sha256(split_json)
    manifest_rows: list[dict[str, Any]] = []

    for record in sorted(voyage_records, key=lambda item: str(item["voyage_id"])):
        voyage_id = str(record["voyage_id"])
        csv_path = staging_dir / str(record["output_name"])
        if not csv_path.is_file():
            raise ValueError(f"Missing staged voyage CSV: {csv_path}")
        frame = pd.read_csv(csv_path, encoding="utf-8-sig")
        if frame.empty:
            raise ValueError(f"Staged voyage CSV is empty: {csv_path}")
        frame["split"] = split_map[voyage_id]
        frame["dataset_version"] = FORMAL_DATASET_VERSION
        frame.to_csv(
            csv_path,
            index=False,
            encoding="utf-8-sig",
            float_format=CSV_FLOAT_FORMAT,
        )
        timestamp = pd.to_datetime(frame["timestamp"], errors="raise")
        if not timestamp.diff().dt.total_seconds().dropna().eq(1.0).all():
            raise ValueError(f"{voyage_id} is not a strict 1 s sequence")
        if timestamp.duplicated().any():
            raise ValueError(f"{voyage_id} contains duplicate 1 s timestamps")
        manifest_rows.append(
            {
                "voyage_id": voyage_id,
                "split": split_map[voyage_id],
                "source_file_name": record["source_file_name"],
                "output_csv": (formal_dir / csv_path.name).relative_to(PROJ).as_posix(),
                "rows": int(len(frame)),
                "start_timestamp": pd.Timestamp(timestamp.iloc[0]).isoformat(),
                "end_timestamp": pd.Timestamp(timestamp.iloc[-1]).isoformat(),
                "common_overlap_start": record["common_overlap_start"],
                "common_overlap_end": record["common_overlap_end"],
                "max_required_raw_gap_s": float(record["max_required_raw_gap_s"]),
                "max_required_raw_gap_channel": record["max_required_raw_gap_channel"],
                "min_total_load_kw_unclipped": float(frame["total_load_kw"].min()),
                "max_total_load_kw_unclipped": float(frame["total_load_kw"].max()),
                "negative_total_load_rows_clipped": int(frame["total_load_kw"].lt(0.0).sum()),
                "source_bundle_sha256": record["source_bundle_sha256"],
                "required_source_channel_hashes_json": record[
                    "required_source_channel_hashes_json"
                ],
                "split_json_sha256": split_sha256,
                "output_sha256": _sha256(csv_path),
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(
        staging_dir / "manifest.csv",
        index=False,
        encoding="utf-8-sig",
        float_format=CSV_FLOAT_FORMAT,
    )
    staged_csvs = {path.name for path in staging_dir.glob("*.csv") if path.name != "manifest.csv"}
    expected_csvs = {str(record["output_name"]) for record in voyage_records}
    if staged_csvs != expected_csvs:
        raise ValueError(
            f"Staging contains stale/missing CSVs: staged={sorted(staged_csvs)}, expected={sorted(expected_csvs)}"
        )

    backup_dir = data_dir / f".{FORMAL_SUBDIRECTORY}_backup"
    if backup_dir.exists() and not formal_dir.exists():
        backup_dir.replace(formal_dir)
    elif backup_dir.exists():
        _safe_remove_directory(backup_dir, data_dir)
    if formal_dir.exists():
        formal_dir.replace(backup_dir)
    try:
        staging_dir.replace(formal_dir)
    except Exception:
        if backup_dir.exists() and not formal_dir.exists():
            backup_dir.replace(formal_dir)
        raise
    if not retain_backup:
        _safe_remove_directory(backup_dir, data_dir)
    return {
        "formal_output_dir": str(formal_dir.resolve()),
        "manifest": str((formal_dir / "manifest.csv").resolve()),
        "num_voyages": len(voyage_records),
        "split_json_sha256": split_sha256,
        "dataset_version": FORMAL_DATASET_VERSION,
        "formal_backup_dir": str(backup_dir.resolve()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the retained device-channel 1 s total-load build entry."
    )
    parser.add_argument("--expected_count", type=int, default=66)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    main_dir = Path(__file__).resolve().parent
    if str(main_dir) not in sys.path:
        sys.path.insert(0, str(main_dir))
    from build_total_load_dataset_721 import build_dataset

    result = build_dataset(expected_count=int(args.expected_count))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
