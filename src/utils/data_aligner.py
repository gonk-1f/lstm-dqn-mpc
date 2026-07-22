"""Reconstruct raw device-power channels on a shared per-voyage 1 s grid."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

from utils.data_loader import VoyageFiles


TIME_COLUMN = "Time"
VOLTAGE_COLUMN = "总电压(V)"
CURRENT_COLUMN = "总电流(A)"
SOC_COLUMN = "SOC(%)"
FUEL_CELL_POWER_COLUMN = "发电功率(kW)"
INVERTER_POWER_COLUMN = "输出有功功率(kW)"
DATASET_VERSION = "device_channel_natural_spline_1s"
MIN_SPLINE_POINTS = 2
VOYAGE_054_TIMESTAMP_ERRATUM_DIR = "7\u67089\u65e508_00_7\u67089\u65e516_00"
VOYAGE_054_TIMESTAMP_ERRATUM_CHANNELS = {
    *(
        f"battery_{side}_cluster_{number}"
        for side in ("left", "right")
        for number in range(1, 7)
    ),
    *(
        f"fuel_cell_{side}_{number}"
        for side in ("left", "right")
        for number in range(1, 5)
    ),
}


class VoyageAlignmentError(ValueError):
    """Raised when the required raw channels cannot define one voyage."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _single_channel_file(
    directory: Path,
    prefix: str,
    *,
    required: bool,
) -> tuple[Path | None, dict[str, Any]]:
    matches = sorted(path for path in directory.glob("*.csv") if path.name.startswith(f"{prefix}_"))
    if len(matches) == 1:
        path = matches[0]
        return path, {
            "matching_file_count": 1,
            "matching_file_names": [path.name],
            "matching_file_sha256": [_sha256(path)],
            "identical_duplicate_files_ignored": 0,
        }
    if not matches and not required:
        return None, {
            "matching_file_count": 0,
            "matching_file_names": [],
            "matching_file_sha256": [],
            "identical_duplicate_files_ignored": 0,
        }
    if not matches:
        raise VoyageAlignmentError(
            "missing_required_channel",
            f"missing required raw channel {prefix!r} under {directory}",
        )
    hashes = [_sha256(path) for path in matches]
    if len(set(hashes)) == 1:
        return matches[0], {
            "matching_file_count": len(matches),
            "matching_file_names": [path.name for path in matches],
            "matching_file_sha256": hashes,
            "identical_duplicate_files_ignored": len(matches) - 1,
        }
    raise VoyageAlignmentError(
        "duplicate_logical_channel",
        f"logical raw channel {prefix!r} has {len(matches)} non-identical files: "
        + "; ".join(f"{path.name} ({file_hash})" for path, file_hash in zip(matches, hashes)),
    )


def _apply_known_timestamp_errata(
    parsed: pd.DataFrame,
    path: Path,
    logical_name: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Apply the single audited voyage_054 timestamp correction."""

    if (
        path.parent.parent.name != VOYAGE_054_TIMESTAMP_ERRATUM_DIR
        or logical_name not in VOYAGE_054_TIMESTAMP_ERRATUM_CHANNELS
    ):
        return parsed, []

    duplicate_timestamp = pd.Timestamp("2024-07-09 09:57:22")
    duplicate_positions = np.flatnonzero(
        parsed["timestamp"].eq(duplicate_timestamp).to_numpy()
    )
    is_right_fuel_cell = logical_name.startswith("fuel_cell_right_")
    expected_previous = pd.Timestamp(
        "2024-07-09 09:56:18" if is_right_fuel_cell else "2024-07-09 09:56:19"
    )
    expected_next = pd.Timestamp(
        "2024-07-09 09:57:48" if is_right_fuel_cell else "2024-07-09 09:57:49"
    )
    pattern_matches = (
        len(duplicate_positions) == 2
        and int(duplicate_positions[1]) == int(duplicate_positions[0]) + 1
        and int(duplicate_positions[0]) > 0
        and int(duplicate_positions[1]) + 1 < len(parsed)
    )
    if pattern_matches:
        first_position = int(duplicate_positions[0])
        previous_timestamp = pd.Timestamp(parsed["timestamp"].iloc[first_position - 1])
        next_timestamp = pd.Timestamp(parsed["timestamp"].iloc[first_position + 2])
        pattern_matches = (
            previous_timestamp == expected_previous and next_timestamp == expected_next
        )
    if not pattern_matches:
        raise VoyageAlignmentError(
            "timestamp_errata_mismatch",
            f"{path.name} ({logical_name}) no longer matches the audited voyage_054 "
            "timestamp erratum precondition",
        )

    corrected_timestamp = expected_previous + pd.Timedelta(seconds=30)
    corrected = parsed.copy()
    corrected.iloc[
        first_position,
        corrected.columns.get_loc("timestamp"),
    ] = corrected_timestamp
    correction = {
        "raw_row_position_zero_based": first_position,
        "csv_line_number": first_position + 2,
        "original_timestamp": duplicate_timestamp.isoformat(),
        "corrected_timestamp": corrected_timestamp.isoformat(),
        "reason": (
            "deterministic inference from adjacent 30 s cadence and synchronized "
            "20-channel row order"
        ),
    }
    return corrected, [correction]


def _read_numeric_channel(
    path: Path,
    value_columns: list[str],
    logical_name: str,
    file_selection_metadata: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    header = pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns.tolist()
    required_columns = [TIME_COLUMN, *value_columns]
    missing_columns = [column for column in required_columns if column not in header]
    if missing_columns:
        raise VoyageAlignmentError(
            "missing_required_field",
            f"{path.name} ({logical_name}) missing fields: {missing_columns}",
        )
    raw = pd.read_csv(path, usecols=required_columns, encoding="utf-8-sig")
    parsed = pd.DataFrame({"timestamp": pd.to_datetime(raw[TIME_COLUMN], errors="coerce")})
    for column in value_columns:
        parsed[column] = pd.to_numeric(raw[column], errors="coerce")
    parsed, timestamp_corrections = _apply_known_timestamp_errata(
        parsed,
        path,
        logical_name,
    )
    invalid_mask = parsed[required_columns[1:]].isna().any(axis=1) | parsed["timestamp"].isna()
    invalid_row_count = int(invalid_mask.sum())
    valid = parsed.loc[~invalid_mask].copy()
    duplicate_mask = valid["timestamp"].duplicated(keep=False)
    duplicate_rows_total = int(duplicate_mask.sum())
    duplicate_timestamp_count = int(valid.loc[duplicate_mask, "timestamp"].nunique())
    exact_duplicate_rows_dropped = 0
    conflicting_duplicate_timestamp_count = 0
    conflicting_duplicate_timestamp_range = ""
    if duplicate_rows_total:
        duplicate_groups = valid.loc[duplicate_mask].groupby("timestamp", sort=False)
        conflict_by_timestamp = duplicate_groups[value_columns].nunique(dropna=False).gt(1).any(axis=1)
        conflicting_duplicate_timestamp_count = int(conflict_by_timestamp.sum())
        if conflicting_duplicate_timestamp_count:
            conflict_times = conflict_by_timestamp.index[conflict_by_timestamp]
            conflicting_duplicate_timestamp_range = (
                f"{pd.Timestamp(conflict_times.min()).isoformat()}.."
                f"{pd.Timestamp(conflict_times.max()).isoformat()}"
            )
        group_sizes = duplicate_groups.size()
        exact_duplicate_rows_dropped = int(
            (group_sizes.loc[~conflict_by_timestamp] - 1).clip(lower=0).sum()
        )
        before = len(valid)
        valid = valid.drop_duplicates(subset=["timestamp"], keep="first")
        duplicate_rows_removed_for_gap_audit = int(before - len(valid))
    else:
        duplicate_rows_removed_for_gap_audit = 0
    was_monotonic = bool(valid["timestamp"].is_monotonic_increasing)
    valid = valid.sort_values("timestamp").reset_index(drop=True)
    if len(valid) < MIN_SPLINE_POINTS:
        raise VoyageAlignmentError(
            "insufficient_spline_points",
            f"{path.name} ({logical_name}) has {len(valid)} valid points; need at least {MIN_SPLINE_POINTS}",
        )
    gaps = valid["timestamp"].diff().dt.total_seconds().dropna()
    metadata = {
        "logical_name": logical_name,
        "file_name": path.name,
        "valid_point_count": int(len(valid)),
        "invalid_row_count": invalid_row_count,
        "duplicate_timestamp_rows_total": duplicate_rows_total,
        "duplicate_timestamp_count": duplicate_timestamp_count,
        "exact_duplicate_rows_dropped": exact_duplicate_rows_dropped,
        "duplicate_rows_removed_for_gap_audit": duplicate_rows_removed_for_gap_audit,
        "conflicting_duplicate_timestamp_count": conflicting_duplicate_timestamp_count,
        "conflicting_duplicate_timestamp_range": conflicting_duplicate_timestamp_range,
        "timestamp_correction_count": len(timestamp_corrections),
        "timestamp_corrections": timestamp_corrections,
        "raw_timestamp_was_monotonic": was_monotonic,
        "start_timestamp": pd.Timestamp(valid["timestamp"].iloc[0]).isoformat(),
        "end_timestamp": pd.Timestamp(valid["timestamp"].iloc[-1]).isoformat(),
        "max_raw_gap_s": float(gaps.max()) if len(gaps) else 0.0,
        "median_raw_gap_s": float(gaps.median()) if len(gaps) else 0.0,
        **file_selection_metadata,
    }
    return valid, metadata


def _natural_spline(
    source: pd.DataFrame,
    value_column: str,
    grid: pd.DatetimeIndex,
    logical_name: str,
) -> np.ndarray:
    source_start = pd.Timestamp(source["timestamp"].iloc[0])
    source_end = pd.Timestamp(source["timestamp"].iloc[-1])
    if pd.Timestamp(grid[0]) < source_start or pd.Timestamp(grid[-1]) > source_end:
        raise VoyageAlignmentError(
            "spline_extrapolation_requested",
            f"{logical_name} grid {grid[0].isoformat()}..{grid[-1].isoformat()} exceeds "
            f"source {source_start.isoformat()}..{source_end.isoformat()}",
        )
    x = (source["timestamp"] - source_start).dt.total_seconds().to_numpy(dtype=float)
    x_new = (pd.Series(grid) - source_start).dt.total_seconds().to_numpy(dtype=float)
    try:
        values = CubicSpline(
            x,
            source[value_column].to_numpy(dtype=float),
            bc_type="natural",
            extrapolate=False,
        )(x_new)
    except Exception as error:
        raise VoyageAlignmentError(
            "spline_interpolation_failed",
            f"{logical_name} natural cubic spline failed: {type(error).__name__}: {error}",
        ) from error
    if not np.isfinite(values).all():
        raise VoyageAlignmentError(
            "spline_interpolation_failed",
            f"{logical_name} natural cubic spline produced non-finite values",
        )
    return np.asarray(values, dtype=float)


def _optional_spline_on_grid(
    source: pd.DataFrame,
    value_column: str,
    grid: pd.DatetimeIndex,
    logical_name: str,
) -> np.ndarray:
    output = np.full(len(grid), np.nan, dtype=float)
    start = pd.Timestamp(source["timestamp"].iloc[0])
    end = pd.Timestamp(source["timestamp"].iloc[-1])
    inside = (grid >= start) & (grid <= end)
    if inside.any():
        output[inside] = _natural_spline(source, value_column, grid[inside], logical_name)
    return output


def _bundle_metadata(
    files: dict[str, Path],
) -> tuple[dict[str, str], dict[str, str], str]:
    channel_files: dict[str, str] = {}
    channel_hashes: dict[str, str] = {}
    bundle = hashlib.sha256()
    for logical_name, path in sorted(files.items()):
        file_hash = _sha256(path)
        channel_files[logical_name] = path.name
        channel_hashes[logical_name] = file_hash
        bundle.update(logical_name.encode("utf-8"))
        bundle.update(b"\0")
        bundle.update(path.name.encode("utf-8"))
        bundle.update(b"\0")
        bundle.update(file_hash.encode("ascii"))
        bundle.update(b"\n")
    return channel_files, channel_hashes, bundle.hexdigest()


def align_single_voyage(voyage: VoyageFiles) -> pd.DataFrame:
    """Spline 8 FC and 12 cluster-power channels to their shared 1 s range.

    Battery power is calculated on each original row before interpolation.
    BDM, inverter, and SOC channels are optional audit signals and never
    determine the formal source-side load.
    """

    required_series: dict[str, pd.DataFrame] = {}
    required_metadata: dict[str, dict[str, Any]] = {}
    required_source_files: dict[str, Path] = {}
    required_errors: list[dict[str, str]] = []
    optional_source_files: dict[str, Path] = {}
    cluster_keys: dict[str, list[str]] = {"left": [], "right": []}
    fuel_cell_keys: list[str] = []

    for side_key, side_name in (("left", "左"), ("right", "右")):
        for cluster_number in range(1, 7):
            logical_name = f"battery_{side_key}_cluster_{cluster_number}"
            cluster_keys[side_key].append(logical_name)
            try:
                path, file_metadata = _single_channel_file(
                    voyage.bms_dir,
                    f"{side_name}电池簇{cluster_number}",
                    required=True,
                )
                assert path is not None
                raw, metadata = _read_numeric_channel(
                    path,
                    [VOLTAGE_COLUMN, CURRENT_COLUMN],
                    logical_name,
                    file_metadata,
                )
                raw["power_kw"] = -(raw[VOLTAGE_COLUMN] * raw[CURRENT_COLUMN]) / 1000.0
                required_series[logical_name] = raw[["timestamp", "power_kw"]]
                required_metadata[logical_name] = metadata
                required_source_files[logical_name] = path
            except VoyageAlignmentError as error:
                required_errors.append(
                    {"logical_name": logical_name, "code": error.code, "message": str(error)}
                )
                required_metadata[logical_name] = {
                    "logical_name": logical_name,
                    "available_for_spline": False,
                    "error_code": error.code,
                    "error_message": str(error),
                    "max_raw_gap_s": None,
                }

    for side_key, side_name in (("left", "左"), ("right", "右")):
        for stack_number in range(1, 5):
            logical_name = f"fuel_cell_{side_key}_{stack_number}"
            fuel_cell_keys.append(logical_name)
            try:
                path, file_metadata = _single_channel_file(
                    voyage.fuel_cell_dir,
                    f"{side_name}氢燃料电池#{stack_number}",
                    required=True,
                )
                assert path is not None
                raw, metadata = _read_numeric_channel(
                    path,
                    [FUEL_CELL_POWER_COLUMN],
                    logical_name,
                    file_metadata,
                )
                required_series[logical_name] = raw[["timestamp", FUEL_CELL_POWER_COLUMN]].rename(
                    columns={FUEL_CELL_POWER_COLUMN: "power_kw"}
                )
                required_metadata[logical_name] = metadata
                required_source_files[logical_name] = path
            except VoyageAlignmentError as error:
                required_errors.append(
                    {"logical_name": logical_name, "code": error.code, "message": str(error)}
                )
                required_metadata[logical_name] = {
                    "logical_name": logical_name,
                    "available_for_spline": False,
                    "error_code": error.code,
                    "error_message": str(error),
                    "max_raw_gap_s": None,
                }

    conflicting_channels = {
        logical_name: metadata
        for logical_name, metadata in required_metadata.items()
        if int(metadata.get("conflicting_duplicate_timestamp_count", 0)) > 0
    }
    required_files, required_hashes, required_bundle_hash = _bundle_metadata(
        required_source_files
    )
    available_gaps = {
        logical_name: float(metadata["max_raw_gap_s"])
        for logical_name, metadata in required_metadata.items()
        if metadata.get("max_raw_gap_s") is not None
    }
    max_gap_channel = max(available_gaps, key=available_gaps.get) if available_gaps else ""
    required_failure_details = {
        "required_channel_metadata": required_metadata,
        "required_channel_files": required_files,
        "required_channel_hashes": required_hashes,
        "required_source_bundle_sha256": required_bundle_hash,
        "max_required_raw_gap_s": available_gaps.get(max_gap_channel),
        "max_required_raw_gap_channel": max_gap_channel,
        "required_channel_errors": required_errors,
    }
    if required_errors or conflicting_channels:
        error_summaries = [
            f"{item['logical_name']} [{item['code']}]: {item['message']}"
            for item in required_errors
        ]
        error_summaries.extend(
            f"{logical_name} [duplicate_raw_timestamp_conflict]: "
            f"{metadata['conflicting_duplicate_timestamp_count']} timestamp(s), "
            f"range={metadata['conflicting_duplicate_timestamp_range']}"
            for logical_name, metadata in conflicting_channels.items()
        )
        error_code = (
            required_errors[0]["code"]
            if required_errors
            else "duplicate_raw_timestamp_conflict"
        )
        raise VoyageAlignmentError(
            error_code,
            "required channel audit failed: " + "; ".join(error_summaries),
            details=required_failure_details,
        )

    common_start_raw = max(pd.Timestamp(frame["timestamp"].iloc[0]) for frame in required_series.values())
    common_end_raw = min(pd.Timestamp(frame["timestamp"].iloc[-1]) for frame in required_series.values())
    common_start = common_start_raw.ceil("s")
    common_end = common_end_raw.floor("s")
    if common_end < common_start:
        raise VoyageAlignmentError(
            "no_common_channel_overlap",
            f"20 required channels have no common 1 s range: start={common_start_raw.isoformat()}, "
            f"end={common_end_raw.isoformat()}",
            details=required_failure_details,
        )
    grid = pd.date_range(common_start, common_end, freq="1s")
    if len(grid) < 2:
        raise VoyageAlignmentError(
            "no_common_channel_overlap",
            f"20 required channels share only {len(grid)} one-second points",
            details=required_failure_details,
        )

    output = pd.DataFrame({"timestamp": grid, "time_s": np.arange(len(grid), dtype=float)})
    component_columns: dict[str, str] = {}
    for logical_name, source in required_series.items():
        column = f"__{logical_name}_power_kw"
        try:
            output[column] = _natural_spline(source, "power_kw", grid, logical_name)
        except VoyageAlignmentError as error:
            raise VoyageAlignmentError(
                error.code,
                str(error),
                details=required_failure_details,
            ) from error
        component_columns[logical_name] = column

    output["fuel_cell_total_kw"] = output[[component_columns[key] for key in fuel_cell_keys]].sum(axis=1)
    output["battery_left_6cluster_kw"] = output[
        [component_columns[key] for key in cluster_keys["left"]]
    ].sum(axis=1)
    output["battery_right_6cluster_kw"] = output[
        [component_columns[key] for key in cluster_keys["right"]]
    ].sum(axis=1)
    output["battery_cluster_total_kw"] = (
        output["battery_left_6cluster_kw"] + output["battery_right_6cluster_kw"]
    )
    output["total_load_kw"] = output["fuel_cell_total_kw"] + output["battery_cluster_total_kw"]
    output["total_load_clipped_kw"] = output["total_load_kw"].clip(lower=0.0)
    output["load_total_kw"] = output["total_load_clipped_kw"]

    optional_metadata: dict[str, dict[str, Any]] = {}
    optional_issues: list[dict[str, str]] = []
    bdm_power_columns: list[str] = []
    soc_columns: list[str] = []
    inverter_columns: list[str] = []

    for side_key, side_name in (("left", "左"), ("right", "右")):
        logical_name = f"battery_bdm_{side_key}"
        try:
            path, file_metadata = _single_channel_file(
                voyage.bms_dir,
                f"{side_name}电池系统BDM",
                required=False,
            )
            if path is None:
                raise VoyageAlignmentError("missing_optional_channel", "not present")
            raw, metadata = _read_numeric_channel(
                path,
                [VOLTAGE_COLUMN, CURRENT_COLUMN, SOC_COLUMN],
                logical_name,
                file_metadata,
            )
            if int(metadata["conflicting_duplicate_timestamp_count"]) > 0:
                raise VoyageAlignmentError(
                    "duplicate_raw_timestamp_conflict",
                    f"{path.name} ({logical_name}) has conflicting values at "
                    f"{metadata['conflicting_duplicate_timestamp_count']} timestamp(s), "
                    f"range={metadata['conflicting_duplicate_timestamp_range']}",
                )
            raw["power_kw"] = -(raw[VOLTAGE_COLUMN] * raw[CURRENT_COLUMN]) / 1000.0
            power_column = f"__{logical_name}_power_kw"
            soc_column = f"soc_{side_key}_pct"
            output[power_column] = _optional_spline_on_grid(raw, "power_kw", grid, logical_name)
            output[soc_column] = _optional_spline_on_grid(raw, SOC_COLUMN, grid, logical_name)
            bdm_power_columns.append(power_column)
            soc_columns.append(soc_column)
            optional_metadata[logical_name] = metadata
            optional_source_files[logical_name] = path
        except VoyageAlignmentError as error:
            optional_issues.append({"logical_name": logical_name, "code": error.code, "message": str(error)})

    for side_key, side_name in (("left", "左"), ("right", "右")):
        logical_name = f"propulsion_inverter_{side_key}"
        try:
            path, file_metadata = _single_channel_file(
                voyage.ems_dir,
                f"{side_name}逆变电源",
                required=False,
            )
            if path is None:
                raise VoyageAlignmentError("missing_optional_channel", "not present")
            raw, metadata = _read_numeric_channel(
                path,
                [INVERTER_POWER_COLUMN],
                logical_name,
                file_metadata,
            )
            if int(metadata["conflicting_duplicate_timestamp_count"]) > 0:
                raise VoyageAlignmentError(
                    "duplicate_raw_timestamp_conflict",
                    f"{path.name} ({logical_name}) has conflicting values at "
                    f"{metadata['conflicting_duplicate_timestamp_count']} timestamp(s), "
                    f"range={metadata['conflicting_duplicate_timestamp_range']}",
                )
            column = f"__{logical_name}_power_kw"
            output[column] = _optional_spline_on_grid(raw, INVERTER_POWER_COLUMN, grid, logical_name)
            inverter_columns.append(column)
            optional_metadata[logical_name] = metadata
            optional_source_files[logical_name] = path
        except VoyageAlignmentError as error:
            optional_issues.append({"logical_name": logical_name, "code": error.code, "message": str(error)})

    for side_key in ("left", "right"):
        if f"soc_{side_key}_pct" not in output:
            output[f"soc_{side_key}_pct"] = np.nan
    output["battery_bdm_total_kw"] = (
        output[bdm_power_columns].sum(axis=1, min_count=2) if len(bdm_power_columns) == 2 else np.nan
    )
    output["battery_cluster_minus_bdm_kw"] = (
        output["battery_cluster_total_kw"] - output["battery_bdm_total_kw"]
    )
    output["soc_mean_pct"] = output[soc_columns].mean(axis=1, skipna=False) if len(soc_columns) == 2 else np.nan
    output["propulsion_inverter_kw"] = (
        output[inverter_columns].sum(axis=1, min_count=2) if len(inverter_columns) == 2 else np.nan
    )
    output["dataset_version"] = DATASET_VERSION
    output["online_feasible"] = False
    output["uses_future_endpoint"] = True

    required_files, required_hashes, required_bundle_hash = _bundle_metadata(required_source_files)
    optional_files, optional_hashes, optional_bundle_hash = _bundle_metadata(optional_source_files)
    channel_files = {**required_files, **optional_files}
    channel_hashes = {**required_hashes, **optional_hashes}
    max_gap_channel = max(required_metadata, key=lambda key: required_metadata[key]["max_raw_gap_s"])
    output.attrs["required_channel_metadata"] = required_metadata
    output.attrs["optional_channel_metadata"] = optional_metadata
    output.attrs["optional_audit_issues"] = optional_issues
    output.attrs["source_channel_files"] = channel_files
    output.attrs["source_channel_hashes"] = channel_hashes
    output.attrs["required_source_bundle_sha256"] = required_bundle_hash
    output.attrs["optional_source_bundle_sha256"] = optional_bundle_hash
    output.attrs["source_bundle_sha256"] = required_bundle_hash
    output.attrs["common_overlap_start"] = common_start.isoformat()
    output.attrs["common_overlap_end"] = common_end.isoformat()
    output.attrs["max_required_raw_gap_s"] = float(required_metadata[max_gap_channel]["max_raw_gap_s"])
    output.attrs["max_required_raw_gap_channel"] = max_gap_channel
    output.attrs["component_columns"] = component_columns
    output.attrs["cluster_keys"] = cluster_keys
    output.attrs["fuel_cell_keys"] = fuel_cell_keys
    output.attrs["interpolation_method"] = "natural cubic spline per device power channel"
    output.attrs["battery_power_formula"] = "-(voltage_v * current_a) / 1000"
    output.attrs["metadata_json"] = json.dumps(required_metadata, ensure_ascii=False, sort_keys=True)
    return output
