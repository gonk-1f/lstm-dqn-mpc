"""Build formal 1 s load from 8 FC and 12 battery-cluster power channels."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd


PROJ = Path(__file__).resolve().parents[2]
SRC = PROJ / "src"
MAIN_DIR = Path(__file__).resolve().parent
for import_path in (SRC, MAIN_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from build_spline_1s_diagnostics import (  # noqa: E402
    CSV_FLOAT_FORMAT,
    publish_formal_voyages,
)
from utils.data_aligner import VoyageAlignmentError, align_single_voyage  # noqa: E402
from utils.data_loader import VoyageFiles  # noqa: E402


DEFAULT_INPUT_DIR = PROJ / "total_load_excels"
DEFAULT_RAW_ROOT = Path.home() / "OneDrive" / "Desktop" / "氢舟一号"
DEFAULT_OUTPUT_DIR = PROJ / "outputs" / "total_load_dataset_build"
DEFAULT_CONFIG_DIR = PROJ / "outputs" / "config"
DEFAULT_SPLINE_OUTPUT_DIR = PROJ / "outputs" / "spline_1s_diagnostics"
DEFAULT_AUDIT_CSV = PROJ / "reports" / "cluster_based_total_load_audit.csv"
DEFAULT_REPORT = PROJ / "docs" / "CLUSTER_BASED_TOTAL_LOAD_REBUILD.md"
FORMAL_30S_FILENAME = "total_load_66_segments.csv"
ACTIVE_SPLIT_FILENAME = "voyage_split_total_load_721.json"
FORMAL_1S_SUBDIRECTORY = "natural_clipped_by_voyage"
POWER_TOLERANCE_KW = 1e-3
LOAD_DEFINITION = "fuel_cell_total_kw + battery_cluster_total_kw"
LOAD_SCOPE = "source_side_device_channel_natural_spline_1s"
DERIVED_30S_ORIGIN = "derived_every_30th_sample_from_device_channel_natural_spline_1s"
TRANSACTION_MARKER_NAME = ".cluster_total_load_publish_transaction.json"
ALLOWED_EXCLUSION_CODES = {
    "missing_raw_directory",
    "missing_required_channel",
    "missing_required_field",
    "insufficient_spline_points",
    "no_common_channel_overlap",
    "duplicate_logical_channel",
    "duplicate_raw_timestamp_conflict",
    "spline_interpolation_failed",
}
NS = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

# Reference only: eligibility is recomputed from the new channel-level build.
LEGACY_BDM_EXCLUDED_IDS = {
    "voyage_001",
    "voyage_003",
    "voyage_004",
    "voyage_011",
    "voyage_017",
    "voyage_022",
    "voyage_024",
    "voyage_026",
    "voyage_032",
    "voyage_033",
    "voyage_045",
    "voyage_052",
    "voyage_058",
    "voyage_059",
    "voyage_060",
    "voyage_062",
}


@dataclass
class VoyageInput:
    voyage_id: str
    excel_path: Path
    legacy_frame: pd.DataFrame
    start_time: pd.Timestamp
    end_time: pd.Timestamp


def _col_ref_to_index(ref: str) -> int:
    letters = re.sub(r"[^A-Z]", "", ref.upper())
    index = 0
    for char in letters:
        index = index * 26 + (ord(char) - ord("A") + 1)
    return index - 1


def _text_content(node: ET.Element | None) -> str:
    return "" if node is None else "".join(node.itertext())


def _read_shared_strings(zf: ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    return [_text_content(si) for si in root.findall("x:si", NS)]


def read_xlsx_first_sheet(path: Path) -> pd.DataFrame:
    """Read the retained workbook without modifying it."""

    with ZipFile(path) as zf:
        shared_strings = _read_shared_strings(zf)
        sheet_path = "xl/worksheets/sheet1.xml"
        if sheet_path not in zf.namelist():
            raise ValueError(f"{path} does not contain {sheet_path}")
        root = ET.fromstring(zf.read(sheet_path))
    rows: list[list[Any]] = []
    for row in root.findall(".//x:sheetData/x:row", NS):
        values: dict[int, Any] = {}
        for cell in row.findall("x:c", NS):
            ref = cell.attrib.get("r", "")
            if not ref:
                continue
            idx = _col_ref_to_index(ref)
            cell_type = cell.attrib.get("t", "")
            value_node = cell.find("x:v", NS)
            if cell_type == "s":
                raw_value = _text_content(value_node)
                value = shared_strings[int(raw_value)] if raw_value else ""
            elif cell_type == "inlineStr":
                value = _text_content(cell.find("x:is", NS))
            else:
                value = _text_content(value_node)
            values[idx] = value
        if values:
            rows.append([values.get(i, "") for i in range(max(values) + 1)])
    if not rows:
        raise ValueError(f"{path} has no worksheet rows")
    header = [str(value).strip() for value in rows[0]]
    data: list[list[Any]] = []
    for row in rows[1:]:
        padded = row + [""] * (len(header) - len(row))
        data.append(padded[: len(header)])
    return pd.DataFrame(data, columns=header)


def _numeric_or_nan(raw: pd.DataFrame, column: str) -> pd.Series:
    if column not in raw.columns:
        return pd.Series(np.nan, index=raw.index, dtype=float)
    return pd.to_numeric(raw[column], errors="coerce")


def _read_legacy_frame(path: Path) -> pd.DataFrame:
    raw = read_xlsx_first_sheet(path)
    if "timestamp" not in raw.columns:
        raise ValueError(f"{path.name} missing timestamp")
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(raw["timestamp"], errors="coerce"),
            "old_fuel_cell_total_kw": _numeric_or_nan(raw, "fuel_cell_total_kw"),
            "old_battery_bdm_total_kw": _numeric_or_nan(raw, "battery_total_kw"),
            "old_propulsion_inverter_kw": _numeric_or_nan(raw, "propulsion_inverter_total_kw"),
            "old_soc_left_pct": _numeric_or_nan(raw, "soc_left_pct"),
            "old_soc_right_pct": _numeric_or_nan(raw, "soc_right_pct"),
            "old_soc_mean_pct": _numeric_or_nan(raw, "soc_mean_pct"),
        }
    )
    if frame["timestamp"].isna().any():
        raise ValueError(f"{path.name} has invalid legacy timestamps")
    if frame.empty or not frame["timestamp"].is_monotonic_increasing:
        raise ValueError(f"{path.name} legacy timestamps are empty or nonmonotonic")
    if frame["timestamp"].duplicated().any():
        raise ValueError(f"{path.name} has duplicate legacy timestamps")
    return frame


def _discover_inputs(input_dir: Path, expected_count: int) -> list[VoyageInput]:
    paths = sorted(Path(input_dir).glob("*.xlsx"))
    if len(paths) != int(expected_count):
        raise ValueError(f"Expected {expected_count} xlsx voyages in {input_dir}, found {len(paths)}")
    prepared = [(path, _read_legacy_frame(path)) for path in paths]
    prepared.sort(key=lambda item: (item[1]["timestamp"].iloc[0], item[0].name))
    return [
        VoyageInput(
            voyage_id=f"voyage_{idx:03d}",
            excel_path=path,
            legacy_frame=frame,
            start_time=pd.Timestamp(frame["timestamp"].iloc[0]),
            end_time=pd.Timestamp(frame["timestamp"].iloc[-1]),
        )
        for idx, (path, frame) in enumerate(prepared, start=1)
    ]


def _raw_voyage_files(raw_root: Path, voyage_name: str) -> VoyageFiles:
    root = Path(raw_root) / voyage_name
    return VoyageFiles(
        root=root,
        bms_dir=root / "BMS",
        ems_dir=root / "EMS",
        fuel_cell_dir=root / "燃料电池系统",
        propulsion_dir=root / "推进系统",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mask_ranges(timestamp: pd.Series, mask: pd.Series, expected_seconds: float) -> str:
    selected = pd.DataFrame({"timestamp": pd.to_datetime(timestamp), "selected": mask.astype(bool)})
    ranges: list[str] = []
    start: pd.Timestamp | None = None
    previous: pd.Timestamp | None = None
    for row in selected.itertuples(index=False):
        current = pd.Timestamp(row.timestamp)
        if row.selected:
            if start is None:
                start = current
            elif previous is not None and abs((current - previous).total_seconds() - expected_seconds) > 1e-9:
                ranges.append(f"{start.isoformat()}..{previous.isoformat()}")
                start = current
            previous = current
        elif start is not None and previous is not None:
            ranges.append(f"{start.isoformat()}..{previous.isoformat()}")
            start = None
            previous = None
    if start is not None and previous is not None:
        ranges.append(f"{start.isoformat()}..{previous.isoformat()}")
    return "; ".join(ranges)


def _range_text(values: pd.Series) -> str:
    valid = pd.to_numeric(values, errors="coerce").dropna()
    if valid.empty:
        return ""
    return f"{float(valid.min()):.9g}..{float(valid.max()):.9g}"


def _legacy_anomaly_mask(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["old_fuel_cell_total_kw"].abs().le(POWER_TOLERANCE_KW)
        & frame["old_battery_bdm_total_kw"].abs().le(POWER_TOLERANCE_KW)
        & frame["old_propulsion_inverter_kw"].gt(POWER_TOLERANCE_KW)
    )


def _contiguous_mask_groups(
    frame: pd.DataFrame,
    mask: pd.Series,
    *,
    expected_seconds: float,
) -> list[pd.DataFrame]:
    selected = frame.loc[mask].copy()
    if selected.empty:
        return []
    selected = selected.sort_values("timestamp")
    group_id = (
        selected["timestamp"].diff().dt.total_seconds().fillna(expected_seconds)
        .ne(expected_seconds)
        .cumsum()
    )
    return [group.copy() for _, group in selected.groupby(group_id, sort=True)]


def _evaluate_anomalies(legacy: pd.DataFrame, reconstructed: pd.DataFrame) -> dict[str, Any]:
    old_mask = _legacy_anomaly_mask(legacy)
    old_rows = legacy.loc[old_mask].copy()
    persistent = (
        reconstructed["fuel_cell_total_kw"].abs().le(POWER_TOLERANCE_KW)
        & reconstructed["battery_cluster_total_kw"].abs().le(POWER_TOLERANCE_KW)
        & reconstructed["propulsion_inverter_kw"].notna()
        & reconstructed["propulsion_inverter_kw"].gt(POWER_TOLERANCE_KW)
    )
    persistent_details: list[dict[str, Any]] = []
    for group in _contiguous_mask_groups(reconstructed, persistent, expected_seconds=1.0):
        group_soc = group["soc_mean_pct"].dropna()
        persistent_details.append(
            {
                "start_timestamp": pd.Timestamp(group["timestamp"].iloc[0]).isoformat(),
                "end_timestamp": pd.Timestamp(group["timestamp"].iloc[-1]).isoformat(),
                "row_count": int(len(group)),
                "reconstructed_fc_kw_range": _range_text(group["fuel_cell_total_kw"]),
                "reconstructed_cluster_battery_kw_range": _range_text(
                    group["battery_cluster_total_kw"]
                ),
                "reconstructed_total_load_kw_range": _range_text(group["total_load_kw"]),
                "reconstructed_inverter_kw_range": _range_text(
                    group["propulsion_inverter_kw"]
                ),
                "reconstructed_soc_pct_range": _range_text(group_soc),
                "reconstructed_soc_delta_pct": (
                    float(group_soc.iloc[-1] - group_soc.iloc[0])
                    if len(group_soc) >= 2
                    else None
                ),
            }
        )
    result: dict[str, Any] = {
        "old_anomaly_count": int(old_mask.sum()),
        "old_anomaly_ranges": _mask_ranges(legacy["timestamp"], old_mask, 30.0),
        "persistent_inconsistency_count_1s": int(persistent.sum()),
        "persistent_inconsistency_ranges": _mask_ranges(
            reconstructed["timestamp"], persistent, 1.0
        ),
        "persistent_inconsistency_details_json": json.dumps(
            persistent_details,
            ensure_ascii=False,
            sort_keys=True,
        ),
    }
    if old_rows.empty:
        result.update(
            {
                "old_anomaly_evaluable_count": 0,
                "old_anomaly_resolved_count": 0,
                "old_anomaly_source_resolved_count": 0,
                "old_anomaly_published_target_resolved_count": 0,
                "old_anomaly_unevaluable_count": 0,
                "old_anomaly_disappeared": False,
                "old_anomaly_reconstructed_fc_kw_range": "",
                "old_anomaly_reconstructed_cluster_battery_kw_range": "",
                "old_anomaly_reconstructed_total_load_kw_range": "",
                "old_anomaly_inverter_kw_range": "",
                "old_anomaly_soc_pct_range": "",
                "old_anomaly_soc_delta_pct_by_range": "",
                "old_anomaly_details_json": "[]",
            }
        )
        return result

    columns = [
        "timestamp",
        "fuel_cell_total_kw",
        "battery_cluster_total_kw",
        "total_load_kw",
        "load_total_kw",
        "propulsion_inverter_kw",
        "soc_mean_pct",
    ]
    comparison = old_rows.merge(reconstructed[columns], on="timestamp", how="left")
    evaluable = comparison["total_load_kw"].notna()
    source_resolved = (
        evaluable
        & comparison["battery_cluster_total_kw"].abs().gt(POWER_TOLERANCE_KW)
        & comparison["total_load_kw"].abs().gt(POWER_TOLERANCE_KW)
    )
    published_target_resolved = evaluable & comparison["load_total_kw"].gt(
        POWER_TOLERANCE_KW
    )
    resolved = source_resolved & published_target_resolved
    old_soc = comparison.loc[evaluable, "old_soc_mean_pct"].dropna()
    details: list[dict[str, Any]] = []
    soc_delta_text: list[str] = []
    for old_group in _contiguous_mask_groups(legacy, old_mask, expected_seconds=30.0):
        group_comparison = old_group.merge(reconstructed[columns], on="timestamp", how="left")
        group_evaluable = group_comparison["total_load_kw"].notna()
        group_source_resolved = (
            group_evaluable
            & group_comparison["battery_cluster_total_kw"].abs().gt(POWER_TOLERANCE_KW)
            & group_comparison["total_load_kw"].abs().gt(POWER_TOLERANCE_KW)
        )
        group_target_resolved = group_evaluable & group_comparison["load_total_kw"].gt(
            POWER_TOLERANCE_KW
        )
        group_resolved = group_source_resolved & group_target_resolved
        group_soc = group_comparison["old_soc_mean_pct"].dropna()
        group_soc_delta = (
            float(group_soc.iloc[-1] - group_soc.iloc[0]) if len(group_soc) >= 2 else None
        )
        if group_soc_delta is not None:
            soc_delta_text.append(
                f"{pd.Timestamp(old_group['timestamp'].iloc[0]).isoformat()}.."
                f"{pd.Timestamp(old_group['timestamp'].iloc[-1]).isoformat()}:{group_soc_delta:.9g}"
            )
        details.append(
            {
                "start_timestamp": pd.Timestamp(old_group["timestamp"].iloc[0]).isoformat(),
                "end_timestamp": pd.Timestamp(old_group["timestamp"].iloc[-1]).isoformat(),
                "legacy_row_count": int(len(old_group)),
                "evaluable_row_count": int(group_evaluable.sum()),
                "resolved_row_count": int(group_resolved.sum()),
                "source_resolved_row_count": int(group_source_resolved.sum()),
                "published_target_resolved_row_count": int(group_target_resolved.sum()),
                "fully_resolved": bool(len(group_comparison) and group_resolved.all()),
                "reconstructed_fc_kw_range": _range_text(
                    group_comparison.loc[group_evaluable, "fuel_cell_total_kw"]
                ),
                "reconstructed_cluster_battery_kw_range": _range_text(
                    group_comparison.loc[group_evaluable, "battery_cluster_total_kw"]
                ),
                "reconstructed_total_load_kw_range": _range_text(
                    group_comparison.loc[group_evaluable, "total_load_kw"]
                ),
                "published_clipped_load_kw_range": _range_text(
                    group_comparison.loc[group_evaluable, "load_total_kw"]
                ),
                "legacy_inverter_kw_range": _range_text(
                    group_comparison["old_propulsion_inverter_kw"]
                ),
                "legacy_soc_pct_range": _range_text(group_soc),
                "legacy_soc_delta_pct": group_soc_delta,
            }
        )
    result.update(
        {
            "old_anomaly_evaluable_count": int(evaluable.sum()),
            "old_anomaly_resolved_count": int(resolved.sum()),
            "old_anomaly_source_resolved_count": int(source_resolved.sum()),
            "old_anomaly_published_target_resolved_count": int(
                published_target_resolved.sum()
            ),
            "old_anomaly_unevaluable_count": int((~evaluable).sum()),
            "old_anomaly_disappeared": bool(len(comparison) and resolved.all()),
            "old_anomaly_reconstructed_fc_kw_range": _range_text(
                comparison.loc[evaluable, "fuel_cell_total_kw"]
            ),
            "old_anomaly_reconstructed_cluster_battery_kw_range": _range_text(
                comparison.loc[evaluable, "battery_cluster_total_kw"]
            ),
            "old_anomaly_reconstructed_total_load_kw_range": _range_text(
                comparison.loc[evaluable, "total_load_kw"]
            ),
            "old_anomaly_inverter_kw_range": _range_text(
                comparison["old_propulsion_inverter_kw"]
            ),
            "old_anomaly_soc_pct_range": _range_text(old_soc),
            "old_anomaly_soc_delta_pct_by_range": "; ".join(soc_delta_text),
            "old_anomaly_details_json": json.dumps(details, ensure_ascii=False, sort_keys=True),
        }
    )
    return result


def _chronological_split(eligible_ids: list[str]) -> dict[str, list[str]]:
    train_count = int(len(eligible_ids) * 0.7)
    validation_count = int(len(eligible_ids) * 0.2)
    return {
        "train_voyages": eligible_ids[:train_count],
        "validation_voyages": eligible_ids[train_count : train_count + validation_count],
        "test_voyages": eligible_ids[train_count + validation_count :],
    }


def _safe_remove_directory(path: Path, root: Path) -> None:
    resolved_path = path.resolve()
    resolved_root = root.resolve()
    if resolved_path == resolved_root or not resolved_path.is_relative_to(resolved_root):
        raise ValueError(f"Refusing to remove directory outside build root: {resolved_path}")
    if path.exists():
        shutil.rmtree(path)


def _write_csv_temp(frame: pd.DataFrame, final_path: Path) -> Path:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = final_path.with_suffix(final_path.suffix + ".tmp")
    frame.to_csv(
        temporary,
        index=False,
        encoding="utf-8-sig",
        float_format=CSV_FLOAT_FORMAT,
    )
    return temporary


def _write_text_temp(text: str, final_path: Path) -> Path:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = final_path.with_suffix(final_path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    return temporary


def _artifact_backup_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.cluster_rebuild_backup")


def _write_transaction_marker(marker: Path, payload: dict[str, Any]) -> None:
    marker.parent.mkdir(parents=True, exist_ok=True)
    temporary = marker.with_suffix(marker.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(marker)


def _cleanup_transaction_backups(
    payload: dict[str, Any],
    *,
    marker: Path,
    data_dir: Path,
) -> None:
    formal_backup = Path(payload["formal_backup_dir"])
    _safe_remove_directory(formal_backup, data_dir)
    for record in payload["scalar_artifacts"]:
        backup = Path(record["backup"])
        if backup.exists():
            backup.unlink()
        backup_temp = backup.with_suffix(backup.suffix + ".tmp")
        if backup_temp.exists():
            backup_temp.unlink()
    marker_temp = marker.with_suffix(marker.suffix + ".tmp")
    if marker_temp.exists():
        marker_temp.unlink()
    if marker.exists():
        marker.unlink()


def _recover_publish_transaction(
    *,
    marker: Path,
    formal_dir: Path,
    data_dir: Path,
    scalar_paths: list[Path],
) -> None:
    formal_backup = data_dir / f".{FORMAL_1S_SUBDIRECTORY}_backup"
    if not marker.exists():
        if formal_backup.exists() and not formal_dir.exists():
            formal_backup.replace(formal_dir)
        elif formal_backup.exists():
            _safe_remove_directory(formal_backup, data_dir)
        for final_path in scalar_paths:
            backup = _artifact_backup_path(final_path)
            for candidate in (backup, backup.with_suffix(backup.suffix + ".tmp")):
                if candidate.exists():
                    candidate.unlink()
        marker_temp = marker.with_suffix(marker.suffix + ".tmp")
        if marker_temp.exists():
            marker_temp.unlink()
        return

    payload = json.loads(marker.read_text(encoding="utf-8"))
    state = str(payload.get("state", "prepared"))
    if state != "committed":
        stored_formal_backup = Path(payload["formal_backup_dir"])
        original_formal_existed = bool(payload["formal_original_existed"])
        if stored_formal_backup.exists():
            _safe_remove_directory(formal_dir, data_dir)
            stored_formal_backup.replace(formal_dir)
        elif not original_formal_existed:
            _safe_remove_directory(formal_dir, data_dir)
        elif state == "formal_swapped":
            raise RuntimeError(
                "Cannot roll back interrupted publication: formal backup is missing"
            )
        for record in payload["scalar_artifacts"]:
            final_path = Path(record["final"])
            backup = Path(record["backup"])
            if bool(record["original_existed"]):
                if not backup.exists():
                    raise RuntimeError(
                        f"Cannot roll back interrupted publication: missing backup {backup}"
                    )
                backup.replace(final_path)
            elif final_path.exists():
                final_path.unlink()
    _cleanup_transaction_backups(payload, marker=marker, data_dir=data_dir)


def _prepare_publish_transaction(
    *,
    marker: Path,
    formal_dir: Path,
    data_dir: Path,
    scalar_paths: list[Path],
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for final_path in scalar_paths:
        resolved = final_path.resolve()
        if not resolved.is_relative_to(PROJ.resolve()):
            raise ValueError(f"Refusing to transact artifact outside repository: {resolved}")
        final_path.parent.mkdir(parents=True, exist_ok=True)
        backup = _artifact_backup_path(final_path)
        backup_temp = backup.with_suffix(backup.suffix + ".tmp")
        if backup.exists() or backup_temp.exists():
            raise RuntimeError(f"Stale transaction backup exists: {backup}")
        original_existed = final_path.exists()
        if original_existed:
            shutil.copy2(final_path, backup_temp)
            backup_temp.replace(backup)
        records.append(
            {
                "final": str(final_path.resolve()),
                "backup": str(backup.resolve()),
                "original_existed": original_existed,
            }
        )
    payload = {
        "state": "prepared",
        "formal_dir": str(formal_dir.resolve()),
        "formal_backup_dir": str(
            (data_dir / f".{FORMAL_1S_SUBDIRECTORY}_backup").resolve()
        ),
        "formal_original_existed": formal_dir.exists(),
        "scalar_artifacts": records,
    }
    _write_transaction_marker(marker, payload)
    return payload


def _public_1s_frame(
    reconstructed: pd.DataFrame,
    *,
    voyage_id: str,
    source_file_name: str,
    required_bundle_sha256: str,
) -> pd.DataFrame:
    output = reconstructed[[column for column in reconstructed.columns if not column.startswith("__")]].copy()
    output.insert(0, "voyage_id", voyage_id)
    output.insert(1, "source_file_name", source_file_name)
    output["file_name"] = source_file_name
    output["battery_total_kw"] = output["battery_cluster_total_kw"]
    output["total_load_fc_plus_batt_kw"] = output["total_load_kw"]
    output["propulsion_load_kw"] = output["propulsion_inverter_kw"]
    output["data_origin"] = "reconstructed_from_20_device_power_channels_by_natural_cubic_spline"
    output["time_original_or_reconstructed"] = "reconstructed_1s_from_device_channels"
    output["is_original_30s_point"] = False
    output["required_source_bundle_sha256"] = required_bundle_sha256
    return output


def _status_from_anomaly(result: dict[str, Any]) -> str:
    if int(result["persistent_inconsistency_count_1s"]) > 0:
        return "persistent_source_power_inconsistency"
    if int(result["old_anomaly_count"]) and bool(result["old_anomaly_disappeared"]):
        return "resolved_by_cluster_reconstruction"
    if int(result["old_anomaly_count"]) and int(result["old_anomaly_unevaluable_count"]):
        return "old_anomaly_not_fully_evaluable"
    if int(result["old_anomaly_count"]):
        return "old_anomaly_not_resolved"
    return "normal"


def _build_report(
    audit: pd.DataFrame,
    split_payload: dict[str, Any],
    formal_30s_csv: Path,
    formal_1s_dir: Path,
) -> str:
    successful = audit.loc[audit["reconstruction_succeeded"].astype(bool)]
    excluded = audit.loc[~audit["reconstruction_succeeded"].astype(bool)]
    resolved = successful.loc[
        successful["old_anomaly_count"].gt(0)
        & successful["old_anomaly_disappeared"].astype(bool),
        "voyage_id",
    ].tolist()
    persistent = successful.loc[
        successful["reconstruction_status"].eq("persistent_source_power_inconsistency"), "voyage_id"
    ].tolist()
    legacy_recovered = [voyage_id for voyage_id in resolved if voyage_id in LEGACY_BDM_EXCLUDED_IDS]
    duplicate_rows_removed = 0
    audit_only_duplicate_rows = 0
    identical_duplicate_files_ignored = 0
    timestamp_corrections: list[tuple[str, str, dict[str, Any]]] = []
    for audit_row in audit.itertuples(index=False):
        metadata_text = audit_row.required_channel_metadata_json
        if pd.isna(metadata_text):
            continue
        if not str(metadata_text).strip():
            continue
        metadata = json.loads(str(metadata_text))
        exact_rows = sum(
            int(channel.get("exact_duplicate_rows_dropped", 0))
            for channel in metadata.values()
        )
        if bool(audit_row.reconstruction_succeeded):
            duplicate_rows_removed += exact_rows
        else:
            audit_only_duplicate_rows += exact_rows
        identical_duplicate_files_ignored += sum(
            int(channel.get("identical_duplicate_files_ignored", 0))
            for channel in metadata.values()
        )
        for logical_name, channel in metadata.items():
            timestamp_corrections.extend(
                (str(audit_row.voyage_id), str(logical_name), correction)
                for correction in channel.get("timestamp_corrections", [])
            )
    correction_voyages = sorted({item[0] for item in timestamp_corrections})
    corrected_time_counts: dict[str, int] = {}
    for _, _, correction in timestamp_corrections:
        corrected_time = str(correction["corrected_timestamp"])
        corrected_time_counts[corrected_time] = corrected_time_counts.get(corrected_time, 0) + 1
    correction_summary = (
        f"{', '.join(correction_voyages)} 共 {len(timestamp_corrections)} 路各1条；"
        + "修正时刻分布："
        + ", ".join(
            f"{timestamp} × {count}"
            for timestamp, count in sorted(corrected_time_counts.items())
        )
        + "。只在构建时修正时间戳，功率值与原始CSV均未修改。"
        if timestamp_corrections
        else "无。"
    )
    negative_rows_clipped = int(successful["negative_total_load_rows_clipped"].sum())
    lines = [
        "# 基于设备通道 1 s 重构的总负荷审计",
        "",
        "## 计算口径",
        "",
        "- FC：左/右各 #1～#4 的 `发电功率(kW)`，8 路分别做 natural cubic spline 后求和。",
        "- 电池：左/右各簇1～6，先在原始同一行按 `-(总电压(V)×总电流(A))/1000` 计算12路簇功率，再分别做 natural cubic spline 后求和。",
        "- 每航段1 s时间轴是20路必需功率通道有效范围的共同交集；不外推、不跨航段、不 `fillna(0)`、不使用BDM回退。",
        "- 原始内部断档允许由 natural cubic spline 连接；逐通道最大断档记录在审计CSV。",
        "- `total_load_kw = fuel_cell_total_kw + battery_cluster_total_kw` 保留未裁剪恒等式；`load_total_kw` 与 `total_load_clipped_kw` 是现有 natural-clipped 非负建模列。",
        "- BDM、逆变器和SOC只用于审计，不参与正式总负荷。",
        "- 实际字段映射：`fuel_cell_left/right_1..4` 对应 `左/右氢燃料电池#1..#4_*.csv::发电功率(kW)`；`battery_left/right_cluster_1..6` 对应 `左/右电池簇1..6_*.csv::总电压(V), 总电流(A)`。",
        "",
        "## 构建与旧异常结论",
        "",
        f"- 全部航段：{len(audit)}；成功重构：{len(successful)}；结构性排除：{len(excluded)}。",
        f"- `resolved_by_cluster_reconstruction`：{', '.join(resolved) if resolved else '无'}。",
        f"- 旧16航段中恢复：{', '.join(legacy_recovered) if legacy_recovered else '无'}。",
        f"- `persistent_source_power_inconsistency`：{', '.join(persistent) if persistent else '无'}。该标签仅作审计，不超出任务列出的结构性排除条件。",
        f"- 可唯一消解的重复：可用航段去除完全相同的重复时间行 {duplicate_rows_removed} 条；排除航段仅为时间断档审计折叠 {audit_only_duplicate_rows} 条；忽略SHA256完全相同的重复文件 {identical_duplicate_files_ignored} 份。未经显式审计勘误的冲突重复值不参与插值。",
        f"- 已审计定点时间勘误：{correction_summary}",
        f"- natural spline 后未裁剪总负荷为负的行数：{negative_rows_clipped}；仅 `load_total_kw`/`total_load_clipped_kw` 按既有规则裁剪为0，源侧恒等式列 `total_load_kw` 不裁剪。",
        "",
        "### 排除航段",
        "",
    ]
    if excluded.empty:
        lines.append("无。")
    else:
        lines.extend(["| 航段 | 原始文件 | 原因 |", "|---|---|---|"])
        for row in excluded.itertuples(index=False):
            lines.append(f"| {row.voyage_id} | {row.source_file_name} | {row.exclusion_reason} |")

    lines.extend(["", "## 各航段最大原始断档", "", "| 航段 | 最大断档(s) | 通道 |", "|---|---:|---|"])
    for row in audit.itertuples(index=False):
        gap = "" if pd.isna(row.max_required_raw_gap_s) else f"{float(row.max_required_raw_gap_s):.3f}"
        channel = "" if pd.isna(row.max_required_raw_gap_channel) else str(row.max_required_raw_gap_channel)
        lines.append(f"| {row.voyage_id} | {gap} | {channel} |")

    lines.extend(
        [
            "",
            "## Chronological 7:2:1 划分",
            "",
            f"- train ({len(split_payload['train_voyages'])})：{', '.join(split_payload['train_voyages'])}",
            f"- validation ({len(split_payload['validation_voyages'])})：{', '.join(split_payload['validation_voyages'])}",
            f"- test ({len(split_payload['test_voyages'])})：{', '.join(split_payload['test_voyages'])}",
            "- 三集合互斥；同一航段不拆分；所有成功重构航段只出现一次。",
            "",
            "## 正式输出与限制",
            "",
            f"- 正式1 s：`{formal_1s_dir.relative_to(PROJ).as_posix()}`。",
            f"- 派生30 s汇总：`{formal_30s_csv.relative_to(PROJ).as_posix()}`；由正式1 s每30点抽样，不是原始30 s实测总负荷。",
            "- natural spline 会跨越记录断档，断档越长，区间内功率的不确定性越高；本任务只记录断档，不新增滤波或修复规则。",
            "- 原始文件没有独立通道倍率、标定和拓扑说明；仅依据字段名称及V、A、kW单位计算。",
            "",
        ]
    )
    return "\n".join(lines)


def build_dataset(
    *,
    input_dir: Path = DEFAULT_INPUT_DIR,
    raw_root: Path = DEFAULT_RAW_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    config_dir: Path = DEFAULT_CONFIG_DIR,
    spline_output_dir: Path = DEFAULT_SPLINE_OUTPUT_DIR,
    audit_csv: Path = DEFAULT_AUDIT_CSV,
    report_path: Path = DEFAULT_REPORT,
    expected_count: int = 66,
) -> dict[str, Any]:
    """Build all formal artifacts without creating an additional entry point."""

    inputs = _discover_inputs(Path(input_dir), int(expected_count))
    data_dir = Path(spline_output_dir) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    formal_dir = data_dir / FORMAL_1S_SUBDIRECTORY
    formal_30s_csv = Path(output_dir) / FORMAL_30S_FILENAME
    split_path = Path(config_dir) / ACTIVE_SPLIT_FILENAME
    audit_path = Path(audit_csv)
    report_final = Path(report_path)
    scalar_artifact_paths = [formal_30s_csv, split_path, audit_path, report_final]
    transaction_marker = data_dir / TRANSACTION_MARKER_NAME
    _recover_publish_transaction(
        marker=transaction_marker,
        formal_dir=formal_dir,
        data_dir=data_dir,
        scalar_paths=scalar_artifact_paths,
    )
    staging_dir = data_dir / f".{FORMAL_1S_SUBDIRECTORY}_staging"
    _safe_remove_directory(staging_dir, data_dir)
    staging_dir.mkdir(parents=True)

    audit_rows: list[dict[str, Any]] = []
    voyage_records: list[dict[str, Any]] = []
    derived_30s_frames: list[pd.DataFrame] = []
    temporary_paths: list[Path] = []
    try:
        for item in inputs:
            legacy_mask = _legacy_anomaly_mask(item.legacy_frame)
            base: dict[str, Any] = {
                "voyage_id": item.voyage_id,
                "source_file_name": item.excel_path.name,
                "raw_voyage_directory": item.excel_path.stem,
                "legacy_start_timestamp": item.start_time.isoformat(),
                "legacy_end_timestamp": item.end_time.isoformat(),
                "old_excluded_reference": item.voyage_id in LEGACY_BDM_EXCLUDED_IDS,
                "old_anomaly_count": int(legacy_mask.sum()),
                "old_anomaly_ranges": _mask_ranges(item.legacy_frame["timestamp"], legacy_mask, 30.0),
            }
            try:
                raw_files = _raw_voyage_files(Path(raw_root), item.excel_path.stem)
                reconstructed = align_single_voyage(raw_files)
                attrs = dict(reconstructed.attrs)
                timestamp = pd.to_datetime(reconstructed["timestamp"])
                if timestamp.duplicated().any() or not timestamp.diff().dt.total_seconds().dropna().eq(1.0).all():
                    raise VoyageAlignmentError(
                        "invalid_reconstructed_timeline",
                        "reconstructed timestamp is not unique and strictly 1 s",
                    )
                component_columns = attrs["component_columns"]
                fc_components = [component_columns[key] for key in attrs["fuel_cell_keys"]]
                cluster_components = [
                    component_columns[key]
                    for side in ("left", "right")
                    for key in attrs["cluster_keys"][side]
                ]
                fc_error = float(
                    (reconstructed[fc_components].sum(axis=1) - reconstructed["fuel_cell_total_kw"]).abs().max()
                )
                cluster_error = float(
                    (
                        reconstructed[cluster_components].sum(axis=1)
                        - reconstructed["battery_cluster_total_kw"]
                    ).abs().max()
                )
                load_error = float(
                    (
                        reconstructed["fuel_cell_total_kw"]
                        + reconstructed["battery_cluster_total_kw"]
                        - reconstructed["total_load_kw"]
                    ).abs().max()
                )
                if max(fc_error, cluster_error, load_error) > 1e-9:
                    raise VoyageAlignmentError(
                        "power_identity_failure",
                        f"fc_error={fc_error:g}, cluster_error={cluster_error:g}, load_error={load_error:g}",
                    )

                anomaly = _evaluate_anomalies(item.legacy_frame, reconstructed)
                status = _status_from_anomaly(anomaly)
                public = _public_1s_frame(
                    reconstructed,
                    voyage_id=item.voyage_id,
                    source_file_name=item.excel_path.name,
                    required_bundle_sha256=attrs["required_source_bundle_sha256"],
                )
                output_name = f"{item.voyage_id}__{item.excel_path.stem}.csv"
                output_path = staging_dir / output_name
                public.to_csv(
                    output_path,
                    index=False,
                    encoding="utf-8-sig",
                    float_format=CSV_FLOAT_FORMAT,
                )
                derived = public.iloc[::30].copy()
                derived["sample_interval_seconds"] = 30.0
                derived["data_origin"] = DERIVED_30S_ORIGIN
                derived_30s_frames.append(derived)
                voyage_records.append(
                    {
                        "voyage_id": item.voyage_id,
                        "source_file_name": item.excel_path.name,
                        "output_name": output_name,
                        "common_overlap_start": attrs["common_overlap_start"],
                        "common_overlap_end": attrs["common_overlap_end"],
                        "max_required_raw_gap_s": attrs["max_required_raw_gap_s"],
                        "max_required_raw_gap_channel": attrs["max_required_raw_gap_channel"],
                        "source_bundle_sha256": attrs["required_source_bundle_sha256"],
                        "required_source_channel_hashes_json": json.dumps(
                            {
                                key: value
                                for key, value in attrs["source_channel_hashes"].items()
                                if key in attrs["required_channel_metadata"]
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                    }
                )
                base.update(
                    {
                        "reconstruction_succeeded": True,
                        "usable": True,
                        "reconstruction_status": status,
                        "exclusion_reason": "",
                        "error_code": "",
                        "formal_1s_rows": int(len(public)),
                        "common_overlap_start": attrs["common_overlap_start"],
                        "common_overlap_end": attrs["common_overlap_end"],
                        "max_required_raw_gap_s": attrs["max_required_raw_gap_s"],
                        "max_required_raw_gap_channel": attrs["max_required_raw_gap_channel"],
                        "max_raw_gap_seconds_by_channel_json": json.dumps(
                            {
                                key: value["max_raw_gap_s"]
                                for key, value in attrs["required_channel_metadata"].items()
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                        "required_channel_metadata_json": json.dumps(
                            attrs["required_channel_metadata"], ensure_ascii=False, sort_keys=True
                        ),
                        "required_channel_files_json": json.dumps(
                            {
                                key: value
                                for key, value in attrs["source_channel_files"].items()
                                if key in attrs["required_channel_metadata"]
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                        "required_channel_hashes_json": json.dumps(
                            {
                                key: value
                                for key, value in attrs["source_channel_hashes"].items()
                                if key in attrs["required_channel_metadata"]
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                        "required_source_bundle_sha256": attrs["required_source_bundle_sha256"],
                        "optional_source_bundle_sha256": attrs["optional_source_bundle_sha256"],
                        "optional_audit_issues_json": json.dumps(
                            attrs["optional_audit_issues"], ensure_ascii=False, sort_keys=True
                        ),
                        "fc_component_sum_max_abs_error_kw": fc_error,
                        "cluster_component_sum_max_abs_error_kw": cluster_error,
                        "total_load_identity_max_abs_error_kw": load_error,
                        "fuel_cell_total_kw_range": _range_text(public["fuel_cell_total_kw"]),
                        "battery_cluster_total_kw_range": _range_text(
                            public["battery_cluster_total_kw"]
                        ),
                        "total_load_kw_unclipped_range": _range_text(public["total_load_kw"]),
                        "negative_total_load_rows_clipped": int(public["total_load_kw"].lt(0.0).sum()),
                        **anomaly,
                    }
                )
            except VoyageAlignmentError as error:
                if error.code not in ALLOWED_EXCLUSION_CODES:
                    raise
                error_details = getattr(error, "details", {})
                failed_metadata = error_details.get("required_channel_metadata", {})
                base.update(
                    {
                        "reconstruction_succeeded": False,
                        "usable": False,
                        "reconstruction_status": "unusable_required_channel_source",
                        "exclusion_reason": str(error),
                        "error_code": getattr(error, "code", type(error).__name__),
                        "formal_1s_rows": 0,
                        "common_overlap_start": "",
                        "common_overlap_end": "",
                        "max_required_raw_gap_s": error_details.get(
                            "max_required_raw_gap_s", np.nan
                        ),
                        "max_required_raw_gap_channel": error_details.get(
                            "max_required_raw_gap_channel", ""
                        ),
                        "max_raw_gap_seconds_by_channel_json": (
                            json.dumps(
                                {
                                    key: value["max_raw_gap_s"]
                                    for key, value in failed_metadata.items()
                                },
                                ensure_ascii=False,
                                sort_keys=True,
                            )
                            if failed_metadata
                            else ""
                        ),
                        "required_channel_metadata_json": (
                            json.dumps(failed_metadata, ensure_ascii=False, sort_keys=True)
                            if failed_metadata
                            else ""
                        ),
                        "required_channel_files_json": (
                            json.dumps(
                                error_details.get("required_channel_files", {}),
                                ensure_ascii=False,
                                sort_keys=True,
                            )
                            if error_details.get("required_channel_files")
                            else ""
                        ),
                        "required_channel_hashes_json": (
                            json.dumps(
                                error_details.get("required_channel_hashes", {}),
                                ensure_ascii=False,
                                sort_keys=True,
                            )
                            if error_details.get("required_channel_hashes")
                            else ""
                        ),
                        "required_source_bundle_sha256": error_details.get(
                            "required_source_bundle_sha256", ""
                        ),
                        "optional_source_bundle_sha256": "",
                        "optional_audit_issues_json": "",
                        "fc_component_sum_max_abs_error_kw": np.nan,
                        "cluster_component_sum_max_abs_error_kw": np.nan,
                        "total_load_identity_max_abs_error_kw": np.nan,
                        "fuel_cell_total_kw_range": "",
                        "battery_cluster_total_kw_range": "",
                        "total_load_kw_unclipped_range": "",
                        "negative_total_load_rows_clipped": 0,
                        "old_anomaly_evaluable_count": 0,
                        "old_anomaly_resolved_count": 0,
                        "old_anomaly_source_resolved_count": 0,
                        "old_anomaly_published_target_resolved_count": 0,
                        "old_anomaly_unevaluable_count": int(legacy_mask.sum()),
                        "old_anomaly_disappeared": False,
                        "old_anomaly_reconstructed_fc_kw_range": "",
                        "old_anomaly_reconstructed_cluster_battery_kw_range": "",
                        "old_anomaly_reconstructed_total_load_kw_range": "",
                        "old_anomaly_inverter_kw_range": "",
                        "old_anomaly_soc_pct_range": "",
                        "old_anomaly_soc_delta_pct_by_range": "",
                        "old_anomaly_details_json": "[]",
                        "persistent_inconsistency_count_1s": 0,
                        "persistent_inconsistency_ranges": "",
                        "persistent_inconsistency_details_json": "[]",
                    }
                )
            audit_rows.append(base)

        audit = pd.DataFrame(audit_rows)
        eligible_ids = audit.loc[audit["reconstruction_succeeded"].astype(bool), "voyage_id"].tolist()
        if not eligible_ids:
            raise RuntimeError("No voyage can be reconstructed; existing formal outputs were not replaced")
        split = _chronological_split(eligible_ids)
        split_map = {
            voyage_id: split_name
            for split_name, key in (
                ("train", "train_voyages"),
                ("validation", "validation_voyages"),
                ("test", "test_voyages"),
            )
            for voyage_id in split[key]
        }
        derived_30s = pd.concat(derived_30s_frames, ignore_index=True)
        derived_30s["split"] = derived_30s["voyage_id"].map(split_map)
        formal_30s_temp = _write_csv_temp(derived_30s, formal_30s_csv)
        temporary_paths.append(formal_30s_temp)
        formal_30s_sha256 = _sha256(formal_30s_temp)

        exclusions = audit.loc[
            ~audit["reconstruction_succeeded"].astype(bool), ["voyage_id", "exclusion_reason"]
        ]
        split_payload: dict[str, Any] = {
            **split,
            "train": split["train_voyages"],
            "validation": split["validation_voyages"],
            "test": split["test_voyages"],
            "file_by_voyage": {item.voyage_id: item.excel_path.name for item in inputs},
            "start_time_by_voyage": {
                item.voyage_id: item.start_time.isoformat() for item in inputs
            },
            "source_voyage_count": len(inputs),
            "eligible_voyage_count": len(eligible_ids),
            "split_counts": {
                "train": len(split["train_voyages"]),
                "validation": len(split["validation_voyages"]),
                "test": len(split["test_voyages"]),
            },
            "split_ratio": [7, 2, 1],
            "split_basis": "chronological_by_original_voyage_start_after_channel_reconstruction",
            "random_seed": None,
            "excluded_voyages": exclusions["voyage_id"].tolist(),
            "exclusion_reasons_by_voyage": dict(
                zip(exclusions["voyage_id"].astype(str), exclusions["exclusion_reason"].astype(str))
            ),
            "resolved_by_cluster_reconstruction_voyages": audit.loc[
                audit["old_anomaly_count"].gt(0)
                & audit["old_anomaly_disappeared"].astype(bool),
                "voyage_id",
            ].tolist(),
            "persistent_source_power_inconsistency_voyages": audit.loc[
                audit["reconstruction_status"].eq("persistent_source_power_inconsistency"), "voyage_id"
            ].tolist(),
            "persistent_status_is_audit_only": True,
            "target_load": "load_total_kw",
            "unclipped_identity_load": "total_load_kw",
            "load_definition": LOAD_DEFINITION,
            "load_scope": LOAD_SCOPE,
            "sample_interval_seconds": 1.0,
            "interpolation_method": "natural cubic spline independently on 8 FC powers and 12 precomputed cluster powers",
            "no_extrapolation": True,
            "no_cross_voyage_interpolation": True,
            "battery_power_formula": "-(voltage_v * current_a) / 1000",
            "power_zero_tolerance_kw": POWER_TOLERANCE_KW,
            "formal_30s_csv": formal_30s_csv.relative_to(PROJ).as_posix(),
            "formal_30s_data_origin": DERIVED_30S_ORIGIN,
            "formal_30s_sha256": formal_30s_sha256,
            "formal_1s_directory": (
                Path(spline_output_dir) / "data" / FORMAL_1S_SUBDIRECTORY
            ).relative_to(PROJ).as_posix(),
            "required_source_bundle_sha256_by_voyage": {
                str(row.voyage_id): str(row.required_source_bundle_sha256)
                for row in audit.loc[audit["reconstruction_succeeded"].astype(bool)].itertuples(index=False)
            },
        }
        split_text = json.dumps(split_payload, indent=2, ensure_ascii=False) + "\n"
        split_temp = _write_text_temp(split_text, split_path)
        temporary_paths.append(split_temp)

        audit_temp = _write_csv_temp(audit, audit_path)
        temporary_paths.append(audit_temp)
        audit_sha256 = _sha256(audit_temp)
        report = _build_report(
            audit,
            split_payload,
            formal_30s_csv,
            Path(spline_output_dir) / "data" / FORMAL_1S_SUBDIRECTORY,
        )
        report_temp = _write_text_temp(report, report_final)
        temporary_paths.append(report_temp)
        report_sha256 = _sha256(report_temp)

        transaction = _prepare_publish_transaction(
            marker=transaction_marker,
            formal_dir=formal_dir,
            data_dir=data_dir,
            scalar_paths=scalar_artifact_paths,
        )
        publish_result = publish_formal_voyages(
            staging_dir=staging_dir,
            voyage_records=voyage_records,
            split_json=split_temp,
            output_dir=Path(spline_output_dir),
            retain_backup=True,
        )
        transaction["state"] = "formal_swapped"
        _write_transaction_marker(transaction_marker, transaction)
        for temporary, final_path in (
            (formal_30s_temp, formal_30s_csv),
            (split_temp, split_path),
            (audit_temp, audit_path),
            (report_temp, report_final),
        ):
            temporary.replace(final_path)
        committed_hashes = {
            formal_30s_csv: formal_30s_sha256,
            split_path: publish_result["split_json_sha256"],
            audit_path: audit_sha256,
            report_final: report_sha256,
        }
        for final_path, expected_hash in committed_hashes.items():
            actual_hash = _sha256(final_path)
            if actual_hash != expected_hash:
                raise RuntimeError(
                    f"Committed artifact hash mismatch for {final_path}: "
                    f"expected={expected_hash}, actual={actual_hash}"
                )
        transaction["state"] = "committed"
        _write_transaction_marker(transaction_marker, transaction)
        _cleanup_transaction_backups(
            transaction,
            marker=transaction_marker,
            data_dir=data_dir,
        )
        temporary_paths.clear()
        return {
            "source_voyages": len(inputs),
            "successfully_reconstructed_voyages": len(eligible_ids),
            "excluded_voyages": len(inputs) - len(eligible_ids),
            "train_count": len(split["train_voyages"]),
            "validation_count": len(split["validation_voyages"]),
            "test_count": len(split["test_voyages"]),
            "formal_1s_directory": publish_result["formal_output_dir"],
            "formal_1s_manifest": publish_result["manifest"],
            "formal_30s_csv": str(formal_30s_csv.resolve()),
            "formal_30s_sha256": formal_30s_sha256,
            "split_json": str(split_path.resolve()),
            "audit_csv": str(audit_path.resolve()),
            "report": str(report_final.resolve()),
        }
    except Exception:
        _recover_publish_transaction(
            marker=transaction_marker,
            formal_dir=formal_dir,
            data_dir=data_dir,
            scalar_paths=scalar_artifact_paths,
        )
        _safe_remove_directory(staging_dir, data_dir)
        for path in temporary_paths:
            if path.exists():
                path.unlink()
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build 1 s total load from 8 FC and 12 battery-cluster power channels."
    )
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--raw_root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config_dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--spline_output_dir", type=Path, default=DEFAULT_SPLINE_OUTPUT_DIR)
    parser.add_argument("--audit_csv", type=Path, default=DEFAULT_AUDIT_CSV)
    parser.add_argument("--report_path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--expected_count", type=int, default=66)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = build_dataset(
        input_dir=args.input_dir,
        raw_root=args.raw_root,
        output_dir=args.output_dir,
        config_dir=args.config_dir,
        spline_output_dir=args.spline_output_dir,
        audit_csv=args.audit_csv,
        report_path=args.report_path,
        expected_count=args.expected_count,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
