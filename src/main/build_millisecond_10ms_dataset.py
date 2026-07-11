"""Read and preserve millisecond-condition workbooks without an Excel engine."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import posixpath
import re
import shutil
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import unquote
from zipfile import ZipFile

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ("time_s", "load_kw", "fuel_cell_kw", "battery_kw", "bus_voltage_v")
HEADER_ALIASES = {
    "\u65f6\u95f4_s": "time_s",
    "\u8d1f\u8f7d\u529f\u7387_kW": "load_kw",
    "\u71c3\u6599\u7535\u6c60\u529f\u7387_kW": "fuel_cell_kw",
    "\u9502\u7535\u6c60\u529f\u7387_kW": "battery_kw",
    "\u6bcd\u7ebf\u7535\u538b_V": "bus_voltage_v",
    **{name: name for name in REQUIRED_COLUMNS},
}
FULL_OVERVIEW_NAME = "\u5168\u7a0b\u603b\u89c8"

SPREADSHEET_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
DOCUMENT_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
NS = {"x": SPREADSHEET_NS}
REL_ID_ATTRIBUTE = f"{{{DOCUMENT_REL_NS}}}id"
HASH_BLOCK_BYTES = 1024 * 1024
SOURCE_STEP_S = 0.001
TARGET_STEP_S = 0.010
TIME_ATOL_S = 1e-9
OVERLAP_VALUE_ATOL = 1e-9
HISTORY_STEPS = 30
PREDICTION_STEPS = 6
DATASET_VERSION = "millisecond_load_10ms_direct_decimation_v1"

DEFAULT_SOURCE_PATHS = (
    Path(
        "C:/Users/20883/OneDrive/Desktop/"
        "26.5.24test\u5404\u5de5\u51b5\u6bb5\u6570\u636e+\u603b\u89c8.xlsx"
    ),
    Path(
        "C:/Users/20883/OneDrive/Desktop/"
        "1036\u5404\u5de5\u51b5\u6bb5\u6570\u636e+\u603b\u89c81.xlsx"
    ),
)
DEFAULT_OVERLAP_PAIRS = (
    (
        (DEFAULT_SOURCE_PATHS[0].name, "\u6bb57_277-304s"),
        (DEFAULT_SOURCE_PATHS[0].name, "\u6bb58_290-310s"),
    ),
    (
        (DEFAULT_SOURCE_PATHS[0].name, "\u6bb59_310-327s"),
        (DEFAULT_SOURCE_PATHS[0].name, "\u6bb510_322-333s"),
    ),
)


def _column_ref_to_index(ref: str) -> int:
    match = re.match(r"([A-Za-z]+)", ref)
    if match is None:
        raise ValueError(f"Invalid worksheet cell reference: {ref!r}")
    index = 0
    for char in match.group(1).upper():
        index = index * 26 + (ord(char) - ord("A") + 1)
    return index - 1


def _text_content(node: ET.Element | None) -> str:
    if node is None:
        return ""
    return "".join(node.itertext())


def _read_shared_strings(zf: ZipFile) -> list[str]:
    member = "xl/sharedStrings.xml"
    if member not in zf.namelist():
        return []
    root = ET.fromstring(zf.read(member))
    return [_text_content(item) for item in root.findall("x:si", NS)]


def _relationship_target_to_member(target: str) -> str:
    normalized_target = unquote(target).replace("\\", "/")
    if normalized_target.startswith("/"):
        member = posixpath.normpath(normalized_target.lstrip("/"))
    else:
        member = posixpath.normpath(posixpath.join("xl", normalized_target))
    if member in {"", ".", ".."} or member.startswith("../"):
        raise ValueError(f"Worksheet relationship target escapes the workbook package: {target!r}")
    return member


def _read_worksheet(zf: ZipFile, member: str, shared_strings: list[str]) -> pd.DataFrame:
    root = ET.fromstring(zf.read(member))
    rows: list[list[Any]] = []
    for row in root.findall(".//x:sheetData/x:row", NS):
        values: dict[int, Any] = {}
        for cell in row.findall("x:c", NS):
            ref = cell.attrib.get("r", "")
            if not ref:
                continue
            index = _column_ref_to_index(ref)
            cell_type = cell.attrib.get("t", "")
            value_node = cell.find("x:v", NS)
            if cell_type == "s":
                raw_index = _text_content(value_node).strip()
                if not raw_index:
                    value: Any = ""
                else:
                    try:
                        value = shared_strings[int(raw_index)]
                    except (IndexError, ValueError) as exc:
                        raise ValueError(
                            f"Invalid shared-string index {raw_index!r} in {member} at {ref}"
                        ) from exc
            elif cell_type == "inlineStr":
                value = _text_content(cell.find("x:is", NS))
            else:
                value = _text_content(value_node)
            values[index] = value
        if values:
            width = max(values) + 1
            rows.append([values.get(index, "") for index in range(width)])

    if not rows:
        raise ValueError(f"Worksheet {member} has no rows")
    header = [str(value).strip() for value in rows[0]]
    data: list[list[Any]] = []
    for row in rows[1:]:
        padded = row + [""] * (len(header) - len(row))
        data.append(padded[: len(header)])
    return pd.DataFrame(data, columns=header)


def normalize_sheet_frame(frame: pd.DataFrame, *, source: str, sheet: str) -> pd.DataFrame:
    """Map known headers and return finite numeric required columns."""
    normalized_columns = []
    for column in frame.columns:
        stripped = str(column).strip()
        normalized_columns.append(HEADER_ALIASES.get(stripped, stripped))

    duplicate_required = sorted(
        column for column in REQUIRED_COLUMNS if normalized_columns.count(column) > 1
    )
    if duplicate_required:
        raise ValueError(
            f"{source} sheet {sheet} has duplicate normalized columns: {duplicate_required}"
        )

    normalized = frame.copy()
    normalized.columns = normalized_columns
    missing = [column for column in REQUIRED_COLUMNS if column not in normalized.columns]
    if missing:
        raise ValueError(f"{source} sheet {sheet} missing required columns: {missing}")

    result = normalized.loc[:, list(REQUIRED_COLUMNS)].copy()
    for column in REQUIRED_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    values = result.to_numpy(dtype=float)
    invalid = ~np.isfinite(values)
    if invalid.any():
        invalid_columns = [
            column for idx, column in enumerate(REQUIRED_COLUMNS) if invalid[:, idx].any()
        ]
        raise ValueError(
            f"{source} sheet {sheet} contains missing or non-finite required values "
            f"in columns: {invalid_columns}"
        )
    return result


def read_condition_sheets(
    path: Path,
    *,
    overview_names: set[str],
) -> dict[str, pd.DataFrame]:
    """Read condition sheets in workbook order using workbook relationships."""
    path = Path(path)
    with ZipFile(path) as zf:
        members = set(zf.namelist())
        workbook_member = "xl/workbook.xml"
        rels_member = "xl/_rels/workbook.xml.rels"
        for required_member in (workbook_member, rels_member):
            if required_member not in members:
                raise ValueError(f"{path} does not contain {required_member}")

        workbook = ET.fromstring(zf.read(workbook_member))
        rels = ET.fromstring(zf.read(rels_member))
        targets: dict[str, str] = {}
        for relationship in rels.findall(f"{{{PACKAGE_REL_NS}}}Relationship"):
            rel_id = relationship.attrib.get("Id")
            target = relationship.attrib.get("Target")
            if rel_id and target and relationship.attrib.get("TargetMode") != "External":
                targets[rel_id] = target

        shared_strings = _read_shared_strings(zf)
        frames: dict[str, pd.DataFrame] = {}
        for sheet_node in workbook.findall("x:sheets/x:sheet", NS):
            sheet_name = sheet_node.attrib.get("name", "")
            if sheet_name in overview_names:
                continue
            rel_id = sheet_node.attrib.get(REL_ID_ATTRIBUTE)
            if not rel_id or rel_id not in targets:
                raise ValueError(
                    f"{path} sheet {sheet_name!r} has no internal worksheet relationship"
                )
            member = _relationship_target_to_member(targets[rel_id])
            if member not in members:
                raise ValueError(
                    f"{path} sheet {sheet_name!r} relationship target is missing: {member}"
                )
            raw = _read_worksheet(zf, member, shared_strings)
            frames[sheet_name] = normalize_sheet_frame(
                raw,
                source=str(path),
                sheet=sheet_name,
            )
    return frames


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(HASH_BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_source_with_hash(source: Path, raw_dir: Path) -> dict[str, object]:
    source = Path(source)
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    destination = raw_dir / source.name

    source_hash = sha256_file(source)
    if destination.exists():
        if not destination.is_file() or sha256_file(destination) != source_hash:
            raise FileExistsError(
                f"Destination already exists with different content: {destination}"
            )
    else:
        shutil.copy2(source, destination)

    copied_hash = sha256_file(destination)
    if copied_hash != source_hash:
        raise OSError(f"Copied source hash verification failed: {destination}")

    return {
        "source_path": str(source.resolve()),
        "copied_path": str(destination.resolve()),
        "sha256": source_hash,
        "bytes": source.stat().st_size,
    }


def _source_member(workbook: object, sheet: object, row_index: object) -> str:
    return f"{workbook}|{sheet}|{int(row_index)}"


def direct_decimate(
    frame: pd.DataFrame,
    *,
    factor: int,
    source_workbook: str,
    source_sheet: str,
) -> pd.DataFrame:
    """Retain source rows 0, 10, 20, ... without interpolation or filtering."""
    if factor != 10:
        raise ValueError("Direct 1 ms to 10 ms decimation requires factor=10")
    normalized = normalize_sheet_frame(
        frame,
        source=source_workbook,
        sheet=source_sheet,
    )
    if normalized.empty:
        raise ValueError(f"{source_workbook}/{source_sheet} contains no data rows")

    times = normalized["time_s"].to_numpy(dtype=np.float64)
    differences = np.diff(times)
    if len(differences) and not np.all(differences > 0.0):
        raise ValueError(f"{source_workbook}/{source_sheet} time must be strictly increasing")
    if len(differences) and not np.allclose(
        differences,
        SOURCE_STEP_S,
        rtol=0.0,
        atol=TIME_ATOL_S,
    ):
        raise ValueError(f"{source_workbook}/{source_sheet} is not contiguous 1 ms data")

    source_indices = np.arange(0, len(normalized), factor, dtype=np.int64)
    output = normalized.iloc[source_indices].copy().reset_index(drop=True)
    output.insert(0, "source_row_index", source_indices)
    output.insert(0, "source_sheet", source_sheet)
    output.insert(0, "source_workbook", source_workbook)
    output["source_members"] = [
        _source_member(source_workbook, source_sheet, source_index)
        for source_index in source_indices
    ]

    output_differences = np.diff(output["time_s"].to_numpy(dtype=np.float64))
    if len(output_differences) and not np.allclose(
        output_differences,
        TARGET_STEP_S,
        rtol=0.0,
        atol=TIME_ATOL_S,
    ):
        raise ValueError(
            f"{source_workbook}/{source_sheet} direct decimation did not produce 10 ms data"
        )
    return output


def _prepare_atomic_frame(frame: pd.DataFrame, *, sequence_id: str) -> pd.DataFrame:
    required_metadata = ("source_workbook", "source_sheet", "source_row_index")
    missing = [column for column in required_metadata + REQUIRED_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"{sequence_id} missing atomic-sequence columns: {missing}")

    output = frame.copy().reset_index(drop=True)
    numeric = output.loc[:, list(REQUIRED_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise ValueError(f"{sequence_id} contains non-finite numeric values")
    output.loc[:, list(REQUIRED_COLUMNS)] = numeric

    time_values = output["time_s"].to_numpy(dtype=np.float64)
    time_ms = np.rint(time_values * 1000.0).astype(np.int64)
    if not np.allclose(time_values, time_ms / 1000.0, rtol=0.0, atol=TIME_ATOL_S):
        raise ValueError(f"{sequence_id} time values do not align to integer milliseconds")
    if len(time_ms) > 1 and not np.all(np.diff(time_ms) == 10):
        raise ValueError(f"{sequence_id} is not a contiguous 10 ms sequence")
    output["time_ms"] = time_ms

    if "source_members" not in output:
        output["source_members"] = [
            _source_member(workbook, sheet, row_index)
            for workbook, sheet, row_index in zip(
                output["source_workbook"],
                output["source_sheet"],
                output["source_row_index"],
            )
        ]
    output["sequence_id"] = sequence_id
    return output


def union_sequence_pair(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    sequence_id: str,
) -> pd.DataFrame:
    """Union one declared overlap pair and reject conflicting duplicate rows."""
    left_prepared = _prepare_atomic_frame(left, sequence_id=sequence_id)
    right_prepared = _prepare_atomic_frame(right, sequence_id=sequence_id)
    left_workbooks = set(left_prepared["source_workbook"].astype(str))
    right_workbooks = set(right_prepared["source_workbook"].astype(str))
    if len(left_workbooks) != 1 or left_workbooks != right_workbooks:
        raise ValueError(f"{sequence_id} overlap pair must come from one source workbook")

    common_time_ms = set(left_prepared["time_ms"]) & set(right_prepared["time_ms"])
    if not common_time_ms:
        raise ValueError(f"{sequence_id} declared overlap pair has no common 10 ms rows")

    combined = pd.concat([left_prepared, right_prepared], ignore_index=True)
    merged_rows: list[pd.Series] = []
    for time_ms, group in combined.groupby("time_ms", sort=True):
        values = group.loc[:, list(REQUIRED_COLUMNS)].to_numpy(dtype=np.float64)
        span = np.ptp(values, axis=0)
        if len(group) > 1 and np.any(span > OVERLAP_VALUE_ATOL):
            differing = [
                column
                for column, difference in zip(REQUIRED_COLUMNS, span)
                if difference > OVERLAP_VALUE_ATOL
            ]
            raise ValueError(
                f"{sequence_id} overlap disagreement at {int(time_ms)} ms in {differing}"
            )
        row = group.iloc[0].copy()
        members: list[str] = []
        sheets: list[str] = []
        for value in group["source_members"].astype(str):
            for member in value.split(";"):
                if member and member not in members:
                    members.append(member)
        for value in group["source_sheet"].astype(str):
            for sheet in value.split(";"):
                if sheet and sheet not in sheets:
                    sheets.append(sheet)
        row["source_members"] = ";".join(members)
        row["source_sheet"] = ";".join(sheets)
        row["sequence_id"] = sequence_id
        merged_rows.append(row)

    output = pd.DataFrame(merged_rows).reset_index(drop=True)
    output["time_ms"] = output["time_ms"].astype(np.int64)
    if len(output) > 1 and not np.all(np.diff(output["time_ms"]) == 10):
        raise ValueError(f"{sequence_id} union is not a contiguous 10 ms sequence")
    return output


def _sequence_id_for_single(key: tuple[str, str]) -> str:
    return f"{key[0]}::{key[1]}"


def _sequence_id_for_pair(left: tuple[str, str], right: tuple[str, str]) -> str:
    return f"{left[0]}::{left[1]}+{right[1]}"


def build_atomic_sequences(
    decimated: dict[tuple[str, str], pd.DataFrame],
    *,
    overlap_pairs: tuple[
        tuple[tuple[str, str], tuple[str, str]],
        ...,
    ],
) -> dict[str, pd.DataFrame]:
    """Build declared overlap unions and untouched independent sequences."""
    pair_by_member: dict[tuple[str, str], tuple[tuple[str, str], tuple[str, str]]] = {}
    for pair in overlap_pairs:
        if len(pair) != 2 or pair[0] == pair[1]:
            raise ValueError(f"Invalid overlap pair: {pair!r}")
        for key in pair:
            if key not in decimated:
                raise KeyError(f"Declared overlap member is missing: {key!r}")
            if key in pair_by_member:
                raise ValueError(f"Overlap member appears in more than one pair: {key!r}")
            pair_by_member[key] = pair

    output: dict[str, pd.DataFrame] = {}
    consumed: set[tuple[str, str]] = set()
    for key, frame in decimated.items():
        if key in consumed:
            continue
        pair = pair_by_member.get(key)
        if pair is None:
            sequence_id = _sequence_id_for_single(key)
            atomic = _prepare_atomic_frame(frame, sequence_id=sequence_id)
            consumed.add(key)
        else:
            left_key, right_key = pair
            sequence_id = _sequence_id_for_pair(left_key, right_key)
            atomic = union_sequence_pair(
                decimated[left_key],
                decimated[right_key],
                sequence_id=sequence_id,
            )
            consumed.update(pair)
        if sequence_id in output:
            raise ValueError(f"Duplicate atomic sequence id: {sequence_id}")
        output[sequence_id] = atomic

    seen: dict[tuple[str, int], str] = {}
    for sequence_id, frame in output.items():
        for workbook, time_ms in zip(frame["source_workbook"].astype(str), frame["time_ms"]):
            key = (workbook, int(time_ms))
            previous = seen.get(key)
            if previous is not None and previous != sequence_id:
                raise ValueError(
                    "Found duplicate workbook/time rows across atomic sequences: "
                    f"{key!r} in {previous!r} and {sequence_id!r}"
                )
            seen[key] = sequence_id
    return output


@dataclass(frozen=True)
class SequenceGroup:
    sequence_id: str
    source_workbook: str
    rows: int
    load_mean: float
    load_std: float
    load_q10: float
    load_q50: float
    load_q90: float


def _load_statistics(values: np.ndarray) -> dict[str, float]:
    numeric = np.asarray(values, dtype=np.float64)
    if numeric.ndim != 1 or len(numeric) == 0 or not np.isfinite(numeric).all():
        raise ValueError("Load statistics require a non-empty finite one-dimensional array")
    return {
        "mean": float(np.mean(numeric)),
        "std": float(np.std(numeric, ddof=0)),
        "q10": float(np.quantile(numeric, 0.10)),
        "q50": float(np.quantile(numeric, 0.50)),
        "q90": float(np.quantile(numeric, 0.90)),
        "min": float(np.min(numeric)),
        "max": float(np.max(numeric)),
    }


def _exact_subsets(groups: Sequence[SequenceGroup], target_rows: int) -> list[tuple[int, ...]]:
    matches: list[tuple[int, ...]] = []
    indices = range(len(groups))
    for subset_size in range(1, len(groups) + 1):
        for subset in itertools.combinations(indices, subset_size):
            if sum(groups[index].rows for index in subset) == target_rows:
                matches.append(subset)
    return matches


def _distribution_score(
    split_indices: Mapping[str, tuple[int, ...]],
    *,
    groups: Sequence[SequenceGroup],
    sequence_loads: Mapping[str, np.ndarray],
) -> float:
    global_values = np.concatenate(
        [np.asarray(sequence_loads[group.sequence_id], dtype=np.float64) for group in groups]
    )
    global_stats = _load_statistics(global_values)
    scale = max(global_stats["std"], abs(global_stats["mean"]) * 0.01, 1e-9)
    score = 0.0
    for split_name in ("train", "validation", "test"):
        values = np.concatenate(
            [
                np.asarray(sequence_loads[groups[index].sequence_id], dtype=np.float64)
                for index in split_indices[split_name]
            ]
        )
        stats = _load_statistics(values)
        score += sum(
            abs(stats[name] - global_stats[name]) / scale
            for name in ("mean", "std", "q10", "q50", "q90")
        )
    return float(score)


def allocate_exact_split(
    groups: Sequence[SequenceGroup],
    *,
    sequence_loads: Mapping[str, np.ndarray],
    targets: Mapping[str, int],
    seed: int,
) -> dict[str, list[SequenceGroup]]:
    """Select an exact group-level split without using model or test results."""
    required_splits = ("train", "validation", "test")
    if set(targets) != set(required_splits):
        raise ValueError(f"Split targets must contain exactly {required_splits}")
    if not groups or any(group.rows <= 0 for group in groups):
        raise ValueError("Sequence groups must be non-empty and have positive row counts")
    group_ids = [group.sequence_id for group in groups]
    if len(group_ids) != len(set(group_ids)):
        raise ValueError("Sequence group IDs must be unique")
    if set(sequence_loads) != set(group_ids):
        raise ValueError("sequence_loads keys must exactly match sequence group IDs")
    for group in groups:
        values = np.asarray(sequence_loads[group.sequence_id], dtype=np.float64)
        if values.ndim != 1 or len(values) != group.rows or not np.isfinite(values).all():
            raise ValueError(f"Invalid load array for sequence {group.sequence_id}")
    if sum(int(targets[name]) for name in required_splits) != sum(group.rows for group in groups):
        raise ValueError("Split target rows do not equal total group rows")

    test_subsets = _exact_subsets(groups, int(targets["test"]))
    validation_subsets = _exact_subsets(groups, int(targets["validation"]))
    all_indices = set(range(len(groups)))
    all_workbooks = {group.source_workbook for group in groups}
    if len(all_workbooks) < 2:
        raise ValueError("Exact split requires at least two source workbooks")

    best_key: tuple[float, str] | None = None
    best_indices: dict[str, tuple[int, ...]] | None = None
    for test_indices in test_subsets:
        test_set = set(test_indices)
        for validation_indices in validation_subsets:
            validation_set = set(validation_indices)
            if test_set & validation_set:
                continue
            train_set = all_indices - test_set - validation_set
            if sum(groups[index].rows for index in train_set) != int(targets["train"]):
                continue
            split_indices = {
                "train": tuple(sorted(train_set)),
                "validation": tuple(sorted(validation_set)),
                "test": tuple(sorted(test_set)),
            }
            if any(
                {groups[index].source_workbook for index in split_indices[name]}
                != all_workbooks
                for name in required_splits
            ):
                continue
            score = _distribution_score(
                split_indices,
                groups=groups,
                sequence_loads=sequence_loads,
            )
            identity = "|".join(
                f"{name}:{','.join(groups[index].sequence_id for index in split_indices[name])}"
                for name in required_splits
            )
            tie_break = hashlib.sha256(f"{seed}|{identity}".encode("utf-8")).hexdigest()
            candidate_key = (round(score, 12), tie_break)
            if best_key is None or candidate_key < best_key:
                best_key = candidate_key
                best_indices = split_indices

    if best_indices is None:
        raise ValueError(
            "There is no exact valid assignment with the requested row targets "
            "and source-workbook representation"
        )
    return {
        name: [groups[index] for index in best_indices[name]]
        for name in required_splits
    }


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig", float_format="%.17g")
    temporary.replace(path)


def _group_from_frame(sequence_id: str, frame: pd.DataFrame) -> SequenceGroup:
    workbooks = set(frame["source_workbook"].astype(str))
    if len(workbooks) != 1:
        raise ValueError(f"{sequence_id} must belong to exactly one source workbook")
    stats = _load_statistics(frame["load_kw"].to_numpy(dtype=np.float64))
    return SequenceGroup(
        sequence_id=sequence_id,
        source_workbook=next(iter(workbooks)),
        rows=len(frame),
        load_mean=stats["mean"],
        load_std=stats["std"],
        load_q10=stats["q10"],
        load_q50=stats["q50"],
        load_q90=stats["q90"],
    )


def _window_count(rows: int) -> int:
    return max(rows - HISTORY_STEPS - PREDICTION_STEPS + 1, 0)


def _audit_written_dataset(
    *,
    combined_path: Path,
    split_assignments: Mapping[str, Sequence[str]],
    split_targets: Mapping[str, int],
) -> dict[str, object]:
    combined = pd.read_csv(combined_path)
    if combined.duplicated(["sequence_id", "time_ms"]).any():
        raise ValueError("Written combined dataset has duplicate sequence_id/time_ms rows")
    actual_rows = {
        name: int((combined["split"] == name).sum())
        for name in ("train", "validation", "test")
    }
    expected_rows = {name: int(value) for name, value in split_targets.items()}
    if actual_rows != expected_rows:
        raise ValueError(f"Written split row counts differ: {actual_rows} != {expected_rows}")
    for split_name, sequence_ids in split_assignments.items():
        actual_ids = set(combined.loc[combined["split"] == split_name, "sequence_id"])
        if actual_ids != set(sequence_ids):
            raise ValueError(f"Written {split_name} sequence assignment differs from manifest")
    for sequence_id, frame in combined.groupby("sequence_id", sort=False):
        time_ms = frame["time_ms"].to_numpy(dtype=np.int64)
        if len(time_ms) > 1 and not np.all(np.diff(time_ms) == 10):
            raise ValueError(f"Written sequence {sequence_id} is not contiguous at 10 ms")
    return {
        "passed": True,
        "rows": int(len(combined)),
        "unique_sequence_time_keys": int(
            len(combined.drop_duplicates(["sequence_id", "time_ms"]))
        ),
        "split_rows": actual_rows,
        "time_step_violations": 0,
    }


def build_dataset(
    *,
    source_paths: Sequence[Path],
    raw_root: Path,
    processed_root: Path,
    split_path: Path,
    split_seed: int = 20260710,
    overlap_pairs: tuple[
        tuple[tuple[str, str], tuple[str, str]],
        ...,
    ] | None = None,
    split_targets: Mapping[str, int] | None = None,
) -> dict[str, object]:
    """Copy, decimate, merge, split, write, and re-audit the 10 ms dataset."""
    source_paths = [Path(path) for path in source_paths]
    raw_root = Path(raw_root)
    processed_root = Path(processed_root)
    split_path = Path(split_path)
    if len(source_paths) < 2 or len({path.name for path in source_paths}) != len(source_paths):
        raise ValueError("At least two source workbooks with unique filenames are required")
    for path in source_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    if processed_root.exists() and any(processed_root.iterdir()):
        raise FileExistsError(f"Processed output directory is not empty: {processed_root}")
    if split_path.exists():
        raise FileExistsError(f"Split manifest already exists: {split_path}")

    created_utc = datetime.now(timezone.utc).isoformat()
    source_records = [
        copy_source_with_hash(path, raw_root / "raw")
        for path in source_paths
    ]
    source_manifest_path = raw_root / "source_manifest.json"
    source_manifest = {
        "dataset_version": DATASET_VERSION,
        "created_utc": created_utc,
        "policy": "project copies are byte-identical to supplied source workbooks",
        "sources": source_records,
    }
    write_json(source_manifest_path, source_manifest)

    decimated: dict[tuple[str, str], pd.DataFrame] = {}
    condition_sheets: list[dict[str, object]] = []
    rows_1ms = 0
    rows_10ms = 0
    source_value_mismatch_count = 0
    for record in source_records:
        copied_path = Path(str(record["copied_path"]))
        frames = read_condition_sheets(copied_path, overview_names={FULL_OVERVIEW_NAME})
        for sheet_name, frame in frames.items():
            key = (copied_path.name, sheet_name)
            if key in decimated:
                raise ValueError(f"Duplicate workbook/sheet key: {key!r}")
            sampled = direct_decimate(
                frame,
                factor=10,
                source_workbook=copied_path.name,
                source_sheet=sheet_name,
            )
            selected_source = frame.iloc[
                sampled["source_row_index"].to_numpy(dtype=np.int64)
            ].loc[:, list(REQUIRED_COLUMNS)]
            sampled_values = sampled.loc[:, list(REQUIRED_COLUMNS)]
            if not np.array_equal(
                selected_source.to_numpy(dtype=np.float64),
                sampled_values.to_numpy(dtype=np.float64),
            ):
                source_value_mismatch_count += 1
            decimated[key] = sampled
            rows_1ms += len(frame)
            rows_10ms += len(sampled)
            condition_sheets.append(
                {
                    "source_workbook": copied_path.name,
                    "source_sheet": sheet_name,
                    "rows_1ms": int(len(frame)),
                    "rows_10ms": int(len(sampled)),
                    "time_start_s": float(frame["time_s"].iloc[0]),
                    "time_end_s": float(frame["time_s"].iloc[-1]),
                }
            )
    if source_value_mismatch_count:
        raise ValueError("Direct-decimation source-value equality audit failed")

    pairs = DEFAULT_OVERLAP_PAIRS if overlap_pairs is None else overlap_pairs
    atomic = build_atomic_sequences(decimated, overlap_pairs=pairs)
    unique_rows = sum(len(frame) for frame in atomic.values())
    if split_targets is None:
        split_targets = {
            "train": int(unique_rows * 0.7),
            "validation": int(unique_rows * 0.2),
            "test": unique_rows - int(unique_rows * 0.7) - int(unique_rows * 0.2),
        }
    targets = {name: int(split_targets[name]) for name in ("train", "validation", "test")}

    groups = [_group_from_frame(sequence_id, frame) for sequence_id, frame in atomic.items()]
    sequence_loads = {
        sequence_id: frame["load_kw"].to_numpy(dtype=np.float64)
        for sequence_id, frame in atomic.items()
    }
    allocated = allocate_exact_split(
        groups,
        sequence_loads=sequence_loads,
        targets=targets,
        seed=split_seed,
    )
    assignments = {
        split_name: [group.sequence_id for group in allocated[split_name]]
        for split_name in ("train", "validation", "test")
    }
    split_by_sequence = {
        sequence_id: split_name
        for split_name, sequence_ids in assignments.items()
        for sequence_id in sequence_ids
    }

    processed_root.mkdir(parents=True, exist_ok=True)
    segment_dir = processed_root / "segments"
    segment_dir.mkdir(parents=True, exist_ok=True)
    combined_frames: list[pd.DataFrame] = []
    sequence_records: list[dict[str, object]] = []
    for sequence_index, (sequence_id, frame) in enumerate(atomic.items(), start=1):
        split_name = split_by_sequence[sequence_id]
        output_frame = frame.copy()
        output_frame.insert(0, "split", split_name)
        segment_path = segment_dir / f"sequence_{sequence_index:03d}.csv"
        _write_csv(segment_path, output_frame)
        stats = _load_statistics(output_frame["load_kw"].to_numpy(dtype=np.float64))
        sequence_records.append(
            {
                "sequence_id": sequence_id,
                "split": split_name,
                "source_workbook": str(output_frame["source_workbook"].iloc[0]),
                "source_sheets": sorted(
                    {
                        member.split("|")[1]
                        for value in output_frame["source_members"].astype(str)
                        for member in value.split(";")
                        if member.count("|") >= 2
                    }
                ),
                "rows": int(len(output_frame)),
                "windows_30_to_6": _window_count(len(output_frame)),
                "time_start_s": float(output_frame["time_s"].iloc[0]),
                "time_end_s": float(output_frame["time_s"].iloc[-1]),
                "load_statistics": stats,
                "csv": str(segment_path.resolve()),
                "sha256": sha256_file(segment_path),
            }
        )
        combined_frames.append(output_frame)

    combined = pd.concat(combined_frames, ignore_index=True)
    combined_path = processed_root / "millisecond_load_10ms.csv"
    _write_csv(combined_path, combined)

    split_rows = {
        split_name: int(sum(group.rows for group in allocated[split_name]))
        for split_name in ("train", "validation", "test")
    }
    split_windows = {
        split_name: int(sum(_window_count(group.rows) for group in allocated[split_name]))
        for split_name in ("train", "validation", "test")
    }
    split_manifest = {
        "dataset_version": DATASET_VERSION,
        "created_utc": created_utc,
        "split_seed": int(split_seed),
        "selection_basis": (
            "exact atomic-sequence row totals, both source workbooks in every split, "
            "then load distribution similarity; no model or test metric"
        ),
        "test_results_used_for_split": False,
        "history_steps": HISTORY_STEPS,
        "prediction_steps": PREDICTION_STEPS,
        "sample_interval_ms": 10,
        "scaler_fit_scope": "train_rows_only",
        "window_formula": "max(rows - 30 - 6 + 1, 0) per atomic sequence",
        "assignments": assignments,
        "split_rows": split_rows,
        "split_windows": split_windows,
        "source_workbooks_by_split": {
            split_name: sorted({group.source_workbook for group in allocated[split_name]})
            for split_name in ("train", "validation", "test")
        },
        "sequence_sha256": {
            record["sequence_id"]: record["sha256"]
            for record in sequence_records
        },
    }
    write_json(split_path, split_manifest)

    written_audit = _audit_written_dataset(
        combined_path=combined_path,
        split_assignments=assignments,
        split_targets=targets,
    )
    dataset_manifest = {
        "dataset_version": DATASET_VERSION,
        "created_utc": created_utc,
        "sample_interval_ms": 10,
        "history_steps": HISTORY_STEPS,
        "prediction_steps": PREDICTION_STEPS,
        "scaler_fit_scope": "train_rows_only",
        "target_column": "load_kw",
        "feature_columns": ["load_kw"],
        "direct_decimation": {
            "method": "direct row selection without interpolation, averaging, or filtering",
            "source_interval_ms": 1,
            "factor": 10,
            "source_indices": "0,10,20,...",
            "source_value_mismatch_count": source_value_mismatch_count,
            "aliasing_caveat": "frequencies above the new 50 Hz Nyquist limit may alias",
        },
        "source_manifest": str(source_manifest_path.resolve()),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "condition_sheets": condition_sheets,
        "condition_sheet_count": int(len(condition_sheets)),
        "rows_1ms": int(rows_1ms),
        "rows_10ms_before_overlap_removal": int(rows_10ms),
        "unique_rows_10ms": int(unique_rows),
        "atomic_sequence_count": int(len(atomic)),
        "atomic_sequences": sequence_records,
        "combined_csv": str(combined_path.resolve()),
        "combined_csv_sha256": sha256_file(combined_path),
        "split_manifest": str(split_path.resolve()),
        "split_manifest_sha256": sha256_file(split_path),
        "split_rows": split_rows,
        "written_artifact_audit": written_audit,
        "online_energy_management_use": False,
        "sensor_provenance_verified": False,
    }
    write_json(processed_root / "dataset_manifest.json", dataset_manifest)

    return {
        "dataset_version": DATASET_VERSION,
        "condition_sheets": int(len(condition_sheets)),
        "rows_1ms": int(rows_1ms),
        "rows_10ms_before_overlap_removal": int(rows_10ms),
        "unique_rows_10ms": int(unique_rows),
        "atomic_sequences": int(len(atomic)),
        "split_rows": split_rows,
        "split_windows": split_windows,
        "dataset_manifest": str((processed_root / "dataset_manifest.json").resolve()),
        "split_manifest": str(split_path.resolve()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the direct-decimated 10 ms forecasting dataset."
    )
    parser.add_argument("--source", action="append", type=Path)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/millisecond_1ms"),
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("data/millisecond_10ms"),
    )
    parser.add_argument(
        "--split-path",
        type=Path,
        default=Path("outputs/config/millisecond_10ms_split_721.json"),
    )
    parser.add_argument("--split-seed", type=int, default=20260710)
    return parser


def _validate_known_production_counts(summary: Mapping[str, object]) -> None:
    expected = {
        "condition_sheets": 21,
        "rows_1ms": 339000,
        "rows_10ms_before_overlap_removal": 33900,
        "unique_rows_10ms": 32000,
        "atomic_sequences": 19,
        "split_rows": {"train": 22400, "validation": 6400, "test": 3200},
    }
    differences = {
        key: {"actual": summary.get(key), "expected": value}
        for key, value in expected.items()
        if summary.get(key) != value
    }
    if differences:
        raise ValueError(f"Known production count audit failed: {differences}")


def main() -> None:
    args = build_parser().parse_args()
    sources = tuple(args.source) if args.source else DEFAULT_SOURCE_PATHS
    summary = build_dataset(
        source_paths=sources,
        raw_root=args.raw_root,
        processed_root=args.processed_root,
        split_path=args.split_path,
        split_seed=args.split_seed,
    )
    if {path.name for path in sources} == {path.name for path in DEFAULT_SOURCE_PATHS}:
        _validate_known_production_counts(summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
