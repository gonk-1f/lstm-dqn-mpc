from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from build_millisecond_10ms_dataset import (
    FULL_OVERVIEW_NAME,
    read_condition_sheets,
    sha256_file,
    write_json,
)


VALUE_COLUMNS = ("time_s", "load_kw", "fuel_cell_kw", "battery_kw", "bus_voltage_v")


def audit_member_rows(
    combined: pd.DataFrame,
    source_frames: Mapping[tuple[str, str], pd.DataFrame],
) -> dict[str, int]:
    checked = 0
    mismatches: list[str] = []
    for output_index, output in combined.iterrows():
        for encoded_member in str(output["source_members"]).split(";"):
            workbook, sheet, source_index_text = encoded_member.rsplit("|", 2)
            source_index = int(source_index_text)
            if source_index % 10 != 0:
                raise ValueError(
                    f"Source member {encoded_member!r} row index is not divisible by 10"
                )
            key = (workbook, sheet)
            if key not in source_frames:
                raise ValueError(f"Source member references an unknown sheet: {key!r}")
            source = source_frames[key]
            if source_index < 0 or source_index >= len(source):
                raise ValueError(f"Source member row is out of range: {encoded_member!r}")
            source_row = source.iloc[source_index]
            expected_time_ms = int(round(float(source_row["time_s"]) * 1000.0))
            if int(output["time_ms"]) != expected_time_ms:
                mismatches.append(f"row {output_index} time_ms")
            for column in VALUE_COLUMNS:
                if float(output[column]) != float(source_row[column]):
                    mismatches.append(f"row {output_index} {column}")
            checked += 1
    if mismatches:
        preview = ", ".join(mismatches[:10])
        raise ValueError(f"Independent source-value audit found {len(mismatches)} mismatches: {preview}")
    return {"checked_members": checked, "mismatch_count": 0}


def audit_dataset(
    *,
    dataset_root: Path,
    split_path: Path,
    audit_path: Path,
    attach_to_manifest: bool,
) -> dict[str, object]:
    dataset_manifest_path = dataset_root / "dataset_manifest.json"
    combined_path = dataset_root / "millisecond_load_10ms.csv"
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    split_manifest = json.loads(split_path.read_text(encoding="utf-8"))
    source_manifest_path = Path(dataset_manifest["source_manifest"])
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))

    source_hash_rows: list[dict[str, object]] = []
    source_frames: dict[tuple[str, str], pd.DataFrame] = {}
    for source in source_manifest["sources"]:
        original = Path(source["source_path"])
        copied = Path(source["copied_path"])
        original_hash = sha256_file(original)
        copied_hash = sha256_file(copied)
        declared_hash = str(source["sha256"])
        if original_hash != copied_hash or original_hash != declared_hash:
            raise ValueError(f"Source/copy hash mismatch for {original}")
        source_hash_rows.append(
            {
                "source_path": str(original),
                "copied_path": str(copied),
                "declared_sha256": declared_hash,
                "original_sha256": original_hash,
                "copied_sha256": copied_hash,
                "match": True,
            }
        )
        for sheet, frame in read_condition_sheets(
            copied, overview_names={FULL_OVERVIEW_NAME}
        ).items():
            source_frames[(original.name, sheet)] = frame

    combined = pd.read_csv(combined_path, float_precision="round_trip")
    if combined.duplicated(["sequence_id", "time_ms"]).any():
        raise ValueError("Duplicate sequence_id/time_ms keys found")
    manifest_sequences = {
        str(sequence["sequence_id"]): sequence
        for sequence in dataset_manifest["atomic_sequences"]
    }
    if len(manifest_sequences) != len(dataset_manifest["atomic_sequences"]):
        raise ValueError("Duplicate sequence_id entries found in dataset manifest")
    time_step_violations: list[str] = []
    window_counts: dict[str, int] = {"train": 0, "validation": 0, "test": 0}
    sequence_windows: dict[str, int] = {}
    for sequence_id, frame in combined.groupby("sequence_id", sort=False):
        times = frame.sort_values("time_ms")["time_ms"].to_numpy(dtype=np.int64)
        if times.size > 1 and not np.all(np.diff(times) == 10):
            time_step_violations.append(str(sequence_id))
        split_names = frame["split"].unique().tolist()
        if len(split_names) != 1:
            raise ValueError(f"Atomic sequence crosses splits: {sequence_id}")
        sequence_key = str(sequence_id)
        windows = max(len(frame) - 30 - 6 + 1, 0)
        sequence_windows[sequence_key] = windows
        if sequence_key not in manifest_sequences:
            raise ValueError(f"Atomic sequence missing from dataset manifest: {sequence_key}")
        declared_windows = int(manifest_sequences[sequence_key]["windows_30_to_6"])
        if windows != declared_windows:
            raise ValueError(
                f"Window count differs for sequence {sequence_key}: "
                f"computed={windows}, declared={declared_windows}"
            )
        window_counts[str(split_names[0])] += windows
    if time_step_violations:
        raise ValueError(f"10 ms time-step violations: {time_step_violations}")
    missing_sequences = sorted(set(manifest_sequences) - set(sequence_windows))
    if missing_sequences:
        raise ValueError(f"Dataset manifest sequences missing from combined CSV: {missing_sequences}")

    actual_assignments = {
        split: sorted(combined.loc[combined["split"] == split, "sequence_id"].unique().tolist())
        for split in ("train", "validation", "test")
    }
    for split, expected in split_manifest["assignments"].items():
        if actual_assignments[split] != sorted(expected):
            raise ValueError(f"Split assignment mismatch for {split}")
    assignment_sets = {name: set(values) for name, values in actual_assignments.items()}
    intersections = {
        "train_validation": sorted(assignment_sets["train"] & assignment_sets["validation"]),
        "train_test": sorted(assignment_sets["train"] & assignment_sets["test"]),
        "validation_test": sorted(assignment_sets["validation"] & assignment_sets["test"]),
    }
    if any(intersections.values()):
        raise ValueError(f"Split intersections are non-empty: {intersections}")
    if window_counts != {key: int(value) for key, value in split_manifest["split_windows"].items()}:
        raise ValueError(f"Window counts differ: {window_counts}")
    workbook_presence = {
        split: sorted(combined.loc[combined["split"] == split, "source_workbook"].unique().tolist())
        for split in ("train", "validation", "test")
    }
    expected_workbooks = sorted(Path(item["source_path"]).name for item in source_manifest["sources"])
    if any(values != expected_workbooks for values in workbook_presence.values()):
        raise ValueError(f"Both source workbooks are not present in every split: {workbook_presence}")

    member_audit = audit_member_rows(combined, source_frames)
    sequence_hashes: list[dict[str, object]] = []
    for sequence in dataset_manifest["atomic_sequences"]:
        path = Path(sequence["csv"])
        actual_hash = sha256_file(path)
        if actual_hash != sequence["sha256"]:
            raise ValueError(f"Atomic sequence hash mismatch: {path}")
        sequence_hashes.append(
            {"sequence_id": sequence["sequence_id"], "sha256": actual_hash, "match": True}
        )
    result: dict[str, object] = {
        "audit_version": "millisecond_10ms_independent_audit_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "source_copy_hashes": source_hash_rows,
        "combined_csv_sha256": sha256_file(combined_path),
        "rows": int(len(combined)),
        "unique_sequence_time_keys": int(
            len(combined.drop_duplicates(["sequence_id", "time_ms"]))
        ),
        "atomic_sequence_count": int(combined["sequence_id"].nunique()),
        "time_step_violations": time_step_violations,
        "split_intersections": intersections,
        "split_rows": {
            split: int((combined["split"] == split).sum())
            for split in ("train", "validation", "test")
        },
        "split_windows": window_counts,
        "sequence_windows": sequence_windows,
        "source_workbooks_by_split": workbook_presence,
        "direct_source_member_audit": member_audit,
        "atomic_sequence_hashes": sequence_hashes,
    }
    write_json(audit_path, result)
    if attach_to_manifest:
        dataset_manifest["independent_audit"] = str(audit_path.resolve())
        dataset_manifest["independent_audit_sha256"] = sha256_file(audit_path)
        write_json(dataset_manifest_path, dataset_manifest)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Independently audit retained 10 ms artifacts")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/millisecond_10ms"))
    parser.add_argument(
        "--split-path", type=Path, default=Path("outputs/config/millisecond_10ms_split_721.json")
    )
    parser.add_argument("--audit-path", type=Path)
    parser.add_argument("--attach-to-manifest", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    audit_path = args.audit_path or args.dataset_root / "independent_audit.json"
    result = audit_dataset(
        dataset_root=args.dataset_root.resolve(),
        split_path=args.split_path.resolve(),
        audit_path=audit_path.resolve(),
        attach_to_manifest=args.attach_to_manifest,
    )
    print(f"passed={result['passed']}")
    print(f"rows={result['rows']}")
    print(f"unique_sequence_time_keys={result['unique_sequence_time_keys']}")
    print(f"atomic_sequence_count={result['atomic_sequence_count']}")
    print(f"checked_source_members={result['direct_source_member_audit']['checked_members']}")
    print(f"audit_path={audit_path.resolve()}")


if __name__ == "__main__":
    main()
