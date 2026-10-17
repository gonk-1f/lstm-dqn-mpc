"""Freeze a Train-only exclusion as a new split version, using metadata only."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data/processed/operating_dataset_independent_v2"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _contained(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root):
        raise ValueError("metadata path escapes dataset root")
    return path


def _records(raw: bytes) -> tuple[bytes, list[tuple[dict, bytes]]]:
    """Keep each frozen CSV record's bytes; reject unsupported multiline records."""
    parsed = list(csv.reader(io.StringIO(raw.decode("utf-8-sig"), newline="")))
    lines = raw.splitlines(keepends=True)
    if not parsed or len(parsed) != len(lines) or any(len(row) != len(parsed[0]) for row in parsed[1:]):
        raise ValueError("source CSV must contain one complete record per line")
    return lines[0], [(dict(zip(parsed[0], row)), line) for row, line in zip(parsed[1:], lines[1:])]


def _save_rows(path: Path, rows: list[dict], columns: list[str]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    path.write_bytes(buffer.getvalue().encode("utf-8-sig"))


def freeze_exclusion(dataset_root: Path, sample_id: str, new_version: str, reason: str) -> dict:
    root = Path(dataset_root).resolve()
    if not re.fullmatch(r"splits_[A-Za-z0-9_-]+", new_version):
        raise ValueError("new split version must be a simple splits_* directory name")
    output = root / "metadata" / new_version
    if output.exists():
        raise FileExistsError(f"frozen version already exists: {output}")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("an explicit user exclusion reason is required")
    active_path = root / "metadata/active_split.json"
    active = json.loads(active_path.read_text(encoding="utf-8"))
    source_manifest = _contained(root, active["manifest_relative_path"])
    source_policy_path = _contained(root, active["policy_relative_path"])
    source_policy = json.loads(source_policy_path.read_text(encoding="utf-8"))
    if (_sha(source_manifest) != active["manifest_sha256"]
            or _sha(source_manifest) != source_policy["sample_manifest_sha256"]):
        raise ValueError("source split manifest SHA-256 mismatch")
    source_directory = source_manifest.parent
    source_payloads = source_directory / "payload_hashes.csv"
    if _sha(source_payloads) != source_policy["payload_hash_manifest_sha256"]:
        raise ValueError("source payload hash manifest SHA-256 mismatch")
    header, records = _records(source_manifest.read_bytes())
    matches = [(row, raw) for row, raw in records if row["candidate_id"] == sample_id]
    if len(matches) != 1 or matches[0][0]["split"] != "train":
        raise ValueError("exclusions require exactly one currently assigned Train sample")
    excluded = matches[0][0]
    kept = [(row, raw) for row, raw in records if row["candidate_id"] != sample_id]
    counts = {split: sum(row["split"] == split for row, _ in kept)
              for split in ("train", "validation", "test")}
    expected_counts = dict(active["split_counts"])
    expected_counts["train"] -= 1
    if counts != expected_counts or any(count <= 0 for count in counts.values()):
        raise ValueError("exclusion must leave all other assignments and nonempty splits intact")
    payload_header, payload_records = _records(source_payloads.read_bytes())
    removed_payloads = [row for row, _ in payload_records if row["candidate_id"] == sample_id]
    if len(removed_payloads) != 2 or {row["kind"] for row in removed_payloads} != {
            "canonical_1s", "existing_30s_preview"}:
        raise ValueError("excluded sample must have both frozen payload identities")
    source_snapshot = {path.relative_to(source_directory).as_posix(): _sha(path)
                       for path in source_directory.rglob("*") if path.is_file()}
    parent_rows = []
    for parent, split in sorted({(row["parent_sample_id"], row["split"]) for row, _ in kept}):
        members = [row for row, _ in kept if row["parent_sample_id"] == parent and row["split"] == split]
        parent_rows.append({"parent_sample_id": parent, "split": split, "candidate_count": len(members),
                            "duration_s": sum(int(row["duration_s"]) for row in members)})
    exclusion = excluded | {
        "excluded_from_split_version": active["split_version"], "new_split_version": new_version,
        "exclusion_reason": reason, "authorization": "explicit_user_instruction",
        "payloads_retained": "true", "source_manifest_sha256": active["manifest_sha256"],
    }
    output.mkdir()
    (output / "sample_manifest.csv").write_bytes(header + b"".join(raw for _, raw in kept))
    (output / "payload_hashes.csv").write_bytes(
        payload_header + b"".join(raw for row, raw in payload_records if row["candidate_id"] != sample_id))
    for split in counts:
        split_rows = [(row, raw) for row, raw in kept if row["split"] == split]
        split_source = source_directory / f"{split}.csv"
        if split != "train" and split_source.is_file():
            # Preserve Validation/Test metadata byte-for-byte, without accessing payloads.
            (output / f"{split}.csv").write_bytes(split_source.read_bytes())
        else:
            (output / f"{split}.csv").write_bytes(header + b"".join(raw for _, raw in split_rows))
        ids_source = source_directory / f"{split}_ids.txt"
        ids = "\n".join(row["candidate_id"] for row, _ in split_rows).encode("utf-8")
        (output / f"{split}_ids.txt").write_bytes(
            ids_source.read_bytes() if split != "train" and ids_source.is_file() else ids)
    _save_rows(output / "parent_assignments.csv", parent_rows,
               ["parent_sample_id", "split", "candidate_count", "duration_s"])
    _save_rows(output / "exclusions.csv", [exclusion], list(exclusion))
    summary = []
    for split in counts:
        members = [row for row, _ in kept if row["split"] == split]
        durations = [int(row["duration_s"]) for row in members]
        item = {"split": split, "candidate_count": len(members),
                "parent_count": len({row["parent_sample_id"] for row in members}),
                "duration_h": sum(durations) / 3600, "minimum_duration_h": min(durations) / 3600,
                "maximum_duration_h": max(durations) / 3600}
        if "load_peak_kw" in members[0]:
            item.update(peak_min_kw=min(float(row["load_peak_kw"]) for row in members),
                        peak_max_kw=max(float(row["load_peak_kw"]) for row in members),
                        nonzero_boundary_B_count=sum(row["quality_class"] == "B" for row in members))
        summary.append(item)
    _save_rows(output / "split_summary.csv", summary, list(summary[0]))
    policy = source_policy | {
        "split_version": new_version, "source_split_version": active["split_version"],
        "source_split_manifest_sha256": active["manifest_sha256"],
        "source_split_policy_sha256": _sha(source_policy_path),
        "source_version_files_sha256": source_snapshot,
        "source_split_algorithm": source_policy["algorithm"] if "algorithm" in source_policy else None,
        "algorithm": "inherit frozen source assignments; explicitly exclude one Train sample; no resampling",
        "split_counts": counts,
        "parent_split_counts": {split: sum(row["split"] == split for row in parent_rows) for split in counts},
        "sample_manifest_sha256": _sha(output / "sample_manifest.csv"),
        "payload_hash_manifest_sha256": _sha(output / "payload_hashes.csv"),
        "checked_payload_hashes": 0, "payload_hash_count": len(payload_records) - 2,
        "payload_hash_verification": "inherited frozen metadata; no payloads opened for this exclusion",
        "environment_integration_status": "INDEPENDENT_TASK_CONTRACT_IN_RUNTIME_METADATA",
        "economic_model_status": "ZHOU_SOC_EQUIVALENT_HYDROGEN_V1",
        "limitations": [
            "Fixed user-selected Test must not be used for hyperparameter selection.",
            "Existing processed interpolation and boundary provenance is inherited unchanged; no raw reconstruction.",
            "Canonical signed load values and nonzero boundaries remain unchanged; runtime records raw loads and applies the existing [-1,0) kW control deadband, rejecting loads below -1 kW.",
            "Execution uses existing canonical values at t=30*k<duration_s with the exact final partial duration; the original endpoint is a boundary only, without extra execution or zero forcing.",
        ],
        "exclusions_relative_path": f"metadata/{new_version}/exclusions.csv",
        "exclusions_sha256": _sha(output / "exclusions.csv"),
        "excluded_candidate_ids": [sample_id], "exclusion_reason": reason,
        "payloads_deleted": False, "remaining_sample_manifest_record_bytes_unchanged": True,
        "validation_and_test_assignments_unchanged": True, "resampling_performed": False,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output / "policy.json").write_text(json.dumps(policy, ensure_ascii=False, indent=2), encoding="utf-8")
    if any(_sha(source_directory / name) != digest for name, digest in source_snapshot.items()):
        raise RuntimeError("source frozen version changed during exclusion")
    new_active = active | {
        "split_version": new_version, "manifest_relative_path": f"metadata/{new_version}/sample_manifest.csv",
        "manifest_sha256": policy["sample_manifest_sha256"],
        "policy_relative_path": f"metadata/{new_version}/policy.json", "split_counts": counts,
        "status": "SPLIT_FROZEN",
    }
    temporary = active_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(new_active, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(active_path)
    return {"split_version": new_version, "split_counts": counts, "excluded_candidate_id": sample_id,
            "manifest_sha256": policy["sample_manifest_sha256"],
            "payload_hash_manifest_sha256": policy["payload_hash_manifest_sha256"],
            "exclusions_sha256": policy["exclusions_sha256"], "source_version_unchanged": True,
            "payloads_opened": 0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--exclude-sample", required=True)
    parser.add_argument("--new-version", required=True)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()
    print(json.dumps(freeze_exclusion(args.dataset_root, args.exclude_sample, args.new_version, args.reason),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
