"""Versioned, label-only v4 mode sidecar on the frozen 30-second axis."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path


FC_ZERO_KW = 1.0
BATTERY_CHARGE_KW = -1.0
SPEED_ZERO_KN = 0.1
BALANCE_DEADBAND_KW = 1.0
SAMPLE_SECONDS = 30.0
MIN_SHORE_SAMPLES = 3
SCHEMA = "v4_mode_labels_30s_v3"
OLD_ROOT_NAME = "operating_dataset_zero_boundary_v2_modes"
NEW_ROOT_NAME = "operating_dataset_zero_boundary_v2_modes_v3"

# A targeted raw-data check found insufficient eight-module FC snapshot coverage
# for these Train spans; an interpolated zero alone cannot establish shore mode.
# They must not become modeled SHORE merely through interpolation.
UNVERIFIED_RAW_FC_TRAIN = {
    "zero_boundary_007": ((5, 10),),
    "zero_boundary_017": ((2, 15),),
    "zero_boundary_037": ((3812, 3828),),
    "zero_boundary_041": ((1, 13),),
    "zero_boundary_042": ((5, 10),),
    "zero_boundary_046": ((2, 14),),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _valid(row: dict[str, str], unverified: bool) -> bool:
    try:
        return (
            not unverified
            and row["channels_complete"] == "True"
            and row["freshness_valid"] == "True"
            and row["duplicate_conflict"] == "False"
            and all(math.isfinite(float(row[key])) for key in (
                "p_fc_total_kw", "p_batt_bus_kw", "speed_kn",
                "load_total_kw", "component_balance_residual_kw",
            ))
        )
    except (KeyError, TypeError, ValueError):
        return False


def classify_rows(rows: list[dict[str, str]], *, sample_id: str, split: str) -> list[dict[str, str]]:
    """Classify existing measurements; pending cannot charge before confirmation."""
    unverified_ranges = UNVERIFIED_RAW_FC_TRAIN.get(sample_id, ()) if split == "train" else ()
    result: list[dict[str, str]] = []
    candidate: list[int] = []
    previous_time: datetime | None = None

    def finish_candidate() -> None:
        if not candidate:
            return
        if len(candidate) >= MIN_SHORE_SAMPLES:
            for offset, index in enumerate(candidate):
                result[index]["mode"] = "shore_pending" if offset < 2 else "shore_charging"
                result[index]["mode_reason"] = (
                    "CAUSAL_ZERO_FC_CHARGE_PENDING" if offset < 2 else
                    "MODELED_SHORE_CONFIRMED_AT_THIRD_SAMPLE"
                )
        else:
            for index in candidate:
                result[index]["mode_reason"] = "UNCONFIRMED_SHORT_ZERO_FC_CHARGE"
        candidate.clear()

    for index, row in enumerate(rows):
        stamp = datetime.fromisoformat(row["timestamp"])
        if previous_time is not None and (stamp - previous_time).total_seconds() != SAMPLE_SECONDS:
            finish_candidate()
        previous_time = stamp
        unverified = any(start <= index < end for start, end in unverified_ranges)
        entry = {
            "timestamp": row["timestamp"], "time_s": row["time_s"],
            "mode": "unknown", "mode_reason": "INVALID_OR_CONFLICTING_POWER_EVIDENCE",
            "load_override_kw": "",
        }
        result.append(entry)
        if not _valid(row, unverified):
            finish_candidate()
            if unverified:
                entry["mode_reason"] = "RAW_EIGHT_MODULE_FC_SNAPSHOT_UNVERIFIED"
            continue
        fc, battery, speed, load = (float(row[key]) for key in (
            "p_fc_total_kw", "p_batt_bus_kw", "speed_kn", "load_total_kw"))
        if fc < -FC_ZERO_KW or speed < 0:
            finish_candidate()
            continue
        reconstructed = fc + battery
        if (fc <= FC_ZERO_KW and battery < BATTERY_CHARGE_KW
                and speed <= SPEED_ZERO_KN
                and abs(float(row["component_balance_residual_kw"])) <= BALANCE_DEADBAND_KW):
            candidate.append(index)
            entry["mode_reason"] = "ZERO_FC_CHARGE_AWAITING_THIRD_SAMPLE"
            continue
        finish_candidate()
        if load < -BALANCE_DEADBAND_KW or reconstructed < -BALANCE_DEADBAND_KW:
            entry["mode_reason"] = "NEGATIVE_UNEXPLAINED_NET_LOAD"
        elif fc > FC_ZERO_KW:
            entry.update(mode="onboard", mode_reason="VALID_POSITIVE_FC_SELF_SUPPLY")
        elif battery >= BATTERY_CHARGE_KW:
            # The zero-load departure boundary and battery-only supply remain
            # onboard; a zero FC measurement alone does not imply shore power.
            entry.update(mode="onboard", mode_reason="VALID_BATTERY_ONLY_OR_ZERO_LOAD")
        else:
            entry["mode_reason"] = "ZERO_FC_CHARGING_WITHOUT_SHORE_SIGNATURE"
    finish_candidate()
    return result


def build_labels(old_root: Path, output_root: Path) -> dict[str, object]:
    old_root, output_root = old_root.resolve(), output_root.resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    source_manifest = old_root / "metadata" / "sample_manifest.csv"
    source = _read_csv(source_manifest)
    if Counter(row["split"] for row in source) != {"train": 30, "validation": 8, "test": 5}:
        raise ValueError("source split differs from frozen 30/8/5 manifest")
    output_root.mkdir(parents=True)
    (output_root / "metadata").mkdir()
    for split in ("train", "validation", "test"):
        (output_root / split).mkdir()
    records = []
    counts = Counter()
    by_split = {split: Counter() for split in ("train", "validation", "test")}
    for identity in source:
        source_path = old_root / identity["relative_path"]
        if _sha256(source_path) != identity["sha256"]:
            raise ValueError(f"{identity['sample_id']}: source mode hash mismatch")
        original = _read_csv(source_path)
        if len(original) != int(identity["supervisory_row_count"]):
            raise ValueError(f"{identity['sample_id']}: source row count mismatch")
        labels = classify_rows(original, sample_id=identity["sample_id"], split=identity["split"])
        target = output_root / identity["relative_path"]
        with target.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=(
                "timestamp", "time_s", "mode", "mode_reason", "load_override_kw"))
            writer.writeheader()
            writer.writerows(labels)
        local = Counter(row["mode"] for row in labels)
        counts.update(local)
        by_split[identity["split"]].update(local)
        records.append({
            **{key: identity[key] for key in ("parent", "sample_id", "relative_path", "split")},
            "supervisory_row_count": len(labels),
            "unresolved_row_count": local["unknown"],
            "sha256": _sha256(target),
        })
    with (output_root / "metadata" / "sample_manifest.csv").open(
        "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(records[0]))
        writer.writeheader()
        writer.writerows(records)
    policy = {
        "schema_version": SCHEMA,
        "source_mode_root": f"../{old_root.name}",
        "source_mode_manifest_sha256": _sha256(source_manifest),
        "rule": {
            "fc_zero_kw": FC_ZERO_KW, "battery_charge_below_kw": BATTERY_CHARGE_KW,
            "speed_at_most_kn": SPEED_ZERO_KN, "sample_seconds": SAMPLE_SECONDS,
            "minimum_confirmed_samples": MIN_SHORE_SAMPLES,
            "balance_deadband_kw": BALANCE_DEADBAND_KW,
            "pending_behavior": "no DQN action or SOC recharge before third sample",
            "shore_meaning": "modeled charging opportunity, not measured grid connection",
            "unknown_behavior": "truncate sample without charging or supply-failure penalty",
        },
        "raw_fc_unverified_train_intervals": {
            key: [list(pair) for pair in ranges]
            for key, ranges in UNVERIFIED_RAW_FC_TRAIN.items()
        },
        "mode_counts": dict(counts),
        "split_mode_counts": {key: dict(value) for key, value in by_split.items()},
    }
    (output_root / "metadata" / "policy.json").write_text(
        json.dumps(policy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return policy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-root", type=Path, default=Path("data/processed") / OLD_ROOT_NAME)
    parser.add_argument("--output-root", type=Path, default=Path("data/processed") / NEW_ROOT_NAME)
    args = parser.parse_args()
    print(json.dumps(build_labels(args.old_root, args.output_root)["split_mode_counts"],
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
