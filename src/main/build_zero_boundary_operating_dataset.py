"""Build the versioned zero-boundary v2 operating dataset from raw telemetry."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Callable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.final_dataset_source import parent_sort_key  # noqa: E402
from v2.data.power_gap_interpolation import interpolate_power_gaps  # noqa: E402
from v2.data.segment_power_source import (  # noqa: E402
    ParentPowerSeries,
    load_parent_power_series,
)
from v2.data.zero_boundary_dataset import (  # noqa: E402
    ACTIVE_THRESHOLD_KW,
    APPROVED_BOUNDARY_EXCLUSIONS,
    FIXED_TEST_PARENTS,
    SUSTAINED_POINTS,
    TEST_COUNT,
    TRAIN_COUNT,
    VALIDATION_COUNT,
    ZERO_DEADBAND_KW,
    assign_parent_splits,
    reconstruct_one_second,
    segment_features,
    trim_to_zero_boundaries,
)


NOMINAL_STEP_SECONDS = 30.0
ALIGNMENT_TOLERANCE_SECONDS = 10.0
EXPECTED_RAW_PARENT_COUNT = 66
EXPECTED_INCLUDED_PARENT_COUNT = TRAIN_COUNT + VALIDATION_COUNT + TEST_COUNT
EXPECTED_EXCLUDED_PARENT_COUNT = len(APPROVED_BOUNDARY_EXCLUSIONS)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def discover_parent_ids(raw_root: Path) -> list[str]:
    """Return recognizable parent folders containing both FC and BMS telemetry."""
    root = Path(raw_root)
    if not root.is_dir():
        raise FileNotFoundError(f"raw telemetry root is missing: {root}")
    return [
        path.name
        for path in root.iterdir()
        if path.is_dir()
        and (path / "燃料电池系统").is_dir()
        and (path / "BMS").is_dir()
    ]


def _boundary_row(parent: str, split: str, boundary: object) -> dict[str, object]:
    row = asdict(boundary)
    row["parent"] = parent
    row["split"] = split
    for key in (
        "left_timestamp",
        "right_timestamp",
        "unrounded_boundary_timestamp",
        "rounded_boundary_timestamp",
        "sustained_block_start_timestamp",
        "sustained_block_end_timestamp",
    ):
        row[key] = pd.Timestamp(row[key]).isoformat()
    columns = [
        "parent",
        "split",
        "side",
        "kind",
        "left_timestamp",
        "left_load_kw",
        "right_timestamp",
        "right_load_kw",
        "original_boundary_load_kw",
        "canonical_boundary_load_kw",
        "crossing_fraction",
        "unrounded_boundary_timestamp",
        "rounded_boundary_timestamp",
        "rounding_adjustment_seconds",
        "removed_point_count",
        "removed_duration_s",
        "sustained_block_start_timestamp",
        "sustained_block_end_timestamp",
    ]
    return {column: row[column] for column in columns}


def _source_file_rows(raw_root: Path, excluded_root: Path) -> list[dict[str, object]]:
    rows = []
    for path in sorted(Path(raw_root).rglob("*")):
        if not path.is_file():
            continue
        resolved = path.resolve()
        try:
            resolved.relative_to(excluded_root.resolve())
            continue
        except ValueError:
            pass
        rows.append(
            {
                "relative_path": path.relative_to(raw_root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return rows


def _validate_outputs(
    frames: dict[str, pd.DataFrame],
    assignment: pd.DataFrame,
) -> dict[str, bool]:
    split_counts = assignment["split"].value_counts().to_dict()
    fixed_test = set(
        assignment.loc[assignment["split"].eq("test"), "parent"]
    )
    finite = all(
        np.isfinite(frame["load_total_kw"].to_numpy(dtype=float)).all()
        for frame in frames.values()
    )
    one_second = all(
        np.array_equal(
            frame["time_s"].to_numpy(dtype=float),
            np.arange(len(frame), dtype=float),
        )
        for frame in frames.values()
    )
    zero_endpoints = all(
        float(frame["load_total_kw"].iloc[0]) == 0.0
        and float(frame["load_total_kw"].iloc[-1]) == 0.0
        for frame in frames.values()
    )
    sustained = all(
        bool(
            np.any(
                np.convolve(
                    (frame["load_total_kw"].to_numpy(dtype=float) > ACTIVE_THRESHOLD_KW).astype(int),
                    np.ones(SUSTAINED_POINTS, dtype=int),
                    mode="valid",
                )
                == SUSTAINED_POINTS
            )
        )
        for frame in frames.values()
    )
    return {
        "exact_parent_and_split_counts": len(frames) == EXPECTED_INCLUDED_PARENT_COUNT
        and split_counts
        == {"train": TRAIN_COUNT, "validation": VALIDATION_COUNT, "test": TEST_COUNT},
        "fixed_test_set": fixed_test == set(FIXED_TEST_PARENTS),
        "no_parent_leakage": int(assignment.groupby("parent")["split"].nunique().max()) == 1,
        "finite_loads": finite,
        "exact_one_second_grid": one_second,
        "zero_endpoints": zero_endpoints,
        "sustained_positive_operation": sustained,
    }


def build_dataset(
    raw_root: Path,
    output_root: Path,
    *,
    discover_parents: Callable[[Path], list[str]] = discover_parent_ids,
    load_parent: Callable[[Path, str], ParentPowerSeries] = load_parent_power_series,
) -> dict[str, object]:
    """Build one immutable dataset root and return its QA summary."""
    raw_root = Path(raw_root).resolve()
    output_root = Path(output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"destination already exists: {output_root}")
    parents = sorted(
        discover_parents(raw_root),
        key=lambda value: parent_sort_key(Path(value)),
    )
    if (
        len(parents) != EXPECTED_RAW_PARENT_COUNT
        or len(set(parents)) != EXPECTED_RAW_PARENT_COUNT
    ):
        raise ValueError("expected exactly 66 recognizable parents")
    temporary = output_root.with_name(
        f"{output_root.name}.building-{os.getpid()}"
    )
    if temporary.exists():
        raise FileExistsError(f"temporary destination already exists: {temporary}")
    temporary.mkdir(parents=True)
    try:
        frames: dict[str, pd.DataFrame] = {}
        trims: dict[str, object] = {}
        interpolation_rows: dict[str, dict[str, object]] = {}
        feature_rows: list[dict[str, object]] = []
        excluded_rows: list[dict[str, object]] = []
        for parent in parents:
            measured = load_parent(raw_root, parent)
            if measured.parent != parent:
                raise ValueError(f"parent loader returned {measured.parent!r} for {parent!r}")
            filled = interpolate_power_gaps(
                measured.timestamps,
                measured.fc_total_kw,
                measured.battery_raw_total_kw,
                step_seconds=NOMINAL_STEP_SECONDS,
            )
            anchor = pd.DataFrame(
                {
                    "timestamp": filled.timestamps,
                    "fc_total_kw": filled.fc_kw,
                    "battery_raw_total_kw": filled.battery_raw_kw,
                    "source_total_kw": filled.source_kw,
                    "is_cubic_imputed": filled.is_interpolated,
                }
            )
            common_interpolation = {
                "aligned_measured_points": len(measured.timestamps),
                "cubic_gap_count": filled.interpolation_gap_count,
                "cubic_imputed_points": filled.interpolated_point_count,
                "cubic_max_gap_seconds": filled.max_interpolated_gap_seconds,
                "cubic_warning_count": len(filled.warning_messages),
                "cubic_warning_messages": "|".join(filled.warning_messages),
                "duplicate_count": measured.duplicate_count,
                "duplicate_conflict_count": measured.duplicate_conflict_count,
                "channel_span_violation_count": measured.channel_span_violation_count,
                "ais_present_points": int(np.count_nonzero(measured.ais_present)),
            }
            try:
                trimmed = trim_to_zero_boundaries(anchor)
            except ValueError as error:
                expected_side = APPROVED_BOUNDARY_EXCLUSIONS.get(parent)
                expected_message = (
                    f"{expected_side} boundary is not bracketed"
                    if expected_side is not None
                    else None
                )
                if expected_message is None or str(error) != expected_message:
                    raise ValueError(
                        f"{parent}: unexpected boundary failure: {error}"
                    ) from error
                values = anchor["source_total_kw"].to_numpy(dtype=float)
                excluded_rows.append(
                    {
                        "parent": parent,
                        "missing_boundary": expected_side,
                        "reason_code": "NO_BRACKET_WITHIN_RECORDING",
                        "reason": str(error),
                        "first_timestamp": pd.Timestamp(anchor["timestamp"].iloc[0]).isoformat(),
                        "last_timestamp": pd.Timestamp(anchor["timestamp"].iloc[-1]).isoformat(),
                        "point_count_30s": len(anchor),
                        "first_load_kw": float(values[0]),
                        "minimum_load_kw": float(np.min(values)),
                        "maximum_load_kw": float(np.max(values)),
                        "last_load_kw": float(values[-1]),
                        "deadband_point_count": int(
                            np.count_nonzero(np.abs(values) <= ZERO_DEADBAND_KW)
                        ),
                        "nonpositive_point_count": int(np.count_nonzero(values <= 0.0)),
                        **common_interpolation,
                    }
                )
                continue
            one_second, pchip_qa = reconstruct_one_second(trimmed.frame)
            frames[parent] = one_second
            trims[parent] = trimmed
            feature_rows.append(segment_features(parent, one_second))
            interpolation_rows[parent] = {
                **common_interpolation,
                "trimmed_anchor_points": len(trimmed.frame),
                "pchip_output_points": int(pchip_qa["output_points"]),
                "pchip_floating_negative_zeroed": int(
                    pchip_qa["floating_negative_zeroed"]
                ),
            }

        actual_exclusions = {
            str(row["parent"]): str(row["missing_boundary"])
            for row in excluded_rows
        }
        if actual_exclusions != APPROVED_BOUNDARY_EXCLUSIONS:
            raise ValueError(
                "actual boundary exclusions do not match the approved 13-parent set"
            )

        assignment = assign_parent_splits(pd.DataFrame(feature_rows))
        checks = _validate_outputs(frames, assignment)
        checks["approved_boundary_exclusions_exact"] = (
            actual_exclusions == APPROVED_BOUNDARY_EXCLUSIONS
        )
        if not all(checks.values()):
            failed = sorted(name for name, passed in checks.items() if not passed)
            raise ValueError(f"dataset acceptance failed before writing: {failed}")

        metadata = temporary / "metadata"
        metadata.mkdir()
        for split in ("train", "validation", "test"):
            (temporary / split).mkdir()
        sample_rows: list[dict[str, object]] = []
        boundary_rows: list[dict[str, object]] = []
        audit_rows: list[dict[str, object]] = []
        for row in assignment.itertuples(index=False):
            parent = str(row.parent)
            split = str(row.split)
            rank = int(row.chronological_rank) + 1
            sample_id = f"zero_boundary_{rank:03d}"
            relative = Path(split) / f"{sample_id}.csv"
            path = temporary / relative
            frame = frames[parent]
            frame.to_csv(path, index=False, encoding="utf-8-sig")
            sample_rows.append(
                {
                    "parent": parent,
                    "sample_id": sample_id,
                    "relative_path": relative.as_posix(),
                    "split": split,
                    "point_count_1s": len(frame),
                    "start_timestamp": pd.Timestamp(frame["timestamp"].iloc[0]).isoformat(),
                    "end_timestamp": pd.Timestamp(frame["timestamp"].iloc[-1]).isoformat(),
                    "duration_s": float(frame["time_s"].iloc[-1]),
                    "sha256": _sha256(path),
                }
            )
            trimmed = trims[parent]
            boundary_rows.append(_boundary_row(parent, split, trimmed.start))
            boundary_rows.append(_boundary_row(parent, split, trimmed.end))
            audit_rows.append(
                {"parent": parent, "split": split, **interpolation_rows[parent]}
            )

        sample_manifest = pd.DataFrame(sample_rows)
        sample_manifest.to_csv(
            metadata / "sample_manifest.csv", index=False, encoding="utf-8-sig"
        )
        parent_columns = [
            "parent",
            "split",
            "chronological_rank",
            "month",
            "duration_s",
            "mean_load_kw",
            "p95_load_kw",
            "duration_quartile",
            "mean_load_quartile",
            "p95_load_quartile",
        ]
        assignment[parent_columns].to_csv(
            metadata / "parent_split_manifest.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(excluded_rows).to_csv(
            metadata / "excluded_parent_manifest.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(boundary_rows).rename(
            columns={"kind": "boundary_kind"}
        ).to_csv(
            metadata / "trim_boundary_audit.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(audit_rows).to_csv(
            metadata / "interpolation_audit.csv",
            index=False,
            encoding="utf-8-sig",
        )
        source_rows = _source_file_rows(raw_root, temporary)
        pd.DataFrame(
            source_rows,
            columns=["relative_path", "size_bytes", "sha256"],
        ).to_csv(
            metadata / "source_files.csv", index=False, encoding="utf-8-sig"
        )
        policy = {
            "dataset_version": "operating_dataset_zero_boundary_v2",
            "parent_count": EXPECTED_INCLUDED_PARENT_COUNT,
            "split_counts": {
                "train": TRAIN_COUNT,
                "validation": VALIDATION_COUNT,
                "test": TEST_COUNT,
            },
            "fixed_test_parents": list(FIXED_TEST_PARENTS),
            "zero_deadband_kw": ZERO_DEADBAND_KW,
            "active_threshold_kw": ACTIVE_THRESHOLD_KW,
            "sustained_points": SUSTAINED_POINTS,
            "nominal_step_seconds": NOMINAL_STEP_SECONDS,
            "alignment_tolerance_seconds": ALIGNMENT_TOLERANCE_SECONDS,
            "gap_interpolation": "local natural cubic on FC and raw-sign battery independently",
            "one_second_interpolation": "PCHIP within selected boundaries only",
            "battery_raw_sign": "negative discharge, positive charge",
            "source_power_identity": "P_source_total = P_fc_total - P_battery_raw_total",
            "source_power_status": "derived from 8 FC and 12 battery clusters; not independently measured",
            "validation_selection": "greedy iterative multilabel stratification to 20 percent targets",
            "validation_tie_break": "chronological parent order then parent identifier",
            "raw_parent_count": EXPECTED_RAW_PARENT_COUNT,
            "included_parent_count": EXPECTED_INCLUDED_PARENT_COUNT,
            "approved_boundary_exclusions": APPROVED_BOUNDARY_EXCLUSIONS,
            "quartile_basis": "48 non-Test included parents only",
        }
        _write_json(metadata / "policy.json", policy)

        split_points = {
            split: int(
                sample_manifest.loc[
                    sample_manifest["split"].eq(split), "point_count_1s"
                ].sum()
            )
            for split in ("train", "validation", "test")
        }
        artifact_hashes = {
            path.relative_to(temporary).as_posix(): _sha256(path)
            for path in sorted(temporary.rglob("*"))
            if path.is_file() and path.name != "qa_summary.json"
        }
        summary: dict[str, object] = {
            "dataset_version": "operating_dataset_zero_boundary_v2",
            "raw_parent_count": EXPECTED_RAW_PARENT_COUNT,
            "excluded_parent_count": EXPECTED_EXCLUDED_PARENT_COUNT,
            "parent_count": EXPECTED_INCLUDED_PARENT_COUNT,
            "segment_count": len(sample_manifest),
            "point_count": int(sample_manifest["point_count_1s"].sum()),
            "split_point_counts": split_points,
            "split_parent_counts": {
                key: int(value)
                for key, value in assignment["split"].value_counts().items()
            },
            "negative_load_point_count": int(
                sum((frame["load_total_kw"] < 0.0).sum() for frame in frames.values())
            ),
            "acceptance_checks": checks,
            "source_file_count": len(source_rows),
            "source_manifest_sha256": _sha256(metadata / "source_files.csv"),
            "artifact_hashes": artifact_hashes,
            "formal_training_status": "NO-GO",
            "formal_training_blockers": [
                "final DQN state audit and freeze",
                "final action catalog 36 -> K screening and freeze",
                "final integrated preflight and solver robustness",
            ],
        }
        _write_json(metadata / "qa_summary.json", summary)
        temporary.replace(output_root)
        return summary
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()
    summary = build_dataset(arguments.raw_root, arguments.output_root)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
