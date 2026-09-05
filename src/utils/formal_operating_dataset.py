"""Read-only access to the frozen formal operating-segment dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OPERATING_DATASET_ROOT = (
    REPO_ROOT / "data" / "processed" / "operating_segments_1s_rebuilt"
)
DEFAULT_SPLIT_MANIFEST = DEFAULT_OPERATING_DATASET_ROOT / "split_manifest.csv"
LOAD_COLUMN = "load_total_kw"
SPLIT_NAMES = ("train", "validation", "test")
EXPECTED_PARENT_VOYAGE_COUNT = 66
EXPECTED_SEGMENT_COUNT = 177
EXPECTED_POINT_COUNT = 1_114_037
EXPECTED_SPLIT_POINT_COUNTS = {
    "train": 796_249,
    "validation": 248_867,
    "test": 68_921,
}


@dataclass(frozen=True)
class OperatingSegmentSplit:
    """The frozen split and ordered segment identifiers from its manifest."""

    train_segments: tuple[str, ...]
    validation_segments: tuple[str, ...]
    test_segments: tuple[str, ...]
    train_parents: tuple[str, ...]
    validation_parents: tuple[str, ...]
    test_parents: tuple[str, ...]
    manifest: pd.DataFrame
    dataset_root: Path


@dataclass(frozen=True)
class FormalDatasetAudit:
    parent_voyage_count: int
    segment_count: int
    point_count: int
    split_point_counts: dict[str, int]
    negative_load_point_count: int
    missing_segment_paths: tuple[str, ...]
    orphan_segment_paths: tuple[str, ...]
    parent_split_leakage: tuple[str, ...]


def _resolve_dataset_root(dataset_root: str | Path) -> Path:
    root = Path(dataset_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"formal operating dataset is missing: {root}")
    return root


def _path_within(root: Path, relative_path: str) -> Path:
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"manifest path escapes formal dataset: {relative_path!r}") from error
    return path


def load_formal_operating_split(
    dataset_root: str | Path = DEFAULT_OPERATING_DATASET_ROOT,
) -> OperatingSegmentSplit:
    """Load the one authoritative CSV split without any legacy fallback."""
    root = _resolve_dataset_root(dataset_root)
    manifest_path = root / "split_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"formal split manifest is missing: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    required = {"parent_voyage", "segment_id", "one_second_csv", "split"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"formal split manifest is missing columns: {sorted(missing)}")
    if manifest["segment_id"].isna().any() or manifest["segment_id"].duplicated().any():
        raise ValueError("formal split manifest has duplicate or missing segment_id")
    manifest = manifest.copy()
    manifest["segment_id"] = manifest["segment_id"].astype(str)
    manifest["parent_voyage"] = manifest["parent_voyage"].astype(str)
    manifest["split"] = manifest["split"].astype(str).str.strip().str.lower()
    if set(manifest["split"]).difference(SPLIT_NAMES):
        raise ValueError("formal split manifest has an unknown split")

    segment_sets = {
        name: tuple(manifest.loc[manifest["split"].eq(name), "segment_id"])
        for name in SPLIT_NAMES
    }
    parent_sets = {
        name: tuple(
            dict.fromkeys(manifest.loc[manifest["split"].eq(name), "parent_voyage"])
        )
        for name in SPLIT_NAMES
    }
    if not all(segment_sets.values()):
        raise ValueError("formal split manifest must contain every split")
    parent_to_split = manifest.groupby("parent_voyage")["split"].nunique()
    if bool(parent_to_split.gt(1).any()):
        raise ValueError("formal split manifest leaks a parent voyage between splits")
    return OperatingSegmentSplit(
        train_segments=segment_sets["train"],
        validation_segments=segment_sets["validation"],
        test_segments=segment_sets["test"],
        train_parents=parent_sets["train"],
        validation_parents=parent_sets["validation"],
        test_parents=parent_sets["test"],
        manifest=manifest,
        dataset_root=root,
    )


def load_operating_segment_loads(
    split_name: str,
    segment_id: str,
    *,
    split: OperatingSegmentSplit,
    allow_test: bool = False,
) -> np.ndarray:
    """Load one independent 1-second episode and enforce its frozen split."""
    name = str(split_name).strip().lower()
    allowed = {
        "train": split.train_segments,
        "validation": split.validation_segments,
        "test": split.test_segments,
    }
    if name not in allowed or (name == "test" and not allow_test):
        raise ValueError(f"formal loader forbids {name!r} access in this context")
    identifier = str(segment_id)
    if identifier not in allowed[name]:
        raise ValueError(f"{identifier} does not belong to formal {name} split")
    row = split.manifest.loc[split.manifest["segment_id"].eq(identifier)]
    if len(row) != 1 or row.iloc[0]["split"] != name:
        raise ValueError(f"formal manifest mismatch for {identifier}")
    path = _path_within(split.dataset_root, str(row.iloc[0]["one_second_csv"]))
    if not path.is_file():
        raise FileNotFoundError(f"formal segment is missing: {path}")
    frame = pd.read_csv(path, usecols=["timestamp", "time_s", LOAD_COLUMN])
    loads = pd.to_numeric(frame[LOAD_COLUMN], errors="coerce").to_numpy(dtype=np.float64)
    time_s = pd.to_numeric(frame["time_s"], errors="coerce").to_numpy(dtype=np.float64)
    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
    if len(loads) < 2 or not np.isfinite(loads).all() or bool((loads < 0.0).any()):
        raise ValueError(f"formal segment has invalid loads: {identifier}")
    if (
        timestamps.isna().any()
        or not timestamps.is_monotonic_increasing
        or timestamps.duplicated().any()
        or not np.allclose(np.diff(time_s), 1.0, rtol=0.0, atol=1.0e-12)
    ):
        raise ValueError(f"formal segment is not a strictly one-second grid: {identifier}")
    return loads


def audit_formal_operating_dataset(
    dataset_root: str | Path = DEFAULT_OPERATING_DATASET_ROOT,
) -> FormalDatasetAudit:
    """Read every final CSV and verify frozen dataset completeness and totals."""
    split = load_formal_operating_split(dataset_root)
    root = split.dataset_root
    referenced: set[Path] = set()
    missing: list[str] = []
    point_count = 0
    negative_count = 0
    split_points = {name: 0 for name in SPLIT_NAMES}
    for row in split.manifest.itertuples(index=False):
        raw_csv = getattr(row, "raw_csv", None)
        if pd.notna(raw_csv) and str(raw_csv).strip():
            raw_path = _path_within(root, str(raw_csv))
            if not raw_path.is_file():
                missing.append(str(raw_path))
        path = _path_within(root, str(row.one_second_csv))
        referenced.add(path)
        if not path.is_file():
            missing.append(str(path))
            continue
        frame = pd.read_csv(path, usecols=["timestamp", "time_s", LOAD_COLUMN])
        loads = pd.to_numeric(frame[LOAD_COLUMN], errors="coerce").to_numpy(dtype=np.float64)
        time_s = pd.to_numeric(frame["time_s"], errors="coerce").to_numpy(dtype=np.float64)
        timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
        if (
            len(loads) < 2
            or not np.isfinite(loads).all()
            or timestamps.isna().any()
            or not timestamps.is_monotonic_increasing
            or timestamps.duplicated().any()
            or not np.allclose(np.diff(time_s), 1.0, rtol=0.0, atol=1.0e-12)
        ):
            raise ValueError(f"invalid final one-second segment: {row.segment_id}")
        point_count += len(loads)
        negative_count += int((loads < 0.0).sum())
        split_points[str(row.split)] += len(loads)
    final_directory = root / "operating_segments_1s"
    actual_paths = {path.resolve() for path in final_directory.glob("*.csv")}
    parent_split_counts = split.manifest.groupby("parent_voyage")["split"].nunique()
    return FormalDatasetAudit(
        parent_voyage_count=int(split.manifest["parent_voyage"].nunique()),
        segment_count=int(len(split.manifest)),
        point_count=point_count,
        split_point_counts=split_points,
        negative_load_point_count=negative_count,
        missing_segment_paths=tuple(sorted(missing)),
        orphan_segment_paths=tuple(sorted(str(path) for path in actual_paths.difference(referenced))),
        parent_split_leakage=tuple(sorted(parent_split_counts[parent_split_counts.gt(1)].index.astype(str))),
    )
