"""Validate and render the frozen zero-boundary dataset power review."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


FROZEN_SPLIT_COUNTS = {"train": 38, "validation": 10, "test": 5}
SPLIT_ORDER = ("train", "validation", "test")
REQUIRED_COLUMNS = ("timestamp", "time_s", "load_total_kw")


@dataclass(frozen=True)
class ReviewEntry:
    parent: str
    sample_id: str
    split: str
    source_relative_path: str
    source_sha256: str
    point_count: int
    duration_s: float
    minimum_kw: float
    maximum_kw: float
    mean_kw: float
    y_min_kw: float
    y_max_kw: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def axis_limits(values: np.ndarray) -> tuple[float, float]:
    """Return zero-inclusive limits padded from this segment's own range."""
    data = np.asarray(values, dtype=float)
    if data.ndim != 1 or data.size == 0 or not np.isfinite(data).all():
        raise ValueError("power values must be a non-empty finite vector")
    lower = min(0.0, float(data.min()))
    upper = max(0.0, float(data.max()))
    span = upper - lower
    if span == 0.0:
        return (-1.0, 1.0)
    pad = 0.05 * span
    return (lower - pad, upper + pad)


def _load_series(
    path: Path, expected_points: int, expected_duration: float
) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=list(REQUIRED_COLUMNS))
    time_s = pd.to_numeric(frame.time_s, errors="coerce").to_numpy(dtype=float)
    load = pd.to_numeric(frame.load_total_kw, errors="coerce").to_numpy(dtype=float)
    if len(frame) != expected_points or len(frame) < 2:
        raise ValueError(f"point count mismatch: {path}")
    if not np.isfinite(time_s).all() or not np.isfinite(load).all():
        raise ValueError(f"non-finite series: {path}")
    if time_s[0] != 0.0 or not np.allclose(
        np.diff(time_s), 1.0, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError(f"time_s is not an exact one-second axis: {path}")
    if time_s[-1] != expected_duration:
        raise ValueError(f"duration mismatch: {path}")
    if load[0] != 0.0 or load[-1] != 0.0:
        raise ValueError(f"zero endpoint contract failed: {path}")
    return frame


def load_review_entries(
    dataset_root: Path,
    *,
    expected_split_counts: dict[str, int] | None = None,
) -> list[ReviewEntry]:
    """Load and validate all source series in frozen split order."""
    root = Path(dataset_root)
    expected = (
        FROZEN_SPLIT_COUNTS
        if expected_split_counts is None
        else expected_split_counts
    )
    manifest = pd.read_csv(root / "metadata" / "sample_manifest.csv")
    if manifest.groupby("split").size().to_dict() != expected:
        raise ValueError("manifest split counts do not match the frozen contract")
    entries: list[ReviewEntry] = []
    for split in SPLIT_ORDER:
        for row in manifest.loc[manifest.split.eq(split)].itertuples(index=False):
            path = root / str(row.relative_path)
            if _sha256(path) != str(row.sha256):
                raise ValueError(f"source SHA-256 mismatch: {path}")
            frame = _load_series(
                path,
                int(row.point_count_1s),
                float(row.duration_s),
            )
            values = frame.load_total_kw.to_numpy(dtype=float)
            y_min, y_max = axis_limits(values)
            entries.append(
                ReviewEntry(
                    parent=str(row.parent),
                    sample_id=str(row.sample_id),
                    split=split,
                    source_relative_path=str(row.relative_path),
                    source_sha256=str(row.sha256),
                    point_count=len(frame),
                    duration_s=float(row.duration_s),
                    minimum_kw=float(values.min()),
                    maximum_kw=float(values.max()),
                    mean_kw=float(values.mean()),
                    y_min_kw=y_min,
                    y_max_kw=y_max,
                )
            )
    return entries
