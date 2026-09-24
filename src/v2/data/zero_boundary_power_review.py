"""Validate and render the frozen zero-boundary dataset power review."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from html import escape
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def _plot_entry(
    dataset_root: Path, output_root: Path, entry: ReviewEntry
) -> str:
    frame = pd.read_csv(
        dataset_root / entry.source_relative_path,
        usecols=list(REQUIRED_COLUMNS),
    )
    image_relative = Path(entry.split) / f"{entry.sample_id}.png"
    image_path = output_root / image_relative
    image_path.parent.mkdir(parents=True, exist_ok=True)

    with plt.rc_context(
        {
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "Arial Unicode MS",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
        }
    ):
        fig, ax = plt.subplots(figsize=(12.0, 4.8), constrained_layout=True)
        ax.plot(
            frame.time_s,
            frame.load_total_kw,
            color="#0B3C5D",
            linewidth=1.0,
        )
        ax.axhline(0.0, color="#666666", linewidth=0.8, linestyle="--")
        ax.scatter(
            [float(frame.time_s.iloc[0]), float(frame.time_s.iloc[-1])],
            [
                float(frame.load_total_kw.iloc[0]),
                float(frame.load_total_kw.iloc[-1]),
            ],
            color="#A61B1B",
            s=18,
            zorder=3,
            label="Zero-power endpoints",
        )
        ax.set_xlim(0.0, entry.duration_s)
        ax.set_ylim(entry.y_min_kw, entry.y_max_kw)
        ax.set_xlabel("Elapsed time (s)")
        ax.set_ylabel("Total power (kW)")
        ax.grid(True, linewidth=0.45, alpha=0.25)
        ax.set_title(
            f"{entry.parent} | {entry.sample_id} | {entry.split.upper()}",
            loc="left",
        )
        fig.suptitle(
            f"duration={entry.duration_s:.0f} s | points={entry.point_count:,} | "
            f"min={entry.minimum_kw:.2f} kW | max={entry.maximum_kw:.2f} kW | "
            f"mean={entry.mean_kw:.2f} kW",
            fontsize=9,
            x=0.99,
            ha="right",
            color="#4B5563",
        )
        fig.savefig(
            image_path,
            dpi=140,
            metadata={"Software": "zero-boundary power review"},
        )
        plt.close(fig)
    return image_relative.as_posix()


def _write_index(output_root: Path, rows: list[dict[str, object]]) -> None:
    sections = []
    for split in SPLIT_ORDER:
        cards = []
        for row in (item for item in rows if item["split"] == split):
            image = escape(str(row["image_relative_path"]), quote=True)
            title = escape(f'{row["parent"]} | {row["sample_id"]}')
            cards.append(
                f'<article><h3>{title}</h3><p>'
                f'duration={row["duration_s"]:.0f} s | '
                f'points={row["point_count"]:,} | '
                f'min={row["minimum_kw"]:.2f} kW | '
                f'max={row["maximum_kw"]:.2f} kW | '
                f'mean={row["mean_kw"]:.2f} kW</p>'
                f'<a href="{image}"><img src="{image}" loading="lazy" '
                f'alt="{title}"></a></article>'
            )
        sections.append(
            f'<section id="{split}"><h2>{split.upper()}</h2>'
            f'{"".join(cards)}</section>'
        )
    document = (
        """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>Zero-boundary v2 total-power review</title><style>
body{font-family:Arial,"Microsoft YaHei",sans-serif;margin:24px;background:#f4f6f8;color:#17202a}
nav a{margin-right:18px}section{margin-top:30px}article{background:#fff;padding:14px;margin:16px 0;border-radius:7px;box-shadow:0 1px 5px #ccd1d1}
h3{font-size:17px;margin:0 0 4px}p{font-size:13px;color:#566573}img{width:100%;height:auto;border:1px solid #ddd}
</style></head><body><h1>Zero-boundary v2 total-power review</h1>
<p>每个航段采用独立纵轴；图像高度不可用于跨航段比较功率幅值。数据按正式清单原值绘制。</p>
<nav><a href="#train">Train</a><a href="#validation">Validation</a><a href="#test">Test</a></nav>
"""
        + "".join(sections)
        + "</body></html>"
    )
    (output_root / "index.html").write_text(document, encoding="utf-8")


def build_review(
    dataset_root: Path,
    output_root: Path,
    *,
    expected_split_counts: dict[str, int] | None = None,
) -> list[dict[str, object]]:
    """Build a new review directory without changing the dataset."""
    output = Path(output_root)
    if output.exists():
        raise FileExistsError(f"review output already exists: {output}")
    entries = load_review_entries(
        dataset_root,
        expected_split_counts=expected_split_counts,
    )
    output.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    try:
        for entry in entries:
            image = _plot_entry(Path(dataset_root), output, entry)
            rows.append({**entry.__dict__, "image_relative_path": image})
        pd.DataFrame(rows).to_csv(
            output / "review_manifest.csv",
            index=False,
            encoding="utf-8-sig",
        )
        _write_index(output, rows)
    except Exception:
        shutil.rmtree(output)
        raise
    return rows
