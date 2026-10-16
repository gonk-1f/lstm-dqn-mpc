"""Render a data-only candidate review without changing any source CSV.

Every input row is sent to Matplotlib.  The PDF contains a reading guide,
an index, every parent overview, and every candidate at a readable page size.
Call ``generate_figures(Path(dataset_root))`` or run this file directly.
"""

from __future__ import annotations

import argparse
from collections import Counter
import html
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import textwrap
from urllib.parse import quote

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


_MARKER_COMPLETED = False
_LOAD_COLOR = "#17466e"
_AIS_COLOR = "#40816c"
_QUALITY_COLORS = {"A": "#43866a", "B": "#c68a2c", "C": "#9f667e", "D": "#8a919a"}
_INDEX_ROWS = 35


def mark_pdf_operation_started(marker_path: Path | None = None) -> None:
    """Run the installed PDF skill marker successfully once per process."""
    global _MARKER_COMPLETED
    if _MARKER_COMPLETED:
        return
    if marker_path is None:
        override = os.environ.get("CODEX_PDF_MARKER")
        if override:
            marker_path = Path(override)
        else:
            installed = Path.home() / ".codex/plugins/cache/openai-primary-runtime/pdf"
            matches = sorted(installed.glob("*/skills/pdf/container_tools/mark_artifact_operation_started.mjs"))
            local = Path("container_tools/mark_artifact_operation_started.mjs")
            marker_path = local if local.is_file() else (matches[-1] if matches else None)
    if marker_path is None or not marker_path.is_file():
        raise FileNotFoundError("PDF skill marker not found; set CODEX_PDF_MARKER to its installed path")
    node = shutil.which("node")
    if node is None:
        raise RuntimeError("Node.js is required to run the PDF skill marker")
    subprocess.run(
        [node, str(marker_path.resolve()), "--operation-kind", "create",
         "--expected-output-count", "1", "--output-format", "pdf"],
        check=True,
    )
    _MARKER_COMPLETED = True


def _font_settings() -> tuple[list[str], bool]:
    available = {entry.name for entry in font_manager.fontManager.ttflist}
    for name in ("Microsoft YaHei", "Microsoft YaHei UI", "Noto Sans CJK SC", "SimHei"):
        if name in available:
            return [name, "DejaVu Sans"], True
    return ["DejaVu Sans"], False


def _safe_file_id(value: object) -> str:
    name = str(value)
    if not name or name in {".", ".."} or any(char in name for char in '/\\:*?"<>|'):
        raise ValueError(f"Invalid figure identifier: {name!r}")
    return name


def _number(value: object, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _text(value: object) -> str:
    return "" if value is None or pd.isna(value) else str(value)


def _inside_root(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Manifest path leaves dataset root: {relative}")
    return path


def _read_trace(path: Path, *, candidate: bool) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = {"timestamp", "time_s", "load_total_kw", "speed_kn"}
    required |= {"raw_snapshot_gap_s", "load_provenance"} if candidate else {"selection_status"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if frame.empty:
        return frame
    for column in ("time_s", "load_total_kw", "speed_kn", "raw_snapshot_gap_s"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    times = frame.time_s.to_numpy(dtype=float)
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError(f"{path}: time_s must be finite and strictly increasing")
    if candidate and not np.isfinite(frame.load_total_kw.to_numpy(dtype=float)).all():
        raise ValueError(f"{path}: candidate load contains missing or nonfinite points")
    return frame


def _axes() -> tuple[plt.Figure, plt.Axes, plt.Axes]:
    figure, (load_ax, speed_ax) = plt.subplots(
        2, 1, figsize=(12.8, 8.0), sharex=True,
        gridspec_kw={"height_ratios": [3.3, 1.0], "hspace": 0.10},
    )
    figure.subplots_adjust(left=0.085, right=0.975, top=0.80, bottom=0.16)
    for axis in (load_ax, speed_ax):
        axis.grid(True, color="#e3e8ec", linewidth=0.65)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=9)
        axis.margins(x=0.025)
    load_ax.set_ylabel("Derived total load (kW)", fontsize=10)
    speed_ax.set_ylabel("AIS (kn)", fontsize=10)
    return figure, load_ax, speed_ax


def _plot_traces(load_ax: plt.Axes, speed_ax: plt.Axes, frame: pd.DataFrame) -> np.ndarray:
    hours = frame.time_s.to_numpy(dtype=float) / 3600.0
    # No resampling, representative points, clipping, or simplification.
    load_ax.plot(hours, frame.load_total_kw.to_numpy(dtype=float), color=_LOAD_COLOR, linewidth=0.95)
    load_ax.axhline(0, color="#8a949e", linewidth=0.7, zorder=0)
    speed_ax.plot(hours, frame.speed_kn.to_numpy(dtype=float), color=_AIS_COLOR, linewidth=0.9)
    speed_ax.set_ylim(bottom=0)
    speed_ax.set_xlabel("Elapsed time in input CSV (h)", fontsize=10)
    return hours


def _candidate_figure(frame: pd.DataFrame, row: pd.Series, *, chinese: bool) -> plt.Figure:
    figure, load_ax, speed_ax = _axes()
    hours = _plot_traces(load_ax, speed_ax, frame)
    identifier = _text(row.candidate_id)
    quality = _text(row.quality_class)
    figure.suptitle(f"{identifier}  |  Candidate review  |  Class {quality}",
                   x=0.085, y=0.975, ha="left", fontsize=15, fontweight="bold")
    parent_line = f"Parent: {_text(row.parent_sample_id)}"
    if chinese and _text(row.get("parent_name")):
        parent_line += f"  |  {_text(row.parent_name)}"
    figure.text(0.085, 0.925, parent_line, fontsize=10)
    figure.text(0.085, 0.889,
                f"{_text(frame.timestamp.iloc[0])}  to  {_text(frame.timestamp.iloc[-1])}", fontsize=9)
    figure.text(0.085, 0.852,
                f"{len(frame):,} input points  |  elapsed {hours[-1] - hours[0]:.3f} h"
                f"  |  mean {frame.load_total_kw.mean():.1f} kW"
                f"  |  peak {frame.load_total_kw.max():.1f} kW", fontsize=9)
    for index, label, kind, fraction, align in (
        (0, "Start", row.start_boundary_kind, 0.015, "left"),
        (-1, "End", row.end_boundary_kind, 0.985, "right"),
    ):
        value = float(frame.load_total_kw.iloc[index])
        load_ax.scatter([hours[index]], [value], color="#bc6245", edgecolor="white",
                        marker="D", s=35, linewidth=0.7, zorder=5)
        kind_text = textwrap.fill(_text(kind).replace("_", " "), width=34)
        provenance = textwrap.fill(_text(frame.load_provenance.iloc[index]).replace("_", " "), width=38)
        load_ax.text(fraction, 0.975,
                     f"{label}: {value:.3f} kW\n{kind_text}\n{provenance}",
                     transform=load_ax.transAxes, ha=align, va="top", fontsize=8,
                     bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#d4dde4", "pad": 4}, zorder=6)
    gap = frame.raw_snapshot_gap_s.max()
    gap_text = f"maximum source bracket {_number(gap):.1f} s" if pd.notna(gap) else "source bracket unavailable"
    provenance_counts = Counter(frame.load_provenance.fillna("unknown").astype(str))
    tags = _text(row.get("feature_tags"))
    footer = f"Boundary values are preserved. {gap_text}. All {len(frame):,} CSV rows are plotted."
    figure.text(0.085, 0.105, footer, fontsize=8, va="top")
    figure.text(0.085, 0.075,
                textwrap.shorten(f"Load provenance: {dict(provenance_counts)}", width=175),
                fontsize=7.5, va="top")
    if tags:
        figure.text(0.085, 0.047, textwrap.shorten(f"Review tags: {tags}", width=175),
                    fontsize=7.5, va="top")
    return figure


def _status_family(status: str) -> tuple[str, str]:
    lowered = status.lower()
    if "unknown" in lowered or "conflict" in lowered or "invalid" in lowered:
        return "Unknown / invalid evidence", "#cf7777"
    if "shore" in lowered or "charg" in lowered:
        return "Charging / external supply", "#d4a56b"
    if "gap" in lowered or "missing" in lowered:
        return "Missing / source gap", "#96a9bf"
    if "idle" in lowered or "zero" in lowered:
        return "Long idle / near zero", "#b5b9bf"
    if any(word in lowered for word in ("exclude", "reject", "not_selected", "unselected", "review")):
        return "Other review / exclusion", "#c5bfcc"
    if any(word in lowered for word in ("candidate", "accept", "retain", "keep", "selected", "eligible")):
        return "Candidate input", "#9dc9b6"
    return "Other review / exclusion", "#c5bfcc"


def _parent_figure(frame: pd.DataFrame, parent_id: str, candidates: pd.DataFrame,
                   *, chinese: bool) -> plt.Figure:
    figure, load_ax, speed_ax = _axes()
    figure.suptitle(f"{parent_id}  |  Complete parent recording", x=0.085, y=0.975,
                   ha="left", fontsize=15, fontweight="bold")
    if frame.empty:
        load_ax.text(0.5, 0.5, "No target-grid rows in this recording", transform=load_ax.transAxes,
                     ha="center", fontsize=12)
        figure.text(0.085, 0.91, "The inventory retains this parent for explicit review.", fontsize=10)
        return figure
    hours = _plot_traces(load_ax, speed_ax, frame)
    status = frame.selection_status.fillna("unclassified").astype(str).to_numpy()
    changes = np.r_[0, np.flatnonzero(status[1:] != status[:-1]) + 1, len(status)]
    legends: dict[str, str] = {}
    half_step = float(np.median(np.diff(hours))) / 2 if len(hours) > 1 else 1 / 240
    for left, right in zip(changes[:-1], changes[1:]):
        label, color = _status_family(str(status[left]))
        legends[label] = color
        for axis in (load_ax, speed_ax):
            axis.axvspan(hours[left] - half_step, hours[right - 1] + half_step,
                         color=color, alpha=0.20, linewidth=0, zorder=0)
    for _, row in candidates.iterrows():
        start, end = _number(row.get("source_start_time_s")), _number(row.get("source_end_time_s"))
        if not np.isfinite([start, end]).all():
            continue
        color = _QUALITY_COLORS.get(_text(row.quality_class), "#43866a")
        for axis in (load_ax, speed_ax):
            axis.axvspan(start / 3600, end / 3600, color=color, alpha=0.15, linewidth=0)
        load_ax.text((start + end) / 7200, 0.97, _text(row.candidate_id),
                     transform=load_ax.get_xaxis_transform(), rotation=90, ha="right", va="top",
                     fontsize=7.5, color="#273847",
                     bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1})
    name = _text(candidates.iloc[0].get("parent_name")) if len(candidates) else ""
    descriptor = f"{len(frame):,} target-grid rows  |  {len(candidates)} candidate intervals"
    if chinese and name:
        descriptor += f"  |  {name}"
    figure.text(0.085, 0.925, descriptor, fontsize=10)
    figure.text(0.085, 0.89,
                f"{_text(frame.timestamp.iloc[0])}  to  {_text(frame.timestamp.iloc[-1])}", fontsize=9)
    handles = [Patch(facecolor=color, alpha=0.45, label=label) for label, color in legends.items()]
    figure.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.077, 0.865),
                  ncol=3, frameon=False, fontsize=8)
    counts = dict(Counter(status))
    footer = f"All {len(frame):,} rows are plotted; missing load remains a visible break. Candidate bands use parent elapsed time."
    figure.text(0.085, 0.095, footer, fontsize=8, va="top")
    figure.text(0.085, 0.06, textwrap.fill(textwrap.shorten(
        f"Selection status counts: {counts}", width=290), 150), fontsize=7.5, va="top")
    return figure


def _cover(parent_count: int, candidate_count: int, *, chinese: bool) -> plt.Figure:
    figure = plt.figure(figsize=(12.8, 8.0))
    figure.text(0.08, 0.9, "Independent voyage candidate review", fontsize=23, weight="bold")
    figure.text(0.08, 0.83,
                f"Stage 1 - data review only  |  {parent_count} parent recordings  |  {candidate_count} candidate intervals",
                fontsize=12)
    paragraphs = [
        "Read a complete parent page first. Shaded exclusions and candidate bands show where each interval came from; the horizontal axis retains the parent's elapsed clock.",
        "Candidate pages show the full supplied 30 s load series above AIS speed. Every CSV row is plotted without point compression. A candidate's elapsed clock may restart at zero.",
        "Diamonds mark the actual endpoint values. Boundary descriptions and source provenance appear beside them. Positive auxiliary load is retained, and this renderer never forces an endpoint to zero.",
        "Derived total load comes from the fuel-cell and battery measurement chain. It is not a separate whole-vessel demand meter. AIS stationarity alone does not prove a shore connection or a completed voyage.",
        "A regular 30 s grid does not prove continuous raw measurement. Inspect raw source gaps, provenance, UNKNOWN intervals, long idle periods, and nonzero endpoints before selecting a candidate.",
        "Quality classes and review tags are supplied by the candidate manifest. These pages do not freeze the dataset or perform training, controller, policy, or Test evaluation.",
    ]
    y = 0.73
    for number, paragraph in enumerate(paragraphs, 1):
        figure.text(0.085, y, f"{number}. " + textwrap.fill(paragraph, width=112,
                     subsequent_indent="   "), fontsize=11, va="top", linespacing=1.45)
        y -= 0.10
    if chinese:
        figure.text(0.085, 0.075, "审查顺序：父记录全图 → 候选段 → 来源间隙与边界说明。候选清单尚未冻结。", fontsize=10)
    return figure


def _index_figure(entries: list[tuple[str, str, int]], index_number: int) -> plt.Figure:
    figure = plt.figure(figsize=(12.8, 8.0))
    figure.text(0.085, 0.93, f"Review index - {index_number}", fontsize=18, weight="bold")
    figure.text(0.085, 0.865, "Type   Identifier / review details", fontsize=10, weight="bold")
    figure.text(0.94, 0.865, "Page", fontsize=10, ha="right", weight="bold")
    for position, (kind, label, page) in enumerate(entries):
        y = 0.825 - position * 0.0208
        figure.text(0.085, y, kind, fontsize=9)
        figure.text(0.14, y, textwrap.shorten(label, width=110), fontsize=9)
        figure.text(0.94, y, str(page), fontsize=9, ha="right")
    return figure


def _write_gallery(path: Path, parents: list[dict], candidates: list[dict]) -> None:
    cards = []
    for kind, records in (("Parent", parents), ("Candidate", candidates)):
        for record in records:
            identifier = record.get("candidate_id", record.get("parent_id", ""))
            subtitle = str(record.get("review_label", ""))
            target = quote(str(record["png"]), safe="/")
            search = html.escape(f"{kind} {identifier} {subtitle}".lower(), quote=True)
            cards.append(
                f'<article data-search="{search}"><h2>{html.escape(kind)}: {html.escape(identifier)}</h2>'
                f'<p>{html.escape(subtitle)}</p><a href="{target}"><img loading="lazy" src="{target}" '
                f'alt="{html.escape(identifier, quote=True)} full resolution review"></a></article>'
            )
    body = "\n".join(cards)
    page = '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Independent candidate review</title>
<style>body{font-family:"Microsoft YaHei",Arial,sans-serif;max-width:1320px;margin:24px auto;padding:0 20px;color:#243647;background:#f4f6f8}header{position:sticky;top:0;background:#f4f6f8ed;padding:12px 0;z-index:2}h1{font-size:24px;margin:0 0 8px}input{width:96%;padding:11px;border:1px solid #abb7c1;border-radius:6px;font-size:16px}article{background:white;padding:12px;margin:20px 0;border:1px solid #d5dde3;border-radius:6px}h2{font-size:17px;margin:4px 0}p{font-size:13px}img{width:100%;height:auto}a{color:#17466e}</style>
<header><h1>Independent voyage candidate review</h1><p>Data review only. Every plotted CSV row is retained. Click a plot to open the full PNG. <a href="independent_voyage_candidate_review.pdf">Complete review PDF</a></p><input id="search" type="search" placeholder="Search parent / candidate ID / class / tags" aria-label="Search plots"><p id="count"></p></header>
''' + body + '''<script>const cards=[...document.querySelectorAll('article')];const input=document.querySelector('#search');function filter(){const q=input.value.trim().toLowerCase();let n=0;for(const card of cards){const visible=card.dataset.search.includes(q);card.hidden=!visible;n+=visible;}document.querySelector('#count').textContent=n+' / '+cards.length+' plots';}input.addEventListener('input',filter);filter();</script></html>'''
    path.write_text(page, encoding="utf-8")


def generate_figures(dataset_root: Path) -> dict:
    """Write every parent/candidate PNG, complete review PDF, and HTML gallery."""
    root = Path(dataset_root).resolve()
    manifest = pd.read_csv(root / "metadata/candidate_manifest.csv", encoding="utf-8-sig")
    required = {"candidate_id", "parent_sample_id", "quality_class", "relative_path",
                "start_boundary_kind", "end_boundary_kind"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"Candidate manifest missing columns: {sorted(missing)}")
    if manifest.candidate_id.isna().any() or manifest.candidate_id.duplicated().any():
        raise ValueError("Candidate identifiers must be present and unique")
    manifest = manifest.sort_values(["parent_sample_id", "candidate_id"], kind="stable").reset_index(drop=True)
    parent_directory = root / "metadata/parent_overview_inputs"
    parent_paths = {path.stem: path for path in sorted(parent_directory.glob("*.csv"))}
    for _, row in manifest.iterrows():
        parent_id = _safe_file_id(row.parent_sample_id)
        if parent_id not in parent_paths:
            optional = _text(row.get("parent_overview_path"))
            if optional and Path(optional).suffix.lower() == ".csv":
                parent_paths[parent_id] = _inside_root(root, optional)
            else:
                raise FileNotFoundError(f"Missing parent overview input for {parent_id}")
    output = root / "figures"
    parent_output, candidate_output = output / "parents", output / "candidates"
    parent_output.mkdir(parents=True, exist_ok=True)
    candidate_output.mkdir(parents=True, exist_ok=True)
    font_names, chinese = _font_settings()
    total = len(parent_paths) + len(manifest)
    index_count = max(1, math.ceil(total / _INDEX_ROWS))
    next_page = 2 + index_count
    parent_records: list[dict] = []
    candidate_records: list[dict] = []
    entries: list[tuple[str, str, int]] = []
    for parent_id in sorted(parent_paths):
        _safe_file_id(parent_id)
        entries.append(("Parent", parent_id, next_page))
        next_page += 1
    for _, row in manifest.iterrows():
        identifier = _safe_file_id(row.candidate_id)
        description = f"{identifier} | {_text(row.parent_sample_id)} | class {_text(row.quality_class)}"
        entries.append(("Cand.", description, next_page))
        next_page += 1
    pdf_path = output / "independent_voyage_candidate_review.pdf"
    mark_pdf_operation_started()
    settings = {"font.family": "sans-serif", "font.sans-serif": font_names,
                "axes.unicode_minus": False, "path.simplify": False,
                "agg.path.chunksize": 0, "pdf.fonttype": 42, "font.size": 10}
    with matplotlib.rc_context(settings), PdfPages(pdf_path) as pdf:
        pdf.infodict().update({"Title": "Independent voyage candidate review",
                               "Subject": "Stage 1 data-only review; full input resolution"})
        page_number = 0

        def save(figure: plt.Figure, png_path: Path | None = None) -> None:
            nonlocal page_number
            try:
                page_number += 1
                figure.text(0.975, 0.015, f"Page {page_number}", fontsize=7, ha="right", color="#657481")
                if png_path is not None:
                    figure.savefig(png_path, dpi=180, facecolor="white")
                pdf.savefig(figure, dpi=140, facecolor="white")
            finally:
                plt.close(figure)

        save(_cover(len(parent_paths), len(manifest), chinese=chinese))
        for index in range(index_count):
            save(_index_figure(entries[index * _INDEX_ROWS:(index + 1) * _INDEX_ROWS], index + 1))
        for parent_id, path in sorted(parent_paths.items()):
            frame = _read_trace(path, candidate=False)
            matching = manifest.loc[manifest.parent_sample_id.eq(parent_id)]
            png_path = parent_output / f"{parent_id}.png"
            save(_parent_figure(frame, parent_id, matching, chinese=chinese), png_path)
            parent_records.append({"parent_id": parent_id, "point_count": len(frame),
                                   "candidate_count": len(matching), "pdf_page": page_number,
                                   "png": png_path.relative_to(output).as_posix(),
                                   "review_label": f"{len(matching)} candidates; complete parent recording"})
            del frame
        for _, row in manifest.iterrows():
            frame = _read_trace(_inside_root(root, _text(row.relative_path)), candidate=True)
            if frame.empty:
                raise ValueError(f"Empty candidate CSV: {row.candidate_id}")
            expected = _number(row.get("point_count"))
            if np.isfinite(expected) and len(frame) != expected:
                raise ValueError(f"{row.candidate_id}: manifest point_count differs from CSV")
            png_path = candidate_output / f"{_safe_file_id(row.candidate_id)}.png"
            save(_candidate_figure(frame, row, chinese=chinese), png_path)
            candidate_records.append({"candidate_id": _text(row.candidate_id),
                                      "parent_id": _text(row.parent_sample_id),
                                      "quality_class": _text(row.quality_class),
                                      "point_count": len(frame), "pdf_page": page_number,
                                      "png": png_path.relative_to(output).as_posix(),
                                      "review_label": f"{_text(row.parent_sample_id)} | class {_text(row.quality_class)} | {_text(row.get('feature_tags'))}"})
            del frame
    gallery_path = output / "index.html"
    _write_gallery(gallery_path, parent_records, candidate_records)
    result = {"parent_figure_count": len(parent_records),
              "candidate_figure_count": len(candidate_records), "pdf_page_count": page_number,
              "all_input_rows_plotted": True, "png_dpi": 180, "pdf_raster_dpi": 140,
              "chinese_font_available": chinese, "font_family": font_names[0],
              "pdf_path": str(pdf_path), "html_path": str(gallery_path),
              "parents": parent_records, "candidates": candidate_records}
    (output / "figure_manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--pdf-marker", type=Path)
    arguments = parser.parse_args()
    mark_pdf_operation_started(arguments.pdf_marker)
    print(json.dumps(generate_figures(arguments.dataset_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
