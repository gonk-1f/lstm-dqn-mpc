"""Deterministic per-episode power plots for one frozen formal policy."""

from __future__ import annotations

import csv
from html import escape
import io
import math
import os
from pathlib import Path
import re
import shutil
import uuid

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .formal_policy import EpisodePowerTrace


_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")
_MANIFEST_FIELDS = (
    "policy_id",
    "sample_id",
    "status",
    "failure_kind",
    "interval_count",
    "duration_s",
    "minimum_load_kw",
    "maximum_load_kw",
    "image_path",
)


def _filename(sample_id: str) -> str:
    value = _SAFE_NAME.sub("_", sample_id).strip("._")
    if not value:
        raise ValueError("sample_id does not contain a usable filename")
    return f"{value}.png"


def _shore_spans(trace: EpisodePowerTrace) -> tuple[tuple[float, float], ...]:
    if not trace.time_s:
        return ()
    step = (
        trace.time_s[1] - trace.time_s[0]
        if len(trace.time_s) > 1
        else trace.soc_time_s[-1] - trace.soc_time_s[0]
    )
    spans: list[tuple[float, float]] = []
    start: float | None = None
    for time_s, mode in zip(trace.time_s, trace.operating_mode):
        shore = mode != "onboard"
        if shore and start is None:
            start = time_s
        elif not shore and start is not None:
            spans.append((start, time_s))
            start = None
    if start is not None:
        spans.append((start, trace.time_s[-1] + step))
    return tuple(spans)


def _plot(path: Path, trace: EpisodePowerTrace, policy_id: str) -> None:
    figure, axis = plt.subplots(figsize=(12.0, 5.2), constrained_layout=True)
    if trace.time_s:
        axis.plot(trace.time_s, trace.load_power_kw, color="#16324F", linewidth=1.4, label="Load / total power")
        axis.plot(trace.time_s, trace.fuel_cell_power_kw, color="#E07A1F", linewidth=1.2, label="Fuel-cell power")
        axis.plot(trace.time_s, trace.battery_bus_power_kw, color="#179C72", linewidth=1.2, label="Battery bus power (+ discharge, - charge)")
        for index, (left, right) in enumerate(_shore_spans(trace)):
            axis.axvspan(
                left,
                right,
                color="#7AA6C2",
                alpha=0.16,
                label="Shore interval" if index == 0 else None,
            )
    axis.axhline(0.0, color="#666666", linewidth=0.8, linestyle="--")
    axis.set_xlabel("Elapsed time (s)")
    axis.set_ylabel("Power (kW)")
    axis.grid(True, linewidth=0.45, alpha=0.25)
    soc_axis = axis.twinx()
    soc_axis.plot(trace.soc_time_s, trace.soc, color="#8A3FFC", linewidth=1.0, linestyle=":", label="SOC")
    soc_axis.set_ylabel("SOC")
    soc_axis.set_ylim(0.18, 0.82)
    status = "COMPLETE" if trace.completed else f"FAILED: {trace.failure_kind}"
    axis.set_title(f"{trace.sample_id} | {policy_id} | {status}", loc="left")
    lines, labels = axis.get_legend_handles_labels()
    soc_lines, soc_labels = soc_axis.get_legend_handles_labels()
    axis.legend(lines + soc_lines, labels + soc_labels, loc="best", fontsize=8)
    figure.savefig(path, dpi=150, metadata={"Software": "v2 H4 formal Test power plots"})
    plt.close(figure)


def _manifest_bytes(rows: list[dict[str, object]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=_MANIFEST_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _index_html(rows: list[dict[str, object]], artifact_title: str) -> str:
    cards = []
    for row in rows:
        title = escape(str(row["sample_id"]))
        status = escape(str(row["status"]))
        failure = escape(str(row["failure_kind"]))
        image = escape(str(row["image_path"]), quote=True)
        cards.append(
            f'<article><h2>{title}</h2><p>{status}'
            f'{(" | " + failure) if failure else ""}</p>'
            f'<a href="{image}"><img src="{image}" alt="{title}"></a></article>'
        )
    return (
        "<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">"
        f"<title>{escape(artifact_title)}</title><style>body{{font-family:sans-serif;"
        "max-width:1280px;margin:auto;padding:20px}article{margin:24px 0}"
        "img{width:100%;height:auto;border:1px solid #ddd}</style></head><body>"
        f"<h1>{escape(artifact_title)}</h1>"
        + "".join(cards)
        + "</body></html>"
    )


def write_power_trace_plots(
    output_directory: Path,
    traces: tuple[EpisodePowerTrace, ...],
    *,
    policy_id: str,
    artifact_title: str = "H4 frozen-policy Test power traces",
) -> Path:
    """Write one plot per trace plus an auditable CSV manifest and HTML index."""

    output = Path(output_directory)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if type(traces) is not tuple or not traces:
        raise ValueError("traces must be a nonempty exact tuple")
    if any(type(trace) is not EpisodePowerTrace for trace in traces):
        raise TypeError("traces must contain exact EpisodePowerTrace values")
    if type(policy_id) is not str or not policy_id:
        raise ValueError("policy_id must be a nonempty exact string")
    if type(artifact_title) is not str or not artifact_title:
        raise ValueError("artifact_title must be a nonempty exact string")
    names = tuple(_filename(trace.sample_id) for trace in traces)
    if len(set(names)) != len(names):
        raise ValueError("trace sample IDs produce duplicate plot filenames")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    temporary.mkdir()
    try:
        plot_root = temporary / "plots"
        plot_root.mkdir()
        rows: list[dict[str, object]] = []
        for trace, filename in zip(traces, names):
            _plot(plot_root / filename, trace, policy_id)
            loads = trace.load_power_kw
            rows.append(
                {
                    "policy_id": policy_id,
                    "sample_id": trace.sample_id,
                    "status": "COMPLETE" if trace.completed else "FAILED",
                    "failure_kind": trace.failure_kind or "",
                    "interval_count": len(trace.time_s),
                    "duration_s": trace.soc_time_s[-1],
                    "minimum_load_kw": min(loads) if loads else math.nan,
                    "maximum_load_kw": max(loads) if loads else math.nan,
                    "image_path": f"plots/{filename}",
                }
            )
        manifest = temporary / "power_trace_manifest.csv"
        manifest.write_bytes(_manifest_bytes(rows))
        (temporary / "index.html").write_text(
            _index_html(rows, artifact_title),
            encoding="utf-8",
        )
        temporary.rename(output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output / "power_trace_manifest.csv"


__all__ = ["write_power_trace_plots"]
