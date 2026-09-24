"""Train-only evidence construction for the v2 DQN state audit."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from ..data.train_supervisory_audit import ParentSupervisoryState


ACTIVE_DATASET_VERSION = "operating_dataset_zero_boundary_v2"
AUDIT_SAMPLE_SECONDS = 30.0
AUDIT_HISTORY_SECONDS = 150.0
AUDIT_TAU_LPF_SECONDS = 90.0
AUDIT_POWER_SCALE_KW = 600.0


@dataclass(frozen=True)
class TrainSegment:
    parent: str
    sample_id: str
    relative_path: str
    start_timestamp: datetime
    end_timestamp: datetime
    sha256: str
    dataset_version: str


@dataclass(frozen=True)
class AuditFeatureRow:
    parent: str
    sample_id: str
    timestamp: datetime
    soc: float
    fc_power_kw: float
    previous_fc_power_kw: float
    battery_power_kw: float
    load_power_kw: float
    recent_load_mean_kw: float
    recent_load_population_std_kw: float
    recent_load_trend_kw_per_s: float
    base_load_kw: float
    recent_delta_soc: float
    delta_load_kw: float
    delta_fc_kw: float
    measured_power_balance_residual_kw: float
    history_sample_count: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _train_path(root: Path, relative_path: str) -> Path:
    relative = PurePosixPath(relative_path)
    if (
        relative.is_absolute()
        or not relative.parts
        or relative.parts[0] != "train"
        or ".." in relative.parts
    ):
        raise ValueError(f"Train path must remain below train/: {relative_path!r}")
    resolved = (root / Path(*relative.parts)).resolve()
    train_root = (root / "train").resolve()
    try:
        resolved.relative_to(train_root)
    except ValueError as error:
        raise ValueError(
            f"Train path escapes the formal Train directory: {relative_path!r}"
        ) from error
    return resolved


def load_train_segments(dataset_root: str | Path) -> tuple[TrainSegment, ...]:
    """Load and authenticate only the active formal Train segment whitelist."""

    root = Path(dataset_root).resolve()
    metadata = root / "metadata"
    qa_path = metadata / "qa_summary.json"
    manifest_path = metadata / "sample_manifest.csv"
    if not qa_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("formal dataset metadata is incomplete")
    qa = json.loads(qa_path.read_text(encoding="utf-8-sig"))
    version = qa.get("dataset_version")
    if version != ACTIVE_DATASET_VERSION:
        raise ValueError(
            f"state audit requires {ACTIVE_DATASET_VERSION}, received {version!r}"
        )

    manifest = pd.read_csv(manifest_path, encoding="utf-8-sig")
    required = {
        "parent",
        "sample_id",
        "relative_path",
        "split",
        "start_timestamp",
        "end_timestamp",
        "sha256",
    }
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"formal manifest is missing columns: {sorted(missing)}")
    train = manifest.loc[manifest["split"].astype(str).str.lower().eq("train")].copy()
    if train.empty:
        raise ValueError("formal manifest has no Train segments")
    if train["sample_id"].isna().any() or train["sample_id"].duplicated().any():
        raise ValueError("Train segment identifiers must be unique and nonempty")

    segments: list[TrainSegment] = []
    for row in train.itertuples(index=False):
        relative_path = str(row.relative_path)
        path = _train_path(root, relative_path)
        if not path.is_file():
            raise FileNotFoundError(f"formal Train segment is missing: {path}")
        expected = str(row.sha256).lower()
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"Train segment SHA-256 mismatch: {relative_path}")
        start = pd.Timestamp(row.start_timestamp)
        end = pd.Timestamp(row.end_timestamp)
        if start.tzinfo is None or end.tzinfo is None or end < start:
            raise ValueError(f"Train segment has invalid timestamps: {row.sample_id}")
        segments.append(
            TrainSegment(
                parent=str(row.parent),
                sample_id=str(row.sample_id),
                relative_path=relative_path,
                start_timestamp=start.to_pydatetime(),
                end_timestamp=end.to_pydatetime(),
                sha256=actual,
                dataset_version=str(version),
            )
        )
    return tuple(
        sorted(segments, key=lambda item: (item.start_timestamp, item.sample_id))
    )


def _load_lookup(load_frame: pd.DataFrame) -> dict[int, float]:
    required = {"timestamp", "load_total_kw"}
    missing = required.difference(load_frame.columns)
    if missing:
        raise ValueError(f"Train load frame is missing columns: {sorted(missing)}")
    timestamps = pd.to_datetime(load_frame["timestamp"], utc=True, errors="raise")
    if timestamps.duplicated().any():
        raise ValueError("Train load timestamps must be unique")
    loads = pd.to_numeric(load_frame["load_total_kw"], errors="raise").to_numpy(
        dtype=float
    )
    if not np.isfinite(loads).all():
        raise ValueError("Train loads must be finite")
    return {
        int(timestamp.value): float(load)
        for timestamp, load in zip(timestamps, loads)
    }


def _timestamp_key(value: datetime) -> int:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError("supervisory timestamps must be timezone-aware")
    return int(timestamp.tz_convert("UTC").value)


def _least_squares_trend(
    timestamps: Sequence[datetime],
    loads: Sequence[float],
) -> float:
    origin = timestamps[0]
    elapsed = np.asarray(
        [(timestamp - origin).total_seconds() for timestamp in timestamps],
        dtype=float,
    )
    values = np.asarray(loads, dtype=float)
    centered_time = elapsed - float(np.mean(elapsed))
    denominator = float(np.dot(centered_time, centered_time))
    if denominator <= 0.0:
        raise ValueError("load trend requires distinct timestamps")
    centered_load = values - float(np.mean(values))
    return float(np.dot(centered_time, centered_load) / denominator)


def build_causal_feature_rows(
    segment: TrainSegment,
    states: Sequence[ParentSupervisoryState],
    load_frame: pd.DataFrame,
) -> tuple[AuditFeatureRow, ...]:
    """Build fully observed Train rows using only a segment-local causal window."""

    if type(segment) is not TrainSegment:
        raise TypeError("segment must be an exact TrainSegment")
    checked = tuple(states)
    if any(type(state) is not ParentSupervisoryState for state in checked):
        raise TypeError("states must contain exact ParentSupervisoryState values")
    if any(
        current.timestamp <= previous.timestamp
        for previous, current in zip(checked, checked[1:])
    ):
        raise ValueError("supervisory timestamps must be strictly increasing")
    lookup = _load_lookup(load_frame)
    alpha = math.exp(-AUDIT_SAMPLE_SECONDS / AUDIT_TAU_LPF_SECONDS)
    base_load: float | None = None
    history: list[tuple[ParentSupervisoryState, float, float]] = []
    rows: list[AuditFeatureRow] = []
    for state in checked:
        if state.timestamp < segment.start_timestamp:
            continue
        if state.timestamp > segment.end_timestamp:
            break
        if state.parent_id != segment.parent:
            raise ValueError("supervisory parent does not match Train segment")
        key = _timestamp_key(state.timestamp)
        if key not in lookup:
            continue
        load = lookup[key]
        if load <= 0.0:
            continue
        base_load = load if base_load is None else alpha * base_load + (1.0 - alpha) * load
        if not state.audit_eligible:
            continue
        if (
            state.p_fc_total_kw is None
            or state.p_batt_total_kw is None
            or state.soc_system is None
            or state.previous_p_fc_total_kw is None
        ):
            continue
        history.append((state, load, base_load))
        cutoff = state.timestamp.timestamp() - AUDIT_HISTORY_SECONDS
        history = [
            item for item in history if item[0].timestamp.timestamp() >= cutoff
        ]
        if len(history) < 6:
            continue
        elapsed = (history[-1][0].timestamp - history[0][0].timestamp).total_seconds()
        if elapsed < AUDIT_HISTORY_SECONDS:
            continue
        window_states = tuple(item[0] for item in history)
        window_loads = tuple(item[1] for item in history)
        current = window_states[-1]
        current_base = history[-1][2]
        mean = float(np.mean(np.asarray(window_loads, dtype=float)))
        std = float(np.std(np.asarray(window_loads, dtype=float), ddof=0))
        trend = _least_squares_trend(
            tuple(item.timestamp for item in window_states),
            window_loads,
        )
        fc = float(current.p_fc_total_kw)
        previous_fc = float(current.previous_p_fc_total_kw)
        battery = float(current.p_batt_total_kw)
        soc = float(current.soc_system)
        rows.append(
            AuditFeatureRow(
                parent=segment.parent,
                sample_id=segment.sample_id,
                timestamp=current.timestamp,
                soc=soc,
                fc_power_kw=fc,
                previous_fc_power_kw=previous_fc,
                battery_power_kw=battery,
                load_power_kw=window_loads[-1],
                recent_load_mean_kw=mean,
                recent_load_population_std_kw=std,
                recent_load_trend_kw_per_s=trend,
                base_load_kw=current_base,
                recent_delta_soc=soc - float(window_states[0].soc_system),
                delta_load_kw=window_loads[-1] - current_base,
                delta_fc_kw=fc - previous_fc,
                measured_power_balance_residual_kw=(
                    window_loads[-1] - fc - battery
                ),
                history_sample_count=len(history),
            )
        )
    return tuple(rows)


def build_train_feature_rows(
    dataset_root: str | Path,
    segments: Sequence[TrainSegment],
    state_loader: Callable[[str], Sequence[ParentSupervisoryState]],
) -> tuple[AuditFeatureRow, ...]:
    """Build rows for an already authenticated Train whitelist only."""

    if not callable(state_loader):
        raise TypeError("state_loader must be callable")
    root = Path(dataset_root).resolve()
    checked = tuple(segments)
    if not checked or any(type(segment) is not TrainSegment for segment in checked):
        raise TypeError("segments must contain TrainSegment values")
    state_cache: dict[str, tuple[ParentSupervisoryState, ...]] = {}
    rows: list[AuditFeatureRow] = []
    for segment in checked:
        path = _train_path(root, segment.relative_path)
        if segment.parent not in state_cache:
            state_cache[segment.parent] = tuple(state_loader(segment.parent))
        load_frame = pd.read_csv(
            path,
            usecols=["timestamp", "time_s", "load_total_kw"],
            encoding="utf-8-sig",
        )
        rows.extend(
            build_causal_feature_rows(
                segment,
                state_cache[segment.parent],
                load_frame,
            )
        )
    return tuple(rows)


__all__ = [
    "ACTIVE_DATASET_VERSION",
    "AUDIT_HISTORY_SECONDS",
    "AUDIT_POWER_SCALE_KW",
    "AUDIT_SAMPLE_SECONDS",
    "AUDIT_TAU_LPF_SECONDS",
    "AuditFeatureRow",
    "TrainSegment",
    "build_causal_feature_rows",
    "build_train_feature_rows",
    "load_train_segments",
]
