"""Authenticated 30 s AIS-speed observations for the formal v2 dataset."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
from numbers import Real
import os
from pathlib import Path
import shutil
from typing import Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator


AIS_RAW_PROVENANCE = "UNIQUE_NEAREST_RAW_WITHIN_10S"
AIS_GAP_PROVENANCE = "INTERPOLATED_GAP_PCHIP"
AIS_UNAVAILABLE_PROVENANCE = "UNAVAILABLE"
AIS_FRESHNESS_SECONDS = 10.0
LOCAL_TIMEZONE = ZoneInfo("Asia/Shanghai")


def _timestamps(values: Sequence[datetime], name: str) -> tuple[datetime, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence")
    checked = tuple(values)
    for value in checked:
        if type(value) is not datetime:
            raise TypeError(f"{name} must contain exact datetime values")
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{name} must contain timezone-aware values")
    if any(right <= left for left, right in zip(checked, checked[1:])):
        raise ValueError(f"{name} must be strictly increasing")
    return checked


def _speeds(values: Sequence[float], expected: int) -> np.ndarray:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("source_speed_kn must be a sequence")
    if len(values) != expected:
        raise ValueError("source timestamp and speed lengths differ")
    output = np.empty(expected, dtype=float)
    for index, value in enumerate(values):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise TypeError("AIS speed must be a real non-bool scalar")
        speed = float(value)
        if not math.isfinite(speed):
            raise ValueError("AIS speed must be finite")
        if speed < 0.0:
            raise ValueError("AIS speed must be nonnegative")
        output[index] = speed
    return output


@dataclass(frozen=True)
class AlignedAisSpeed:
    speed_kn: np.ndarray
    provenance: tuple[str, ...]


def align_supervisory_speed(
    supervisory_timestamps: Sequence[datetime],
    source_timestamps: Sequence[datetime],
    source_speed_kn: Sequence[float],
    *,
    freshness_seconds: float = AIS_FRESHNESS_SECONDS,
    fill_internal_gaps: bool = True,
) -> AlignedAisSpeed:
    """Align unique near-synchronous AIS and PCHIP-fill only bracketed gaps.

    A source record can represent at most one supervisory point. Gap filling is
    offline reconstruction evidence, is never labelled measured, and cannot
    extrapolate outside the raw AIS time span.
    """

    targets = _timestamps(supervisory_timestamps, "supervisory_timestamps")
    sources = _timestamps(source_timestamps, "source_timestamps")
    speed = _speeds(source_speed_kn, len(sources))
    if isinstance(freshness_seconds, bool) or not isinstance(
        freshness_seconds, Real
    ):
        raise TypeError("freshness_seconds must be a real non-bool scalar")
    freshness = float(freshness_seconds)
    if not math.isfinite(freshness) or freshness < 0.0:
        raise ValueError("freshness_seconds must be finite and nonnegative")
    if type(fill_internal_gaps) is not bool:
        raise TypeError("fill_internal_gaps must be an exact bool")

    output = np.full(len(targets), np.nan, dtype=float)
    labels = [AIS_UNAVAILABLE_PROVENANCE] * len(targets)
    if not targets or not sources:
        return AlignedAisSpeed(output, tuple(labels))

    source_seconds = np.array(
        [(timestamp - sources[0]).total_seconds() for timestamp in sources],
        dtype=float,
    )
    target_seconds = np.array(
        [(timestamp - sources[0]).total_seconds() for timestamp in targets],
        dtype=float,
    )
    last_used = -1
    for index, target in enumerate(target_seconds):
        insertion = int(np.searchsorted(source_seconds, target, side="left"))
        candidates = tuple(
            candidate
            for candidate in (insertion - 1, insertion)
            if last_used < candidate < len(sources)
        )
        distances = sorted(
            (abs(source_seconds[candidate] - target), candidate)
            for candidate in candidates
            if abs(source_seconds[candidate] - target) <= freshness
        )
        if not distances:
            continue
        if len(distances) > 1 and distances[0][0] == distances[1][0]:
            continue
        candidate = distances[0][1]
        output[index] = speed[candidate]
        labels[index] = AIS_RAW_PROVENANCE
        last_used = candidate

    if fill_internal_gaps and len(sources) >= 2:
        interpolator = PchipInterpolator(source_seconds, speed, extrapolate=False)
        for index in np.flatnonzero(~np.isfinite(output)):
            target = target_seconds[index]
            right = int(np.searchsorted(source_seconds, target, side="right"))
            left = right - 1
            if left < 0 or right >= len(sources):
                continue
            value = float(interpolator(target))
            if not math.isfinite(value):
                continue
            lower = min(speed[left], speed[right])
            upper = max(speed[left], speed[right])
            output[index] = min(upper, max(lower, value))
            labels[index] = AIS_GAP_PROVENANCE

    return AlignedAisSpeed(output, tuple(labels))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_parent_ais(raw_root: Path, parent: str) -> tuple[tuple[datetime, ...], tuple[float, ...], tuple[Path, ...]]:
    paths = tuple(sorted((raw_root / parent / "推进系统").glob("AIS航速_*.csv")))
    if not paths:
        raise FileNotFoundError(f"{parent}: no AIS speed file")
    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_csv(path, encoding="utf-8-sig")
        if not {"Time", "航速(节)"}.issubset(frame.columns):
            raise ValueError(f"{path}: AIS schema is incomplete")
        values = frame[["Time", "航速(节)"]].copy()
        timestamps = pd.to_datetime(values["Time"], errors="coerce")
        if timestamps.isna().any():
            raise ValueError(f"{path}: AIS timestamp is invalid")
        if timestamps.dt.tz is None:
            timestamps = timestamps.dt.tz_localize(LOCAL_TIMEZONE)
        else:
            timestamps = timestamps.dt.tz_convert(LOCAL_TIMEZONE)
        values["timestamp"] = timestamps
        values["speed_kn"] = pd.to_numeric(
            values["航速(节)"].astype(str).str.replace(r"\s*kn$", "", regex=True),
            errors="coerce",
        )
        if values["speed_kn"].isna().any():
            raise ValueError(f"{path}: AIS speed is invalid")
        if values["speed_kn"].lt(0.0).any():
            raise ValueError(f"{path}: AIS speed must be nonnegative")
        frames.append(values[["timestamp", "speed_kn"]])
    merged = pd.concat(frames, ignore_index=True).sort_values("timestamp", kind="stable")
    conflicts = merged.groupby("timestamp")["speed_kn"].nunique().gt(1)
    if conflicts.any():
        raise ValueError(f"{parent}: conflicting AIS timestamps")
    merged = merged.drop_duplicates("timestamp", keep="first").reset_index(drop=True)
    return (
        tuple(value.to_pydatetime() for value in merged["timestamp"]),
        tuple(float(value) for value in merged["speed_kn"]),
        paths,
    )


def build_ais_speed_sidecar(
    dataset_root: Path,
    raw_root: Path,
    output_root: Path,
) -> dict[str, object]:
    """Build an atomic per-segment 30 s AIS sidecar without changing power data."""

    dataset = Path(dataset_root).resolve()
    raw = Path(raw_root).resolve()
    output = Path(output_root).resolve()
    if output.exists():
        raise FileExistsError(f"destination already exists: {output}")
    manifest_path = dataset / "metadata" / "sample_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"dataset manifest is missing: {manifest_path}")
    manifest = pd.read_csv(manifest_path, encoding="utf-8-sig")
    required = {"parent", "sample_id", "relative_path", "split", "duration_s"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"dataset manifest is missing columns: {sorted(missing)}")
    if manifest.empty or manifest["sample_id"].duplicated().any():
        raise ValueError("dataset manifest must contain unique segments")

    temporary = output.with_name(f"{output.name}.building-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"temporary destination already exists: {temporary}")
    (temporary / "metadata").mkdir(parents=True)
    rows: list[dict[str, object]] = []
    source_rows: list[dict[str, object]] = []
    try:
        for item in manifest.itertuples(index=False):
            split = str(item.split)
            relative = Path(str(item.relative_path))
            segment_path = (dataset / relative).resolve()
            try:
                segment_path.relative_to(dataset)
            except ValueError as exc:
                raise ValueError("segment path escapes dataset root") from exc
            frame = pd.read_csv(segment_path, encoding="utf-8-sig")
            if not {"timestamp", "time_s"}.issubset(frame.columns):
                raise ValueError(f"{item.sample_id}: segment time schema is incomplete")
            time_s = pd.to_numeric(frame["time_s"], errors="coerce")
            duration = float(item.duration_s)
            selected = frame.loc[
                time_s.mod(30.0).eq(0.0) & time_s.add(30.0).le(duration),
                ["timestamp", "time_s"],
            ].copy()
            timestamps = pd.to_datetime(selected["timestamp"], errors="coerce")
            if timestamps.isna().any():
                raise ValueError(f"{item.sample_id}: invalid supervisory timestamp")
            if timestamps.dt.tz is None:
                timestamps = timestamps.dt.tz_localize(LOCAL_TIMEZONE)
            else:
                timestamps = timestamps.dt.tz_convert(LOCAL_TIMEZONE)
            source_times, source_speed, source_paths = _read_parent_ais(
                raw, str(item.parent)
            )
            aligned = align_supervisory_speed(
                tuple(value.to_pydatetime() for value in timestamps),
                source_times,
                source_speed,
            )
            if not np.isfinite(aligned.speed_kn).all():
                raise ValueError(
                    f"{item.sample_id}: AIS coverage cannot be reconstructed without extrapolation"
                )
            destination = temporary / split / f"{item.sample_id}.csv"
            destination.parent.mkdir(parents=True, exist_ok=True)
            output_frame = pd.DataFrame(
                {
                    "timestamp": [value.isoformat() for value in timestamps],
                    "time_s": selected["time_s"].to_numpy(dtype=float),
                    "speed_kn": aligned.speed_kn,
                    "speed_provenance": aligned.provenance,
                }
            )
            output_frame.to_csv(destination, index=False, encoding="utf-8-sig")
            counts = output_frame["speed_provenance"].value_counts().to_dict()
            rows.append(
                {
                    "parent": str(item.parent),
                    "sample_id": str(item.sample_id),
                    "split": split,
                    "relative_path": destination.relative_to(temporary).as_posix(),
                    "supervisory_row_count": len(output_frame),
                    "aligned_raw_count": int(counts.get(AIS_RAW_PROVENANCE, 0)),
                    "interpolated_gap_count": int(counts.get(AIS_GAP_PROVENANCE, 0)),
                    "minimum_speed_kn": float(output_frame["speed_kn"].min()),
                    "maximum_speed_kn": float(output_frame["speed_kn"].max()),
                    "sha256": _sha256(destination),
                }
            )
            for source_path in source_paths:
                source_rows.append(
                    {
                        "parent": str(item.parent),
                        "relative_path": source_path.relative_to(raw).as_posix(),
                        "sha256": _sha256(source_path),
                    }
                )
        sidecar_manifest = pd.DataFrame(rows)
        sidecar_manifest.to_csv(
            temporary / "metadata" / "sample_manifest.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(source_rows).drop_duplicates().to_csv(
            temporary / "metadata" / "source_files.csv",
            index=False,
            encoding="utf-8-sig",
        )
        split_counts = {
            str(key): int(value)
            for key, value in sidecar_manifest["split"].value_counts().sort_index().items()
        }
        summary: dict[str, object] = {
            "schema_version": "v2_ais_speed_30s_v1",
            "segment_count": len(sidecar_manifest),
            "split_segment_counts": split_counts,
            "supervisory_row_count": int(sidecar_manifest["supervisory_row_count"].sum()),
            "aligned_raw_count": int(sidecar_manifest["aligned_raw_count"].sum()),
            "interpolated_gap_count": int(sidecar_manifest["interpolated_gap_count"].sum()),
            "speed_normalization_kn": 20.0,
        }
        (temporary / "metadata" / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
        return summary
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


__all__ = [
    "AIS_FRESHNESS_SECONDS",
    "AIS_GAP_PROVENANCE",
    "AIS_RAW_PROVENANCE",
    "AIS_UNAVAILABLE_PROVENANCE",
    "AlignedAisSpeed",
    "align_supervisory_speed",
    "build_ais_speed_sidecar",
]
