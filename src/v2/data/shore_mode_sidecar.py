"""Authenticated FC/BMS/AIS evidence on the frozen 30 s training axis."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Callable

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from .ais_speed_sidecar import AIS_GAP_PROVENANCE, AIS_RAW_PROVENANCE
from .power_gap_interpolation import interpolate_power_gaps
from .segment_power_source import ParentPowerSeries, load_parent_power_series
from .supervisory_rules import ModeSample, OperatingMode, classify_operating_modes


MODE_SIDECAR_SCHEMA_VERSION = "v2_shore_mode_30s_v1"
POWER_RAW_PROVENANCE = "ALIGNED_OR_CUBIC_ANCHOR"
POWER_PCHIP_PROVENANCE = "INTERPOLATED_COMPONENT_PCHIP"
EXPECTED_SPLIT_COUNTS = {"train": 38, "validation": 10, "test": 5}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_mode_payload(path: Path, expected_sha256: str) -> None:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(target)
    if type(expected_sha256) is not str or _sha256(target) != expected_sha256:
        raise ValueError(f"{target}: mode payload SHA-256 mismatch")


def _contained(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("manifest path escapes dataset root") from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _identity(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"parent", "sample_id", "relative_path", "split", "sha256"}
    if required.difference(frame.columns):
        raise ValueError("sidecar input manifest schema mismatch")
    if frame["sample_id"].duplicated().any():
        raise ValueError("sidecar input sample IDs must be unique")
    return frame[["parent", "sample_id", "split"]].sort_values("sample_id").reset_index(drop=True)


def _component_values(
    series: ParentPowerSeries,
    target: tuple[pd.Timestamp, ...],
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    filled = interpolate_power_gaps(
        series.timestamps,
        series.fc_total_kw,
        series.battery_raw_total_kw,
        step_seconds=30.0,
    )
    source_ns = np.asarray([pd.Timestamp(value).value for value in filled.timestamps], dtype=np.int64)
    target_ns = np.asarray([value.value for value in target], dtype=np.int64)
    if len(source_ns) < 2 or target_ns.min() < source_ns.min() or target_ns.max() > source_ns.max():
        raise ValueError(f"{series.parent}: component reconstruction would extrapolate")
    origin = int(source_ns[0])
    source_s = (source_ns - origin) / 1.0e9
    target_s = (target_ns - origin) / 1.0e9
    fc = np.asarray(PchipInterpolator(source_s, filled.fc_kw, extrapolate=False)(target_s), dtype=float)
    battery_raw = np.asarray(
        PchipInterpolator(source_s, filled.battery_raw_kw, extrapolate=False)(target_s),
        dtype=float,
    )
    if not np.isfinite(fc).all() or not np.isfinite(battery_raw).all():
        raise ValueError(f"{series.parent}: component reconstruction is nonfinite")
    anchors = set(int(value) for value in source_ns)
    provenance = tuple(
        POWER_RAW_PROVENANCE if int(value) in anchors else POWER_PCHIP_PROVENANCE
        for value in target_ns
    )
    return fc, -battery_raw, provenance


def _reason(mode: OperatingMode) -> str:
    return {
        OperatingMode.ONBOARD: "VALID_NON_SHORE_COMPOSITE",
        OperatingMode.SHORE_PENDING: "CAUSAL_SHORE_CANDIDATE_PENDING",
        OperatingMode.SHORE_CHARGING: "THREE_SAMPLE_SHORE_SIGNATURE",
        OperatingMode.UNRESOLVED: "INSUFFICIENT_OR_CONTRADICTORY_EVIDENCE",
    }[mode]


def build_shore_mode_sidecar(
    power_root: Path,
    ais_root: Path,
    raw_root: Path,
    output_root: Path,
    *,
    load_parent: Callable[[Path, str], ParentPowerSeries] = load_parent_power_series,
    expected_split_counts: dict[str, int] = EXPECTED_SPLIT_COUNTS,
) -> dict[str, object]:
    """Build one immutable mode sidecar bound to power/AIS manifests."""

    power = Path(power_root).resolve()
    ais = Path(ais_root).resolve()
    raw = Path(raw_root).resolve()
    output = Path(output_root).resolve()
    if output.exists():
        raise FileExistsError(f"destination already exists: {output}")
    power_manifest_path = power / "metadata" / "sample_manifest.csv"
    ais_manifest_path = ais / "metadata" / "sample_manifest.csv"
    if not power_manifest_path.is_file() or not ais_manifest_path.is_file():
        raise FileNotFoundError("power/AIS manifest is missing")
    power_manifest = pd.read_csv(power_manifest_path, encoding="utf-8-sig")
    ais_manifest = pd.read_csv(ais_manifest_path, encoding="utf-8-sig")
    if not _identity(power_manifest).equals(_identity(ais_manifest)):
        raise ValueError("power and AIS identities/splits differ")
    counts = {
        split: int((power_manifest["split"] == split).sum())
        for split in ("train", "validation", "test")
    }
    if counts != expected_split_counts:
        raise ValueError(f"sidecar split counts differ: {counts}")

    temporary = output.with_name(f"{output.name}.building-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"temporary destination already exists: {temporary}")
    temporary.mkdir(parents=True)
    try:
        (temporary / "metadata").mkdir()
        for split in ("train", "validation", "test"):
            (temporary / split).mkdir()
        ais_by_id = ais_manifest.set_index("sample_id")
        rows: list[dict[str, object]] = []
        mode_counts: Counter[str] = Counter()
        split_mode_counts: dict[str, Counter[str]] = {
            split: Counter() for split in ("train", "validation", "test")
        }
        for power_row in power_manifest.sort_values("sample_id").itertuples(index=False):
            ais_row = ais_by_id.loc[str(power_row.sample_id)]
            power_path = _contained(power, str(power_row.relative_path))
            ais_path = _contained(ais, str(ais_row.relative_path))
            if _sha256(power_path) != str(power_row.sha256):
                raise ValueError(f"{power_row.sample_id}: power SHA-256 mismatch")
            if _sha256(ais_path) != str(ais_row.sha256):
                raise ValueError(f"{power_row.sample_id}: AIS SHA-256 mismatch")
            power_frame = pd.read_csv(power_path, encoding="utf-8-sig")
            ais_frame = pd.read_csv(ais_path, encoding="utf-8-sig")
            duration = float(power_row.duration_s)
            time_values = pd.to_numeric(power_frame["time_s"], errors="coerce")
            selected = power_frame.loc[
                time_values.mod(30.0).eq(0.0) & time_values.add(30.0).le(duration),
                ["timestamp", "time_s", "load_total_kw"],
            ].reset_index(drop=True)
            if len(selected) != len(ais_frame):
                raise ValueError(f"{power_row.sample_id}: power/AIS axes differ")
            target = tuple(pd.to_datetime(selected["timestamp"], errors="raise"))
            ais_timestamps = tuple(pd.to_datetime(ais_frame["timestamp"], errors="raise"))
            if target != ais_timestamps or not np.array_equal(
                selected["time_s"].to_numpy(dtype=float),
                ais_frame["time_s"].to_numpy(dtype=float),
            ):
                raise ValueError(f"{power_row.sample_id}: power/AIS axes differ")
            series = load_parent(raw, str(power_row.parent))
            if type(series) is not ParentPowerSeries or series.parent != str(power_row.parent):
                raise ValueError("raw parent loader returned the wrong identity")
            fc, battery_bus, power_provenance = _component_values(series, target)
            load = pd.to_numeric(
                selected["load_total_kw"], errors="coerce"
            ).to_numpy(dtype=float)
            if not np.isfinite(load).all():
                raise ValueError(f"{power_row.sample_id}: invalid frozen load")
            balance_residual = fc + battery_bus - load
            speed = pd.to_numeric(ais_frame["speed_kn"], errors="coerce").to_numpy(dtype=float)
            speed_provenance = tuple(str(value) for value in ais_frame["speed_provenance"])
            if not np.isfinite(speed).all() or (speed < 0.0).any():
                raise ValueError(f"{power_row.sample_id}: invalid AIS speed")
            if not set(speed_provenance).issubset({AIS_RAW_PROVENANCE, AIS_GAP_PROVENANCE}):
                raise ValueError(f"{power_row.sample_id}: invalid AIS provenance")
            conflict = series.duplicate_conflict_count > 0
            complete = series.channel_span_violation_count == 0 and not conflict
            samples = tuple(
                ModeSample(
                    timestamp.to_pydatetime(),
                    float(speed[index]),
                    float(fc[index]),
                    float(battery_bus[index]),
                    channels_complete=complete,
                    conflicting_duplicate=conflict,
                    long_gap_contaminated=False,
                    p_load_kw=float(load[index]),
                )
                for index, timestamp in enumerate(target)
            )
            modes = classify_operating_modes(samples)
            payload = pd.DataFrame(
                {
                    "timestamp": [value.isoformat() for value in target],
                    "time_s": selected["time_s"].to_numpy(dtype=float),
                    "load_total_kw": load,
                    "p_fc_total_kw": fc,
                    "p_batt_bus_kw": battery_bus,
                    "component_balance_residual_kw": balance_residual,
                    "power_provenance": power_provenance,
                    "speed_kn": speed,
                    "speed_provenance": speed_provenance,
                    "channels_complete": [complete] * len(target),
                    "freshness_valid": [True] * len(target),
                    "duplicate_conflict": [conflict] * len(target),
                    "mode": [mode.value for mode in modes],
                    "mode_reason": [_reason(mode) for mode in modes],
                }
            )
            relative = Path(str(power_row.split)) / f"{power_row.sample_id}.csv"
            path = temporary / relative
            payload.to_csv(path, index=False, encoding="utf-8-sig")
            local_counts = Counter(mode.value for mode in modes)
            mode_counts.update(local_counts)
            split_mode_counts[str(power_row.split)].update(local_counts)
            rows.append(
                {
                    "parent": str(power_row.parent),
                    "sample_id": str(power_row.sample_id),
                    "relative_path": relative.as_posix(),
                    "split": str(power_row.split),
                    "supervisory_row_count": len(payload),
                    "unresolved_row_count": local_counts[OperatingMode.UNRESOLVED.value],
                    "sha256": _sha256(path),
                }
            )
        pd.DataFrame(rows).to_csv(
            temporary / "metadata" / "sample_manifest.csv",
            index=False,
            encoding="utf-8-sig",
        )
        policy = {
            "schema_version": MODE_SIDECAR_SCHEMA_VERSION,
            "power_manifest_sha256": _sha256(power_manifest_path),
            "ais_manifest_sha256": _sha256(ais_manifest_path),
            "battery_bus_sign": "positive discharge, negative charge",
            "component_interpolation": "PCHIP within bracketed reconstructed raw-component span",
            "mode_counts": dict(sorted(mode_counts.items())),
            "split_mode_counts": {
                split: dict(sorted(values.items()))
                for split, values in split_mode_counts.items()
            },
        }
        (temporary / "metadata" / "policy.json").write_text(
            json.dumps(policy, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
        return policy
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


__all__ = [
    "EXPECTED_SPLIT_COUNTS",
    "MODE_SIDECAR_SCHEMA_VERSION",
    "POWER_PCHIP_PROVENANCE",
    "POWER_RAW_PROVENANCE",
    "build_shore_mode_sidecar",
    "verify_mode_payload",
]
