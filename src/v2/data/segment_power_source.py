"""Audited raw-channel assembly for parent-segment power review and datasets."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from datetime import datetime
import math
from pathlib import Path
from typing import Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from utils.final_dataset_source import _read, collapse_channel
from v2.data.train_supervisory_audit import (
    ParentRawChannels,
    RawChannel,
    RawRecord,
    resolve_duplicates,
)


LOCAL_TIMEZONE = ZoneInfo("Asia/Shanghai")
ALIGNMENT_TOLERANCE_SECONDS = 10.0


@dataclass(frozen=True)
class LoadedParent:
    channels: ParentRawChannels
    duplicate_count: int
    duplicate_conflict_count: int


@dataclass(frozen=True)
class CycleAlignment:
    channel_indices: tuple[tuple[int | None, ...], ...]
    valid_reference_positions: tuple[int, ...]
    snapshot_timestamps: tuple[datetime, ...]
    channel_span_violation_count: int


@dataclass(frozen=True)
class ParentPowerSeries:
    parent: str
    timestamps: tuple[datetime, ...]
    fc_total_kw: np.ndarray
    battery_raw_total_kw: np.ndarray
    source_total_kw: np.ndarray
    ais_present: np.ndarray
    duplicate_count: int
    duplicate_conflict_count: int
    channel_span_violation_count: int


def power_conventions(
    fc_kw: Sequence[float],
    battery_bus_kw: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert bus-positive battery power to raw sign and derive source power."""
    fc = np.asarray(fc_kw, dtype=float)
    battery_bus = np.asarray(battery_bus_kw, dtype=float)
    if fc.shape != battery_bus.shape:
        raise ValueError("FC and battery arrays must have identical shapes")
    if not np.isfinite(fc).all() or not np.isfinite(battery_bus).all():
        raise ValueError("FC and battery power must be finite")
    battery_raw = -battery_bus
    source = fc + battery_bus
    if not np.allclose(source, fc - battery_raw, rtol=0.0, atol=1.0e-12):
        raise AssertionError("source-power identity failed")
    return battery_raw, source


def align_near_synchronous_cycles(
    reference_timestamps: Sequence[datetime],
    channel_timestamps: Sequence[Sequence[datetime]],
    *,
    tolerance_seconds: float,
    max_channel_span_seconds: float = ALIGNMENT_TOLERANCE_SECONDS,
) -> CycleAlignment:
    """Match one unused nearest sample per channel at the latest arrival."""
    reference = tuple(reference_timestamps)
    sources = tuple(tuple(values) for values in channel_timestamps)
    if not sources:
        raise ValueError("at least one channel is required")
    if tolerance_seconds < 0.0 or max_channel_span_seconds < 0.0:
        raise ValueError("alignment tolerances must be nonnegative")
    if any(right <= left for left, right in zip(reference, reference[1:])):
        raise ValueError("reference timestamps must be strictly increasing")

    mappings: list[tuple[int | None, ...]] = []
    for source in sources:
        if any(right <= left for left, right in zip(source, source[1:])):
            raise ValueError("source timestamps must be strictly increasing")
        last_used = -1
        matched: list[int | None] = []
        for timestamp in reference:
            insertion = bisect_left(source, timestamp)
            candidates = tuple(
                index
                for index in (insertion - 1, insertion)
                if last_used < index < len(source)
            )
            within = tuple(
                (abs((source[index] - timestamp).total_seconds()), index)
                for index in candidates
                if abs((source[index] - timestamp).total_seconds())
                <= tolerance_seconds
            )
            if not within:
                matched.append(None)
                continue
            ordered = sorted(within)
            if len(ordered) > 1 and ordered[0][0] == ordered[1][0]:
                matched.append(None)
                continue
            selected = ordered[0][1]
            matched.append(selected)
            last_used = selected
        mappings.append(tuple(matched))

    valid_positions: list[int] = []
    snapshots: list[datetime] = []
    span_violations = 0
    for position in range(len(reference)):
        selected_indices = tuple(mapping[position] for mapping in mappings)
        if any(index is None for index in selected_indices):
            continue
        selected_times = tuple(
            source[index]
            for source, index in zip(sources, selected_indices)
            if index is not None
        )
        snapshot = max(selected_times)
        if (snapshot - min(selected_times)).total_seconds() > max_channel_span_seconds:
            span_violations += 1
            continue
        valid_positions.append(position)
        snapshots.append(snapshot)
    return CycleAlignment(
        tuple(mappings),
        tuple(valid_positions),
        tuple(snapshots),
        span_violations,
    )


def assemble_parent_power_series(
    channels: ParentRawChannels,
    *,
    reader_duplicate_count: int = 0,
    reader_duplicate_conflict_count: int = 0,
    tolerance_seconds: float = ALIGNMENT_TOLERANCE_SECONDS,
) -> ParentPowerSeries:
    """Assemble complete near-synchronous 8-FC and 12-BMS power snapshots."""
    all_channels = (
        channels.fuel_cell_channels
        + channels.battery_channels
        + (channels.speed_channel,)
    )
    resolved = tuple(resolve_duplicates(channel.records) for channel in all_channels)
    clock = tuple(record.timestamp for record in resolved[0].records)
    if not clock:
        raise ValueError(f"{channels.parent_id}: empty FC reference channel")
    alignment = align_near_synchronous_cycles(
        clock,
        tuple(
            tuple(record.timestamp for record in resolution.records)
            for resolution in resolved[:20]
        ),
        tolerance_seconds=tolerance_seconds,
        max_channel_span_seconds=tolerance_seconds,
    )
    valid_positions = alignment.valid_reference_positions
    if not valid_positions:
        raise ValueError(
            f"{channels.parent_id}: no complete near-synchronous FC+BMS snapshot"
        )
    fc_values: list[float] = []
    battery_bus_values: list[float] = []
    for position in valid_positions:
        fc_values.append(
            math.fsum(
                resolved[channel].records[
                    alignment.channel_indices[channel][position]
                ].values[0]
                for channel in range(8)
                if alignment.channel_indices[channel][position] is not None
            )
        )
        battery_bus_values.append(
            math.fsum(
                resolved[channel].records[
                    alignment.channel_indices[channel][position]
                ].values[0]
                for channel in range(8, 20)
                if alignment.channel_indices[channel][position] is not None
            )
        )
    fc = np.asarray(fc_values, dtype=float)
    battery_raw, source = power_conventions(fc, battery_bus_values)
    timestamps = alignment.snapshot_timestamps
    ais_alignment = align_near_synchronous_cycles(
        timestamps,
        (tuple(record.timestamp for record in resolved[20].records),),
        tolerance_seconds=tolerance_seconds,
        max_channel_span_seconds=tolerance_seconds,
    )
    ais_present = np.asarray(
        [index is not None for index in ais_alignment.channel_indices[0]],
        dtype=bool,
    )
    duplicate_count = reader_duplicate_count + sum(
        resolution.exact_duplicate_rows_removed for resolution in resolved
    )
    duplicate_conflict_count = reader_duplicate_conflict_count + sum(
        len(resolution.conflicting_timestamps) for resolution in resolved
    )
    return ParentPowerSeries(
        parent=channels.parent_id,
        timestamps=timestamps,
        fc_total_kw=fc,
        battery_raw_total_kw=battery_raw,
        source_total_kw=source,
        ais_present=ais_present,
        duplicate_count=int(duplicate_count),
        duplicate_conflict_count=int(duplicate_conflict_count),
        channel_span_violation_count=alignment.channel_span_violation_count,
    )


def _aware(value: object) -> datetime:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(LOCAL_TIMEZONE)
    else:
        timestamp = timestamp.tz_convert(LOCAL_TIMEZONE)
    return timestamp.to_pydatetime()


def _raw_channel(
    channel_id: str,
    frame: pd.DataFrame,
    value_columns: tuple[str, ...],
    source_paths: tuple[str, ...],
) -> RawChannel:
    records = []
    for row_number, row in frame.iterrows():
        values = tuple(float(row[column]) for column in value_columns)
        if not all(np.isfinite(values)):
            continue
        timestamp = _aware(row["timestamp"])
        records.append(
            RawRecord(
                timestamp,
                values,
                f"{'|'.join(source_paths)}@{timestamp.isoformat()}#collapsed-row-{row_number}",
            )
        )
    return RawChannel(channel_id, tuple(records))


def _load_parent(raw_root: Path, parent_id: str) -> LoadedParent:
    """Read one parent with the audited raw CSV collapse rules."""
    parent_root = raw_root / parent_id
    if not parent_root.is_dir():
        raise FileNotFoundError(f"missing parent directory: {parent_root}")
    fc_channels: list[RawChannel] = []
    battery_channels: list[RawChannel] = []
    qa: list[dict[str, object]] = []
    for side in ("左", "右"):
        for number in range(1, 5):
            channel_id = f"{side}氢燃料电池#{number}"
            source_rows: list[dict[str, object]] = []
            frame, stats = _read(
                parent_root / "燃料电池系统",
                channel_id,
                {"发电功率(kW)": "power_kw"},
                source_rows,
            )
            qa.append(stats)
            fc_channels.append(
                _raw_channel(
                    channel_id,
                    frame.dropna(subset=["power_kw"]),
                    ("power_kw",),
                    tuple(str(row["path"]) for row in source_rows),
                )
            )
        for number in range(1, 7):
            channel_id = f"{side}电池簇{number}"
            source_rows = []
            frame, stats = _read(
                parent_root / "BMS",
                channel_id,
                {
                    "总电压(V)": "voltage_v",
                    "总电流(A)": "current_a",
                    "SOC(%)": "soc_pct",
                },
                source_rows,
            )
            qa.append(stats)
            frame["power_kw"] = -(
                frame["voltage_v"] * frame["current_a"]
            ) / 1000.0
            frame.loc[frame["voltage_v"].le(0), "power_kw"] = np.nan
            frame.loc[~frame["soc_pct"].between(0, 100), "soc_pct"] = np.nan
            frame["soc"] = frame["soc_pct"] / 100.0
            battery_channels.append(
                _raw_channel(
                    channel_id,
                    frame.dropna(subset=["power_kw", "soc"]),
                    ("power_kw", "soc"),
                    tuple(str(row["path"]) for row in source_rows),
                )
            )

    ais_paths = sorted((parent_root / "推进系统").glob("AIS航速_*.csv"))
    if not ais_paths:
        speed = RawChannel("ais-speed", ())
    else:
        frames = tuple(pd.read_csv(path, encoding="utf-8-sig") for path in ais_paths)
        raw = pd.concat(frames, ignore_index=True).rename(
            columns={"Time": "timestamp", "航速(节)": "speed_kn"}
        )
        raw["speed_kn"] = pd.to_numeric(
            raw["speed_kn"].astype(str).str.replace(r"\s*kn$", "", regex=True),
            errors="coerce",
        )
        raw.loc[raw["speed_kn"].lt(0), "speed_kn"] = np.nan
        collapsed, stats = collapse_channel(raw, ["speed_kn"])
        qa.append(stats)
        speed = _raw_channel(
            "ais-speed",
            collapsed,
            ("speed_kn",),
            tuple(str(path) for path in ais_paths),
        )
    return LoadedParent(
        ParentRawChannels(
            parent_id,
            tuple(fc_channels),
            tuple(battery_channels),
            speed,
        ),
        sum(int(row.get("exact_duplicate_rows", 0)) for row in qa),
        sum(int(row.get("conflicting_timestamps", 0)) for row in qa),
    )


def load_parent_power_series(raw_root: Path, parent: str) -> ParentPowerSeries:
    """Load and assemble one raw parent into the reviewed power series."""
    loaded = _load_parent(Path(raw_root), str(parent))
    return assemble_parent_power_series(
        loaded.channels,
        reader_duplicate_count=loaded.duplicate_count,
        reader_duplicate_conflict_count=loaded.duplicate_conflict_count,
    )
