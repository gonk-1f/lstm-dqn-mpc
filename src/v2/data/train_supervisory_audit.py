"""Strict causal construction of Train supervisory states for one raw parent."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime
import math
from numbers import Real
from typing import Sequence

from .supervisory_rules import (
    LONG_GAP_SECONDS,
    ModeSample,
    OperatingMode,
    classify_operating_modes,
    is_fresh_causal_age,
    reconstruct_sailing_load,
)


def _timestamp(value: object, name: str) -> datetime:
    if type(value) is not datetime:
        raise TypeError(f"{name} must be an exact datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real non-bool scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _text(value: object, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be a nonempty exact string")
    return value


@dataclass(frozen=True)
class RawRecord:
    timestamp: datetime
    values: tuple[float, ...]
    provenance: str

    def __post_init__(self) -> None:
        _timestamp(self.timestamp, "timestamp")
        if type(self.values) is not tuple or not self.values:
            raise TypeError("values must be a nonempty exact tuple")
        object.__setattr__(
            self,
            "values",
            tuple(_finite(value, "record value") for value in self.values),
        )
        object.__setattr__(self, "provenance", _text(self.provenance, "provenance"))


@dataclass(frozen=True)
class RawChannel:
    channel_id: str
    records: tuple[RawRecord, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "channel_id", _text(self.channel_id, "channel_id"))
        if type(self.records) is not tuple or any(
            type(record) is not RawRecord for record in self.records
        ):
            raise TypeError("records must be an exact tuple of RawRecord values")


@dataclass(frozen=True)
class DuplicateResolution:
    records: tuple[RawRecord, ...]
    exact_duplicate_rows_removed: int
    conflicting_timestamps: tuple[datetime, ...]


def resolve_duplicates(records: Sequence[RawRecord]) -> DuplicateResolution:
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise TypeError("records must be a sequence")
    checked = tuple(records)
    if any(type(record) is not RawRecord for record in checked):
        raise TypeError("records must contain exact RawRecord values")
    groups: dict[datetime, list[RawRecord]] = {}
    for record in checked:
        groups.setdefault(record.timestamp, []).append(record)

    kept: list[RawRecord] = []
    exact_removed = 0
    conflicts: list[datetime] = []
    for timestamp in sorted(groups):
        group = groups[timestamp]
        distinct = {record.values for record in group}
        if len(distinct) != 1:
            conflicts.append(timestamp)
            continue
        kept.append(group[0])
        exact_removed += len(group) - 1
    return DuplicateResolution(tuple(kept), exact_removed, tuple(conflicts))


def causal_align_latest_without_reuse(
    supervisory_timestamps: Sequence[datetime],
    source_timestamps: Sequence[datetime],
) -> tuple[int | None, ...]:
    supervisory = tuple(
        _timestamp(value, "supervisory timestamp") for value in supervisory_timestamps
    )
    source = tuple(_timestamp(value, "source timestamp") for value in source_timestamps)
    if any(right <= left for left, right in zip(supervisory, supervisory[1:])):
        raise ValueError("supervisory timestamps must be strictly increasing")
    if any(right <= left for left, right in zip(source, source[1:])):
        raise ValueError("source timestamps must be strictly increasing")

    aligned: list[int | None] = []
    last_used = -1
    for timestamp in supervisory:
        candidate = bisect_right(source, timestamp) - 1
        if candidate <= last_used:
            aligned.append(None)
            continue
        age = (timestamp - source[candidate]).total_seconds() if candidate >= 0 else -1.0
        if candidate < 0 or not is_fresh_causal_age(age):
            aligned.append(None)
            continue
        aligned.append(candidate)
        last_used = candidate
    return tuple(aligned)


@dataclass(frozen=True)
class ParentRawChannels:
    parent_id: str
    fuel_cell_channels: tuple[RawChannel, ...]
    battery_channels: tuple[RawChannel, ...]
    speed_channel: RawChannel

    def __post_init__(self) -> None:
        object.__setattr__(self, "parent_id", _text(self.parent_id, "parent_id"))
        if type(self.fuel_cell_channels) is not tuple or len(self.fuel_cell_channels) != 8:
            raise ValueError("exactly eight FC channels are required")
        if type(self.battery_channels) is not tuple or len(self.battery_channels) != 12:
            raise ValueError("exactly twelve battery channels are required")
        if any(type(channel) is not RawChannel for channel in self.fuel_cell_channels):
            raise TypeError("fuel_cell_channels must contain exact RawChannel values")
        if any(type(channel) is not RawChannel for channel in self.battery_channels):
            raise TypeError("battery_channels must contain exact RawChannel values")
        if type(self.speed_channel) is not RawChannel:
            raise TypeError("speed_channel must be an exact RawChannel")
        identifiers = tuple(
            channel.channel_id
            for channel in self.fuel_cell_channels + self.battery_channels
        ) + (self.speed_channel.channel_id,)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("channel IDs must be unique")


@dataclass(frozen=True)
class ParentSupervisoryState:
    parent_id: str
    timestamp: datetime
    mode: OperatingMode
    p_fc_total_kw: float | None
    p_batt_total_kw: float | None
    p_load_kw: float | None
    soc_system: float | None
    previous_p_fc_total_kw: float | None
    channels_complete: bool
    conflicting_duplicate: bool
    long_gap_contaminated: bool
    source_provenance: tuple[tuple[str, datetime, str], ...]

    @property
    def audit_eligible(self) -> bool:
        return (
            self.mode is OperatingMode.SAILING_ISLAND
            and self.channels_complete
            and not self.conflicting_duplicate
            and not self.long_gap_contaminated
            and self.p_load_kw is not None
            and self.soc_system is not None
            and self.previous_p_fc_total_kw is not None
        )


@dataclass(frozen=True)
class ParentSupervisoryResult:
    parent_id: str
    states: tuple[ParentSupervisoryState, ...]
    exact_duplicate_rows_removed: int
    conflicting_duplicate_count: int


def build_parent_supervisory_states(
    channels: ParentRawChannels,
) -> ParentSupervisoryResult:
    if type(channels) is not ParentRawChannels:
        raise TypeError("channels must be an exact ParentRawChannels")
    all_channels = (
        channels.fuel_cell_channels
        + channels.battery_channels
        + (channels.speed_channel,)
    )
    resolutions = tuple(resolve_duplicates(channel.records) for channel in all_channels)
    exact_removed = sum(result.exact_duplicate_rows_removed for result in resolutions)
    conflicting = {
        timestamp
        for result in resolutions
        for timestamp in result.conflicting_timestamps
    }
    clock = tuple(record.timestamp for record in resolutions[0].records)
    aligned = tuple(
        causal_align_latest_without_reuse(
            clock,
            tuple(record.timestamp for record in resolution.records),
        )
        for resolution in resolutions
    )

    raw_rows: list[dict[str, object]] = []
    previous_complete_fc: float | None = None
    previous_timestamp: datetime | None = None
    for position, timestamp in enumerate(clock):
        gap = (
            previous_timestamp is None
            or (timestamp - previous_timestamp).total_seconds() > LONG_GAP_SECONDS
        )
        indices = tuple(mapping[position] for mapping in aligned)
        complete = all(index is not None for index in indices)
        conflict_here = timestamp in conflicting
        provenance: list[tuple[str, datetime, str]] = []
        p_fc: float | None = None
        p_batt: float | None = None
        soc: float | None = None
        if complete and not conflict_here:
            selected = tuple(
                resolution.records[index]
                for resolution, index in zip(resolutions, indices)
                if index is not None
            )
            fc_records = selected[:8]
            battery_records = selected[8:20]
            speed_record = selected[20]
            if (
                all(len(record.values) == 1 for record in fc_records)
                and all(len(record.values) == 2 for record in battery_records)
                and len(speed_record.values) == 1
            ):
                p_fc = math.fsum(record.values[0] for record in fc_records)
                p_batt = math.fsum(record.values[0] for record in battery_records)
                soc = math.fsum(record.values[1] for record in battery_records) / 12.0
                complete = 0.0 <= soc <= 1.0
                if complete:
                    provenance = [
                        (channel.channel_id, record.timestamp, record.provenance)
                        for channel, record in zip(all_channels, selected)
                    ]
            else:
                complete = False

        previous_fc = (
            previous_complete_fc
            if complete and not gap and previous_complete_fc is not None
            else None
        )
        speed = 0.0
        if complete:
            speed_index = indices[20]
            assert speed_index is not None
            speed = resolutions[20].records[speed_index].values[0]
        mode_sample = ModeSample(
            timestamp,
            speed,
            p_fc if p_fc is not None else 0.0,
            p_batt if p_batt is not None else 0.0,
            channels_complete=complete,
            conflicting_duplicate=conflict_here,
            long_gap_contaminated=gap,
        )
        raw_rows.append(
            {
                "sample": mode_sample,
                "p_fc": p_fc,
                "p_batt": p_batt,
                "soc": soc,
                "previous_fc": previous_fc,
                "provenance": tuple(provenance),
            }
        )
        previous_complete_fc = p_fc if complete else None
        previous_timestamp = timestamp

    modes = classify_operating_modes(tuple(row["sample"] for row in raw_rows))
    states: list[ParentSupervisoryState] = []
    for row, mode in zip(raw_rows, modes):
        sample = row["sample"]
        assert type(sample) is ModeSample
        load = reconstruct_sailing_load(sample, mode) if mode is OperatingMode.SAILING_ISLAND else None
        states.append(
            ParentSupervisoryState(
                channels.parent_id,
                sample.timestamp,
                mode,
                row["p_fc"],
                row["p_batt"],
                load,
                row["soc"],
                row["previous_fc"],
                sample.channels_complete,
                sample.conflicting_duplicate,
                sample.long_gap_contaminated,
                row["provenance"],
            )
        )
    return ParentSupervisoryResult(
        channels.parent_id,
        tuple(states),
        exact_removed,
        len(conflicting),
    )


__all__ = [
    "DuplicateResolution",
    "ParentRawChannels",
    "ParentSupervisoryResult",
    "ParentSupervisoryState",
    "RawChannel",
    "RawRecord",
    "build_parent_supervisory_states",
    "causal_align_latest_without_reuse",
    "resolve_duplicates",
]
