"""Frozen Train-only eligibility rules for the objective-scale audit."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import math
from numbers import Real
from typing import Sequence


FRESHNESS_CAP_SECONDS = 10.0
SPEED_ZERO_TOLERANCE_KN = 0.1
FC_ZERO_TOLERANCE_KW = 8.0
LONG_GAP_SECONDS = 45.0
SHORE_MIN_CONSECUTIVE_SAMPLES = 2


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real non-bool scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def is_fresh_causal_age(age_seconds: object) -> bool:
    age = _finite(age_seconds, "age_seconds")
    return 0.0 <= age <= FRESHNESS_CAP_SECONDS


class OperatingMode(Enum):
    SAILING_ISLAND = "sailing_island"
    SHORE_CONNECTED = "shore_connected"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ModeSample:
    timestamp: datetime
    speed_kn: float
    p_fc_total_kw: float
    p_batt_total_kw: float
    channels_complete: bool = True
    conflicting_duplicate: bool = False
    long_gap_contaminated: bool = False

    def __post_init__(self) -> None:
        if type(self.timestamp) is not datetime:
            raise TypeError("timestamp must be an exact datetime")
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        for name in ("speed_kn", "p_fc_total_kw", "p_batt_total_kw"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if self.speed_kn < 0.0:
            raise ValueError("speed_kn must be nonnegative")
        for name in (
            "channels_complete",
            "conflicting_duplicate",
            "long_gap_contaminated",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be an exact bool")

    @property
    def quality_valid(self) -> bool:
        return (
            self.channels_complete
            and not self.conflicting_duplicate
            and not self.long_gap_contaminated
        )


def _shore_candidate(sample: ModeSample) -> bool:
    return (
        sample.quality_valid
        and sample.speed_kn <= SPEED_ZERO_TOLERANCE_KN
        and abs(sample.p_fc_total_kw) <= FC_ZERO_TOLERANCE_KW
        and sample.p_batt_total_kw < 0.0
    )


def classify_operating_modes(samples: Sequence[ModeSample]) -> tuple[OperatingMode, ...]:
    if isinstance(samples, (str, bytes)) or not isinstance(samples, Sequence):
        raise TypeError("samples must be a sequence")
    checked = tuple(samples)
    if any(type(sample) is not ModeSample for sample in checked):
        raise TypeError("samples must contain exact ModeSample values")
    if any(
        right.timestamp <= left.timestamp
        for left, right in zip(checked, checked[1:])
    ):
        raise ValueError("sample timestamps must be strictly increasing")

    modes = [OperatingMode.UNKNOWN] * len(checked)
    shore_flags = [_shore_candidate(sample) for sample in checked]
    start = 0
    while start < len(checked):
        if not shore_flags[start]:
            start += 1
            continue
        end = start + 1
        while (
            end < len(checked)
            and shore_flags[end]
            and (checked[end].timestamp - checked[end - 1].timestamp).total_seconds()
            <= LONG_GAP_SECONDS
        ):
            end += 1
        if end - start >= SHORE_MIN_CONSECUTIVE_SAMPLES:
            modes[start:end] = [OperatingMode.SHORE_CONNECTED] * (end - start)
        start = end

    for index, sample in enumerate(checked):
        if modes[index] is OperatingMode.SHORE_CONNECTED or not sample.quality_valid:
            continue
        reconstructed = sample.p_fc_total_kw + sample.p_batt_total_kw
        if sample.speed_kn > SPEED_ZERO_TOLERANCE_KN and reconstructed >= 0.0:
            modes[index] = OperatingMode.SAILING_ISLAND
    return tuple(modes)


def reconstruct_sailing_load(sample: ModeSample, mode: OperatingMode) -> float:
    if type(sample) is not ModeSample:
        raise TypeError("sample must be an exact ModeSample")
    if type(mode) is not OperatingMode:
        raise TypeError("mode must be an exact OperatingMode")
    if mode is not OperatingMode.SAILING_ISLAND:
        raise ValueError("load reconstruction is restricted to sailing_island")
    if not sample.quality_valid or sample.speed_kn <= SPEED_ZERO_TOLERANCE_KN:
        raise ValueError("sample does not satisfy sailing eligibility")
    load = sample.p_fc_total_kw + sample.p_batt_total_kw
    if load < 0.0:
        raise ValueError("negative reconstructed sailing load is contradictory")
    return float(load)


__all__ = [
    "FC_ZERO_TOLERANCE_KW",
    "FRESHNESS_CAP_SECONDS",
    "LONG_GAP_SECONDS",
    "SHORE_MIN_CONSECUTIVE_SAMPLES",
    "SPEED_ZERO_TOLERANCE_KN",
    "ModeSample",
    "OperatingMode",
    "classify_operating_modes",
    "is_fresh_causal_age",
    "reconstruct_sailing_load",
]
