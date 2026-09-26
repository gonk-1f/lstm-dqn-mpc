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
BATTERY_CHARGE_THRESHOLD_KW = 1.0
SOURCE_LOAD_DEADBAND_KW = 1.0
LONG_GAP_SECONDS = 45.0
SHORE_MIN_CONSECUTIVE_SAMPLES = 3


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


def normalize_onboard_load_kw(load_kw: object) -> float:
    """Map the frozen numerical deadband to zero before MPC execution."""

    load = _finite(load_kw, "load_kw")
    if load < -SOURCE_LOAD_DEADBAND_KW:
        raise ValueError("ONBOARD load lies below the frozen source-load deadband")
    return max(0.0, load)


class OperatingMode(Enum):
    ONBOARD = "onboard"
    SHORE_PENDING = "shore_pending"
    SHORE_CHARGING = "shore_charging"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class ModeSample:
    timestamp: datetime
    speed_kn: float
    p_fc_total_kw: float
    p_batt_total_kw: float
    channels_complete: bool = True
    conflicting_duplicate: bool = False
    long_gap_contaminated: bool = False
    p_load_kw: float | None = None

    def __post_init__(self) -> None:
        if type(self.timestamp) is not datetime:
            raise TypeError("timestamp must be an exact datetime")
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        for name in ("speed_kn", "p_fc_total_kw", "p_batt_total_kw"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if self.speed_kn < 0.0:
            raise ValueError("speed_kn must be nonnegative")
        if self.p_load_kw is not None:
            object.__setattr__(self, "p_load_kw", _finite(self.p_load_kw, "p_load_kw"))
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
        and sample.p_batt_total_kw < -BATTERY_CHARGE_THRESHOLD_KW
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

    modes = [OperatingMode.UNRESOLVED] * len(checked)
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
            modes[start : start + SHORE_MIN_CONSECUTIVE_SAMPLES - 1] = [
                OperatingMode.SHORE_PENDING
            ] * (SHORE_MIN_CONSECUTIVE_SAMPLES - 1)
            modes[start + SHORE_MIN_CONSECUTIVE_SAMPLES - 1 : end] = [
                OperatingMode.SHORE_CHARGING
            ] * (end - start - SHORE_MIN_CONSECUTIVE_SAMPLES + 1)
        start = end

    for index, sample in enumerate(checked):
        if modes[index] in {
            OperatingMode.SHORE_PENDING,
            OperatingMode.SHORE_CHARGING,
        } or not sample.quality_valid:
            continue
        reconstructed = (
            sample.p_fc_total_kw + sample.p_batt_total_kw
            if sample.p_load_kw is None
            else sample.p_load_kw
        )
        if reconstructed >= -SOURCE_LOAD_DEADBAND_KW:
            modes[index] = OperatingMode.ONBOARD
    return tuple(modes)


def reconstruct_sailing_load(sample: ModeSample, mode: OperatingMode) -> float:
    if type(sample) is not ModeSample:
        raise TypeError("sample must be an exact ModeSample")
    if type(mode) is not OperatingMode:
        raise TypeError("mode must be an exact OperatingMode")
    if mode is not OperatingMode.ONBOARD:
        raise ValueError("load reconstruction is restricted to onboard mode")
    if not sample.quality_valid:
        raise ValueError("sample does not satisfy onboard eligibility")
    load = sample.p_fc_total_kw + sample.p_batt_total_kw
    return normalize_onboard_load_kw(load)


__all__ = [
    "BATTERY_CHARGE_THRESHOLD_KW",
    "FRESHNESS_CAP_SECONDS",
    "LONG_GAP_SECONDS",
    "SHORE_MIN_CONSECUTIVE_SAMPLES",
    "SOURCE_LOAD_DEADBAND_KW",
    "SPEED_ZERO_TOLERANCE_KN",
    "ModeSample",
    "OperatingMode",
    "classify_operating_modes",
    "is_fresh_causal_age",
    "normalize_onboard_load_kw",
    "reconstruct_sailing_load",
]
