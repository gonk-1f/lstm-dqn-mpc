"""Formal signed-power mode classification and fuel-cell safety interlock."""

from __future__ import annotations

from enum import Enum
import math
from numbers import Real


SIGNED_LOAD_DEADBAND_KW = 1.0


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real non-bool scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


class FormalOperatingMode(Enum):
    SAILING_OR_ISLANDED = "sailing_or_islanded"
    SHORE_CONNECTED = "shore_connected"
    IDLE = "idle"

    @property
    def shore_connected(self) -> float:
        return 1.0 if self is FormalOperatingMode.SHORE_CONNECTED else 0.0


def classify_formal_mode(load_total_kw: float, speed_kn: float) -> FormalOperatingMode:
    """Classify from signed power; AIS is validated context, never the interlock."""

    load = _finite(load_total_kw, "load_total_kw")
    speed = _finite(speed_kn, "speed_kn")
    if speed < 0.0:
        raise ValueError("speed_kn must be nonnegative")
    if load < -SIGNED_LOAD_DEADBAND_KW:
        return FormalOperatingMode.SHORE_CONNECTED
    if load > SIGNED_LOAD_DEADBAND_KW:
        return FormalOperatingMode.SAILING_OR_ISLANDED
    return FormalOperatingMode.IDLE


def interlocked_fc_power_kw(
    mode: FormalOperatingMode, requested_fc_power_kw: float
) -> float:
    if type(mode) is not FormalOperatingMode:
        raise TypeError("mode must be an exact FormalOperatingMode")
    requested = _finite(requested_fc_power_kw, "requested_fc_power_kw")
    if requested < 0.0 or requested > 600.0:
        raise ValueError("requested_fc_power_kw must lie in [0, 600]")
    if mode is not FormalOperatingMode.SAILING_OR_ISLANDED:
        return 0.0
    return requested


__all__ = [
    "FormalOperatingMode",
    "SIGNED_LOAD_DEADBAND_KW",
    "classify_formal_mode",
    "interlocked_fc_power_kw",
]
