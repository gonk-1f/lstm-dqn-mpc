"""Candidate causal operating state for the v2 DQN.

The state is deliberately *not* a finalized training schema.  Its nine
semantic groups flatten to ten scalars because the fuel-cell group contains
both current and previous power.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

import numpy as np


CANDIDATE_STATE_STATUS = "NO-GO"
CANDIDATE_STATE_GROUP_NAMES = (
    "soc",
    "fuel_cell_power_current_and_previous",
    "battery_power",
    "load_power",
    "recent_load_mean",
    "recent_load_population_std",
    "recent_load_trend",
    "causal_base_load",
    "recent_delta_soc",
)
CANDIDATE_STATE_FEATURE_NAMES = (
    "soc",
    "fuel_cell_power_fraction",
    "previous_fuel_cell_power_fraction",
    "battery_power_fraction",
    "load_power_fraction",
    "recent_load_mean_fraction",
    "recent_load_population_std_fraction",
    "recent_load_window_trend_fraction",
    "causal_base_load_fraction",
    "recent_delta_soc",
)


def _finite_scalar(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real numeric scalar, not bool or text")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_scalar(value: object, name: str) -> float:
    result = _finite_scalar(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


@dataclass(frozen=True)
class OperatingHistorySample:
    """One immutable, timestamped, physical operating record."""

    timestamp_seconds: float
    soc: float
    fuel_cell_power_kw: float
    battery_power_kw: float
    load_power_kw: float
    causal_base_load_kw: float

    def __post_init__(self) -> None:
        for name in (
            "timestamp_seconds",
            "soc",
            "fuel_cell_power_kw",
            "battery_power_kw",
            "load_power_kw",
            "causal_base_load_kw",
        ):
            object.__setattr__(self, name, _finite_scalar(getattr(self, name), name))
        if not 0.0 <= self.soc <= 1.0:
            raise ValueError("soc must lie in [0, 1]")


def _validate_sample(value: object) -> OperatingHistorySample:
    if type(value) is not OperatingHistorySample:
        raise TypeError("history must contain exact OperatingHistorySample records")
    for name in (
        "timestamp_seconds",
        "soc",
        "fuel_cell_power_kw",
        "battery_power_kw",
        "load_power_kw",
        "causal_base_load_kw",
    ):
        if type(getattr(value, name)) is not float:
            raise TypeError(f"stored {name} must remain an exact float")
        _finite_scalar(getattr(value, name), name)
    if not 0.0 <= value.soc <= 1.0:
        raise ValueError("stored soc must remain in [0, 1]")
    return value


@dataclass(frozen=True)
class StateNormalization:
    """Explicit positive denominators for dimensional power features."""

    fuel_cell_rated_kw: float
    battery_power_scale_kw: float
    load_power_scale_kw: float
    base_load_power_scale_kw: float

    def __post_init__(self) -> None:
        for name in (
            "fuel_cell_rated_kw",
            "battery_power_scale_kw",
            "load_power_scale_kw",
            "base_load_power_scale_kw",
        ):
            object.__setattr__(self, name, _positive_scalar(getattr(self, name), name))


def _validate_normalization(value: object) -> StateNormalization:
    if type(value) is not StateNormalization:
        raise TypeError("normalization must be an exact StateNormalization")
    for name in (
        "fuel_cell_rated_kw",
        "battery_power_scale_kw",
        "load_power_scale_kw",
        "base_load_power_scale_kw",
    ):
        if type(getattr(value, name)) is not float:
            raise TypeError(f"stored {name} must remain an exact float")
        _positive_scalar(getattr(value, name), name)
    return value


def _least_squares_slope_per_second(
    samples: tuple[OperatingHistorySample, ...],
) -> float:
    origin = samples[0].timestamp_seconds
    elapsed = tuple(sample.timestamp_seconds - origin for sample in samples)
    loads = tuple(sample.load_power_kw for sample in samples)
    mean_time = math.fsum(elapsed) / len(elapsed)
    mean_load = math.fsum(loads) / len(loads)
    numerator = math.fsum(
        (time - mean_time) * (load - mean_load)
        for time, load in zip(elapsed, loads)
    )
    denominator = math.fsum((time - mean_time) ** 2 for time in elapsed)
    if denominator <= 0.0:
        raise ValueError("load trend requires distinct physical timestamps")
    slope = numerator / denominator
    if not math.isfinite(slope):
        raise ValueError("load trend must remain finite")
    return slope


def build_candidate_operating_state(
    history: tuple[OperatingHistorySample, ...],
    *,
    current_time_seconds: float,
    window_seconds: float,
    normalization: StateNormalization,
) -> tuple[float, ...]:
    """Build the ten-scalar candidate state from a causal seconds window.

    The inclusive window is ``[current_time_seconds - window_seconds,
    current_time_seconds]``.  Future records may be present in the supplied
    immutable history but are never selected.  A sample exactly at
    ``current_time_seconds`` and at least two samples inside the window are
    required, so neither the previous fuel-cell value nor the trend can be
    silently fabricated.
    """

    if type(history) is not tuple:
        raise TypeError("history must be an immutable tuple")
    if not history:
        raise ValueError("history must not be empty")
    checked_history = tuple(_validate_sample(sample) for sample in history)
    for previous, current in zip(checked_history, checked_history[1:]):
        if current.timestamp_seconds <= previous.timestamp_seconds:
            raise ValueError("history timestamps must be strictly increasing")

    now = _finite_scalar(current_time_seconds, "current_time_seconds")
    window = _positive_scalar(window_seconds, "window_seconds")
    scales = _validate_normalization(normalization)
    start = now - window
    selected = tuple(
        sample
        for sample in checked_history
        if start <= sample.timestamp_seconds <= now
    )
    if not selected or selected[-1].timestamp_seconds != now:
        raise ValueError("history requires an exact current-time sample")
    if len(selected) < 2:
        raise ValueError(
            "at least two causal samples inside the seconds window are required "
            "for previous fuel-cell power and load trend"
        )

    current = selected[-1]
    previous = selected[-2]
    load_mean = math.fsum(sample.load_power_kw for sample in selected) / len(selected)
    load_variance = math.fsum(
        (sample.load_power_kw - load_mean) ** 2 for sample in selected
    ) / len(selected)
    load_std = math.sqrt(load_variance)
    load_slope_kw_per_second = _least_squares_slope_per_second(selected)

    state = (
        current.soc,
        current.fuel_cell_power_kw / scales.fuel_cell_rated_kw,
        previous.fuel_cell_power_kw / scales.fuel_cell_rated_kw,
        current.battery_power_kw / scales.battery_power_scale_kw,
        current.load_power_kw / scales.load_power_scale_kw,
        load_mean / scales.load_power_scale_kw,
        load_std / scales.load_power_scale_kw,
        load_slope_kw_per_second * window / scales.load_power_scale_kw,
        current.causal_base_load_kw / scales.base_load_power_scale_kw,
        current.soc - selected[0].soc,
    )
    if len(state) != len(CANDIDATE_STATE_FEATURE_NAMES) or not all(
        math.isfinite(value) for value in state
    ):
        raise ValueError("candidate state must contain ten finite scalars")
    return state


__all__ = [
    "CANDIDATE_STATE_FEATURE_NAMES",
    "CANDIDATE_STATE_GROUP_NAMES",
    "CANDIDATE_STATE_STATUS",
    "OperatingHistorySample",
    "StateNormalization",
    "build_candidate_operating_state",
]
