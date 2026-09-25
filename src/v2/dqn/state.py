"""Candidate causal operating state for the v2 DQN.

The state is deliberately *not* a finalized training schema.  Its nine
semantic groups flatten to ten scalars because the fuel-cell group contains
both current and previous power.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
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

FORMAL_STATE_STATUS = "FROZEN_PROJECT_BASELINE"
FORMAL_STATE_SCHEMA_VERSION = "v2_s8_onboard_ais_v1"
FORMAL_STATE_FEATURE_NAMES = (
    "soc",
    "causal_base_load_fraction",
    "load_residual_fraction",
    "recent_load_population_std_fraction",
    "recent_load_window_trend_fraction",
    "fuel_cell_power_fraction",
    "fuel_cell_delta_fraction",
    "speed_fraction",
)
FORMAL_STATE_DIMENSION = len(FORMAL_STATE_FEATURE_NAMES)
FORMAL_STATE_POWER_SCALE_KW = 600.0
FORMAL_STATE_SPEED_SCALE_KN = 20.0
FORMAL_STATE_WINDOW_SECONDS = 150.0
FORMAL_STATE_SCHEMA_DIGEST = hashlib.sha256(
    json.dumps(
        {
            "version": FORMAL_STATE_SCHEMA_VERSION,
            "features": FORMAL_STATE_FEATURE_NAMES,
            "power_scale_kw": FORMAL_STATE_POWER_SCALE_KW,
            "speed_scale_kn": FORMAL_STATE_SPEED_SCALE_KN,
            "window_seconds": FORMAL_STATE_WINDOW_SECONDS,
        },
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()


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


def _finite_sum(values: tuple[float, ...], name: str) -> float:
    try:
        result = math.fsum(values)
    except OverflowError as exc:
        raise ValueError(f"{name} must remain finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must remain finite")
    return result


def _finite_product_ratio(
    numerators: tuple[float, ...],
    denominators: tuple[float, ...],
    name: str,
) -> float:
    """Evaluate a product ratio without overflowing representable results."""

    if any(value == 0.0 for value in numerators):
        return 0.0
    if any(value == 0.0 for value in denominators):
        raise ValueError(f"{name} denominator must be non-zero")

    sign = 1.0
    mantissa = 1.0
    exponent = 0
    for value in numerators:
        checked = _finite_scalar(value, name)
        if checked < 0.0:
            sign = -sign
        part, part_exponent = math.frexp(abs(checked))
        mantissa *= part
        exponent += part_exponent
    for value in denominators:
        checked = _finite_scalar(value, name)
        if checked < 0.0:
            sign = -sign
        part, part_exponent = math.frexp(abs(checked))
        mantissa /= part
        exponent -= part_exponent
    try:
        result = math.ldexp(sign * mantissa, exponent)
    except OverflowError as exc:
        raise ValueError(f"{name} is not representable as a finite float") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} is not representable as a finite float")
    return result


def _scaled_population_mean_and_std(
    values: tuple[float, ...],
) -> tuple[float, float]:
    """Return finite population moments using a common magnitude scale."""

    scale = max(abs(value) for value in values)
    if scale == 0.0:
        return 0.0, 0.0
    normalized = tuple(value / scale for value in values)
    normalized_mean = _finite_sum(normalized, "normalized load sum") / len(values)
    normalized_variance = _finite_sum(
        tuple((value - normalized_mean) ** 2 for value in normalized),
        "normalized load squared-deviation sum",
    ) / len(values)
    if normalized_variance < 0.0 or not math.isfinite(normalized_variance):
        raise ValueError("normalized load population variance must be finite")
    mean = _finite_product_ratio((scale, normalized_mean), (1.0,), "load mean")
    std = _finite_product_ratio(
        (scale, math.sqrt(normalized_variance)),
        (1.0,),
        "load population standard deviation",
    )
    return mean, std


def _sample_age_seconds(current_time_seconds: float, timestamp_seconds: float) -> float:
    """Return signed age while classifying finite-subtraction overflow."""

    try:
        age = current_time_seconds - timestamp_seconds
    except OverflowError:
        return math.inf if timestamp_seconds < current_time_seconds else -math.inf
    if math.isnan(age):
        raise ValueError("sample age must not be NaN")
    return age


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
    time_center = _finite_sum(
        (
            samples[0].timestamp_seconds / 2.0,
            samples[-1].timestamp_seconds / 2.0,
        ),
        "timestamp center",
    )
    elapsed = tuple(sample.timestamp_seconds - time_center for sample in samples)
    if not all(math.isfinite(value) for value in elapsed):
        raise ValueError("centered physical timestamps must remain finite")
    time_scale = max(abs(value) for value in elapsed)
    if time_scale == 0.0:
        raise ValueError("load trend requires distinct physical timestamps")
    loads = tuple(sample.load_power_kw for sample in samples)
    load_scale = max(abs(value) for value in loads)
    if load_scale == 0.0:
        return 0.0

    normalized_time = tuple(value / time_scale for value in elapsed)
    normalized_load = tuple(value / load_scale for value in loads)
    mean_time = _finite_sum(normalized_time, "normalized timestamp sum") / len(
        normalized_time
    )
    mean_load = _finite_sum(normalized_load, "normalized load sum") / len(
        normalized_load
    )
    covariance = _finite_sum(
        tuple(
            (time - mean_time) * (load - mean_load)
            for time, load in zip(normalized_time, normalized_load)
        ),
        "normalized load-time covariance sum",
    )
    variance = _finite_sum(
        tuple((time - mean_time) ** 2 for time in normalized_time),
        "normalized timestamp squared-deviation sum",
    )
    if variance <= 0.0:
        raise ValueError("load trend requires distinct physical timestamps")
    return _finite_product_ratio(
        (load_scale, covariance),
        (time_scale, variance),
        "load trend slope in kW/s",
    )


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
    selected_list: list[OperatingHistorySample] = []
    for sample in checked_history:
        age = _sample_age_seconds(now, sample.timestamp_seconds)
        if math.isinf(age):
            # +inf is older than every finite window; -inf is future data.
            continue
        if 0.0 <= age <= window:
            selected_list.append(sample)
    selected = tuple(selected_list)
    if not selected or selected[-1].timestamp_seconds != now:
        raise ValueError("history requires an exact current-time sample")
    if len(selected) < 2:
        raise ValueError(
            "at least two causal samples inside the seconds window are required "
            "for previous fuel-cell power and load trend"
        )

    current = selected[-1]
    previous = selected[-2]
    load_mean, load_std = _scaled_population_mean_and_std(
        tuple(sample.load_power_kw for sample in selected)
    )
    load_slope_kw_per_second = _least_squares_slope_per_second(selected)

    state = (
        current.soc,
        current.fuel_cell_power_kw / scales.fuel_cell_rated_kw,
        previous.fuel_cell_power_kw / scales.fuel_cell_rated_kw,
        current.battery_power_kw / scales.battery_power_scale_kw,
        current.load_power_kw / scales.load_power_scale_kw,
        _finite_product_ratio(
            (load_mean,), (scales.load_power_scale_kw,), "normalized load mean"
        ),
        _finite_product_ratio(
            (load_std,),
            (scales.load_power_scale_kw,),
            "normalized load population standard deviation",
        ),
        _finite_product_ratio(
            (load_slope_kw_per_second, window),
            (scales.load_power_scale_kw,),
            "normalized load window trend",
        ),
        current.causal_base_load_kw / scales.base_load_power_scale_kw,
        current.soc - selected[0].soc,
    )
    if len(state) != len(CANDIDATE_STATE_FEATURE_NAMES) or not all(
        math.isfinite(value) for value in state
    ):
        raise ValueError("candidate state must contain ten finite scalars")
    return state


def build_formal_operating_state(
    history: tuple[OperatingHistorySample, ...],
    *,
    current_time_seconds: float,
    speed_kn: float,
) -> tuple[float, ...]:
    """Build the frozen ONBOARD-only S8 from an inclusive causal history."""

    if type(history) is not tuple or not history:
        raise TypeError("history must be a nonempty immutable tuple")
    checked = tuple(_validate_sample(sample) for sample in history)
    if any(
        current.timestamp_seconds <= previous.timestamp_seconds
        for previous, current in zip(checked, checked[1:])
    ):
        raise ValueError("history timestamps must be strictly increasing")
    now = _finite_scalar(current_time_seconds, "current_time_seconds")
    speed = _finite_scalar(speed_kn, "speed_kn")
    if speed < 0.0:
        raise ValueError("speed_kn must be nonnegative")
    selected = tuple(
        sample
        for sample in checked
        if 0.0 <= _sample_age_seconds(now, sample.timestamp_seconds)
        <= FORMAL_STATE_WINDOW_SECONDS
    )
    if not selected or selected[-1].timestamp_seconds != now:
        raise ValueError("history requires an exact current-time sample")
    current = selected[-1]
    previous_fc = (
        selected[-2].fuel_cell_power_kw
        if len(selected) >= 2
        else current.fuel_cell_power_kw
    )
    _, load_std = _scaled_population_mean_and_std(
        tuple(sample.load_power_kw for sample in selected)
    )
    load_slope = (
        _least_squares_slope_per_second(selected) if len(selected) >= 2 else 0.0
    )
    state = (
        float(current.soc),
        current.causal_base_load_kw / FORMAL_STATE_POWER_SCALE_KW,
        (current.load_power_kw - current.causal_base_load_kw)
        / FORMAL_STATE_POWER_SCALE_KW,
        load_std / FORMAL_STATE_POWER_SCALE_KW,
        load_slope * FORMAL_STATE_WINDOW_SECONDS / FORMAL_STATE_POWER_SCALE_KW,
        current.fuel_cell_power_kw / FORMAL_STATE_POWER_SCALE_KW,
        (current.fuel_cell_power_kw - previous_fc) / FORMAL_STATE_POWER_SCALE_KW,
        speed / FORMAL_STATE_SPEED_SCALE_KN,
    )
    if len(state) != FORMAL_STATE_DIMENSION or not all(
        type(value) is float and math.isfinite(value) for value in state
    ):
        raise ValueError("formal S8 state must contain eight finite floats")
    return state


__all__ = [
    "CANDIDATE_STATE_FEATURE_NAMES",
    "CANDIDATE_STATE_GROUP_NAMES",
    "CANDIDATE_STATE_STATUS",
    "FORMAL_STATE_DIMENSION",
    "FORMAL_STATE_FEATURE_NAMES",
    "FORMAL_STATE_POWER_SCALE_KW",
    "FORMAL_STATE_SCHEMA_DIGEST",
    "FORMAL_STATE_SCHEMA_VERSION",
    "FORMAL_STATE_SPEED_SCALE_KN",
    "FORMAL_STATE_STATUS",
    "FORMAL_STATE_WINDOW_SECONDS",
    "OperatingHistorySample",
    "StateNormalization",
    "build_candidate_operating_state",
    "build_formal_operating_state",
]
