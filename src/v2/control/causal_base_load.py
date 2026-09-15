"""Causal persistence load and first-order base-reference forecasts."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Sequence

import numpy as np


def _finite_scalar(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real numeric scalar, not bool or text")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _finite_vector(values: object, name: str) -> tuple[float, ...]:
    if isinstance(values, np.ndarray):
        if values.ndim != 1:
            raise TypeError(f"{name} must be one-dimensional")
        source = values
    elif isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a numeric sequence")
    else:
        source = values
    return tuple(
        _finite_scalar(value, f"{name}[{index}]")
        for index, value in enumerate(source)
    )


@dataclass(frozen=True)
class CausalLoadForecast:
    """Forecast produced from one current observation and past filter state."""

    load_kw: tuple[float, ...]
    base_reference_kw: tuple[float, ...]

    def __post_init__(self) -> None:
        loads = _finite_vector(self.load_kw, "load_kw")
        references = _finite_vector(self.base_reference_kw, "base_reference_kw")
        if not loads or len(loads) != len(references):
            raise ValueError("forecast vectors must have the same nonzero length")
        object.__setattr__(self, "load_kw", loads)
        object.__setattr__(self, "base_reference_kw", references)


class CausalBaseLoadFilter:
    """First-order LPF whose only measurement input is the current scalar load."""

    def __init__(self, *, sample_seconds: float, tau_seconds: float) -> None:
        sample = _finite_scalar(sample_seconds, "sample_seconds")
        tau = _finite_scalar(tau_seconds, "tau_seconds")
        if sample <= 0.0:
            raise ValueError("sample_seconds must be positive")
        if tau <= 0.0:
            raise ValueError("tau_seconds must be positive")
        self.sample_seconds = sample
        self.tau_seconds = tau
        self.alpha = math.exp(-sample / tau)
        self._observed_base_kw: float | None = None

    @property
    def observed_base_kw(self) -> float | None:
        """Return the state based only on observations consumed so far."""

        return self._observed_base_kw

    def observe(self, load_kw: float, *, horizon: int) -> CausalLoadForecast:
        """Consume one observation and make a persistence forecast.

        The scalar-only signature intentionally provides no place for future
        actual measurements.
        """

        forecast = self.preview(load_kw, horizon=horizon)
        self.commit(load_kw)
        return forecast

    def preview(self, load_kw: float, *, horizon: int) -> CausalLoadForecast:
        """Forecast one observation without changing the committed filter state."""

        current = _finite_scalar(load_kw, "load_kw")
        if current < 0.0:
            raise ValueError("load_kw must be nonnegative")
        if type(horizon) is not int:
            raise TypeError("horizon must be an exact positive integer")
        if horizon <= 0:
            raise ValueError("horizon must be positive")

        if self._observed_base_kw is None:
            observed_base = current
        else:
            observed_base = (
                self.alpha * self._observed_base_kw
                + (1.0 - self.alpha) * current
            )

        forecast_base = observed_base
        references: list[float] = []
        for _ in range(horizon):
            forecast_base = self.alpha * forecast_base + (1.0 - self.alpha) * current
            references.append(float(forecast_base))
        return CausalLoadForecast(
            load_kw=(current,) * horizon,
            base_reference_kw=tuple(references),
        )

    def commit(self, load_kw: float) -> None:
        """Commit exactly one already-validated current observation."""

        current = _finite_scalar(load_kw, "load_kw")
        if current < 0.0:
            raise ValueError("load_kw must be nonnegative")
        if self._observed_base_kw is None:
            self._observed_base_kw = current
        else:
            self._observed_base_kw = (
                self.alpha * self._observed_base_kw
                + (1.0 - self.alpha) * current
            )
