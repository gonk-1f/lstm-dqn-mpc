"""Causal persistence load and first-order base-reference forecasts."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

import numpy as np


def _finite_scalar(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real numeric scalar, not bool or text")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class CausalLoadForecast:
    """Forecast produced from one current observation and past filter state."""

    load_kw: tuple[float, ...]
    base_reference_kw: tuple[float, ...]


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

        current = _finite_scalar(load_kw, "load_kw")
        if current < 0.0:
            raise ValueError("load_kw must be nonnegative")
        if type(horizon) is not int:
            raise TypeError("horizon must be an exact positive integer")
        if horizon <= 0:
            raise ValueError("horizon must be positive")

        if self._observed_base_kw is None:
            self._observed_base_kw = current
        else:
            self._observed_base_kw = (
                self.alpha * self._observed_base_kw
                + (1.0 - self.alpha) * current
            )

        forecast_base = self._observed_base_kw
        references: list[float] = []
        for _ in range(horizon):
            forecast_base = self.alpha * forecast_base + (1.0 - self.alpha) * current
            references.append(float(forecast_base))
        return CausalLoadForecast(
            load_kw=(current,) * horizon,
            base_reference_kw=tuple(references),
        )
