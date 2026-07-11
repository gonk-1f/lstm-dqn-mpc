from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FuelCellStackConfig:
    name: str
    power_min_kw: float
    power_max_kw: float
    ramp_rate_kw: float
    initial_power_kw: float = 0.0


class FuelCellStack:
    """Fuel-cell stack with output and ramp constraints."""

    def __init__(self, config: FuelCellStackConfig):
        self.config = config
        self.power_kw = float(config.initial_power_kw)

    def reset(self, power_kw: float | None = None) -> None:
        self.power_kw = float(self.config.initial_power_kw if power_kw is None else power_kw)

    def step(self, power_reference_kw: float) -> dict:
        lower = self.power_kw - self.config.ramp_rate_kw
        upper = self.power_kw + self.config.ramp_rate_kw
        ramp_limited = max(lower, min(upper, power_reference_kw))
        clipped = max(self.config.power_min_kw, min(self.config.power_max_kw, ramp_limited))
        ramp_violation = abs(power_reference_kw - ramp_limited)
        bound_violation = abs(ramp_limited - clipped)
        self.power_kw = float(clipped)
        return {
            "name": self.config.name,
            "power_kw": self.power_kw,
            "ramp_violation": float(ramp_violation),
            "bound_violation": float(bound_violation),
        }
