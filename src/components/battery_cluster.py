from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BatteryClusterConfig:
    name: str
    capacity_kwh: float
    soc_min: float
    soc_max: float
    charge_power_max_kw: float
    discharge_power_max_kw: float
    initial_soc: float = 0.6


class BatteryCluster:
    """Simple battery-cluster model used by ship EMS layers.

    Positive power means discharge to the bus.
    Negative power means charge from the bus.
    """

    def __init__(self, config: BatteryClusterConfig):
        self.config = config
        self.soc = float(config.initial_soc)
        self.power_kw = 0.0

    def reset(self, soc: float | None = None) -> None:
        self.soc = float(self.config.initial_soc if soc is None else soc)
        self.power_kw = 0.0

    def clip_power(self, power_kw: float) -> tuple[float, float]:
        clipped = max(-self.config.charge_power_max_kw, min(self.config.discharge_power_max_kw, power_kw))
        violation = abs(power_kw - clipped)
        return float(clipped), float(violation)

    def step(self, power_kw: float, dt_hours: float) -> dict:
        clipped_power, power_violation = self.clip_power(power_kw)
        next_soc = self.soc - clipped_power * dt_hours / self.config.capacity_kwh
        clipped_soc = max(self.config.soc_min, min(self.config.soc_max, next_soc))
        soc_violation = abs(next_soc - clipped_soc)
        self.soc = float(clipped_soc)
        self.power_kw = float(clipped_power)
        return {
            "name": self.config.name,
            "power_kw": self.power_kw,
            "soc": self.soc,
            "power_violation": float(power_violation),
            "soc_violation": float(soc_violation),
        }
