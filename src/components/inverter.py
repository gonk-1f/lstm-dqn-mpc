from __future__ import annotations

from dataclasses import dataclass


@dataclass
class InverterConfig:
    name: str
    efficiency: float = 0.97
    power_max_kw: float = 250.0


class Inverter:
    """Static inverter model with efficiency and power limit."""

    def __init__(self, config: InverterConfig):
        self.config = config

    def transfer(self, dc_power_kw: float) -> dict:
        dc_power_kw = max(-self.config.power_max_kw, min(self.config.power_max_kw, dc_power_kw))
        ac_power_kw = dc_power_kw * self.config.efficiency if dc_power_kw >= 0 else dc_power_kw / self.config.efficiency
        return {
            "name": self.config.name,
            "dc_power_kw": float(dc_power_kw),
            "ac_power_kw": float(ac_power_kw),
            "loss_kw": float(abs(dc_power_kw - ac_power_kw)),
        }
