from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DCDCConverterConfig:
    name: str
    efficiency: float = 0.98
    power_max_kw: float = 200.0


class DCDCConverter:
    """Static DC/DC converter model."""

    def __init__(self, config: DCDCConverterConfig):
        self.config = config

    def transfer(self, input_power_kw: float) -> dict:
        clipped = max(-self.config.power_max_kw, min(self.config.power_max_kw, input_power_kw))
        output = clipped * self.config.efficiency if clipped >= 0 else clipped / self.config.efficiency
        return {
            "name": self.config.name,
            "input_power_kw": float(clipped),
            "output_power_kw": float(output),
            "loss_kw": float(abs(clipped - output)),
        }
