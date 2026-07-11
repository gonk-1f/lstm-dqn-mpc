from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PropulsionLoadConfig:
    propulsion_gain: float = 6.0
    hotel_load_kw: float = 15.0
    auxiliary_gain: float = 1.5


class PropulsionLoadModel:
    """Simple vessel load model from speed and auxiliary operating level."""

    def __init__(self, config: PropulsionLoadConfig | None = None):
        self.config = config or PropulsionLoadConfig()

    def estimate(self, speed_knots: float, auxiliary_level: float = 1.0) -> dict:
        propulsion_kw = self.config.propulsion_gain * (speed_knots ** 3)
        hotel_kw = self.config.hotel_load_kw
        auxiliary_kw = self.config.auxiliary_gain * auxiliary_level
        total_kw = propulsion_kw + hotel_kw + auxiliary_kw
        return {
            "speed_knots": float(speed_knots),
            "propulsion_kw": float(propulsion_kw),
            "hotel_kw": float(hotel_kw),
            "auxiliary_kw": float(auxiliary_kw),
            "total_kw": float(total_kw),
        }
