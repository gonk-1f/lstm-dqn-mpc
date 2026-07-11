from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ShipSideState:
    name: str
    fuel_cell_power_kw: float = 0.0
    battery_power_kw: float = 0.0
    load_power_kw: float = 0.0

    @property
    def supplied_power_kw(self) -> float:
        return self.fuel_cell_power_kw + self.battery_power_kw

    @property
    def balance_error_kw(self) -> float:
        return self.supplied_power_kw - self.load_power_kw


@dataclass
class ShipPowerBus:
    """Aggregate left/right vessel-side power flows."""

    sides: list[ShipSideState] = field(default_factory=list)

    def total_load_kw(self) -> float:
        return float(sum(side.load_power_kw for side in self.sides))

    def total_supply_kw(self) -> float:
        return float(sum(side.supplied_power_kw for side in self.sides))

    def total_balance_error_kw(self) -> float:
        return float(sum(side.balance_error_kw for side in self.sides))

    def as_dict(self) -> dict:
        return {
            "total_load_kw": self.total_load_kw(),
            "total_supply_kw": self.total_supply_kw(),
            "total_balance_error_kw": self.total_balance_error_kw(),
            "sides": [
                {
                    "name": side.name,
                    "fuel_cell_power_kw": side.fuel_cell_power_kw,
                    "battery_power_kw": side.battery_power_kw,
                    "load_power_kw": side.load_power_kw,
                    "supplied_power_kw": side.supplied_power_kw,
                    "balance_error_kw": side.balance_error_kw,
                }
                for side in self.sides
            ],
        }
