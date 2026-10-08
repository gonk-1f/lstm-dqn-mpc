"""Linear battery energy value for reward timing, never a ledger charge."""
from dataclasses import dataclass
from math import isfinite

from v2.economics import FORMAL_PRICE_CATALOG
from v3.control import SHORE_TARGET_SOC


@dataclass(frozen=True)
class BatteryEnergyValue:
    coefficient_cny: float
    reference_soc: float

    @classmethod
    def from_accountant(cls, accountant):
        eta_chg, _ = accountant.efficiency.require_calibrated()
        tariff = FORMAL_PRICE_CATALOG.shore_cny_per_kwh
        if tariff is None:
            raise ValueError('battery energy feedback requires the formal shore tariff')
        return cls(tariff * accountant.plant.battery_nominal_energy_kwh / eta_chg,
                   SHORE_TARGET_SOC)

    def __call__(self, soc: float) -> float:
        if not isfinite(soc):
            raise ValueError('SOC must be finite')
        return self.coefficient_cny * (self.reference_soc - soc)

    def terminal_correction(self, start_soc: float, actual_end_soc: float) -> float:
        return self(actual_end_soc)-self(start_soc)
