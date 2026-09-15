"""Battery bus-to-cell energy and SOC accounting."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

import numpy as np

FORMAL_BATTERY_ETA_CHG = 0.95
FORMAL_BATTERY_ETA_DIS = 0.95
BATTERY_EFFICIENCY_CALIBRATION_STATUS = "SOURCE_BACKED"
BATTERY_EFFICIENCY_SOURCE_DOI = "10.11930/j.issn.1004-9649.202507065"
BATTERY_EFFICIENCY_SOURCE_LOCATION = "Table 3"


def _strict_scalar(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real numeric scalar, not bool or text")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class BatteryEfficiency:
    eta_chg: float | None = None
    eta_dis: float | None = None
    source_reference: str | None = None
    source_location: str | None = None

    def require_calibrated(self) -> tuple[float, float]:
        if self.eta_chg is None or self.eta_dis is None:
            raise ValueError("eta_chg and eta_dis are uncalibrated")
        charge = _strict_scalar(self.eta_chg, "eta_chg")
        discharge = _strict_scalar(self.eta_dis, "eta_dis")
        if not 0.0 < charge <= 1.0 or not 0.0 < discharge <= 1.0:
            raise ValueError("battery efficiencies must lie in (0, 1]")
        if (
            not isinstance(self.source_reference, str)
            or not self.source_reference.strip()
            or not isinstance(self.source_location, str)
            or not self.source_location.strip()
        ):
            raise ValueError("battery efficiency source reference and location are required")
        return charge, discharge

    @classmethod
    def formal_default(cls) -> BatteryEfficiency:
        """Return the explicit source-backed formal efficiency calibration."""

        return cls(
            eta_chg=FORMAL_BATTERY_ETA_CHG,
            eta_dis=FORMAL_BATTERY_ETA_DIS,
            source_reference=BATTERY_EFFICIENCY_SOURCE_DOI,
            source_location=BATTERY_EFFICIENCY_SOURCE_LOCATION,
        )


def formal_battery_efficiency() -> BatteryEfficiency:
    """Build the formal 0.95/0.95 source-backed battery efficiency object."""

    return BatteryEfficiency.formal_default()


def next_soc(
    soc: float,
    p_batt_bus_kw: float,
    dt_seconds: float,
    capacity_kwh: float,
    eta_chg: float,
    eta_dis: float,
) -> float:
    """Advance SOC without clamping; positive bus power means discharge."""

    state = _strict_scalar(soc, "soc")
    bus_power = _strict_scalar(p_batt_bus_kw, "p_batt_bus_kw")
    duration = _strict_scalar(dt_seconds, "dt_seconds")
    capacity = _strict_scalar(capacity_kwh, "capacity_kwh")
    charge = _strict_scalar(eta_chg, "eta_chg")
    discharge = _strict_scalar(eta_dis, "eta_dis")
    if duration <= 0.0 or capacity <= 0.0:
        raise ValueError("duration and capacity must be positive")
    if not 0.0 < charge <= 1.0 or not 0.0 < discharge <= 1.0:
        raise ValueError("battery efficiencies must lie in (0, 1]")
    battery_side_kw = bus_power / discharge if bus_power > 0.0 else charge * bus_power
    return state - battery_side_kw * (duration / 3600.0) / capacity
