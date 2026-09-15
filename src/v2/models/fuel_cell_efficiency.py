"""Source-backed fuel-cell efficiency and hydrogen-energy accounting."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real
from typing import Sequence

import numpy as np
from scipy.interpolate import PchipInterpolator

LHV_H2_MJ_PER_KG = 120.0
LHV_H2_KWH_PER_KG = LHV_H2_MJ_PER_KG / 3.6
FC_EFFICIENCY_CALIBRATION_STATUS = "SOURCE_BACKED"

FC_DATA_WORKBOOK_PATH = r"C:\Users\20883\OneDrive\Desktop\氢耗\FC_Data.xlsx"
FC_DATA_WORKBOOK_SHA256 = (
    "906a0383f6e427a938a8343e9fb1428bfea0f8e2766f5ac5449fea0ccbde4a21"
)
FC_DATA_SHEET = "Sheet1"
FC_DATA_RANGE = "A2:B12"
FC_SOURCE_SYSTEM_RATED_POWER_KW = 100.0
FORMAL_FC_RATED_POWER_KW = 600.0
FC_DATA_SOURCE_COLUMNS = "A: net system output kW; B: system efficiency percent points"
FC_DATA_AXIS_TRANSFORM = (
    "Treat the source as a 100 kW characteristic and preserve load fraction: "
    "P_formal/600 kW = P_source/100 kW."
)
FC_DATA_EFFICIENCY_TRANSFORM = (
    "Convert percent points to fractions; efficiency is otherwise unchanged."
)
FC_DATA_ENDPOINT_TRANSFORM = (
    "Use source points above 100 kW only as PCHIP support to interpolate the "
    "source endpoint at exactly 100 kW before mapping it to 600 kW."
)
FC_DATA_HYDROGEN_CROSS_CHECK = (
    "The old m_h2 column is excluded from the formal model; interpreted as g/min, "
    "it implies about 115.1--116.9 MJ/kg and conflicts with the adopted 120 MJ/kg LHV."
)

# Values are stored exactly as extracted: column A is kW and column B is percent.
FC_DATA_RAW_POINTS = (
    (0.0, 0.0),
    (11.3938, 63.8554),
    (22.3676, 62.4096),
    (35.9133, 60.4819),
    (49.116, 58.7952),
    (62.146, 58.0723),
    (74.8327, 56.8675),
    (87.1766, 56.1446),
    (98.3215, 54.9398),
    (109.123, 53.7349),
    (116.668, 52.0482),
)


def _strict_scalar(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real numeric scalar, not bool or text")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _strict_array(values: object, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.kind not in "iuf" or array.dtype.kind == "b":
        raise TypeError(f"{name} must contain only real non-bool numeric values")
    result = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


@dataclass(frozen=True)
class FuelCellEfficiencyProvenance:
    workbook_path: str
    workbook_sha256: str
    worksheet: str
    cell_range: str
    source_columns: str
    raw_points: tuple[tuple[float, float], ...]
    axis_transform: str
    efficiency_transform: str
    endpoint_transform: str
    hydrogen_column_cross_check: str

    def __post_init__(self) -> None:
        text_fields = (
            self.workbook_path,
            self.workbook_sha256,
            self.worksheet,
            self.cell_range,
            self.source_columns,
            self.axis_transform,
            self.efficiency_transform,
            self.endpoint_transform,
            self.hydrogen_column_cross_check,
        )
        if any(not isinstance(value, str) or not value.strip() for value in text_fields):
            raise ValueError("complete fuel-cell efficiency provenance is required")
        if len(self.workbook_sha256) != 64 or any(
            character not in "0123456789abcdefABCDEF"
            for character in self.workbook_sha256
        ):
            raise ValueError("workbook_sha256 must be a 64-character hexadecimal digest")
        raw = _strict_array(self.raw_points, "provenance raw_points")
        if raw.ndim != 2 or raw.shape[0] < 2 or raw.shape[1] != 2:
            raise ValueError("provenance raw_points must be an Nx2 numeric table")
        if not np.all(np.diff(raw[:, 0]) > 0.0):
            raise ValueError("provenance source powers must be strictly increasing")
        if np.any(raw[:, 1] < 0.0) or np.any(raw[:, 1] > 100.0):
            raise ValueError("provenance efficiency percent points must lie in [0, 100]")
        object.__setattr__(
            self,
            "raw_points",
            tuple((float(power), float(eta)) for power, eta in raw),
        )


FC_DATA_PROVENANCE = FuelCellEfficiencyProvenance(
    workbook_path=FC_DATA_WORKBOOK_PATH,
    workbook_sha256=FC_DATA_WORKBOOK_SHA256,
    worksheet=FC_DATA_SHEET,
    cell_range=FC_DATA_RANGE,
    source_columns=FC_DATA_SOURCE_COLUMNS,
    raw_points=FC_DATA_RAW_POINTS,
    axis_transform=FC_DATA_AXIS_TRANSFORM,
    efficiency_transform=FC_DATA_EFFICIENCY_TRANSFORM,
    endpoint_transform=FC_DATA_ENDPOINT_TRANSFORM,
    hydrogen_column_cross_check=FC_DATA_HYDROGEN_CROSS_CHECK,
)


def _formal_curve_points() -> tuple[tuple[float, ...], tuple[float, ...]]:
    raw = np.asarray(FC_DATA_RAW_POINTS, dtype=np.float64)
    source_power = raw[:, 0]
    source_efficiency_percent = raw[:, 1]
    endpoint_percent = float(
        PchipInterpolator(source_power, source_efficiency_percent, extrapolate=False)(
            FC_SOURCE_SYSTEM_RATED_POWER_KW
        )
    )
    within_domain = source_power < FC_SOURCE_SYSTEM_RATED_POWER_KW
    source_domain_power = np.append(
        source_power[within_domain], FC_SOURCE_SYSTEM_RATED_POWER_KW
    )
    source_domain_eta = np.append(
        source_efficiency_percent[within_domain], endpoint_percent
    )
    axis_scale = FORMAL_FC_RATED_POWER_KW / FC_SOURCE_SYSTEM_RATED_POWER_KW
    return (
        tuple(float(value) for value in source_domain_power * axis_scale),
        tuple(float(value) for value in source_domain_eta / 100.0),
    )


FORMAL_FC_POWER_POINTS_KW, FORMAL_FC_EFFICIENCIES = _formal_curve_points()


@dataclass(frozen=True)
class FuelCellEfficiencyMap:
    """PCHIP efficiency map on an explicit absolute-power domain."""

    power_kw: Sequence[float]
    efficiencies: Sequence[float]
    rated_power_kw: float
    provenance: FuelCellEfficiencyProvenance
    _interpolator: PchipInterpolator = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        power = _strict_array(self.power_kw, "power_kw").reshape(-1)
        efficiency = _strict_array(self.efficiencies, "efficiencies").reshape(-1)
        rated = _strict_scalar(self.rated_power_kw, "rated_power_kw")
        if power.size < 2 or power.size != efficiency.size:
            raise ValueError("efficiency map requires matching arrays with at least two points")
        if not np.all(np.diff(power) > 0.0):
            raise ValueError("power_kw must be strictly increasing")
        if rated <= 0.0 or power[0] != 0.0 or power[-1] != rated:
            raise ValueError("efficiency map must cover exactly 0..rated_power_kw")
        if np.any(efficiency < 0.0) or np.any(efficiency > 1.0):
            raise ValueError("efficiencies must lie in [0, 1]")
        if np.any(efficiency[power > 0.0] <= 0.0):
            raise ValueError("efficiency must be positive at every positive-power point")
        if not isinstance(self.provenance, FuelCellEfficiencyProvenance):
            raise TypeError("a complete FuelCellEfficiencyProvenance is required")
        object.__setattr__(self, "power_kw", tuple(float(value) for value in power))
        object.__setattr__(self, "efficiencies", tuple(float(value) for value in efficiency))
        object.__setattr__(self, "rated_power_kw", rated)
        object.__setattr__(
            self,
            "_interpolator",
            PchipInterpolator(power, efficiency, extrapolate=False),
        )

    def eta(self, p_fc_kw: float | np.ndarray) -> float | np.ndarray:
        values = _strict_array(p_fc_kw, "p_fc_kw")
        if np.any(values < 0.0) or np.any(values > self.rated_power_kw):
            raise ValueError(
                f"p_fc_kw must lie in the calibrated range [0, {self.rated_power_kw:g}]"
            )
        result = np.asarray(self._interpolator(values), dtype=np.float64)
        if np.any(result < -1.0e-12) or np.any(result > 1.0 + 1.0e-12):
            raise RuntimeError("shape-preserving interpolation escaped physical efficiency bounds")
        if values.ndim == 0:
            return float(result)
        return result

    def require_formal_calibration(self) -> FuelCellEfficiencyMap:
        """Reject any map not identical to the authoritative formal calibration."""

        if self.provenance != FC_DATA_PROVENANCE:
            raise ValueError("formal fuel-cell accounting requires authoritative provenance")
        if (
            self.rated_power_kw != FORMAL_FC_RATED_POWER_KW
            or self.power_kw != FORMAL_FC_POWER_POINTS_KW
            or self.efficiencies != FORMAL_FC_EFFICIENCIES
        ):
            raise ValueError("formal fuel-cell accounting requires the authoritative curve")
        return self


def calibrated_fuel_cell_efficiency_map() -> FuelCellEfficiencyMap:
    """Build the formal 600 kW map from the authorized 100 kW source curve."""

    return FuelCellEfficiencyMap(
        power_kw=FORMAL_FC_POWER_POINTS_KW,
        efficiencies=FORMAL_FC_EFFICIENCIES,
        rated_power_kw=FORMAL_FC_RATED_POWER_KW,
        provenance=FC_DATA_PROVENANCE,
    )


def formal_fuel_cell_efficiency_map() -> FuelCellEfficiencyMap:
    """Alias that emphasizes this is the source-backed formal default."""

    return calibrated_fuel_cell_efficiency_map()


def hydrogen_mass_kg_unverified(
    p_fc_kw: float,
    dt_seconds: float,
    *,
    efficiency: float,
    rated_power_kw: float = FORMAL_FC_RATED_POWER_KW,
) -> float:
    """Pure hydrogen math for synthetic checks; not a formal v2 boundary."""

    power = _strict_scalar(p_fc_kw, "p_fc_kw")
    duration = _strict_scalar(dt_seconds, "dt_seconds")
    eta = _strict_scalar(efficiency, "efficiency")
    rated = _strict_scalar(rated_power_kw, "rated_power_kw")
    if rated <= 0.0:
        raise ValueError("rated_power_kw must be positive")
    if power < 0.0 or power > rated:
        raise ValueError("p_fc_kw must lie within 0..rated_power_kw")
    if duration <= 0.0:
        raise ValueError("dt_seconds must be positive")
    if eta < 0.0 or eta > 1.0 or (power > 0.0 and eta == 0.0):
        raise ValueError("efficiency must be in (0, 1] at positive power")
    if power == 0.0:
        return 0.0
    dt_hours = duration / 3600.0
    return power * dt_hours / (eta * LHV_H2_KWH_PER_KG)


def hydrogen_mass_from_map_kg(
    p_fc_kw: float,
    dt_seconds: float,
    efficiency_map: FuelCellEfficiencyMap,
) -> float:
    """Return a step's hydrogen mass using the map's domain and efficiency."""

    if type(efficiency_map) is not FuelCellEfficiencyMap:
        raise TypeError("efficiency_map must be an exact FuelCellEfficiencyMap")
    FuelCellEfficiencyMap.require_formal_calibration(efficiency_map)
    power = _strict_scalar(p_fc_kw, "p_fc_kw")
    eta = float(efficiency_map.eta(power))
    return hydrogen_mass_kg_unverified(
        power,
        dt_seconds,
        efficiency=eta,
        rated_power_kw=efficiency_map.rated_power_kw,
    )
