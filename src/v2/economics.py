"""Raw-CNY macro-interval economics with fail-closed formal boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
from numbers import Real

import numpy as np

from .analysis.action_screening import (
    DataSplit,
    DatasetProvenance,
    HeldOutSelectionError,
)
from .contracts import REWARD_VERSION
from .models.battery_degradation import (
    BatteryLifetimeNormalization,
    formal_battery_degradation_cost_cny,
)
from .models.battery_energy import BatteryEfficiency
from .models.fuel_cell_degradation import (
    formal_fuel_cell_degradation_cost_cny,
)


HYDROGEN_PRICE_CNY_PER_KG = 35.0
FUEL_CELL_PRICE_CNY_PER_KW = 3500.0
BATTERY_PRICE_CNY_PER_KWH = 2000.0
SHORE_TARIFF_CNY_PER_KWH = 1.10

EQUIPMENT_PRICE_SOURCE_DOI = "10.3390/jmse13010034"
SHORE_TARIFF_SOURCE_DOI = "10.11930/j.issn.1004-9649.202507065"
SHORE_CONVERTER_CALIBRATION_STATUS = "NO-GO"
SHORE_CHARGING_EFFICIENCY_STATUS = "VERIFIED"
SHORE_CHARGING_EFFICIENCY_EVIDENCE = (
    "literature-based aggregate assumption; not vessel-measured"
)
DEGRADATION_COST_STATUS = "INTERVAL-CALIBRATED"


def _finite_scalar(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real numeric scalar, not bool or text")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _nonnegative_scalar(value: object, name: str) -> float:
    result = _finite_scalar(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _positive_scalar(value: object, name: str) -> float:
    result = _finite_scalar(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _finite_sum(values: tuple[float, ...], name: str) -> float:
    try:
        result = math.fsum(values)
    except OverflowError as exc:
        raise ValueError(f"{name} must remain finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must remain finite")
    return result


def _exact_text(value: object, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact str")
    if not value.strip():
        raise ValueError(f"{name} must be non-empty")
    return value


@dataclass(frozen=True)
class PriceSource:
    """Closed provenance record for the two approved price-source roles."""

    source_doi: str
    source_location: str
    role: str
    classification: str

    def __post_init__(self) -> None:
        values = tuple(
            _exact_text(getattr(self, name), name)
            for name in ("source_doi", "source_location", "role", "classification")
        )
        approved = {
            (
                EQUIPMENT_PRICE_SOURCE_DOI,
                "economic parameter table",
                "hydrogen and equipment unit-price source only",
                "published economic assumptions",
            ),
            (
                SHORE_TARIFF_SOURCE_DOI,
                "Table 2",
                "shore electricity scenario tariff source only",
                "scenario_not_measured",
            ),
        }
        if values not in approved:
            raise ValueError("price provenance must exactly match an approved source role")


EQUIPMENT_PRICE_SOURCE = PriceSource(
    source_doi=EQUIPMENT_PRICE_SOURCE_DOI,
    source_location="economic parameter table",
    role="hydrogen and equipment unit-price source only",
    classification="published economic assumptions",
)
SHORE_TARIFF_SOURCE = PriceSource(
    source_doi=SHORE_TARIFF_SOURCE_DOI,
    source_location="Table 2",
    role="shore electricity scenario tariff source only",
    classification="scenario_not_measured",
)


@dataclass(frozen=True)
class EconomicPriceCatalog:
    """Formal fixed unit prices; ``None`` is an explicit missing-price gate."""

    hydrogen_cny_per_kg: float | None
    fuel_cell_cny_per_kw: float | None
    battery_cny_per_kwh: float | None
    shore_cny_per_kwh: float | None

    def __post_init__(self) -> None:
        expected = {
            "hydrogen_cny_per_kg": HYDROGEN_PRICE_CNY_PER_KG,
            "fuel_cell_cny_per_kw": FUEL_CELL_PRICE_CNY_PER_KW,
            "battery_cny_per_kwh": BATTERY_PRICE_CNY_PER_KWH,
            "shore_cny_per_kwh": SHORE_TARIFF_CNY_PER_KWH,
        }
        for name, approved in expected.items():
            value = getattr(self, name)
            if value is None:
                continue
            checked = _positive_scalar(value, name)
            if checked != approved:
                raise ValueError(f"{name} must equal the approved source-backed value")
            object.__setattr__(self, name, checked)

    @classmethod
    def formal_scenario(cls) -> EconomicPriceCatalog:
        return cls(
            hydrogen_cny_per_kg=HYDROGEN_PRICE_CNY_PER_KG,
            fuel_cell_cny_per_kw=FUEL_CELL_PRICE_CNY_PER_KW,
            battery_cny_per_kwh=BATTERY_PRICE_CNY_PER_KWH,
            shore_cny_per_kwh=SHORE_TARIFF_CNY_PER_KWH,
        )


FORMAL_PRICE_CATALOG = EconomicPriceCatalog.formal_scenario()


def _validate_prices(value: object) -> EconomicPriceCatalog:
    if type(value) is not EconomicPriceCatalog:
        raise TypeError("prices must be an exact EconomicPriceCatalog")
    expected = (
        ("hydrogen_cny_per_kg", HYDROGEN_PRICE_CNY_PER_KG),
        ("fuel_cell_cny_per_kw", FUEL_CELL_PRICE_CNY_PER_KW),
        ("battery_cny_per_kwh", BATTERY_PRICE_CNY_PER_KWH),
        ("shore_cny_per_kwh", SHORE_TARIFF_CNY_PER_KWH),
    )
    for name, approved in expected:
        stored = getattr(value, name)
        if stored is not None and (type(stored) is not float or stored != approved):
            raise ValueError(f"stored {name} no longer matches approved provenance")
    return value


class ShoreEnergyClassification(Enum):
    MEASURED = "measured"
    MODELED = "modeled"


@dataclass(frozen=True)
class ShoreEnergy:
    """Explicit shore-energy channel; measured status is never inferred."""

    energy_kwh: float
    classification: ShoreEnergyClassification

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "energy_kwh", _nonnegative_scalar(self.energy_kwh, "energy_kwh")
        )
        if type(self.classification) is not ShoreEnergyClassification:
            raise TypeError("classification must be an exact ShoreEnergyClassification")


def _validate_shore_energy(value: object) -> ShoreEnergy:
    if type(value) is not ShoreEnergy:
        raise TypeError("shore_energy must be an exact ShoreEnergy")
    if type(value.energy_kwh) is not float:
        raise TypeError("stored shore energy must remain an exact float")
    _nonnegative_scalar(value.energy_kwh, "energy_kwh")
    if type(value.classification) is not ShoreEnergyClassification:
        raise TypeError("stored shore classification is invalid")
    return value


def hydrogen_cost_cny(
    hydrogen_mass_kg: float,
    *,
    prices: EconomicPriceCatalog = FORMAL_PRICE_CATALOG,
) -> float:
    mass = _nonnegative_scalar(hydrogen_mass_kg, "hydrogen_mass_kg")
    catalog = _validate_prices(prices)
    if catalog.hydrogen_cny_per_kg is None:
        raise ValueError("hydrogen price is missing")
    result = mass * catalog.hydrogen_cny_per_kg
    if not math.isfinite(result):
        raise ValueError("hydrogen cost must remain finite")
    return result


def shore_energy_cost_cny(
    shore_energy: ShoreEnergy,
    *,
    prices: EconomicPriceCatalog = FORMAL_PRICE_CATALOG,
) -> float:
    energy = _validate_shore_energy(shore_energy)
    catalog = _validate_prices(prices)
    if catalog.shore_cny_per_kwh is None:
        raise ValueError("shore tariff is missing")
    result = energy.energy_kwh * catalog.shore_cny_per_kwh
    if not math.isfinite(result):
        raise ValueError("shore energy cost must remain finite")
    return result


@dataclass(frozen=True)
class ShoreConverterCalibration:
    """Proposed converter calibration; no formal value is approved yet."""

    eta_shore_converter: float
    status: str
    source_reference: str

    def __post_init__(self) -> None:
        efficiency = _positive_scalar(
            self.eta_shore_converter, "eta_shore_converter"
        )
        if efficiency > 1.0:
            raise ValueError("eta_shore_converter must lie in (0, 1]")
        object.__setattr__(self, "eta_shore_converter", efficiency)
        status = _exact_text(self.status, "status")
        _exact_text(self.source_reference, "source_reference")
        if status != "UNVERIFIED":
            raise ValueError("no verified shore-converter status is currently approved")

    def require_verified(self) -> float:
        raise ValueError(
            "formal modeled grid recharge is NO-GO: no verified shore-converter "
            "efficiency calibration is available"
        )


def terminal_recharge_grid_energy_unverified(
    *,
    episode_initial_soc: float,
    episode_end_soc: float,
    battery_capacity_kwh: float,
    battery_efficiency: BatteryEfficiency,
    eta_shore_converter: float,
) -> ShoreEnergy:
    """Pure synthetic calculator requiring an explicit converter efficiency."""

    initial = _finite_scalar(episode_initial_soc, "episode_initial_soc")
    end = _finite_scalar(episode_end_soc, "episode_end_soc")
    if not 0.0 <= initial <= 1.0 or not 0.0 <= end <= 1.0:
        raise ValueError("episode SOC values must lie in [0, 1]")
    capacity = _positive_scalar(battery_capacity_kwh, "battery_capacity_kwh")
    if type(battery_efficiency) is not BatteryEfficiency:
        raise TypeError("battery_efficiency must be an exact BatteryEfficiency")
    eta_chg, _ = BatteryEfficiency.require_calibrated(battery_efficiency)
    converter = _positive_scalar(eta_shore_converter, "eta_shore_converter")
    if converter > 1.0:
        raise ValueError("eta_shore_converter must lie in (0, 1]")
    battery_side_needed_kwh = max(0.0, initial - end) * capacity
    grid_energy = battery_side_needed_kwh / eta_chg / converter
    if not math.isfinite(grid_energy):
        raise ValueError("terminal grid recharge energy must remain finite")
    return ShoreEnergy(grid_energy, ShoreEnergyClassification.MODELED)


def terminal_recharge_grid_energy(
    *,
    episode_initial_soc: float,
    episode_end_soc: float,
    battery_capacity_kwh: float,
    battery_efficiency: BatteryEfficiency,
) -> ShoreEnergy:
    """Apply the approved aggregate 0.95 charge-path efficiency exactly once."""

    initial = _finite_scalar(episode_initial_soc, "episode_initial_soc")
    end = _finite_scalar(episode_end_soc, "episode_end_soc")
    if not 0.0 <= initial <= 1.0 or not 0.0 <= end <= 1.0:
        raise ValueError("episode SOC values must lie in [0, 1]")
    capacity = _positive_scalar(battery_capacity_kwh, "battery_capacity_kwh")
    if type(battery_efficiency) is not BatteryEfficiency:
        raise TypeError("battery_efficiency must be an exact BatteryEfficiency")
    eta_chg, _ = BatteryEfficiency.require_calibrated(battery_efficiency)
    battery_side_needed_kwh = max(0.0, initial - end) * capacity
    grid_energy = battery_side_needed_kwh / eta_chg
    if not math.isfinite(grid_energy):
        raise ValueError("terminal grid recharge energy must remain finite")
    return ShoreEnergy(
        grid_energy,
        ShoreEnergyClassification.MODELED,
    )


@dataclass(frozen=True)
class RawCnyIntervalLedger:
    """Exactly four unweighted raw-CNY cost components for one interval."""

    h2_cost_cny: float
    fuel_cell_degradation_cost_cny: float
    battery_degradation_cost_cny: float
    shore_cost_cny: float

    def __post_init__(self) -> None:
        for name in (
            "h2_cost_cny",
            "fuel_cell_degradation_cost_cny",
            "battery_degradation_cost_cny",
            "shore_cost_cny",
        ):
            object.__setattr__(
                self, name, _nonnegative_scalar(getattr(self, name), name)
            )
        _finite_sum(self.components_cny, "interval cost sum")

    @property
    def components_cny(self) -> tuple[float, float, float, float]:
        components = (
            self.h2_cost_cny,
            self.fuel_cell_degradation_cost_cny,
            self.battery_degradation_cost_cny,
            self.shore_cost_cny,
        )
        if any(type(value) is not float for value in components):
            raise TypeError("stored ledger components must remain exact floats")
        for index, value in enumerate(components):
            _nonnegative_scalar(value, f"component_{index}_cny")
        return components

    @property
    def total_cost_cny(self) -> float:
        return _finite_sum(self.components_cny, "interval cost sum")

    @property
    def reward_cny(self) -> float:
        return -self.total_cost_cny


def build_formal_interval_ledger(
    *,
    hydrogen_mass_kg: float,
    fuel_cell_cumulative_voltage_loss_before_uv: float,
    fuel_cell_cumulative_voltage_loss_after_uv: float,
    fuel_cell_rated_kw: float,
    battery_cumulative_weighted_ah_before: float,
    battery_cumulative_weighted_ah_after: float,
    battery_capacity_kwh: float,
    battery_normalization: BatteryLifetimeNormalization,
    shore_energy: ShoreEnergy | None,
    prices: EconomicPriceCatalog = FORMAL_PRICE_CATALOG,
) -> RawCnyIntervalLedger:
    """Build one physical interval ledger from cumulative before/after states."""

    catalog = _validate_prices(prices)
    fc_rated = _positive_scalar(fuel_cell_rated_kw, "fuel_cell_rated_kw")
    battery_capacity = _positive_scalar(
        battery_capacity_kwh, "battery_capacity_kwh"
    )
    if fc_rated != 600.0:
        raise ValueError("fuel_cell_rated_kw must equal the approved 600 kW rating")
    if battery_capacity != 624.0:
        raise ValueError(
            "battery_capacity_kwh must equal the approved 624 kWh capacity"
        )
    if type(battery_normalization) is not BatteryLifetimeNormalization:
        raise TypeError(
            "battery_normalization must be an exact BatteryLifetimeNormalization"
        )
    if shore_energy is not None:
        _validate_shore_energy(shore_energy)
    if catalog.fuel_cell_cny_per_kw is None:
        raise ValueError("fuel-cell equipment price is missing")
    if catalog.battery_cny_per_kwh is None:
        raise ValueError("battery equipment price is missing")

    h2_cost = hydrogen_cost_cny(hydrogen_mass_kg, prices=catalog)
    fc_cost = formal_fuel_cell_degradation_cost_cny(
        fuel_cell_cumulative_voltage_loss_before_uv,
        fuel_cell_cumulative_voltage_loss_after_uv,
        replacement_cost_cny=(
            fc_rated * catalog.fuel_cell_cny_per_kw
        ),
    )
    battery_cost = formal_battery_degradation_cost_cny(
        battery_cumulative_weighted_ah_before,
        battery_cumulative_weighted_ah_after,
        replacement_cost_cny=(
            battery_capacity * catalog.battery_cny_per_kwh
        ),
        normalization=battery_normalization,
    )
    shore_cost = (
        0.0
        if shore_energy is None
        else shore_energy_cost_cny(shore_energy, prices=catalog)
    )
    return RawCnyIntervalLedger(h2_cost, fc_cost, battery_cost, shore_cost)


def _provenance_payload(value: DatasetProvenance) -> tuple[str, str, str]:
    if type(value) is not DatasetProvenance:
        raise TypeError("provenance must be an exact DatasetProvenance")
    if type(value.dataset_version) is not str or not value.dataset_version.strip():
        raise ValueError("dataset_version must remain a non-empty exact string")
    if type(value.provenance_id) is not str or not value.provenance_id.strip():
        raise ValueError("provenance_id must remain a non-empty exact string")
    if type(value.split) is not DataSplit:
        raise TypeError("split must remain an exact DataSplit")
    return value.dataset_version, value.provenance_id, value.split.value


REWARD_SCALE_DERIVATION_RULE = "positive_arithmetic_mean_v1"
_REWARD_SCALE_SEAL = object()


def _derive_reward_scale(train_raw_costs_cny: tuple[float, ...]) -> float:
    scale = _finite_sum(train_raw_costs_cny, "Train raw-cost sum") / len(
        train_raw_costs_cny
    )
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("Train-calibrated reward scale must be finite and positive")
    return scale


def _reward_scale_digest(
    *,
    scale_cny: float,
    train_raw_costs_cny: tuple[float, ...],
    sample_count: int,
    provenance: DatasetProvenance,
    derivation_rule: str,
    audit_id: str,
    reason: str,
) -> str:
    payload = {
        "reward_version": REWARD_VERSION,
        "scale_cny": scale_cny.hex(),
        "train_raw_costs_cny": [value.hex() for value in train_raw_costs_cny],
        "sample_count": sample_count,
        "provenance": _provenance_payload(provenance),
        "derivation_rule": derivation_rule,
        "audit_id": audit_id,
        "reason": reason,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, init=False)
class RewardScaleCalibration:
    """Sealed Train evidence and its deterministically derived reward scale."""

    scale_cny: float
    train_raw_costs_cny: tuple[float, ...]
    sample_count: int
    provenance: DatasetProvenance
    derivation_rule: str
    audit_id: str
    reason: str
    digest: str = field(repr=False)
    _seal: object = field(repr=False, compare=False)

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError(
            "RewardScaleCalibration is sealed; use calibrate_reward_scale with "
            "immutable Train costs and audit evidence"
        )


def _validate_reward_scale(value: object) -> RewardScaleCalibration:
    if type(value) is not RewardScaleCalibration:
        raise TypeError("calibration must be an exact RewardScaleCalibration")
    if getattr(value, "_seal", None) is not _REWARD_SCALE_SEAL:
        raise ValueError("reward-scale calibration was not issued by the sealed factory")
    if type(value.scale_cny) is not float:
        raise TypeError("stored reward scale must remain an exact float")
    scale = _positive_scalar(value.scale_cny, "scale_cny")
    if type(value.train_raw_costs_cny) is not tuple or not value.train_raw_costs_cny:
        raise ValueError("stored Train raw costs must remain a non-empty tuple")
    costs = tuple(
        _nonnegative_scalar(cost, f"train_raw_costs_cny[{index}]")
        for index, cost in enumerate(value.train_raw_costs_cny)
    )
    if any(type(cost) is not float for cost in value.train_raw_costs_cny):
        raise TypeError("stored Train raw costs must remain exact floats")
    if type(value.sample_count) is not int or value.sample_count != len(costs):
        raise ValueError("stored reward-scale sample count is inconsistent")
    derivation_rule = _exact_text(value.derivation_rule, "derivation_rule")
    if derivation_rule != REWARD_SCALE_DERIVATION_RULE:
        raise ValueError("stored reward-scale derivation rule is unsupported")
    _exact_text(value.audit_id, "audit_id")
    _exact_text(value.reason, "reason")
    _provenance_payload(value.provenance)
    if value.provenance.split is not DataSplit.TRAIN:
        raise HeldOutSelectionError("stored reward-scale provenance must remain Train")
    derived_scale = _derive_reward_scale(costs)
    if scale != derived_scale:
        raise ValueError("stored reward scale does not match its bound Train costs")
    expected = _reward_scale_digest(
        scale_cny=scale,
        train_raw_costs_cny=costs,
        sample_count=value.sample_count,
        provenance=value.provenance,
        derivation_rule=value.derivation_rule,
        audit_id=value.audit_id,
        reason=value.reason,
    )
    if type(value.digest) is not str or value.digest != expected:
        raise ValueError("reward-scale calibration has been mutated or forged")
    return value


def calibrate_reward_scale(
    train_raw_costs_cny: tuple[float, ...],
    *,
    provenance: DatasetProvenance,
    audit_id: str,
    reason: str,
) -> RewardScaleCalibration:
    """Calibrate ``C_ref`` as the arithmetic Train mean raw interval cost."""

    if type(provenance) is not DatasetProvenance:
        raise TypeError("provenance must be an exact DatasetProvenance")
    _provenance_payload(provenance)
    if provenance.split is not DataSplit.TRAIN:
        raise HeldOutSelectionError(
            "reward-scale calibration is Train-only; Validation/Test/unknown "
            "provenance is forbidden"
        )
    if type(train_raw_costs_cny) is not tuple:
        raise TypeError("Train raw costs must be an immutable tuple")
    if not train_raw_costs_cny:
        raise ValueError("Train raw costs must not be empty")
    checked_audit_id = _exact_text(audit_id, "audit_id")
    checked_reason = _exact_text(reason, "reason")
    costs = tuple(
        _nonnegative_scalar(value, f"train_raw_costs_cny[{index}]")
        for index, value in enumerate(train_raw_costs_cny)
    )
    scale = _derive_reward_scale(costs)
    sample_count = len(costs)
    digest = _reward_scale_digest(
        scale_cny=scale,
        train_raw_costs_cny=costs,
        sample_count=sample_count,
        provenance=provenance,
        derivation_rule=REWARD_SCALE_DERIVATION_RULE,
        audit_id=checked_audit_id,
        reason=checked_reason,
    )
    calibration = object.__new__(RewardScaleCalibration)
    object.__setattr__(calibration, "scale_cny", scale)
    object.__setattr__(calibration, "train_raw_costs_cny", costs)
    object.__setattr__(calibration, "sample_count", sample_count)
    object.__setattr__(calibration, "provenance", provenance)
    object.__setattr__(
        calibration, "derivation_rule", REWARD_SCALE_DERIVATION_RULE
    )
    object.__setattr__(calibration, "audit_id", checked_audit_id)
    object.__setattr__(calibration, "reason", checked_reason)
    object.__setattr__(calibration, "digest", digest)
    object.__setattr__(calibration, "_seal", _REWARD_SCALE_SEAL)
    return calibration


def scaled_reward(
    ledger: RawCnyIntervalLedger,
    *,
    calibration: RewardScaleCalibration,
) -> float:
    if type(ledger) is not RawCnyIntervalLedger:
        raise TypeError("ledger must be an exact RawCnyIntervalLedger")
    checked = _validate_reward_scale(calibration)
    result = ledger.reward_cny / checked.scale_cny
    if not math.isfinite(result):
        raise ValueError("scaled reward must remain finite")
    return result


__all__ = [
    "BATTERY_PRICE_CNY_PER_KWH",
    "DEGRADATION_COST_STATUS",
    "EQUIPMENT_PRICE_SOURCE",
    "EconomicPriceCatalog",
    "FORMAL_PRICE_CATALOG",
    "FUEL_CELL_PRICE_CNY_PER_KW",
    "HYDROGEN_PRICE_CNY_PER_KG",
    "PriceSource",
    "REWARD_VERSION",
    "RawCnyIntervalLedger",
    "REWARD_SCALE_DERIVATION_RULE",
    "RewardScaleCalibration",
    "SHORE_CONVERTER_CALIBRATION_STATUS",
    "SHORE_CHARGING_EFFICIENCY_EVIDENCE",
    "SHORE_CHARGING_EFFICIENCY_STATUS",
    "SHORE_TARIFF_CNY_PER_KWH",
    "SHORE_TARIFF_SOURCE",
    "ShoreConverterCalibration",
    "ShoreEnergy",
    "ShoreEnergyClassification",
    "build_formal_interval_ledger",
    "calibrate_reward_scale",
    "hydrogen_cost_cny",
    "scaled_reward",
    "shore_energy_cost_cny",
    "terminal_recharge_grid_energy",
    "terminal_recharge_grid_energy_unverified",
]
