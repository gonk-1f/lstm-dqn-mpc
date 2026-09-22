"""SOC/current-weighted battery throughput with guarded lifetime conversion."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real


BATTERY_DEGRADATION_MODEL_VERSION = "soc_current_weighted_throughput_v1"
BATTERY_DEGRADATION_SOURCE_DOI = "10.3390/en14133810"
BATTERY_ENERGY_CAPACITY_KWH = 624.0
BATTERY_NOMINAL_VOLTAGE_V = 432.0
BATTERY_NOMINAL_CHARGE_CAPACITY_AH = 624_000.0 / 432.0
BATTERY_CURRENT_REF_1C_A = BATTERY_NOMINAL_CHARGE_CAPACITY_AH
BATTERY_LIFETIME_THROUGHPUT_FACTOR = 15_000.0
BATTERY_LIFETIME_Q_AH = (
    BATTERY_LIFETIME_THROUGHPUT_FACTOR
    * BATTERY_NOMINAL_CHARGE_CAPACITY_AH
)
BATTERY_REPLACEMENT_COST_CNY = 2_000.0 * BATTERY_ENERGY_CAPACITY_KWH
BATTERY_LIFETIME_SENSITIVITY_FACTORS = (10_000.0, 15_000.0, 20_000.0)
BATTERY_LIFETIME_CONFIGURATION_STATUS = "FROZEN"
BATTERY_LIFETIME_EVIDENCE_STATUS = (
    "SECONDARY_LITERATURE / LITERATURE-CALIBRATED"
)
BATTERY_LIFETIME_PROVENANCE_CLASSIFICATION = (
    "literature-based lifetime-throughput modeling assumption"
)
BATTERY_LIFETIME_EVIDENCE_BASIS = "secondary literature basis"
BATTERY_LIFETIME_APPLICABILITY = (
    "frozen project baseline battery lifetime normalization; not vessel measured"
)


def _strict_scalar(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real numeric scalar, not bool or text")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _strict_text(value: object, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact str")
    if not value.strip():
        raise ValueError(f"{name} must be non-empty")
    return value


def _exact_nonnegative_float(value: object, name: str) -> float:
    if type(value) is not float:
        raise TypeError(f"{name} must be an exact float")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return value


def soc_stress(soc: float) -> float:
    """Return F(SOC) = 1 + 3.25 (1 - SOC)^2."""

    state = _strict_scalar(soc, "soc")
    if state < 0.0 or state > 1.0:
        raise ValueError("soc must lie in [0, 1]")
    return 1.0 + 3.25 * (1.0 - state) ** 2


def current_stress(current_a: float, nominal_current_a: float) -> float:
    """Return G(I), with positive current defined as discharge."""

    current = _strict_scalar(current_a, "current_a")
    nominal = _strict_scalar(nominal_current_a, "nominal_current_a")
    if nominal <= 0.0:
        raise ValueError("nominal_current_a must be positive")
    if current >= 0.0:
        return 1.0 + 0.45 * current / nominal
    return 1.0 + 0.55 * abs(current) / nominal


@dataclass(frozen=True)
class BatteryDegradationStep:
    """Raw and stress-weighted ampere-hour throughput for one interval."""

    raw_ah: float
    weighted_ah: float
    soc_stress: float
    current_stress: float

    def __post_init__(self) -> None:
        for name in ("raw_ah", "weighted_ah", "soc_stress", "current_stress"):
            _exact_nonnegative_float(getattr(self, name), name)


def battery_degradation_step(
    soc: float,
    current_a: float,
    dt_seconds: float,
    nominal_current_a: float,
) -> BatteryDegradationStep:
    """Calculate throughput without charge/discharge energy efficiencies."""

    state = _strict_scalar(soc, "soc")
    current = _strict_scalar(current_a, "current_a")
    duration = _strict_scalar(dt_seconds, "dt_seconds")
    nominal = _strict_scalar(nominal_current_a, "nominal_current_a")
    if state < 0.0 or state > 1.0:
        raise ValueError("soc must lie in [0, 1]")
    if duration <= 0.0:
        raise ValueError("dt_seconds must be positive")
    if nominal <= 0.0:
        raise ValueError("nominal_current_a must be positive")

    f_soc = soc_stress(state)
    g_current = current_stress(current, nominal)
    raw_ah = abs(current) * duration / 3600.0
    return BatteryDegradationStep(
        raw_ah=raw_ah,
        weighted_ah=raw_ah * f_soc * g_current,
        soc_stress=f_soc,
        current_stress=g_current,
    )


@dataclass
class BatteryThroughputAccount:
    """Cumulative raw and weighted Ah, with no implied lifetime denominator."""

    raw_ah: float = 0.0
    weighted_ah: float = 0.0

    def __post_init__(self) -> None:
        self._validated_values()

    def _validated_values(self) -> tuple[float, float]:
        return (
            _exact_nonnegative_float(self.raw_ah, "raw_ah"),
            _exact_nonnegative_float(self.weighted_ah, "weighted_ah"),
        )

    def add(self, step: BatteryDegradationStep) -> None:
        if type(step) is not BatteryDegradationStep:
            raise TypeError("step must be an exact BatteryDegradationStep")
        raw_ah, weighted_ah = self._validated_values()
        updated_raw_ah = raw_ah + step.raw_ah
        updated_weighted_ah = weighted_ah + step.weighted_ah
        if not math.isfinite(updated_raw_ah) or not math.isfinite(updated_weighted_ah):
            raise ValueError("cumulative throughput addition must remain finite")
        self.raw_ah = updated_raw_ah
        self.weighted_ah = updated_weighted_ah


@dataclass(frozen=True)
class BatteryLifetimeNormalization:
    """Frozen lifetime-throughput baseline with an explicit evidence boundary."""

    q_lifetime_ah: float
    throughput_factor: float
    nominal_charge_capacity_ah: float
    provenance_classification: str
    evidence_basis: str
    applicability: str

    def __post_init__(self) -> None:
        numeric = (
            ("q_lifetime_ah", self.q_lifetime_ah, BATTERY_LIFETIME_Q_AH),
            (
                "throughput_factor",
                self.throughput_factor,
                BATTERY_LIFETIME_THROUGHPUT_FACTOR,
            ),
            (
                "nominal_charge_capacity_ah",
                self.nominal_charge_capacity_ah,
                BATTERY_NOMINAL_CHARGE_CAPACITY_AH,
            ),
        )
        for name, value, expected in numeric:
            if type(value) is not float:
                raise TypeError(f"{name} must be an exact float")
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            if value != expected:
                raise ValueError(f"{name} must equal the approved derived value")

        text = (
            (
                "provenance_classification",
                self.provenance_classification,
                BATTERY_LIFETIME_PROVENANCE_CLASSIFICATION,
            ),
            (
                "evidence_basis",
                self.evidence_basis,
                BATTERY_LIFETIME_EVIDENCE_BASIS,
            ),
            (
                "applicability",
                self.applicability,
                BATTERY_LIFETIME_APPLICABILITY,
            ),
        )
        for name, value, expected in text:
            _strict_text(value, name)
            if value != expected:
                raise ValueError(f"{name} must preserve the approved source role")

    def require_literature_calibrated(self) -> BatteryLifetimeNormalization:
        BatteryLifetimeNormalization.__post_init__(self)
        return self


def formal_battery_lifetime_normalization() -> BatteryLifetimeNormalization:
    """Return the frozen baseline with secondary-literature provenance."""

    return BatteryLifetimeNormalization(
        q_lifetime_ah=BATTERY_LIFETIME_Q_AH,
        throughput_factor=BATTERY_LIFETIME_THROUGHPUT_FACTOR,
        nominal_charge_capacity_ah=BATTERY_NOMINAL_CHARGE_CAPACITY_AH,
        provenance_classification=BATTERY_LIFETIME_PROVENANCE_CLASSIFICATION,
        evidence_basis=BATTERY_LIFETIME_EVIDENCE_BASIS,
        applicability=BATTERY_LIFETIME_APPLICABILITY,
    )


@dataclass(frozen=True)
class BatteryLifeState:
    """Cumulative weighted-Ah life state at one interval boundary."""

    cumulative_weighted_ah: float
    raw_life_fraction: float
    economic_life_fraction: float
    eol_reached: bool

    def __post_init__(self) -> None:
        _exact_nonnegative_float(
            self.cumulative_weighted_ah,
            "cumulative_weighted_ah",
        )
        _exact_nonnegative_float(self.raw_life_fraction, "raw_life_fraction")
        economic = _exact_nonnegative_float(
            self.economic_life_fraction,
            "economic_life_fraction",
        )
        if economic > 1.0:
            raise ValueError("economic_life_fraction must lie in [0, 1]")
        if type(self.eol_reached) is not bool:
            raise TypeError("eol_reached must be an exact bool")


@dataclass(frozen=True)
class BatteryLifeIncrement:
    """Non-duplicating economic life consumed by one physical interval."""

    before: BatteryLifeState
    after: BatteryLifeState
    delta_economic_fraction: float

    def __post_init__(self) -> None:
        if type(self.before) is not BatteryLifeState:
            raise TypeError("before must be an exact BatteryLifeState")
        if type(self.after) is not BatteryLifeState:
            raise TypeError("after must be an exact BatteryLifeState")
        delta = _exact_nonnegative_float(
            self.delta_economic_fraction,
            "delta_economic_fraction",
        )
        if delta > 1.0:
            raise ValueError("delta_economic_fraction must lie in [0, 1]")


def battery_life_state(
    cumulative_weighted_ah: float,
    *,
    normalization: BatteryLifetimeNormalization,
) -> BatteryLifeState:
    """Return raw, clipped-economic, and EOL cumulative diagnostics."""

    throughput = _strict_scalar(
        cumulative_weighted_ah,
        "cumulative_weighted_ah",
    )
    if throughput < 0.0:
        raise ValueError("cumulative_weighted_ah must be non-negative")
    if type(normalization) is not BatteryLifetimeNormalization:
        raise TypeError("normalization must use the exact provenance-bearing type")
    checked = normalization.require_literature_calibrated()
    raw = throughput / checked.q_lifetime_ah
    economic = min(raw, 1.0)
    return BatteryLifeState(
        cumulative_weighted_ah=throughput,
        raw_life_fraction=raw,
        economic_life_fraction=economic,
        eol_reached=raw >= 1.0,
    )


def formal_battery_interval_life_loss(
    cumulative_weighted_ah_before: float,
    cumulative_weighted_ah_after: float,
    *,
    normalization: BatteryLifetimeNormalization,
) -> BatteryLifeIncrement:
    """Return the clipped cumulative-life difference for one interval."""

    before = battery_life_state(
        cumulative_weighted_ah_before,
        normalization=normalization,
    )
    after = battery_life_state(
        cumulative_weighted_ah_after,
        normalization=normalization,
    )
    if after.cumulative_weighted_ah < before.cumulative_weighted_ah:
        raise ValueError("cumulative weighted Ah must not decrease")
    return BatteryLifeIncrement(
        before=before,
        after=after,
        delta_economic_fraction=(
            after.economic_life_fraction - before.economic_life_fraction
        ),
    )


def battery_relative_life_loss_unverified(
    weighted_ah: float,
    *,
    q_lifetime_ah: float,
) -> float:
    """Divide weighted Ah by a caller-supplied synthetic denominator."""

    throughput = _strict_scalar(weighted_ah, "weighted_ah")
    lifetime = _strict_scalar(q_lifetime_ah, "q_lifetime_ah")
    if throughput < 0.0:
        raise ValueError("weighted_ah must be non-negative")
    if lifetime <= 0.0:
        raise ValueError("q_lifetime_ah must be positive")
    return throughput / lifetime


def formal_battery_relative_life_loss(
    weighted_ah: float,
    *,
    normalization: BatteryLifetimeNormalization,
) -> float:
    """Return cumulative raw life under the frozen literature calibration."""

    throughput = _strict_scalar(weighted_ah, "weighted_ah")
    if throughput < 0.0:
        raise ValueError("weighted_ah must be non-negative")
    if type(normalization) is not BatteryLifetimeNormalization:
        raise TypeError("formal normalization must use the exact provenance-bearing type")
    checked = normalization.require_literature_calibrated()
    return throughput / checked.q_lifetime_ah


def require_lifetime_normalization(
    *,
    q_nominal_ah: float,
    q_lifetime_ah: object,
) -> None:
    """Reject legacy bare-number lifetime normalization attempts."""

    nominal = _strict_scalar(q_nominal_ah, "q_nominal_ah")
    if nominal <= 0.0:
        raise ValueError("q_nominal_ah must be positive")
    if type(q_lifetime_ah) is not BatteryLifetimeNormalization:
        raise TypeError(
            "bare q_lifetime_ah is not formal calibration; provenance is required"
        )
    q_lifetime_ah.require_literature_calibrated()


def formal_battery_degradation_cost_cny(
    cumulative_weighted_ah_before: float,
    cumulative_weighted_ah_after: float,
    *,
    replacement_cost_cny: float,
    normalization: BatteryLifetimeNormalization,
) -> float:
    """Charge only newly consumed clipped life in the current interval."""

    increment = formal_battery_interval_life_loss(
        cumulative_weighted_ah_before,
        cumulative_weighted_ah_after,
        normalization=normalization,
    )
    price = _strict_scalar(replacement_cost_cny, "replacement_cost_cny")
    if price < 0.0:
        raise ValueError("replacement_cost_cny must be non-negative")
    if price != BATTERY_REPLACEMENT_COST_CNY:
        raise ValueError(
            "replacement_cost_cny must equal 2000 CNY/kWh * 624 kWh"
        )
    result = increment.delta_economic_fraction * price
    if not math.isfinite(result):
        raise ValueError("battery interval degradation cost must remain finite")
    return result
