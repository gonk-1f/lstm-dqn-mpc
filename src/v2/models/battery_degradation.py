"""SOC/current-weighted battery throughput with guarded lifetime conversion."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real


BATTERY_DEGRADATION_MODEL_VERSION = "soc_current_weighted_throughput_v1"
BATTERY_DEGRADATION_SOURCE_DOI = "10.3390/en14133810"
BATTERY_LIFETIME_NORMALIZATION_STATUS = "NO-GO"


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

    def add(self, step: BatteryDegradationStep) -> None:
        if type(step) is not BatteryDegradationStep:
            raise TypeError("step must be an exact BatteryDegradationStep")
        if (
            not math.isfinite(step.raw_ah)
            or not math.isfinite(step.weighted_ah)
            or step.raw_ah < 0.0
            or step.weighted_ah < 0.0
        ):
            raise ValueError("throughput values must be finite and non-negative")
        self.raw_ah += step.raw_ah
        self.weighted_ah += step.weighted_ah


@dataclass(frozen=True)
class BatteryLifetimeNormalization:
    """Proposed lifetime-throughput record; none is formally verified yet."""

    q_lifetime_ah: float
    source_doi: str
    unit: str
    chemistry_system_applicability: str

    def __post_init__(self) -> None:
        if type(self.q_lifetime_ah) is not float:
            raise TypeError("formal q_lifetime_ah must be an exact float")
        if not math.isfinite(self.q_lifetime_ah) or self.q_lifetime_ah <= 0.0:
            raise ValueError("q_lifetime_ah must be finite and positive")
        _strict_text(self.source_doi, "source_doi")
        _strict_text(self.unit, "unit")
        _strict_text(
            self.chemistry_system_applicability,
            "chemistry_system_applicability",
        )

    def require_verified(self) -> BatteryLifetimeNormalization:
        raise ValueError(
            "formal battery lifetime normalization is NO-GO: no authoritative "
            "Q_lifetime with unit and chemistry/system applicability is available"
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
    """Fail closed until lifetime throughput has verified applicability."""

    throughput = _strict_scalar(weighted_ah, "weighted_ah")
    if throughput < 0.0:
        raise ValueError("weighted_ah must be non-negative")
    if type(normalization) is not BatteryLifetimeNormalization:
        raise TypeError("formal normalization must use the exact provenance-bearing type")
    BatteryLifetimeNormalization.require_verified(normalization)
    raise AssertionError("unreachable until a formal calibration is authorized")


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
    BatteryLifetimeNormalization.require_verified(q_lifetime_ah)


def formal_battery_degradation_cost_cny(
    weighted_ah: float,
    *,
    replacement_cost_cny: float,
    normalization: BatteryLifetimeNormalization,
) -> float:
    """Fail closed before raw/weighted Ah can be multiplied by equipment price."""

    relative_loss = formal_battery_relative_life_loss(
        weighted_ah, normalization=normalization
    )
    price = _strict_scalar(replacement_cost_cny, "replacement_cost_cny")
    if price < 0.0:
        raise ValueError("replacement_cost_cny must be non-negative")
    return relative_loss * price
