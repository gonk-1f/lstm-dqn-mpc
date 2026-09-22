"""Aggregate fuel-cell voltage-loss accounting with fail-closed normalization."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real


FC_DEGRADATION_MODEL_VERSION = "aggregate_four_condition_voltage_loss_v1"
FC_DEGRADATION_SOURCE_DOI = "10.1016/j.ijhydene.2024.02.349"
FC_ACCOUNTING_STRUCTURE_SOURCE_DOI = "10.3390/jmse13010034"

FC_LOW_RUNTIME_LOSS_UV_PER_HOUR = 10.17
FC_HIGH_RUNTIME_LOSS_UV_PER_HOUR = 11.74
FC_TRANSIENT_LOSS_UV_PER_DELTA_KW = 0.0441
FC_START_STOP_LOSS_UV_PER_CYCLE = 23.91
FC_HIGH_LOAD_FRACTION = 0.8
FC_SOURCE_POWER_BASIS = "source-compatible reference-unit power"
FC_SINGLE_CELL_VOLTAGE_BASIS = "single-cell voltage"

FC_EOL_VOLTAGE_LOSS_UV = 70_000.0
FC_AGGREGATE_REPLACEMENT_COST_CNY = 3_500.0 * 600.0
FC_LIFETIME_NORMALIZATION_STATUS = "VERIFIED"
FC_LIFETIME_EVIDENCE_CLASS = "literature/model verified; not vessel-measured"
FC_AGGREGATE_POWER_MAPPING_STATUS = "NO-GO"


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


def _validated_voltage_loss_components(
    low_runtime_uv: object,
    high_runtime_uv: object,
    transient_uv: object,
    start_stop_uv: object,
) -> tuple[float, float, float, float]:
    components = (
        _exact_nonnegative_float(low_runtime_uv, "low_runtime_uv"),
        _exact_nonnegative_float(high_runtime_uv, "high_runtime_uv"),
        _exact_nonnegative_float(transient_uv, "transient_uv"),
        _exact_nonnegative_float(start_stop_uv, "start_stop_uv"),
    )
    runtime_uv = components[0] + components[1]
    if not math.isfinite(runtime_uv):
        raise ValueError("combined runtime voltage loss must remain finite")
    total_uv = runtime_uv + components[2] + components[3]
    if not math.isfinite(total_uv):
        raise ValueError("total voltage loss must remain finite")
    return components


@dataclass(frozen=True)
class FuelCellVoltageLoss:
    """Raw one-step voltage-loss components, all in microvolts."""

    low_runtime_uv: float
    high_runtime_uv: float
    transient_uv: float
    start_stop_uv: float

    def __post_init__(self) -> None:
        _validated_voltage_loss_components(
            self.low_runtime_uv,
            self.high_runtime_uv,
            self.transient_uv,
            self.start_stop_uv,
        )

    @property
    def runtime_uv(self) -> float:
        return self.low_runtime_uv + self.high_runtime_uv

    @property
    def total_uv(self) -> float:
        return self.runtime_uv + self.transient_uv + self.start_stop_uv


def reference_unit_voltage_loss_step_uv(
    previous_reference_power_kw: float,
    reference_power_kw: float,
    dt_seconds: float,
    reference_rated_power_kw: float,
    *,
    is_on: bool,
    aggregate_start_stop_cycles: int = 0,
) -> FuelCellVoltageLoss:
    """Return per-cell voltage loss from a source-compatible reference trace.

    This low-level equation does not accept an aggregate plant power trace.
    """

    previous = _strict_scalar(
        previous_reference_power_kw, "previous_reference_power_kw"
    )
    power = _strict_scalar(reference_power_kw, "reference_power_kw")
    duration = _strict_scalar(dt_seconds, "dt_seconds")
    rated = _strict_scalar(reference_rated_power_kw, "reference_rated_power_kw")
    if type(is_on) is not bool:
        raise TypeError("is_on must be an exact bool")
    if isinstance(aggregate_start_stop_cycles, bool) or not isinstance(
        aggregate_start_stop_cycles, Integral
    ):
        raise TypeError("aggregate_start_stop_cycles must be a non-negative integer")
    cycles = int(aggregate_start_stop_cycles)

    if rated <= 0.0:
        raise ValueError("reference_rated_power_kw must be positive")
    if previous < 0.0 or previous > rated:
        raise ValueError(
            "previous_reference_power_kw must lie in [0, reference_rated_power_kw]"
        )
    if power < 0.0 or power > rated:
        raise ValueError(
            "reference_power_kw must lie in [0, reference_rated_power_kw]"
        )
    if duration <= 0.0:
        raise ValueError("dt_seconds must be positive")
    if cycles < 0:
        raise ValueError("aggregate_start_stop_cycles must be non-negative")

    low_runtime_uv = 0.0
    high_runtime_uv = 0.0
    if is_on:
        runtime_hours = duration / 3600.0
        if power >= FC_HIGH_LOAD_FRACTION * rated:
            high_runtime_uv = FC_HIGH_RUNTIME_LOSS_UV_PER_HOUR * runtime_hours
        else:
            low_runtime_uv = FC_LOW_RUNTIME_LOSS_UV_PER_HOUR * runtime_hours

    return FuelCellVoltageLoss(
        low_runtime_uv=low_runtime_uv,
        high_runtime_uv=high_runtime_uv,
        transient_uv=FC_TRANSIENT_LOSS_UV_PER_DELTA_KW * abs(power - previous),
        start_stop_uv=FC_START_STOP_LOSS_UV_PER_CYCLE * cycles,
    )


@dataclass
class FuelCellVoltageLossAccount:
    """Cumulative raw accounting; this class deliberately has no life fraction."""

    low_runtime_uv: float = 0.0
    high_runtime_uv: float = 0.0
    transient_uv: float = 0.0
    start_stop_uv: float = 0.0

    def __post_init__(self) -> None:
        self._validated_components()

    def _validated_components(self) -> tuple[float, float, float, float]:
        return _validated_voltage_loss_components(
            self.low_runtime_uv,
            self.high_runtime_uv,
            self.transient_uv,
            self.start_stop_uv,
        )

    def add(self, step: FuelCellVoltageLoss) -> None:
        if type(step) is not FuelCellVoltageLoss:
            raise TypeError("step must be an exact FuelCellVoltageLoss")
        current = self._validated_components()
        additions = _validated_voltage_loss_components(
            step.low_runtime_uv, step.high_runtime_uv, step.transient_uv, step.start_stop_uv
        )
        updated = _validated_voltage_loss_components(
            *(left + right for left, right in zip(current, additions))
        )
        (
            self.low_runtime_uv,
            self.high_runtime_uv,
            self.transient_uv,
            self.start_stop_uv,
        ) = updated

    @property
    def runtime_uv(self) -> float:
        return self.low_runtime_uv + self.high_runtime_uv

    @property
    def total_uv(self) -> float:
        return self.runtime_uv + self.transient_uv + self.start_stop_uv


@dataclass(frozen=True)
class FuelCellLifeState:
    """Cumulative aggregate-equivalent life state at one interval boundary."""

    cumulative_voltage_loss_uv: float
    raw_life_fraction: float
    economic_life_fraction: float
    eol_reached: bool

    def __post_init__(self) -> None:
        _exact_nonnegative_float(
            self.cumulative_voltage_loss_uv,
            "cumulative_voltage_loss_uv",
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
class FuelCellLifeIncrement:
    """Non-duplicating economic life consumed by one physical interval."""

    before: FuelCellLifeState
    after: FuelCellLifeState
    delta_economic_fraction: float

    def __post_init__(self) -> None:
        if type(self.before) is not FuelCellLifeState:
            raise TypeError("before must be an exact FuelCellLifeState")
        if type(self.after) is not FuelCellLifeState:
            raise TypeError("after must be an exact FuelCellLifeState")
        delta = _exact_nonnegative_float(
            self.delta_economic_fraction,
            "delta_economic_fraction",
        )
        if delta > 1.0:
            raise ValueError("delta_economic_fraction must lie in [0, 1]")


def fuel_cell_life_state(
    cumulative_voltage_loss_uv: float,
) -> FuelCellLifeState:
    """Return raw, clipped-economic, and EOL cumulative diagnostics."""

    loss = _strict_scalar(
        cumulative_voltage_loss_uv,
        "cumulative_voltage_loss_uv",
    )
    if loss < 0.0:
        raise ValueError("cumulative_voltage_loss_uv must be non-negative")
    raw = loss / FC_EOL_VOLTAGE_LOSS_UV
    economic = min(raw, 1.0)
    return FuelCellLifeState(
        cumulative_voltage_loss_uv=loss,
        raw_life_fraction=raw,
        economic_life_fraction=economic,
        eol_reached=raw >= 1.0,
    )


def formal_fuel_cell_interval_life_loss(
    cumulative_voltage_loss_before_uv: float,
    cumulative_voltage_loss_after_uv: float,
) -> FuelCellLifeIncrement:
    """Return the clipped cumulative-life difference for one interval."""

    before = fuel_cell_life_state(cumulative_voltage_loss_before_uv)
    after = fuel_cell_life_state(cumulative_voltage_loss_after_uv)
    if after.cumulative_voltage_loss_uv < before.cumulative_voltage_loss_uv:
        raise ValueError("cumulative fuel-cell voltage loss must not decrease")
    return FuelCellLifeIncrement(
        before=before,
        after=after,
        delta_economic_fraction=(
            after.economic_life_fraction - before.economic_life_fraction
        ),
    )


@dataclass(frozen=True)
class AggregateFcStateTransition:
    is_on: bool
    starts: int
    stops: int
    cumulative_starts: int
    cumulative_stops: int


class AggregateFcOnOffTracker:
    """Aggregate hysteresis/dwell proxy; it does not detect individual stacks."""

    def __init__(
        self,
        p_off_threshold_kw: float,
        p_on_threshold_kw: float,
        minimum_dwell_seconds: float,
        rated_power_kw: float,
        *,
        initially_on: bool = False,
    ) -> None:
        p_off = _strict_scalar(p_off_threshold_kw, "p_off_threshold_kw")
        p_on = _strict_scalar(p_on_threshold_kw, "p_on_threshold_kw")
        dwell = _strict_scalar(minimum_dwell_seconds, "minimum_dwell_seconds")
        rated = _strict_scalar(rated_power_kw, "rated_power_kw")
        if type(initially_on) is not bool:
            raise TypeError("initially_on must be an exact bool")
        if rated <= 0.0:
            raise ValueError("rated_power_kw must be positive")
        if p_off < 0.0 or not p_off < p_on:
            raise ValueError("thresholds must satisfy 0 <= p_off_threshold_kw < p_on_threshold_kw")
        if p_on > rated:
            raise ValueError("p_on_threshold_kw must not exceed rated_power_kw")
        if dwell <= 0.0:
            raise ValueError("minimum_dwell_seconds must be positive")

        self.p_off_threshold_kw = p_off
        self.p_on_threshold_kw = p_on
        self.minimum_dwell_seconds = dwell
        self.rated_power_kw = rated
        self.is_on = initially_on
        self.cumulative_starts = 0
        self.cumulative_stops = 0
        self._pending_seconds = 0.0

    def update(self, power_kw: float, dt_seconds: float) -> AggregateFcStateTransition:
        power = _strict_scalar(power_kw, "power_kw")
        duration = _strict_scalar(dt_seconds, "dt_seconds")
        if power < 0.0 or power > self.rated_power_kw:
            raise ValueError("power_kw must lie in [0, rated_power_kw]")
        if duration <= 0.0:
            raise ValueError("dt_seconds must be positive")

        starts = 0
        stops = 0
        if self.is_on:
            if power <= self.p_off_threshold_kw:
                self._pending_seconds += duration
                if self._pending_seconds >= self.minimum_dwell_seconds:
                    self.is_on = False
                    self._pending_seconds = 0.0
                    self.cumulative_stops += 1
                    stops = 1
            else:
                self._pending_seconds = 0.0
        else:
            if power >= self.p_on_threshold_kw:
                self._pending_seconds += duration
                if self._pending_seconds >= self.minimum_dwell_seconds:
                    self.is_on = True
                    self._pending_seconds = 0.0
                    self.cumulative_starts += 1
                    starts = 1
            else:
                self._pending_seconds = 0.0

        return AggregateFcStateTransition(
            is_on=self.is_on,
            starts=starts,
            stops=stops,
            cumulative_starts=self.cumulative_starts,
            cumulative_stops=self.cumulative_stops,
        )


@dataclass(frozen=True)
class AggregateFcPowerMapping:
    """Proposed aggregate-to-reference mapping; none is verified."""

    aggregate_to_reference_power_ratio: float
    source_doi: str
    applicability: str

    def __post_init__(self) -> None:
        if type(self.aggregate_to_reference_power_ratio) is not float:
            raise TypeError("aggregate_to_reference_power_ratio must be an exact float")
        if (
            not math.isfinite(self.aggregate_to_reference_power_ratio)
            or self.aggregate_to_reference_power_ratio <= 0.0
        ):
            raise ValueError(
                "aggregate_to_reference_power_ratio must be finite and positive"
            )
        _strict_text(self.source_doi, "source_doi")
        _strict_text(self.applicability, "applicability")

    def require_verified(self) -> AggregateFcPowerMapping:
        raise ValueError(
            "formal aggregate fuel-cell degradation is NO-GO: no authoritative "
            "aggregate-to-reference-unit power mapping is available"
        )


def formal_aggregate_fc_voltage_loss_step_uv(
    previous_aggregate_power_kw: float,
    aggregate_power_kw: float,
    dt_seconds: float,
    aggregate_rated_power_kw: float,
    *,
    is_on: bool,
    mapping: AggregateFcPowerMapping,
    aggregate_start_stop_cycles: int = 0,
) -> FuelCellVoltageLoss:
    """Fail closed before applying per-reference-unit coefficients to aggregate power."""

    previous = _strict_scalar(
        previous_aggregate_power_kw, "previous_aggregate_power_kw"
    )
    power = _strict_scalar(aggregate_power_kw, "aggregate_power_kw")
    duration = _strict_scalar(dt_seconds, "dt_seconds")
    rated = _strict_scalar(aggregate_rated_power_kw, "aggregate_rated_power_kw")
    if type(is_on) is not bool:
        raise TypeError("is_on must be an exact bool")
    if isinstance(aggregate_start_stop_cycles, bool) or not isinstance(
        aggregate_start_stop_cycles, Integral
    ):
        raise TypeError("aggregate_start_stop_cycles must be a non-negative integer")
    if rated <= 0.0:
        raise ValueError("aggregate_rated_power_kw must be positive")
    if previous < 0.0 or previous > rated:
        raise ValueError(
            "previous_aggregate_power_kw must lie in [0, aggregate_rated_power_kw]"
        )
    if power < 0.0 or power > rated:
        raise ValueError(
            "aggregate_power_kw must lie in [0, aggregate_rated_power_kw]"
        )
    if duration <= 0.0:
        raise ValueError("dt_seconds must be positive")
    if int(aggregate_start_stop_cycles) < 0:
        raise ValueError("aggregate_start_stop_cycles must be non-negative")
    if type(mapping) is not AggregateFcPowerMapping:
        raise TypeError("mapping must use the exact provenance-bearing type")
    AggregateFcPowerMapping.require_verified(mapping)
    raise AssertionError("unreachable until a formal mapping is authorized")


@dataclass(frozen=True)
class FuelCellLifetimeNormalization:
    """Proposed calibration record; no instance is currently formally verified."""

    v_init_v: float
    voltage_basis: str
    source_doi: str
    applicability: str

    def __post_init__(self) -> None:
        if type(self.v_init_v) is not float:
            raise TypeError("formal v_init_v must be an exact float")
        if not math.isfinite(self.v_init_v) or self.v_init_v <= 0.0:
            raise ValueError("v_init_v must be finite and positive")
        _strict_text(self.voltage_basis, "voltage_basis")
        if self.voltage_basis != FC_SINGLE_CELL_VOLTAGE_BASIS:
            raise ValueError("voltage_basis must be exactly 'single-cell voltage'")
        _strict_text(self.source_doi, "source_doi")
        _strict_text(self.applicability, "applicability")

    def require_verified(self) -> FuelCellLifetimeNormalization:
        raise ValueError(
            "formal fuel-cell lifetime normalization is NO-GO: no applicable "
            "numeric single-cell V_init has authoritative calibration"
        )


def fuel_cell_relative_life_loss_unverified(
    delta_v_uv: float,
    *,
    v_init_v: float,
    voltage_basis: str,
) -> float:
    """Evaluate D_fc = DeltaV/(0.1 V_init) for explicitly synthetic use only."""

    loss = _strict_scalar(delta_v_uv, "delta_v_uv")
    initial_voltage = _strict_scalar(v_init_v, "v_init_v")
    _strict_text(voltage_basis, "voltage_basis")
    if loss < 0.0:
        raise ValueError("delta_v_uv must be non-negative")
    if initial_voltage <= 0.0:
        raise ValueError("v_init_v must be positive")
    if voltage_basis != FC_SINGLE_CELL_VOLTAGE_BASIS:
        raise ValueError("voltage_basis must be exactly 'single-cell voltage'")
    return loss * 1.0e-6 / (0.1 * initial_voltage)


def formal_fuel_cell_relative_life_loss(
    delta_v_uv: float,
    *,
    normalization: FuelCellLifetimeNormalization,
) -> float:
    """Fail closed until a provenance-bearing V_init basis is authorized."""

    loss = _strict_scalar(delta_v_uv, "delta_v_uv")
    if loss < 0.0:
        raise ValueError("delta_v_uv must be non-negative")
    if type(normalization) is not FuelCellLifetimeNormalization:
        raise TypeError("formal normalization must use the exact provenance-bearing type")
    FuelCellLifetimeNormalization.require_verified(normalization)
    raise AssertionError("unreachable until a formal calibration is authorized")


def formal_fuel_cell_degradation_cost_cny(
    cumulative_voltage_loss_before_uv: float,
    cumulative_voltage_loss_after_uv: float,
    *,
    replacement_cost_cny: float,
) -> float:
    """Charge only newly consumed clipped life in the current interval."""

    increment = formal_fuel_cell_interval_life_loss(
        cumulative_voltage_loss_before_uv,
        cumulative_voltage_loss_after_uv,
    )
    price = _strict_scalar(replacement_cost_cny, "replacement_cost_cny")
    if price < 0.0:
        raise ValueError("replacement_cost_cny must be non-negative")
    if price != FC_AGGREGATE_REPLACEMENT_COST_CNY:
        raise ValueError(
            "replacement_cost_cny must equal 3500 CNY/kW * 600 kW"
        )
    result = increment.delta_economic_fraction * price
    if not math.isfinite(result):
        raise ValueError("fuel-cell interval degradation cost must remain finite")
    return result
