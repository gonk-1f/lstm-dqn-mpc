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

FC_LIFETIME_NORMALIZATION_STATUS = "NO-GO"


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


@dataclass(frozen=True)
class FuelCellVoltageLoss:
    """Raw one-step voltage-loss components, all in microvolts."""

    low_runtime_uv: float
    high_runtime_uv: float
    transient_uv: float
    start_stop_uv: float

    @property
    def runtime_uv(self) -> float:
        return self.low_runtime_uv + self.high_runtime_uv

    @property
    def total_uv(self) -> float:
        return self.runtime_uv + self.transient_uv + self.start_stop_uv


def fc_voltage_loss_step_uv(
    previous_power_kw: float,
    power_kw: float,
    dt_seconds: float,
    rated_power_kw: float,
    *,
    is_on: bool,
    aggregate_start_stop_cycles: int = 0,
) -> FuelCellVoltageLoss:
    """Return raw aggregate voltage loss for one executed-power interval."""

    previous = _strict_scalar(previous_power_kw, "previous_power_kw")
    power = _strict_scalar(power_kw, "power_kw")
    duration = _strict_scalar(dt_seconds, "dt_seconds")
    rated = _strict_scalar(rated_power_kw, "rated_power_kw")
    if type(is_on) is not bool:
        raise TypeError("is_on must be an exact bool")
    if isinstance(aggregate_start_stop_cycles, bool) or not isinstance(
        aggregate_start_stop_cycles, Integral
    ):
        raise TypeError("aggregate_start_stop_cycles must be a non-negative integer")
    cycles = int(aggregate_start_stop_cycles)

    if rated <= 0.0:
        raise ValueError("rated_power_kw must be positive")
    if previous < 0.0 or previous > rated:
        raise ValueError("previous_power_kw must lie in [0, rated_power_kw]")
    if power < 0.0 or power > rated:
        raise ValueError("power_kw must lie in [0, rated_power_kw]")
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

    def add(self, step: FuelCellVoltageLoss) -> None:
        if type(step) is not FuelCellVoltageLoss:
            raise TypeError("step must be an exact FuelCellVoltageLoss")
        components = (
            step.low_runtime_uv,
            step.high_runtime_uv,
            step.transient_uv,
            step.start_stop_uv,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in components):
            raise ValueError("voltage-loss components must be finite and non-negative")
        self.low_runtime_uv += step.low_runtime_uv
        self.high_runtime_uv += step.high_runtime_uv
        self.transient_uv += step.transient_uv
        self.start_stop_uv += step.start_stop_uv

    @property
    def runtime_uv(self) -> float:
        return self.low_runtime_uv + self.high_runtime_uv

    @property
    def total_uv(self) -> float:
        return self.runtime_uv + self.transient_uv + self.start_stop_uv


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
        _strict_text(self.source_doi, "source_doi")
        _strict_text(self.applicability, "applicability")

    def require_verified(self) -> FuelCellLifetimeNormalization:
        raise ValueError(
            "formal fuel-cell lifetime normalization is NO-GO: V_init and its "
            "cell/system voltage basis have no authoritative calibration"
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
    delta_v_uv: float,
    *,
    replacement_cost_cny: float,
    normalization: FuelCellLifetimeNormalization,
) -> float:
    """Fail closed before raw microvolts can be multiplied by equipment price."""

    relative_loss = formal_fuel_cell_relative_life_loss(
        delta_v_uv, normalization=normalization
    )
    price = _strict_scalar(replacement_cost_cny, "replacement_cost_cny")
    if price < 0.0:
        raise ValueError("replacement_cost_cny must be non-negative")
    return relative_loss * price
