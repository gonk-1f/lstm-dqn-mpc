"""Three-component constrained nonlinear receding-horizon controller."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import minimize

from ..config import TimeScaleConfig
from ..contracts import MPC_OBJECTIVE_VERSION
from ..models.battery_energy import BatteryEfficiency, next_soc
from .causal_base_load import CausalBaseLoadFilter


def _finite_scalar(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real numeric scalar, not bool or text")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _finite_vector(values: object, name: str) -> tuple[float, ...]:
    if isinstance(values, np.ndarray):
        if values.ndim != 1:
            raise TypeError(f"{name} must be one-dimensional")
        source = values
    elif isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a numeric sequence")
    else:
        source = values
    return tuple(_finite_scalar(value, f"{name}[{index}]") for index, value in enumerate(source))


def _integral(value: object, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integral non-bool value")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


@dataclass(frozen=True)
class ObjectiveWeights:
    q_base: float
    q_smooth: float
    q_soc: float

    def __post_init__(self) -> None:
        values = tuple(
            _finite_scalar(getattr(self, name), name)
            for name in ("q_base", "q_smooth", "q_soc")
        )
        if any(value <= 0.0 for value in values):
            raise ValueError("all objective weights must be positive")
        if not math.isclose(sum(values), 1.0, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("objective weights must sum to one")
        for name, value in zip(("q_base", "q_smooth", "q_soc"), values):
            object.__setattr__(self, name, value)


MPCWeights = ObjectiveWeights


@dataclass(frozen=True)
class ObjectiveComponents:
    j_base: float
    j_smooth: float
    j_soc: float

    def __post_init__(self) -> None:
        for name in ("j_base", "j_smooth", "j_soc"):
            value = _finite_scalar(getattr(self, name), name)
            if value < 0.0:
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(self, name, value)


def _validate_deadband(low: float, high: float, scale: float) -> tuple[float, float, float]:
    low_value = _finite_scalar(low, "soc_deadband_low")
    high_value = _finite_scalar(high, "soc_deadband_high")
    scale_value = _finite_scalar(scale, "soc_scale")
    if not 0.0 <= low_value < high_value <= 1.0:
        raise ValueError("SOC deadband must satisfy 0 <= low < high <= 1")
    if scale_value <= 0.0:
        raise ValueError("soc_scale must be positive")
    return low_value, high_value, scale_value


def soc_deadband_penalty(soc: float, low: float, high: float, scale: float) -> float:
    """Return the squared, normalized distance outside an inclusive deadband."""

    state = _finite_scalar(soc, "soc")
    low_value, high_value, scale_value = _validate_deadband(low, high, scale)
    if state < low_value:
        return ((low_value - state) / scale_value) ** 2
    if state > high_value:
        return ((state - high_value) / scale_value) ** 2
    return 0.0


def objective_components(
    *,
    p_fc_kw: Sequence[float],
    base_reference_kw: Sequence[float],
    soc_path: Sequence[float],
    previous_executed_p_fc_kw: float,
    p_fc_scale_kw: float,
    delta_p_fc_scale_kw: float,
    soc_deadband_low: float,
    soc_deadband_high: float,
    soc_scale: float,
) -> ObjectiveComponents:
    """Evaluate exactly the base, smoothness, and SOC components."""

    powers = _finite_vector(p_fc_kw, "p_fc_kw")
    references = _finite_vector(base_reference_kw, "base_reference_kw")
    states = _finite_vector(soc_path, "soc_path")
    if not powers or len(powers) != len(references) or len(powers) != len(states):
        raise ValueError("objective sequences must have the same nonzero length")
    previous = _finite_scalar(previous_executed_p_fc_kw, "previous_executed_p_fc_kw")
    power_scale = _finite_scalar(p_fc_scale_kw, "p_fc_scale_kw")
    ramp_scale = _finite_scalar(delta_p_fc_scale_kw, "delta_p_fc_scale_kw")
    low, high, state_scale = _validate_deadband(
        soc_deadband_low, soc_deadband_high, soc_scale
    )
    if power_scale <= 0.0 or ramp_scale <= 0.0:
        raise ValueError("power objective scales must be positive")

    j_base = sum(((power - reference) / power_scale) ** 2 for power, reference in zip(powers, references))
    prior = (previous,) + powers[:-1]
    j_smooth = sum(((power - old) / ramp_scale) ** 2 for power, old in zip(powers, prior))
    j_soc = sum(soc_deadband_penalty(state, low, high, state_scale) for state in states)
    return ObjectiveComponents(float(j_base), float(j_smooth), float(j_soc))


def weighted_objective(components: ObjectiveComponents, weights: ObjectiveWeights) -> float:
    if type(components) is not ObjectiveComponents:
        raise TypeError("components must be an exact ObjectiveComponents instance")
    if type(weights) is not ObjectiveWeights:
        raise TypeError("weights must be an exact ObjectiveWeights instance")
    return float(
        weights.q_base * components.j_base
        + weights.q_smooth * components.j_smooth
        + weights.q_soc * components.j_soc
    )


@dataclass(frozen=True)
class MPCConfig:
    timescale: TimeScaleConfig
    fuel_cell_rated_kw: float
    battery_capacity_kwh: float
    battery_efficiency: BatteryEfficiency
    battery_charge_min_kw: float
    battery_discharge_max_kw: float
    fuel_cell_ramp_kw_per_step: float
    soc_min: float
    soc_max: float
    soc_deadband_low: float
    soc_deadband_high: float
    p_fc_scale_kw: float
    delta_p_fc_scale_kw: float
    soc_scale: float

    def __post_init__(self) -> None:
        if type(self.timescale) is not TimeScaleConfig:
            raise TypeError("timescale must be an exact TimeScaleConfig instance")
        if type(self.battery_efficiency) is not BatteryEfficiency:
            raise TypeError("battery_efficiency must be an exact BatteryEfficiency instance")
        self.battery_efficiency.require_calibrated()

        numeric_names = (
            "fuel_cell_rated_kw",
            "battery_capacity_kwh",
            "battery_charge_min_kw",
            "battery_discharge_max_kw",
            "fuel_cell_ramp_kw_per_step",
            "soc_min",
            "soc_max",
            "soc_deadband_low",
            "soc_deadband_high",
            "p_fc_scale_kw",
            "delta_p_fc_scale_kw",
            "soc_scale",
        )
        for name in numeric_names:
            object.__setattr__(self, name, _finite_scalar(getattr(self, name), name))
        for name in (
            "fuel_cell_rated_kw",
            "battery_capacity_kwh",
            "battery_discharge_max_kw",
            "fuel_cell_ramp_kw_per_step",
            "p_fc_scale_kw",
            "delta_p_fc_scale_kw",
            "soc_scale",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.battery_charge_min_kw >= 0.0:
            raise ValueError("battery_charge_min_kw must be negative")
        if not 0.0 <= self.soc_min < self.soc_max <= 1.0:
            raise ValueError("SOC hard bounds must satisfy 0 <= min < max <= 1")
        low, high, _ = _validate_deadband(
            self.soc_deadband_low, self.soc_deadband_high, self.soc_scale
        )
        if low < self.soc_min or high > self.soc_max:
            raise ValueError("SOC deadband must lie inside the SOC hard bounds")


@dataclass(frozen=True)
class SolverDiagnostics:
    success: bool
    status: int
    message: str
    iterations: int
    objective_value: float

    def __post_init__(self) -> None:
        if not isinstance(self.success, (bool, np.bool_)):
            raise TypeError("success must be an exact bool or numpy bool")
        status = _integral(self.status, "status")
        iterations = _integral(self.iterations, "iterations", minimum=0)
        if type(self.message) is not str:
            raise TypeError("message must be an exact string")
        objective_value = _finite_scalar(self.objective_value, "objective_value")
        if objective_value < 0.0:
            raise ValueError("objective_value must be nonnegative")
        object.__setattr__(self, "success", bool(self.success))
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "iterations", iterations)
        object.__setattr__(self, "objective_value", objective_value)


@dataclass(frozen=True)
class MPCCommand:
    p_fc_kw: float
    p_batt_bus_kw: float
    predicted_next_soc: float

    def __post_init__(self) -> None:
        for name in ("p_fc_kw", "p_batt_bus_kw", "predicted_next_soc"):
            object.__setattr__(
                self,
                name,
                _finite_scalar(getattr(self, name), name),
            )


@dataclass(frozen=True)
class MPCPlan:
    p_fc_kw: tuple[float, ...]
    p_batt_bus_kw: tuple[float, ...]
    soc_path: tuple[float, ...]
    load_forecast_kw: tuple[float, ...]
    base_reference_kw: tuple[float, ...]
    components: ObjectiveComponents
    objective_value: float
    diagnostics: SolverDiagnostics

    def __post_init__(self) -> None:
        vector_names = (
            "p_fc_kw",
            "p_batt_bus_kw",
            "soc_path",
            "load_forecast_kw",
            "base_reference_kw",
        )
        vectors = tuple(
            _finite_vector(getattr(self, name), name)
            for name in vector_names
        )
        if not vectors[0] or any(len(vector) != len(vectors[0]) for vector in vectors[1:]):
            raise ValueError("MPC plan vectors must have the same nonzero length")
        if type(self.components) is not ObjectiveComponents:
            raise TypeError("components must be an exact ObjectiveComponents instance")
        if type(self.diagnostics) is not SolverDiagnostics:
            raise TypeError("diagnostics must be an exact SolverDiagnostics instance")
        if not self.diagnostics.success:
            raise ValueError("a returned MPC plan requires successful solver diagnostics")
        objective_value = _finite_scalar(self.objective_value, "objective_value")
        if objective_value < 0.0:
            raise ValueError("objective_value must be nonnegative")
        if objective_value != self.diagnostics.objective_value:
            raise ValueError("plan and diagnostic objective values must match")
        for name, vector in zip(vector_names, vectors):
            object.__setattr__(self, name, vector)
        object.__setattr__(self, "objective_value", objective_value)

    def first_command(self) -> MPCCommand:
        """Expose only the command that a receding-horizon loop may execute."""

        return MPCCommand(
            p_fc_kw=self.p_fc_kw[0],
            p_batt_bus_kw=self.p_batt_bus_kw[0],
            predicted_next_soc=self.soc_path[0],
        )


class MPCSolveError(RuntimeError):
    kind = "mpc_solve_error"

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class PhysicalInfeasibilityError(MPCSolveError):
    kind = "physical_infeasibility"


class NumericalSolverError(MPCSolveError):
    kind = "numerical_solver_failure"


def _soc_path(
    initial_soc: float,
    battery_powers: Sequence[float],
    config: MPCConfig,
) -> tuple[float, ...]:
    state = initial_soc
    states: list[float] = []
    for power in battery_powers:
        state = next_soc(
            state,
            power,
            config.timescale.ts_mpc_seconds,
            config.battery_capacity_kwh,
            efficiency=config.battery_efficiency,
        )
        states.append(state)
    return tuple(states)


def shifted_warm_start(previous_plan: MPCPlan) -> tuple[float, ...]:
    """Explicitly shift a prior plan without retaining controller state."""

    if type(previous_plan) is not MPCPlan:
        raise TypeError("previous_plan must be an exact MPCPlan instance")
    return previous_plan.p_fc_kw[1:] + (previous_plan.p_fc_kw[-1],)


Optimizer = Callable[..., object]


class NonlinearMPC:
    """Deterministic SLSQP MPC with explicit cold or caller-supplied warm start."""

    _PHYSICAL_TOLERANCE = 1e-6

    def __init__(self, config: MPCConfig, *, optimizer: Optimizer = minimize) -> None:
        if type(config) is not MPCConfig:
            raise TypeError("config must be an exact MPCConfig instance")
        if not callable(optimizer):
            raise TypeError("optimizer must be callable")
        self.config = config
        self._optimizer = optimizer

    def _validate_initial_state(self, current_soc: float, previous_fc: float) -> None:
        cfg = self.config
        if not cfg.soc_min <= current_soc <= cfg.soc_max:
            raise PhysicalInfeasibilityError("current SOC is outside the configured hard bounds")
        if not 0.0 <= previous_fc <= cfg.fuel_cell_rated_kw:
            raise PhysicalInfeasibilityError("previous executed fuel-cell power is outside its bounds")

    def _reachable_intervals(
        self,
        loads: Sequence[float],
        current_soc: float,
        previous_fc: float,
    ) -> tuple[tuple[float, float], ...]:
        cfg = self.config
        prior_low = previous_fc
        prior_high = previous_fc
        soc_low = current_soc
        soc_high = current_soc
        intervals: list[tuple[float, float]] = []
        for step, load in enumerate(loads):
            low = max(
                0.0,
                load - cfg.battery_discharge_max_kw,
                prior_low - cfg.fuel_cell_ramp_kw_per_step,
            )
            high = min(
                cfg.fuel_cell_rated_kw,
                load - cfg.battery_charge_min_kw,
                prior_high + cfg.fuel_cell_ramp_kw_per_step,
            )
            if low > high + self._PHYSICAL_TOLERANCE:
                raise PhysicalInfeasibilityError(
                    f"no power satisfying fuel-cell, battery, and ramp bounds at horizon step {step}"
                )
            maximum_battery = load - low
            minimum_battery = load - high
            next_low = next_soc(
                soc_low,
                maximum_battery,
                cfg.timescale.ts_mpc_seconds,
                cfg.battery_capacity_kwh,
                efficiency=cfg.battery_efficiency,
            )
            next_high = next_soc(
                soc_high,
                minimum_battery,
                cfg.timescale.ts_mpc_seconds,
                cfg.battery_capacity_kwh,
                efficiency=cfg.battery_efficiency,
            )
            if next_high < cfg.soc_min - self._PHYSICAL_TOLERANCE or next_low > cfg.soc_max + self._PHYSICAL_TOLERANCE:
                raise PhysicalInfeasibilityError(
                    f"no SOC satisfying hard bounds at horizon step {step}"
                )
            soc_low = max(next_low, cfg.soc_min)
            soc_high = min(next_high, cfg.soc_max)
            intervals.append((low, high))
            prior_low, prior_high = low, high
        return tuple(intervals)

    def _cold_start(
        self,
        references: Sequence[float],
        intervals: Sequence[tuple[float, float]],
        previous_fc: float,
    ) -> np.ndarray:
        cfg = self.config
        values: list[float] = []
        prior = previous_fc
        for reference, (reachable_low, reachable_high) in zip(references, intervals):
            low = max(reachable_low, prior - cfg.fuel_cell_ramp_kw_per_step)
            high = min(reachable_high, prior + cfg.fuel_cell_ramp_kw_per_step)
            value = min(max(reference, low), high)
            values.append(float(value))
            prior = value
        return np.asarray(values, dtype=float)

    def _postsolve_check(
        self,
        powers: np.ndarray,
        loads: np.ndarray,
        states: np.ndarray,
        previous_fc: float,
        solver_status: int,
    ) -> None:
        cfg = self.config
        tolerance = self._PHYSICAL_TOLERANCE
        if powers.shape != (cfg.timescale.n_mpc,) or not np.all(np.isfinite(powers)):
            raise NumericalSolverError(
                "solver returned a malformed or nonfinite decision vector",
                status=solver_status,
            )
        batteries = loads - powers
        deltas = np.diff(np.concatenate(([previous_fc], powers)))
        residuals = (
            np.min(powers),
            np.min(cfg.fuel_cell_rated_kw - powers),
            np.min(batteries - cfg.battery_charge_min_kw),
            np.min(cfg.battery_discharge_max_kw - batteries),
            np.min(cfg.fuel_cell_ramp_kw_per_step - np.abs(deltas)),
            np.min(states - cfg.soc_min),
            np.min(cfg.soc_max - states),
        )
        if not np.all(np.isfinite(states)) or min(residuals) < -tolerance:
            raise NumericalSolverError(
                "solver success output failed independent physical residual checks",
                status=solver_status,
            )

    def solve(
        self,
        observed_load_kw: float,
        current_soc: float,
        previous_executed_p_fc_kw: float,
        weights: ObjectiveWeights,
        base_load_filter: CausalBaseLoadFilter,
        *,
        warm_start: Sequence[float] | None = None,
    ) -> MPCPlan:
        cfg = self.config
        load = _finite_scalar(observed_load_kw, "observed_load_kw")
        state = _finite_scalar(current_soc, "current_soc")
        previous = _finite_scalar(
            previous_executed_p_fc_kw, "previous_executed_p_fc_kw"
        )
        if load < 0.0:
            raise ValueError("observed_load_kw must be nonnegative")
        if type(weights) is not ObjectiveWeights:
            raise TypeError("weights must be an exact ObjectiveWeights instance")
        if type(base_load_filter) is not CausalBaseLoadFilter:
            raise TypeError("base_load_filter must be an exact CausalBaseLoadFilter instance")
        if not math.isclose(
            base_load_filter.sample_seconds,
            cfg.timescale.ts_mpc_seconds,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError("base-load filter sample time must equal the MPC sample time")
        self._validate_initial_state(state, previous)

        warm: tuple[float, ...] | None = None
        if warm_start is not None:
            warm = _finite_vector(warm_start, "warm_start")
            if len(warm) != cfg.timescale.n_mpc:
                raise ValueError("warm_start length must equal timescale.n_mpc")

        forecast = base_load_filter.preview(load, horizon=cfg.timescale.n_mpc)
        loads = np.asarray(forecast.load_kw, dtype=float)
        references = np.asarray(forecast.base_reference_kw, dtype=float)
        intervals = self._reachable_intervals(loads, state, previous)

        if warm is None:
            initial = self._cold_start(references, intervals, previous)
        else:
            initial = np.asarray(warm, dtype=float)

        def derive(decision: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            batteries = loads - decision
            states = np.asarray(_soc_path(state, batteries, cfg), dtype=float)
            return batteries, states

        def objective(decision: np.ndarray) -> float:
            _, states = derive(decision)
            components = objective_components(
                p_fc_kw=tuple(decision),
                base_reference_kw=forecast.base_reference_kw,
                soc_path=tuple(states),
                previous_executed_p_fc_kw=previous,
                p_fc_scale_kw=cfg.p_fc_scale_kw,
                delta_p_fc_scale_kw=cfg.delta_p_fc_scale_kw,
                soc_deadband_low=cfg.soc_deadband_low,
                soc_deadband_high=cfg.soc_deadband_high,
                soc_scale=cfg.soc_scale,
            )
            return weighted_objective(components, weights)

        def physical_inequalities(decision: np.ndarray) -> np.ndarray:
            batteries, states = derive(decision)
            deltas = np.diff(np.concatenate(([previous], decision)))
            return np.concatenate(
                (
                    batteries - cfg.battery_charge_min_kw,
                    cfg.battery_discharge_max_kw - batteries,
                    cfg.fuel_cell_ramp_kw_per_step - deltas,
                    cfg.fuel_cell_ramp_kw_per_step + deltas,
                    states - cfg.soc_min,
                    cfg.soc_max - states,
                )
            )

        try:
            raw = self._optimizer(
                objective,
                initial,
                method="SLSQP",
                bounds=[(0.0, cfg.fuel_cell_rated_kw)] * cfg.timescale.n_mpc,
                constraints=({"type": "ineq", "fun": physical_inequalities},),
                options={"ftol": 1e-10, "maxiter": 500, "disp": False},
            )
        except Exception as exc:
            raise NumericalSolverError(f"optimizer raised {type(exc).__name__}: {exc}") from exc
        status: int | None = None
        try:
            status = _integral(getattr(raw, "status"), "optimizer status")
            success_value = getattr(raw, "success")
            if not isinstance(success_value, (bool, np.bool_)):
                raise TypeError("optimizer success must be an exact bool or numpy bool")
            iterations = _integral(
                getattr(raw, "nit"),
                "optimizer iterations",
                minimum=0,
            )
            message_value = getattr(raw, "message")
            if type(message_value) is not str:
                raise TypeError("optimizer message must be an exact string")
            success = bool(success_value)
            message = message_value
        except Exception as exc:
            raise NumericalSolverError(
                f"optimizer returned malformed metadata: {type(exc).__name__}: {exc}",
                status=status,
            ) from exc
        if not success:
            raise NumericalSolverError(message, status=status)

        try:
            powers = np.asarray(getattr(raw, "x", ()), dtype=float)
            valid_shape = powers.shape == (cfg.timescale.n_mpc,)
            all_finite = bool(np.all(np.isfinite(powers)))
        except Exception as exc:
            raise NumericalSolverError(
                f"solver returned an unreadable decision vector: {type(exc).__name__}: {exc}",
                status=status,
            ) from exc
        if not valid_shape or not all_finite:
            raise NumericalSolverError(
                "solver returned a malformed or nonfinite decision vector",
                status=status,
            )
        try:
            batteries, states = derive(powers)
        except Exception as exc:
            raise NumericalSolverError(
                f"solver decision could not be evaluated: {type(exc).__name__}: {exc}",
                status=status,
            ) from exc
        self._postsolve_check(powers, loads, states, previous, status)
        components = objective_components(
            p_fc_kw=tuple(powers),
            base_reference_kw=forecast.base_reference_kw,
            soc_path=tuple(states),
            previous_executed_p_fc_kw=previous,
            p_fc_scale_kw=cfg.p_fc_scale_kw,
            delta_p_fc_scale_kw=cfg.delta_p_fc_scale_kw,
            soc_deadband_low=cfg.soc_deadband_low,
            soc_deadband_high=cfg.soc_deadband_high,
            soc_scale=cfg.soc_scale,
        )
        value = weighted_objective(components, weights)
        diagnostics = SolverDiagnostics(
            success=True,
            status=status,
            message=message,
            iterations=iterations,
            objective_value=value,
        )
        plan = MPCPlan(
            p_fc_kw=tuple(float(value) for value in powers),
            p_batt_bus_kw=tuple(float(value) for value in batteries),
            soc_path=tuple(float(value) for value in states),
            load_forecast_kw=forecast.load_kw,
            base_reference_kw=forecast.base_reference_kw,
            components=components,
            objective_value=value,
            diagnostics=diagnostics,
        )
        base_load_filter.commit(load)
        return plan
