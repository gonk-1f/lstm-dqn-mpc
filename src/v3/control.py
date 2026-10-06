"""Five-step economic MPC with separate predicted and executed interval ledgers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import minimize

from v2.config import PlantConfig
from v2.control.causal_base_load import CausalBaseLoadFilter
from v2.data.supervisory_rules import normalize_onboard_load_kw
from v2.economics import (
    RawCnyIntervalLedger, ShoreEnergy, ShoreEnergyClassification,
    build_formal_interval_ledger,
)
from v2.models.battery_degradation import (
    BATTERY_NOMINAL_VOLTAGE_V,
    battery_degradation_step,
    formal_battery_lifetime_normalization,
)
from v2.models.battery_energy import formal_battery_efficiency, next_soc
from v2.models.fuel_cell_degradation import (
    FC_AGGREGATE_REPLACEMENT_COST_CNY,
    FC_EOL_VOLTAGE_LOSS_UV,
    FC_LOW_RUNTIME_LOSS_UV_PER_HOUR,
    aggregate_start_cycle_count,
    formal_aggregate_fc_power_mapping,
    formal_aggregate_fc_voltage_loss_step_uv,
)
from v2.models.fuel_cell_efficiency import formal_fuel_cell_efficiency_map, hydrogen_mass_from_map_kg


DT_SECONDS = 30.0
HORIZON = 5
SOC_MIN = 0.2
SOC_MAX = 0.8
SHORE_TARGET_SOC = 0.6


def _vector(values: Sequence[float], name: str) -> tuple[float, ...]:
    result = tuple(float(v) for v in values)
    if len(result) != HORIZON or not np.isfinite(result).all() or min(result) < 0:
        raise ValueError(f"{name} must contain five finite nonnegative powers")
    return result


@dataclass(frozen=True)
class MPCWeights:
    lambda_ref: float
    lambda_soc: float

    def __post_init__(self) -> None:
        if not np.isfinite([self.lambda_ref, self.lambda_soc]).all() or min(self.lambda_ref, self.lambda_soc) < 0:
            raise ValueError("MPC weights must be finite and nonnegative")


@dataclass(frozen=True)
class MPCObjectiveTerms:
    economic_cost_cny: float
    economic_normalized: float
    reference_penalty: float
    soc_penalty: float

    def objective(self, weights: MPCWeights) -> float:
        return (
            self.economic_normalized
            + weights.lambda_ref * self.reference_penalty
            + weights.lambda_soc * self.soc_penalty
        ) / HORIZON


@dataclass(frozen=True)
class AccountState:
    soc: float = 0.6
    previous_fc_kw: float = 0.0
    fc_loss_uv: float = 0.0
    battery_weighted_ah: float = 0.0


@dataclass(frozen=True)
class Plan:
    forecast_kw: tuple[float, ...]
    reference_kw: tuple[float, ...]
    fc_power_kw: tuple[float, ...]
    battery_power_kw: tuple[float, ...]
    soc_path: tuple[float, ...]
    first_predicted_ledger: RawCnyIntervalLedger
    objective_value: float


@dataclass(frozen=True)
class ExecutedStep:
    actual_fc_power_kw: float
    actual_battery_power_kw: float
    actual_soc: float
    actual_ledger: RawCnyIntervalLedger
    prediction_error_kw: float

    @property
    def reward_cny(self) -> float:
        return self.actual_ledger.reward_cny


@dataclass(frozen=True)
class DecisionTransition:
    state: tuple[float, ...]
    action: MPCWeights
    reward_cny: float
    next_state: tuple[float, ...]
    executed: ExecutedStep
    done: bool = False
    shore_ledger: RawCnyIntervalLedger | None = None
    shore_requested_battery_bus_kw: tuple[float, ...] = ()
    shore_accepted_battery_bus_kw: tuple[float, ...] = ()
    shore_soc_path: tuple[float, ...] = ()


class EconomicMPC:
    """Reuses formal v2 physics and prices; nominal cost must come from Train."""

    def __init__(self, *, nominal_cost_cny: float):
        if not math.isfinite(nominal_cost_cny) or nominal_cost_cny <= 0:
            raise ValueError("nominal_cost_cny must be finite and positive")
        self.nominal_cost_cny = float(nominal_cost_cny)
        self.plant = PlantConfig.research_simulation()
        self.efficiency = formal_battery_efficiency()
        self.battery_normalization = formal_battery_lifetime_normalization()
        self.fc_efficiency = formal_fuel_cell_efficiency_map()
        self.fc_mapping = formal_aggregate_fc_power_mapping()

    def interval(self, state: AccountState, fc_kw: float, load_kw: float) -> tuple[AccountState, RawCnyIntervalLedger, float]:
        """One shared accounting function for prediction and actual execution."""
        fc = float(fc_kw)
        load = float(load_kw)
        if not np.isfinite([fc, load]).all() or not 0 <= fc <= self.plant.fuel_cell_rated_total_kw or load < 0:
            raise ValueError("FC and load must be finite and within physical domains")
        batt = load - fc
        soc = next_soc(state.soc, batt, DT_SECONDS, self.plant.battery_nominal_energy_kwh, efficiency=self.efficiency)
        fc_step = formal_aggregate_fc_voltage_loss_step_uv(
            state.previous_fc_kw, fc, DT_SECONDS, self.plant.fuel_cell_rated_total_kw,
            is_on=fc > 0.0, mapping=self.fc_mapping,
            aggregate_start_stop_cycles=aggregate_start_cycle_count(state.previous_fc_kw, fc),
        )
        battery_step = battery_degradation_step(
            state.soc, batt * 1000.0 / BATTERY_NOMINAL_VOLTAGE_V,
            DT_SECONDS, self.battery_normalization.nominal_charge_capacity_ah,
        )
        next_state = AccountState(
            soc=soc, previous_fc_kw=fc,
            fc_loss_uv=state.fc_loss_uv + fc_step.total_uv,
            battery_weighted_ah=state.battery_weighted_ah + battery_step.weighted_ah,
        )
        ledger = build_formal_interval_ledger(
            hydrogen_mass_kg=hydrogen_mass_from_map_kg(fc, DT_SECONDS, self.fc_efficiency),
            fuel_cell_cumulative_voltage_loss_before_uv=state.fc_loss_uv,
            fuel_cell_cumulative_voltage_loss_after_uv=next_state.fc_loss_uv,
            fuel_cell_rated_kw=self.plant.fuel_cell_rated_total_kw,
            battery_cumulative_weighted_ah_before=state.battery_weighted_ah,
            battery_cumulative_weighted_ah_after=next_state.battery_weighted_ah,
            battery_capacity_kwh=self.plant.battery_nominal_energy_kwh,
            battery_normalization=self.battery_normalization,
            shore_energy=None,
        )
        return next_state, ledger, batt

    def shore_interval(self, state: AccountState, requested_battery_bus_kw: float) -> tuple[AccountState, RawCnyIntervalLedger, float]:
        """Accept a bounded charge request up to the formal 0.6 SOC target.

        The accepted battery-side energy is converted to modeled grid energy
        using the formal charge efficiency. FC remains off during this step.
        """
        requested = float(requested_battery_bus_kw)
        if not math.isfinite(requested) or requested > 0.0:
            raise ValueError("shore charge request must be finite and nonpositive")
        if not SOC_MIN <= state.soc <= SOC_MAX:
            raise ValueError("shore starting SOC is outside hard bounds")
        eta_chg, _ = self.efficiency.require_calibrated()
        room_kwh = max(0.0, SHORE_TARGET_SOC - state.soc) * self.plant.battery_nominal_energy_kwh
        room_kw = room_kwh * 3600.0 / DT_SECONDS
        accepted_kw = min(-requested, -self.plant.battery_charge_min_kw, room_kw)
        batt = -accepted_kw
        soc = state.soc + (
            accepted_kw * DT_SECONDS / 3600.0 / self.plant.battery_nominal_energy_kwh
        )
        if state.soc <= SHORE_TARGET_SOC:
            soc = min(soc, SHORE_TARGET_SOC)
        fc_step = formal_aggregate_fc_voltage_loss_step_uv(
            state.previous_fc_kw, 0.0, DT_SECONDS,
            self.plant.fuel_cell_rated_total_kw, is_on=False,
            mapping=self.fc_mapping,
            aggregate_start_stop_cycles=aggregate_start_cycle_count(state.previous_fc_kw, 0.0),
        )
        battery_step = battery_degradation_step(
            state.soc, batt * 1000.0 / BATTERY_NOMINAL_VOLTAGE_V,
            DT_SECONDS, self.battery_normalization.nominal_charge_capacity_ah,
        )
        next_state = AccountState(
            soc=soc, previous_fc_kw=0.0,
            fc_loss_uv=state.fc_loss_uv + fc_step.total_uv,
            battery_weighted_ah=state.battery_weighted_ah + battery_step.weighted_ah,
        )
        grid_kwh = accepted_kw * DT_SECONDS / 3600.0 / eta_chg
        ledger = build_formal_interval_ledger(
            hydrogen_mass_kg=0.0,
            fuel_cell_cumulative_voltage_loss_before_uv=state.fc_loss_uv,
            fuel_cell_cumulative_voltage_loss_after_uv=next_state.fc_loss_uv,
            fuel_cell_rated_kw=self.plant.fuel_cell_rated_total_kw,
            battery_cumulative_weighted_ah_before=state.battery_weighted_ah,
            battery_cumulative_weighted_ah_after=next_state.battery_weighted_ah,
            battery_capacity_kwh=self.plant.battery_nominal_energy_kwh,
            battery_normalization=self.battery_normalization,
            shore_energy=(
                ShoreEnergy(grid_kwh, ShoreEnergyClassification.MODELED)
                if grid_kwh > 0.0 else None
            ),
        )
        return next_state, ledger, batt

    def _predicted_path(self, powers: np.ndarray, loads: np.ndarray, state: AccountState):
        current = state
        states, ledgers = [], []
        for fc, load in zip(powers, loads):
            current, ledger, _ = self.interval(current, float(fc), float(load))
            states.append(current.soc)
            ledgers.append(ledger)
        return np.asarray(states), ledgers

    def objective_terms(
        self, powers_kw: Sequence[float], forecast_kw: Sequence[float],
        reference_kw: Sequence[float], state: AccountState,
    ) -> MPCObjectiveTerms:
        """Evaluate the exact unweighted terms used by the five-step optimizer."""
        powers = np.asarray(_vector(powers_kw, "powers_kw"))
        loads = np.asarray(_vector(forecast_kw, "forecast_kw"))
        references = np.asarray(_vector(reference_kw, "reference_kw"))
        states, ledgers = self._predicted_path(powers, loads, state)
        # The discontinuous formal start cost is recorded in each ledger; the
        # optimizer uses smooth FC proxies to compare candidate powers.
        fc_price_per_uv = FC_AGGREGATE_REPLACEMENT_COST_CNY / FC_EOL_VOLTAGE_LOSS_UV
        runtime_proxy = (
            FC_LOW_RUNTIME_LOSS_UV_PER_HOUR * DT_SECONDS / 3600.0
            * fc_price_per_uv * np.sum(powers / self.plant.fuel_cell_rated_total_kw)
        )
        deltas = np.diff(np.concatenate(([state.previous_fc_kw], powers)))
        transient_proxy = 0.001 * np.dot(deltas, deltas)
        economic_cost = (
            sum(item.h2_cost_cny + item.battery_degradation_cost_cny for item in ledgers)
            + runtime_proxy + transient_proxy
        )
        reference_penalty = np.sum(
            ((powers - references) / self.plant.fuel_cell_rated_total_kw) ** 2
        )
        soc_penalty = np.sum(((states - 0.5) / 0.1) ** 2)
        return MPCObjectiveTerms(
            float(economic_cost), float(economic_cost / self.nominal_cost_cny),
            float(reference_penalty), float(soc_penalty),
        )

    def solve(self, forecast_kw: Sequence[float], reference_kw: Sequence[float], state: AccountState, weights: MPCWeights) -> Plan:
        loads = np.asarray(_vector(forecast_kw, "forecast_kw"))
        references = np.asarray(_vector(reference_kw, "reference_kw"))
        if not SOC_MIN <= state.soc <= SOC_MAX:
            raise ValueError("initial SOC is outside hard bounds")
        initial = np.clip(references, 0.0, self.plant.fuel_cell_rated_total_kw)

        def objective(powers: np.ndarray) -> float:
            return self.objective_terms(powers, loads, references, state).objective(weights)

        def margins(powers: np.ndarray) -> np.ndarray:
            batteries = loads - powers
            states, _ = self._predicted_path(powers, loads, state)
            return np.concatenate((
                batteries - self.plant.battery_charge_min_kw,
                self.plant.battery_discharge_max_kw - batteries,
                states - SOC_MIN, SOC_MAX - states,
            ))

        def optimize(start: np.ndarray):
            return minimize(
                objective, start, method="SLSQP",
                bounds=[(0.0, self.plant.fuel_cell_rated_total_kw)] * HORIZON,
                constraints=({"type": "ineq", "fun": margins},),
                options={"ftol": 1e-8, "maxiter": 200},
            )

        result = optimize(initial)
        if not result.success:
            # A feasible zero-FC start often avoids SLSQP's false
            # "constraints incompatible" result at the on/off boundary.
            zero = np.zeros(HORIZON)
            if min(margins(zero)) >= 0:
                retry = optimize(zero)
                if retry.success:
                    result = retry
        if not result.success or not np.isfinite(result.x).all() or min(margins(result.x)) < -1e-6:
            raise RuntimeError(f"economic MPC failed: {result.message}")
        powers = tuple(float(p) for p in result.x)
        states, ledgers = self._predicted_path(result.x, loads, state)
        return Plan(
            tuple(loads), tuple(references), powers,
            tuple(float(load - fc) for load, fc in zip(loads, powers)),
            tuple(float(value) for value in states), ledgers[0], objective(result.x),
        )


class PredictiveController:
    """One real LPF commit per measurement and one FC command per MPC plan.

    With no forecaster, hold the latest observed load over all five steps.
    """

    def __init__(self, *, mpc: EconomicMPC,
                 forecaster: Callable[[tuple[float, ...]], Sequence[float]] | None = None,
                 history_steps: int = 1):
        if history_steps <= 0:
            raise ValueError("history_steps must be positive")
        self.forecaster = forecaster
        self.history_steps = history_steps
        self._needs_voyage_seed = forecaster is None
        self.mpc = mpc
        self.filter = CausalBaseLoadFilter(sample_seconds=DT_SECONDS, tau_seconds=180.0)
        self.history: list[float] = []
        self.state = AccountState()
        self.pending_plan: Plan | None = None
        self._prepared: tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]] | None = None
        self._pending_action: MPCWeights | None = None
        self._pending_decision_state: tuple[float, ...] | None = None
        self.previous_action: MPCWeights | None = None
        self.errors_kw: list[float] = []
        self.previous_fc_delta_kw = 0.0

    @property
    def observed_base_kw(self) -> float | None:
        return self.filter.observed_base_kw

    def begin_voyage(self) -> tuple[float, ...]:
        """Seed a virtual zero-load decision boundary without physical cost."""
        if self.pending_plan is not None or self._pending_decision_state is not None or self.history:
            raise RuntimeError("begin_voyage requires a fresh control boundary")
        self._needs_voyage_seed = False
        self.observe(0.0)
        return self.prepare_decision()

    def observe(self, actual_load_kw: float) -> None:
        if self._needs_voyage_seed:
            raise RuntimeError("begin_voyage must seed the zero-load boundary first")
        if self.pending_plan is not None:
            raise RuntimeError("execute pending FC command before observing another load")
        load = normalize_onboard_load_kw(actual_load_kw)
        self.history.append(load)
        self.filter.commit(load)
        self._prepared = None

    def prepare_decision(self) -> tuple[float, ...]:
        """Build s_k from measurements and a forecast before DQN selects a_k."""
        if self.pending_plan is not None:
            raise RuntimeError("execute pending FC command before preparing the next state")
        if self._prepared is not None:
            return self._prepared[0]
        if len(self.history) < self.history_steps:
            raise RuntimeError("insufficient real load history")
        forecast = _vector(
            (self.history[-1],) * HORIZON if self.forecaster is None
            else self.forecaster(tuple(self.history[-self.history_steps:])),
            "load forecast",
        )
        base = self.filter.observed_base_kw
        assert base is not None
        current_base = base
        references = []
        for load in forecast:
            current_base = self.filter.alpha * current_base + (1.0 - self.filter.alpha) * load
            references.append(current_base)
        # Most recent error first; padding is used only at episode startup.
        recent_errors = list(reversed(self.errors_kw[-3:])) + [0.0] * max(0, 3 - len(self.errors_kw))
        prior_action = self.previous_action or MPCWeights(0.0, 0.0)
        rated = self.mpc.plant.fuel_cell_rated_total_kw
        physical_state = (
            float(self.state.soc), self.history[-1] / rated, base / rated,
            self.state.previous_fc_kw / rated, self.previous_fc_delta_kw / rated,
        )
        state = (
            *physical_state,
            *((value / rated for value in forecast) if self.forecaster is not None else ()),
            *(value / rated for value in recent_errors),
            float(prior_action.lambda_ref), float(prior_action.lambda_soc),
        )
        self._prepared = (state, forecast, tuple(references))
        return state

    def plan(self, weights: MPCWeights) -> Plan:
        if self.pending_plan is not None:
            raise RuntimeError("a plan is already awaiting execution")
        self.prepare_decision()
        assert self._prepared is not None
        _, forecast, references = self._prepared
        plan = self.mpc.solve(forecast, references, self.state, weights)
        self.pending_plan = plan
        self._pending_action = weights
        return plan

    def execute_next(self, actual_load_kw: float) -> ExecutedStep:
        plan = self.pending_plan
        if plan is None:
            raise RuntimeError("no MPC plan is awaiting execution")
        load = normalize_onboard_load_kw(actual_load_kw)
        command = plan.fc_power_kw[0]
        battery = load - command
        if not self.mpc.plant.battery_charge_min_kw <= battery <= self.mpc.plant.battery_discharge_max_kw:
            raise RuntimeError("actual prediction error exceeded battery power bounds")
        next_state, ledger, _ = self.mpc.interval(self.state, command, load)
        if not SOC_MIN <= next_state.soc <= SOC_MAX:
            raise RuntimeError("actual prediction error exceeded SOC hard bounds")
        self.previous_fc_delta_kw = command - self.state.previous_fc_kw
        self.state = next_state
        self.previous_action = self._pending_action
        self._pending_action = None
        self.errors_kw.append(load - plan.forecast_kw[0])
        self.pending_plan = None
        self.observe(load)
        return ExecutedStep(command, battery, next_state.soc, ledger, load - plan.forecast_kw[0])

    def run_decision(self, policy: Callable[[tuple[float, ...]], MPCWeights], *, next_actual_load_kw: float) -> DecisionTransition:
        """Simulation convenience wrapper; live callers use the two phases."""
        self.start_decision(policy)
        return self.finish_decision(next_actual_load_kw)

    def start_decision(self, policy: Callable[[tuple[float, ...]], MPCWeights]) -> tuple[tuple[float, ...], MPCWeights, Plan]:
        """At k: forecast, expose s_k, select a_k, then solve MPC."""
        current_state = self.prepare_decision()
        action = policy(current_state)
        if type(action) is not MPCWeights:
            raise TypeError("policy must return MPCWeights")
        plan = self.plan(action)
        self._pending_decision_state = current_state
        return current_state, action, plan

    def finish_decision(self, actual_load_kw: float) -> DecisionTransition:
        """At k+1: reveal actual load, settle r_k, build s_(k+1)."""
        current_state = self._pending_decision_state
        action = self._pending_action
        if current_state is None or action is None:
            raise RuntimeError("start_decision must run before the actual outcome")
        executed = self.execute_next(actual_load_kw)
        next_state = self.prepare_decision()
        self._pending_decision_state = None
        return DecisionTransition(current_state, action, executed.reward_cny, next_state, executed)

    def finish_terminal_decision(
        self, actual_load_kw: float, shore_charge_requests_kw: Sequence[float],
    ) -> DecisionTransition:
        """Settle the final ONBOARD command and all subsequent shore samples."""
        current_state = self._pending_decision_state
        action = self._pending_action
        if current_state is None or action is None:
            raise RuntimeError("start_decision must precede terminal settlement")
        requests = tuple(float(value) for value in shore_charge_requests_kw)
        if not requests or any(not math.isfinite(value) or value > 0.0 for value in requests):
            raise ValueError("terminal shore segment requires nonpositive finite charge requests")
        executed = self.execute_next(actual_load_kw)
        from .shore import settle_shore_segment

        shore = settle_shore_segment(self.mpc, self.state, requests)
        self.state = shore.end_state
        self.filter = CausalBaseLoadFilter(sample_seconds=DT_SECONDS, tau_seconds=180.0)
        self.history.clear()
        self.errors_kw.clear()
        self.previous_action = None
        self.previous_fc_delta_kw = 0.0
        self._prepared = None
        self._pending_decision_state = None
        self._needs_voyage_seed = self.forecaster is None
        terminal_state = (self.state.soc,) + (0.0,) * (len(current_state) - 1)
        return DecisionTransition(
            current_state, action,
            executed.reward_cny + shore.ledger.reward_cny,
            terminal_state, executed, done=True, shore_ledger=shore.ledger,
            shore_requested_battery_bus_kw=shore.requested_battery_bus_kw,
            shore_accepted_battery_bus_kw=shore.accepted_battery_bus_kw,
            shore_soc_path=shore.soc_path,
        )
