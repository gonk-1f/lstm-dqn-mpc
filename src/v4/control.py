"""Causal direct-power replay with the existing formal physical ledger."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import ceil, fsum, isfinite
from typing import Callable

from v2.data.supervisory_rules import OperatingMode, normalize_onboard_load_kw
from v2.economics import RawCnyIntervalLedger, ShoreEnergyClassification
from v2.models.battery_energy import next_soc
from v2.models.fuel_cell_degradation import (
    FC_AGGREGATE_REPLACEMENT_COST_CNY, FC_AGGREGATE_TO_REFERENCE_POWER_RATIO,
    FC_EOL_VOLTAGE_LOSS_UV, FC_TRANSIENT_LOSS_UV_PER_DELTA_KW,
)
from v3.control import AccountState, EconomicMPC, SOC_MAX, SOC_MIN, SHORE_TARGET_SOC
from v3.episode_replay import SHORE_MODES, ShoreBlock, _ledger_total
from v3.shore import settle_shore_segment

from .reward_feedback import BatteryEnergyValue
from .independent_economics import (
    ZhouEconomicEvaluation, ZhouEvaluationConfig, evaluate_zhou_economics,
    soc_equivalent_hydrogen_cost_cny,
)
from .actions import FC_ACTION_MODES, action_values_kw, candidate_fc_actions


ACTION_KW = tuple(range(0, 601, 10))
SOC_REF = 0.60
DEFAULT_BETA_SOC = 200.0
DEFAULT_STATE_MODE = 'history7'
STATE_MODES = ('history7', 'minimal3', 'load_delta4')
STATE_FEATURES_BY_MODE = {
    'history7': (
        'soc', 'previous_fc_kw_over_600', 'previous_fc_delta_kw_over_600',
        'second_previous_fc_delta_kw_over_600', 'current_load_kw_over_600',
        'current_load_delta_kw_over_600', 'previous_load_delta_kw_over_600',
    ),
    'minimal3': ('soc', 'current_load_kw_over_600', 'previous_fc_kw_over_600'),
    'load_delta4': (
        'soc', 'current_load_kw_over_600', 'previous_fc_kw_over_600',
        'current_load_delta_kw_over_600',
    ),
}
REWARD_COST_MODES = ('full_economic', 'energy_only', 'energy_with_fc_ramp', 'control_only',
                     'zhou_joint', 'zhou_economic_only')
# Fixed CNY/kW slope of formal aggregate FC transient degradation before EOL clipping.
# This is a separate training term; the four-component ledger is not changed.
FC_RAMP_REWARD_CNY_PER_KW = (
    FC_TRANSIENT_LOSS_UV_PER_DELTA_KW * FC_AGGREGATE_TO_REFERENCE_POWER_RATIO
    * FC_AGGREGATE_REPLACEMENT_COST_CNY / FC_EOL_VOLTAGE_LOSS_UV
)
FC_SQUARED_RAMP_REWARD_PER_KW2 = FC_RAMP_REWARD_CNY_PER_KW / 60.0


def ledger_training_reward(ledger: RawCnyIntervalLedger, reward_cost_mode: str) -> float:
    """Select the training objective without altering the four-cost ledger."""
    if reward_cost_mode not in REWARD_COST_MODES:
        raise ValueError(f'reward_cost_mode must be one of {REWARD_COST_MODES}')
    if reward_cost_mode == 'control_only':
        return 0.0
    if reward_cost_mode in ('full_economic', 'zhou_joint', 'zhou_economic_only'):
        return ledger.reward_cny
    return -fsum((ledger.h2_cost_cny, ledger.shore_cost_cny))


def soc_deviation_squared(soc: float) -> float:
    """Auxiliary control objective; not an actual economic ledger expense."""
    if not isfinite(soc):
        raise ValueError("SOC must be finite")
    return (soc - SOC_REF) ** 2


class ReplayExecutionError(RuntimeError):
    def __init__(
        self, row_index: int, mode: OperatingMode, cause: Exception, *,
        executed_transitions: tuple[DirectTransition, ...] = (),
        failure_kind: str = "execution_error",
        failure_cause: str | None = None, failed_state: tuple[float,...] | None = None,
    ):
        self.row_index = row_index
        self.mode = mode
        self.cause = cause
        self.executed_transitions = executed_transitions
        self.failure_kind = failure_kind
        self.failure_cause = failure_cause
        self.failed_state = failed_state
        super().__init__(f"row {row_index} ({mode.value}): {cause}")


class NoFeasibleFCActionError(RuntimeError):
    """No grid action can satisfy the present battery power and SOC bounds."""


@dataclass(frozen=True)
class DirectTransition:
    state: tuple[float, ...]
    action_kw: int
    reward_cny: float
    next_state: tuple[float, ...]
    executed_ledger: RawCnyIntervalLedger
    actual_battery_kw: float
    actual_soc: float
    done: bool
    next_feasible_actions: tuple[int, ...]
    shore_ledger: RawCnyIntervalLedger | None = None
    modeled_terminal_ledger: RawCnyIntervalLedger | None = None
    soc_soft_penalty_cny: float = 0.0
    original_reward: float | None = None
    immediate_battery_energy_adjustment: float = 0.0
    terminal_correction: float = 0.0
    terminal_reason: str | None = None
    failure_penalty_equivalent_cny: float = 0.0
    physical_feasible_actions: tuple[int, ...] = ()
    policy_candidate_actions: tuple[int, ...] = ()
    excluded_degradation_cost_cny: float = 0.0
    fc_ramp_penalty_cny: float = 0.0
    duration_s: float = 30.0
    raw_load_kw: float | None = None
    relative_time_s: float | None = None
    action_mode: str = 'absolute61'
    action_index: int | None = None
    action_delta_kw: int | None = None
    soc_equivalent_hydrogen_cost_cny: float = 0.0

    @property
    def is_successful_terminal(self) -> bool:
        return self.done and self.terminal_reason in (None,'completed')

    @property
    def new_reward(self) -> float:
        return self.reward_cny

    @property
    def original_economic_ledger(self) -> RawCnyIntervalLedger:
        """Executed four-component ledger, including any modeled port charge."""
        if self.shore_ledger is None:
            return self.executed_ledger
        return _ledger_total((self.executed_ledger, self.shore_ledger))


@dataclass(frozen=True)
class ModeledTerminalSettlement:
    """Accounting-only recharge scenario; the actual vessel state is unchanged."""

    start_soc: float
    reference_soc: float
    grid_energy_kwh: float
    ledger: RawCnyIntervalLedger
    classification: ShoreEnergyClassification = ShoreEnergyClassification.MODELED


@dataclass(frozen=True)
class DirectReplay:
    sample_id: str
    split: str
    transitions: tuple[DirectTransition, ...]
    shore_blocks: tuple[ShoreBlock, ...]
    total_ledger: RawCnyIntervalLedger
    final_state: AccountState
    fc_power_kw_by_row: tuple[float, ...]
    battery_bus_kw_by_row: tuple[float, ...]
    soc_by_row: tuple[float, ...]
    modeled_terminal_settlement: ModeledTerminalSettlement | None = None
    economic_evaluation: ZhouEconomicEvaluation | None = None
    task_mode: str = 'legacy'

    @property
    def total_cost_cny(self) -> float:
        return self.total_ledger.total_cost_cny

    @property
    def comparable_cost_cny(self) -> float:
        if self.economic_evaluation is not None:
            return self.economic_evaluation.total_cost_cny
        modeled = self.modeled_terminal_settlement
        return self.total_cost_cny + (0.0 if modeled is None else modeled.ledger.total_cost_cny)

    @property
    def soc_soft_penalty_cny(self) -> float:
        return fsum(item.soc_soft_penalty_cny for item in self.transitions)


def modeled_terminal_settlement(
    physical: AccountState, accountant: EconomicMPC,
) -> ModeledTerminalSettlement | None:
    """Value missing recharge to SOC 0.6 at the bounded 624 kW charge rate."""
    reference_soc = SHORE_TARGET_SOC
    if physical.soc >= reference_soc:
        return None
    charge_kw = -accountant.plant.battery_charge_min_kw
    duration = 30.0
    energy_kwh = (reference_soc - physical.soc) * accountant.plant.battery_nominal_energy_kwh
    steps = max(1, ceil(energy_kwh * 3600.0 / (charge_kw * duration)))
    modeled_start = AccountState(
        soc=physical.soc, previous_fc_kw=0.0,
        fc_loss_uv=physical.fc_loss_uv,
        battery_weighted_ah=physical.battery_weighted_ah,
    )
    settlement = settle_shore_segment(accountant, modeled_start, (-charge_kw,) * steps)
    eta_chg, _ = accountant.efficiency.require_calibrated()
    grid_kwh = fsum(-power * duration / 3600.0 / eta_chg for power in settlement.accepted_battery_bus_kw)
    return ModeledTerminalSettlement(
        physical.soc, reference_soc, grid_kwh, settlement.ledger,
    )


def _modeled_fixed_target_shore(physical: AccountState, accountant: EconomicMPC,
                                recorded_requests: tuple[float, ...]):
    """Settle a port call to SOC 0.6; logged powers do not limit this model.

    The existing formal shore interval supplies the FC shutdown, charging
    efficiency, battery ageing and tariff. Synthetic charge intervals exist
    only inside this accounting calculation, not as measured 30 s trace rows.
    """
    charge_kw = -accountant.plant.battery_charge_min_kw
    deficit_kwh = max(0.0, SHORE_TARGET_SOC - physical.soc) * accountant.plant.battery_nominal_energy_kwh
    steps = max(1, ceil(deficit_kwh * 3600.0 / (charge_kw * 30.0)))
    settlement = settle_shore_segment(accountant, physical, (-charge_kw,) * steps)
    if deficit_kwh > 0 and abs(settlement.end_state.soc - SHORE_TARGET_SOC) > 1e-9:
        raise RuntimeError('modeled fixed-target SHORE failed to reach SOC 0.6')
    if deficit_kwh > 0:
        settlement = replace(settlement, end_state=replace(settlement.end_state, soc=SHORE_TARGET_SOC))
    block_data = (recorded_requests, (), (), 'modeled_fixed_target_soc_0.6')
    return settlement, block_data


def _validated_rows_v4(episode: object):
    """Keep raw shore power only for provenance; it is not a charge request."""
    try:
        modes = tuple(OperatingMode(value) for value in episode.operating_mode)
        loads = tuple(float(value) for value in episode.load_kw)
        independent = getattr(episode, 'task_mode', 'legacy') == 'independent_voyage'
        recorded_battery = ((0.0,) * len(modes) if independent else
                            tuple(float(value) for value in episode.battery_bus_kw))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError('episode lacks valid operating modes and power rows') from exc
    if not modes or len(modes) != len(loads) or len(modes) != len(recorded_battery):
        raise ValueError('episode mode and power rows must have the same positive length')
    if OperatingMode.UNRESOLVED in modes:
        raise ValueError('unresolved operating mode blocks v4 replay')
    if independent and any(mode is not OperatingMode.ONBOARD for mode in modes):
        raise ValueError('independent voyages require all rows to be ONBOARD')
    if any(not isfinite(value) for value in (*loads, *recorded_battery)):
        raise ValueError('episode power rows must be finite')
    return modes, loads, recorded_battery


def _independent_timing(episode: object, row_count: int):
    try:
        times = tuple(float(value) for value in episode.time_s)
        durations = tuple(float(value) for value in episode.dt_s)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError('independent voyages require time_s and dt_s interval arrays') from exc
    if (len(times) != row_count or len(durations) != row_count or times[0] != 0.0
            or any(not isfinite(value) for value in (*times, *durations))
            or any(duration <= 0.0 for duration in durations)
            or any(abs(right - left - duration) > 1e-9
                   for left, right, duration in zip(times, times[1:], durations))):
        raise ValueError('independent interval times and durations must be finite, positive and contiguous')
    return times, durations


def build_state(
    physical: AccountState, fc_history_kw: tuple[float, float, float],
    observed_loads_kw: tuple[float, float, float],
    *, state_mode: str = DEFAULT_STATE_MODE,
) -> tuple[float, ...]:
    """Select causal features; FC history contains executed commands only."""
    if state_mode not in STATE_FEATURES_BY_MODE:
        raise ValueError(f'state_mode must be one of {STATE_MODES}')
    if (len(fc_history_kw) != 3 or len(observed_loads_kw) != 3 or
            any(not isfinite(value) for value in (*fc_history_kw, *observed_loads_kw))):
        raise ValueError("state needs three finite FC and observed load samples")
    if abs(fc_history_kw[-1] - physical.previous_fc_kw) > 1e-9:
        raise ValueError("FC history must match the executed physical state")
    fc_prior2, fc_prior1, fc_previous = fc_history_kw
    load_prior2, load_prior1, current = observed_loads_kw
    rated = float(ACTION_KW[-1])
    if state_mode == 'minimal3':
        return float(physical.soc), current / rated, fc_previous / rated
    if state_mode == 'load_delta4':
        return (float(physical.soc), current / rated, fc_previous / rated,
                (current - load_prior1) / rated)
    return (
        float(physical.soc), fc_previous / rated,
        (fc_previous - fc_prior1) / rated, (fc_prior1 - fc_prior2) / rated,
        current / rated, (current - load_prior1) / rated,
        (load_prior1 - load_prior2) / rated,
    )


def state_feature_value(
    state: tuple[float, ...], feature_name: str, *, state_mode: str | None = None,
) -> float:
    """Resolve a named input feature without assuming a layout-specific index."""
    if state_mode is None:
        matches = [mode for mode, features in STATE_FEATURES_BY_MODE.items()
                   if len(features) == len(state)]
        if len(matches) != 1:
            raise ValueError('state dimension does not identify a unique state mode')
        state_mode = matches[0]
    if state_mode not in STATE_FEATURES_BY_MODE:
        raise ValueError(f'state_mode must be one of {STATE_MODES}')
    features = STATE_FEATURES_BY_MODE[state_mode]
    if len(state) != len(features):
        raise ValueError('state dimension does not match its named features')
    if any(not isfinite(value) for value in state):
        raise ValueError('state features must be finite')
    if feature_name not in features:
        raise ValueError(f'{feature_name!r} is not a feature of {state_mode}')
    return float(state[features.index(feature_name)])


def _checked_action(value: object, feasible_actions: tuple[int, ...]) -> int:
    if type(value) is not int or value not in ACTION_KW:
        raise ValueError("FC action must be an exact 10 kW-grid integer in [0, 600]")
    if value not in feasible_actions:
        raise ValueError("FC action violates the current strategy candidate mask")
    return value


def feasible_fc_actions(
    physical: AccountState, measured_load_kw: float, accountant: EconomicMPC,
    *, duration_s: float = 30.0,
) -> tuple[int, ...]:
    """Screen actions using the load observed before this decision only."""
    load = normalize_onboard_load_kw(measured_load_kw)
    duration = float(duration_s)
    if isinstance(duration_s, bool) or not isfinite(duration) or duration <= 0.0:
        raise ValueError('mask duration must be finite and positive')
    plant = accountant.plant
    feasible: list[int] = []
    for fc_kw in ACTION_KW:
        battery_kw = load - fc_kw
        if not plant.battery_charge_min_kw <= battery_kw <= plant.battery_discharge_max_kw:
            continue
        soc = next_soc(
            physical.soc, battery_kw, duration, plant.battery_nominal_energy_kwh,
            efficiency=accountant.efficiency,
        )
        if SOC_MIN <= soc <= SOC_MAX:
            feasible.append(fc_kw)
    return tuple(feasible)


def infeasibility_cause(measured_load_kw: float, accountant: EconomicMPC) -> str:
    """Diagnose an empty physical mask without changing its definition."""
    load = normalize_onboard_load_kw(measured_load_kw)
    plant = accountant.plant
    power_only = any(plant.battery_charge_min_kw <= load-power <= plant.battery_discharge_max_kw
                     for power in ACTION_KW)
    return 'soc_limited' if power_only else 'structural_power'


def replay_episode(
    episode: object, policy: Callable[[tuple[float, ...], tuple[int, ...]], int], *,
    accountant: EconomicMPC, initial_state: AccountState | None = None,
    beta_soc: float = 0.0,
    redistribute_battery_energy: bool = False,
    reward_cost_mode: str = 'full_economic',
    state_mode: str = DEFAULT_STATE_MODE,
    economic_evaluation_config: ZhouEvaluationConfig | None = None,
    action_mode: str = 'absolute61',
    on_transition: Callable[[int, DirectTransition], None] | None = None,
) -> DirectReplay:
    """Execute ONBOARD FC actions; route shore rows outside the DQN policy.

    The current exogenous load is observed before the FC decision. Physical
    feasibility is screened from that measurement; an invalid command fails
    closed without power clipping or a synthetic cost. Reward redistribution
    is opt-in so historical callers keep their original reward timing.
    Energy-based candidates exclude degradation from reward only. The ramp
    candidate adds one ONBOARD training term; physical ledgers are unchanged.
    Control-only uses just that ONBOARD ramp term and the SOC deviation;
    all physical settlement ledgers still run without entering its reward.
    An explicitly marked independent voyage starts from the default fresh
    state, integrates its real interval durations, and has no shore or
    synthetic terminal action. Zhou rewards use the signed per-step SOC
    conversion, whose sum equals its evaluation correction without another
    terminal reward. Economic-only removes auxiliary SOC/ramp terms while
    retaining all original physical depreciation. Increment actions restrict executed FC candidates only;
    there is no clipping or fallback to the absolute action set.
    """
    modes, loads, requests = _validated_rows_v4(episode)
    task_mode = getattr(episode, 'task_mode', 'legacy')
    if task_mode not in ('legacy', 'independent_voyage'):
        raise ValueError('task_mode must be legacy or independent_voyage')
    independent = task_mode == 'independent_voyage'
    if on_transition is not None and (
            not callable(on_transition) or not independent
            or any(mode is not OperatingMode.ONBOARD for mode in modes)):
        raise ValueError('transition notifications require an independent ONBOARD-only voyage')
    if independent:
        if type(economic_evaluation_config) is not ZhouEvaluationConfig:
            raise ValueError('independent voyages require an explicit ZhouEvaluationConfig')
        if initial_state is not None and initial_state != AccountState():
            raise ValueError('independent voyages require the default SOC 0.6, FC off and fresh loss state')
        relative_times, durations = _independent_timing(episode, len(modes))
    else:
        if economic_evaluation_config is not None:
            raise ValueError('Zhou economic evaluation applies only to independent voyages')
        relative_times, durations = (None,) * len(modes), (30.0,) * len(modes)
    if not callable(policy):
        raise TypeError("policy must be callable")
    if not isinstance(accountant, EconomicMPC):
        raise TypeError("accountant must supply the formal v3 interval ledger")
    if initial_state is not None and type(initial_state) is not AccountState:
        raise TypeError("initial_state must be an AccountState")
    if not isfinite(beta_soc) or beta_soc < 0.0:
        raise ValueError("beta_soc must be finite and nonnegative")
    if type(redistribute_battery_energy) is not bool:
        raise ValueError('redistribute_battery_energy must be bool')
    if reward_cost_mode not in REWARD_COST_MODES:
        raise ValueError(f'reward_cost_mode must be one of {REWARD_COST_MODES}')
    if state_mode not in STATE_FEATURES_BY_MODE:
        raise ValueError(f'state_mode must be one of {STATE_MODES}')
    if action_mode not in FC_ACTION_MODES:
        raise ValueError(f'action_mode must be one of {FC_ACTION_MODES}')
    if action_mode == 'increment7' and (not independent or state_mode not in ('minimal3', 'load_delta4')):
        raise ValueError('increment7 requires independent voyages with minimal3 or load_delta4 states')
    if reward_cost_mode in ('zhou_joint', 'zhou_economic_only') and not independent:
        raise ValueError(f'{reward_cost_mode} requires independent voyages')
    if reward_cost_mode == 'zhou_economic_only' and beta_soc != 0.0:
        raise ValueError('zhou_economic_only requires beta_soc=0')
    if reward_cost_mode in ('control_only', 'zhou_joint', 'zhou_economic_only') and redistribute_battery_energy:
        raise ValueError(f'{reward_cost_mode} does not allow battery energy redistribution')
    energy_value = BatteryEnergyValue.from_accountant(accountant)
    physical = initial_state or AccountState()
    transitions: list[DirectTransition] = []
    shore_blocks: list[ShoreBlock] = []
    ledgers: list[RawCnyIntervalLedger] = []
    fc_trace = [0.0] * len(modes)
    battery_trace = [0.0] * len(modes)
    soc_trace = [0.0] * len(modes)
    terminal_settlement: ModeledTerminalSettlement | None = None
    index = 0
    while index < len(modes):
        if modes[index] is OperatingMode.UNKNOWN:
            raise ReplayExecutionError(
                index, modes[index], ValueError('unverified data boundary truncates this sample'),
                executed_transitions=tuple(transitions), failure_kind='data_truncation',
            )
        if modes[index] in SHORE_MODES:
            end = index
            while end < len(modes) and modes[end] in SHORE_MODES:
                end += 1
            if OperatingMode.SHORE_CHARGING not in modes[index:end]:
                raise ReplayExecutionError(index, modes[index], ValueError('unconfirmed SHORE block'))
            start_soc = physical.soc
            try:
                shore, block_data = _modeled_fixed_target_shore(physical, accountant, requests[index:end])
            except (RuntimeError, ValueError) as exc:
                raise ReplayExecutionError(
                    index, modes[index], exc, executed_transitions=tuple(transitions),
                ) from exc
            physical = shore.end_state
            block = ShoreBlock(
                index, end, start_soc, physical.soc, shore.ledger,
                *block_data,
            )
            shore_blocks.append(block)
            ledgers.append(shore.ledger)
            for row in range(index, end):
                soc_trace[row] = start_soc if modes[row] is OperatingMode.SHORE_PENDING else physical.soc
            index = end
            continue

        history: tuple[float, float, float] | None = None
        fc_history = (physical.previous_fc_kw,) * 3
        voyage_initial_soc = physical.soc
        while index < len(modes) and modes[index] is OperatingMode.ONBOARD:
            empty_physical_mask=False
            empty_increment_mask=False
            try:
                actual_load = normalize_onboard_load_kw(loads[index])
                # V5 exposes the measured causal load difference, including
                # small raw negative readings. Physical execution retains its
                # existing normalization; historical state modes are unchanged.
                observed_load = loads[index] if state_mode == 'load_delta4' else actual_load
                current_history = ((observed_load,) * 3 if history is None
                                   else (*history[-2:], observed_load))
                decision_state = build_state(physical, fc_history, current_history, state_mode=state_mode)
                physical_actions = feasible_fc_actions(
                    physical, actual_load, accountant, duration_s=durations[index])
                if not physical_actions:
                    empty_physical_mask=True
                    raise NoFeasibleFCActionError(
                        "no feasible FC action for actual battery power and SOC bounds"
                    )
                candidate_actions = candidate_fc_actions(
                    physical_actions, physical.previous_fc_kw, action_mode=action_mode)
                if not candidate_actions:
                    empty_increment_mask=True
                    raise NoFeasibleFCActionError(
                        'no feasible FC increment; physically feasible absolute powers exist')
                fc_kw = _checked_action(policy(decision_state, candidate_actions), candidate_actions)
                previous_fc_kw = physical.previous_fc_kw
                next_physical, ledger, battery_kw = (
                    accountant.interval(physical, fc_kw, actual_load, duration_s=durations[index])
                    if independent else accountant.interval(physical, fc_kw, actual_load))
                if not accountant.plant.battery_charge_min_kw <= battery_kw <= accountant.plant.battery_discharge_max_kw:
                    raise RuntimeError("actual battery power exceeded physical bounds")
                if not SOC_MIN <= next_physical.soc <= SOC_MAX:
                    raise RuntimeError("actual SOC exceeded physical bounds")
            except (RuntimeError, ValueError, TypeError) as exc:
                raise ReplayExecutionError(
                    index, modes[index], exc, executed_transitions=tuple(transitions),
                    failure_kind=(
                        "no_feasible_action" if (empty_physical_mask or empty_increment_mask)
                        and isinstance(exc, NoFeasibleFCActionError)
                        else "execution_error"
                    ),
                    failure_cause=('increment_action_limited' if empty_increment_mask else
                                   infeasibility_cause(actual_load,accountant)
                                   if empty_physical_mask and isinstance(exc,NoFeasibleFCActionError) else None),
                    failed_state=(decision_state if (empty_physical_mask or empty_increment_mask)
                                  and isinstance(exc,NoFeasibleFCActionError) else None),
                ) from exc
            physical = next_physical
            ledgers.append(ledger)
            fc_trace[index] = float(fc_kw)
            battery_trace[index] = battery_kw
            soc_trace[index] = physical.soc
            history = current_history
            fc_history = (*fc_history[-2:], float(fc_kw))
            last_onboard = index + 1 == len(modes) or modes[index + 1] is not OperatingMode.ONBOARD
            shore_ledger = None
            modeled_ledger = None
            # Once per executed ONBOARD action, before any SHORE/terminal charge.
            soc_penalty = (0.0 if reward_cost_mode == 'zhou_economic_only'
                           else beta_soc * soc_deviation_squared(next_physical.soc))
            fc_ramp_penalty = (FC_RAMP_REWARD_CNY_PER_KW * abs(fc_kw - previous_fc_kw)
                               if reward_cost_mode in ('energy_with_fc_ramp', 'control_only') else 0.0)
            soc_energy_cost = 0.0
            if reward_cost_mode in ('zhou_joint', 'zhou_economic_only'):
                if reward_cost_mode == 'zhou_joint':
                    fc_ramp_penalty = FC_SQUARED_RAMP_REWARD_PER_KW2 * (fc_kw - previous_fc_kw) ** 2
                soc_energy_cost = soc_equivalent_hydrogen_cost_cny(
                    state_feature_value(decision_state, 'soc', state_mode=state_mode),
                    next_physical.soc, economic_evaluation_config)
            reward = (ledger_training_reward(ledger, reward_cost_mode)
                      - soc_energy_cost - soc_penalty - fc_ramp_penalty)
            if last_onboard and index + 1 < len(modes) and modes[index + 1] in SHORE_MODES:
                shore_start = index + 1
                shore_end = shore_start
                while shore_end < len(modes) and modes[shore_end] in SHORE_MODES:
                    shore_end += 1
                if OperatingMode.SHORE_CHARGING not in modes[shore_start:shore_end]:
                    raise ReplayExecutionError(
                        shore_start, modes[shore_start], ValueError('unconfirmed SHORE block'),
                        executed_transitions=tuple(transitions), failure_kind='data_truncation',
                    )
                try:
                    shore, block_data = _modeled_fixed_target_shore(
                        physical, accountant, requests[shore_start:shore_end])
                except (RuntimeError, ValueError) as exc:
                    raise ReplayExecutionError(
                        shore_start, modes[shore_start], exc,
                        executed_transitions=tuple(transitions),
                    ) from exc
                block = ShoreBlock(
                    shore_start, shore_end, physical.soc, shore.end_state.soc,
                    shore.ledger, *block_data,
                )
                shore_blocks.append(block)
                ledgers.append(shore.ledger)
                for row in range(shore_start, shore_end):
                    soc_trace[row] = (
                        block.start_soc if modes[row] is OperatingMode.SHORE_PENDING
                        else shore.end_state.soc)
                physical = shore.end_state
                shore_ledger = shore.ledger
                reward += ledger_training_reward(shore_ledger, reward_cost_mode)
            elif last_onboard and index + 1 == len(modes) and not independent:
                terminal_settlement = modeled_terminal_settlement(physical, accountant)
                if terminal_settlement is not None:
                    modeled_ledger = terminal_settlement.ledger
                    reward += ledger_training_reward(modeled_ledger, reward_cost_mode)
            excluded_degradation = (fsum(
                cost
                for attached_ledger in (ledger, shore_ledger, modeled_ledger)
                if attached_ledger is not None
                for cost in (attached_ledger.fuel_cell_degradation_cost_cny,
                             attached_ledger.battery_degradation_cost_cny)
            ) if reward_cost_mode not in ('full_economic', 'zhou_joint', 'zhou_economic_only') else 0.0)
            original_reward = reward
            # Actual onboard SOC comes from next_physical, before any shore
            # settlement changes physical. No extra efficiency is applied.
            adjustment = correction = 0.0
            if redistribute_battery_energy:
                adjustment = -(energy_value(next_physical.soc) - energy_value(decision_state[0]))
                if last_onboard:
                    correction = energy_value.terminal_correction(voyage_initial_soc,next_physical.soc)
                reward = original_reward + adjustment + correction
            if shore_ledger is not None:
                successor_load = ((loads[shore_end] if state_mode == 'load_delta4'
                                   else normalize_onboard_load_kw(loads[shore_end]))
                                  if shore_end < len(modes) and modes[shore_end] is OperatingMode.ONBOARD
                                  else observed_load)
                successor_history = (successor_load,) * 3
                successor_fc_history = (physical.previous_fc_kw,) * 3
            elif index + 1 < len(modes) and modes[index + 1] is OperatingMode.ONBOARD:
                successor_load = (loads[index + 1] if state_mode == 'load_delta4'
                                  else normalize_onboard_load_kw(loads[index + 1]))
                successor_history = (*history[-2:], successor_load)
                successor_fc_history = fc_history
            else:
                successor_history = history
                successor_fc_history = fc_history
            next_state = build_state(physical, successor_fc_history, successor_history, state_mode=state_mode)
            has_next_onboard = (shore_ledger is not None and shore_end < len(modes)
                                and modes[shore_end] is OperatingMode.ONBOARD)
            data_truncation = ((shore_ledger is not None and shore_end < len(modes)
                                and modes[shore_end] is OperatingMode.UNKNOWN)
                               or (shore_ledger is None and index + 1 < len(modes)
                                   and modes[index + 1] is OperatingMode.UNKNOWN))
            done = last_onboard and not has_next_onboard and not data_truncation
            next_load = loads[shore_end] if has_next_onboard else loads[index + 1] if not last_onboard else None
            next_physical_actions = (
                feasible_fc_actions(physical, next_load, accountant,
                                    duration_s=durations[shore_end if has_next_onboard else index + 1])
                if next_load is not None else ()
            )
            next_feasible = candidate_fc_actions(
                next_physical_actions, physical.previous_fc_kw, action_mode=action_mode)
            transitions.append(DirectTransition(
                decision_state, fc_kw, reward, next_state, ledger,
                battery_kw, next_physical.soc, done, next_feasible,
                shore_ledger, modeled_ledger, soc_penalty,
                original_reward, adjustment, correction,
                'data_truncation' if data_truncation else 'completed' if done else None,
                physical_feasible_actions=physical_actions,
                policy_candidate_actions=candidate_actions,
                excluded_degradation_cost_cny=excluded_degradation,
                fc_ramp_penalty_cny=fc_ramp_penalty,
                duration_s=durations[index],
                raw_load_kw=loads[index] if independent else None,
                relative_time_s=relative_times[index],
                action_mode=action_mode,
                action_index=action_values_kw(action_mode).index(
                    int(fc_kw - previous_fc_kw) if action_mode == 'increment7' else fc_kw),
                action_delta_kw=int(fc_kw - previous_fc_kw),
                soc_equivalent_hydrogen_cost_cny=soc_energy_cost,
            ))
            # An empty nonterminal successor must first pass through the existing
            # physical-failure handler. Never expose its unpenalized tail to TD.
            if on_transition is not None and (done or next_feasible):
                on_transition(index, transitions[-1])
            if last_onboard and shore_ledger is not None:
                index = shore_end
                break
            index += 1
    total_ledger = _ledger_total(ledgers)
    economic_evaluation = (evaluate_zhou_economics(
        total_ledger, soc_initial=0.6, soc_terminal=physical.soc,
        config=economic_evaluation_config) if independent else None)
    return DirectReplay(
        str(episode.sample_id), str(episode.split), tuple(transitions),
        tuple(shore_blocks), total_ledger, physical,
        tuple(fc_trace), tuple(battery_trace), tuple(soc_trace),
        terminal_settlement, economic_evaluation, task_mode,
    )
