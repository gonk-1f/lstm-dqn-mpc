"""Causal direct-power replay with the existing formal physical ledger."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Callable

from v2.data.supervisory_rules import OperatingMode, normalize_onboard_load_kw
from v2.economics import RawCnyIntervalLedger
from v2.models.battery_energy import next_soc
from v2.models.battery_degradation import battery_life_state
from v2.models.fuel_cell_degradation import fuel_cell_life_state
from v3.control import AccountState, EconomicMPC, SOC_MAX, SOC_MIN
from v3.episode_replay import SHORE_MODES, ShoreBlock, _ledger_total, _validated_rows
from v3.shore import settle_shore_segment


ACTION_KW = tuple(range(0, 601, 10))


class ReplayExecutionError(RuntimeError):
    def __init__(self, row_index: int, mode: OperatingMode, cause: Exception):
        self.row_index = row_index
        self.mode = mode
        self.cause = cause
        super().__init__(f"row {row_index} ({mode.value}): {cause}")


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

    @property
    def total_cost_cny(self) -> float:
        return self.total_ledger.total_cost_cny


def build_state(
    physical: AccountState, observed_loads_kw: tuple[float, ...],
    accountant: EconomicMPC, *, departure: bool,
) -> tuple[float, ...]:
    """Eight features measured before the commanded interval; no future load."""
    if not observed_loads_kw or any(not isfinite(value) for value in observed_loads_kw):
        raise ValueError("state needs finite observed loads")
    current = observed_loads_kw[-1]
    previous = observed_loads_kw[-2] if len(observed_loads_kw) >= 2 else 0.0
    prior = observed_loads_kw[-3] if len(observed_loads_kw) >= 3 else 0.0
    fc_life = fuel_cell_life_state(physical.fc_loss_uv)
    battery_life = battery_life_state(
        physical.battery_weighted_ah, normalization=accountant.battery_normalization,
    )
    rated = accountant.plant.fuel_cell_rated_total_kw
    return (
        float(physical.soc), current / rated, (current - previous) / rated,
        (previous - prior) / rated, physical.previous_fc_kw / rated,
        fc_life.economic_life_fraction, battery_life.economic_life_fraction,
        float(departure),
    )


def _checked_action(value: object, feasible_actions: tuple[int, ...]) -> int:
    if type(value) is not int or value not in ACTION_KW:
        raise ValueError("FC action must be an exact 10 kW-grid integer in [0, 600]")
    if value not in feasible_actions:
        raise ValueError("FC action violates actual battery power or SOC bounds")
    return value


def feasible_fc_actions(
    physical: AccountState, measured_load_kw: float, accountant: EconomicMPC,
) -> tuple[int, ...]:
    """Screen actions using the load observed before this decision only."""
    load = normalize_onboard_load_kw(measured_load_kw)
    plant = accountant.plant
    feasible: list[int] = []
    for fc_kw in ACTION_KW:
        battery_kw = load - fc_kw
        if not plant.battery_charge_min_kw <= battery_kw <= plant.battery_discharge_max_kw:
            continue
        soc = next_soc(
            physical.soc, battery_kw, 30.0, plant.battery_nominal_energy_kwh,
            efficiency=accountant.efficiency,
        )
        if SOC_MIN - 1e-9 <= soc <= SOC_MAX + 1e-9:
            feasible.append(fc_kw)
    return tuple(feasible)


def replay_episode(
    episode: object, policy: Callable[[tuple[float, ...], tuple[int, ...]], int], *,
    accountant: EconomicMPC, initial_state: AccountState | None = None,
) -> DirectReplay:
    """Execute ONBOARD FC actions; route shore rows outside the DQN policy.

    The current exogenous load is observed before the FC decision. Physical
    feasibility is screened from that measurement; an invalid command fails
    closed without power clipping or a synthetic cost.
    """
    modes, loads, requests = _validated_rows(episode)
    if not callable(policy):
        raise TypeError("policy must be callable")
    if not isinstance(accountant, EconomicMPC):
        raise TypeError("accountant must supply the formal v3 interval ledger")
    if initial_state is not None and type(initial_state) is not AccountState:
        raise TypeError("initial_state must be an AccountState")
    physical = initial_state or AccountState()
    transitions: list[DirectTransition] = []
    shore_blocks: list[ShoreBlock] = []
    ledgers: list[RawCnyIntervalLedger] = []
    fc_trace = [0.0] * len(modes)
    battery_trace = [0.0] * len(modes)
    soc_trace = [0.0] * len(modes)
    index = 0
    while index < len(modes):
        if modes[index] in SHORE_MODES:
            end = index
            while end < len(modes) and modes[end] in SHORE_MODES:
                end += 1
            start_soc = physical.soc
            try:
                shore = settle_shore_segment(accountant, physical, requests[index:end])
            except (RuntimeError, ValueError) as exc:
                raise ReplayExecutionError(index, modes[index], exc) from exc
            physical = shore.end_state
            block = ShoreBlock(
                index, end, start_soc, physical.soc, shore.ledger,
                shore.requested_battery_bus_kw, shore.accepted_battery_bus_kw,
                shore.soc_path,
            )
            shore_blocks.append(block)
            ledgers.append(shore.ledger)
            for offset, (power, soc) in enumerate(zip(block.accepted_battery_bus_kw, block.soc_path)):
                battery_trace[index + offset] = power
                soc_trace[index + offset] = soc
            index = end
            continue

        history = (0.0,)  # virtual departure boundary, never a physical row
        while index < len(modes) and modes[index] is OperatingMode.ONBOARD:
            try:
                actual_load = normalize_onboard_load_kw(loads[index])
                current_history = (*history[-2:], actual_load)
                decision_state = build_state(physical, current_history, accountant, departure=len(history) == 1)
                feasible_actions = feasible_fc_actions(physical, actual_load, accountant)
                if not feasible_actions:
                    raise RuntimeError("no feasible FC action for actual battery power and SOC bounds")
                fc_kw = _checked_action(policy(decision_state, feasible_actions), feasible_actions)
                next_physical, ledger, battery_kw = accountant.interval(physical, fc_kw, actual_load)
                if not accountant.plant.battery_charge_min_kw <= battery_kw <= accountant.plant.battery_discharge_max_kw:
                    raise RuntimeError("actual battery power exceeded physical bounds")
                if not SOC_MIN <= next_physical.soc <= SOC_MAX:
                    raise RuntimeError("actual SOC exceeded physical bounds")
            except (RuntimeError, ValueError, TypeError) as exc:
                raise ReplayExecutionError(index, modes[index], exc) from exc
            physical = next_physical
            ledgers.append(ledger)
            fc_trace[index] = float(fc_kw)
            battery_trace[index] = battery_kw
            soc_trace[index] = physical.soc
            history = current_history
            last_onboard = index + 1 == len(modes) or modes[index + 1] is not OperatingMode.ONBOARD
            shore_ledger = None
            reward = ledger.reward_cny
            if last_onboard and index + 1 < len(modes):
                shore_start = index + 1
                shore_end = shore_start
                while shore_end < len(modes) and modes[shore_end] in SHORE_MODES:
                    shore_end += 1
                try:
                    shore = settle_shore_segment(accountant, physical, requests[shore_start:shore_end])
                except (RuntimeError, ValueError) as exc:
                    raise ReplayExecutionError(shore_start, modes[shore_start], exc) from exc
                block = ShoreBlock(
                    shore_start, shore_end, physical.soc, shore.end_state.soc,
                    shore.ledger, shore.requested_battery_bus_kw,
                    shore.accepted_battery_bus_kw, shore.soc_path,
                )
                shore_blocks.append(block)
                ledgers.append(shore.ledger)
                for offset, (power, soc) in enumerate(zip(block.accepted_battery_bus_kw, block.soc_path)):
                    battery_trace[shore_start + offset] = power
                    soc_trace[shore_start + offset] = soc
                physical = shore.end_state
                shore_ledger = shore.ledger
                reward += shore_ledger.reward_cny
            if shore_ledger is not None:
                successor_history = (0.0, normalize_onboard_load_kw(loads[shore_end])) if shore_end < len(modes) else (0.0,)
            elif index + 1 < len(modes) and modes[index + 1] is OperatingMode.ONBOARD:
                successor_history = (*history[-2:], normalize_onboard_load_kw(loads[index + 1]))
            else:
                successor_history = history
            next_state = build_state(physical, successor_history, accountant, departure=shore_ledger is not None)
            next_feasible = (
                feasible_fc_actions(physical, loads[index + 1], accountant)
                if not last_onboard else ()
            )
            transitions.append(DirectTransition(
                decision_state, fc_kw, reward, next_state, ledger,
                battery_kw, next_physical.soc, last_onboard, next_feasible, shore_ledger,
            ))
            if last_onboard and shore_ledger is not None:
                index = shore_end
                break
            index += 1
    return DirectReplay(
        str(episode.sample_id), str(episode.split), tuple(transitions),
        tuple(shore_blocks), _ledger_total(ledgers), physical,
        tuple(fc_trace), tuple(battery_trace), tuple(soc_trace),
    )
