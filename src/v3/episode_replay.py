"""Explicit operating-mode gate for causal v3 ONBOARD and shore replay."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import fsum, isfinite
from typing import Callable, Sequence

from v2.data.supervisory_rules import OperatingMode
from v2.economics import RawCnyIntervalLedger

from .control import AccountState, DecisionTransition, EconomicMPC, MPCWeights, PredictiveController
from .shore import settle_shore_segment


SHORE_MODES = frozenset((OperatingMode.SHORE_PENDING, OperatingMode.SHORE_CHARGING))


class ReplayExecutionError(RuntimeError):
    def __init__(self, row_index: int, mode: OperatingMode, cause: Exception):
        self.row_index = row_index
        self.mode = mode
        self.cause = cause
        super().__init__(f"row {row_index} ({mode.value}): {cause}")


@dataclass(frozen=True)
class ShoreBlock:
    start_index: int
    end_index_exclusive: int
    start_soc: float
    end_soc: float
    ledger: RawCnyIntervalLedger
    requested_battery_bus_kw: tuple[float, ...]
    accepted_battery_bus_kw: tuple[float, ...]
    soc_path: tuple[float, ...]
    settlement_basis: str = 'logged_charge_request'


@dataclass(frozen=True)
class EpisodeReplay:
    sample_id: str
    split: str
    start_soc: float
    final_state: AccountState
    transitions: tuple[DecisionTransition, ...]
    shore_blocks: tuple[ShoreBlock, ...]
    total_ledger: RawCnyIntervalLedger
    fc_power_kw_by_row: tuple[float, ...]
    battery_bus_kw_by_row: tuple[float, ...]
    soc_by_row: tuple[float, ...]
    onboard_steps: int
    shore_steps: int

    @property
    def total_cost_cny(self) -> float:
        return self.total_ledger.total_cost_cny


def _validated_rows(episode: object) -> tuple[tuple[OperatingMode, ...], tuple[float, ...], tuple[float, ...]]:
    try:
        modes = tuple(OperatingMode(value) for value in episode.operating_mode)
        loads = tuple(float(value) for value in episode.load_kw)
        battery = tuple(float(value) for value in episode.battery_bus_kw)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("episode lacks valid operating modes and power rows") from exc
    if not modes or len(modes) != len(loads) or len(modes) != len(battery):
        raise ValueError("episode mode and power rows must have the same positive length")
    if OperatingMode.UNRESOLVED in modes:
        raise ValueError("unresolved operating mode blocks v3 replay")
    if any(not isfinite(load) or not isfinite(power) for load, power in zip(loads, battery)):
        raise ValueError("episode power rows must be finite")
    if any(power > 0.0 for mode, power in zip(modes, battery) if mode in SHORE_MODES):
        raise ValueError("shore battery power must be a nonpositive charge request")
    return modes, loads, battery


def _ledger_total(ledgers: Sequence[RawCnyIntervalLedger]) -> RawCnyIntervalLedger:
    return RawCnyIntervalLedger(*(
        fsum(ledger.components_cny[index] for ledger in ledgers) for index in range(4)
    ))


def replay_episode(
    episode: object, policy: Callable[[tuple[float, ...]], MPCWeights], *,
    mpc: EconomicMPC, initial_state: AccountState | None = None,
) -> EpisodeReplay:
    """Route frozen mode rows before invoking a policy or MPC solve.

    The policy sees only the causal state prepared before the next ONBOARD
    measurement. Consecutive shore rows never create DQN actions.
    """
    modes, loads, battery_requests = _validated_rows(episode)
    if not callable(policy):
        raise TypeError("policy must be callable")
    controller = PredictiveController(mpc=mpc)
    if initial_state is not None:
        if type(initial_state) is not AccountState:
            raise TypeError("initial_state must be an AccountState")
        controller.state = initial_state
    start_soc = controller.state.soc
    transitions: list[DecisionTransition] = []
    shore_blocks: list[ShoreBlock] = []
    ledgers: list[RawCnyIntervalLedger] = []
    fc_trace: list[float | None] = [None] * len(modes)
    battery_trace: list[float | None] = [None] * len(modes)
    soc_trace: list[float | None] = [None] * len(modes)
    index = 0
    while index < len(modes):
        if modes[index] in SHORE_MODES:
            # Handles leading shore rows and shore-only episodes. Shore after an
            # ONBOARD run is settled with its preceding terminal transition.
            end = index
            while end < len(modes) and modes[end] in SHORE_MODES:
                end += 1
            start = controller.state.soc
            try:
                settlement = settle_shore_segment(mpc, controller.state, battery_requests[index:end])
            except (RuntimeError, ValueError) as exc:
                raise ReplayExecutionError(index, modes[index], exc) from exc
            controller.state = settlement.end_state
            block = ShoreBlock(
                index, end, start, settlement.end_state.soc, settlement.ledger,
                settlement.requested_battery_bus_kw,
                settlement.accepted_battery_bus_kw, settlement.soc_path,
            )
            shore_blocks.append(block)
            ledgers.append(settlement.ledger)
            for offset, (accepted, soc) in enumerate(zip(block.accepted_battery_bus_kw, block.soc_path)):
                fc_trace[index + offset] = 0.0
                battery_trace[index + offset] = accepted
                soc_trace[index + offset] = soc
            index = end
            continue

        # Only ONBOARD reaches this branch. The virtual zero-load boundary
        # precedes the first actual sample of every new voyage.
        controller.begin_voyage()
        while index < len(modes) and modes[index] is OperatingMode.ONBOARD:
            try:
                controller.start_decision(policy)
            except (RuntimeError, ValueError) as exc:
                raise ReplayExecutionError(index, modes[index], exc) from exc
            last_onboard = index + 1 == len(modes) or modes[index + 1] is not OperatingMode.ONBOARD
            if last_onboard and index + 1 < len(modes):
                shore_start = index + 1
                shore_end = shore_start
                while shore_end < len(modes) and modes[shore_end] in SHORE_MODES:
                    shore_end += 1
                try:
                    transition = controller.finish_terminal_decision(
                        loads[index], battery_requests[shore_start:shore_end],
                    )
                except (RuntimeError, ValueError) as exc:
                    raise ReplayExecutionError(index, modes[index], exc) from exc
                assert transition.shore_ledger is not None
                shore_blocks.append(ShoreBlock(
                    shore_start, shore_end, transition.executed.actual_soc,
                    controller.state.soc, transition.shore_ledger,
                    transition.shore_requested_battery_bus_kw,
                    transition.shore_accepted_battery_bus_kw,
                    transition.shore_soc_path,
                ))
                ledgers.append(transition.shore_ledger)
            else:
                try:
                    transition = controller.finish_decision(loads[index])
                except (RuntimeError, ValueError) as exc:
                    raise ReplayExecutionError(index, modes[index], exc) from exc
                if last_onboard:
                    transition = replace(transition, done=True)
            transitions.append(transition)
            ledgers.append(transition.executed.actual_ledger)
            fc_trace[index] = transition.executed.actual_fc_power_kw
            battery_trace[index] = transition.executed.actual_battery_power_kw
            soc_trace[index] = transition.executed.actual_soc
            if last_onboard and index + 1 < len(modes):
                block = shore_blocks[-1]
                for offset, (accepted, soc) in enumerate(zip(block.accepted_battery_bus_kw, block.soc_path)):
                    fc_trace[block.start_index + offset] = 0.0
                    battery_trace[block.start_index + offset] = accepted
                    soc_trace[block.start_index + offset] = soc
                index = block.end_index_exclusive
                break
            index += 1

    if any(value is None for trace in (fc_trace, battery_trace, soc_trace) for value in trace):
        raise RuntimeError("mode replay did not settle every physical row")
    return EpisodeReplay(
        str(episode.sample_id), str(episode.split), start_soc, controller.state,
        tuple(transitions), tuple(shore_blocks), _ledger_total(ledgers),
        tuple(float(value) for value in fc_trace),
        tuple(float(value) for value in battery_trace),
        tuple(float(value) for value in soc_trace),
        sum(mode is OperatingMode.ONBOARD for mode in modes),
        sum(mode in SHORE_MODES for mode in modes),
    )
