"""Actual shore-segment accounting with no DQN action or MPC solve."""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum
from typing import Sequence

from v2.economics import RawCnyIntervalLedger

from .control import AccountState, EconomicMPC


@dataclass(frozen=True)
class ShoreSettlement:
    end_state: AccountState
    ledger: RawCnyIntervalLedger
    steps: tuple[RawCnyIntervalLedger, ...]
    requested_battery_bus_kw: tuple[float, ...]
    accepted_battery_bus_kw: tuple[float, ...]
    soc_path: tuple[float, ...]


def settle_shore_segment(
    mpc: EconomicMPC, start_state: AccountState,
    requested_battery_bus_kw: Sequence[float],
) -> ShoreSettlement:
    """Limit logged charge requests and return one consolidated actual ledger."""
    powers = tuple(requested_battery_bus_kw)
    if not powers:
        raise ValueError("shore segment must contain at least one 30-second charge request")
    state = start_state
    steps: list[RawCnyIntervalLedger] = []
    accepted: list[float] = []
    soc_path: list[float] = []
    for power in powers:
        state, ledger, accepted_power = mpc.shore_interval(state, power)
        steps.append(ledger)
        accepted.append(accepted_power)
        soc_path.append(state.soc)
    consolidated = RawCnyIntervalLedger(*(
        fsum(item.components_cny[index] for item in steps) for index in range(4)
    ))
    return ShoreSettlement(state, consolidated, tuple(steps), powers, tuple(accepted), tuple(soc_path))
