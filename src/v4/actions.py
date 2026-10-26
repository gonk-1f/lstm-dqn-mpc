"""Action identities; policy APIs continue to exchange executed FC power."""

from __future__ import annotations

import math
from typing import Sequence


FC_ACTION_MODES = ('absolute61', 'increment7', 'increment13')
FC_INCREMENT_KW = (-60, -30, -10, 0, 10, 30, 60)
FC_INCREMENT_13_KW = tuple(range(-60, 61, 10))
FC_ABSOLUTE_KW = tuple(range(0, 601, 10))


def action_values_kw(action_mode: str = 'absolute61') -> tuple[int, ...]:
    if action_mode not in FC_ACTION_MODES:
        raise ValueError(f'action_mode must be one of {FC_ACTION_MODES}')
    if action_mode == 'increment13':
        return FC_INCREMENT_13_KW
    return FC_INCREMENT_KW if action_mode == 'increment7' else FC_ABSOLUTE_KW


def previous_grid_power_kw(previous_fc_kw: float) -> int:
    """Recover a grid power from the observable normalized state, without clipping."""
    if not math.isfinite(previous_fc_kw):
        raise ValueError('increment action requires finite previous FC power')
    power = int(round(previous_fc_kw / 10)) * 10
    if power not in FC_ABSOLUTE_KW or not math.isclose(power, previous_fc_kw, abs_tol=1e-4):
        raise ValueError('increment action requires previous FC power on the physical grid')
    return power


def candidate_fc_actions(physical_actions: Sequence[int], previous_fc_kw: float,
                         action_mode: str = 'absolute61') -> tuple[int, ...]:
    """Intersect already physical-feasible powers with the incremental action set.

    Empty intersections remain empty. There is no clipping and no replacement
    with absolute controls when an incremental action cannot be executed.
    """
    values = action_values_kw(action_mode)
    physical = tuple(physical_actions)
    if (any(type(power) is not int or power not in FC_ABSOLUTE_KW for power in physical)
            or len(set(physical)) != len(physical)):
        raise ValueError('physical actions must be distinct powers on the 0-600 kW grid')
    if action_mode == 'absolute61':
        return physical
    previous = previous_grid_power_kw(previous_fc_kw)
    return tuple(power for power in physical if power - previous in values)
