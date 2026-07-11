from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DQNObservation:
    load_kw: float
    speed_knots: float
    soc: float
    fuel_cell_power_kw: float
    battery_power_kw: float
    fuel_cell_ref_kw: float
    battery_ref_kw: float
    tracking_error_kw: float


@dataclass
class DQNTransition:
    """One replay-buffer sample for ship lower-layer Q-learning.

    Physical meaning:
    - state: current ship microgrid operating condition
    - action: discrete power-allocation decision chosen by the lower layer
    - reward: immediate control quality under the chosen action
    - next_state: operating condition after the environment transition
    - done: whether the episode terminates after this transition
    """

    state: Any
    action: int
    reward: float
    next_state: Any
    done: bool


@dataclass
class ControlAction:
    action_id: int
    fuel_cell_power_kw: float
    battery_power_kw: float
    split_ratio: dict = field(default_factory=dict)
