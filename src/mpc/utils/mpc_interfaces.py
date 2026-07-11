from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MPCCommand:
    timestamp: str
    fuel_cell_ref_total_kw: float
    battery_ref_total_kw: float
    fuel_cell_ref_left_kw: float = 0.0
    fuel_cell_ref_right_kw: float = 0.0
    battery_ref_left_kw: float = 0.0
    battery_ref_right_kw: float = 0.0
    soc_target: float
    update_period_seconds: int = 30
    objective_info: dict = field(default_factory=dict)
    constraints: dict = field(default_factory=dict)


@dataclass
class MPCState:
    soc: float
    soc_left: float = 0.0
    soc_right: float = 0.0
    load_prediction_kw: list[float]
    speed_prediction_knots: list[float]
    previous_fuel_cell_kw: float
    timestamp: str = ""


@dataclass
class MultiRateSchedule:
    mpc_update_seconds: int
    dqn_control_seconds: int
    dqn_steps_per_mpc: int
