from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class TimeScaleConfig:
    """Independent lower-MPC and upper-DQN timing parameters.

    ``n_mpc`` is a prediction-horizon length. ``dqn_switch_steps`` is the
    count of real receding-horizon control periods for which an action is held.
    They deliberately remain distinct even when their provisional values match.
    """

    ts_mpc_seconds: float
    n_mpc: int
    dqn_switch_steps: int

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.ts_mpc_seconds)) or self.ts_mpc_seconds <= 0:
            raise ValueError("ts_mpc_seconds must be finite and positive")
        if type(self.n_mpc) is not int or self.n_mpc <= 0:
            raise ValueError("n_mpc must be a positive integer")
        if type(self.dqn_switch_steps) is not int or self.dqn_switch_steps <= 0:
            raise ValueError("dqn_switch_steps must be a positive integer")

    @classmethod
    def provisional(cls) -> "TimeScaleConfig":
        return cls(ts_mpc_seconds=30.0, n_mpc=5, dqn_switch_steps=5)

    @property
    def prediction_seconds(self) -> float:
        return float(self.ts_mpc_seconds) * self.n_mpc

    @property
    def switch_seconds(self) -> float:
        return float(self.ts_mpc_seconds) * self.dqn_switch_steps

    @property
    def mpc_solves_per_action(self) -> int:
        return self.dqn_switch_steps

    @property
    def n_mpc_semantics(self) -> str:
        return "future MPC prediction steps per rolling solve"

    @property
    def dqn_switch_steps_semantics(self) -> str:
        return "executed MPC control periods per selected DQN action"
