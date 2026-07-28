from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MPCWeightAction:
    action_id: int
    q_h2: float
    q_batt: float
    q_soc: float
    q_fc_var: float
    name: str

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.q_h2, self.q_batt, self.q_soc, self.q_fc_var)


DQN_MPC_WEIGHT_ACTIONS: tuple[MPCWeightAction, ...] = (
    MPCWeightAction(0, 0.25, 0.40, 12.0, 20.0, "candidate_C"),
    MPCWeightAction(1, 0.60, 0.15, 4.0, 2.0, "hydrogen_economy"),
    MPCWeightAction(2, 0.25, 0.50, 20.0, 12.0, "balanced"),
    MPCWeightAction(3, 0.20, 0.45, 28.0, 18.0, "soc_maintenance"),
    MPCWeightAction(4, 0.30, 0.45, 50.0, 18.0, "strong_soc_recovery"),
    MPCWeightAction(5, 0.15, 0.80, 12.0, 1.0, "fast_fc_response"),
    MPCWeightAction(6, 0.15, 0.15, 8.0, 50.0, "fc_smoothing"),
)


def get_weight_action(action_id: int) -> MPCWeightAction:
    if type(action_id) is not int:
        raise ValueError(f"Invalid action_id {action_id!r}; expected an integer from 0 to 6.")
    if action_id < 0 or action_id >= len(DQN_MPC_WEIGHT_ACTIONS):
        raise IndexError(f"Invalid action_id {action_id}; expected 0 to 6.")
    return DQN_MPC_WEIGHT_ACTIONS[action_id]
