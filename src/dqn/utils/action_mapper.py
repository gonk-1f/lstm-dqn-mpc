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
    MPCWeightAction(
        0,
        0.20,
        0.50,
        40.0,
        16.0,
        "balanced",
    ),
    MPCWeightAction(
        1,
        0.40,
        0.25,
        8.0,
        8.0,
        "hydrogen_economy",
    ),
    MPCWeightAction(
        2,
        0.25,
        0.50,
        30.0,
        40.0,
        "fc_smoothing",
    ),
    MPCWeightAction(
        3,
        0.15,
        0.80,
        120.0,
        8.0,
        "soc_protection",
    ),
)


def get_weight_action(action_id: int) -> MPCWeightAction:
    max_action_id = len(DQN_MPC_WEIGHT_ACTIONS) - 1

    if type(action_id) is not int:
        raise ValueError(
            f"Invalid action_id {action_id!r}; "
            f"expected an integer from 0 to {max_action_id}."
        )

    if action_id < 0 or action_id >= len(DQN_MPC_WEIGHT_ACTIONS):
        raise IndexError(
            f"Invalid action_id {action_id}; "
            f"expected 0 to {max_action_id}."
        )

    return DQN_MPC_WEIGHT_ACTIONS[action_id]
