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
    soc_penalty_mode: str = "symmetric"

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.q_h2, self.q_batt, self.q_soc, self.q_fc_var)


DQN_MPC_WEIGHT_ACTIONS: tuple[MPCWeightAction, ...] = (
    MPCWeightAction(
        0,
        0.25,
        0.40,
        12.0,
        20.0,
        "nominal",
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
        0.45,
        200.0,
        8.0,
        "soc_regulation",
        "deficit_only",
    ),
    MPCWeightAction(
        3,
        0.15,
        0.80,
        12.0,
        8.0,
        "fast_fc_response",
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
