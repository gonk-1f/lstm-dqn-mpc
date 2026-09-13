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


LEGACY_FOUR_WEIGHT_ACTIONS: tuple[MPCWeightAction, ...] = (
    MPCWeightAction(
        0,
        0.05,
        0.15,
        0.70,
        0.10,
        "balanced",
    ),
    MPCWeightAction(
        1,
        0.10,
        0.25,
        0.55,
        0.10,
        "hydrogen_economy",
    ),
    MPCWeightAction(
        2,
        0.05,
        0.10,
        0.60,
        0.25,
        "fc_smoothing",
    ),
    MPCWeightAction(
        3,
        0.05,
        0.20,
        0.70,
        0.05,
        "soc_protection",
    ),
)


ACTION_TABLE_VERSION = "positive_integer_simplex_10_lexicographic_v1"
REWARD_METHOD = "executed_closed_loop_reward"
REWARD_VERSION = "dp0_executed_linear_h_half_quadratics_v1"


def _integer_weight_compositions() -> tuple[tuple[int, int, int, int], ...]:
    return tuple(
        (h, b, s, 10 - h - b - s)
        for h in range(1, 8)
        for b in range(1, 9 - h)
        for s in range(1, 10 - h - b)
    )


DQN_MPC_WEIGHT_ACTIONS = tuple(
    MPCWeightAction(i, *(n / 10 for n in row), f"grid_{i:02d}")
    for i, row in enumerate(_integer_weight_compositions())
)


def control_semantics() -> dict:
    """Version action IDs and reward together for models and replay."""
    return {
        "action_table_version": ACTION_TABLE_VERSION,
        "reward_method": REWARD_METHOD,
        "reward_version": REWARD_VERSION,
        "switch_seconds": 1.0,
        "actions": [
            {"action_id": a.action_id, "weights": list(a.as_tuple()),
             "integer_weights": list(row)}
            for a, row in zip(DQN_MPC_WEIGHT_ACTIONS, _integer_weight_compositions())
        ],
    }


def require_control_semantics(value: object) -> None:
    if value != control_semantics():
        raise ValueError("incompatible checkpoint/replay action table or reward semantics; "
                         "legacy four-action artifacts cannot resume the 84-action method")


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
