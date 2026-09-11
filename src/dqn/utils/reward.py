from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def _validated_action_weights(
    action_weights: Sequence[float],
) -> np.ndarray:
    weights = np.asarray(action_weights, dtype=np.float64).reshape(-1)
    if weights.size != 4:
        raise ValueError("action_weights must contain exactly four values")
    if not np.all(np.isfinite(weights)):
        raise ValueError("action_weights must be finite")
    if np.any(weights < 0.0):
        raise ValueError("action_weights must be non-negative")
    if float(weights.sum()) <= 0.0:
        raise ValueError("action weight sum must be positive")
    return weights


def mpc_action_weight_sum(action_weights: Sequence[float]) -> float:
    """Return the positive sum of one selected MPC action's weights."""

    return float(_validated_action_weights(action_weights).sum())


def calculate_mpc_weight_reward(
    *,
    raw_mpc_objective: float,
    action_weights: Sequence[float],
) -> tuple[float, dict[str, Any]]:
    """Map the selected action's complete N-step MPC objective to reward.

    The MPC itself is solved with the original action weights. Only the
    returned objective value is normalized for the DQN reward:

        normalized_objective = raw_mpc_objective / sum(action_weights)
        reward = 1 / (1 + normalized_objective)
    """

    objective = float(raw_mpc_objective)
    if not np.isfinite(objective):
        raise ValueError("raw_mpc_objective must be finite")
    if objective < 0.0:
        raise ValueError("raw_mpc_objective must be non-negative")

    weight_sum = mpc_action_weight_sum(action_weights)
    normalized_objective = objective / weight_sum
    reward = 1.0 / (1.0 + normalized_objective)

    if not np.isfinite(reward) or not 0.0 < reward <= 1.0:
        raise RuntimeError("normalized MPC-objective reward is invalid")

    info: dict[str, Any] = {
        "raw_mpc_objective": objective,
        "weight_sum": weight_sum,
        "normalized_objective": float(normalized_objective),
        "total_reward": float(reward),
    }
    return float(reward), info
