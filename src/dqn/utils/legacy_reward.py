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


def legacy_yuan_like_self_cost(
    *,
    raw_mpc_objective: float,
) -> tuple[float, dict[str, Any]]:
    """Map the selected action's complete N-step MPC objective to reward.

    Historical project baseline, not an exact reproduction of Yuan Eq. 51.
    Action-dependent objectives are not a common physical scoring ruler:

        reward = 1 / (1 + raw_mpc_objective)
    """

    objective = float(raw_mpc_objective)
    if not np.isfinite(objective):
        raise ValueError("raw_mpc_objective must be finite")
    if objective < 0.0:
        raise ValueError("raw_mpc_objective must be non-negative")

    reward = 1.0 / (1.0 + objective)

    if not np.isfinite(reward) or not 0.0 < reward <= 1.0:
        raise RuntimeError("MPC-objective reward is invalid")

    info: dict[str, Any] = {
        "raw_mpc_objective": objective,
        "total_reward": float(reward),
    }
    return float(reward), info


def legacy_predicted_common_reward(terms, *, horizon: int = 6):
    """Frozen RMS/L2: 1/(1+sqrt((H/N)^2+B/N+S/N+F/N)).

    Input order H/B/S/F matches unweighted predicted N-step MPC terms.
    Source: audit_rms_l2_84_train.py. H is squared only in this legacy method.
    """
    values = np.asarray(terms, dtype=float)
    if values.shape[-1:] != (4,) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('four finite nonnegative predicted terms required')
    if horizon != 6:
        raise ValueError('historical baseline is fixed at N=6')
    v = values / horizon
    return 1. / (1. + np.sqrt(v[..., 0]**2 + v[..., 1:].sum(axis=-1)))
