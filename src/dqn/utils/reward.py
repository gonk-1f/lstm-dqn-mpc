from __future__ import annotations

from typing import Any

import numpy as np

from mpc.solvers.fc_dp0_curve import h2_kg_step_dp0_quadratic


# Fixed common-reward weights. These evaluate every MPC action with
# the same criterion and do not change the A0-A3 MPC objectives.
REWARD_Q_H2 = 0.25
REWARD_Q_BATT = 0.40
REWARD_Q_FC_VAR = 20.0

FUEL_CELL_MAX_KW = 600.0
BATTERY_POWER_REF_KW = 624.0
SOC_REFERENCE = 0.55
SOC_SOFT_MIN = 0.50
SOC_SOFT_MAX = 0.60
SOC_SOFT_SCALE = 0.05
FC_VARIATION_REF_KW = 48.0
DT_SECONDS = 1.0


def soc_soft_working_range_penalty(next_soc: float) -> float:
    """Return the squared distance outside the closed SOC soft range."""

    soc = float(next_soc)
    if not np.isfinite(soc):
        raise ValueError("next_soc must be finite")
    if soc < SOC_SOFT_MIN:
        return float(((SOC_SOFT_MIN - soc) / SOC_SOFT_SCALE) ** 2)
    if soc > SOC_SOFT_MAX:
        return float(((soc - SOC_SOFT_MAX) / SOC_SOFT_SCALE) ** 2)
    return 0.0


def calculate_mpc_weight_reward(
    *,
    p_fc_kw: float,
    p_batt_kw: float,
    next_soc: float,
    previous_fc_kw: float,
) -> tuple[float, dict[str, Any]]:
    """Calculate the four-term common reward for DQN-MPC actions.

    The fixed evaluation criterion shared by MLP and KAN is

        r_t = -(
            0.25 * H_t
            + 0.40 * B_t
            + Phi_SOC(SOC_t+1)
            + 20.0 * F_t
        )

    where H_t is Dp0 hydrogen consumption normalized at 600 kW,
    B_t = (P_batt,t / 624)^2, and
    F_t = ((P_fc,t - P_fc,t-1) / 48)^2. Phi_SOC is zero throughout
    the closed soft working range [0.50, 0.60] and grows quadratically
    with distance outside it, normalized by 0.05. Its coefficient is 1.0.
    """

    values = np.asarray(
        [p_fc_kw, p_batt_kw, next_soc, previous_fc_kw],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("reward inputs must all be finite")

    p_fc = float(p_fc_kw)
    p_batt = float(p_batt_kw)
    soc_next = float(next_soc)
    p_fc_previous = float(previous_fc_kw)

    h2_kg_step = float(
        np.asarray(
            h2_kg_step_dp0_quadratic(
                p_fc,
                dt_seconds=DT_SECONDS,
                p_rated_total_kw=FUEL_CELL_MAX_KW,
            )
        )
    )
    h2_reference_kg_step = float(
        np.asarray(
            h2_kg_step_dp0_quadratic(
                FUEL_CELL_MAX_KW,
                dt_seconds=DT_SECONDS,
                p_rated_total_kw=FUEL_CELL_MAX_KW,
            )
        )
    )
    if h2_reference_kg_step <= 0.0:
        raise RuntimeError(
            "hydrogen normalization reference must be positive"
        )

    h2_norm = h2_kg_step / h2_reference_kg_step
    battery_power_sq_norm = (
        p_batt / BATTERY_POWER_REF_KW
    ) ** 2
    phi_soc = soc_soft_working_range_penalty(soc_next)
    fc_variation_sq_norm = (
        (p_fc - p_fc_previous) / FC_VARIATION_REF_KW
    ) ** 2

    weighted_h2 = REWARD_Q_H2 * h2_norm
    weighted_batt = REWARD_Q_BATT * battery_power_sq_norm
    weighted_soc = phi_soc
    weighted_fc_var = REWARD_Q_FC_VAR * fc_variation_sq_norm
    cost = (
        weighted_h2
        + weighted_batt
        + weighted_soc
        + weighted_fc_var
    )
    reward = -float(cost)

    info: dict[str, Any] = {
        "h2_norm": float(h2_norm),
        "battery_power_sq_norm": float(battery_power_sq_norm),
        "phi_soc": float(phi_soc),
        "fc_variation_sq_norm": float(fc_variation_sq_norm),
        "weighted_h2": float(weighted_h2),
        "weighted_batt": float(weighted_batt),
        "weighted_soc": float(weighted_soc),
        "weighted_fc_var": float(weighted_fc_var),
        "total_cost": float(cost),
        "total_reward": float(reward),
    }
    return reward, info
