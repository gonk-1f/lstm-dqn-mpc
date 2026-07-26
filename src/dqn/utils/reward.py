from __future__ import annotations

from typing import Any

import numpy as np

from mpc.solvers.fc_dp0_curve import h2_kg_step_dp0_quadratic


# Fixed DQN evaluation weights.
#
# These are the Candidate C weights used only as the common
# performance criterion for DQN training. They do NOT change
# with the selected MPC action A0-A6.
REWARD_Q_H2 = 0.25
REWARD_Q_BATT = 0.40
REWARD_Q_SOC = 12.0
REWARD_Q_FC_VAR = 20.0

FUEL_CELL_MAX_KW = 560.0
BATTERY_POWER_REF_KW = 624.0
SOC_REFERENCE = 0.55
SOC_BAND = 0.05
FC_VARIATION_REF_KW = 48.0
DT_SECONDS = 1.0


def calculate_mpc_weight_reward(
    *,
    p_fc_kw: float,
    p_batt_kw: float,
    next_soc: float,
    previous_fc_kw: float,
) -> tuple[float, dict[str, Any]]:
    """
    Calculate the fixed four-objective reward for DQN-MPC
    objective-weight selection.

    The reward is

        r_t = -[
            0.25 * H2_norm
            + 0.40 * Batt_power_sq_norm
            + 12.0 * SOC_tracking_sq_norm
            + 20.0 * FC_variation_sq_norm
        ]

    where

        H2_norm
            = m_H2(P_fc,t) / m_H2(560 kW)

        Batt_power_sq_norm
            = (P_batt,t / 624 kW)^2

        SOC_tracking_sq_norm
            = ((SOC_t+1 - 0.55) / 0.05)^2

        FC_variation_sq_norm
            = ((P_fc,t - P_fc,t-1) / 48 kW)^2

    This reward is a fixed evaluation criterion. The four reward
    weights do not change when the DQN selects different MPC
    weight actions.

    No additional SOC guard, battery-direction term, action-switch
    penalty, terminal penalty, or constraint penalty is included.
    """

    values = np.asarray(
        [
            p_fc_kw,
            p_batt_kw,
            next_soc,
            previous_fc_kw,
        ],
        dtype=np.float64,
    )

    if not np.all(np.isfinite(values)):
        raise ValueError("reward inputs must all be finite")

    p_fc = float(p_fc_kw)
    p_batt = float(p_batt_kw)
    soc_next = float(next_soc)
    p_fc_previous = float(previous_fc_kw)

    # Use the same project Dp0 quadratic hydrogen model as the MPC.
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

    soc_tracking_sq_norm = (
        (soc_next - SOC_REFERENCE) / SOC_BAND
    ) ** 2

    fc_variation_sq_norm = (
        (p_fc - p_fc_previous) / FC_VARIATION_REF_KW
    ) ** 2

    weighted_h2 = REWARD_Q_H2 * h2_norm
    weighted_batt = REWARD_Q_BATT * battery_power_sq_norm
    weighted_soc = REWARD_Q_SOC * soc_tracking_sq_norm
    weighted_fc_var = (
        REWARD_Q_FC_VAR * fc_variation_sq_norm
    )

    cost = (
        weighted_h2
        + weighted_batt
        + weighted_soc
        + weighted_fc_var
    )

    reward = -float(cost)

    info: dict[str, Any] = {
        "h2_norm": float(h2_norm),
        "battery_power_sq_norm": float(
            battery_power_sq_norm
        ),
        "soc_tracking_sq_norm": float(
            soc_tracking_sq_norm
        ),
        "fc_variation_sq_norm": float(
            fc_variation_sq_norm
        ),
        "weighted_h2": float(weighted_h2),
        "weighted_batt": float(weighted_batt),
        "weighted_soc": float(weighted_soc),
        "weighted_fc_var": float(weighted_fc_var),
        "total_cost": float(cost),
        "total_reward": float(reward),
        "reward_weights": {
            "q_h2": REWARD_Q_H2,
            "q_batt": REWARD_Q_BATT,
            "q_soc": REWARD_Q_SOC,
            "q_fc_var": REWARD_Q_FC_VAR,
        },
    }

    return reward, info