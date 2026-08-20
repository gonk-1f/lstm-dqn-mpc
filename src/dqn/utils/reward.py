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
REWARD_Q_LOW_SOC_DISCHARGE = 360.0
REWARD_Q_FAST_RISE_RESPONSE = 4.0
REWARD_Q_FC_VAR = 20.0

FUEL_CELL_MAX_KW = 600.0
BATTERY_POWER_REF_KW = 624.0
SOC_REFERENCE = 0.55
SOC_MIN = 0.20
SOC_BAND = 0.05
FC_VARIATION_REF_KW = 48.0
# TRAIN positive backward load-delta P99, P_load[t] - P_load[t-1].
LOAD_DELTA_RISE_REFERENCE_KW = 6.116713499697
DT_SECONDS = 1.0


def calculate_mpc_weight_reward(
    *,
    p_fc_kw: float,
    p_batt_kw: float,
    next_soc: float,
    previous_fc_kw: float,
    soc_before: float | None = None,
    load_delta_kw: float = 0.0,
) -> tuple[float, dict[str, Any]]:
    """
    Calculate the fixed six-term common reward for DQN-MPC
    objective-weight selection.

    The reward is

        r_t = -[
            0.25 * H2_norm
            + 0.40 * Batt_power_sq_norm
            + 12.0 * SOC_tracking_sq_norm
            + 360.0 * g_low * Low_SOC_discharge_sq_norm
            + 4.0 * g_rise * Fast_rise_response_sq_norm
            + 20.0 * (1 - g_rise) * (1 - g_low) * FC_variation_sq_norm
        ]

    where

        H2_norm
            = m_H2(P_fc,t) / m_H2(600 kW)

        Batt_power_sq_norm
            = (P_batt,t / 624 kW)^2

        SOC_tracking_sq_norm
            = ((SOC_t+1 - 0.55) / 0.05)^2

        Low_SOC_discharge_sq_norm
            = (max(P_batt,t, 0) / 624 kW)^2

        g_low
            = clip((0.55 - SOC_t) / (0.55 - 0.20), 0, 1)

        g_rise
            = clip(max(load_delta, 0) / 6.116713 kW, 0, 1)

        Fast_rise_response_sq_norm
            = (max(min(max(load_delta, 0), 48) - max(FC_delta, 0), 0) / 48)^2

        FC_variation_sq_norm
            = ((P_fc,t - P_fc,t-1) / 48 kW)^2

    This reward is a fixed evaluation criterion. The reward
    weights do not change when the DQN selects different MPC
    weight actions.

    Runtime environment and probe callers provide actual
    ``soc_before`` and observed backward ``load_delta_kw`` values. Legacy direct callers
    default to ``next_soc`` and a zero load delta.
    """

    soc_before_value = float(next_soc if soc_before is None else soc_before)
    load_delta = float(load_delta_kw)
    values = np.asarray(
        [
            p_fc_kw,
            p_batt_kw,
            next_soc,
            previous_fc_kw,
            soc_before_value,
            load_delta,
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

    low_soc_discharge_sq_norm = (
        max(p_batt, 0.0) / BATTERY_POWER_REF_KW
    ) ** 2

    g_low = float(
        np.clip(
            (SOC_REFERENCE - soc_before_value)
            / (SOC_REFERENCE - SOC_MIN),
            0.0,
            1.0,
        )
    )
    load_rise_kw = max(load_delta, 0.0)
    g_rise = float(
        np.clip(
            load_rise_kw / LOAD_DELTA_RISE_REFERENCE_KW,
            0.0,
            1.0,
        )
    )
    fc_delta_kw = p_fc - p_fc_previous
    desired_fc_rise_kw = min(load_rise_kw, FC_VARIATION_REF_KW)
    fast_rise_response_sq_norm = (
        max(desired_fc_rise_kw - max(fc_delta_kw, 0.0), 0.0)
        / FC_VARIATION_REF_KW
    ) ** 2
    fc_variation_sq_norm = (
        fc_delta_kw / FC_VARIATION_REF_KW
    ) ** 2

    weighted_h2 = REWARD_Q_H2 * h2_norm
    weighted_batt = REWARD_Q_BATT * battery_power_sq_norm
    weighted_soc = REWARD_Q_SOC * soc_tracking_sq_norm
    weighted_low_soc_discharge = (
        REWARD_Q_LOW_SOC_DISCHARGE
        * g_low
        * low_soc_discharge_sq_norm
    )
    weighted_fast_rise_response = (
        REWARD_Q_FAST_RISE_RESPONSE
        * g_rise
        * fast_rise_response_sq_norm
    )
    weighted_fc_var = (
        REWARD_Q_FC_VAR
        * (1.0 - g_rise)
        * (1.0 - g_low)
        * fc_variation_sq_norm
    )

    cost = (
        weighted_h2
        + weighted_batt
        + weighted_soc
        + weighted_low_soc_discharge
        + weighted_fast_rise_response
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
        "low_soc_discharge_sq_norm": float(
            low_soc_discharge_sq_norm
        ),
        "fast_rise_response_sq_norm": float(
            fast_rise_response_sq_norm
        ),
        "fc_variation_sq_norm": float(
            fc_variation_sq_norm
        ),
        "g_low": float(g_low),
        "g_rise": float(g_rise),
        "weighted_h2": float(weighted_h2),
        "weighted_batt": float(weighted_batt),
        "weighted_soc": float(weighted_soc),
        "weighted_low_soc_discharge": float(
            weighted_low_soc_discharge
        ),
        "weighted_fast_rise_response": float(
            weighted_fast_rise_response
        ),
        "weighted_fc_var": float(weighted_fc_var),
        "total_cost": float(cost),
        "total_reward": float(reward),
        "reward_weights": {
            "q_h2": REWARD_Q_H2,
            "q_batt": REWARD_Q_BATT,
            "q_soc": REWARD_Q_SOC,
            "q_low_soc_discharge": REWARD_Q_LOW_SOC_DISCHARGE,
            "q_fast_rise_response": REWARD_Q_FAST_RISE_RESPONSE,
            "q_fc_var": REWARD_Q_FC_VAR,
        },
    }

    return reward, info
