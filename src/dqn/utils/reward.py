"""Action-independent scoring of the physical step actually executed."""
from __future__ import annotations

import numpy as np
from mpc.solvers.fc_dp0_curve import h2_rate_gps_dp0
from utils.physical_config import (
    BATTERY_POWER_REF_KW, SOC_REFERENCE, SOC_SOFT_SCALE, FUEL_CELL_RAMP_KW_PER_S,
)

HYDROGEN_REFERENCE_POWER_KW = 600.0


def calculate_executed_reward(*, p_fc_kw: float, p_batt_kw: float,
                              soc_after: float, p_fc_prev_kw: float) -> tuple[float, dict]:
    """r=-(h + b**2/2 + s**2/2 + f**2/2), at Ts=T_sw=1 second.

    No action, objective value, predicted SOC or future plan is accepted.
    Uses the existing tabulated Dp0 physical curve, not the QP fit objective.
    """
    if not np.isfinite([p_fc_kw, p_batt_kw, soc_after, p_fc_prev_kw]).all():
        raise ValueError('executed reward inputs must be finite')
    rate = float(h2_rate_gps_dp0(p_fc_kw, p_rated_total_kw=HYDROGEN_REFERENCE_POWER_KW))
    reference = float(h2_rate_gps_dp0(HYDROGEN_REFERENCE_POWER_KW,
                                   p_rated_total_kw=HYDROGEN_REFERENCE_POWER_KW))
    h = rate / reference
    b = float(p_batt_kw) / BATTERY_POWER_REF_KW
    s = (float(soc_after) - SOC_REFERENCE) / SOC_SOFT_SCALE
    f = (float(p_fc_kw) - float(p_fc_prev_kw)) / FUEL_CELL_RAMP_KW_PER_S
    cost = h + .5 * b**2 + .5 * s**2 + .5 * f**2
    if not np.isfinite(cost):
        raise ValueError('executed reward cost must be finite')
    return -cost, {
        'method': 'executed_closed_loop_reward',
        'hydrogen': h, 'hydrogen_rate_g_s': rate, 'h_E_g_s': reference,
        'b': b, 's': s, 'f': f,
        'battery_power_stress_proxy': b**2,
        'soc_deviation_sq': s**2, 'fc_variation_sq': f**2,
        'stage_cost': cost, 'total_reward': -cost,
    }
