from __future__ import annotations

from .mpc_qp_formulation import QpMpcConfig
from utils.physical_config import (
    MPC_HORIZON as N6_HORIZON, DT_SECONDS as N6_DT_SECONDS,
    SOC_REFERENCE as FIXED_SOC_REFERENCE, SOC_SOFT_MIN, SOC_SOFT_MAX,
    SOC_SOFT_SCALE,
    FUEL_CELL_MIN_KW, FUEL_CELL_MAX_KW,
    FUEL_CELL_RAMP_KW_PER_S, BATTERY_CAPACITY_KWH, BATTERY_CHARGE_MAX_KW,
    BATTERY_DISCHARGE_MAX_KW, BATTERY_POWER_REF_KW, SOC_MIN, SOC_MAX,
)


OBJECTIVE_VARIANT = "n6_h2_batt_socref_fcvar_normalized_v2"

N6_STATE_COMMIT_TOLERANCES: dict[str, float] = {
    "actual_balance_kw": 0.01,
    "qp_balance_kw": 0.1,
    "power_bound_kw": 0.1,
    "ramp_kw": 0.1,
    "soc": 1.0e-5,
    "soc_prediction": 1.0e-5,
}


def build_formal_mpc_config() -> QpMpcConfig:
    """Return the frozen physical N=6 MPC configuration.

    The solver bank applies only the selected action weights to this common
    physical configuration and shared continuous SOC reference objective.
    """

    return QpMpcConfig(
        horizon=N6_HORIZON,
        dt_seconds=N6_DT_SECONDS,
        battery_capacity_kwh=BATTERY_CAPACITY_KWH,
        battery_charge_max_kw=BATTERY_CHARGE_MAX_KW,
        battery_discharge_max_kw=BATTERY_DISCHARGE_MAX_KW,
        battery_power_ref_kw=BATTERY_POWER_REF_KW,
        fuel_cell_min_kw=FUEL_CELL_MIN_KW,
        fuel_cell_max_kw=FUEL_CELL_MAX_KW,
        fuel_cell_ramp_rate_kw_per_s=FUEL_CELL_RAMP_KW_PER_S,
        fuel_cell_ramp_kw=None,
        soc_min=SOC_MIN,
        soc_max=SOC_MAX,
        soc_reference=FIXED_SOC_REFERENCE,
        soc_scale=SOC_SOFT_SCALE,
        objective_variant=OBJECTIVE_VARIANT,
        q_h2=0.25,
        q_fc_var=0.25,
        q_soc=0.25,
        q_batt=0.25,
        q_ramp=0.0,
        q_terminal_soc=0.0,
    )
