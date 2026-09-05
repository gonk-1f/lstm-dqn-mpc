from __future__ import annotations

from .mpc_qp_formulation import QpMpcConfig


OBJECTIVE_VARIANT = "n6_h2_batt_soc_fcvar_normalized_v1"
N6_HORIZON = 6
N6_DT_SECONDS = 1.0
FIXED_SOC_REFERENCE = 0.55
SOC_SOFT_MIN = 0.50
SOC_SOFT_MAX = 0.60
SOC_SOFT_SCALE = 0.05

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
    physical configuration and shared SOC soft working range.
    """

    return QpMpcConfig(
        horizon=N6_HORIZON,
        dt_seconds=N6_DT_SECONDS,
        battery_capacity_kwh=624.0,
        battery_charge_max_kw=624.0,
        battery_discharge_max_kw=1248.0,
        battery_power_ref_kw=624.0,
        fuel_cell_min_kw=0.0,
        fuel_cell_max_kw=600.0,
        fuel_cell_ramp_rate_kw_per_s=48.0,
        fuel_cell_ramp_kw=None,
        soc_min=0.2,
        soc_max=0.8,
        soc_soft_min=SOC_SOFT_MIN,
        soc_soft_max=SOC_SOFT_MAX,
        soc_band=SOC_SOFT_SCALE,
        objective_variant=OBJECTIVE_VARIANT,
        q_h2=0.25,
        q_fc_var=20.0,
        q_soc=12.0,
        q_batt=0.40,
        q_ramp=0.0,
        q_terminal_soc=0.0,
    )
