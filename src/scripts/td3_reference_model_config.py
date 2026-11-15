"""Environment settings for the SimulationCode comparison."""

import numpy as np

from td3_config import PROJECT_ROOT
from ship_ems import reference_model as source


MODEL_VERSION = "simulationcode_ship_no_fc_dynamics_v1"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "td3_no_fc_dynamics"
SOC_WEIGHT = 10.0
FAILURE_COST_PER_REMAINING_STEP = 2.0
NORMAL_STEP_COST_UPPER_BOUND = source.maximum_step_cost(SOC_WEIGHT)

MODEL_PARAMETERS = {
    "fc_rated_kw": source.FC_RATED_KW,
    "fc_minimum_output_kw": source.FC_MIN_KW,
    "fc_degradation_normalization_kw": source.FC_NORMALIZATION_KW,
    "fc_running_threshold_kw": source.FC_RUNNING_THRESHOLD_KW,
    "fc_low_load_threshold_kw": source.FC_LOW_LOAD_FRACTION * source.FC_RATED_KW,
    "fc_high_load_threshold_kw": source.FC_HIGH_LOAD_FRACTION * source.FC_RATED_KW,
    "fc_startup_uv": source.FC_STARTUP_UV,
    "fc_low_uv_per_hour": source.FC_LOW_UV_PER_HOUR,
    "fc_high_uv_per_hour": source.FC_HIGH_UV_PER_HOUR,
    "fc_transient_uv_per_normalized_unit": source.FC_TRANSIENT_UV_PER_NORMALIZED_UNIT,
    "fc_eol_uv": source.FC_EOL_UV,
    "fc_step_cost_cap": source.FC_STEP_COST_CAP,
    "fc_replacement_cost_per_kw": source.FC_REPLACEMENT_COST_PER_KW,
    "battery_capacity_kwh": source.BATTERY_KWH,
    "battery_voltage_v": source.BATTERY_VOLTAGE_V,
    "battery_max_charge_kw": source.BATTERY_CHARGE_KW,
    "battery_max_discharge_kw": source.BATTERY_DISCHARGE_KW,
    "battery_cost_per_qloss": source.BATTERY_COST_PER_QLOSS,
    "soc_target": source.SOC_TARGET,
    "soc_initial": source.SOC_INITIAL,
    "soc_boundary_range": (source.SOC_MIN, source.SOC_MAX),
    "soc_boundary_cost": source.SOC_BOUNDARY_COST,
    "hydrogen_lhv_kj_per_kg": source.LHV_KJ_PER_KG,
    "hydrogen_cost_per_kg": source.HYDROGEN_COST_PER_KG,
    "cost_unit": "SimulationCode cost unit (currency unspecified)",
    "failure_cost_per_remaining_step": FAILURE_COST_PER_REMAINING_STEP,
    "normal_step_cost_upper_bound": NORMAL_STEP_COST_UPPER_BOUND,
    "failure_cost_exceeds_normal_step_cost_bound": (
        FAILURE_COST_PER_REMAINING_STEP > NORMAL_STEP_COST_UPPER_BOUND
    ),
}


def load_values(profile) -> np.ndarray:
    """Extract the unchanged 1 s load column from a project voyage frame."""
    if hasattr(profile, "columns"):
        profile = profile["load_total_kw"]
    return np.asarray(profile, dtype=float)


def observation_scales(train_profiles) -> tuple[float, float]:
    """Use common normalization from the unchanged Train35 split."""
    profiles = [load_values(profile) for profile in train_profiles]
    return (
        max(float(np.max(profile)) for profile in profiles) or 1.0,
        max(float(np.max(np.abs(np.diff(profile)))) for profile in profiles) or 1.0,
    )


def make_env(
    load_profiles, sample_ids, *, load_scale_kw: float, load_delta_scale_kw: float,
    train_window_steps: int | None = None,
) -> source.SimulationCodeShipEnv:
    return source.SimulationCodeShipEnv(
        [load_values(profile) for profile in load_profiles], sample_ids, load_scale_kw=load_scale_kw,
        load_delta_scale_kw=load_delta_scale_kw, train_window_steps=train_window_steps,
        soc_weight=SOC_WEIGHT, failure_cost_per_remaining_step=FAILURE_COST_PER_REMAINING_STEP,
    )
