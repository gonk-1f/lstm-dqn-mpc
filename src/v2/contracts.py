from __future__ import annotations

from .config import TimeScaleConfig

METHOD_VERSION = "multiscale_dqn_wmpc_v2"
MPC_OBJECTIVE_VERSION = "fc_base_smooth_soc_deadband_v1"
ACTION_TABLE_VERSION = "three_weight_simplex_behavior_filtered_v1"
REWARD_VERSION = "macro_interval_real_economic_cost_v1"
FC_ENERGY_VERSION = "eta_fc_lhv_h2_v1"
FC_DEGRADATION_VERSION = "aggregate_four_condition_voltage_loss_v1"
BATTERY_DEGRADATION_VERSION = "soc_current_weighted_throughput_v1"
DATASET_VERSION = "mode_aware_operating_cycle_v2"


class IncompatibleArtifactError(ValueError):
    """Raised before any v1 or mismatched artifact can be resumed as v2."""


def control_semantics(
    timescale: TimeScaleConfig | None = None,
) -> dict[str, object]:
    scale = timescale or TimeScaleConfig.provisional()
    return {
        "method_version": METHOD_VERSION,
        "mpc_objective_version": MPC_OBJECTIVE_VERSION,
        "action_table_version": ACTION_TABLE_VERSION,
        "reward_version": REWARD_VERSION,
        "fc_energy_version": FC_ENERGY_VERSION,
        "fc_degradation_version": FC_DEGRADATION_VERSION,
        "battery_degradation_version": BATTERY_DEGRADATION_VERSION,
        "dataset_version": DATASET_VERSION,
        "ts_mpc_seconds": scale.ts_mpc_seconds,
        "n_mpc": scale.n_mpc,
        "dqn_switch_steps": scale.dqn_switch_steps,
        "prediction_seconds": scale.prediction_seconds,
        "switch_seconds": scale.switch_seconds,
    }


def require_v2_semantics(
    value: object,
    timescale: TimeScaleConfig | None = None,
) -> None:
    if value != control_semantics(timescale):
        raise IncompatibleArtifactError(
            "checkpoint/replay semantics are not exactly compatible with "
            f"{METHOD_VERSION}; v1 artifacts cannot be resumed"
        )
