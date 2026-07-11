from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MultiRateConfig:
    base_sample_seconds: int = 60
    mpc_update_seconds: int = 60
    dqn_control_seconds: int = 5

    @property
    def dqn_steps_per_base_sample(self) -> int:
        return max(1, int(round(self.base_sample_seconds / self.dqn_control_seconds)))

    @property
    def dqn_steps_per_mpc(self) -> int:
        return max(1, int(round(self.mpc_update_seconds / self.dqn_control_seconds)))


def expand_multirate_dataset(dataset: pd.DataFrame, config: MultiRateConfig) -> pd.DataFrame:
    """Expand coarse processed data to a faster DQN control scale.

    The vessel measurements are treated as piecewise-constant over each base sample.
    MPC references are refreshed every `mpc_update_seconds` and held between updates.
    """

    df = dataset.copy().reset_index(drop=True)
    repeat = config.dqn_steps_per_base_sample
    expanded = df.loc[df.index.repeat(repeat)].reset_index(drop=True)
    expanded["base_index"] = np.repeat(np.arange(len(df)), repeat)
    expanded["dqn_substep"] = np.tile(np.arange(repeat), len(df))
    expanded["dqn_time_seconds"] = (
        expanded["base_index"] * config.base_sample_seconds + expanded["dqn_substep"] * config.dqn_control_seconds
    )

    step_per_mpc = config.dqn_steps_per_mpc
    expanded["mpc_refresh_flag"] = (expanded["dqn_time_seconds"] // config.dqn_control_seconds) % step_per_mpc == 0

    held_reference_columns = [
        "mpc_fuel_cell_ref_kw",
        "mpc_battery_ref_kw",
        "mpc_soc_ref",
        "mpc_fuel_cell_ref_left_kw",
        "mpc_fuel_cell_ref_right_kw",
        "mpc_battery_ref_left_kw",
        "mpc_battery_ref_right_kw",
        "mpc_objective_value",
        "mpc_fuel_cost_norm_mean",
        "mpc_fc_smooth_norm_mean",
        "mpc_soc_dev_norm_mean",
        "mpc_battery_use_norm_mean",
        "mpc_terminal_soc_norm",
        "mpc_fuel_cost_term",
        "mpc_fc_smooth_term",
        "mpc_soc_dev_term",
        "mpc_battery_use_term",
        "mpc_terminal_soc_term",
        "mpc_total_cost",
        "mpc_enable_battery_use_in_mpc",
        "mpc_success",
        "mpc_solver_message",
        "mpc_solve_stride",
        "mpc_source",
    ]
    for column in held_reference_columns:
        if column in expanded.columns:
            expanded[column] = expanded[column].where(expanded["mpc_refresh_flag"]).ffill()

    expanded["dqn_control_seconds"] = config.dqn_control_seconds
    expanded["mpc_update_seconds"] = config.mpc_update_seconds
    return expanded


def summarize_multirate_feasibility(config: MultiRateConfig) -> dict:
    return {
        "base_sample_seconds": config.base_sample_seconds,
        "mpc_update_seconds": config.mpc_update_seconds,
        "dqn_control_seconds": config.dqn_control_seconds,
        "dqn_steps_per_base_sample": config.dqn_steps_per_base_sample,
        "dqn_steps_per_mpc": config.dqn_steps_per_mpc,
    }
