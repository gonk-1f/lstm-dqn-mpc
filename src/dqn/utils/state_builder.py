from __future__ import annotations

from typing import Sequence

import numpy as np
from utils.physical_config import (
    MPC_HORIZON as DQN_MPC_PREVIEW_STEPS, SOC_REFERENCE,
    SOC_SOFT_SCALE as SOC_SCALE, FUEL_CELL_MAX_KW as FUEL_CELL_POWER_SCALE_KW,
    BATTERY_POWER_REF_KW as BATTERY_POWER_SCALE_KW,
    FUEL_CELL_MAX_KW as LOAD_POWER_SCALE_KW,
    FUEL_CELL_RAMP_KW_PER_S as LOAD_DELTA_SCALE_KW,
)


DQN_MPC_STATE_DIM = 7


def build_dqn_mpc_state(
    *,
    current_soc: float,
    previous_fc_kw: float,
    previous_batt_kw: float,
    load_history_kw: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Build the normalized causal 7-dimensional DQN-MPC state.

    ``load_history_kw`` contains current-voyage samples through time t.
    State order: current SOC, previous FC, previous battery, current load,
    backward load delta, latest 10 s mean load, latest 60 s mean load.
    """

    history = np.asarray(
        load_history_kw,
        dtype=np.float64,
    ).reshape(-1)
    if history.size == 0:
        raise ValueError(
            "load_history_kw must contain at least one value"
        )

    scalar_values = np.asarray(
        [
            current_soc,
            previous_fc_kw,
            previous_batt_kw,
        ],
        dtype=np.float64,
    )

    if not np.all(np.isfinite(scalar_values)):
        raise ValueError("state scalar inputs must all be finite")

    if not np.all(np.isfinite(history)):
        raise ValueError(
            "load_history_kw must contain only finite values"
        )

    current_load_kw = float(history[-1])
    previous_load_kw = (
        float(history[-2]) if history.size >= 2 else current_load_kw
    )
    delta_load_kw = current_load_kw - previous_load_kw
    mean_load_10s_kw = float(np.mean(history[-10:]))
    mean_load_60s_kw = float(np.mean(history[-60:]))

    state = np.asarray(
        [
            (float(current_soc) - SOC_REFERENCE) / SOC_SCALE,
            float(previous_fc_kw) / FUEL_CELL_POWER_SCALE_KW,
            float(previous_batt_kw) / BATTERY_POWER_SCALE_KW,
            current_load_kw / LOAD_POWER_SCALE_KW,
            delta_load_kw / LOAD_DELTA_SCALE_KW,
            mean_load_10s_kw / LOAD_POWER_SCALE_KW,
            mean_load_60s_kw / LOAD_POWER_SCALE_KW,
        ],
        dtype=np.float64,
    )

    if state.shape != (DQN_MPC_STATE_DIM,):
        raise RuntimeError(
            f"unexpected DQN state shape: {state.shape}"
        )

    return state.astype(np.float32, copy=False)
