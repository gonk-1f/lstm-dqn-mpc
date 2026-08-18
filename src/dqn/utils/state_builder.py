from __future__ import annotations

from typing import Sequence

import numpy as np


DQN_MPC_STATE_DIM = 11
DQN_MPC_PREVIEW_STEPS = 6

SOC_REFERENCE = 0.55
SOC_SCALE = 0.05

FUEL_CELL_POWER_SCALE_KW = 600.0
BATTERY_POWER_SCALE_KW = 624.0
LOAD_POWER_SCALE_KW = 600.0
LOAD_DELTA_SCALE_KW = 48.0


def build_dqn_mpc_state(
    *,
    current_soc: float,
    previous_fc_kw: float,
    previous_batt_kw: float,
    current_load_kw: float,
    previous_load_kw: float,
    future_load_kw: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """
    Build the normalized 11-dimensional DQN state used for MPC
    weight selection.

    State order:
        0: normalized current SOC
        1: normalized previous fuel-cell power
        2: normalized previous battery power
        3: normalized current load
        4: normalized current load change
        5-10: normalized load preview for t+1 ... t+6

    No clipping is applied. Values outside [-1, 1] or [0, 1]
    remain available to the DQN because they carry physical
    operating-point information.
    """

    preview = np.asarray(
        future_load_kw,
        dtype=np.float64,
    ).reshape(-1)

    if preview.size != DQN_MPC_PREVIEW_STEPS:
        raise ValueError(
            "future_load_kw must contain exactly "
            f"{DQN_MPC_PREVIEW_STEPS} values, got {preview.size}"
        )

    scalar_values = np.asarray(
        [
            current_soc,
            previous_fc_kw,
            previous_batt_kw,
            current_load_kw,
            previous_load_kw,
        ],
        dtype=np.float64,
    )

    if not np.all(np.isfinite(scalar_values)):
        raise ValueError("state scalar inputs must all be finite")

    if not np.all(np.isfinite(preview)):
        raise ValueError("future_load_kw must contain only finite values")

    delta_load_kw = float(current_load_kw) - float(previous_load_kw)

    state = np.concatenate(
        [
            np.asarray(
                [
                    (float(current_soc) - SOC_REFERENCE) / SOC_SCALE,
                    float(previous_fc_kw) / FUEL_CELL_POWER_SCALE_KW,
                    float(previous_batt_kw) / BATTERY_POWER_SCALE_KW,
                    float(current_load_kw) / LOAD_POWER_SCALE_KW,
                    delta_load_kw / LOAD_DELTA_SCALE_KW,
                ],
                dtype=np.float64,
            ),
            preview / LOAD_POWER_SCALE_KW,
        ]
    )

    if state.shape != (DQN_MPC_STATE_DIM,):
        raise RuntimeError(
            f"unexpected DQN state shape: {state.shape}"
        )

    return state.astype(np.float32, copy=False)
