from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PredictionWindow:
    load_kw: np.ndarray
    speed_knots: np.ndarray


class MovingAveragePredictor:
    """Lightweight predictor for phase-1 and phase-2 MPC studies."""

    def __init__(self, horizon_steps: int = 8):
        self.horizon_steps = horizon_steps

    def predict_from_dataframe(self, dataset: pd.DataFrame, start_index: int) -> PredictionWindow:
        load = dataset["load_total_kw"].iloc[start_index : start_index + self.horizon_steps].to_numpy(dtype=float)
        speed = dataset["speed_knots"].iloc[start_index : start_index + self.horizon_steps].to_numpy(dtype=float)
        if len(load) == 0:
            load = np.zeros(self.horizon_steps, dtype=float)
            speed = np.zeros(self.horizon_steps, dtype=float)
        if len(load) < self.horizon_steps:
            load = np.pad(load, (0, self.horizon_steps - len(load)), mode="edge")
            speed = np.pad(speed, (0, self.horizon_steps - len(speed)), mode="edge")
        return PredictionWindow(load_kw=load, speed_knots=speed)
