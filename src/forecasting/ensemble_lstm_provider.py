from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from forecasting.lstm_load_predictor import add_time_features, inverse_target, load_checkpoint, transform


@dataclass(frozen=True)
class LSTMCheckpointSpec:
    name: str
    checkpoint: Path
    weight: float


class ActiveLSTMForecastProvider:
    """Current/history-only multi-step load forecast provider.

    Paper-facing name: formal LSTM load predictor.

    Implementation detail: the active checkpoint blend is fixed before formal
    evaluation and is not exposed in thesis figures or baseline labels.
    """

    def __init__(self, specs: list[LSTMCheckpointSpec], device: str = "auto"):
        self.specs = specs
        if not specs:
            raise ValueError("At least one LSTM checkpoint spec is required.")
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
        self.models: list[tuple[LSTMCheckpointSpec, Any, dict[str, Any]]] = []
        for spec in specs:
            model, payload = load_checkpoint(Path(spec.checkpoint), device=self.device)
            self.models.append((spec, model, payload))

    @staticmethod
    def formal_ensemble(project_root: Path, device: str = "auto") -> "ActiveLSTMForecastProvider":
        return ActiveLSTMForecastProvider(
            [
                LSTMCheckpointSpec(
                    "LSTM18_p18_h128_l2_lr2e4",
                    project_root / "outputs/lstm_hyperparam_refine_h128_local/checkpoints/LSTM18_p18_h128_l2_lr2e4/best_lstm_load_predictor.pt",
                    1.0,
                ),
            ],
            device=device,
        )

    def _prepare_features(self, df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
        out = add_time_features(df.copy()).reset_index(drop=True)
        group_indices = (
            [list(idx) for idx in out.groupby("voyage_name", sort=False).groups.values()]
            if "voyage_name" in out.columns
            else [list(out.index)]
        )
        if any(col.startswith("delta_") for col in features):
            for raw_col, delta_col in [
                ("load_total_kw", "delta_load_total"),
                ("load_left_kw", "delta_load_left"),
                ("load_right_kw", "delta_load_right"),
                ("speed_knots", "delta_speed"),
            ]:
                out[delta_col] = 0.0
                if raw_col in out.columns:
                    for idx_list in group_indices:
                        out.loc[idx_list, delta_col] = out.loc[idx_list, raw_col].astype(float).diff().fillna(0.0)
        if any(col.startswith("rolling_") for col in features) and "load_total_kw" in out.columns:
            for window in [3, 6]:
                for col in [f"rolling_mean_load_total_{window}", f"rolling_std_load_total_{window}"]:
                    out[col] = 0.0
                for idx_list in group_indices:
                    load = out.loc[idx_list, "load_total_kw"].astype(float)
                    out.loc[idx_list, f"rolling_mean_load_total_{window}"] = load.rolling(window, min_periods=1).mean()
                    out.loc[idx_list, f"rolling_std_load_total_{window}"] = load.rolling(window, min_periods=1).std().fillna(0.0)
            if "speed_knots" in out.columns:
                for col in ["rolling_mean_speed_3", "rolling_std_speed_3"]:
                    out[col] = 0.0
                for idx_list in group_indices:
                    speed = out.loc[idx_list, "speed_knots"].astype(float)
                    out.loc[idx_list, "rolling_mean_speed_3"] = speed.rolling(3, min_periods=1).mean()
                    out.loc[idx_list, "rolling_std_speed_3"] = speed.rolling(3, min_periods=1).std().fillna(0.0)
        for col in features:
            if col not in out.columns:
                out[col] = 0.0
        return out

    def _full_features(self, payload: dict[str, Any], df: pd.DataFrame) -> list[str]:
        features = list(payload["features"])
        scaler_dim = len(payload["feature_scaler"]["mean"])
        full = list(features)
        for col in ["time_sin", "time_cos"]:
            if len(full) < scaler_dim and col in df.columns and col not in full:
                full.append(col)
        if len(full) != scaler_dim:
            raise ValueError(f"LSTM feature dimension mismatch: features={len(full)}, scaler={scaler_dim}")
        return full

    def _single_matrix(self, base_df: pd.DataFrame, model: Any, payload: dict[str, Any]) -> np.ndarray:
        cfg = payload["config"]
        history_len = int(cfg["history_len"])
        horizon = int(cfg["pred_horizon"])
        df = self._prepare_features(base_df, list(payload["features"]))
        features = self._full_features(payload, df)
        values = df[features].astype(float).ffill().bfill().fillna(0.0).to_numpy()
        values = transform(values, payload["feature_scaler"])
        actual = df["load_total_kw"].astype(float).ffill().bfill().fillna(0.0).to_numpy()
        forecasts: list[np.ndarray] = []
        model.eval()
        with torch.no_grad():
            for idx in range(len(df)):
                start = max(0, idx - history_len + 1)
                hist = values[start : idx + 1]
                if len(hist) < history_len:
                    hist = np.vstack([np.repeat(hist[:1], history_len - len(hist), axis=0), hist])
                x = torch.as_tensor(hist[None, :, :], dtype=torch.float32, device=self.device)
                pred = model(x).detach().cpu().numpy()[0]
                pred = np.maximum(inverse_target(pred, payload["target_scaler"]), 0.0)
                row = np.concatenate([[actual[idx]], pred[: max(0, horizon - 1)]])
                if len(row) < horizon:
                    row = np.pad(row, (0, horizon - len(row)), mode="edge")
                forecasts.append(row[:horizon])
        return np.vstack(forecasts)

    def forecast_matrix(self, base_df: pd.DataFrame) -> np.ndarray:
        weighted = None
        total_weight = 0.0
        for spec, model, payload in self.models:
            matrix = self._single_matrix(base_df, model, payload)
            weighted = matrix * float(spec.weight) if weighted is None else weighted + matrix * float(spec.weight)
            total_weight += float(spec.weight)
        if weighted is None or total_weight <= 0:
            raise ValueError("No forecasts produced.")
        return np.maximum(weighted / total_weight, 0.0)

    def predict(self, sequence_window: pd.DataFrame) -> np.ndarray:
        matrix = self.forecast_matrix(sequence_window)
        return matrix[-1]
