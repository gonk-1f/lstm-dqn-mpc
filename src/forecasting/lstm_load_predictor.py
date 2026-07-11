from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset


BASE_FEATURES = ["load_total_kw", "load_left_kw", "load_right_kw", "speed_knots", "soc_left", "soc_right"]


@dataclass
class LSTMForecastConfig:
    history_len: int = 18
    pred_horizon: int = 18
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.2
    batch_size: int = 64
    lr: float = 1.0e-3
    epochs: int = 20
    patience: int = 4
    seed: int = 42
    grad_clip: float = 1.0


class MultiStepLoadLSTM(nn.Module):
    def __init__(self, feature_dim: int, config: LSTMForecastConfig):
        super().__init__()
        self.dropout_rate = float(config.dropout)
        self.input_dropout = nn.Dropout(self.dropout_rate)
        lstm_dropout = self.dropout_rate if int(config.num_layers) > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=int(feature_dim),
            hidden_size=int(config.hidden_size),
            num_layers=int(config.num_layers),
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.head = nn.Linear(int(config.hidden_size), int(config.pred_horizon))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_dropout(x)
        _, (hidden, _) = self.lstm(x)
        return self.head(hidden[-1])

    def mc_forward(self, x: torch.Tensor, n_samples: int = 10) -> tuple[torch.Tensor, torch.Tensor]:
        """Monte Carlo Dropout: N stochastic forward passes, return (mean, std)."""
        self.train()  # keep dropout active
        samples = []
        with torch.no_grad():
            for _ in range(n_samples):
                samples.append(self.forward(x))
        stacked = torch.stack(samples, dim=0)  # (N, batch, horizon)
        return stacked.mean(dim=0), stacked.std(dim=0)


class SequenceDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


def set_seed(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def base_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "dqn_substep" in out.columns:
        out = out[out["dqn_substep"].astype(int) == 0].copy()
    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
        sort_cols = ["voyage_name", "timestamp"] if "voyage_name" in out.columns else ["timestamp"]
        out = out.sort_values(sort_cols)
    return out.reset_index(drop=True)


def available_features(df: pd.DataFrame) -> list[str]:
    features = [col for col in BASE_FEATURES if col in df.columns]
    if "load_total_kw" not in features:
        raise ValueError("LSTM load forecasting requires at least load_total_kw.")
    return features


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "timestamp" not in out.columns:
        return out
    ts = pd.to_datetime(out["timestamp"], errors="coerce")
    minute = ts.dt.hour.fillna(0) * 60 + ts.dt.minute.fillna(0)
    angle = 2.0 * np.pi * minute.astype(float) / 1440.0
    out["time_sin"] = np.sin(angle)
    out["time_cos"] = np.cos(angle)
    return out


def fit_scaler(values: np.ndarray) -> dict[str, list[float]]:
    mean = np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return {"mean": mean.astype(float).tolist(), "std": std.astype(float).tolist()}


def transform(values: np.ndarray, scaler: dict[str, list[float]]) -> np.ndarray:
    mean = np.asarray(scaler["mean"], dtype=float)
    std = np.asarray(scaler["std"], dtype=float)
    return (values - mean) / std


def inverse_target(values: np.ndarray, target_scaler: dict[str, float]) -> np.ndarray:
    return values * float(target_scaler["std"]) + float(target_scaler["mean"])


def make_windows(
    df: pd.DataFrame,
    features: list[str],
    config: LSTMForecastConfig,
    feature_scaler: dict[str, list[float]] | None = None,
    target_scaler: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, list[float]], dict[str, float]]:
    data = add_time_features(base_rows(df))
    full_features = list(features)
    for col in ["time_sin", "time_cos"]:
        if col in data.columns and col not in full_features:
            full_features.append(col)
    values = data[full_features].astype(float).ffill().bfill().fillna(0.0).to_numpy()
    target = data["load_total_kw"].astype(float).ffill().bfill().fillna(0.0).to_numpy()
    if feature_scaler is None:
        feature_scaler = fit_scaler(values)
    if target_scaler is None:
        std = float(np.nanstd(target))
        target_scaler = {"mean": float(np.nanmean(target)), "std": std if std >= 1e-6 else 1.0}
    values = transform(values, feature_scaler)
    target_norm = (target - float(target_scaler["mean"])) / float(target_scaler["std"])

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    group_key = "voyage_name" if "voyage_name" in data.columns else None
    groups = data.groupby(group_key, sort=False).indices.values() if group_key else [np.arange(len(data))]
    for idx_values in groups:
        idx = np.asarray(list(idx_values), dtype=int)
        if len(idx) < int(config.history_len) + int(config.pred_horizon):
            continue
        for pos in range(int(config.history_len), len(idx) - int(config.pred_horizon) + 1):
            hist_idx = idx[pos - int(config.history_len) : pos]
            fut_idx = idx[pos : pos + int(config.pred_horizon)]
            xs.append(values[hist_idx])
            ys.append(target_norm[fut_idx])
    if not xs:
        raise ValueError("No LSTM windows generated; check history_len, pred_horizon, and split lengths.")
    return np.stack(xs), np.stack(ys), feature_scaler, target_scaler


def metrics_from_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    err = y_pred - y_true
    horizon = err.shape[1] if err.ndim > 1 else 1
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    denom = np.maximum(np.abs(y_true), 1.0)
    mape = float(np.mean(np.abs(err) / denom) * 100.0)
    smape = float(np.mean(2.0 * np.abs(err) / np.maximum(np.abs(y_true) + np.abs(y_pred), 1.0)) * 100.0)
    bias = float(np.mean(err))

    out = {"MAE": mae, "RMSE": rmse, "MAPE": mape, "sMAPE": smape, "bias": bias}

    # Per-horizon metrics
    horizon_specs = [("h1", 0), ("h3", 2), ("h6", 5), ("h8", 7), ("h18", 17)]
    for label, h in horizon_specs:
        if h < horizon:
            e_h = err[:, h]
            out[f"{label}_MAE"] = float(np.mean(np.abs(e_h)))
            out[f"{label}_bias"] = float(np.mean(e_h))

    # Peak underestimation ratio (top 10% of actual load)
    top10_thresh = float(np.percentile(y_true.reshape(-1), 90))
    peak_mask = y_true.reshape(-1) >= top10_thresh
    if peak_mask.sum() > 0:
        peak_err = err.reshape(-1)[peak_mask]
        out["top10_load_bias"] = float(np.mean(peak_err))
        out["peak_underestimation_ratio"] = float(np.mean(peak_err < -1e-6))
    else:
        out["top10_load_bias"] = 0.0
        out["peak_underestimation_ratio"] = 0.0

    # Ramp RMSE (step-to-step load change error)
    if y_true.shape[0] > 1:
        true_ramp = np.diff(y_true.reshape(-1))
        pred_ramp = np.diff(y_pred.reshape(-1))
        out["ramp_RMSE"] = float(np.sqrt(np.mean((pred_ramp - true_ramp) ** 2)))
    else:
        out["ramp_RMSE"] = 0.0

    return out


def control_aware_validation_score(
    metrics: dict[str, float],
    alpha: float = 0.5,
    beta: float = 0.3,
    gamma: float = 1.0,
) -> float:
    return float(
        metrics.get("RMSE", 999.0)
        + alpha * abs(metrics.get("bias", 0.0))
        + beta * metrics.get("ramp_RMSE", 0.0)
        + gamma * metrics.get("peak_underestimation_ratio", 0.0)
    )


def save_checkpoint(
    path: Path,
    model: MultiStepLoadLSTM,
    config: LSTMForecastConfig,
    features: list[str],
    feature_scaler: dict[str, list[float]],
    target_scaler: dict[str, float],
    metrics: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": asdict(config),
            "features": features,
            "feature_scaler": feature_scaler,
            "target_scaler": target_scaler,
            "metrics": metrics,
        },
        path,
    )
    meta = {k: v for k, v in metrics.items()}
    meta.update({"config": asdict(config), "features": features, "checkpoint": str(path)})
    path.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def load_checkpoint(path: Path, device: str = "cpu") -> tuple[MultiStepLoadLSTM, dict[str, Any]]:
    payload = torch.load(path, map_location=device)
    config = LSTMForecastConfig(**payload["config"])
    model = MultiStepLoadLSTM(len(payload["feature_scaler"]["mean"]), config).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload
