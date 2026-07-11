from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd
import torch
from torch import nn


@dataclass(frozen=True)
class StandardScaler1D:
    mean: float
    std: float

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=np.float32) - self.mean) / self.std

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float32) * self.std + self.mean


@dataclass(frozen=True)
class WindowSet:
    x: np.ndarray
    y: np.ndarray
    sequence_ids: np.ndarray
    target_start_indices: np.ndarray


def fit_standard_scaler(train_sequences: Mapping[str, np.ndarray]) -> StandardScaler1D:
    if not train_sequences:
        raise ValueError("Training sequences must not be empty")
    values = np.concatenate(
        [np.asarray(value, dtype=np.float64).reshape(-1) for value in train_sequences.values()]
    )
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Training load values must be finite and non-empty")
    std = float(values.std(ddof=0))
    if not np.isfinite(std) or std <= 0.0:
        raise ValueError("Training load standard deviation must be positive")
    return StandardScaler1D(mean=float(values.mean()), std=std)


def build_windows(
    sequences: Mapping[str, np.ndarray], *, history_steps: int, prediction_steps: int
) -> WindowSet:
    if history_steps <= 0 or prediction_steps <= 0:
        raise ValueError("history_steps and prediction_steps must be positive")
    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    sequence_ids: list[str] = []
    target_starts: list[int] = []
    for sequence_id, raw_values in sequences.items():
        values = np.asarray(raw_values, dtype=np.float32).reshape(-1)
        if not np.isfinite(values).all():
            raise ValueError(f"Sequence {sequence_id} contains non-finite values")
        count = max(values.size - history_steps - prediction_steps + 1, 0)
        for start in range(count):
            target_start = start + history_steps
            x_rows.append(values[start:target_start, None])
            y_rows.append(values[target_start : target_start + prediction_steps])
            sequence_ids.append(str(sequence_id))
            target_starts.append(target_start)
    if not x_rows:
        return WindowSet(
            x=np.empty((0, history_steps, 1), dtype=np.float32),
            y=np.empty((0, prediction_steps), dtype=np.float32),
            sequence_ids=np.empty((0,), dtype=object),
            target_start_indices=np.empty((0,), dtype=np.int64),
        )
    return WindowSet(
        x=np.stack(x_rows).astype(np.float32, copy=False),
        y=np.stack(y_rows).astype(np.float32, copy=False),
        sequence_ids=np.asarray(sequence_ids, dtype=object),
        target_start_indices=np.asarray(target_starts, dtype=np.int64),
    )


def baseline_forecasts(histories: np.ndarray, *, prediction_steps: int) -> dict[str, np.ndarray]:
    values = np.asarray(histories, dtype=np.float64)
    if values.ndim != 3 or values.shape[2] != 1 or values.shape[1] < 2:
        raise ValueError("histories must have shape (samples, history, 1) with history >= 2")
    if prediction_steps <= 0 or not np.isfinite(values).all():
        raise ValueError("prediction_steps must be positive and histories finite")
    series = values[:, :, 0]
    horizons = np.arange(1, prediction_steps + 1, dtype=np.float64)[None, :]
    current = np.repeat(series[:, -1, None], prediction_steps, axis=1)
    last_slope = series[:, -1, None] + (series[:, -1] - series[:, -2])[:, None] * horizons
    positions = np.arange(series.shape[1], dtype=np.float64)
    design = np.column_stack([positions, np.ones_like(positions)])
    future_positions = np.arange(series.shape[1], series.shape[1] + prediction_steps, dtype=np.float64)
    trend = np.empty((series.shape[0], prediction_steps), dtype=np.float64)
    for index, row in enumerate(series):
        slope, intercept = np.linalg.lstsq(design, row, rcond=None)[0]
        trend[index] = slope * future_positions + intercept
    return {
        "current_hold": current,
        "last_slope": last_slope,
        "local_linear_trend": trend,
    }


def _metric_row(truth: np.ndarray, prediction: np.ndarray, horizon: object) -> dict[str, object]:
    error = prediction - truth
    absolute_error = np.abs(error)
    denominator = float(np.abs(truth).sum())
    residual_sum = float(np.square(error).sum())
    total_sum = float(np.square(truth - truth.mean()).sum())
    return {
        "horizon": horizon,
        "mae_kw": float(absolute_error.mean()),
        "rmse_kw": float(np.sqrt(np.square(error).mean())),
        "wape_pct": float(100.0 * absolute_error.sum() / denominator) if denominator > 0.0 else np.nan,
        "bias_kw": float(error.mean()),
        "r2": float(1.0 - residual_sum / total_sum) if total_sum > 0.0 else np.nan,
        "samples": int(truth.size),
        "negative_prediction_count": int(np.count_nonzero(prediction < 0.0)),
    }


def metrics_by_horizon(truth: np.ndarray, prediction: np.ndarray) -> pd.DataFrame:
    expected = np.asarray(truth, dtype=np.float64)
    actual = np.asarray(prediction, dtype=np.float64)
    if expected.ndim != 2 or actual.shape != expected.shape:
        raise ValueError("truth and prediction must have identical two-dimensional shapes")
    if not np.isfinite(expected).all() or not np.isfinite(actual).all():
        raise ValueError("truth and prediction must be finite")
    rows = [
        _metric_row(expected[:, index], actual[:, index], index + 1)
        for index in range(expected.shape[1])
    ]
    rows.append(_metric_row(expected.reshape(-1), actual.reshape(-1), "aggregate"))
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class ModelConfig:
    hidden_size: int
    num_layers: int
    dropout: float
    mlp_head: tuple[int, ...]


class SequenceToVectorLSTM(nn.Module):
    def __init__(self, *, config: ModelConfig, prediction_steps: int = 6) -> None:
        super().__init__()
        recurrent_dropout = config.dropout if config.num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            1,
            config.hidden_size,
            config.num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )
        widths = (config.hidden_size,) + config.mlp_head + (prediction_steps,)
        layers: list[nn.Module] = []
        for index in range(len(widths) - 1):
            layers.append(nn.Linear(widths[index], widths[index + 1]))
            if index < len(widths) - 2:
                layers.append(nn.ReLU())
        self.head = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(inputs)
        return self.head(output[:, -1, :])
