"""Causal direct multi-output LSTM load forecasting."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn

from v2.data.supervisory_rules import normalize_onboard_load_kw


DEFAULT_HISTORY_STEPS = 24
FORECAST_HORIZON = 5
HISTORY_CANDIDATES = (6, 12, 24)


def make_windows(
    loads_kw: np.ndarray,
    modes: tuple[str, ...],
    *,
    history_steps: int = DEFAULT_HISTORY_STEPS,
    horizon: int = FORECAST_HORIZON,
) -> tuple[np.ndarray, np.ndarray]:
    """Use only contiguous ONBOARD observations; targets start after the origin."""
    loads = np.asarray(loads_kw, dtype=np.float64)
    if loads.ndim != 1 or not np.isfinite(loads).all():
        raise ValueError("loads must be a finite one-dimensional array")
    if len(loads) != len(modes):
        raise ValueError("modes and loads must have equal length")
    if type(history_steps) is not int or history_steps <= 0:
        raise ValueError("history_steps must be a positive integer")
    if type(horizon) is not int or horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    clean = loads.copy()
    onboard = tuple(isinstance(mode, str) and mode.lower() == "onboard" for mode in modes)
    for index, is_onboard in enumerate(onboard):
        if is_onboard:
            clean[index] = normalize_onboard_load_kw(float(loads[index]))
    x, y = [], []
    for end in range(history_steps, len(loads) - horizon + 1):
        start = end - history_steps
        if all(onboard[start : end + horizon]):
            x.append(clean[start:end, None])
            y.append(clean[end : end + horizon])
    return (
        np.asarray(x, dtype=np.float32).reshape(-1, history_steps, 1),
        np.asarray(y, dtype=np.float32).reshape(-1, horizon),
    )


def windows_from_episodes(
    episodes: Sequence[object], *, history_steps: int, horizon: int = FORECAST_HORIZON
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate complete per-episode windows without making boundary pairs."""
    xs, ys = [], []
    for episode in episodes:
        x, y = make_windows(
            episode.load_kw, episode.operating_mode,
            history_steps=history_steps, horizon=horizon,
        )
        if len(x):
            xs.append(x)
            ys.append(y)
    if not xs:
        return (
            np.empty((0, history_steps, 1), dtype=np.float32),
            np.empty((0, horizon), dtype=np.float32),
        )
    return np.concatenate(xs), np.concatenate(ys)


class DirectLSTM(nn.Module):
    """Predict five changes from the last observed load in one pass."""

    def __init__(self, *, history_steps: int = DEFAULT_HISTORY_STEPS, horizon: int = FORECAST_HORIZON, hidden_size: int = 32):
        super().__init__()
        if history_steps <= 0 or horizon <= 0 or hidden_size <= 0:
            raise ValueError("model dimensions must be positive")
        self.history_steps = history_steps
        self.horizon = horizon
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, horizon)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 3 or history.shape[1:] != (self.history_steps, 1):
            raise ValueError(f"history must have shape [batch, {self.history_steps}, 1]")
        _, (hidden, _) = self.lstm(history)
        return history[:, -1, 0, None] + self.head(hidden[-1])


class FittedForecaster:
    """Inference adapter with a Train-fitted fixed load scale."""

    def __init__(self, model: DirectLSTM, *, mean_kw: float, scale_kw: float):
        if not np.isfinite([mean_kw, scale_kw]).all() or scale_kw <= 0:
            raise ValueError("forecaster normalization must be finite and positive")
        self.model = model.eval()
        self.mean_kw = float(mean_kw)
        self.scale_kw = float(scale_kw)
        self.history_steps = model.history_steps

    def __call__(self, history_kw: tuple[float, ...]) -> tuple[float, ...]:
        if len(history_kw) != self.history_steps:
            raise ValueError("history length does not match fitted LSTM")
        values = np.asarray(history_kw, dtype=np.float32)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("history must contain finite nonnegative loads")
        inputs = torch.from_numpy(((values - self.mean_kw) / self.scale_kw).reshape(1, -1, 1))
        with torch.no_grad():
            normalized = self.model(inputs).numpy()[0]
        powers = np.maximum(0.0, normalized * self.scale_kw + self.mean_kw)
        return tuple(float(value) for value in powers)
