"""Train-only fitting and equal-origin Validation selection of LSTM history."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from v2.data.formal_training_dataset import FormalTrainingDataset
from v2.data.supervisory_rules import normalize_onboard_load_kw
from v2.main.train_formal_dqn import DEFAULT_AIS_ROOT, DEFAULT_MODE_ROOT, DEFAULT_POWER_ROOT

from .forecasting import (
    DirectLSTM,
    FittedForecaster,
    FORECAST_HORIZON,
    HISTORY_CANDIDATES,
    windows_from_episodes,
)
from .metrics import forecast_metrics, persistence_forecast


def validation_windows_for_candidates(
    episodes: Sequence[object], candidates: Sequence[int], *, horizon: int = FORECAST_HORIZON
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Every H is scored on the same forecast origins, allowed by max(H)."""
    histories = tuple(sorted(set(candidates)))
    if not histories or histories[0] <= 0:
        raise ValueError("candidate histories must be positive")
    long_x, y = windows_from_episodes(episodes, history_steps=histories[-1], horizon=horizon)
    if len(y) == 0:
        raise ValueError("Validation contains no shared ONBOARD forecast origins")
    return {h: (long_x[:, -h:, :].copy(), y.copy()) for h in histories}


def _train_load_scale(episodes: Sequence[object]) -> tuple[float, float]:
    values = [
        normalize_onboard_load_kw(float(load))
        for episode in episodes
        for load, mode in zip(episode.load_kw, episode.operating_mode)
        if isinstance(mode, str) and mode.lower() == "onboard"
    ]
    if not values:
        raise ValueError("Train contains no ONBOARD loads")
    mean = float(np.mean(values))
    std = float(np.std(values))
    return mean, max(std, 1.0)


def fit_and_select_history(
    train_episodes: Sequence[object], validation_episodes: Sequence[object],
    *, candidates: Sequence[int] = HISTORY_CANDIDATES,
    epochs: int = 20, patience: int = 4, batch_size: int = 256,
    seed: int = 42, hidden_size: int = 32,
) -> tuple[FittedForecaster, dict[str, object]]:
    """Select H by five-step Validation WAPE; never opens Test payloads."""
    if min(epochs, patience, batch_size, hidden_size) <= 0:
        raise ValueError("training dimensions must be positive")
    torch.manual_seed(seed)
    np.random.seed(seed)
    mean, scale = _train_load_scale(train_episodes)
    shared_val = validation_windows_for_candidates(validation_episodes, candidates)
    common_x, common_y = shared_val[max(shared_val)]
    persistence_metrics = forecast_metrics(
        common_y, persistence_forecast(common_x, horizon=FORECAST_HORIZON)
    )
    if persistence_metrics["wape_percent"] is None:
        raise ValueError("Validation WAPE is undefined because all actual loads are zero")
    results: list[dict[str, object]] = []
    selected: tuple[float, FittedForecaster] | None = None
    for history_steps, (val_x, val_y) in shared_val.items():
        train_x, train_y = windows_from_episodes(train_episodes, history_steps=history_steps)
        if not len(train_x):
            raise ValueError(f"Train contains no H={history_steps} windows")
        model = DirectLSTM(history_steps=history_steps, hidden_size=hidden_size)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        normalized_x = torch.from_numpy((train_x - mean) / scale)
        normalized_y = torch.from_numpy((train_y - mean) / scale)
        loader = DataLoader(
            TensorDataset(normalized_x, normalized_y), batch_size=batch_size,
            shuffle=True, generator=torch.Generator().manual_seed(seed + history_steps),
        )
        validation_x = torch.from_numpy((val_x - mean) / scale)
        best_wape = float("inf")
        best_weights = None
        best_epoch = 0
        stale = 0
        for epoch in range(1, epochs + 1):
            model.train()
            for x_batch, y_batch in loader:
                optimizer.zero_grad()
                loss = torch.nn.functional.smooth_l1_loss(model(x_batch), y_batch)
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                predictions = np.maximum(0.0, np.concatenate([
                    model(chunk).numpy() for chunk in validation_x.split(batch_size)
                ]) * scale + mean)
            metrics = forecast_metrics(val_y, predictions)
            wape = metrics["wape_percent"]
            if wape is None:
                raise ValueError("Validation WAPE is undefined because all actual loads are zero")
            if wape < best_wape - 1e-7:
                best_wape, best_epoch = wape, epoch
                best_weights = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break
        assert best_weights is not None
        model.load_state_dict(best_weights)
        model.eval()
        with torch.no_grad():
            predictions = np.maximum(0.0, np.concatenate([
                model(chunk).numpy() for chunk in validation_x.split(batch_size)
            ]) * scale + mean)
        metrics = forecast_metrics(val_y, predictions)
        results.append({
            "history_steps": history_steps,
            "train_windows": len(train_x),
            "validation_windows": len(val_x),
            "best_epoch": best_epoch,
            "validation_metrics": metrics,
        })
        if selected is None or best_wape < selected[0]:
            selected = (best_wape, FittedForecaster(model, mean_kw=mean, scale_kw=scale))
    assert selected is not None
    report = {
        "selection_rule": "minimum equal-origin Validation five-step WAPE",
        "history_candidates": list(shared_val),
        "horizon_steps": FORECAST_HORIZON,
        "sample_seconds": 30,
        "selected_history_steps": selected[1].history_steps,
        "train_mean_kw": mean,
        "train_scale_kw": scale,
        "persistence_metrics": persistence_metrics,
        "results": results,
    }
    report["wape_skill_percent"] = float(
        100.0 * (1.0 - selected[0] / persistence_metrics["wape_percent"])
    )
    report["selected_beats_persistence"] = bool(
        selected[0] < persistence_metrics["wape_percent"]
    )
    return selected[1], report


def load_fitted_forecaster(path: Path) -> FittedForecaster:
    """Load only the v3 state dictionary and its frozen Train normalization."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("architecture") != "residual_direct_lstm_v1":
        raise ValueError("checkpoint architecture is not the v3 residual direct LSTM")
    if payload.get("horizon_steps") != FORECAST_HORIZON:
        raise ValueError("checkpoint does not predict exactly five future steps")
    model = DirectLSTM(
        history_steps=int(payload["history_steps"]),
        horizon=int(payload["horizon_steps"]),
        hidden_size=int(payload["hidden_size"]),
    )
    model.load_state_dict(payload["model_state"])
    return FittedForecaster(
        model,
        mean_kw=float(payload["train_mean_kw"]),
        scale_kw=float(payload["train_scale_kw"]),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--power-root", type=Path, default=DEFAULT_POWER_ROOT)
    parser.add_argument("--ais-root", type=Path, default=DEFAULT_AIS_ROOT)
    parser.add_argument("--mode-root", type=Path, default=DEFAULT_MODE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/v3_lstm"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    args = parser.parse_args(argv)
    dataset = FormalTrainingDataset.open(args.power_root, args.ais_root, args.mode_root)
    fitted, report = fit_and_select_history(
        dataset.load_train(), dataset.load_validation(),
        epochs=args.epochs, patience=args.patience,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "architecture": "residual_direct_lstm_v1",
        "model_state": fitted.model.state_dict(),
        "history_steps": fitted.history_steps,
        "horizon_steps": FORECAST_HORIZON,
        "hidden_size": fitted.model.lstm.hidden_size,
        "train_mean_kw": fitted.mean_kw,
        "train_scale_kw": fitted.scale_kw,
        "seed": 42,
    }, args.output_dir / "selected_lstm.pt")
    (args.output_dir / "selection.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
