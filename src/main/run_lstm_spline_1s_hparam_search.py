"""Optuna LSTM search for offline natural-clipped spline 1 s load data.

This experiment is intentionally isolated from the current 30 s LSTM-MPC
mainline. The input data are offline cubic-spline reconstructions, not measured
1 s load and not online-feasible forecast labels.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJ = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_DIR = PROJ / "outputs" / "spline_1s_diagnostics" / "data" / "natural_clipped_by_voyage"
DEFAULT_SPLIT_JSON = PROJ / "outputs" / "mpc_solver_benchmark_1s" / "data" / "voyage_split_spline_1s_total_load_721.json"
DEFAULT_OUTPUT_DIR = PROJ / "outputs" / "lstm_spline_1s_hparam_search"
DATA_LABEL = "natural clipped cubic-spline reconstructed 1s load profile"
DATA_CAVEAT = (
    "The data are offline natural-boundary cubic-spline reconstructions from "
    "original 30 s vessel load voyages with nonnegative clipping. They are not "
    "native measured 1 s load data, not online prediction data, and intermediate "
    "1 s points use future 30 s endpoint information."
)


@dataclass(frozen=True)
class SearchTask:
    name: str
    pred_horizon: int
    n_trials: int
    fixed_history_len: int | None = None


@dataclass(frozen=True)
class TrialConfig:
    history_len: int
    pred_horizon: int
    hidden_size: int
    num_layers: int
    dropout: float
    mlp_head: tuple[int, ...]
    loss: str
    learning_rate: float
    batch_size: int
    gradient_clip: float
    weight_decay: float
    seed: int
    epochs_max: int
    early_stopping_patience: int


def config_fingerprint(config: TrialConfig) -> tuple[Any, ...]:
    return (
        int(config.history_len),
        int(config.pred_horizon),
        int(config.hidden_size),
        int(config.num_layers),
        float(config.dropout),
        tuple(int(x) for x in config.mlp_head),
        str(config.loss),
        float(config.learning_rate),
        int(config.batch_size),
        float(config.gradient_clip),
        float(config.weight_decay),
        int(config.seed),
    )


def write_trial_snapshot(output_dir: Path, task_name: str, rows: list[dict[str, Any]]) -> Path:
    path = output_dir / f"hparam_trials_{task_name}.partial.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return path


class MultiHorizonLSTM(nn.Module):
    def __init__(self, *, feature_dim: int, config: TrialConfig):
        super().__init__()
        lstm_dropout = float(config.dropout) if int(config.num_layers) > 1 else 0.0
        self.input_dropout = nn.Dropout(float(config.dropout))
        self.lstm = nn.LSTM(
            input_size=int(feature_dim),
            hidden_size=int(config.hidden_size),
            num_layers=int(config.num_layers),
            dropout=lstm_dropout,
            batch_first=True,
        )
        layers: list[nn.Module] = []
        in_dim = int(config.hidden_size)
        for hidden in config.mlp_head:
            layers.append(nn.Linear(in_dim, int(hidden)))
            layers.append(nn.ReLU())
            if float(config.dropout) > 0:
                layers.append(nn.Dropout(float(config.dropout)))
            in_dim = int(hidden)
        layers.append(nn.Linear(in_dim, int(config.pred_horizon)))
        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_dropout(x)
        _, (hidden, _) = self.lstm(x)
        return self.head(hidden[-1])


def set_seed(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_default_tasks() -> dict[str, SearchTask]:
    return {
        "taskA": SearchTask(name="taskA", pred_horizon=60, n_trials=40),
        "taskB": SearchTask(name="taskB", pred_horizon=180, n_trials=30),
    }


def build_horizon_steps(task: SearchTask) -> list[int]:
    steps = [1, 6, 30, 60]
    if int(task.pred_horizon) >= 120:
        steps.append(120)
    if int(task.pred_horizon) >= 180:
        steps.append(180)
    return [step for step in steps if step <= int(task.pred_horizon)]


def build_windows_for_series(
    load: np.ndarray,
    *,
    history_len: int,
    pred_horizon: int,
    stride: int = 1,
    max_windows: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(load, dtype=np.float32).reshape(-1)
    history_len = int(history_len)
    pred_horizon = int(pred_horizon)
    stride = max(1, int(stride))
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    stop = len(values) - pred_horizon + 1
    for pos in range(history_len, stop, stride):
        xs.append(values[pos - history_len : pos].reshape(history_len, 1))
        ys.append(values[pos : pos + pred_horizon])
        if max_windows is not None and len(xs) >= int(max_windows):
            break
    if not xs:
        return (
            np.empty((0, history_len, 1), dtype=np.float32),
            np.empty((0, pred_horizon), dtype=np.float32),
        )
    return np.stack(xs).astype(np.float32, copy=False), np.stack(ys).astype(np.float32, copy=False)


def current_hold_forecast(history: np.ndarray, pred_horizon: int) -> np.ndarray:
    values = np.asarray(history, dtype=float).reshape(-1)
    return np.full(int(pred_horizon), float(values[-1]), dtype=float)


def last_slope_forecast(history: np.ndarray, pred_horizon: int) -> np.ndarray:
    values = np.asarray(history, dtype=float).reshape(-1)
    current = float(values[-1])
    prev = float(values[-2]) if len(values) >= 2 else current
    slope = current - prev
    return current + np.arange(1, int(pred_horizon) + 1, dtype=float) * slope


def moving_average_hold_forecast(history: np.ndarray, pred_horizon: int, *, window: int = 60) -> np.ndarray:
    values = np.asarray(history, dtype=float).reshape(-1)
    tail = values[-min(len(values), int(window)) :]
    return np.full(int(pred_horizon), float(np.mean(tail)), dtype=float)


def ema_hold_forecast(history: np.ndarray, pred_horizon: int, *, span: int = 60) -> np.ndarray:
    values = np.asarray(history, dtype=float).reshape(-1)
    alpha = 2.0 / (float(span) + 1.0)
    ema = float(values[0])
    for value in values[1:]:
        ema = alpha * float(value) + (1.0 - alpha) * ema
    return np.full(int(pred_horizon), ema, dtype=float)


def horizon_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    truth = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    horizon = min(truth.shape[1], pred.shape[1])
    out: dict[str, float] = {}
    for h_idx in range(horizon):
        label = f"h{h_idx + 1}"
        t = truth[:, h_idx]
        p = pred[:, h_idx]
        mask = np.isfinite(t) & np.isfinite(p)
        if not np.any(mask):
            out[f"MAE_{label}"] = float("nan")
            out[f"RMSE_{label}"] = float("nan")
            out[f"WAPE_{label}"] = float("nan")
            out[f"Bias_{label}"] = float("nan")
            continue
        err = p[mask] - t[mask]
        denom = float(np.sum(np.abs(t[mask])) + 1e-6)
        out[f"MAE_{label}"] = float(np.mean(np.abs(err)))
        out[f"RMSE_{label}"] = float(np.sqrt(np.mean(err**2)))
        out[f"WAPE_{label}"] = float(np.sum(np.abs(err)) / denom * 100.0)
        out[f"Bias_{label}"] = float(np.mean(err))
    return out


def aggregate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    metrics = horizon_metrics(y_true, y_pred)
    horizon = min(y_true.shape[1], y_pred.shape[1])
    mae_values = [metrics[f"MAE_h{i}"] for i in range(1, horizon + 1)]
    rmse_values = [metrics[f"RMSE_h{i}"] for i in range(1, horizon + 1)]
    wape_values = [metrics[f"WAPE_h{i}"] for i in range(1, horizon + 1)]
    bias_values = [metrics[f"Bias_h{i}"] for i in range(1, horizon + 1)]
    metrics["MAE_mean_all_horizons"] = float(np.nanmean(mae_values))
    metrics["RMSE_mean_all_horizons"] = float(np.nanmean(rmse_values))
    metrics["WAPE_mean_all_horizons"] = float(np.nanmean(wape_values))
    metrics["Bias_mean"] = float(np.nanmean(bias_values))
    return metrics


def primary_score_from_metrics(metrics: dict[str, float], task: SearchTask) -> float:
    values = [float(metrics[f"WAPE_h{idx}"]) for idx in range(1, int(task.pred_horizon) + 1)]
    return float(np.nanmean(values))


def load_split(split_json: Path) -> dict[str, Any]:
    split = load_json(split_json)
    for key in ["train_voyages", "validation_voyages", "test_voyages"]:
        if key not in split or not split[key]:
            raise ValueError(f"Split JSON missing non-empty {key}: {split_json}")
    return split


def voyage_csv_path(source_dir: Path, voyage_id: str) -> Path:
    matches = sorted(Path(source_dir).glob(f"{voyage_id}__*.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one CSV for {voyage_id} in {source_dir}, found {len(matches)}")
    return matches[0]


def read_voyage_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["voyage_id", "split", "time_s", "load_total_kw", "online_feasible", "uses_future_endpoint"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    df = df.sort_values("time_s").reset_index(drop=True)
    return df


def load_voyage_frames(source_dir: Path, voyages: list[str]) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for voyage_id in voyages:
        frames[voyage_id] = read_voyage_frame(voyage_csv_path(source_dir, voyage_id))
    return frames


def split_stats(frames: dict[str, pd.DataFrame], voyages: list[str]) -> dict[str, Any]:
    chunks = [frames[voyage_id] for voyage_id in voyages]
    df = pd.concat(chunks, ignore_index=True)
    duplicate_time = any(bool(frame["time_s"].duplicated().any()) for frame in chunks)
    load = df["load_total_kw"].astype(float)
    return {
        "voyages": len(voyages),
        "rows": int(len(df)),
        "load_min": float(load.min()),
        "load_max": float(load.max()),
        "load_mean": float(load.mean()),
        "load_std": float(load.std(ddof=0)),
        "has_nan": bool(df[["time_s", "load_total_kw"]].isna().any().any()),
        "has_negative_load": bool((load < 0).any()),
        "has_duplicate_time_s_within_voyage": duplicate_time,
    }


def write_data_check(output_dir: Path, source_dir: Path, split: dict[str, Any], frames: dict[str, pd.DataFrame]) -> None:
    stats = {
        "train": split_stats(frames, split["train_voyages"]),
        "validation": split_stats(frames, split["validation_voyages"]),
        "test": split_stats(frames, split["test_voyages"]),
    }
    lines = [
        "# Spline 1s LSTM Data Check",
        "",
        f"- Data source: `{source_dir}`",
        f"- Data label: {DATA_LABEL}",
        f"- Caveat: {DATA_CAVEAT}",
        "- Window crossing voyage boundaries: false",
        "- Scaler fit scope: train split only",
        "- online_feasible=false",
        "- uses_future_endpoint=true",
        "- not_measured_1s=true",
        "",
        "| split | voyages | rows | load_min | load_max | load_mean | load_std | has_nan | has_negative_load | duplicate_time_s |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for name, row in stats.items():
        lines.append(
            "| {name} | {voyages} | {rows} | {load_min:.6f} | {load_max:.6f} | "
            "{load_mean:.6f} | {load_std:.6f} | {has_nan} | {has_negative_load} | "
            "{has_duplicate_time_s_within_voyage} |".format(name=name, **row)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "data_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def fit_train_scaler(frames: dict[str, pd.DataFrame], train_voyages: list[str]) -> dict[str, float]:
    values = np.concatenate([frames[v]["load_total_kw"].astype(float).to_numpy() for v in train_voyages])
    std = float(np.nanstd(values))
    return {"mean": float(np.nanmean(values)), "std": std if std >= 1e-6 else 1.0}


def split_windows_raw(
    frames: dict[str, pd.DataFrame],
    voyages: list[str],
    *,
    history_len: int,
    pred_horizon: int,
    stride: int,
    max_windows: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    x_chunks: list[np.ndarray] = []
    y_chunks: list[np.ndarray] = []
    per_voyage_cap = None
    if max_windows is not None:
        per_voyage_cap = max(1, int(math.ceil(float(max_windows) / max(len(voyages), 1))))
    for voyage_id in voyages:
        series = frames[voyage_id]["load_total_kw"].astype(float).to_numpy()
        x, y = build_windows_for_series(
            series,
            history_len=history_len,
            pred_horizon=pred_horizon,
            stride=stride,
            max_windows=per_voyage_cap,
        )
        if len(x):
            x_chunks.append(x)
            y_chunks.append(y)
    if not x_chunks:
        raise ValueError("No windows generated; check history_len, pred_horizon, and voyage lengths.")
    x_all = np.concatenate(x_chunks, axis=0)
    y_all = np.concatenate(y_chunks, axis=0)
    if max_windows is not None and len(x_all) > int(max_windows):
        pick = np.linspace(0, len(x_all) - 1, int(max_windows)).round().astype(int)
        x_all = x_all[pick]
        y_all = y_all[pick]
    return x_all.astype(np.float32, copy=False), y_all.astype(np.float32, copy=False)


def normalize_xy(x_raw: np.ndarray, y_raw: np.ndarray, scaler: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    mean = float(scaler["mean"])
    std = float(scaler["std"])
    return ((x_raw - mean) / std).astype(np.float32), ((y_raw - mean) / std).astype(np.float32)


def inverse_y(y_norm: np.ndarray, scaler: dict[str, float]) -> np.ndarray:
    return y_norm * float(scaler["std"]) + float(scaler["mean"])


def baseline_predictions(x_raw: np.ndarray, pred_horizon: int) -> dict[str, np.ndarray]:
    n_rows = int(x_raw.shape[0])
    pred_horizon = int(pred_horizon)
    out = {
        "current-hold": np.empty((n_rows, pred_horizon), dtype=float),
        "last-slope": np.empty((n_rows, pred_horizon), dtype=float),
        "moving-average-hold": np.empty((n_rows, pred_horizon), dtype=float),
        "ema-hold": np.empty((n_rows, pred_horizon), dtype=float),
    }
    for idx in range(n_rows):
        history = x_raw[idx, :, 0]
        out["current-hold"][idx] = current_hold_forecast(history, pred_horizon)
        out["last-slope"][idx] = last_slope_forecast(history, pred_horizon)
        out["moving-average-hold"][idx] = moving_average_hold_forecast(history, pred_horizon, window=60)
        out["ema-hold"][idx] = ema_hold_forecast(history, pred_horizon, span=60)
    return out


def loss_function(name: str) -> nn.Module:
    if name == "MSE":
        return nn.MSELoss()
    if name == "Huber":
        return nn.SmoothL1Loss(beta=1.0)
    raise ValueError(f"Unsupported loss: {name}")


def run_training(
    *,
    task: SearchTask,
    config: TrialConfig,
    x_train_raw: np.ndarray,
    y_train_raw: np.ndarray,
    x_val_raw: np.ndarray,
    y_val_raw: np.ndarray,
    scaler: dict[str, float],
    device: str,
    trial_time_limit_sec: float | None = None,
) -> dict[str, Any]:
    set_seed(int(config.seed))
    x_train, y_train = normalize_xy(x_train_raw, y_train_raw, scaler)
    x_val, y_val = normalize_xy(x_val_raw, y_val_raw, scaler)
    train_loader = DataLoader(
        TensorDataset(torch.as_tensor(x_train), torch.as_tensor(y_train)),
        batch_size=int(config.batch_size),
        shuffle=True,
        generator=torch.Generator().manual_seed(int(config.seed)),
    )
    val_loader = DataLoader(
        TensorDataset(torch.as_tensor(x_val), torch.as_tensor(y_val)),
        batch_size=max(int(config.batch_size), 1) * 4,
        shuffle=False,
    )
    model = MultiHorizonLSTM(feature_dim=1, config=config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    criterion = loss_function(config.loss)
    best_score = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, float] = {}
    best_epoch = 0
    stale_epochs = 0
    start_time = time.perf_counter()
    stopped_by_trial_time_limit = False
    for epoch in range(1, int(config.epochs_max) + 1):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.gradient_clip))
            optimizer.step()
        model.eval()
        pred_chunks: list[np.ndarray] = []
        with torch.no_grad():
            for x_batch, _ in val_loader:
                pred_chunks.append(model(x_batch.to(device)).detach().cpu().numpy())
        val_pred_kw = inverse_y(np.vstack(pred_chunks), scaler)
        metrics = aggregate_metrics(y_val_raw, val_pred_kw)
        score = primary_score_from_metrics(metrics, task)
        if score < best_score - 1e-9:
            best_score = score
            best_metrics = metrics
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= int(config.early_stopping_patience):
                break
        if trial_time_limit_sec is not None and float(trial_time_limit_sec) > 0:
            if time.perf_counter() - start_time >= float(trial_time_limit_sec):
                stopped_by_trial_time_limit = True
                break
    if best_state is None:
        raise RuntimeError("Training produced no best state.")
    elapsed = time.perf_counter() - start_time
    model.load_state_dict(best_state)
    row = {
        "task": task.name,
        "primary_score": float(best_score),
        "val_MAE_mean_all_horizons": float(best_metrics["MAE_mean_all_horizons"]),
        "val_RMSE_mean_all_horizons": float(best_metrics["RMSE_mean_all_horizons"]),
        "val_WAPE_mean_all_horizons": float(best_metrics["WAPE_mean_all_horizons"]),
        "val_Bias_mean": float(best_metrics["Bias_mean"]),
        "train_time_sec": float(elapsed),
        "best_epoch": int(best_epoch),
        "stopped_by_trial_time_limit": bool(stopped_by_trial_time_limit),
    }
    for step in [1, 6, 30, 60, 120, 180]:
        key = f"MAE_h{step}"
        row[f"val_h{step}_MAE"] = float(best_metrics[key]) if key in best_metrics else float("nan")
    row.update(asdict(config))
    row["lstm_layers"] = int(config.num_layers)
    row["mlp_head"] = "-".join(str(x) for x in config.mlp_head)
    return {"row": row, "model": model, "state": best_state, "metrics": best_metrics}


def sample_trial_config(trial: optuna.Trial, task: SearchTask, args: argparse.Namespace) -> TrialConfig:
    if task.fixed_history_len is None:
        history_candidates = [60, 180, 300, 540]
        if bool(getattr(args, "include_history_900", False)):
            history_candidates.append(900)
        history_len = int(trial.suggest_categorical("history_len", history_candidates))
    else:
        history_len = int(task.fixed_history_len)
    mlp_map = {"128": (128,), "256": (256,), "256-128": (256, 128)}
    return TrialConfig(
        history_len=history_len,
        pred_horizon=int(task.pred_horizon),
        hidden_size=int(trial.suggest_categorical("hidden_size", [64, 128, 256])),
        num_layers=int(trial.suggest_categorical("lstm_layers", [1, 2, 3])),
        dropout=float(trial.suggest_categorical("dropout", [0.0, 0.1, 0.2, 0.3])),
        mlp_head=mlp_map[str(trial.suggest_categorical("mlp_head", list(mlp_map.keys())))],
        loss=str(trial.suggest_categorical("loss", ["MSE", "Huber"])),
        learning_rate=float(trial.suggest_categorical("learning_rate", [1e-3, 5e-4, 2e-4, 1e-4])),
        batch_size=int(trial.suggest_categorical("batch_size", [32, 64, 128])),
        gradient_clip=float(trial.suggest_categorical("gradient_clip", [0.5, 1.0, 5.0])),
        weight_decay=float(trial.suggest_categorical("weight_decay", [0.0, 1e-6, 1e-5])),
        seed=int(trial.suggest_categorical("seed", [42, 123])),
        epochs_max=int(args.epochs),
        early_stopping_patience=int(args.patience),
    )


def trial_config_from_row(row: dict[str, Any], *, task: SearchTask, seed: int | None = None, args: argparse.Namespace) -> TrialConfig:
    mlp_head = tuple(int(part) for part in str(row["mlp_head"]).split("-") if part)
    return TrialConfig(
        history_len=int(row["history_len"]),
        pred_horizon=int(task.pred_horizon),
        hidden_size=int(row["hidden_size"]),
        num_layers=int(row["num_layers"]),
        dropout=float(row["dropout"]),
        mlp_head=mlp_head,
        loss=str(row["loss"]),
        learning_rate=float(row["learning_rate"]),
        batch_size=int(row["batch_size"]),
        gradient_clip=float(row["gradient_clip"]),
        weight_decay=float(row["weight_decay"]),
        seed=int(seed if seed is not None else row["seed"]),
        epochs_max=int(args.epochs),
        early_stopping_patience=int(args.patience),
    )


def build_fixed_taskC_config(args: argparse.Namespace) -> TrialConfig:
    return TrialConfig(
        history_len=30,
        pred_horizon=6,
        hidden_size=128,
        num_layers=3,
        dropout=0.0,
        mlp_head=(128,),
        loss="Huber",
        learning_rate=0.0001,
        batch_size=32,
        gradient_clip=1.0,
        weight_decay=1e-5,
        seed=123,
        epochs_max=int(args.epochs),
        early_stopping_patience=int(args.patience),
    )


class WindowCache:
    def __init__(self, *, frames: dict[str, pd.DataFrame], split: dict[str, Any], args: argparse.Namespace):
        self.frames = frames
        self.split = split
        self.args = args
        self.cache: dict[tuple[str, int, int], tuple[np.ndarray, np.ndarray]] = {}

    def get(self, split_name: str, config: TrialConfig) -> tuple[np.ndarray, np.ndarray]:
        key = (split_name, int(config.history_len), int(config.pred_horizon))
        if key in self.cache:
            return self.cache[key]
        voyages = {
            "train": self.split["train_voyages"],
            "validation": self.split["validation_voyages"],
            "test": self.split["test_voyages"],
        }[split_name]
        max_windows = {
            "train": self.args.max_train_windows,
            "validation": self.args.max_val_windows,
            "test": self.args.max_test_windows,
        }[split_name]
        stride = {
            "train": self.args.train_window_stride,
            "validation": self.args.val_window_stride,
            "test": self.args.test_window_stride,
        }[split_name]
        value = split_windows_raw(
            self.frames,
            voyages,
            history_len=int(config.history_len),
            pred_horizon=int(config.pred_horizon),
            stride=int(stride),
            max_windows=max_windows,
        )
        self.cache[key] = value
        return value


def run_task_search(
    *,
    task: SearchTask,
    frames: dict[str, pd.DataFrame],
    split: dict[str, Any],
    scaler: dict[str, float],
    output_dir: Path,
    args: argparse.Namespace,
    device: str,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    seen_scores: dict[tuple[Any, ...], dict[str, Any]] = {}
    cache = WindowCache(frames=frames, split=split, args=args)

    def objective(trial: optuna.Trial) -> float:
        config = sample_trial_config(trial, task, args)
        fingerprint = config_fingerprint(config)
        if fingerprint in seen_scores:
            original = seen_scores[fingerprint]
            row = dict(original)
            row["trial_number"] = int(trial.number)
            row["status"] = "duplicate_skipped"
            row["duplicate_of_trial"] = int(original["trial_number"])
            row["train_time_sec"] = 0.0
            rows.append(row)
            write_trial_snapshot(output_dir, task.name, rows)
            trial.set_user_attr("row", row)
            return float(row["primary_score"])
        x_train, y_train = cache.get("train", config)
        x_val, y_val = cache.get("validation", config)
        result = run_training(
            task=task,
            config=config,
            x_train_raw=x_train,
            y_train_raw=y_train,
            x_val_raw=x_val,
            y_val_raw=y_val,
            scaler=scaler,
            device=device,
            trial_time_limit_sec=args.trial_time_limit_sec,
        )
        row = dict(result["row"])
        row["trial_number"] = int(trial.number)
        row["status"] = "completed"
        row["duplicate_of_trial"] = ""
        rows.append(row)
        seen_scores[fingerprint] = row
        write_trial_snapshot(output_dir, task.name, rows)
        trial.set_user_attr("row", row)
        return float(row["primary_score"])

    study = optuna.create_study(direction="minimize", study_name=f"spline_1s_{task.name}")
    study.optimize(objective, n_trials=int(task.n_trials), show_progress_bar=False)
    trials_df = pd.DataFrame(rows).sort_values("primary_score").reset_index(drop=True)
    trials_path = output_dir / f"hparam_trials_{task.name}.csv"
    trials_df.to_csv(trials_path, index=False, encoding="utf-8-sig")
    candidate_df = trials_df
    if "status" in trials_df.columns:
        non_duplicate_df = trials_df[trials_df["status"] != "duplicate_skipped"]
        if len(non_duplicate_df) > 0:
            candidate_df = non_duplicate_df
    best_row = candidate_df.iloc[0].to_dict()
    best_config = trial_config_from_row(best_row, task=task, args=args)
    x_train, y_train = cache.get("train", best_config)
    x_val, y_val = cache.get("validation", best_config)
    best_result = run_training(
        task=task,
        config=best_config,
        x_train_raw=x_train,
        y_train_raw=y_train,
        x_val_raw=x_val,
        y_val_raw=y_val,
            scaler=scaler,
            device=device,
            trial_time_limit_sec=args.trial_time_limit_sec,
        )
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{task.name}_best.pt"
    torch.save(
        {
            "model_state": best_result["state"],
            "config": asdict(best_config),
            "scaler": scaler,
            "data_label": DATA_LABEL,
            "data_caveat": DATA_CAVEAT,
            "task": asdict(task),
        },
        model_path,
    )
    write_json(output_dir / f"best_configs_{task.name}.json", {"config": asdict(best_config), "model_path": str(model_path)})

    seed_rows: list[dict[str, Any]] = []
    top_rows = candidate_df.head(min(3, len(candidate_df))).to_dict(orient="records")
    seeds = [42] if bool(args.smoke) else [42, 123, 2026]
    for rank, row in enumerate(top_rows, start=1):
        for seed in seeds:
            cfg = trial_config_from_row(row, task=task, seed=seed, args=args)
            x_train, y_train = cache.get("train", cfg)
            x_val, y_val = cache.get("validation", cfg)
            result = run_training(
                task=task,
                config=cfg,
                x_train_raw=x_train,
                y_train_raw=y_train,
                x_val_raw=x_val,
                y_val_raw=y_val,
                scaler=scaler,
                device=device,
                trial_time_limit_sec=args.trial_time_limit_sec,
            )
            seed_row = dict(result["row"])
            seed_row["config_rank"] = rank
            seed_rows.append(seed_row)
    seed_df = pd.DataFrame(seed_rows)
    seed_df.to_csv(output_dir / f"best_seed_check_{task.name}.csv", index=False, encoding="utf-8-sig")

    test_cfg = best_config
    x_test_raw, y_test_raw = cache.get("test", test_cfg)
    x_test_norm, _ = normalize_xy(x_test_raw, y_test_raw, scaler)
    model = best_result["model"].to(device)
    model.eval()
    pred_chunks: list[np.ndarray] = []
    with torch.no_grad():
        loader = DataLoader(torch.as_tensor(x_test_norm), batch_size=max(int(test_cfg.batch_size), 1) * 4, shuffle=False)
        for x_batch in loader:
            pred_chunks.append(model(x_batch.to(device)).detach().cpu().numpy())
    y_pred_kw = inverse_y(np.vstack(pred_chunks), scaler)
    metrics = aggregate_metrics(y_test_raw, y_pred_kw)
    metrics_rows = []
    for step in build_horizon_steps(task):
        metrics_rows.append(
            {
                "model": "LSTM",
                "task": task.name,
                "horizon_step": step,
                "horizon_seconds": step,
                "MAE": metrics[f"MAE_h{step}"],
                "RMSE": metrics[f"RMSE_h{step}"],
                "WAPE": metrics[f"WAPE_h{step}"],
                "Bias": metrics[f"Bias_h{step}"],
            }
        )
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(output_dir / f"metrics_by_horizon_{task.name}.csv", index=False, encoding="utf-8-sig")

    compare_rows = list(metrics_rows)
    for model_name, pred in baseline_predictions(x_test_raw, int(task.pred_horizon)).items():
        model_metrics = horizon_metrics(y_test_raw, pred)
        for step in build_horizon_steps(task):
            compare_rows.append(
                {
                    "model": model_name,
                    "task": task.name,
                    "horizon_step": step,
                    "horizon_seconds": step,
                    "MAE": model_metrics[f"MAE_h{step}"],
                    "RMSE": model_metrics[f"RMSE_h{step}"],
                    "WAPE": model_metrics[f"WAPE_h{step}"],
                    "Bias": model_metrics[f"Bias_h{step}"],
                }
            )
    compare_df = pd.DataFrame(compare_rows)
    compare_df.to_csv(output_dir / f"baseline_compare_{task.name}.csv", index=False, encoding="utf-8-sig")
    write_task_figures(task=task, output_dir=output_dir, y_true=y_test_raw, y_pred=y_pred_kw, compare_df=compare_df)
    write_per_voyage_test_figures(
        task=task,
        output_dir=output_dir,
        frames=frames,
        test_voyages=split["test_voyages"],
        config=test_cfg,
        scaler=scaler,
        model=model,
        device=device,
        stride=int(args.test_window_stride),
    )
    return {
        "task": task,
        "best_config": best_config,
        "trials": trials_df,
        "metrics": metrics_df,
        "compare": compare_df,
        "model_path": model_path,
    }


def run_task_fixed(
    *,
    task: SearchTask,
    config: TrialConfig,
    frames: dict[str, pd.DataFrame],
    split: dict[str, Any],
    scaler: dict[str, float],
    output_dir: Path,
    args: argparse.Namespace,
    device: str,
) -> dict[str, Any]:
    cache = WindowCache(frames=frames, split=split, args=args)
    x_train, y_train = cache.get("train", config)
    x_val, y_val = cache.get("validation", config)
    result = run_training(
        task=task,
        config=config,
        x_train_raw=x_train,
        y_train_raw=y_train,
        x_val_raw=x_val,
        y_val_raw=y_val,
        scaler=scaler,
        device=device,
        trial_time_limit_sec=args.trial_time_limit_sec,
    )
    fixed_row = dict(result["row"])
    fixed_row["mode"] = "fixed_hyperparameters"
    pd.DataFrame([fixed_row]).to_csv(
        output_dir / f"fixed_validation_metrics_{task.name}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{task.name}_fixed.pt"
    torch.save(
        {
            "model_state": result["state"],
            "config": asdict(config),
            "scaler": scaler,
            "data_label": DATA_LABEL,
            "data_caveat": DATA_CAVEAT,
            "task": asdict(task),
            "mode": "fixed_hyperparameters",
        },
        model_path,
    )
    write_json(
        output_dir / f"fixed_config_{task.name}.json",
        {"config": asdict(config), "model_path": str(model_path), "validation_row": fixed_row},
    )

    x_test_raw, y_test_raw = cache.get("test", config)
    x_test_norm, _ = normalize_xy(x_test_raw, y_test_raw, scaler)
    model = result["model"].to(device)
    model.eval()
    pred_chunks: list[np.ndarray] = []
    with torch.no_grad():
        loader = DataLoader(torch.as_tensor(x_test_norm), batch_size=max(int(config.batch_size), 1) * 4, shuffle=False)
        for x_batch in loader:
            pred_chunks.append(model(x_batch.to(device)).detach().cpu().numpy())
    y_pred_kw = inverse_y(np.vstack(pred_chunks), scaler)
    metrics = aggregate_metrics(y_test_raw, y_pred_kw)
    metrics_rows = []
    for step in build_horizon_steps(task):
        metrics_rows.append(
            {
                "model": "LSTM",
                "task": task.name,
                "horizon_step": step,
                "horizon_seconds": step,
                "MAE": metrics[f"MAE_h{step}"],
                "RMSE": metrics[f"RMSE_h{step}"],
                "WAPE": metrics[f"WAPE_h{step}"],
                "Bias": metrics[f"Bias_h{step}"],
            }
        )
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(output_dir / f"metrics_by_horizon_{task.name}.csv", index=False, encoding="utf-8-sig")

    compare_rows = list(metrics_rows)
    for model_name, pred in baseline_predictions(x_test_raw, int(task.pred_horizon)).items():
        model_metrics = horizon_metrics(y_test_raw, pred)
        for step in build_horizon_steps(task):
            compare_rows.append(
                {
                    "model": model_name,
                    "task": task.name,
                    "horizon_step": step,
                    "horizon_seconds": step,
                    "MAE": model_metrics[f"MAE_h{step}"],
                    "RMSE": model_metrics[f"RMSE_h{step}"],
                    "WAPE": model_metrics[f"WAPE_h{step}"],
                    "Bias": model_metrics[f"Bias_h{step}"],
                }
            )
    compare_df = pd.DataFrame(compare_rows)
    compare_df.to_csv(output_dir / f"baseline_compare_{task.name}.csv", index=False, encoding="utf-8-sig")
    write_task_figures(task=task, output_dir=output_dir, y_true=y_test_raw, y_pred=y_pred_kw, compare_df=compare_df)
    write_per_voyage_test_figures(
        task=task,
        output_dir=output_dir,
        frames=frames,
        test_voyages=split["test_voyages"],
        config=config,
        scaler=scaler,
        model=model,
        device=device,
        stride=int(args.test_window_stride),
    )
    return {
        "task": task,
        "best_config": config,
        "fixed_validation": pd.DataFrame([fixed_row]),
        "metrics": metrics_df,
        "compare": compare_df,
        "model_path": model_path,
    }


def write_task_figures(
    *,
    task: SearchTask,
    output_dir: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    compare_df: pd.DataFrame,
) -> None:
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    limit = min(1000, y_true.shape[0])
    x_axis = np.arange(limit)
    for step in build_horizon_steps(task):
        idx = step - 1
        plt.figure(figsize=(10, 4))
        plt.plot(x_axis, y_true[:limit, idx], label=f"actual h{step}", linewidth=0.8)
        plt.plot(x_axis, y_pred[:limit, idx], label=f"LSTM h{step}", linewidth=0.8)
        plt.xlabel("test window index")
        plt.ylabel("load_total_kw")
        plt.title(f"{task.name} h{step} prediction on spline-reconstructed 1s data")
        plt.grid(alpha=0.25)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(fig_dir / f"{task.name}_prediction_h{step}.png", dpi=150)
        plt.close()

    plt.figure(figsize=(8, 4))
    for model_name, chunk in compare_df.groupby("model", sort=False):
        plt.plot(chunk["horizon_seconds"], chunk["MAE"], marker="o", label=model_name)
    plt.xlabel("horizon (s)")
    plt.ylabel("MAE (kW)")
    plt.title(f"{task.name} error vs horizon")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / f"{task.name}_error_vs_horizon.png", dpi=150)
    plt.close()


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value))


def test_voyage_figure_path(output_dir: Path, task: SearchTask, voyage_id: str) -> Path:
    return Path(output_dir) / "figures" / f"{task.name}_test_voyages" / f"{task.name}_{_safe_name(voyage_id)}_prediction_h1_h6.png"


def write_one_voyage_prediction_figure(
    *,
    task: SearchTask,
    output_dir: Path,
    voyage_id: str,
    decision_time_s: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Path:
    path = test_voyage_figure_path(output_dir, task, voyage_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    truth = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    decision_time = np.asarray(decision_time_s, dtype=float).reshape(-1)
    if truth.ndim != 2 or pred.ndim != 2:
        raise ValueError("y_true and y_pred must be 2D arrays.")
    if truth.shape != pred.shape:
        raise ValueError(f"y_true and y_pred shape mismatch: {truth.shape} vs {pred.shape}")
    if len(decision_time) != truth.shape[0]:
        raise ValueError(f"decision_time_s length {len(decision_time)} does not match rows {truth.shape[0]}")

    plot_steps = [step for step in (1, 6) if step <= int(task.pred_horizon)]
    if not plot_steps:
        plot_steps = build_horizon_steps(task)[:1]
    metrics = horizon_metrics(truth, pred)
    fig, axes_obj = plt.subplots(len(plot_steps), 1, figsize=(14, 3.5 * len(plot_steps)), sharex=False)
    axes = np.asarray(axes_obj).reshape(-1)
    for ax, step in zip(axes, plot_steps):
        idx = step - 1
        x_min = (decision_time + float(idx)) / 60.0
        ax.plot(x_min, truth[:, idx], color="black", linewidth=0.7, alpha=0.80, label=f"actual h{step}")
        ax.plot(x_min, pred[:, idx], color="tab:blue", linewidth=0.7, alpha=0.85, label=f"LSTM h{step}")
        ax.set_ylabel("load_total_kw")
        ax.set_title(
            f"{voyage_id} h{step}: MAE={metrics[f'MAE_h{step}']:.3f} kW, "
            f"WAPE={metrics[f'WAPE_h{step}']:.3f}%"
        )
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("target time within voyage (min)")
    fig.suptitle(f"{task.name} per-voyage test prediction on spline-reconstructed 1s data", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _predict_kw(
    *,
    model: nn.Module,
    x_raw: np.ndarray,
    y_raw: np.ndarray,
    config: TrialConfig,
    scaler: dict[str, float],
    device: str,
) -> np.ndarray:
    x_norm, _ = normalize_xy(x_raw, y_raw, scaler)
    pred_chunks: list[np.ndarray] = []
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        loader = DataLoader(torch.as_tensor(x_norm), batch_size=max(int(config.batch_size), 1) * 4, shuffle=False)
        for x_batch in loader:
            pred_chunks.append(model(x_batch.to(device)).detach().cpu().numpy())
    return inverse_y(np.vstack(pred_chunks), scaler)


def write_per_voyage_test_figures(
    *,
    task: SearchTask,
    output_dir: Path,
    frames: dict[str, pd.DataFrame],
    test_voyages: list[str],
    config: TrialConfig,
    scaler: dict[str, float],
    model: nn.Module,
    device: str,
    stride: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    stride = max(1, int(stride))
    for voyage_id in test_voyages:
        frame = frames[voyage_id]
        load = frame["load_total_kw"].astype(float).to_numpy()
        stop = len(load) - int(config.pred_horizon) + 1
        positions = np.arange(int(config.history_len), stop, stride, dtype=int)
        x_raw, y_true = build_windows_for_series(
            load,
            history_len=int(config.history_len),
            pred_horizon=int(config.pred_horizon),
            stride=stride,
            max_windows=None,
        )
        if len(x_raw) == 0:
            continue
        positions = positions[: len(x_raw)]
        decision_time_s = frame["time_s"].astype(float).to_numpy()[positions]
        y_pred = _predict_kw(
            model=model,
            x_raw=x_raw,
            y_raw=y_true,
            config=config,
            scaler=scaler,
            device=device,
        )
        figure_path = write_one_voyage_prediction_figure(
            task=task,
            output_dir=output_dir,
            voyage_id=voyage_id,
            decision_time_s=decision_time_s,
            y_true=y_true,
            y_pred=y_pred,
        )
        metrics = horizon_metrics(y_true, y_pred)
        for step in build_horizon_steps(task):
            rows.append(
                {
                    "voyage_id": voyage_id,
                    "task": task.name,
                    "horizon_step": step,
                    "horizon_seconds": step,
                    "rows": int(len(y_true)),
                    "MAE": metrics[f"MAE_h{step}"],
                    "RMSE": metrics[f"RMSE_h{step}"],
                    "WAPE": metrics[f"WAPE_h{step}"],
                    "Bias": metrics[f"Bias_h{step}"],
                    "figure": str(figure_path),
                }
            )
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(output_dir / f"per_voyage_test_metrics_{task.name}.csv", index=False, encoding="utf-8-sig")
    return metrics_df


def write_cross_task_baseline_figure(output_dir: Path, task_name: str) -> None:
    compare_path = output_dir / f"baseline_compare_{task_name}.csv"
    if not compare_path.exists():
        return
    compare_df = pd.read_csv(compare_path)
    plt.figure(figsize=(8, 4))
    for model_name, chunk in compare_df.groupby("model", sort=False):
        plt.plot(chunk["horizon_seconds"], chunk["WAPE"], marker="o", label=model_name)
    plt.xlabel("horizon (s)")
    plt.ylabel("WAPE (%)")
    plt.title(f"LSTM vs baselines {task_name}")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "figures" / f"lstm_vs_baselines_{task_name}.png", dpi=150)
    plt.close()


def reference_rows() -> list[dict[str, str]]:
    return [
        {
            "paper_id": "ref1",
            "paper_title": "Short-Term Load Forecasting for Smart Home Appliances with Sequence to Sequence Learning",
            "link": "https://arxiv.org/abs/2106.15348",
            "task_type": "short-term appliance load forecasting",
            "data_domain": "GreenD appliance-level electricity consumption",
            "sampling_interval": "1 Hz source; smoothed to 10 min in the described setup",
            "interpolation_or_preprocessing": "smoothing / aggregation to T=10 min",
            "history_length_points": "144",
            "history_time": "24 h",
            "prediction_horizon_points": "not_reported",
            "prediction_time": "next hour",
            "lstm_layers": "2",
            "hidden_units": "not_reported",
            "dropout": "not_reported",
            "batch_size": "not_reported",
            "learning_rate": "not_reported",
            "optimizer": "not_reported",
            "loss": "MSE",
            "epochs": "not_reported",
            "early_stopping": "not_reported",
            "metrics": "RMSE; MAE; NRMSE",
            "directly_applicable_to_this_project": "no",
            "usable_hparam_clue": "use physical-time history windows and evaluate the whole horizon",
            "limitations": "smart-home appliance data, not shipboard spline-reconstructed 1s load",
        },
        {
            "paper_id": "ref2",
            "paper_title": "Load Forecasting on A Highly Sparse Electrical Load Dataset Using Gaussian Interpolation",
            "link": "https://arxiv.org/abs/2508.14069",
            "task_type": "load forecasting with sparse data interpolation",
            "data_domain": "hourly sparse electrical load",
            "sampling_interval": "hourly sparse",
            "interpolation_or_preprocessing": "linear; polynomial; spline; moving average; Gaussian interpolation",
            "history_length_points": "not_reported",
            "history_time": "not_reported",
            "prediction_horizon_points": "not_reported",
            "prediction_time": "not_reported",
            "lstm_layers": "not_reported",
            "hidden_units": "not_reported",
            "dropout": "not_reported",
            "batch_size": "not_reported",
            "learning_rate": "not_reported",
            "optimizer": "not_reported",
            "loss": "not_reported",
            "epochs": "not_reported",
            "early_stopping": "not_reported",
            "metrics": "not_reported",
            "directly_applicable_to_this_project": "no",
            "usable_hparam_clue": "compare LSTM against naive baselines after interpolation",
            "limitations": "hourly sparse load, not measured 1s ship load",
        },
        {
            "paper_id": "ref3",
            "paper_title": "Neural NILM: Deep Neural Networks Applied to Energy Disaggregation",
            "link": "https://arxiv.org/abs/1507.06594",
            "task_type": "energy disaggregation",
            "data_domain": "household electricity",
            "sampling_interval": "seconds to low-frequency electricity series",
            "interpolation_or_preprocessing": "not_reported",
            "history_length_points": "not_reported",
            "history_time": "not_reported",
            "prediction_horizon_points": "not_applicable",
            "prediction_time": "not_applicable",
            "lstm_layers": "not_reported",
            "hidden_units": "not_reported",
            "dropout": "not_reported",
            "batch_size": "not_reported",
            "learning_rate": "not_reported",
            "optimizer": "not_reported",
            "loss": "not_reported",
            "epochs": "not_reported",
            "early_stopping": "not_reported",
            "metrics": "not_reported",
            "directly_applicable_to_this_project": "no",
            "usable_hparam_clue": "LSTM is applicable to electricity time series, but task-specific windows are required",
            "limitations": "NILM task, not future load forecasting",
        },
        {
            "paper_id": "ref4",
            "paper_title": "Deep Learning Based Energy Disaggregation and On/Off Detection of Household Appliances",
            "link": "https://arxiv.org/abs/1908.00941",
            "task_type": "NILM and on/off detection",
            "data_domain": "household energy datasets including REFIT",
            "sampling_interval": "not_reported",
            "interpolation_or_preprocessing": "not_reported",
            "history_length_points": "not_reported",
            "history_time": "receptive field concept reported by model family",
            "prediction_horizon_points": "not_applicable",
            "prediction_time": "target field concept reported by model family",
            "lstm_layers": "not_reported",
            "hidden_units": "not_reported",
            "dropout": "not_reported",
            "batch_size": "not_reported",
            "learning_rate": "not_reported",
            "optimizer": "not_reported",
            "loss": "not_reported",
            "epochs": "not_reported",
            "early_stopping": "not_reported",
            "metrics": "not_reported",
            "directly_applicable_to_this_project": "no",
            "usable_hparam_clue": "separate receptive/history field from target/forecast field",
            "limitations": "disaggregation/detection task, not load forecasting",
        },
        {
            "paper_id": "ref5",
            "paper_title": "Application Research of Spline Interpolation and ARIMA in the Field of Stock Market Forecasting",
            "link": "https://arxiv.org/abs/2311.10759",
            "task_type": "stock time-series forecasting",
            "data_domain": "stock market",
            "sampling_interval": "not_reported",
            "interpolation_or_preprocessing": "cubic spline interpolation with ARIMA-style forecasting",
            "history_length_points": "not_applicable",
            "history_time": "not_applicable",
            "prediction_horizon_points": "not_applicable",
            "prediction_time": "not_applicable",
            "lstm_layers": "not_applicable",
            "hidden_units": "not_applicable",
            "dropout": "not_applicable",
            "batch_size": "not_applicable",
            "learning_rate": "not_applicable",
            "optimizer": "not_applicable",
            "loss": "not_applicable",
            "epochs": "not_applicable",
            "early_stopping": "not_applicable",
            "metrics": "not_reported",
            "directly_applicable_to_this_project": "no",
            "usable_hparam_clue": "spline interpolation can be used as offline smoothing/filling before forecasting",
            "limitations": "not electricity load and not LSTM",
        },
        {
            "paper_id": "ref6",
            "paper_title": "IOP hydrogen vessel LSTM multi-step forecasting paper",
            "link": "https://iopscience.iop.org/article/10.1088/1742-6596/2876/1/012052",
            "task_type": "ship or hydrogen-vessel-related load forecasting",
            "data_domain": "not_reported",
            "sampling_interval": "not_reported",
            "interpolation_or_preprocessing": "not_reported",
            "history_length_points": "not_reported",
            "history_time": "not_reported",
            "prediction_horizon_points": "not_reported",
            "prediction_time": "not_reported",
            "lstm_layers": "not_reported",
            "hidden_units": "not_reported",
            "dropout": "not_reported",
            "batch_size": "not_reported",
            "learning_rate": "not_reported",
            "optimizer": "not_reported",
            "loss": "not_reported",
            "epochs": "not_reported",
            "early_stopping": "not_reported",
            "metrics": "not_reported",
            "directly_applicable_to_this_project": "limited",
            "usable_hparam_clue": "if verified as 1s and Np=5, it only supports short-horizon settings",
            "limitations": "does not replace h60/h180 evaluation and requires direct paper verification before strong claims",
        },
    ]


def write_reference_files(output_dir: Path) -> None:
    rows = reference_rows()
    pd.DataFrame(rows).to_csv(output_dir / "reference_hparam_table.csv", index=False, encoding="utf-8-sig")
    lines = [
        "# Reference Hyperparameter Evidence",
        "",
        "This table is for hyperparameter clues only. It is not a literature review and not evidence that spline-reconstructed 1 s ship load is valid online ground truth.",
        "",
        "| paper_id | usable_hparam_clue | limitations |",
        "|---|---|---|",
    ]
    for row in rows:
        lines.append(f"| {row['paper_id']} | {row['usable_hparam_clue']} | {row['limitations']} |")
    (output_dir / "reference_hparam_evidence.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(output_dir: Path, results: dict[str, dict[str, Any]], *, smoke: bool, fixed: bool = False) -> None:
    def best_line(task_name: str) -> str:
        cfg = results[task_name]["best_config"]
        return json.dumps(asdict(cfg), ensure_ascii=False)

    task_setting_lines = []
    for result in results.values():
        task = result["task"]
        history_text = (
            f"fixed_history_len={int(task.fixed_history_len)}"
            if task.fixed_history_len is not None
            else "history_len searched"
        )
        task_setting_lines.append(
            f"- {task.name}: dt=1s, {history_text}, pred_horizon={int(task.pred_horizon)}, "
            f"forecast_time={int(task.pred_horizon)}s."
        )
    best_config_lines: list[str] = []
    for task_name in results:
        best_config_lines.extend([f"## Best {task_name} Configuration", "", best_line(task_name), ""])
    compare_files = ", ".join(f"`baseline_compare_{task_name}.csv`" for task_name in results)
    metrics_files = ", ".join(f"`metrics_by_horizon_{task_name}.csv`" for task_name in results)

    lines = [
        "# Spline 1s LSTM Hyperparameter Search Report",
        "",
        f"Mode: {'fixed hyperparameter train/validation/test run' if fixed else ('smoke verification run' if smoke else 'full search run')}",
        "",
        "## 1. Data",
        "",
        f"The experiment uses `{DEFAULT_SOURCE_DIR}`: {DATA_LABEL}.",
        "",
        "## 2. Why This Is Not Measured 1s Data",
        "",
        DATA_CAVEAT,
        "",
        "## 3. Why Run The Search",
        "",
        "The purpose is to find best hyperparameters on spline-reconstructed 1s data for diagnostics. It is not a claim of valid online 1s forecasting capability.",
        "",
        "## 4. Reference Clues",
        "",
        "The reference table records only sampling, preprocessing, LSTM, horizon, and metric clues. Missing values are marked `not_reported` or `not_applicable`.",
        "",
        "## 5. Task Settings",
        "",
        *task_setting_lines,
        "",
        "## 6. Search Space",
        "",
        "Search includes task-specific fixed history lengths when configured; otherwise history_len [60, 180, 300, 540]. It searches hidden_size [64, 128, 256], LSTM layers [1, 2, 3], dropout [0.0, 0.1, 0.2, 0.3], MLP heads [128], [256], [256, 128], loss [MSE, Huber], Adam learning rates [1e-3, 5e-4, 2e-4, 1e-4], batch size [32, 64, 128], gradient clip [0.5, 1.0, 5.0], weight_decay [0.0, 1e-6, 1e-5], and seed [42, 123].",
        "",
        "## 7. Search Strategy",
        "",
        "Optuna minimizes mean validation WAPE across the full task horizon, not h1 only.",
        "",
        *best_config_lines,
        "",
        "## 10-11. LSTM Versus Baselines",
        "",
        f"See {compare_files} for current-hold, last-slope, moving-average hold, EMA hold, and LSTM metrics.",
        "",
        "## 12. Spline Regularity",
        "",
        "Very small errors or strong naive baselines are expected diagnostics because the labels are smooth spline reconstructions using future endpoints. This is reported as a limitation, not a stop condition.",
        "",
        "## 13. h1 Versus Long Horizons",
        "",
        f"See {metrics_files} and the error-vs-horizon figures.",
        "",
        "## 14. Use Of Current Best Hyperparameters",
        "",
        "They can be used for later diagnostics on the same spline-reconstructed data source. They should not be treated as validated real 1s predictors.",
        "",
        "## 15. Main Paper Recommendation",
        "",
        "Do not use this as main paper validity evidence.",
        "",
        "## 16. Appendix Recommendation",
        "",
        "Use only as appendix or sensitivity evidence, with the offline spline caveat stated explicitly.",
        "",
        "## 17. Data Needed For Real Online Forecasting",
        "",
        "A real online forecasting claim requires native measured 1s load, causal acquisition timestamps, no future endpoint dependence, and a split built from those measured data.",
        "",
        "## Conclusion Boundary",
        "",
        "- A. best hyperparameters on spline-reconstructed 1s data: yes, within the run mode above.",
        "- B. valid online 1s load forecasting capability: no.",
    ]
    (output_dir / "REPORT_SPLINE_1S_LSTM_HPARAM_SEARCH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run isolated Optuna LSTM search on spline-reconstructed 1s load.")
    parser.add_argument("--source_dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--split_json", type=Path, default=DEFAULT_SPLIT_JSON)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--n_trials_taskA", type=int, default=None)
    parser.add_argument("--n_trials_taskB", type=int, default=None)
    parser.add_argument("--run_taskC_30_to_6", action="store_true")
    parser.add_argument("--n_trials_taskC", type=int, default=None)
    parser.add_argument("--run_fixed_taskC_30_to_6", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--trial_time_limit_sec", type=float, default=1800)
    parser.add_argument("--max_train_windows", type=int, default=120000)
    parser.add_argument("--max_val_windows", type=int, default=30000)
    parser.add_argument("--max_test_windows", type=int, default=None)
    parser.add_argument("--train_window_stride", type=int, default=10)
    parser.add_argument("--val_window_stride", type=int, default=5)
    parser.add_argument("--test_window_stride", type=int, default=1)
    parser.add_argument("--include_history_900", action="store_true")
    parser.add_argument("--no_reference_refresh", action="store_true")
    return parser


def apply_smoke_overrides(args: argparse.Namespace) -> None:
    if not bool(args.smoke):
        return
    args.n_trials_taskA = 1 if args.n_trials_taskA is None else args.n_trials_taskA
    args.n_trials_taskB = 1 if args.n_trials_taskB is None else args.n_trials_taskB
    args.n_trials_taskC = 1 if args.n_trials_taskC is None else args.n_trials_taskC
    args.epochs = min(int(args.epochs), 2)
    args.patience = min(int(args.patience), 1)
    args.max_train_windows = min(int(args.max_train_windows), 512)
    args.max_val_windows = min(int(args.max_val_windows), 256)
    args.max_test_windows = 256 if args.max_test_windows is None else min(int(args.max_test_windows), 256)
    args.train_window_stride = max(int(args.train_window_stride), 30)
    args.val_window_stride = max(int(args.val_window_stride), 30)
    args.test_window_stride = max(int(args.test_window_stride), 30)


def build_tasks_from_args(args: argparse.Namespace) -> dict[str, SearchTask]:
    if bool(getattr(args, "run_fixed_taskC_30_to_6", False)):
        return {
            "taskC_30_to_6": SearchTask(
                name="taskC_30_to_6",
                pred_horizon=6,
                n_trials=0,
                fixed_history_len=30,
            )
        }
    if bool(getattr(args, "run_taskC_30_to_6", False)):
        n_trials = 10 if args.n_trials_taskC is None else int(args.n_trials_taskC)
        return {
            "taskC_30_to_6": SearchTask(
                name="taskC_30_to_6",
                pred_horizon=6,
                n_trials=n_trials,
                fixed_history_len=30,
            )
        }

    tasks = build_default_tasks()
    if args.n_trials_taskA is not None:
        tasks["taskA"] = SearchTask(name="taskA", pred_horizon=60, n_trials=int(args.n_trials_taskA))
    if args.n_trials_taskB is not None:
        tasks["taskB"] = SearchTask(name="taskB", pred_horizon=180, n_trials=int(args.n_trials_taskB))
    return tasks


def main() -> None:
    args = build_parser().parse_args()
    apply_smoke_overrides(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_reference_files(output_dir)
    split = load_split(Path(args.split_json))
    all_voyages = split["train_voyages"] + split["validation_voyages"] + split["test_voyages"]
    frames = load_voyage_frames(Path(args.source_dir), all_voyages)
    write_data_check(output_dir, Path(args.source_dir), split, frames)
    scaler = fit_train_scaler(frames, split["train_voyages"])
    tasks = build_tasks_from_args(args)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    results: dict[str, dict[str, Any]] = {}
    if bool(getattr(args, "run_fixed_taskC_30_to_6", False)):
        task_name = "taskC_30_to_6"
        results[task_name] = run_task_fixed(
            task=tasks[task_name],
            config=build_fixed_taskC_config(args),
            frames=frames,
            split=split,
            scaler=scaler,
            output_dir=output_dir,
            args=args,
            device=device,
        )
        write_cross_task_baseline_figure(output_dir, task_name)
    else:
        for task_name in tasks:
            results[task_name] = run_task_search(
                task=tasks[task_name],
                frames=frames,
                split=split,
                scaler=scaler,
                output_dir=output_dir,
                args=args,
                device=device,
            )
            write_cross_task_baseline_figure(output_dir, task_name)
    write_report(output_dir, results, smoke=bool(args.smoke), fixed=bool(getattr(args, "run_fixed_taskC_30_to_6", False)))
    run_summary = {
        "output_dir": str(output_dir.resolve()),
        "source_dir": str(Path(args.source_dir).resolve()),
        "split_json": str(Path(args.split_json).resolve()),
        "smoke": bool(args.smoke),
        "device": device,
        "scaler_fit_scope": "train_voyages_only",
        "online_feasible": False,
        "uses_future_endpoint": True,
        "not_measured_1s": True,
        "tasks": {name: asdict(task) for name, task in tasks.items()},
    }
    write_json(output_dir / "run_summary.json", run_summary)
    print(json.dumps(run_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
