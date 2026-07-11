from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import random
import re
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import optuna
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[2]
FORECASTING_ROOT = ROOT / "src" / "forecasting"
if str(FORECASTING_ROOT) not in sys.path:
    sys.path.insert(0, str(FORECASTING_ROOT))

from millisecond_multistep_lstm import (  # noqa: E402
    ModelConfig,
    SequenceToVectorLSTM,
    StandardScaler1D,
    WindowSet,
    baseline_forecasts,
    build_windows,
    fit_standard_scaler,
    metrics_by_horizon,
)


SEARCH_SPACE_VERSION = "millisecond_10ms_lstm_search_v1"


@dataclass(frozen=True)
class TrialConfig:
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
    max_epochs: int
    patience: int


@dataclass(frozen=True)
class EpochResult:
    score: float
    mae: float


@dataclass(frozen=True)
class TrainingLoopResult:
    best_score: float
    best_mae: float
    best_epoch: int
    epochs_completed: int
    stopped_by_timeout: bool
    stopped_by_early_stopping: bool


@dataclass(frozen=True)
class CandidateResult:
    config_id: str
    validation_wape: tuple[float, ...]
    validation_mae: tuple[float, ...]


@dataclass
class TrainingArtifact:
    config: TrialConfig
    model_state_dict: dict[str, torch.Tensor]
    best_score: float
    best_mae: float
    best_epoch: int
    epochs_completed: int
    elapsed_s: float
    stopped_by_timeout: bool
    stopped_by_early_stopping: bool
    validation_metrics: list[dict[str, object]]
    learning_curve: list[dict[str, float | int]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bounded 10 ms millisecond LSTM Optuna search")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/millisecond_10ms"))
    parser.add_argument(
        "--split-path", type=Path, default=Path("outputs/config/millisecond_10ms_split_721.json")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/lstm_millisecond_10ms_30_to_6")
    )
    parser.add_argument("--n-trials", type=int, default=24)
    parser.add_argument("--study-timeout-s", type=int, default=3600)
    parser.add_argument("--trial-timeout-s", type=int, default=180)
    parser.add_argument("--max-epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--history-steps", type=int, default=30)
    parser.add_argument("--prediction-steps", type=int, default=6)
    parser.add_argument("--robust-top-k", type=int, default=3)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 20260710])
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume-run", type=str)
    return parser


def sample_trial_config(
    trial: optuna.Trial,
    *,
    fixed_seed: int,
    max_epochs: int = 25,
    patience: int = 4,
) -> TrialConfig:
    num_layers = int(trial.suggest_categorical("num_layers", [1, 2, 3]))
    dropout = (
        0.0
        if num_layers == 1
        else float(trial.suggest_categorical("dropout", [0.0, 0.1, 0.2, 0.3]))
    )
    head_map = {"none": (), "64": (64,), "128": (128,), "128-64": (128, 64)}
    return TrialConfig(
        hidden_size=int(trial.suggest_categorical("hidden_size", [32, 64, 128, 256])),
        num_layers=num_layers,
        dropout=dropout,
        mlp_head=head_map[str(trial.suggest_categorical("mlp_head", list(head_map)))],
        loss=str(trial.suggest_categorical("loss", ["MSE", "Huber"])),
        learning_rate=float(trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True)),
        batch_size=int(trial.suggest_categorical("batch_size", [64, 128, 256])),
        gradient_clip=float(trial.suggest_categorical("gradient_clip", [0.5, 1.0, 5.0])),
        weight_decay=float(trial.suggest_categorical("weight_decay", [0.0, 1e-6, 1e-5, 1e-4])),
        seed=int(fixed_seed),
        max_epochs=int(max_epochs),
        patience=int(patience),
    )


def set_deterministic_seed(seed: int) -> dict[str, object]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        deterministic = True
    except Exception:
        deterministic = False
    return {"seed": int(seed), "deterministic_algorithms_requested": deterministic}


def run_training_loop(
    *,
    state: object,
    max_epochs: int,
    patience: int,
    min_delta: float,
    trial_timeout_s: float,
    clock: Callable[[], float],
    run_epoch: Callable[[object, int], EpochResult],
) -> TrainingLoopResult:
    started = clock()
    best_score = float("inf")
    best_mae = float("inf")
    best_epoch = 0
    stale_epochs = 0
    epochs_completed = 0
    timed_out = False
    early_stopped = False
    for epoch in range(1, max_epochs + 1):
        if clock() - started >= trial_timeout_s:
            timed_out = True
            break
        result = run_epoch(state, epoch)
        epochs_completed = epoch
        if result.score < best_score - min_delta:
            best_score = float(result.score)
            best_mae = float(result.mae)
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        if clock() - started >= trial_timeout_s:
            timed_out = True
            break
        if stale_epochs >= patience:
            early_stopped = True
            break
    return TrainingLoopResult(
        best_score=best_score,
        best_mae=best_mae,
        best_epoch=best_epoch,
        epochs_completed=epochs_completed,
        stopped_by_timeout=timed_out,
        stopped_by_early_stopping=early_stopped,
    )


def _persist_trials(study: optuna.Study, trial_csv: Path) -> None:
    trial_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary = trial_csv.with_suffix(trial_csv.suffix + ".tmp")
    study.trials_dataframe().to_csv(temporary, index=False)
    temporary.replace(trial_csv)


def run_study(
    *,
    objective: Callable[[optuna.Trial], float],
    storage_path: Path,
    study_name: str,
    n_trials: int,
    timeout_s: float,
    sampler_seed: int,
    trial_csv: Path,
    study_user_attrs: Mapping[str, object] | None = None,
) -> optuna.Study:
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{storage_path.resolve().as_posix()}"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=sampler_seed),
    )
    expected_attrs = dict(study_user_attrs or {})
    for key, expected in expected_attrs.items():
        stored_attrs = study.user_attrs
        if key in stored_attrs and stored_attrs[key] != expected:
            stored = stored_attrs[key]
            close_study_storage(study)
            raise ValueError(
                f"Study metadata mismatch for {key}: stored={stored!r}, requested={expected!r}"
            )
        study.set_user_attr(key, expected)
    remaining = max(int(n_trials) - len(study.trials), 0)
    if remaining:
        study.optimize(
            objective,
            n_trials=remaining,
            timeout=float(timeout_s),
            n_jobs=1,
            callbacks=[lambda current, trial: _persist_trials(current, trial_csv)],
            catch=(RuntimeError,),
        )
    _persist_trials(study, trial_csv)
    return study


def close_study_storage(study: optuna.Study) -> None:
    """Release Optuna's SQLite handles so Windows can move/delete the database."""
    storage = study._storage
    storage.remove_session()
    backend = getattr(storage, "_backend", storage)
    engine = getattr(backend, "engine", None)
    if engine is not None:
        engine.dispose()


def load_sequences(combined_csv: Path, split: str) -> dict[str, np.ndarray]:
    required = ["split", "sequence_id", "time_ms", "load_kw"]
    frame = pd.read_csv(combined_csv, usecols=required)
    missing = set(required).difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    selected = frame.loc[frame["split"] == split].copy()
    if selected.empty:
        raise ValueError(f"No rows found for split {split!r}")
    if selected[["time_ms", "load_kw"]].isna().any().any():
        raise ValueError(f"Split {split!r} contains missing time or load values")
    sequences: dict[str, np.ndarray] = {}
    for sequence_id, group in selected.groupby("sequence_id", sort=False):
        ordered = group.sort_values("time_ms", kind="stable")
        times = ordered["time_ms"].to_numpy(dtype=np.int64)
        if times.size > 1 and not np.all(np.diff(times) == 10):
            raise ValueError(f"Sequence {sequence_id!r} is not sampled at exactly 10 ms")
        values = ordered["load_kw"].to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError(f"Sequence {sequence_id!r} contains non-finite load values")
        sequences[str(sequence_id)] = values
    return sequences


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def scale_windows(windows: WindowSet, scaler: StandardScaler1D) -> WindowSet:
    return WindowSet(
        x=scaler.transform(windows.x),
        y=scaler.transform(windows.y),
        sequence_ids=windows.sequence_ids,
        target_start_indices=windows.target_start_indices,
    )


def _model_config(config: TrialConfig) -> ModelConfig:
    return ModelConfig(
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
        mlp_head=config.mlp_head,
    )


def _mean_horizon_wape(metrics: pd.DataFrame) -> float:
    return float(metrics.iloc[:-1]["wape_pct"].mean())


def train_model(
    *,
    config: TrialConfig,
    train_windows_raw: WindowSet,
    validation_windows_raw: WindowSet,
    scaler: StandardScaler1D,
    prediction_steps: int,
    device: torch.device,
    trial_timeout_s: float,
    min_delta: float = 1e-6,
) -> TrainingArtifact:
    if train_windows_raw.x.shape[0] == 0 or validation_windows_raw.x.shape[0] == 0:
        raise ValueError("Training and validation windows must both be non-empty")
    set_deterministic_seed(config.seed)
    train_windows = scale_windows(train_windows_raw, scaler)
    validation_windows = scale_windows(validation_windows_raw, scaler)
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_windows.x), torch.from_numpy(train_windows.y)),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        TensorDataset(torch.from_numpy(validation_windows.x), torch.from_numpy(validation_windows.y)),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model = SequenceToVectorLSTM(
        config=_model_config(config), prediction_steps=prediction_steps
    ).to(device)
    criterion: nn.Module = nn.MSELoss() if config.loss == "MSE" else nn.HuberLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    started = time.monotonic()
    best_score = float("inf")
    best_mae = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: list[dict[str, object]] = []
    curve: list[dict[str, float | int]] = []
    stale_epochs = 0
    timed_out = False
    early_stopped = False
    epochs_completed = 0
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for x_batch, y_batch in train_loader:
            if time.monotonic() - started >= trial_timeout_s:
                timed_out = True
                break
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x_batch)
            loss = criterion(prediction, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            train_loss_sum += float(loss.detach().cpu()) * x_batch.shape[0]
            train_count += int(x_batch.shape[0])
            if time.monotonic() - started >= trial_timeout_s:
                timed_out = True
                break
        if timed_out:
            break
        model.eval()
        scaled_predictions: list[np.ndarray] = []
        validation_complete = True
        with torch.no_grad():
            for x_batch, _ in validation_loader:
                if time.monotonic() - started >= trial_timeout_s:
                    timed_out = True
                    validation_complete = False
                    break
                output = model(x_batch.to(device, non_blocking=True))
                scaled_predictions.append(output.cpu().numpy())
                if time.monotonic() - started >= trial_timeout_s:
                    timed_out = True
                    validation_complete = False
                    break
        if not validation_complete:
            break
        prediction_kw = scaler.inverse_transform(np.concatenate(scaled_predictions, axis=0))
        validation_metrics_frame = metrics_by_horizon(validation_windows_raw.y, prediction_kw)
        score = _mean_horizon_wape(validation_metrics_frame)
        mae = float(validation_metrics_frame.iloc[-1]["mae_kw"])
        epochs_completed = epoch
        curve.append(
            {
                "epoch": epoch,
                "train_loss": train_loss_sum / max(train_count, 1),
                "validation_mean_horizon_wape_pct": score,
                "validation_aggregate_mae_kw": mae,
            }
        )
        if score < best_score - min_delta:
            best_score = score
            best_mae = mae
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            best_metrics = validation_metrics_frame.to_dict(orient="records")
            stale_epochs = 0
        else:
            stale_epochs += 1
        if timed_out:
            break
        if stale_epochs >= config.patience:
            early_stopped = True
            break
    elapsed = time.monotonic() - started
    if best_state is None:
        raise optuna.TrialPruned("trial timeout before validation")
    return TrainingArtifact(
        config=config,
        model_state_dict=best_state,
        best_score=best_score,
        best_mae=best_mae,
        best_epoch=best_epoch,
        epochs_completed=epochs_completed,
        elapsed_s=elapsed,
        stopped_by_timeout=timed_out,
        stopped_by_early_stopping=early_stopped,
        validation_metrics=best_metrics,
        learning_curve=curve,
    )


def predict_windows(
    *,
    config: TrialConfig,
    model_state_dict: Mapping[str, torch.Tensor],
    windows_raw: WindowSet,
    scaler: StandardScaler1D,
    prediction_steps: int,
    device: torch.device,
) -> np.ndarray:
    model = SequenceToVectorLSTM(
        config=_model_config(config), prediction_steps=prediction_steps
    ).to(device)
    model.load_state_dict(model_state_dict)
    model.eval()
    scaled = scale_windows(windows_raw, scaler)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(scaled.x)),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    rows: list[np.ndarray] = []
    with torch.no_grad():
        for (x_batch,) in loader:
            rows.append(model(x_batch.to(device, non_blocking=True)).cpu().numpy())
    return scaler.inverse_transform(np.concatenate(rows, axis=0))


def trial_config_from_mapping(payload: Mapping[str, object]) -> TrialConfig:
    values = dict(payload)
    values["mlp_head"] = tuple(int(value) for value in values.get("mlp_head", ()))
    return TrialConfig(**values)  # type: ignore[arg-type]


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        raise TypeError("Tensors must not be serialized to JSON")
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    temporary.replace(path)


def select_configuration(candidates: Sequence[CandidateResult]) -> CandidateResult:
    if not candidates:
        raise ValueError("At least one validation candidate is required")
    return min(
        candidates,
        key=lambda item: (
            float(np.mean(item.validation_wape)),
            float(np.mean(item.validation_mae)),
            item.config_id,
        ),
    )


def canonical_config_id(config: TrialConfig) -> str:
    payload = dataclasses.asdict(config)
    payload.pop("seed", None)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _experiment_title() -> str:
    return "dt=10 ms | history=30 (300 ms) | horizon=6 (60 ms)"


def _checkpoint_payload(
    *,
    artifact: TrainingArtifact,
    scaler: StandardScaler1D,
    dataset_hash: str,
    split_hash: str,
    source_hashes: Sequence[str],
) -> dict[str, object]:
    return {
        "model_state_dict": artifact.model_state_dict,
        "trial_config": dataclasses.asdict(artifact.config),
        "scaler": dataclasses.asdict(scaler),
        "dataset_manifest_sha256": dataset_hash,
        "split_manifest_sha256": split_hash,
        "source_sha256": list(source_hashes),
        "best_epoch": artifact.best_epoch,
        "validation_metrics": artifact.validation_metrics,
        "seed": artifact.config.seed,
        "torch_version": torch.__version__,
        "optuna_version": optuna.__version__,
        "search_space_version": SEARCH_SPACE_VERSION,
    }


def _plot_optuna(study: optuna.Study, figure_root: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_root.mkdir(parents=True, exist_ok=True)
    complete = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    if complete:
        numbers = [trial.number for trial in complete]
        values = np.asarray([float(trial.value) for trial in complete])
        ax.plot(numbers, values, "o-", label="trial validation WAPE")
        ax.plot(numbers, np.minimum.accumulate(values), "-", label="best so far")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No completed trials", ha="center", va="center")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Mean h1-h6 validation WAPE (%)")
    ax.set_title(f"Optuna optimization history\n{_experiment_title()}")
    fig.tight_layout()
    fig.savefig(figure_root / "optuna_optimization_history.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    try:
        if len(complete) < 2:
            raise ValueError("At least two completed trials are required")
        importances = optuna.importance.get_param_importances(study)
        if not importances:
            raise ValueError("No importances returned")
        names = list(importances)
        values = [importances[name] for name in names]
        ax.barh(names[::-1], values[::-1])
        ax.set_xlabel("Importance")
    except Exception as exc:
        ax.text(
            0.5,
            0.5,
            f"Parameter importance unavailable\n{type(exc).__name__}: {exc}",
            ha="center",
            va="center",
            wrap=True,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_title(f"Optuna parameter importance\n{_experiment_title()}")
    fig.tight_layout()
    fig.savefig(figure_root / "optuna_parameter_importance.png", dpi=180)
    plt.close(fig)


def _plot_evaluation(
    *,
    figure_root: Path,
    learning_curves: pd.DataFrame,
    test_metrics: pd.DataFrame,
    baseline_metrics: pd.DataFrame,
    truth: np.ndarray,
    primary_prediction: np.ndarray,
    test_windows: WindowSet,
) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_root.mkdir(parents=True, exist_ok=True)
    title = _experiment_title()
    fig, ax = plt.subplots(figsize=(9, 5))
    for seed, group in learning_curves.groupby("seed"):
        ax.plot(group["epoch"], group["validation_mean_horizon_wape_pct"], label=f"seed {seed}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation mean h1-h6 WAPE (%)")
    ax.set_title(f"Selected configuration learning curves\n{title}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_root / "learning_curves_selected_seeds.png", dpi=180)
    plt.close(fig)

    horizon_model = test_metrics.loc[test_metrics["horizon"] != "aggregate"].copy()
    horizon_model["horizon"] = horizon_model["horizon"].astype(int)
    model_summary = horizon_model.groupby("horizon", as_index=False)["mae_kw"].mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(model_summary["horizon"], model_summary["mae_kw"], "o-", label="LSTM seed mean")
    ax.set_xlabel("Horizon step")
    ax.set_ylabel("MAE (kW)")
    ax.set_title(f"Error versus forecast horizon\n{title}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_root / "error_vs_horizon.png", dpi=180)
    plt.close(fig)

    aggregate_model = float(
        test_metrics.loc[test_metrics["horizon"] == "aggregate", "wape_pct"].mean()
    )
    aggregate_baselines = baseline_metrics.loc[baseline_metrics["horizon"] == "aggregate"]
    labels = ["LSTM"] + aggregate_baselines["baseline"].astype(str).tolist()
    values = [aggregate_model] + aggregate_baselines["wape_pct"].astype(float).tolist()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(labels, values)
    ax.set_ylabel("Aggregate WAPE (%)")
    ax.set_title(f"LSTM versus declared baselines\n{title}")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(figure_root / "lstm_vs_baselines.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(truth.reshape(-1), primary_prediction.reshape(-1), s=5, alpha=0.25)
    bounds = [
        min(float(truth.min()), float(primary_prediction.min())),
        max(float(truth.max()), float(primary_prediction.max())),
    ]
    ax.plot(bounds, bounds, "k--", linewidth=1)
    ax.set_xlabel("Actual load (kW)")
    ax.set_ylabel("Predicted load (kW)")
    ax.set_title(f"Prediction versus actual\n{title}")
    fig.tight_layout()
    fig.savefig(figure_root / "prediction_vs_actual_scatter.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    errors = primary_prediction - truth
    ax.boxplot([errors[:, index] for index in range(errors.shape[1])], showfliers=False)
    ax.set_xlabel("Horizon step")
    ax.set_ylabel("Residual prediction - actual (kW)")
    ax.set_title(f"Residual distribution by horizon\n{title}")
    fig.tight_layout()
    fig.savefig(figure_root / "residual_distribution_by_horizon.png", dpi=180)
    plt.close(fig)

    sequence_root = figure_root / "test_sequences"
    sequence_root.mkdir(parents=True, exist_ok=True)
    sequence_files: dict[str, str] = {}
    for index, sequence_id in enumerate(dict.fromkeys(test_windows.sequence_ids.tolist())):
        mask = test_windows.sequence_ids == sequence_id
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(sequence_id)).strip("_")[:50]
        suffix = hashlib.sha256(str(sequence_id).encode("utf-8")).hexdigest()[:8]
        filename = f"{index:02d}_{safe or 'sequence'}_{suffix}_prediction_h1_h6.png"
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        positions = np.arange(int(np.count_nonzero(mask)))
        for axis, horizon in zip(axes, (0, prediction_steps_index(primary_prediction))):
            axis.plot(positions, truth[mask, horizon], label="actual", linewidth=1)
            axis.plot(positions, primary_prediction[mask, horizon], label="prediction", linewidth=1)
            axis.set_ylabel(f"h{horizon + 1} load (kW)")
            axis.legend()
        axes[-1].set_xlabel("Window index within atomic sequence")
        fig.suptitle(f"Test atomic sequence {index:02d} ({suffix})\n{title}")
        fig.tight_layout()
        fig.savefig(sequence_root / filename, dpi=170)
        plt.close(fig)
        sequence_files[str(sequence_id)] = (Path("figures") / "test_sequences" / filename).as_posix()
    return sequence_files


def prediction_steps_index(prediction: np.ndarray) -> int:
    return int(prediction.shape[1] - 1)


def _write_artifact_manifest(run_root: Path) -> None:
    files = []
    for path in sorted(run_root.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            files.append(
                {
                    "path": path.relative_to(run_root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    write_json(run_root / "artifact_manifest.json", {"files": files})


def run_experiment(args: argparse.Namespace) -> Path:
    dataset_root = args.dataset_root.resolve()
    split_path = args.split_path.resolve()
    combined_csv = dataset_root / "millisecond_load_10ms.csv"
    dataset_manifest_path = dataset_root / "dataset_manifest.json"
    for required in (combined_csv, dataset_manifest_path, split_path):
        if not required.exists():
            raise FileNotFoundError(required)
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    split_manifest = json.loads(split_path.read_text(encoding="utf-8"))
    if int(split_manifest["history_steps"]) != args.history_steps:
        raise ValueError("History steps do not match the retained split manifest")
    if int(split_manifest["prediction_steps"]) != args.prediction_steps:
        raise ValueError("Prediction steps do not match the retained split manifest")
    dataset_hash = sha256_file(dataset_manifest_path)
    split_hash = sha256_file(split_path)
    source_manifest_path = ROOT / str(dataset_manifest["source_manifest"])
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_hashes = [str(item["sha256"]) for item in source_manifest["sources"]]
    device = resolve_device(args.device)

    if args.resume_run:
        candidate = Path(args.resume_run)
        run_root = candidate if candidate.is_absolute() else args.output_root.resolve() / candidate
        if not run_root.exists():
            raise FileNotFoundError(f"Resume run does not exist: {run_root}")
    else:
        run_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
        run_root = args.output_root.resolve() / run_id
        run_root.mkdir(parents=True, exist_ok=False)
    for child in ("study", "selection", "checkpoints", "metrics", "predictions", "figures"):
        (run_root / child).mkdir(parents=True, exist_ok=True)
    run_started_utc = datetime.now(timezone.utc).isoformat()
    run_config = vars(args).copy()
    run_config.update(
        {
            "dataset_manifest_sha256": dataset_hash,
            "split_manifest_sha256": split_hash,
            "source_sha256": source_hashes,
            "resolved_device": str(device),
            "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
            "search_space_version": SEARCH_SPACE_VERSION,
            "run_started_utc": run_started_utc,
            "forecasting_only": True,
        }
    )
    write_json(run_root / "run_config.json", run_config)

    train_sequences = load_sequences(combined_csv, "train")
    validation_sequences = load_sequences(combined_csv, "validation")
    scaler = fit_standard_scaler(train_sequences)
    train_windows = build_windows(
        train_sequences, history_steps=args.history_steps, prediction_steps=args.prediction_steps
    )
    validation_windows = build_windows(
        validation_sequences, history_steps=args.history_steps, prediction_steps=args.prediction_steps
    )
    trial_checkpoint_root = run_root / "study" / "trial_checkpoints"
    trial_checkpoint_root.mkdir(exist_ok=True)

    def objective(trial: optuna.Trial) -> float:
        config = sample_trial_config(
            trial,
            fixed_seed=int(args.seeds[0]),
            max_epochs=args.max_epochs,
            patience=args.patience,
        )
        artifact = train_model(
            config=config,
            train_windows_raw=train_windows,
            validation_windows_raw=validation_windows,
            scaler=scaler,
            prediction_steps=args.prediction_steps,
            device=device,
            trial_timeout_s=args.trial_timeout_s,
        )
        trial.set_user_attr("trial_config", dataclasses.asdict(config))
        trial.set_user_attr("best_epoch", artifact.best_epoch)
        trial.set_user_attr("validation_mean_horizon_wape_pct", artifact.best_score)
        trial.set_user_attr("validation_aggregate_mae_kw", artifact.best_mae)
        trial.set_user_attr("elapsed_s", artifact.elapsed_s)
        trial.set_user_attr("stopped_by_timeout", artifact.stopped_by_timeout)
        checkpoint_path = trial_checkpoint_root / f"trial_{trial.number:03d}.pt"
        torch.save(
            _checkpoint_payload(
                artifact=artifact,
                scaler=scaler,
                dataset_hash=dataset_hash,
                split_hash=split_hash,
                source_hashes=source_hashes,
            ),
            checkpoint_path,
        )
        trial.set_user_attr("checkpoint", checkpoint_path.relative_to(run_root).as_posix())
        pd.DataFrame(artifact.learning_curve).to_csv(
            trial_checkpoint_root / f"trial_{trial.number:03d}_curve.csv", index=False
        )
        return artifact.best_score

    study_started = time.monotonic()
    study = run_study(
        objective=objective,
        storage_path=run_root / "study" / "study.sqlite3",
        study_name="millisecond_10ms_lstm_30_to_6",
        n_trials=args.n_trials,
        timeout_s=args.study_timeout_s,
        sampler_seed=20260710,
        trial_csv=run_root / "study" / "trials_partial.csv",
        study_user_attrs={
            "dataset_manifest_sha256": dataset_hash,
            "split_manifest_sha256": split_hash,
            "history_steps": args.history_steps,
            "prediction_steps": args.prediction_steps,
            "search_space_version": SEARCH_SPACE_VERSION,
        },
    )
    study_elapsed_s = time.monotonic() - study_started
    _plot_optuna(study, run_root / "figures")
    complete_trials = sorted(
        (
            trial
            for trial in study.trials
            if trial.state == optuna.trial.TrialState.COMPLETE
            and trial.value is not None
            and "trial_config" in trial.user_attrs
        ),
        key=lambda trial: (float(trial.value), trial.number),
    )
    unique_configs: list[TrialConfig] = []
    seen: set[str] = set()
    for trial in complete_trials:
        config = trial_config_from_mapping(trial.user_attrs["trial_config"])
        config_id = canonical_config_id(config)
        if config_id not in seen:
            seen.add(config_id)
            unique_configs.append(config)
        if len(unique_configs) >= args.robust_top_k:
            break
    close_study_storage(study)
    if not unique_configs:
        raise RuntimeError("No completed Optuna trial is available for robust selection")

    candidate_artifacts: dict[tuple[str, int], TrainingArtifact] = {}
    selection_rows: list[dict[str, object]] = []
    all_curve_rows: list[dict[str, object]] = []
    candidates: list[CandidateResult] = []
    for base_config in unique_configs:
        config_id = canonical_config_id(base_config)
        wapes: list[float] = []
        maes: list[float] = []
        for seed in args.seeds:
            config = dataclasses.replace(base_config, seed=int(seed))
            artifact = train_model(
                config=config,
                train_windows_raw=train_windows,
                validation_windows_raw=validation_windows,
                scaler=scaler,
                prediction_steps=args.prediction_steps,
                device=device,
                trial_timeout_s=args.trial_timeout_s,
            )
            candidate_artifacts[(config_id, int(seed))] = artifact
            wapes.append(artifact.best_score)
            maes.append(artifact.best_mae)
            selection_rows.append(
                {
                    "config_id": config_id,
                    "seed": int(seed),
                    "validation_mean_horizon_wape_pct": artifact.best_score,
                    "validation_aggregate_mae_kw": artifact.best_mae,
                    "best_epoch": artifact.best_epoch,
                    "elapsed_s": artifact.elapsed_s,
                    "stopped_by_timeout": artifact.stopped_by_timeout,
                }
            )
            for row in artifact.learning_curve:
                all_curve_rows.append({"config_id": config_id, "seed": int(seed), **row})
        candidates.append(CandidateResult(config_id, tuple(wapes), tuple(maes)))
    selected = select_configuration(candidates)
    ranking_rows = [
        {
            "config_id": candidate.config_id,
            "validation_wape_mean_pct": float(np.mean(candidate.validation_wape)),
            "validation_wape_std_pct": float(np.std(candidate.validation_wape, ddof=0)),
            "validation_mae_mean_kw": float(np.mean(candidate.validation_mae)),
            "validation_mae_std_kw": float(np.std(candidate.validation_mae, ddof=0)),
        }
        for candidate in candidates
    ]
    ranking_rows.sort(
        key=lambda row: (
            float(row["validation_wape_mean_pct"]),
            float(row["validation_mae_mean_kw"]),
            str(row["config_id"]),
        )
    )
    for rank, row in enumerate(ranking_rows, 1):
        row["rank"] = rank
        row["selected"] = row["config_id"] == selected.config_id
    pd.DataFrame(selection_rows).to_csv(
        run_root / "selection" / "top3_validation_by_seed.csv", index=False
    )
    pd.DataFrame(ranking_rows).to_csv(
        run_root / "selection" / "configuration_ranking.csv", index=False
    )
    selected_base = next(
        config for config in unique_configs if canonical_config_id(config) == selected.config_id
    )
    checkpoint_paths: dict[int, Path] = {}
    for seed in args.seeds:
        artifact = candidate_artifacts[(selected.config_id, int(seed))]
        path = run_root / "checkpoints" / f"selected_seed_{int(seed)}.pt"
        torch.save(
            _checkpoint_payload(
                artifact=artifact,
                scaler=scaler,
                dataset_hash=dataset_hash,
                split_hash=split_hash,
                source_hashes=source_hashes,
            ),
            path,
        )
        checkpoint_paths[int(seed)] = path
    selection_completed_utc = datetime.now(timezone.utc).isoformat()
    write_json(
        run_root / "selection" / "selection_complete.json",
        {
            "completed_utc": selection_completed_utc,
            "selected_config_id": selected.config_id,
            "selected_hyperparameters": dataclasses.asdict(selected_base),
            "selection_uses_test_metrics": False,
            "primary_checkpoint_seed": 42 if 42 in args.seeds else int(args.seeds[0]),
        },
    )

    # Test windows are intentionally constructed only after selection_complete.json exists.
    test_sequences = load_sequences(combined_csv, "test")
    test_windows = build_windows(
        test_sequences, history_steps=args.history_steps, prediction_steps=args.prediction_steps
    )
    test_metric_frames: list[pd.DataFrame] = []
    predictions: dict[int, np.ndarray] = {}
    for seed in args.seeds:
        artifact = candidate_artifacts[(selected.config_id, int(seed))]
        prediction = predict_windows(
            config=artifact.config,
            model_state_dict=artifact.model_state_dict,
            windows_raw=test_windows,
            scaler=scaler,
            prediction_steps=args.prediction_steps,
            device=device,
        )
        predictions[int(seed)] = prediction
        metrics = metrics_by_horizon(test_windows.y, prediction)
        metrics.insert(0, "seed", int(seed))
        test_metric_frames.append(metrics)
    test_metrics = pd.concat(test_metric_frames, ignore_index=True)
    test_metrics.to_csv(run_root / "metrics" / "test_metrics_by_seed_horizon.csv", index=False)
    metric_columns = [
        "mae_kw",
        "rmse_kw",
        "wape_pct",
        "bias_kw",
        "r2",
        "samples",
        "negative_prediction_count",
    ]
    seed_summary = test_metrics.groupby("horizon", as_index=False)[metric_columns].agg(["mean", "std"])
    seed_summary.columns = [
        "horizon" if left == "horizon" else f"{left}_{right}" for left, right in seed_summary.columns
    ]
    seed_summary.to_csv(run_root / "metrics" / "test_metrics_seed_mean_std.csv", index=False)

    baseline_frames: list[pd.DataFrame] = []
    for name, prediction in baseline_forecasts(
        test_windows.x, prediction_steps=args.prediction_steps
    ).items():
        metrics = metrics_by_horizon(test_windows.y, prediction)
        metrics.insert(0, "baseline", name)
        baseline_frames.append(metrics)
    baseline_metrics = pd.concat(baseline_frames, ignore_index=True)
    baseline_metrics.to_csv(
        run_root / "metrics" / "test_baseline_metrics_by_horizon.csv", index=False
    )
    primary_seed = 42 if 42 in predictions else int(args.seeds[0])
    primary_prediction = predictions[primary_seed]
    sequence_metric_frames: list[pd.DataFrame] = []
    for sequence_id in dict.fromkeys(test_windows.sequence_ids.tolist()):
        mask = test_windows.sequence_ids == sequence_id
        frame = metrics_by_horizon(test_windows.y[mask], primary_prediction[mask])
        frame.insert(0, "sequence_id", sequence_id)
        frame.insert(1, "seed", primary_seed)
        sequence_metric_frames.append(frame)
    pd.concat(sequence_metric_frames, ignore_index=True).to_csv(
        run_root / "metrics" / "test_metrics_by_sequence.csv", index=False
    )
    gap_rows: list[dict[str, object]] = []
    for seed in args.seeds:
        artifact = candidate_artifacts[(selected.config_id, int(seed))]
        for split_name, windows in (("train", train_windows), ("validation", validation_windows), ("test", test_windows)):
            prediction = (
                predictions[int(seed)]
                if split_name == "test"
                else predict_windows(
                    config=artifact.config,
                    model_state_dict=artifact.model_state_dict,
                    windows_raw=windows,
                    scaler=scaler,
                    prediction_steps=args.prediction_steps,
                    device=device,
                )
            )
            aggregate = metrics_by_horizon(windows.y, prediction).iloc[-1]
            gap_rows.append(
                {
                    "seed": int(seed),
                    "split": split_name,
                    "mae_kw": aggregate["mae_kw"],
                    "wape_pct": aggregate["wape_pct"],
                    "samples": aggregate["samples"],
                }
            )
    pd.DataFrame(gap_rows).to_csv(
        run_root / "metrics" / "train_validation_test_gap.csv", index=False
    )
    prediction_frame = pd.DataFrame(
        {
            "sequence_id": test_windows.sequence_ids,
            "target_start_index": test_windows.target_start_indices,
        }
    )
    for horizon in range(args.prediction_steps):
        prediction_frame[f"actual_h{horizon + 1}_kw"] = test_windows.y[:, horizon]
        prediction_frame[f"prediction_h{horizon + 1}_kw"] = primary_prediction[:, horizon]
    prediction_frame.to_parquet(
        run_root / "predictions" / f"test_predictions_seed_{primary_seed}.parquet", index=False
    )

    learning_curves = pd.DataFrame(
        row for row in all_curve_rows if row["config_id"] == selected.config_id
    )
    sequence_figures = _plot_evaluation(
        figure_root=run_root / "figures",
        learning_curves=learning_curves,
        test_metrics=test_metrics,
        baseline_metrics=baseline_metrics,
        truth=test_windows.y,
        primary_prediction=primary_prediction,
        test_windows=test_windows,
    )
    aggregate_lstm = test_metrics.loc[test_metrics["horizon"] == "aggregate"]
    lstm_mae = float(aggregate_lstm["mae_kw"].mean())
    lstm_wape = float(aggregate_lstm["wape_pct"].mean())
    aggregate_baseline = baseline_metrics.loc[baseline_metrics["horizon"] == "aggregate"]
    beats_all = bool(
        (lstm_mae < aggregate_baseline["mae_kw"]).all()
        and (lstm_wape < aggregate_baseline["wape_pct"]).all()
    )
    any_no_worse_both = bool(
        ((aggregate_baseline["mae_kw"] <= lstm_mae) & (aggregate_baseline["wape_pct"] <= lstm_wape)).any()
    )
    comparison = (
        "LSTM_BEATS_ALL_BASELINES"
        if beats_all
        else "BASELINE_MATCHES_OR_BEATS_LSTM"
        if any_no_worse_both
        else "LSTM_MIXED_RESULT"
    )
    run_completed_utc = datetime.now(timezone.utc).isoformat()
    summary = {
        "run_id": run_root.name,
        "run_started_utc": run_started_utc,
        "selection_completed_utc": selection_completed_utc,
        "run_completed_utc": run_completed_utc,
        "forecasting_only": True,
        "dataset_manifest_sha256": dataset_hash,
        "split_manifest_sha256": split_hash,
        "source_sha256": source_hashes,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "study_elapsed_s": study_elapsed_s,
        "study_trial_limit": args.n_trials,
        "study_timeout_s": args.study_timeout_s,
        "completed_trials": len(complete_trials),
        "selected_config_id": selected.config_id,
        "selected_hyperparameters": dataclasses.asdict(selected_base),
        "primary_seed": primary_seed,
        "test_seed_mean_aggregate_mae_kw": lstm_mae,
        "test_seed_mean_aggregate_wape_pct": lstm_wape,
        "baseline_decision": comparison,
        "negative_predictions_total": int(aggregate_lstm["negative_prediction_count"].sum()),
        "test_sequence_figures": sequence_figures,
        "sensor_provenance_verified": False,
        "direct_decimation_aliasing_caveat": True,
    }
    write_json(run_root / "run_summary.json", summary)
    report = f"""# Millisecond 10 ms LSTM Forecast Report

## Scope

This is a forecasting-only experiment. It is not approved for online energy-management use.

- {_experiment_title()}
- Direct row decimation: every 10th 1 ms row; no averaging or interpolation.
- Dataset rows: {dataset_manifest['unique_rows_10ms']} unique 10 ms rows across {dataset_manifest['atomic_sequence_count']} atomic sequences.
- Split rows: {json.dumps(split_manifest['split_rows'], ensure_ascii=False)}.
- Source SHA-256: {', '.join(source_hashes)}.
- Sensor provenance in the supplied workbooks is unverified.
- Direct decimation does not provide anti-alias filtering; sub-100 Hz content may alias.

## Search and selection

- Device: {summary['device_name']}.
- Optuna limits: at most {args.n_trials} trials, {args.study_timeout_s} s study time, {args.trial_timeout_s} s per trial.
- Completed trials: {len(complete_trials)}; measured study call duration: {study_elapsed_s:.3f} s.
- Selection used validation metrics only and completed at `{selection_completed_utc}` before test windows were constructed.
- Selected hyperparameters: `{json.dumps(dataclasses.asdict(selected_base), sort_keys=True)}`.
- Seed {primary_seed} is the designated primary checkpoint; no best-test-seed selection was performed.

## Held-out test result

- Three-seed aggregate MAE mean: {lstm_mae:.6f} kW.
- Three-seed aggregate WAPE mean: {lstm_wape:.6f}%.
- Raw negative prediction count summed across seed aggregate rows: {summary['negative_predictions_total']}.
- Predeclared baseline decision: **{comparison}**.

Detailed per-horizon, per-seed, per-sequence, baseline, and train/validation/test-gap tables are under `metrics/`. Raw primary-seed predictions are under `predictions/`.
"""
    (run_root / "REPORT_MILLISECOND_10MS_LSTM.md").write_text(report, encoding="utf-8")
    _write_artifact_manifest(run_root)
    print(f"run_root={run_root}")
    print(f"completed_trials={len(complete_trials)}")
    print(f"selected_config_id={selected.config_id}")
    print(f"test_seed_mean_aggregate_mae_kw={lstm_mae:.9f}")
    print(f"test_seed_mean_aggregate_wape_pct={lstm_wape:.9f}")
    print(f"baseline_decision={comparison}")
    return run_root


def main() -> None:
    args = build_parser().parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
