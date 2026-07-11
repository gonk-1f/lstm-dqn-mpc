"""Retrain the current 7-2-1 LSTM load forecaster without overwriting baseline checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

SRC = Path(__file__).resolve().parents[1]
PROJ = Path(__file__).resolve().parents[2]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from forecasting.feature_pipeline import prepare_lstm_features  # noqa: E402
from forecasting.lstm_load_predictor import (  # noqa: E402
    LSTMForecastConfig,
    MultiStepLoadLSTM,
    fit_scaler,
    inverse_target,
    metrics_from_predictions,
    save_checkpoint,
    set_seed,
    transform,
)


DEFAULT_SOURCE_CSV = PROJ / "data/processed/aligned_timeseries.csv"
DEFAULT_SPLIT_JSON = PROJ / "outputs/config/voyage_split_721.json"
DEFAULT_REFERENCE_META = (
    PROJ
    / "outputs/lstm_721/candidate_asym_weighted_huber_delta10/checkpoints/"
    / "candidate_asym_weighted_huber_delta10/best_lstm_load_predictor.json"
)
DEFAULT_OUTPUT_DIR = PROJ / "outputs/lstm_721_retrain"
DEFAULT_CANDIDATE = "candidate_asym_weighted_huber_delta10_retrain"
CURRENT_BASELINE_CKPT = DEFAULT_REFERENCE_META.with_suffix(".pt")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def config_from_reference_meta(reference_meta: Path) -> tuple[LSTMForecastConfig, str, dict[str, Any]]:
    meta = load_json(Path(reference_meta))
    feature_meta_path = Path(reference_meta).with_suffix(".feature_set.json")
    feature_meta = load_json(feature_meta_path) if feature_meta_path.exists() else {}
    cfg = LSTMForecastConfig(**dict(meta["config"]))
    loss_meta: dict[str, Any] = {}
    for key in [
        "loss",
        "huber_delta_kw",
        "huber_delta_norm",
        "asym_under_weight",
        "asym_high_load_bonus",
        "asym_ramp_bonus",
        "asym_high_load_threshold_kw",
        "asym_ramp_threshold_kw",
        "horizon_weight",
        "features",
    ]:
        if key in feature_meta:
            loss_meta[key] = feature_meta[key]
        elif key in meta:
            loss_meta[key] = meta[key]
    loss_meta.setdefault("loss", meta.get("loss", "asym_weighted_huber"))
    loss_meta.setdefault("huber_delta_kw", float(meta.get("huber_delta_kw", 10.0)))
    loss_meta.setdefault("asym_under_weight", float(meta.get("asym_under_weight", 3.0)))
    loss_meta.setdefault("asym_high_load_bonus", float(feature_meta.get("asym_high_load_bonus", 0.5)))
    loss_meta.setdefault("asym_ramp_bonus", float(feature_meta.get("asym_ramp_bonus", 0.2)))
    loss_meta.setdefault("asym_high_load_threshold_kw", float(feature_meta.get("asym_high_load_threshold_kw", 66.5)))
    loss_meta.setdefault("asym_ramp_threshold_kw", float(feature_meta.get("asym_ramp_threshold_kw", 3.4)))
    loss_meta.setdefault("horizon_weight", [1.0] * int(cfg.pred_horizon))
    loss_meta.setdefault("features", list(meta["features"]))
    return cfg, str(meta.get("feature_set", feature_meta.get("feature_set", "rolling"))), loss_meta


def asym_weighted_huber_loss(
    *,
    y_pred_norm: torch.Tensor,
    y_true_norm: torch.Tensor,
    y_true_kw: torch.Tensor,
    target_std_kw: float,
    huber_delta_kw: float,
    asym_under_weight: float,
    high_load_threshold_kw: float,
    ramp_threshold_kw: float,
    high_load_bonus: float = 0.5,
    ramp_bonus: float = 0.2,
    horizon_weight: list[float] | None = None,
) -> torch.Tensor:
    delta_norm = float(huber_delta_kw) / max(float(target_std_kw), 1e-6)
    err = y_pred_norm - y_true_norm
    abs_err = torch.abs(err)
    base = torch.where(abs_err <= delta_norm, 0.5 * err**2, delta_norm * (abs_err - 0.5 * delta_norm))
    under_w = torch.where(y_pred_norm < y_true_norm, float(asym_under_weight), 1.0)
    sample_w = torch.ones_like(base)
    sample_w = sample_w + float(high_load_bonus) * (y_true_kw > float(high_load_threshold_kw)).to(base.dtype)
    ramp = torch.zeros_like(y_true_kw)
    if y_true_kw.shape[1] > 1:
        ramp[:, 1:] = y_true_kw[:, 1:] - y_true_kw[:, :-1]
    sample_w = sample_w + float(ramp_bonus) * (ramp > float(ramp_threshold_kw)).to(base.dtype)
    if horizon_weight is not None:
        h_w = torch.as_tensor(horizon_weight, dtype=base.dtype, device=base.device).reshape(1, -1)
        sample_w = sample_w * h_w
    return torch.mean(base * under_w * sample_w)


def _parse_horizon_weight(value: str | list[float] | tuple[float, ...] | None, horizon: int) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, str):
        weights = [float(part.strip()) for part in value.split(",") if part.strip()]
    else:
        weights = [float(part) for part in value]
    if len(weights) != int(horizon):
        raise ValueError(f"horizon_weight length {len(weights)} does not match pred_horizon {horizon}.")
    return weights


def apply_training_overrides(
    config: LSTMForecastConfig,
    loss_meta: dict[str, Any],
    args: argparse.Namespace,
) -> str:
    for field in [
        "history_len",
        "pred_horizon",
        "hidden_size",
        "num_layers",
        "dropout",
        "batch_size",
        "lr",
        "epochs",
        "patience",
        "seed",
        "grad_clip",
    ]:
        value = getattr(args, field, None)
        if value is not None:
            current = getattr(config, field)
            cast = int if isinstance(current, int) and not isinstance(current, bool) else float
            setattr(config, field, cast(value))

    for field in [
        "huber_delta_kw",
        "asym_under_weight",
        "asym_high_load_bonus",
        "asym_ramp_bonus",
    ]:
        value = getattr(args, field, None)
        if value is not None:
            loss_meta[field] = float(value)

    horizon_weight = _parse_horizon_weight(getattr(args, "horizon_weight", None), int(config.pred_horizon))
    if horizon_weight is not None:
        loss_meta["horizon_weight"] = horizon_weight

    return str(getattr(args, "selection_metric", None) or "validation_MAE")


def compute_train_loss_thresholds(train_df: pd.DataFrame, quantile: float = 0.75) -> tuple[float, float]:
    if "load_total_kw" not in train_df.columns:
        raise ValueError("Cannot compute loss thresholds without load_total_kw.")
    load = train_df["load_total_kw"].astype(float).ffill().bfill().fillna(0.0).to_numpy()
    high_load_threshold = float(np.nanquantile(load, float(quantile)))

    group_key = "voyage_name" if "voyage_name" in train_df.columns else "voyage_id" if "voyage_id" in train_df.columns else None
    ramp_values: list[np.ndarray] = []
    groups = train_df.groupby(group_key, sort=False).groups.values() if group_key else [train_df.index]
    for idx_values in groups:
        series = train_df.loc[list(idx_values), "load_total_kw"].astype(float).ffill().bfill().fillna(0.0).to_numpy()
        if len(series) > 1:
            ramp_values.append(np.abs(np.diff(series)))
    if ramp_values:
        ramp = np.concatenate(ramp_values)
        ramp_threshold = float(np.nanquantile(ramp, float(quantile)))
    else:
        ramp_threshold = 0.0
    return high_load_threshold, ramp_threshold


def detailed_horizon_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    out: dict[str, float | int] = {}
    horizon = min(y_true.shape[1], y_pred.shape[1])
    for h_idx in range(horizon):
        label = f"h{h_idx + 1}"
        truth = y_true[:, h_idx]
        pred = y_pred[:, h_idx]
        mask = np.isfinite(truth) & np.isfinite(pred)
        out[f"count_{label}"] = int(np.sum(mask))
        if not np.any(mask):
            for key in ["RMSE", "MAE", "WAPE", "Bias", "under_prediction_rate", "over_prediction_rate"]:
                out[f"{key}_{label}"] = float("nan")
            continue
        err = pred[mask] - truth[mask]
        out[f"RMSE_{label}"] = float(np.sqrt(np.mean(err**2)))
        out[f"MAE_{label}"] = float(np.mean(np.abs(err)))
        out[f"WAPE_{label}"] = float(np.sum(np.abs(err)) / (np.sum(np.abs(truth[mask])) + 1e-6) * 100.0)
        out[f"Bias_{label}"] = float(np.mean(err))
        out[f"under_prediction_rate_{label}"] = float(np.mean(err < -1e-6))
        out[f"over_prediction_rate_{label}"] = float(np.mean(err > 1e-6))
    return out


def weighted_mae_h1_h3(metrics: dict[str, Any]) -> float:
    return float(0.5 * metrics["MAE_h1"] + 0.3 * metrics["MAE_h2"] + 0.2 * metrics["MAE_h3"])


def _prepare_fixed_feature_frame(df: pd.DataFrame, feature_set: str, features: list[str]) -> pd.DataFrame:
    prepared = prepare_lstm_features(df, feature_set)
    for col in features:
        if col not in prepared.columns:
            prepared[col] = 0.0
    return prepared


def _voyage_group_indices(prepared: pd.DataFrame) -> list[np.ndarray]:
    group_key = "voyage_name" if "voyage_name" in prepared.columns else "voyage_id" if "voyage_id" in prepared.columns else None
    if group_key is None:
        return [np.arange(len(prepared))]
    return [np.asarray(list(idx_values), dtype=int) for idx_values in prepared.groupby(group_key, sort=False).indices.values()]


def _windows_from_frame(
    prepared: pd.DataFrame,
    *,
    features: list[str],
    config: LSTMForecastConfig,
    feature_scaler: dict[str, list[float]],
    target_scaler: dict[str, float],
    max_windows: int | None = None,
    window_stride: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = prepared[features].astype(float).ffill().bfill().fillna(0.0).to_numpy()
    values = transform(values, feature_scaler).astype(np.float32, copy=False)
    target_kw = prepared["load_total_kw"].astype(float).ffill().bfill().fillna(0.0).to_numpy()
    target_norm = ((target_kw - float(target_scaler["mean"])) / float(target_scaler["std"])).astype(np.float32, copy=False)
    target_kw = target_kw.astype(np.float32, copy=False)
    stride = max(1, int(window_stride))

    xs: list[np.ndarray] = []
    ys_norm: list[np.ndarray] = []
    ys_kw: list[np.ndarray] = []
    for idx in _voyage_group_indices(prepared):
        if len(idx) < int(config.history_len) + int(config.pred_horizon):
            continue
        for pos in range(int(config.history_len), len(idx) - int(config.pred_horizon) + 1, stride):
            hist_idx = idx[pos - int(config.history_len) : pos]
            fut_idx = idx[pos : pos + int(config.pred_horizon)]
            xs.append(values[hist_idx])
            ys_norm.append(target_norm[fut_idx])
            ys_kw.append(target_kw[fut_idx])
            if max_windows is not None and len(xs) >= int(max_windows):
                return np.stack(xs), np.stack(ys_norm), np.stack(ys_kw)
    if not xs:
        raise ValueError("No LSTM windows generated for the requested split.")
    return np.stack(xs), np.stack(ys_norm), np.stack(ys_kw)


def make_training_dataset(x: np.ndarray, y_norm: np.ndarray, y_kw: np.ndarray) -> TensorDataset:
    return TensorDataset(
        torch.as_tensor(x, dtype=torch.float32),
        torch.as_tensor(y_norm, dtype=torch.float32),
        torch.as_tensor(y_kw, dtype=torch.float32),
    )


def best_checkpoint_score(row: dict[str, Any], selection_metric: str = "validation_MAE") -> float:
    """Select checkpoints by physical validation metrics in kW, not by shaped training loss."""
    if selection_metric == "validation_MAE":
        return float(row["val_MAE"])
    if selection_metric == "validation_weighted_MAE_h1_h3":
        return float(row["val_weighted_MAE_h1_h3"])
    key = selection_metric
    if key.startswith("validation_"):
        key = "val_" + key[len("validation_") :]
    return float(row[key])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Retrain the current 7-2-1 LSTM load forecaster.")
    parser.add_argument("--source_csv", default=str(DEFAULT_SOURCE_CSV))
    parser.add_argument("--split_json", default=str(DEFAULT_SPLIT_JSON))
    parser.add_argument("--reference_meta", default=str(DEFAULT_REFERENCE_META))
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--candidate", default=DEFAULT_CANDIDATE)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--max_train_windows", type=int, default=None)
    parser.add_argument("--max_val_windows", type=int, default=None)
    parser.add_argument("--train_window_stride", type=int, default=1)
    parser.add_argument("--val_window_stride", type=int, default=1)
    parser.add_argument("--feature_set", default=None)
    parser.add_argument("--overwrite_current", action="store_true")
    parser.add_argument("--history_len", type=int, default=None)
    parser.add_argument("--pred_horizon", type=int, default=None)
    parser.add_argument("--hidden_size", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", "--learning_rate", dest="lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--grad_clip", type=float, default=None)
    parser.add_argument("--huber_delta_kw", type=float, default=None)
    parser.add_argument("--asym_under_weight", type=float, default=None)
    parser.add_argument("--asym_high_load_bonus", type=float, default=None)
    parser.add_argument("--asym_ramp_bonus", type=float, default=None)
    parser.add_argument("--horizon_weight", default=None)
    parser.add_argument("--selection_metric", default="validation_MAE")
    parser.add_argument("--auto_loss_thresholds", action="store_true")
    parser.add_argument("--threshold_quantile", type=float, default=0.75)
    return parser


def train_lstm_721(args: argparse.Namespace) -> dict[str, Any]:
    reference_meta = Path(args.reference_meta)
    config, feature_set, loss_meta = config_from_reference_meta(reference_meta)
    if getattr(args, "feature_set", None):
        feature_set = str(args.feature_set)
    if args.max_epochs is not None:
        config.epochs = int(args.max_epochs)
    selection_metric = apply_training_overrides(config, loss_meta, args)
    features = list(loss_meta["features"])
    forced_features = getattr(args, "feature_list", None)
    if forced_features is not None:
        features = [str(col) for col in forced_features]
        loss_meta["features"] = features
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    ckpt_path = output_dir / "checkpoints" / str(args.candidate) / "best_lstm_load_predictor.pt"
    if ckpt_path.resolve() == CURRENT_BASELINE_CKPT.resolve() and not bool(args.overwrite_current):
        raise ValueError("Refusing to overwrite the current baseline LSTM checkpoint without --overwrite_current.")

    split = load_json(Path(args.split_json))
    df_all = pd.read_csv(args.source_csv)
    train_df = df_all[df_all["voyage_name"].isin(split["train_voyages"])].reset_index(drop=True)
    val_df = df_all[df_all["voyage_name"].isin(split["validation_voyages"])].reset_index(drop=True)
    if train_df.empty or val_df.empty:
        raise ValueError("Train/validation split produced an empty dataframe.")
    if bool(getattr(args, "auto_loss_thresholds", False)):
        high_load_threshold, ramp_threshold = compute_train_loss_thresholds(
            train_df,
            quantile=float(getattr(args, "threshold_quantile", 0.75)),
        )
        loss_meta["asym_high_load_threshold_kw"] = high_load_threshold
        loss_meta["asym_ramp_threshold_kw"] = ramp_threshold

    set_seed(int(config.seed))
    train_prepared = _prepare_fixed_feature_frame(train_df, feature_set, features)
    val_prepared = _prepare_fixed_feature_frame(val_df, feature_set, features)
    train_values = train_prepared[features].astype(float).ffill().bfill().fillna(0.0).to_numpy()
    feature_scaler = fit_scaler(train_values)
    target = train_prepared["load_total_kw"].astype(float).ffill().bfill().fillna(0.0).to_numpy()
    target_std = float(np.nanstd(target))
    target_scaler = {"mean": float(np.nanmean(target)), "std": target_std if target_std >= 1e-6 else 1.0}
    x_train, y_train, y_train_kw = _windows_from_frame(
        train_prepared,
        features=features,
        config=config,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        max_windows=args.max_train_windows,
        window_stride=int(getattr(args, "train_window_stride", 1)),
    )
    x_val, y_val, y_val_kw = _windows_from_frame(
        val_prepared,
        features=features,
        config=config,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        max_windows=args.max_val_windows,
        window_stride=int(getattr(args, "val_window_stride", 1)),
    )

    train_dataset = make_training_dataset(x_train, y_train, y_train_kw)
    val_dataset = make_training_dataset(x_val, y_val, y_val_kw)
    generator = torch.Generator()
    generator.manual_seed(int(config.seed))
    loader = DataLoader(
        train_dataset,
        batch_size=int(config.batch_size),
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=max(int(config.batch_size), 1) * 4,
        shuffle=False,
    )
    model = MultiStepLoadLSTM(feature_dim=len(features), config=config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.lr))

    best_score = float("inf")
    best_metrics: dict[str, Any] = {}
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    rows: list[dict[str, Any]] = []
    horizon_weight = [float(x) for x in loss_meta.get("horizon_weight", [1.0] * int(config.pred_horizon))]
    for epoch in range(1, int(config.epochs) + 1):
        model.train()
        batch_losses: list[float] = []
        for x_batch, y_batch, y_kw_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            y_kw_batch = y_kw_batch.to(device)
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = asym_weighted_huber_loss(
                y_pred_norm=pred,
                y_true_norm=y_batch,
                y_true_kw=y_kw_batch,
                target_std_kw=float(target_scaler["std"]),
                huber_delta_kw=float(loss_meta["huber_delta_kw"]),
                asym_under_weight=float(loss_meta["asym_under_weight"]),
                high_load_threshold_kw=float(loss_meta["asym_high_load_threshold_kw"]),
                ramp_threshold_kw=float(loss_meta["asym_ramp_threshold_kw"]),
                high_load_bonus=float(loss_meta.get("asym_high_load_bonus", 0.5)),
                ramp_bonus=float(loss_meta.get("asym_ramp_bonus", 0.2)),
                horizon_weight=horizon_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.grad_clip))
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            val_losses: list[float] = []
            val_rows: list[int] = []
            pred_kw_chunks: list[np.ndarray] = []
            true_kw_chunks: list[np.ndarray] = []
            for x_batch, y_batch, y_kw_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                y_kw_batch = y_kw_batch.to(device)
                val_pred = model(x_batch)
                batch_val_loss = asym_weighted_huber_loss(
                    y_pred_norm=val_pred,
                    y_true_norm=y_batch,
                    y_true_kw=y_kw_batch,
                    target_std_kw=float(target_scaler["std"]),
                    huber_delta_kw=float(loss_meta["huber_delta_kw"]),
                    asym_under_weight=float(loss_meta["asym_under_weight"]),
                    high_load_threshold_kw=float(loss_meta["asym_high_load_threshold_kw"]),
                    ramp_threshold_kw=float(loss_meta["asym_ramp_threshold_kw"]),
                    high_load_bonus=float(loss_meta.get("asym_high_load_bonus", 0.5)),
                    ramp_bonus=float(loss_meta.get("asym_ramp_bonus", 0.2)),
                    horizon_weight=horizon_weight,
                )
                batch_rows = int(x_batch.shape[0])
                val_losses.append(float(batch_val_loss.detach().cpu()) * batch_rows)
                val_rows.append(batch_rows)
                pred_kw_chunks.append(inverse_target(val_pred.detach().cpu().numpy(), target_scaler))
                true_kw_chunks.append(y_kw_batch.detach().cpu().numpy())
            val_loss = float(np.sum(val_losses) / max(int(np.sum(val_rows)), 1))
            pred_kw = np.vstack(pred_kw_chunks)
            true_kw = np.vstack(true_kw_chunks)
            metrics = metrics_from_predictions(true_kw, pred_kw)
            detailed_metrics = detailed_horizon_metrics(true_kw, pred_kw)
            metrics.update(detailed_metrics)
            metrics["weighted_MAE_h1_h3"] = weighted_mae_h1_h3(metrics)
        row = {"epoch": epoch, "train_loss": float(np.mean(batch_losses)), "val_loss": float(val_loss)}
        row.update({f"val_{key}": value for key, value in metrics.items()})
        rows.append(row)
        score = best_checkpoint_score(row, selection_metric)
        if score < best_score - 1e-9:
            best_score = score
            best_metrics = dict(row)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= int(config.patience):
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a best checkpoint.")
    model.load_state_dict(best_state)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_metrics = dict(best_metrics)
    checkpoint_metrics.update(
        {
            "candidate": str(args.candidate),
            "feature_set": feature_set,
            "loss": str(loss_meta["loss"]),
            "huber_delta_kw": float(loss_meta["huber_delta_kw"]),
            "asym_under_weight": float(loss_meta["asym_under_weight"]),
            "asym_high_load_threshold_kw": float(loss_meta["asym_high_load_threshold_kw"]),
            "asym_ramp_threshold_kw": float(loss_meta["asym_ramp_threshold_kw"]),
            "epochs_ran": len(rows),
            "best_epoch": int(best_metrics["epoch"]),
            "selection_metric": selection_metric,
            "best_selection_score": float(best_score),
        }
    )
    save_checkpoint(ckpt_path, model, config, features, feature_scaler, target_scaler, checkpoint_metrics)
    write_json(
        ckpt_path.with_suffix(".feature_set.json"),
        {key: value for key, value in loss_meta.items() if key != "features"} | {"feature_set": feature_set, "features": features},
    )
    pd.DataFrame(rows).to_csv(ckpt_path.parent / "training_curve.csv", index=False, encoding="utf-8-sig")
    write_json(
        output_dir / "run_config.json",
        {
            "source_csv": str(Path(args.source_csv).resolve()),
            "split_json": str(Path(args.split_json).resolve()),
            "reference_meta": str(reference_meta.resolve()),
            "output_dir": str(output_dir.resolve()),
            "candidate": str(args.candidate),
            "checkpoint": str(ckpt_path.resolve()),
            "overwrite_current": bool(args.overwrite_current),
            "train_windows": int(x_train.shape[0]),
            "val_windows": int(x_val.shape[0]),
            "train_window_stride": int(getattr(args, "train_window_stride", 1)),
            "val_window_stride": int(getattr(args, "val_window_stride", 1)),
            "selection_metric": selection_metric,
            "best_selection_score": float(best_score),
            "config": config.__dict__,
            "loss_meta": {key: value for key, value in loss_meta.items() if key != "features"},
            "features": features,
        },
    )
    return {
        "checkpoint": str(ckpt_path),
        "epochs_ran": len(rows),
        "best_epoch": int(best_metrics["epoch"]),
        "best_val_loss": float(best_metrics["val_loss"]),
        "selection_metric": selection_metric,
        "best_selection_score": float(best_score),
        "val_MAE": float(best_metrics["val_MAE"]),
        "val_RMSE": float(best_metrics["val_RMSE"]),
    }


def main() -> None:
    args = build_parser().parse_args()
    summary = train_lstm_721(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
