"""Train/evaluate load-only LSTM on offline cubic-spline 1 s reconstructions.

These data are reconstructed 1 s sequences, not measured 1 s data. This script
continues the offline diagnostic even when physical audit warnings exist.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJ = Path(__file__).resolve().parents[2]
SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from forecasting.feature_pipeline import prepare_lstm_features  # noqa: E402
from forecasting.lstm_load_predictor import inverse_target, load_checkpoint, transform  # noqa: E402
from run_train_lstm_721 import (  # noqa: E402
    DEFAULT_REFERENCE_META,
    detailed_horizon_metrics,
    train_lstm_721,
)


DEFAULT_SPLINE_ROOT = PROJ / "outputs" / "spline_1s_diagnostics"
DEFAULT_BASE_SPLIT_JSON = PROJ / "outputs" / "config" / "voyage_split_total_load_721.json"
DEFAULT_DATASET_VERSION = "cubic_spline_1s_natural"
DEFAULT_SOURCE_CSV = DEFAULT_SPLINE_ROOT / "data" / f"{DEFAULT_DATASET_VERSION}.csv"
DEFAULT_OUTPUT_ROOT = DEFAULT_SPLINE_ROOT / "models" / "spline_1s_short_h180_p60"
HORIZONS_TO_REPORT = (1, 6, 30, 60)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def prepare_spline_lstm_source(
    *,
    source_csv: Path,
    base_split_json: Path,
    output_dir: Path,
    dataset_version: str,
) -> dict[str, str]:
    """Create a training-compatible source CSV and split JSON for one spline version."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    usecols = [
        "dataset_version",
        "voyage_id",
        "split",
        "timestamp",
        "time_s",
        "load_total_kw",
        "online_feasible",
        "uses_future_endpoint",
        "file_name",
    ]
    available = pd.read_csv(source_csv, nrows=0).columns.tolist()
    selected = [col for col in usecols if col in available]
    df = pd.read_csv(source_csv, usecols=selected)
    if "dataset_version" in df.columns:
        df = df[df["dataset_version"].eq(dataset_version)].copy()
    if df.empty:
        raise ValueError(f"No rows found for dataset_version={dataset_version} in {source_csv}")
    if "voyage_id" not in df.columns:
        raise ValueError("Spline source must include voyage_id")
    df["voyage_name"] = df["voyage_id"].astype(str)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "load_total_kw"]).sort_values(["voyage_name", "time_s"]).reset_index(drop=True)
    df["load_definition"] = "offline cubic-spline reconstruction of 30 s load_total_kw"
    df["load_scope"] = "offline_reconstructed_1s_total_load"
    prepared_source = output_dir / f"{dataset_version}_lstm_source.csv"
    df.to_csv(prepared_source, index=False, encoding="utf-8-sig")

    base_split = load_json(Path(base_split_json))
    split_payload = {
        "train_voyages": list(base_split.get("train_voyages", base_split.get("train", []))),
        "validation_voyages": list(base_split.get("validation_voyages", base_split.get("validation", []))),
        "test_voyages": list(base_split.get("test_voyages", base_split.get("test", []))),
        "train": list(base_split.get("train_voyages", base_split.get("train", []))),
        "validation": list(base_split.get("validation_voyages", base_split.get("validation", []))),
        "test": list(base_split.get("test_voyages", base_split.get("test", []))),
        "sample_interval_seconds": 1.0,
        "dataset_version": dataset_version,
        "source_csv": str(prepared_source.resolve()),
        "online_feasible": False,
        "uses_future_endpoint": True,
        "feature_policy": "load_total_kw only; no speed, time, rolling, ramp, or SOC features",
        "data_label": "offline cubic-spline 1 s reconstruction, not measured 1 s data",
    }
    split_path = output_dir / f"{dataset_version}_split_721.json"
    write_json(split_path, split_payload)
    return {"source_csv": str(prepared_source.resolve()), "split_json": str(split_path.resolve())}


def _forecast_voyage_load_only(
    df_voyage: pd.DataFrame,
    *,
    model: torch.nn.Module,
    payload: dict[str, Any],
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cfg = payload["config"]
    history_len = int(cfg["history_len"])
    pred_horizon = int(cfg["pred_horizon"])
    prepared = prepare_lstm_features(df_voyage, "base")
    features = ["load_total_kw"]
    values = prepared[features].astype(float).ffill().bfill().fillna(0.0).to_numpy()
    values = transform(values, payload["feature_scaler"])
    actual = prepared["load_total_kw"].astype(float).ffill().bfill().fillna(0.0).to_numpy()
    decision_indices = np.arange(history_len - 1, len(prepared) - 1, dtype=int)
    pred = np.full((len(decision_indices), pred_horizon), np.nan, dtype=float)
    true = np.full((len(decision_indices), pred_horizon), np.nan, dtype=float)
    model.eval()
    with torch.no_grad():
        for row_idx, decision_t in enumerate(decision_indices):
            hist = values[decision_t - history_len + 1 : decision_t + 1]
            if hist.shape[0] != history_len:
                continue
            x = torch.as_tensor(hist[None, :, :], dtype=torch.float32, device=device)
            raw = model(x).detach().cpu().numpy()[0]
            pred[row_idx] = inverse_target(raw, payload["target_scaler"])
            for horizon_idx in range(pred_horizon):
                true_idx = decision_t + horizon_idx + 1
                if true_idx < len(actual):
                    true[row_idx, horizon_idx] = actual[true_idx]
    return decision_indices, pred, true


def _baseline_arrays(actual: np.ndarray, decision_indices: np.ndarray, pred_horizon: int) -> dict[str, np.ndarray]:
    out = {
        "current_hold": np.full((len(decision_indices), pred_horizon), np.nan, dtype=float),
        "last_slope": np.full((len(decision_indices), pred_horizon), np.nan, dtype=float),
        "moving_average": np.full((len(decision_indices), pred_horizon), np.nan, dtype=float),
    }
    for row_idx, decision_t in enumerate(decision_indices):
        current = actual[decision_t]
        prev = actual[decision_t - 1] if decision_t >= 1 else current
        slope = current - prev
        ma_start = max(0, decision_t - 59)
        moving = float(np.mean(actual[ma_start : decision_t + 1]))
        for horizon_idx in range(pred_horizon):
            step = horizon_idx + 1
            out["current_hold"][row_idx, horizon_idx] = current
            out["last_slope"][row_idx, horizon_idx] = current + step * slope
            out["moving_average"][row_idx, horizon_idx] = moving
    return out


def _metrics_for_horizons(true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    metrics = detailed_horizon_metrics(true, pred)
    summary: dict[str, float] = {}
    for step in HORIZONS_TO_REPORT:
        for key in ("MAE", "RMSE", "WAPE", "Bias"):
            summary[f"{key}_h{step}"] = float(metrics.get(f"{key}_h{step}", np.nan))
    return summary


def evaluate_spline_checkpoint(
    *,
    checkpoint: Path,
    source_csv: Path,
    split_json: Path,
    output_dir: Path,
    dataset_version: str,
    device: str | None = None,
) -> pd.DataFrame:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, payload = load_checkpoint(Path(checkpoint), device=device)
    split = load_json(Path(split_json))
    df_all = pd.read_csv(source_csv)
    pred_horizon = int(payload["config"]["pred_horizon"])
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    all_baselines = {"current_hold": [], "last_slope": [], "moving_average": []}
    plot_payload: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for voyage_id in split["test_voyages"]:
        df_voyage = df_all[df_all["voyage_name"].eq(voyage_id)].reset_index(drop=True)
        if df_voyage.empty:
            continue
        decision_indices, pred, true = _forecast_voyage_load_only(
            df_voyage,
            model=model,
            payload=payload,
            device=device,
        )
        actual = df_voyage["load_total_kw"].to_numpy(dtype=float)
        baselines = _baseline_arrays(actual, decision_indices, pred_horizon)
        all_true.append(true)
        all_pred.append(pred)
        for name, arr in baselines.items():
            all_baselines[name].append(arr)
        if not plot_payload:
            for step in HORIZONS_TO_REPORT:
                horizon_idx = step - 1
                if horizon_idx < pred.shape[1]:
                    valid = np.isfinite(true[:, horizon_idx]) & np.isfinite(pred[:, horizon_idx])
                    plot_payload[step] = (
                        decision_indices[valid] + step,
                        true[valid, horizon_idx],
                        pred[valid, horizon_idx],
                    )
    true_all = np.vstack(all_true)
    pred_all = np.vstack(all_pred)
    baseline_all = {name: np.vstack(chunks) for name, chunks in all_baselines.items()}
    lstm_metrics = _metrics_for_horizons(true_all, pred_all)
    rows: list[dict[str, Any]] = []
    for step in HORIZONS_TO_REPORT:
        horizon_idx = step - 1
        row: dict[str, Any] = {
            "dataset_version": dataset_version,
            "model": "LSTM_load_only_h180_p60",
            "horizon_step": step,
            "horizon_seconds": step,
            "LSTM_MAE": lstm_metrics.get(f"MAE_h{step}", np.nan),
            "LSTM_RMSE": lstm_metrics.get(f"RMSE_h{step}", np.nan),
            "LSTM_WAPE": lstm_metrics.get(f"WAPE_h{step}", np.nan),
            "LSTM_Bias": lstm_metrics.get(f"Bias_h{step}", np.nan),
        }
        for name, arr in baseline_all.items():
            metrics = _metrics_for_horizons(true_all[:, [horizon_idx]], arr[:, [horizon_idx]])
            row[f"{name}_MAE"] = metrics.get("MAE_h1", np.nan)
            row[f"{name}_RMSE"] = metrics.get("RMSE_h1", np.nan)
            row[f"{name}_WAPE"] = metrics.get("WAPE_h1", np.nan)
            row[f"{name}_Bias"] = metrics.get("Bias_h1", np.nan)
        row["LSTM_better_than_current_hold"] = bool(row["LSTM_MAE"] < row["current_hold_MAE"])
        row["LSTM_better_than_last_slope"] = bool(row["LSTM_MAE"] < row["last_slope_MAE"])
        row["LSTM_better_than_moving_average"] = bool(row["LSTM_MAE"] < row["moving_average_MAE"])
        row["comment"] = "offline spline diagnostic; not measured 1 s data"
        rows.append(row)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(output_dir / "baseline_compare_metrics.csv", index=False, encoding="utf-8-sig")
    metrics_df.to_csv(output_dir / "metrics_by_horizon.csv", index=False, encoding="utf-8-sig")

    for step, (x_idx, true_values, pred_values) in plot_payload.items():
        limit = min(len(x_idx), 3000)
        plt.figure(figsize=(12, 4))
        plt.plot(x_idx[:limit] / 3600.0, true_values[:limit], label=f"actual h{step}", lw=0.8)
        plt.plot(x_idx[:limit] / 3600.0, pred_values[:limit], label=f"LSTM h{step}", lw=0.8)
        plt.xlabel("time in voyage (h)")
        plt.ylabel("load_total_kw")
        plt.title(f"{dataset_version} h{step} offline spline LSTM prediction")
        plt.grid(alpha=0.25)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(output_dir / f"test_prediction_h{step}.png", dpi=160)
        plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(metrics_df["horizon_seconds"], metrics_df["LSTM_MAE"], marker="o", label="LSTM")
    plt.plot(metrics_df["horizon_seconds"], metrics_df["current_hold_MAE"], marker="s", label="current-hold")
    plt.plot(metrics_df["horizon_seconds"], metrics_df["last_slope_MAE"], marker="^", label="last-slope")
    plt.plot(metrics_df["horizon_seconds"], metrics_df["moving_average_MAE"], marker="x", label="moving-average")
    plt.xlabel("horizon (s)")
    plt.ylabel("MAE (kW)")
    plt.title(f"{dataset_version} error vs horizon")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "error_vs_horizon.png", dpi=160)
    plt.close()
    return metrics_df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run load-only LSTM diagnostic on offline spline 1 s data.")
    parser.add_argument("--source_csv", type=Path, default=DEFAULT_SOURCE_CSV)
    parser.add_argument("--base_split_json", type=Path, default=DEFAULT_BASE_SPLIT_JSON)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset_version", default=DEFAULT_DATASET_VERSION)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max_train_windows", type=int, default=120000)
    parser.add_argument("--max_val_windows", type=int, default=30000)
    parser.add_argument("--train_window_stride", type=int, default=10)
    parser.add_argument("--val_window_stride", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset_dir = Path(args.output_root) / str(args.dataset_version)
    prepared = prepare_spline_lstm_source(
        source_csv=Path(args.source_csv),
        base_split_json=Path(args.base_split_json),
        output_dir=dataset_dir / "prepared",
        dataset_version=str(args.dataset_version),
    )
    horizon_weight = ",".join(["1.0"] * 60)
    train_args = argparse.Namespace(
        source_csv=prepared["source_csv"],
        split_json=prepared["split_json"],
        reference_meta=str(DEFAULT_REFERENCE_META),
        output_dir=dataset_dir,
        candidate="spline_1s_short_h180_p60_load_only",
        device=args.device,
        max_epochs=None,
        max_train_windows=int(args.max_train_windows),
        max_val_windows=int(args.max_val_windows),
        train_window_stride=int(args.train_window_stride),
        val_window_stride=int(args.val_window_stride),
        feature_set="base",
        feature_list=["load_total_kw"],
        overwrite_current=False,
        history_len=180,
        pred_horizon=60,
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
        batch_size=64,
        lr=2.0e-4,
        epochs=int(args.epochs),
        patience=int(args.patience),
        seed=42,
        grad_clip=1.0,
        huber_delta_kw=20.0,
        asym_under_weight=1.0,
        asym_high_load_bonus=0.0,
        asym_ramp_bonus=0.0,
        horizon_weight=horizon_weight,
        selection_metric="validation_MAE",
        auto_loss_thresholds=False,
        threshold_quantile=0.75,
    )
    summary = train_lstm_721(train_args)
    checkpoint = Path(summary["checkpoint"]).resolve()
    metrics_df = evaluate_spline_checkpoint(
        checkpoint=checkpoint,
        source_csv=Path(prepared["source_csv"]),
        split_json=Path(prepared["split_json"]),
        output_dir=dataset_dir,
        dataset_version=str(args.dataset_version),
        device=args.device,
    )
    config = {
        "dataset_version": str(args.dataset_version),
        "source_csv": prepared["source_csv"],
        "split_json": prepared["split_json"],
        "checkpoint": str(checkpoint),
        "history_len": 180,
        "pred_horizon": 60,
        "history_time_seconds": 180,
        "forecast_time_seconds": 60,
        "features": ["load_total_kw"],
        "sample_interval_seconds": 1.0,
        "online_feasible": False,
        "uses_future_endpoint": True,
        "data_label": "offline cubic-spline 1 s reconstruction, not measured 1 s data",
        "train_summary": summary,
        "metrics_by_horizon": metrics_df.to_dict(orient="records"),
    }
    write_json(dataset_dir / "config.json", config)
    print(json.dumps(config, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
