"""Train the LSTM forecaster on the 66-voyage energy-side total-load dataset."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

SRC = Path(__file__).resolve().parents[1]
PROJ = Path(__file__).resolve().parents[2]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from forecasting.feature_pipeline import (  # noqa: E402
    clean_total_load_feature_columns,
    clean_total_load_feature_columns_1s,
    clean_total_load_speed_feature_columns,
    prepare_lstm_features,
)
from forecasting.lstm_load_predictor import inverse_target, load_checkpoint, transform  # noqa: E402
from run_train_lstm_721 import (  # noqa: E402
    DEFAULT_REFERENCE_META,
    compute_train_loss_thresholds,
    detailed_horizon_metrics,
    train_lstm_721,
)


DEFAULT_SOURCE_CSV = PROJ / "outputs" / "total_load_dataset_build" / "total_load_66_segments.csv"
DEFAULT_SPLIT_JSON = PROJ / "outputs" / "config" / "voyage_split_total_load_721.json"
DEFAULT_OUTPUT_DIR = PROJ / "outputs" / "lstm_total_load_721"
DEFAULT_CANDIDATE = "candidate_A_loss_recalib"
FLAT_CHECKPOINT = DEFAULT_OUTPUT_DIR / "checkpoints" / "best_lstm_load_predictor.pt"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _feature_set_for_checkpoint(checkpoint: Path) -> str:
    feature_meta_path = checkpoint.with_suffix(".feature_set.json")
    if feature_meta_path.exists():
        meta = load_json(feature_meta_path)
        return str(meta.get("feature_set", "rolling"))
    return "rolling"


def _forecast_voyage(
    df_voyage: pd.DataFrame,
    *,
    model: torch.nn.Module,
    payload: dict[str, Any],
    feature_set: str,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cfg = payload["config"]
    history_len = int(cfg["history_len"])
    pred_horizon = int(cfg["pred_horizon"])
    prepared = prepare_lstm_features(df_voyage, feature_set)
    features = list(payload["features"])
    for col in ["time_sin", "time_cos"]:
        if col in prepared.columns and col not in features:
            features.append(col)
    for col in features:
        if col not in prepared.columns:
            prepared[col] = 0.0
    values = prepared[features].astype(float).ffill().bfill().fillna(0.0).to_numpy()
    values = transform(values, payload["feature_scaler"])
    actual = prepared["load_total_kw"].astype(float).ffill().bfill().fillna(0.0).to_numpy()
    decision_indices = np.arange(history_len - 1, len(prepared), dtype=int)
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
            pred[row_idx] = np.maximum(inverse_target(raw, payload["target_scaler"]), 0.0)
            for horizon_idx in range(pred_horizon):
                true_idx = decision_t + horizon_idx + 1
                if true_idx < len(actual):
                    true[row_idx, horizon_idx] = actual[true_idx]
    return decision_indices, pred, true


def _horizon_metrics(true: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    return detailed_horizon_metrics(true, pred)


def _region_metrics(
    true: np.ndarray,
    pred: np.ndarray,
    *,
    high_load_threshold_kw: float,
    ramp_threshold_kw: float,
) -> dict[str, float]:
    truth = np.asarray(true, dtype=float)
    prediction = np.asarray(pred, dtype=float)
    mask = np.isfinite(truth) & np.isfinite(prediction)
    abs_err = np.abs(prediction - truth)

    ramp = np.zeros_like(truth)
    if truth.shape[1] > 1:
        ramp[:, 1:] = np.abs(np.diff(truth, axis=1))

    def masked_mae(region: np.ndarray) -> float:
        combined = mask & region
        if not np.any(combined):
            return float("nan")
        return float(np.mean(abs_err[combined]))

    return {
        "high_load_MAE": masked_mae(truth > float(high_load_threshold_kw)),
        "ramp_region_MAE": masked_mae(ramp > float(ramp_threshold_kw)),
        "zero_load_MAE": masked_mae(truth < 1.0),
        "nonzero_load_MAE": masked_mae(truth >= 1.0),
    }


def evaluate_checkpoint_on_test(
    *,
    checkpoint: Path,
    source_csv: Path,
    split_json: Path,
    output_dir: Path,
    device: str | None = None,
    high_load_threshold_kw: float | None = None,
    ramp_threshold_kw: float | None = None,
    sample_interval_seconds: float = 30.0,
) -> pd.DataFrame:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, payload = load_checkpoint(checkpoint, device=device)
    feature_set = _feature_set_for_checkpoint(checkpoint)
    split = load_json(split_json)
    df_all = pd.read_csv(source_csv)
    if high_load_threshold_kw is None or ramp_threshold_kw is None:
        train_df = df_all[df_all["voyage_name"].isin(split["train_voyages"])].reset_index(drop=True)
        high_load_threshold_kw, ramp_threshold_kw = compute_train_loss_thresholds(train_df, quantile=0.75)
    rows: list[dict[str, Any]] = []
    plot_specs: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    for voyage_id in split["test_voyages"]:
        df_voyage = df_all[df_all["voyage_name"] == voyage_id].reset_index(drop=True)
        if df_voyage.empty:
            raise ValueError(f"Test voyage not found in source_csv: {voyage_id}")
        decision_indices, pred, true = _forecast_voyage(
            df_voyage,
            model=model,
            payload=payload,
            feature_set=feature_set,
            device=device,
        )
        metrics = _horizon_metrics(true, pred)
        metrics.update(
            _region_metrics(
                true,
                pred,
                high_load_threshold_kw=float(high_load_threshold_kw),
                ramp_threshold_kw=float(ramp_threshold_kw),
            )
        )
        rows.append(
            {
                "voyage_id": voyage_id,
                "file_name": str(df_voyage["file_name"].iloc[0]) if "file_name" in df_voyage.columns else voyage_id,
                "rows": int(len(df_voyage)),
                "duration_h": float(len(df_voyage) * float(sample_interval_seconds) / 3600.0),
                **metrics,
            }
        )
        all_true.append(true)
        all_pred.append(pred)
        plot_specs.append((voyage_id, decision_indices, pred[:, 0], df_voyage["load_total_kw"].to_numpy(dtype=float)))

    combined_true = np.vstack(all_true)
    combined_pred = np.vstack(all_pred)
    combined_metrics = _horizon_metrics(combined_true, combined_pred)
    combined_metrics.update(
        _region_metrics(
            combined_true,
            combined_pred,
            high_load_threshold_kw=float(high_load_threshold_kw),
            ramp_threshold_kw=float(ramp_threshold_kw),
        )
    )
    rows.append({"voyage_id": "all", "file_name": "all", "rows": int(len(combined_true)), "duration_h": np.nan, **combined_metrics})
    metrics_df = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(output_dir / "metrics.csv", index=False, encoding="utf-8-sig")

    n = min(len(plot_specs), 7)
    fig, axes = plt.subplots(n, 1, figsize=(14, max(3, 2.2 * n)), sharex=False)
    if n == 1:
        axes = [axes]
    for ax, (voyage_id, decision_indices, pred_h1, actual) in zip(axes, plot_specs[:n]):
        valid = decision_indices
        t = valid * float(sample_interval_seconds) / 3600.0
        actual_h1 = np.full_like(pred_h1, np.nan, dtype=float)
        mask = valid + 1 < len(actual)
        actual_h1[mask] = actual[valid[mask] + 1]
        ax.plot(t, actual_h1, color="black", lw=0.7, alpha=0.75, label="Actual h1")
        ax.plot(t, pred_h1, color="#1f77b4", lw=0.7, alpha=0.85, label="Predicted h1")
        ax.set_ylabel("kW")
        ax.set_title(str(voyage_id), fontsize=9)
        ax.grid(alpha=0.18)
    axes[0].legend(fontsize=8)
    axes[-1].set_xlabel("Time (hours)")
    fig.tight_layout()
    fig.savefig(output_dir / "prediction_examples.png", dpi=180)
    plt.close(fig)
    return metrics_df


def _copy_flat_checkpoint(nested_checkpoint: Path, flat_checkpoint: Path) -> None:
    flat_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for suffix in [".pt", ".json", ".feature_set.json"]:
        src = nested_checkpoint.with_suffix(suffix)
        if src.exists():
            shutil.copy2(src, flat_checkpoint.with_suffix(suffix))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train total-load LSTM on the 66-voyage 7:2:1 split.")
    parser.add_argument("--source_csv", type=Path, default=DEFAULT_SOURCE_CSV)
    parser.add_argument("--split_json", type=Path, default=DEFAULT_SPLIT_JSON)
    parser.add_argument("--reference_meta", type=Path, default=DEFAULT_REFERENCE_META)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--candidate", default=DEFAULT_CANDIDATE)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--max_train_windows", type=int, default=None)
    parser.add_argument("--max_val_windows", type=int, default=None)
    parser.add_argument("--train_window_stride", type=int, default=1)
    parser.add_argument("--val_window_stride", type=int, default=1)
    parser.add_argument("--overwrite_current", action="store_true")
    parser.add_argument("--history_len", type=int, default=18)
    parser.add_argument("--pred_horizon", type=int, default=6)
    parser.add_argument("--hidden_size", type=int, default=192)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", "--learning_rate", dest="lr", type=float, default=2.0e-4)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--huber_delta_kw", type=float, default=20.0)
    parser.add_argument("--asym_under_weight", type=float, default=1.5)
    parser.add_argument("--asym_high_load_bonus", type=float, default=0.2)
    parser.add_argument("--asym_ramp_bonus", type=float, default=0.1)
    parser.add_argument("--horizon_weight", default="1.5,1.3,1.1,1.0,0.8,0.6")
    parser.add_argument("--selection_metric", default="validation_weighted_MAE_h1_h3")
    parser.add_argument("--feature_mode", choices=["load_only", "speed", "rolling_1s"], default="load_only")
    parser.add_argument("--sample_interval_seconds", type=float, default=30.0)
    parser.add_argument("--no_auto_loss_thresholds", dest="auto_loss_thresholds", action="store_false")
    parser.add_argument("--threshold_quantile", type=float, default=0.75)
    parser.set_defaults(auto_loss_thresholds=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.feature_mode == "speed":
        args.feature_list = clean_total_load_speed_feature_columns()
        args.feature_set = "rolling"
    elif args.feature_mode == "rolling_1s":
        args.feature_list = clean_total_load_feature_columns_1s()
        args.feature_set = "rolling_1s"
    else:
        args.feature_list = clean_total_load_feature_columns()
        args.feature_set = "rolling"
    summary = train_lstm_721(args)
    nested_checkpoint = Path(summary["checkpoint"]).resolve()
    flat_checkpoint = Path(args.output_dir) / "checkpoints" / "best_lstm_load_predictor.pt"
    _copy_flat_checkpoint(nested_checkpoint, flat_checkpoint)
    metrics_df = evaluate_checkpoint_on_test(
        checkpoint=flat_checkpoint,
        source_csv=Path(args.source_csv),
        split_json=Path(args.split_json),
        output_dir=Path(args.output_dir),
        device=args.device,
        sample_interval_seconds=float(args.sample_interval_seconds),
    )
    run_config_path = Path(args.output_dir) / "run_config.json"
    config_payload = load_json(run_config_path) if run_config_path.exists() else {}
    config_payload.update(
        {
            "source_csv": str(Path(args.source_csv).resolve()),
            "split_json": str(Path(args.split_json).resolve()),
            "output_dir": str(Path(args.output_dir).resolve()),
            "checkpoint": str(flat_checkpoint.resolve()),
            "nested_checkpoint": str(nested_checkpoint),
            "load_definition": "fuel_cell_total_kw + battery_total_kw",
            "load_scope": "energy_side_equivalent_total_load",
            "target_load": "load_total_kw",
            "feature_mode": str(args.feature_mode),
            "sample_interval_seconds": float(args.sample_interval_seconds),
            "test_metrics_csv": str((Path(args.output_dir) / "metrics.csv").resolve()),
        }
    )
    write_json(Path(args.output_dir) / "config.json", config_payload)
    result = dict(summary)
    all_row = metrics_df[metrics_df["voyage_id"] == "all"].iloc[0].to_dict()
    result.update({f"test_{key}": value for key, value in all_row.items() if key.startswith(("RMSE_", "MAE_", "WAPE_"))})
    result["checkpoint"] = str(flat_checkpoint)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
