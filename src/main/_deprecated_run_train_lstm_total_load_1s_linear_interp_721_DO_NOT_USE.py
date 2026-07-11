"""DEPRECATED: train on invalid linear 1 s reconstructed data.

DEPRECATED: this entrypoint trains on 30 s to 1 s linear-interpolation outputs.
The reconstructed labels are non-causal and must not be used as valid online
high-frequency forecasting evidence.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parents[1]
PROJ = Path(__file__).resolve().parents[2]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from forecasting.feature_pipeline import clean_total_load_feature_columns_1s  # noqa: E402
from run_train_lstm_721 import DEFAULT_REFERENCE_META, train_lstm_721  # noqa: E402
from run_train_lstm_total_load_721 import evaluate_checkpoint_on_test, load_json, write_json  # noqa: E402


DEFAULT_SOURCE_CSV = PROJ / "outputs" / "total_load_dataset_1s_build" / "total_load_66_segments.csv"
DEFAULT_SPLIT_JSON = PROJ / "outputs" / "config" / "voyage_split_total_load_1s_721.json"
DEFAULT_OUTPUT_DIR = PROJ / "outputs" / "lstm_total_load_1s_721"
DEFAULT_CANDIDATE = "candidate_1s_h30_p6_weighted_huber"


def _copy_flat_checkpoint(nested_checkpoint: Path, flat_checkpoint: Path) -> None:
    flat_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for suffix in [".pt", ".json", ".feature_set.json"]:
        src = nested_checkpoint.with_suffix(suffix)
        if src.exists():
            shutil.copy2(src, flat_checkpoint.with_suffix(suffix))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train 1 s total-load LSTM on the chronological 66-voyage split.")
    parser.add_argument("--source_csv", type=Path, default=DEFAULT_SOURCE_CSV)
    parser.add_argument("--split_json", type=Path, default=DEFAULT_SPLIT_JSON)
    parser.add_argument("--reference_meta", type=Path, default=DEFAULT_REFERENCE_META)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--candidate", default=DEFAULT_CANDIDATE)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--max_train_windows", type=int, default=None)
    parser.add_argument("--max_val_windows", type=int, default=None)
    parser.add_argument("--train_window_stride", type=int, default=10)
    parser.add_argument("--val_window_stride", type=int, default=5)
    parser.add_argument("--overwrite_current", action="store_true")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--pred_horizon", type=int, default=6)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", "--learning_rate", dest="lr", type=float, default=1.0e-4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--huber_delta_kw", type=float, default=20.0)
    parser.add_argument("--asym_under_weight", type=float, default=1.5)
    parser.add_argument("--asym_high_load_bonus", type=float, default=0.2)
    parser.add_argument("--asym_ramp_bonus", type=float, default=0.1)
    parser.add_argument("--horizon_weight", default="2.0,1.5,1.2,1.0,0.8,0.6")
    parser.add_argument("--selection_metric", default="validation_weighted_MAE_h1_h3")
    parser.add_argument("--feature_mode", choices=["rolling_1s"], default="rolling_1s")
    parser.add_argument("--sample_interval_seconds", type=float, default=1.0)
    parser.add_argument("--no_auto_loss_thresholds", dest="auto_loss_thresholds", action="store_false")
    parser.add_argument("--threshold_quantile", type=float, default=0.75)
    parser.set_defaults(auto_loss_thresholds=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.feature_list = clean_total_load_feature_columns_1s()
    args.feature_set = "rolling_1s"
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
            "load_scope": "energy_side_equivalent_total_load_1s",
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
