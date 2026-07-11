"""Run fixed LSTM-H2-MPC on the 66-voyage energy-side total-load test split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
PROJ = Path(__file__).resolve().parents[2]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import run_lstm_mpc_test as base  # noqa: E402


DEFAULT_LSTM_CKPT = PROJ / "outputs" / "lstm_total_load_721" / "checkpoints" / "best_lstm_load_predictor.pt"
DEFAULT_SPLIT_JSON = PROJ / "outputs" / "config" / "voyage_split_total_load_721.json"
DEFAULT_SOURCE_CSV = PROJ / "outputs" / "total_load_dataset_build" / "total_load_66_segments.csv"
DEFAULT_OUT_DIR = PROJ / "outputs" / "lstm_mpc_total_load_test"
LOAD_DEFINITION = "fuel_cell_total_kw + battery_total_kw"
CAPACITY_BASIS = "full ship energy storage capacity for energy-side equivalent total load"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run fixed LSTM-MPC on total-load test voyages.")
    parser.add_argument("--weight_set", default=base.TOTAL_LOAD_WEIGHT_SET)
    parser.add_argument("--lstm_ckpt", type=Path, default=DEFAULT_LSTM_CKPT)
    parser.add_argument("--split_json", type=Path, default=DEFAULT_SPLIT_JSON)
    parser.add_argument("--source_csv", type=Path, default=DEFAULT_SOURCE_CSV)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--voyage", default=None)
    parser.add_argument("--soc", type=float, default=0.55)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--no_plots", action="store_true")
    return parser


def configure_base_module(args: argparse.Namespace) -> None:
    base.LSTM_CKPT = Path(args.lstm_ckpt)
    base.SPLIT_JSON = Path(args.split_json)
    base.SOURCE_CSV = Path(args.source_csv)
    base.OUT_DIR = Path(args.output_dir)
    base.LOAD_DEFINITION = LOAD_DEFINITION
    base.CAPACITY_BASIS = CAPACITY_BASIS
    base.P6_WEIGHT_SOURCE = str(args.weight_set)
    base.CURRENT_FIXED_WEIGHT_SET = str(args.weight_set)


def main() -> None:
    args = build_parser().parse_args()
    configure_base_module(args)
    metrics_df, horizon_df, _ = base.run_all(
        weight_set=str(args.weight_set),
        output_dir=Path(args.output_dir),
        voyage=args.voyage,
        init_soc=float(args.soc),
        make_plots=not bool(args.no_plots),
        write_outputs=True,
        max_steps=args.max_steps,
        device=args.device,
    )
    print(f"Saved: {Path(args.output_dir)}")
    if not metrics_df.empty:
        cols = [
            "voyage_id",
            "file_name",
            "duration_h",
            "soc_end",
            "soc_min",
            "H2_total_kg",
            "charge_sustaining_adjusted_H2",
            "solver_success_rate",
        ]
        print(metrics_df[[col for col in cols if col in metrics_df.columns]].to_string(index=False))
    if not horizon_df.empty:
        cols = ["voyage_id", "MAE_h1", "WAPE_h1", "MAE_h6", "WAPE_h6"]
        print(horizon_df[[col for col in cols if col in horizon_df.columns]].to_string(index=False))


if __name__ == "__main__":
    main()
