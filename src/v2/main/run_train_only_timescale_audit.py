"""Run the bounded Task 9 diagnostic through its lazy Train-only guard."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import sys

from ..analysis.action_screening import DataSplit, DatasetProvenance, HeldOutSelectionError
from ..analysis.timescale_audit import run_timescale_audit
from ..contracts import DATASET_VERSION


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train-only N=5, M={5,10} diagnostic. It never selects a formal "
            "time scale and never starts DQN training."
        )
    )
    parser.add_argument("--split", default="Train", choices=[item.value for item in DataSplit])
    parser.add_argument("--train-json", type=Path)
    parser.add_argument("--provenance-id", default="repository-current-unverified")
    parser.add_argument("--sample-seconds", type=float, default=30.0)
    parser.add_argument("--autocorrelation-lags", default="1,5")
    parser.add_argument("--rolling-window-samples", type=int, default=5)
    parser.add_argument("--change-threshold", type=float)
    return parser


def _parse_lags(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError("autocorrelation lags must be comma-separated integers") from exc
    if not result:
        raise ValueError("at least one autocorrelation lag is required")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    split = DataSplit(args.split)
    provenance = DatasetProvenance(DATASET_VERSION, args.provenance_id, split)

    if args.train_json is None:
        print(
            "NO-GO: no audited Train JSON payload was supplied; current repository "
            "raw-data provenance is insufficient.",
            file=sys.stderr,
        )
        return 2
    if args.change_threshold is None:
        print(
            "NO-GO: --change-threshold must be pre-registered from Train evidence; "
            "the entrypoint will not invent a default.",
            file=sys.stderr,
        )
        return 2

    def payload_loader() -> object:
        return json.loads(args.train_json.read_text(encoding="utf-8"))

    try:
        result = run_timescale_audit(
            provenance=provenance,
            payload_loader=payload_loader,
            sample_seconds=args.sample_seconds,
            autocorrelation_lags=_parse_lags(args.autocorrelation_lags),
            rolling_window_samples=args.rolling_window_samples,
            change_threshold=args.change_threshold,
        )
    except (HeldOutSelectionError, OSError, TypeError, ValueError) as exc:
        print(f"NO-GO: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "dataset_version": result.provenance.dataset_version,
                "provenance_id": result.provenance.provenance_id,
                "split": result.provenance.split.value,
                "sample_count": result.sample_count,
                "sample_seconds": result.sample_seconds,
                "n_mpc": result.n_mpc,
                "candidate_switch_steps": result.candidate_switch_steps,
                "formal_selection_status": result.formal_selection_status,
                "selected_dqn_switch_steps": result.selected_dqn_switch_steps,
                "digest": result.digest,
            },
            sort_keys=True,
        )
    )
    return 0 if result.formal_selection_status == "GO" else 2


if __name__ == "__main__":
    raise SystemExit(main())
