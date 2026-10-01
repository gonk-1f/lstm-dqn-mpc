"""Screen LPF time constants on Validation without opening Test."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
from pathlib import Path
import shutil
from typing import Sequence
import uuid

from ..config import TAU_LPF_SECONDS
from ..data.formal_training_dataset import FormalTrainingDataset
from ..evaluation.checkpoint_selection import load_selection_outputs
from ..evaluation.formal_policy import (
    GreedyDqnPolicy,
    evaluate_formal_policy_with_power_traces,
)
from ..evaluation.power_trace_plots import write_power_trace_plots
from ..evaluation.tau_lpf_screen import build_tau_screen_summary
from ..training.checkpoint import load_checkpoint
from ..training.dqn import DqnAgent
from ..training.schedule import EpisodeShuffleSchedule
from .history_dqn_evaluation import load_evaluation_profile
from .run_reward_scale_calibration import _manifest_hashes
from .train_formal_dqn import (
    DEFAULT_AIS_ROOT,
    DEFAULT_MODE_ROOT,
    DEFAULT_POWER_ROOT,
    REPOSITORY_ROOT,
)


DEFAULT_SELECTION_ROOT = (
    REPOSITORY_ROOT / "outputs" / "v2_history_dqn_study" / "H4_formal_selection_40"
)
DEFAULT_REWARD_SCALE = (
    REPOSITORY_ROOT
    / "outputs"
    / "v2_history_dqn_study"
    / "reward_scale_calibration.json"
)
DEFAULT_OUTPUT_ROOT = (
    REPOSITORY_ROOT / "outputs" / "v2_history_dqn_study" / "tau_lpf_validation_screen"
)


def _positive_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("tau must be positive and finite")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Screen LPF tau values using H4 Validation episodes only"
    )
    parser.add_argument("--selection-dir", type=Path, default=DEFAULT_SELECTION_ROOT)
    parser.add_argument("--reward-scale", type=Path, default=DEFAULT_REWARD_SCALE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--power-root", type=Path, default=DEFAULT_POWER_ROOT)
    parser.add_argument("--ais-root", type=Path, default=DEFAULT_AIS_ROOT)
    parser.add_argument("--mode-root", type=Path, default=DEFAULT_MODE_ROOT)
    parser.add_argument("--taus", type=_positive_float, nargs="+", default=(180.0, 300.0))
    parser.add_argument("--device", default="cpu")
    return parser


def _csv_bytes(rows: tuple[dict[str, object], ...]) -> bytes:
    if not rows:
        raise ValueError("CSV rows must not be empty")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _tau_label(value: float) -> str:
    return str(int(value)) if value.is_integer() else str(value).replace(".", "p")


def write_tau_screen_outputs(
    output_directory: Path,
    results: tuple[
        tuple[dict[str, object], tuple[dict[str, object], ...], tuple[object, ...]],
        ...,
    ],
    *,
    selected_round: int,
    input_manifest_hashes: dict[str, str],
    test_payloads_opened: int,
) -> Path:
    """Atomically write comparison tables, diagnostics, and Validation plots."""

    output = Path(output_directory)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if not results:
        raise ValueError("tau screen results must not be empty")
    if test_payloads_opened != 0:
        raise RuntimeError("Test payload was opened during tau screen")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    temporary.mkdir()
    try:
        summaries: list[dict[str, object]] = []
        for summary, episode_rows, traces in results:
            tau = float(summary["tau_lpf_seconds"])
            label = _tau_label(tau)
            tau_root = temporary / f"tau_{label}"
            tau_root.mkdir()
            (tau_root / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            (tau_root / "episode_metrics.csv").write_bytes(_csv_bytes(episode_rows))
            write_power_trace_plots(
                tau_root / "power_plots",
                traces,
                policy_id=str(summary["policy_id"]),
                artifact_title=f"H4 round {selected_round} tau={label} s Validation power traces",
            )
            summaries.append(summary)

        summary_rows = tuple(summaries)
        (temporary / "tau_comparison.csv").write_bytes(_csv_bytes(summary_rows))
        manifest = {
            "screening_status": "PRETRAIN_ENVIRONMENT_SCREEN_ONLY",
            "dataset_split": "VALIDATION_ONLY",
            "selected_h4_round": selected_round,
            "checkpoint_trained_tau_seconds": TAU_LPF_SECONDS,
            "candidate_tau_seconds": [row["tau_lpf_seconds"] for row in summaries],
            "input_manifest_sha256": input_manifest_hashes,
            "test_payloads_opened": test_payloads_opened,
            "interpretation": (
                f"The frozen tau={TAU_LPF_SECONDS:g} s H4 checkpoint was evaluated under changed plant/control "
                "dynamics. These results screen tau candidates and are not final retrained-model results."
            ),
            "summaries": summaries,
        }
        (temporary / "screening_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.rename(output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output / "screening_manifest.json"


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = Path(args.output_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    taus = tuple(float(value) for value in args.taus)
    if len(set(taus)) != len(taus):
        raise ValueError("tau candidates must be unique")

    selection = load_selection_outputs(Path(args.selection_dir))
    current_hashes = _manifest_hashes(args.power_root, args.ais_root, args.mode_root)
    if dict(selection.input_manifest_hashes) != current_hashes:
        raise ValueError("current dataset manifests differ from H4 selection")

    dataset = FormalTrainingDataset.open(args.power_root, args.ais_root, args.mode_root)
    train_ids = dataset.split_episode_ids("train")
    validation = dataset.load_validation()
    if dataset.opened_test_payloads != 0:
        raise RuntimeError("Test payload was opened before tau screen")

    profile, config, _ = load_evaluation_profile(
        experiment_id="H4",
        reward_scale_path=args.reward_scale,
        expected_manifest_hashes=current_hashes,
        rounds=40,
    )
    agent = DqnAgent(config, seed=42, device=args.device)
    schedule = EpisodeShuffleSchedule(train_ids, seed=42)
    metadata = load_checkpoint(
        Path(args.selection_dir) / "best_validation.pt",
        agent=agent,
        schedule=schedule,
    )
    if metadata.round_index != selection.selected_round or metadata.episode_position != 0:
        raise ValueError("selected checkpoint metadata differs from selection")

    results = []
    for tau in taus:
        policy_id = f"H4_round_{selection.selected_round:03d}_tau_{_tau_label(tau)}"
        evaluation, traces = evaluate_formal_policy_with_power_traces(
            episodes=validation,
            policy=GreedyDqnPolicy(agent, policy_id=policy_id),
            tau_lpf_seconds=tau,
        )
        summary, episode_rows = build_tau_screen_summary(
            tau_lpf_seconds=tau,
            evaluation=evaluation,
            traces=traces,
            checkpoint_trained_tau_seconds=TAU_LPF_SECONDS,
        )
        results.append((summary, episode_rows, traces))
        print(
            f"validation_tau={tau:g} completed={summary['completed_episodes']}/"
            f"{summary['episode_count']} raw_cost_cny={summary['raw_economic_cost_cny']:.9f} "
            f"fc_variation_kw={summary['fc_total_variation_kw']:.9f} "
            f"battery_abs_energy_kwh={summary['battery_absolute_energy_kwh']:.9f}",
            flush=True,
        )

    if dataset.opened_test_payloads != 0:
        raise RuntimeError("Test payload was opened during tau screen")
    manifest_path = write_tau_screen_outputs(
        output,
        tuple(results),
        selected_round=selection.selected_round,
        input_manifest_hashes=current_hashes,
        test_payloads_opened=dataset.opened_test_payloads,
    )
    print(
        f"TAU_SCREEN=COMPLETE experiment={profile.experiment_id} "
        f"selected_round={selection.selected_round} candidates={','.join(f'{value:g}' for value in taus)} "
        f"dataset_split=VALIDATION_ONLY test_payloads_opened={dataset.opened_test_payloads} "
        f"manifest={manifest_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
