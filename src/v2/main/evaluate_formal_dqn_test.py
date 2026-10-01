"""One-time final Test evaluation for the sealed formal v2 DQN selection."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from ..data.formal_training_dataset import FormalTrainingDataset
from ..evaluation.final_test import (
    FINAL_TEST_CONFIRMATION,
    authenticate_final_test_selection,
    create_final_test_lock,
    write_final_test_results,
)
from ..evaluation.formal_policy import (
    FixedActionPolicy,
    GreedyDqnPolicy,
    evaluate_formal_policy,
    evaluate_formal_policy_with_power_traces,
)
from ..evaluation.power_trace_plots import write_power_trace_plots
from ..training.checkpoint import load_checkpoint
from ..training.dqn import DqnAgent
from ..training.schedule import EpisodeShuffleSchedule
from .history_dqn_evaluation import load_evaluation_profile
from .select_formal_dqn_checkpoint import DEFAULT_SELECTION_ROOT
from .train_formal_dqn import (
    DEFAULT_AIS_ROOT,
    DEFAULT_MODE_ROOT,
    DEFAULT_POWER_ROOT,
    REPOSITORY_ROOT,
)


DEFAULT_TEST_OUTPUT_ROOT = REPOSITORY_ROOT / "outputs" / "v2_formal_dqn_test"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the one-time formal v2 final Test evaluation"
    )
    parser.add_argument("--selection-dir", type=Path, default=DEFAULT_SELECTION_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_TEST_OUTPUT_ROOT)
    parser.add_argument("--power-root", type=Path, default=DEFAULT_POWER_ROOT)
    parser.add_argument("--ais-root", type=Path, default=DEFAULT_AIS_ROOT)
    parser.add_argument("--mode-root", type=Path, default=DEFAULT_MODE_ROOT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--experiment", choices=("H1", "H2", "H3", "H4"), default="H1")
    parser.add_argument("--reward-scale", type=Path)
    parser.add_argument("--plot-dir", type=Path)
    parser.add_argument("--confirm-final-test", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.confirm_final_test != FINAL_TEST_CONFIRMATION:
        _parser().error(
            f"--confirm-final-test must be exactly {FINAL_TEST_CONFIRMATION}"
        )
    output = Path(args.output_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if args.plot_dir is not None and (
        Path(args.plot_dir).exists() or Path(args.plot_dir).is_symlink()
    ):
        raise FileExistsError(args.plot_dir)

    authorization = authenticate_final_test_selection(args.selection_dir)
    create_final_test_lock(output, authorization)

    dataset = FormalTrainingDataset.open(args.power_root, args.ais_root, args.mode_root)
    train_ids = dataset.split_episode_ids("train")
    profile, config, _ = load_evaluation_profile(
        experiment_id=args.experiment,
        reward_scale_path=args.reward_scale,
        expected_manifest_hashes=dict(authorization.input_manifest_hashes),
        rounds=40,
    )
    agent = DqnAgent(config, seed=42, device=args.device)
    schedule = EpisodeShuffleSchedule(train_ids, seed=42)
    best_path = Path(args.selection_dir) / "best_validation.pt"
    metadata = load_checkpoint(best_path, agent=agent, schedule=schedule)
    if metadata.round_index != authorization.selected_round:
        raise ValueError("best checkpoint round differs from selection authorization")
    if metadata.episode_position != 0:
        raise ValueError("final Test requires a completed-round checkpoint")

    test_episodes = dataset.load_final_test(authorization)
    traces = None
    if args.plot_dir is None:
        dqn_result = evaluate_formal_policy(
            episodes=test_episodes,
            policy=GreedyDqnPolicy(agent, policy_id="greedy_dqn"),
        )
    else:
        dqn_result, traces = evaluate_formal_policy_with_power_traces(
            episodes=test_episodes,
            policy=GreedyDqnPolicy(agent, policy_id="greedy_dqn"),
        )
    fixed_result = evaluate_formal_policy(
        episodes=test_episodes,
        policy=FixedActionPolicy("w_8_1_1"),
    )
    result_digest = write_final_test_results(
        output,
        authorization,
        (dqn_result, fixed_result),
    )
    if traces is not None:
        write_power_trace_plots(
            Path(args.plot_dir),
            traces,
            policy_id=f"{profile.experiment_id}_round_{authorization.selected_round:03d}",
        )
    print(
        f"FINAL_TEST=COMPLETE experiment={profile.experiment_id} "
        f"selected_round={authorization.selected_round} "
        f"episodes={len(test_episodes)} "
        f"dqn_completed={dqn_result.completed_episodes} "
        f"dqn_failed={dqn_result.failed_episodes} "
        f"dqn_raw_economic_cost_cny={dqn_result.raw_economic_cost_cny:.9f} "
        f"fixed_completed={fixed_result.completed_episodes} "
        f"fixed_failed={fixed_result.failed_episodes} "
        f"fixed_raw_economic_cost_cny={fixed_result.raw_economic_cost_cny:.9f} "
        f"test_payloads_opened={dataset.opened_test_payloads} "
        f"result_digest={result_digest}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
