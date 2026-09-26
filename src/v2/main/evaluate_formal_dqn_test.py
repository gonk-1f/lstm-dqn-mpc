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
)
from ..training.checkpoint import load_checkpoint
from ..training.dqn import DqnAgent, DqnTrainingConfig
from ..training.schedule import EpisodeShuffleSchedule
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

    authorization = authenticate_final_test_selection(args.selection_dir)
    create_final_test_lock(output, authorization)

    dataset = FormalTrainingDataset.open(args.power_root, args.ais_root, args.mode_root)
    train_ids = dataset.split_episode_ids("train")
    agent = DqnAgent(DqnTrainingConfig.formal_baseline(), seed=42, device=args.device)
    schedule = EpisodeShuffleSchedule(train_ids, seed=42)
    best_path = Path(args.selection_dir) / "best_validation.pt"
    metadata = load_checkpoint(best_path, agent=agent, schedule=schedule)
    if metadata.round_index != authorization.selected_round:
        raise ValueError("best checkpoint round differs from selection authorization")
    if metadata.episode_position != 0:
        raise ValueError("final Test requires a completed-round checkpoint")

    test_episodes = dataset.load_final_test(authorization)
    dqn_result = evaluate_formal_policy(
        test_episodes,
        GreedyDqnPolicy(agent, policy_id="greedy_dqn"),
    )
    fixed_result = evaluate_formal_policy(
        test_episodes,
        FixedActionPolicy("w_8_1_1"),
    )
    result_digest = write_final_test_results(
        output,
        authorization,
        (dqn_result, fixed_result),
    )
    print(
        f"FINAL_TEST=COMPLETE selected_round={authorization.selected_round} "
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
