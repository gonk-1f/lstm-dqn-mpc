"""Validation-only selector for formal v2 DQN round checkpoints."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import time
from typing import Sequence

from ..data.formal_training_dataset import FormalTrainingDataset
from ..evaluation.checkpoint_selection import (
    ValidationCandidate,
    authenticate_checkpoint_candidates,
    make_selection_manifest,
    write_selection_outputs,
)
from ..evaluation.formal_policy import GreedyDqnPolicy, evaluate_formal_policy
from ..training.checkpoint import load_checkpoint
from ..training.dqn import DqnAgent
from ..training.schedule import EpisodeShuffleSchedule
from .history_dqn_evaluation import load_evaluation_profile
from .train_formal_dqn import (
    DEFAULT_AIS_ROOT,
    DEFAULT_MODE_ROOT,
    DEFAULT_POWER_ROOT,
    REPOSITORY_ROOT,
)


DEFAULT_CHECKPOINT_ROOT = REPOSITORY_ROOT / "outputs" / "v2_formal_dqn_v3"
DEFAULT_SELECTION_ROOT = REPOSITORY_ROOT / "outputs" / "v2_formal_dqn_selection"


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select the formal v2 DQN checkpoint using Validation only"
    )
    parser.add_argument("--power-root", type=Path, default=DEFAULT_POWER_ROOT)
    parser.add_argument("--ais-root", type=Path, default=DEFAULT_AIS_ROOT)
    parser.add_argument("--mode-root", type=Path, default=DEFAULT_MODE_ROOT)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_SELECTION_ROOT)
    parser.add_argument("--first-round", type=_positive_int, default=1)
    parser.add_argument("--last-round", type=_positive_int, default=40)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--experiment", choices=("H1", "H2", "H3", "H4"), default="H1")
    parser.add_argument("--reward-scale", type=Path)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _input_manifest_hashes(args: argparse.Namespace) -> dict[str, str]:
    paths = {
        "power": Path(args.power_root) / "metadata" / "sample_manifest.csv",
        "ais": Path(args.ais_root) / "metadata" / "sample_manifest.csv",
        "modes": Path(args.mode_root) / "metadata" / "sample_manifest.csv",
    }
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("formal power/AIS/mode manifest is missing")
    return {name: _sha256(path) for name, path in paths.items()}


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = Path(args.output_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if args.first_round > args.last_round:
        raise ValueError("first round must not exceed last round")
    required_rounds = tuple(range(args.first_round, args.last_round + 1))

    dataset = FormalTrainingDataset.open(args.power_root, args.ais_root, args.mode_root)
    train_ids = dataset.split_episode_ids("train")
    validation = dataset.load_validation()
    input_hashes = _input_manifest_hashes(args)
    profile, config, expected_identity = load_evaluation_profile(
        experiment_id=args.experiment,
        reward_scale_path=args.reward_scale,
        expected_manifest_hashes=input_hashes,
        rounds=args.last_round,
    )
    checkpoints = authenticate_checkpoint_candidates(
        args.checkpoint_dir,
        required_rounds=required_rounds,
        expected_training_identity=expected_identity,
    )
    if dataset.opened_test_payloads != 0:
        raise RuntimeError("Test payload was opened during Validation selection")

    started = time.perf_counter()
    candidates: list[ValidationCandidate] = []
    by_round = {item.round_index: item for item in checkpoints}
    for item in checkpoints:
        agent = DqnAgent(config, seed=42, device=args.device)
        schedule = EpisodeShuffleSchedule(train_ids, seed=42)
        metadata = load_checkpoint(item.path, agent=agent, schedule=schedule)
        if metadata.round_index != item.round_index:
            raise ValueError("checkpoint metadata round differs from filename")
        if metadata.episode_position != 0:
            raise ValueError("selection requires a completed-round checkpoint")
        result = evaluate_formal_policy(
            episodes=validation,
            policy=GreedyDqnPolicy(
                agent,
                policy_id=f"greedy_dqn_round_{item.round_index:03d}",
            ),
        )
        candidate = ValidationCandidate(
            round_index=item.round_index,
            checkpoint_sha256=item.checkpoint_sha256,
            completed_episodes=result.completed_episodes,
            failed_episodes=result.failed_episodes,
            failure_penalty_score=result.failure_penalty_score,
            raw_economic_cost_cny=result.raw_economic_cost_cny,
            learning_reward=result.learning_reward,
        )
        candidates.append(candidate)
        print(
            f"validation_candidate round={item.round_index} "
            f"completed_episodes={candidate.completed_episodes} "
            f"failed_episodes={candidate.failed_episodes} "
            f"failure_penalty_score={candidate.failure_penalty_score:.9f} "
            f"raw_economic_cost_cny={candidate.raw_economic_cost_cny:.9f} "
            f"learning_reward={candidate.learning_reward:.9f} "
            f"elapsed_s={time.perf_counter() - started:.1f}",
            flush=True,
        )

    if dataset.opened_test_payloads != 0:
        raise RuntimeError("Test payload was opened during Validation selection")
    manifest = make_selection_manifest(
        input_hashes,
        tuple(candidates),
        required_rounds=required_rounds,
    )
    selected = by_round[manifest.selected_round]
    write_selection_outputs(output, manifest, selected.path)
    print(
        f"selection_complete experiment={profile.experiment_id} "
        f"selected_round={manifest.selected_round} "
        f"completed_episodes={next(value.completed_episodes for value in candidates if value.round_index == manifest.selected_round)} "
        f"failed_episodes={next(value.failed_episodes for value in candidates if value.round_index == manifest.selected_round)} "
        f"result_digest={manifest.result_digest} "
        f"best_sha256={manifest.copied_best_sha256} "
        f"best_path={output / 'best_validation.pt'} "
        f"test_payloads_opened={dataset.opened_test_payloads} "
        f"elapsed_s={time.perf_counter() - started:.1f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
