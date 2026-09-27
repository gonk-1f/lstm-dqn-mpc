"""Rank completed H1-H4 pilots using ordered Validation episodes only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Sequence

from ..data.formal_training_dataset import FormalTrainingDataset
from ..dqn.action_space import ACTION_CATALOG_DIGEST
from ..dqn.state import FORMAL_STATE_SCHEMA_DIGEST
from ..evaluation.checkpoint_selection import (
    ValidationCandidate,
    authenticate_checkpoint_candidates,
    canonical_result_digest,
)
from ..evaluation.formal_policy import GreedyDqnPolicy, evaluate_formal_policy
from ..evaluation.study_selection import (
    STUDY_PILOT_ROUNDS,
    StudyProfileRun,
    select_study_candidates,
    write_study_selection_outputs,
)
from ..training.checkpoint import load_checkpoint
from ..training.dqn import DqnAgent
from ..training.experiments import history_study_profile
from ..training.reward_scaling import load_reward_scale_document
from ..training.schedule import EpisodeShuffleSchedule
from .run_reward_scale_calibration import DEFAULT_OUTPUT as DEFAULT_SCALE_PATH
from .select_formal_dqn_checkpoint import _input_manifest_hashes, _positive_int
from .train_formal_dqn import (
    DEFAULT_AIS_ROOT,
    DEFAULT_MODE_ROOT,
    DEFAULT_POWER_ROOT,
    REPOSITORY_ROOT,
)


DEFAULT_STUDY_ROOT = REPOSITORY_ROOT / "outputs" / "v2_history_dqn_study"
DEFAULT_OUTPUT_ROOT = DEFAULT_STUDY_ROOT / "pilot_selection"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validation-only H1-H4 pilot selector")
    parser.add_argument("--power-root", type=Path, default=DEFAULT_POWER_ROOT)
    parser.add_argument("--ais-root", type=Path, default=DEFAULT_AIS_ROOT)
    parser.add_argument("--mode-root", type=Path, default=DEFAULT_MODE_ROOT)
    parser.add_argument("--study-root", type=Path, default=DEFAULT_STUDY_ROOT)
    parser.add_argument("--reward-scale", type=Path, default=DEFAULT_SCALE_PATH)
    parser.add_argument("--rounds", type=_positive_int, default=STUDY_PILOT_ROUNDS)
    parser.add_argument("--top-k", type=_positive_int, default=2)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="cpu")
    return parser


def _open_selection_data(args: argparse.Namespace):
    dataset = FormalTrainingDataset.open(args.power_root, args.ais_root, args.mode_root)
    train_ids = dataset.split_episode_ids("train")
    validation = dataset.load_validation()
    if dataset.opened_test_payloads != 0:
        raise RuntimeError("Test payload was opened during pilot selection")
    return dataset, train_ids, validation


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.rounds != STUDY_PILOT_ROUNDS:
        raise ValueError("pilot selection requires exactly ten completed rounds")
    if args.top_k != 2:
        raise ValueError("pilot continuation requires exactly top-k two profiles")
    output = Path(args.output_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)

    input_hashes = _input_manifest_hashes(args)
    scale_document = json.loads(Path(args.reward_scale).read_text(encoding="utf-8"))
    calibration = load_reward_scale_document(
        scale_document,
        expected_manifest_hashes=input_hashes,
    )
    profiles = {
        "H1": history_study_profile("H1", None),
        **{
            experiment_id: history_study_profile(experiment_id, calibration)
            for experiment_id in ("H2", "H3", "H4")
        },
    }
    dataset, train_ids, validation = _open_selection_data(args)
    dataset_identity = canonical_result_digest(
        {"input_manifest_sha256": input_hashes}
    )
    started = time.perf_counter()
    runs: list[StudyProfileRun] = []
    required_rounds = range(1, args.rounds + 1)

    for experiment_id, profile in profiles.items():
        checkpoint_dir = Path(args.study_root) / experiment_id
        expected_identity = {
            "experiment_id": profile.experiment_id,
            "reward_mode": profile.reward_mode,
            "reward_scaling_identity": profile.reward_scaling_identity,
        }
        checkpoints = authenticate_checkpoint_candidates(
            checkpoint_dir,
            required_rounds=required_rounds,
            expected_training_identity=expected_identity,
        )
        config = profile.dqn_config(rounds=args.rounds)
        values: list[ValidationCandidate] = []
        for item in checkpoints:
            agent = DqnAgent(config, seed=42, device=args.device)
            schedule = EpisodeShuffleSchedule(train_ids, seed=42)
            metadata = load_checkpoint(item.path, agent=agent, schedule=schedule)
            if metadata.round_index != item.round_index or metadata.episode_position != 0:
                raise ValueError("study checkpoint is not a completed matching round")
            result = evaluate_formal_policy(
                episodes=validation,
                policy=GreedyDqnPolicy(
                    agent,
                    policy_id=f"{experiment_id}_round_{item.round_index:03d}",
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
            values.append(candidate)
            print(
                f"study_validation experiment={experiment_id} "
                f"round={item.round_index}/{args.rounds} "
                f"completed_episodes={candidate.completed_episodes} "
                f"failed_episodes={candidate.failed_episodes} "
                f"failure_penalty_score={candidate.failure_penalty_score:.9f} "
                f"raw_economic_cost_cny={candidate.raw_economic_cost_cny:.9f} "
                f"elapsed_s={time.perf_counter() - started:.1f}",
                flush=True,
            )

        latest = checkpoint_dir / "latest.pt"
        latest_agent = DqnAgent(config, seed=42, device=args.device)
        latest_schedule = EpisodeShuffleSchedule(train_ids, seed=42)
        latest_metadata = load_checkpoint(
            latest,
            agent=latest_agent,
            schedule=latest_schedule,
        )
        if (
            latest_metadata.round_index != args.rounds
            or latest_metadata.episode_position != 0
        ):
            raise ValueError("latest.pt is not resumable from completed pilot round ten")
        runs.append(
            StudyProfileRun(
                experiment_id=experiment_id,
                reward_mode=profile.reward_mode,
                reward_scaling_identity=profile.reward_scaling_identity,
                state_schema_digest=FORMAL_STATE_SCHEMA_DIGEST,
                action_catalog_digest=ACTION_CATALOG_DIGEST,
                dataset_identity=dataset_identity,
                completed_rounds=args.rounds,
                resume_path=latest.resolve(),
                validation_candidates=tuple(values),
            )
        )

    if dataset.opened_test_payloads != 0:
        raise RuntimeError("Test payload was opened during pilot selection")
    selected = select_study_candidates(
        tuple(runs),
        required_profiles=("H1", "H2", "H3", "H4"),
        top_k=args.top_k,
    )
    write_study_selection_outputs(
        output,
        runs=tuple(runs),
        selected=selected,
        input_manifest_hashes=input_hashes,
    )
    print(
        "study_selection_complete "
        f"top_profiles={','.join(item.experiment_id for item in selected)} "
        f"output={output} test_payloads_opened={dataset.opened_test_payloads} "
        f"elapsed_s={time.perf_counter() - started:.1f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
