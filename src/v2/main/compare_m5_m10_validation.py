"""Validation-only M=5 versus M=10 ablation comparison; never opens Test."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

from ..config import TimeScaleConfig
from ..data.formal_training_dataset import FormalTrainingDataset
from ..evaluation.formal_policy import (
    FixedActionPolicy,
    GreedyDqnPolicy,
    PolicyEvaluation,
    evaluate_formal_policy,
)
from ..training.checkpoint import load_checkpoint
from ..training.dqn import DqnAgent
from ..training.experiments import history_study_profile, m10_ablation_profile
from ..training.reward_scaling import load_reward_scale_document
from ..training.schedule import EpisodeShuffleSchedule
from .run_m10_reward_scale_calibration import (
    DEFAULT_OUTPUT as DEFAULT_M10_REWARD_SCALE,
    M10_TIMESCALE,
)
from .run_reward_scale_calibration import (
    DEFAULT_OUTPUT as DEFAULT_M5_REWARD_SCALE,
    _manifest_hashes,
    _write_atomic,
)
from .train_formal_dqn import (
    DEFAULT_AIS_ROOT,
    DEFAULT_MODE_ROOT,
    DEFAULT_POWER_ROOT,
    REPOSITORY_ROOT,
    _positive_int,
)


M5_TIMESCALE = TimeScaleConfig.formal_baseline()
DEFAULT_M5_CHECKPOINT_DIR = (
    REPOSITORY_ROOT / "outputs" / "v2_m10_dqn_study" / "M5_control"
)
DEFAULT_M10_CHECKPOINT_DIR = (
    REPOSITORY_ROOT / "outputs" / "v2_m10_dqn_study" / "M10"
)
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "outputs"
    / "v2_m10_dqn_study"
    / "validation_comparison.json"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validation-only M=5 versus M=10 DQN comparison"
    )
    parser.add_argument("--power-root", type=Path, default=DEFAULT_POWER_ROOT)
    parser.add_argument("--ais-root", type=Path, default=DEFAULT_AIS_ROOT)
    parser.add_argument("--mode-root", type=Path, default=DEFAULT_MODE_ROOT)
    parser.add_argument("--m5-checkpoint", type=Path)
    parser.add_argument("--m10-checkpoint", type=Path)
    parser.add_argument("--m5-checkpoint-dir", type=Path, default=DEFAULT_M5_CHECKPOINT_DIR)
    parser.add_argument("--m10-checkpoint-dir", type=Path, default=DEFAULT_M10_CHECKPOINT_DIR)
    parser.add_argument("--m5-reward-scale", type=Path, default=DEFAULT_M5_REWARD_SCALE)
    parser.add_argument("--m10-reward-scale", type=Path, default=DEFAULT_M10_REWARD_SCALE)
    parser.add_argument("--m5-rounds", type=_positive_int, default=40)
    parser.add_argument("--m10-rounds", type=_positive_int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def _action_diagnostics(
    action_counts: tuple[tuple[str, int], ...],
) -> dict[str, object]:
    if type(action_counts) is not tuple or not action_counts:
        raise ValueError("action_counts must be a nonempty exact tuple")
    distribution = {action_id: count for action_id, count in action_counts}
    if len(distribution) != len(action_counts) or any(
        type(action_id) is not str
        or not action_id
        or type(count) is not int
        or count <= 0
        for action_id, count in action_counts
    ):
        raise ValueError("action_counts contains invalid or duplicate entries")
    total = sum(distribution.values())
    shares = tuple(count / total for count in distribution.values())
    return {
        "transition_count": total,
        "unique_action_count": len(distribution),
        "max_action_share": max(shares),
        "shannon_entropy": -math.fsum(
            share * math.log(share) for share in shares
        ),
        "action_distribution": distribution,
    }


def _evaluation_row(
    result: PolicyEvaluation,
    *,
    method: str,
    timescale: TimeScaleConfig,
) -> dict[str, object]:
    diagnostics = _action_diagnostics(result.action_counts)
    return {
        "method": method,
        "dqn_switch_steps": timescale.dqn_switch_steps,
        "switch_seconds": timescale.switch_seconds,
        "completed_episodes": result.completed_episodes,
        "failed_episodes": result.failed_episodes,
        "h2_cost_cny": result.h2_cost_cny,
        "fc_degradation_cost_cny": result.fc_degradation_cost_cny,
        "battery_degradation_cost_cny": result.battery_degradation_cost_cny,
        "shore_cost_cny": result.shore_cost_cny,
        "raw_economic_cost_cny": result.raw_economic_cost_cny,
        "failure_penalty_score": result.failure_penalty_score,
        **diagnostics,
    }


def _load_agent(
    *,
    checkpoint: Path,
    profile,
    rounds: int,
    train_ids: tuple[str, ...],
    seed: int,
    device: str,
    timescale: TimeScaleConfig,
) -> tuple[DqnAgent, object]:
    agent = DqnAgent(profile.dqn_config(rounds=rounds), seed=seed, device=device)
    schedule = EpisodeShuffleSchedule(train_ids, seed=seed)
    metadata = load_checkpoint(
        checkpoint,
        agent=agent,
        schedule=schedule,
        timescale=timescale,
    )
    if metadata.episode_position != 0:
        raise ValueError("comparison checkpoint must be a completed-round checkpoint")
    return agent, metadata


def _select_dqn_validation_row(
    *,
    method: str,
    checkpoint: Path | None,
    checkpoint_dir: Path,
    profile,
    rounds: int,
    train_ids: tuple[str, ...],
    validation,
    seed: int,
    device: str,
    timescale: TimeScaleConfig,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    paths = (
        (Path(checkpoint),)
        if checkpoint is not None
        else tuple(Path(checkpoint_dir) / f"round_{index:03d}.pt" for index in range(1, rounds + 1))
    )
    candidates: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        agent, metadata = _load_agent(
            checkpoint=path,
            profile=profile,
            rounds=rounds,
            train_ids=train_ids,
            seed=seed,
            device=device,
            timescale=timescale,
        )
        round_index = metadata.round_index
        if checkpoint is None and path.name != f"round_{round_index:03d}.pt":
            raise ValueError("checkpoint filename and completed round differ")
        result = evaluate_formal_policy(
            episodes=validation,
            policy=GreedyDqnPolicy(
                agent,
                policy_id=f"{method}_round_{round_index:03d}",
            ),
            timescale=timescale,
        )
        row = _evaluation_row(result, method=method, timescale=timescale)
        row["selected_round"] = round_index
        row["checkpoint"] = str(path.resolve())
        rows.append(row)
        candidate = {
            "round": round_index,
            "completed_episodes": row["completed_episodes"],
            "failed_episodes": row["failed_episodes"],
            "failure_penalty_score": row["failure_penalty_score"],
            "raw_economic_cost_cny": row["raw_economic_cost_cny"],
            "unique_action_count": row["unique_action_count"],
            "shannon_entropy": row["shannon_entropy"],
            "max_action_share": row["max_action_share"],
            "checkpoint": row["checkpoint"],
        }
        candidates.append(candidate)
        print(
            f"validation_candidate method={method} round={round_index}/{rounds} "
            f"raw_economic_cost_cny={row['raw_economic_cost_cny']:.9f} "
            f"completed_episodes={row['completed_episodes']} "
            f"failed_episodes={row['failed_episodes']} "
            f"unique_actions={row['unique_action_count']} "
            f"action_entropy={row['shannon_entropy']:.9g} "
            f"max_action_share={row['max_action_share']:.9g}",
            flush=True,
        )
    selected = min(
        rows,
        key=lambda row: (
            -int(row["completed_episodes"]),
            float(row["failure_penalty_score"]),
            float(row["raw_economic_cost_cny"]),
            int(row["selected_round"]),
        ),
    )
    return selected, candidates


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    hashes = _manifest_hashes(args.power_root, args.ais_root, args.mode_root)
    m5_calibration = load_reward_scale_document(
        json.loads(Path(args.m5_reward_scale).read_text(encoding="utf-8")),
        expected_manifest_hashes=hashes,
        timescale=M5_TIMESCALE,
    )
    m10_calibration = load_reward_scale_document(
        json.loads(Path(args.m10_reward_scale).read_text(encoding="utf-8")),
        expected_manifest_hashes=hashes,
        timescale=M10_TIMESCALE,
    )
    m5_profile = history_study_profile("H4", m5_calibration)
    m10_profile = m10_ablation_profile(m10_calibration)

    dataset = FormalTrainingDataset.open(
        args.power_root,
        args.ais_root,
        args.mode_root,
    )
    train_ids = dataset.split_episode_ids("train")
    validation = dataset.load_validation()
    if dataset.opened_test_payloads != 0:
        raise RuntimeError("Test payload was opened before Validation comparison")

    m5_row, m5_candidates = _select_dqn_validation_row(
        method="DQN_M5",
        checkpoint=args.m5_checkpoint,
        checkpoint_dir=args.m5_checkpoint_dir,
        profile=m5_profile,
        rounds=args.m5_rounds,
        train_ids=train_ids,
        validation=validation,
        seed=args.seed,
        device=args.device,
        timescale=M5_TIMESCALE,
    )
    m10_row, m10_candidates = _select_dqn_validation_row(
        method="DQN_M10",
        checkpoint=args.m10_checkpoint,
        checkpoint_dir=args.m10_checkpoint_dir,
        profile=m10_profile,
        rounds=args.m10_rounds,
        train_ids=train_ids,
        validation=validation,
        seed=args.seed,
        device=args.device,
        timescale=M10_TIMESCALE,
    )
    runs = (
        ("FIXED_M5", M5_TIMESCALE, FixedActionPolicy("w_8_1_1")),
        ("FIXED_M10", M10_TIMESCALE, FixedActionPolicy("w_8_1_1")),
    )
    rows: list[dict[str, object]] = [m5_row, m10_row]
    for method, timescale, policy in runs:
        result = evaluate_formal_policy(
            episodes=validation,
            policy=policy,
            timescale=timescale,
        )
        row = _evaluation_row(result, method=method, timescale=timescale)
        rows.append(row)
        print(
            f"validation_compare method={method} M={timescale.dqn_switch_steps} "
            f"raw_economic_cost_cny={row['raw_economic_cost_cny']:.9f} "
            f"completed_episodes={row['completed_episodes']} "
            f"failed_episodes={row['failed_episodes']} "
            f"unique_actions={row['unique_action_count']} "
            f"action_entropy={row['shannon_entropy']:.9g} "
            f"max_action_share={row['max_action_share']:.9g}",
            flush=True,
        )
    by_method = {row["method"]: row for row in rows}
    fixed_cost = float(by_method["FIXED_M5"]["raw_economic_cost_cny"])
    fixed_m10_cost = float(by_method["FIXED_M10"]["raw_economic_cost_cny"])
    if not math.isclose(fixed_cost, fixed_m10_cost, rel_tol=0.0, abs_tol=1.0e-9):
        raise RuntimeError("fixed-policy cost changed with action grouping")
    payload: dict[str, object] = {
        "scope": "validation_only",
        "test_payloads_opened": dataset.opened_test_payloads,
        "rows": rows,
        "validation_candidates": {
            "M5": m5_candidates,
            "M10": m10_candidates,
        },
        "fixed_baseline_cost_cny": fixed_cost,
        "dqn_advantage_cny": {
            "M5": fixed_cost - float(by_method["DQN_M5"]["raw_economic_cost_cny"]),
            "M10": fixed_cost - float(by_method["DQN_M10"]["raw_economic_cost_cny"]),
        },
    }
    if dataset.opened_test_payloads != 0:
        raise RuntimeError("Test payload was opened during Validation comparison")
    _write_atomic(args.output, payload)
    print(
        f"VALIDATION_COMPARISON=PASS output={args.output} "
        f"test_payloads_opened={dataset.opened_test_payloads}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
