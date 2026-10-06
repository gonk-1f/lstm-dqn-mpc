"""Independent Train+Validation entrypoint for the v2 M=10 ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from ..dqn.state import FORMAL_STATE_DIMENSION
from ..preflight import require_formal_training_ready
from ..training.experiments import m10_ablation_profile
from ..training.reward_scaling import load_reward_scale_document
from .run_m10_reward_scale_calibration import (
    DEFAULT_OUTPUT as DEFAULT_M10_REWARD_SCALE,
    M10_TIMESCALE,
)
from .run_reward_scale_calibration import _manifest_hashes
from .train_formal_dqn import (
    DEFAULT_AIS_ROOT,
    DEFAULT_MODE_ROOT,
    DEFAULT_POWER_ROOT,
    REPOSITORY_ROOT,
    _positive_int,
)
from .train_formal_dqn import _live_preflight, _smoke, _train
from .train_history_dqn_study import _guard_output_directory


DEFAULT_M10_STUDY_ROOT = REPOSITORY_ROOT / "outputs" / "v2_m10_dqn_study" / "M10"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="v2 M=10 DQN action-hold ablation")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--preflight-only", action="store_true")
    modes.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--power-root", type=Path, default=DEFAULT_POWER_ROOT)
    parser.add_argument("--ais-root", type=Path, default=DEFAULT_AIS_ROOT)
    parser.add_argument("--mode-root", type=Path, default=DEFAULT_MODE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_M10_STUDY_ROOT)
    parser.add_argument("--reward-scale", type=Path, default=DEFAULT_M10_REWARD_SCALE)
    parser.add_argument("--rounds", type=_positive_int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log-every", type=_positive_int, default=50)
    parser.add_argument("--resume", type=Path)
    return parser


def _load_profile(args: argparse.Namespace):
    scale_path = Path(args.reward_scale)
    if not scale_path.is_file():
        raise FileNotFoundError(scale_path)
    document = json.loads(scale_path.read_text(encoding="utf-8"))
    calibration = load_reward_scale_document(
        document,
        expected_manifest_hashes=_manifest_hashes(
            args.power_root,
            args.ais_root,
            args.mode_root,
        ),
        timescale=M10_TIMESCALE,
    )
    return m10_ablation_profile(calibration), calibration


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.resume is not None and (args.preflight_only or args.smoke_only):
        parser.error("--resume is available only in normal training mode")
    profile, calibration = _load_profile(args)
    report, dataset, train, validation = _live_preflight(
        args,
        timescale=M10_TIMESCALE,
    )
    config = profile.dqn_config(rounds=args.rounds)
    print(
        f"M10_STUDY experiment={profile.experiment_id} "
        f"state_dimension={FORMAL_STATE_DIMENSION} action_dimension={config.action_dim} "
        f"ts_mpc_seconds={M10_TIMESCALE.ts_mpc_seconds:g} "
        f"n_mpc={M10_TIMESCALE.n_mpc} "
        f"dqn_switch_steps={M10_TIMESCALE.dqn_switch_steps} "
        f"switch_seconds={M10_TIMESCALE.switch_seconds:g} gamma={config.gamma:g} "
        f"learning_rate={config.learning_rate:.9g} warmup_steps={config.warmup_steps} "
        f"epsilon_decay_steps={config.epsilon_decay_steps} "
        f"replay_capacity={config.replay_capacity} target_sync_steps={config.target_sync_steps} "
        f"rounds={config.rounds} test_payloads_opened={dataset.opened_test_payloads}",
        flush=True,
    )
    if args.preflight_only:
        return 0 if report.ready else 2
    if args.smoke_only:
        _smoke(train, timescale=M10_TIMESCALE)
        return 0
    _guard_output_directory(args.output_dir, args.resume)
    require_formal_training_ready()
    _train(
        args,
        train,
        validation,
        profile=profile,
        calibration=calibration,
        timescale=M10_TIMESCALE,
    )
    if dataset.opened_test_payloads != 0:
        raise RuntimeError("Test payload was opened by the M=10 study")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
