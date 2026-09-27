"""Independent H1-H4 pilot entrypoint for the 90-value history DQN."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from ..dqn.state import FORMAL_STATE_DIMENSION
from ..preflight import require_formal_training_ready
from ..training.experiments import (
    DqnExperimentProfile,
    history_study_profile,
)
from ..training.reward_scaling import load_reward_scale_document
from .run_reward_scale_calibration import DEFAULT_OUTPUT as DEFAULT_SCALE_PATH
from .run_reward_scale_calibration import _manifest_hashes
from .train_formal_dqn import (
    DEFAULT_AIS_ROOT,
    DEFAULT_MODE_ROOT,
    DEFAULT_POWER_ROOT,
    REPOSITORY_ROOT,
    _live_preflight,
    _positive_int,
    _smoke,
    _train,
)


DEFAULT_STUDY_ROOT = REPOSITORY_ROOT / "outputs" / "v2_history_dqn_study"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="v2 90-state DQN history pilot study")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--preflight-only", action="store_true")
    modes.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--experiment", choices=("H1", "H2", "H3", "H4"), default="H1")
    parser.add_argument("--power-root", type=Path, default=DEFAULT_POWER_ROOT)
    parser.add_argument("--ais-root", type=Path, default=DEFAULT_AIS_ROOT)
    parser.add_argument("--mode-root", type=Path, default=DEFAULT_MODE_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--reward-scale", type=Path)
    parser.add_argument("--rounds", type=_positive_int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log-every", type=_positive_int, default=1)
    parser.add_argument("--resume", type=Path)
    return parser


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    return (
        Path(args.output_dir)
        if args.output_dir is not None
        else DEFAULT_STUDY_ROOT / args.experiment
    )


def _guard_output_directory(output_dir: Path, resume: Path | None) -> None:
    destination = Path(output_dir).resolve()
    if resume is not None:
        expected = (destination / "latest.pt").resolve()
        if Path(resume).resolve() != expected or not expected.is_file():
            raise ValueError("resume must be this experiment directory's exact latest.pt")
        return
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"study output directory is not empty: {destination}")


def _load_profile(
    args: argparse.Namespace,
) -> tuple[DqnExperimentProfile, object | None]:
    if args.experiment == "H1":
        if args.reward_scale is not None:
            raise ValueError("H1 raw reward forbids --reward-scale")
        return history_study_profile("H1", None), None
    scale_path = Path(args.reward_scale or DEFAULT_SCALE_PATH)
    if not scale_path.is_file():
        raise FileNotFoundError(scale_path)
    document = json.loads(scale_path.read_text(encoding="utf-8"))
    expected_hashes = None
    if all(hasattr(args, name) for name in ("power_root", "ais_root", "mode_root")):
        expected_hashes = _manifest_hashes(
            args.power_root,
            args.ais_root,
            args.mode_root,
        )
    calibration = load_reward_scale_document(
        document,
        expected_manifest_hashes=expected_hashes,
    )
    return history_study_profile(args.experiment, calibration), calibration


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.resume is not None and (args.preflight_only or args.smoke_only):
        parser.error("--resume is available only in normal training mode")
    args.output_dir = _resolve_output_dir(args)
    profile, calibration = _load_profile(args)
    report, dataset, train, validation = _live_preflight(args)
    print(
        f"HISTORY_STUDY experiment={profile.experiment_id} "
        f"state_dimension={FORMAL_STATE_DIMENSION} reward_mode={profile.reward_mode} "
        f"learning_rate={profile.learning_rate:.9g} rounds={args.rounds}",
        flush=True,
    )
    if args.preflight_only:
        return 0 if report.ready else 2
    if args.smoke_only:
        _smoke(train)
        return 0
    _guard_output_directory(args.output_dir, args.resume)
    require_formal_training_ready()
    _train(
        args,
        train,
        validation,
        profile=profile,
        calibration=calibration,
    )
    if dataset.opened_test_payloads != 0:
        raise RuntimeError("Test payload was opened by history study")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
