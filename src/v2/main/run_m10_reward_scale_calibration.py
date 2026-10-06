"""Generate an isolated Train-only reward-scale document for M=10."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from ..config import TimeScaleConfig
from .run_reward_scale_calibration import (
    DEFAULT_AIS_ROOT,
    DEFAULT_MODE_ROOT,
    DEFAULT_POWER_ROOT,
    REPOSITORY_ROOT,
    _write_atomic,
    generate_calibration,
)


M10_TIMESCALE = TimeScaleConfig(30.0, 5, 10)
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "outputs"
    / "v2_m10_dqn_study"
    / "reward_scale_calibration.json"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train-only v2 M=10 reward-scale calibration"
    )
    parser.add_argument("--power-root", type=Path, default=DEFAULT_POWER_ROOT)
    parser.add_argument("--ais-root", type=Path, default=DEFAULT_AIS_ROOT)
    parser.add_argument("--mode-root", type=Path, default=DEFAULT_MODE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    document = generate_calibration(args, timescale=M10_TIMESCALE)
    _write_atomic(args.output, document)
    print(
        f"M10_REWARD_SCALE_CALIBRATION=PASS output={args.output} "
        f"sample_count={document['sample_count']} "
        f"scale_cny={document['scale_cny']:.12g} "
        f"dqn_switch_steps={M10_TIMESCALE.dqn_switch_steps} "
        f"switch_seconds={M10_TIMESCALE.switch_seconds:g} "
        f"test_payloads_opened={document['test_payloads_opened']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
