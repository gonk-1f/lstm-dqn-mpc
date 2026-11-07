"""Deterministic TD3 Validation8; evaluation never trains or opens Test payloads."""
from __future__ import annotations

from pathlib import Path
from .evaluation import DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_ROOT, argument_parser, evaluate


def run_validation(
    model_path: Path, data_root: Path | None = None, output_root: Path | None = None,
    device: str = "cpu", seed: int = 42, episodes: int = 8,
    max_steps_per_episode: int | None = None, torch_threads: int = 1,
    output_dir: Path | None = None, max_steps: int | None = None,
    reward_mode: str = "economic",
) -> dict:
    if max_steps is not None and max_steps_per_episode is not None:
        raise ValueError("Specify only one max_steps argument")
    return evaluate(model_path, split="validation", data_root=data_root, output_root=output_root,
                    output_dir=output_dir, device=device, seed=seed, episodes=episodes,
                    max_steps=max_steps if max_steps is not None else max_steps_per_episode,
                    torch_threads=torch_threads, reward_mode=reward_mode)


def main(argv=None):
    parser = argument_parser(__doc__)
    parser.add_argument("--episodes", type=int, default=8, help="1..8 fixed-order voyages; default full Validation8")
    return run_validation(**vars(parser.parse_args(argv)))


if __name__ == "__main__":
    main()
