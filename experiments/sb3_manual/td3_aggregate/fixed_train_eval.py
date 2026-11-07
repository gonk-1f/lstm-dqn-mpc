"""Deterministic TD3 diagnostic on the fixed 9258-second Train voyage."""
from __future__ import annotations

from pathlib import Path
from .evaluation import DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_ROOT, FIXED_TRAIN_ID, argument_parser, evaluate


def run_fixed_train_evaluation(
    model_path: Path, data_root: Path | None = None, output_root: Path | None = None,
    episode: int = 500, sample_id: str = FIXED_TRAIN_ID, device: str = "cpu", seed: int = 42,
    max_steps: int | None = None, output_dir: Path | None = None, torch_threads: int = 1,
    reward_mode: str = "economic",
) -> dict:
    if sample_id != FIXED_TRAIN_ID:
        raise ValueError("The fixed TD3 diagnostic requires parent_005_seg01")
    return evaluate(model_path, split="fixed_train", data_root=data_root, output_root=output_root,
                    output_dir=output_dir, device=device, seed=seed, max_steps=max_steps,
                    episode=episode, torch_threads=torch_threads, reward_mode=reward_mode)


def main(argv=None):
    parser = argument_parser(__doc__)
    parser.add_argument("--episode", type=int, default=500)
    parser.add_argument("--sample-id", default=FIXED_TRAIN_ID)
    return run_fixed_train_evaluation(**vars(parser.parse_args(argv)))


if __name__ == "__main__":
    main()
