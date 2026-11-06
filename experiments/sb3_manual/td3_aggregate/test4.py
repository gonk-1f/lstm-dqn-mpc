"""Final deterministic TD3 evaluation on the independent frozen Test4 voyages."""
from __future__ import annotations

from pathlib import Path
from .evaluation import DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_ROOT, argument_parser, evaluate


def run_test4(
    model_path: Path, data_root: Path | None = None, output_root: Path | None = None,
    device: str = "cpu", seed: int = 42, test_source_root: Path | None = None,
    torch_threads: int = 1, output_dir: Path | None = None, max_steps: int | None = None,
) -> dict:
    return evaluate(model_path, split="test", data_root=data_root, output_root=output_root,
                    output_dir=output_dir, device=device, seed=seed, max_steps=max_steps,
                    torch_threads=torch_threads, test_source_root=test_source_root)


def main(argv=None):
    parser = argument_parser(__doc__)
    parser.add_argument("--test-source-root", type=Path)
    return run_test4(**vars(parser.parse_args(argv)))


if __name__ == "__main__":
    main()
