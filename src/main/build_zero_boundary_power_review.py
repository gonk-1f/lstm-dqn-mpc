"""Build the final zero-boundary v2 total-power review."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from v2.data.zero_boundary_power_review import build_review  # noqa: E402


DEFAULT_DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "operating_dataset_zero_boundary_v2"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "v2_zero_boundary_power_review"


def validated_output_path(project_root: Path, output_root: Path) -> Path:
    """Resolve an output path and require a child of repository outputs."""
    root = Path(project_root).resolve()
    outputs = (root / "outputs").resolve()
    target = Path(output_root).resolve()
    try:
        relative = target.relative_to(outputs)
    except ValueError as exc:
        raise ValueError(
            "review output must stay under repository outputs"
        ) from exc
    if not relative.parts:
        raise ValueError("repository outputs itself cannot be replaced")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build final zero-boundary total-power review"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()

    target = validated_output_path(PROJECT_ROOT, args.output_root)
    if target.exists() and args.replace:
        shutil.rmtree(target)
    rows = build_review(args.dataset_root, target)
    print(f"generated {len(rows)} segment figures at {target}")


if __name__ == "__main__":
    main()
