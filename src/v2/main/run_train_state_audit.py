"""Run the reproducible Train-only v2 DQN state audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Sequence

from ..analysis.train_state_audit import (
    build_train_feature_rows,
    load_train_segments,
    write_audit_artifacts,
)
from ..data.segment_power_source import _load_parent
from ..data.train_supervisory_audit import (
    ParentSupervisoryState,
    build_parent_supervisory_states,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_ROOT = (
    PROJECT_ROOT / "data" / "processed" / "operating_dataset_zero_boundary_v2"
)
DEFAULT_RAW_ROOT = Path.home() / "OneDrive" / "Desktop" / "氢舟一号"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "v2_dqn_state_audit"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "docs" / "v2_dqn_state_audit.md"


StateLoader = Callable[[str], Sequence[ParentSupervisoryState]]


def _raw_state_loader(raw_root: Path) -> StateLoader:
    root = Path(raw_root).resolve()

    def load(parent: str) -> tuple[ParentSupervisoryState, ...]:
        loaded = _load_parent(root, parent)
        result = build_parent_supervisory_states(loaded.channels)
        return result.states

    return load


def run_train_state_audit(
    *,
    dataset_root: str | Path,
    raw_root: str | Path,
    output_root: str | Path,
    report_path: str | Path,
    state_loader: StateLoader | None = None,
) -> dict[str, object]:
    """Build evidence from the authenticated active Train split only."""

    dataset = Path(dataset_root).resolve()
    raw = Path(raw_root).resolve()
    segments = load_train_segments(dataset)
    loader = state_loader if state_loader is not None else _raw_state_loader(raw)
    rows = build_train_feature_rows(dataset, segments, loader)
    return write_audit_artifacts(
        rows=rows,
        segments=segments,
        output_root=output_root,
        report_path=report_path,
        dataset_root=dataset,
        raw_root=raw,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the Train-only v2 DQN state audit evidence bundle."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    manifest = run_train_state_audit(
        dataset_root=arguments.dataset_root,
        raw_root=arguments.raw_root,
        output_root=arguments.output_root,
        report_path=arguments.report_path,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
