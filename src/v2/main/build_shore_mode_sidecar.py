"""Build the authenticated FC/BMS/AIS shore-mode sidecar."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..data.shore_mode_sidecar import build_shore_mode_sidecar


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--ais-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()
    summary = build_shore_mode_sidecar(
        arguments.dataset_root,
        arguments.ais_root,
        arguments.raw_root,
        arguments.output_root,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
