"""Build the authenticated AIS-speed sidecar for the frozen v2 dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..data.ais_speed_sidecar import build_ais_speed_sidecar


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()
    summary = build_ais_speed_sidecar(
        arguments.dataset_root,
        arguments.raw_root,
        arguments.output_root,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
