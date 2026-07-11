"""DEPRECATED: dataset builder for invalid linear 1 s reconstruction.

DEPRECATED: this script consumes 30 s to 1 s linear-interpolation outputs.
The resulting reconstructed data are non-causal and must not be used as valid
online high-frequency forecasting evidence.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
PROJ = Path(__file__).resolve().parents[2]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from build_total_load_dataset_721 import build_dataset  # noqa: E402


DEFAULT_INPUT_DIR = PROJ / "total_load_excels_1s"
DEFAULT_OUTPUT_DIR = PROJ / "outputs" / "total_load_dataset_1s_build"
DEFAULT_CONFIG_DIR = PROJ / "outputs" / "config"


def main() -> None:
    result = build_dataset(
        input_dir=DEFAULT_INPUT_DIR,
        output_dir=DEFAULT_OUTPUT_DIR,
        config_dir=DEFAULT_CONFIG_DIR,
        expected_count=66,
        split_counts=(46, 13, 7),
        include_existing_speed=True,
        sample_interval_seconds=1.0,
        split_json_name="voyage_split_total_load_1s_721.json",
        split_txt_name="SPLIT_TOTAL_LOAD_1S_721.txt",
    )
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
    print()


if __name__ == "__main__":
    main()
