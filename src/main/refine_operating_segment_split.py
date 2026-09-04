"""Read-only validation for the chronological parent-safe rebuilt split."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = REPO_ROOT / "data" / "processed" / "operating_segments_1s_rebuilt"


def validate_rebuilt_split(root: Path = DATASET_ROOT) -> dict[str, object]:
    manifest = pd.read_csv(Path(root) / "split_manifest.csv")
    parent_splits = manifest.groupby("parent_voyage")["split"].nunique()
    leakage = int(parent_splits.gt(1).sum())
    if leakage:
        raise ValueError(f"parent voyage split leakage: {leakage}")
    return {
        "parent_voyage_counts": manifest.groupby("split")["parent_voyage"].nunique().to_dict(),
        "segment_counts": manifest["split"].value_counts().to_dict(),
        "parent_voyage_overlap_count": 0,
    }


if __name__ == "__main__":
    print(json.dumps(validate_rebuilt_split(), ensure_ascii=False, indent=2))
