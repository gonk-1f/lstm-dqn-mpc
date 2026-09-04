from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "src" / "main"
if str(MAIN) not in sys.path:
    sys.path.insert(0, str(MAIN))

from refine_operating_segment_split import validate_rebuilt_split  # noqa: E402


class TestOperatingSegmentRepartition(unittest.TestCase):
    def test_validator_accepts_parent_safe_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pd.DataFrame(
                {
                    "parent_voyage": ["voyage_001", "voyage_001", "voyage_002"],
                    "segment_id": ["a", "b", "c"],
                    "split": ["train", "train", "validation"],
                }
            ).to_csv(root / "split_manifest.csv", index=False)
            report = validate_rebuilt_split(root)
        self.assertEqual(report["parent_voyage_overlap_count"], 0)
        self.assertEqual(report["parent_voyage_counts"], {"train": 1, "validation": 1})
