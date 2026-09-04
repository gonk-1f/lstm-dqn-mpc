from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from utils.operating_segment_dataset import pchip_to_one_second  # noqa: E402


class TestOperatingSegmentDatasetPublicAPI(unittest.TestCase):
    def test_public_pchip_api_keeps_real_negative_source_values_without_clipping(self) -> None:
        source = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=2, freq="30s"),
                "load_total_kw": [-2.0, 4.0],
            }
        )
        result, qa = pchip_to_one_second(source)
        self.assertEqual(float(result.loc[0, "load_total_kw"]), -2.0)
        self.assertEqual(qa["floating_negative_zeroed"], 0)
