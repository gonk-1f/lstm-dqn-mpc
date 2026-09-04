from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "src" / "main"
if str(MAIN) not in sys.path:
    sys.path.insert(0, str(MAIN))


from zero_residual_numerical_negative_loads import zero_residual_numerical_negatives  # noqa: E402


class TestZeroResidualNumericalNegativeLoads(unittest.TestCase):
    def _dataset(self, values: list[float]) -> Path:
        temporary = Path(tempfile.mkdtemp())
        segment_dir = temporary / "operating_segments_1s"
        segment_dir.mkdir()
        pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=len(values), freq="s"), "load_total_kw": values}).to_csv(segment_dir / "segment.csv", index=False)
        (temporary / "qa_summary.json").write_text(json.dumps({"pchip_1s": {"min_kw": min(values), "zero_count": values.count(0.0)}}), encoding="utf-8")
        return temporary

    def test_zeroes_only_residual_negative_values_and_records_qa(self) -> None:
        root = self._dataset([-0.5, 0.0, 2.0])

        report = zero_residual_numerical_negatives(root)

        values = pd.read_csv(root / "operating_segments_1s" / "segment.csv")["load_total_kw"].tolist()
        qa = json.loads((root / "qa_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(values, [0.0, 0.0, 2.0])
        self.assertEqual(report["numerical_zero_clipped_points"], 1)
        self.assertEqual(qa["numerical_zero_clipping"]["before_min_load_kw"], -0.5)
        self.assertEqual(qa["numerical_zero_clipping"]["after_min_load_kw"], 0.0)
        self.assertEqual(qa["pchip_1s"]["negative_count"], 0)

    def test_refuses_non_tolerance_negative_value_without_writing(self) -> None:
        root = self._dataset([-1.0, 0.0])

        with self.assertRaisesRegex(ValueError, "outside the approved"):
            zero_residual_numerical_negatives(root)

        values = pd.read_csv(root / "operating_segments_1s" / "segment.csv")["load_total_kw"].tolist()
        self.assertEqual(values, [-1.0, 0.0])
