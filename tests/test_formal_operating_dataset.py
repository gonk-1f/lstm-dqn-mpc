from __future__ import annotations

import sys
import unittest
import inspect
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
MAIN = SRC / "main"

for path in (SRC, MAIN):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


from utils.formal_operating_dataset import (  # noqa: E402
    audit_formal_operating_dataset,
    load_formal_operating_split,
)
import test_dqn_mpc_causal as dqn_test  # noqa: E402
import test_mpc_nominal_causal as a0_test  # noqa: E402


class FormalOperatingDatasetTests(unittest.TestCase):
    def test_frozen_rebuilt_dataset_is_complete_and_disjoint(self) -> None:
        audit = audit_formal_operating_dataset()

        self.assertEqual(audit.parent_voyage_count, 66)
        self.assertEqual(audit.segment_count, 177)
        self.assertEqual(audit.point_count, 1_114_037)
        self.assertEqual(
            audit.split_point_counts,
            {"train": 796_249, "validation": 248_867, "test": 68_921},
        )
        self.assertEqual(audit.negative_load_point_count, 0)
        self.assertEqual(audit.orphan_segment_paths, ())
        self.assertEqual(audit.missing_segment_paths, ())
        self.assertEqual(audit.parent_split_leakage, ())

    def test_dqn_and_fixed_a0_load_the_same_ordered_test_segments(self) -> None:
        split = load_formal_operating_split()

        dqn_loads = dqn_test.load_test_voyage(
            split.test_segments[0], split=split
        )
        a0_loads = a0_test.load_test_voyage(
            split.test_segments[0], split=split
        )

        self.assertIn("initial_soc=0.55", inspect.getsource(dqn_test.run_test_episode))
        self.assertIn("initial_soc=0.55", inspect.getsource(a0_test.run_test_episode))
        self.assertEqual(dqn_loads.tolist(), a0_loads.tolist())


if __name__ == "__main__":
    unittest.main()
