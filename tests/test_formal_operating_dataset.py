from __future__ import annotations

import sys
import unittest
import inspect
import tempfile
from pathlib import Path
import pandas as pd


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
    def make_fixture(self, root: Path):
        """Exercise test-loader plumbing without opening the held-out dataset."""
        rows = []
        (root / 'operating_segments_1s').mkdir()
        for name in ('train', 'validation', 'test'):
            relative = f'operating_segments_1s/fixture_{name}.csv'
            pd.DataFrame(dict(timestamp=pd.date_range('2024-01-01', periods=3, freq='s'),
                time_s=[0., 1., 2.], load_total_kw=[0., 220., 221.])).to_csv(root / relative, index=False)
            rows.append(dict(parent_voyage=f'parent_{name}', segment_id=f'fixture_{name}',
                one_second_csv=relative, split=name, raw_csv=relative))
        pd.DataFrame(rows).to_csv(root / 'split_manifest.csv', index=False)
        return load_formal_operating_split(root)

    def test_frozen_manifest_totals_and_parent_groups_are_unchanged(self) -> None:
        # Manifest metadata only. A unit-test run must never open held-out loads.
        frame = load_formal_operating_split().manifest
        self.assertEqual(frame.parent_voyage.nunique(), 66)
        self.assertEqual(len(frame), 177)
        self.assertEqual(int(frame.num_1s_points.sum()), 1_114_037)
        self.assertEqual(frame.groupby('split').num_1s_points.sum().to_dict(),
                         {'train': 796249, 'validation': 248867, 'test': 68921})
        self.assertEqual(int((frame.groupby('parent_voyage').split.nunique() > 1).sum()), 0)

    def test_dataset_auditor_checks_complete_disjoint_synthetic_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.make_fixture(Path(directory))
            audit = audit_formal_operating_dataset(Path(directory))
        self.assertEqual(audit.parent_voyage_count, 3)
        self.assertEqual(audit.segment_count, 3)
        self.assertEqual(audit.point_count, 9)
        self.assertEqual(
            audit.split_point_counts,
            {"train": 3, "validation": 3, "test": 3},
        )
        self.assertEqual(audit.negative_load_point_count, 0)
        self.assertEqual(audit.orphan_segment_paths, ())
        self.assertEqual(audit.missing_segment_paths, ())
        self.assertEqual(audit.parent_split_leakage, ())

    def test_dqn_and_fixed_a0_load_the_same_ordered_test_segments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            split = self.make_fixture(Path(directory))
            dqn_loads = dqn_test.load_test_voyage(split.test_segments[0], split=split)
            a0_loads = a0_test.load_test_voyage(split.test_segments[0], split=split)

        self.assertIn("initial_soc=training.SOC_REFERENCE", inspect.getsource(dqn_test.run_test_episode))
        self.assertIn("initial_soc=0.55", inspect.getsource(a0_test.run_test_episode))
        self.assertEqual(dqn_loads.tolist(), a0_loads.tolist())


if __name__ == "__main__":
    unittest.main()
