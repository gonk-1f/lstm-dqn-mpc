from __future__ import annotations

import sys
import unittest
import inspect
import json
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
    load_operating_segment_loads,
)
import test_dqn_mpc_causal as dqn_test  # noqa: E402
import test_mpc_nominal_causal as a0_test  # noqa: E402
import compare_mpc_vs_dqn as comparison  # noqa: E402


class FormalOperatingDatasetTests(unittest.TestCase):
    def make_fixture(self, root: Path):
        """Exercise test-loader plumbing without opening the held-out dataset."""
        rows = []
        (root / 'metadata').mkdir()
        for name in ('train', 'validation', 'test'):
            (root / name).mkdir()
            relative = f'{name}/fixture_{name}.csv'
            loads = [0., -5., 0.] if name == 'validation' else [0., 220., 0.]
            pd.DataFrame(dict(timestamp=pd.date_range('2024-01-01', periods=3, freq='s'),
                time_s=[0., 1., 2.], load_total_kw=loads)).to_csv(root / relative, index=False)
            rows.append(dict(parent=f'parent_{name}', sample_id=f'fixture_{name}',
                relative_path=relative, split=name, point_count_1s=3))
        pd.DataFrame(rows).to_csv(root / 'metadata' / 'sample_manifest.csv', index=False)
        (root / 'metadata' / 'qa_summary.json').write_text(json.dumps({
            'parent_count': 3,
            'segment_count': 3,
            'point_count': 9,
            'split_point_counts': {'train': 3, 'validation': 3, 'test': 3},
        }), encoding='utf-8')
        return load_formal_operating_split(root)

    def test_legacy_split_manifest_is_not_a_runtime_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame([dict(parent_voyage='old', segment_id='old',
                one_second_csv='old.csv', split='train')]).to_csv(
                    root / 'split_manifest.csv', index=False)
            with self.assertRaises(FileNotFoundError):
                load_formal_operating_split(root)

    def test_frozen_manifest_totals_and_parent_groups_match_final_dataset(self) -> None:
        # Manifest metadata only. A unit-test run must never open held-out loads.
        frame = load_formal_operating_split().manifest
        self.assertEqual(frame.parent_voyage.nunique(), 43)
        self.assertEqual(len(frame), 43)
        self.assertEqual(int(frame.num_1s_points.sum()), 929_709)
        self.assertEqual(frame.groupby('split').num_1s_points.sum().to_dict(),
                         {'train': 707941, 'validation': 138763, 'test': 83005})
        self.assertEqual(frame.groupby('split').size().to_dict(),
                         {'test': 5, 'train': 30, 'validation': 8})
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
        self.assertEqual(audit.negative_load_point_count, 1)
        self.assertEqual(audit.orphan_segment_paths, ())
        self.assertEqual(audit.missing_segment_paths, ())
        self.assertEqual(audit.parent_split_leakage, ())

    def test_formal_loader_retains_finite_internal_negative_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            split = self.make_fixture(Path(directory))

            loads = load_operating_segment_loads(
                'validation',
                split.validation_segments[0],
                split=split,
            )

        self.assertEqual(loads.tolist(), [0.0, -5.0, 0.0])

    def test_dataset_auditor_rejects_metadata_point_count_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_fixture(root)
            qa_path = root / 'metadata' / 'qa_summary.json'
            qa = json.loads(qa_path.read_text(encoding='utf-8'))
            qa['point_count'] = 10
            qa_path.write_text(json.dumps(qa), encoding='utf-8')

            with self.assertRaisesRegex(
                ValueError,
                'formal dataset point_count mismatch',
            ):
                audit_formal_operating_dataset(root)

    def test_dqn_and_fixed_a0_load_the_same_ordered_test_segments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            split = self.make_fixture(Path(directory))
            dqn_loads = dqn_test.load_test_voyage(split.test_segments[0], split=split)
            a0_loads = a0_test.load_test_voyage(split.test_segments[0], split=split)

        self.assertIn("initial_soc=training.SOC_REFERENCE", inspect.getsource(dqn_test.run_test_episode))
        self.assertIn("initial_soc=0.55", inspect.getsource(a0_test.run_test_episode))
        self.assertEqual(dqn_loads.tolist(), a0_loads.tolist())

    def test_comparison_uses_the_frozen_final_test_ids(self) -> None:
        self.assertEqual(
            comparison.formal_test_voyage_ids(),
            load_formal_operating_split().test_segments,
        )


if __name__ == "__main__":
    unittest.main()
