import sys
import pathlib
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
from main.build_final_operating_dataset import assign_parent_roles, summarize_load, ledger_intervals, validate_sample
from utils.final_dataset_split import representative_parents, consecutive_boundary_low_count


class FinalBuilderTests(unittest.TestCase):
    def test_stable_low_boundary_requires_consecutive_observed_points(self):
        values=np.array([5.,6.,7.,20.,4.])
        self.assertEqual(consecutive_boundary_low_count(values),3)
        self.assertEqual(consecutive_boundary_low_count(values,from_start=False),1)
        self.assertEqual(consecutive_boundary_low_count([5.,np.nan,4.]),1)

    def test_feature_selection_is_deterministic_and_includes_hard_feasible_parent(self):
        inventory=pd.DataFrame({'parent':[f'p{x}' for x in range(12)],'chronological_index':range(12),
            'feasible_natural_voyages':[1]*11+[0],'natural_duration_s':np.arange(12)*300+600,
            'mean_load_kw':np.arange(12)*30+100,'above600_deficit_energy_kwh':np.arange(12)**2,
            'delta_abs_p95_kw_per_s':np.arange(12)[::-1]})
        a=representative_parents(inventory,7);b=representative_parents(inventory.sample(frac=1,random_state=1),7)
        self.assertEqual(a,b);self.assertEqual(len(set(a)),7);self.assertIn('p10',a);self.assertNotIn('p11',a)

    def test_parent_roles_are_fixed_before_sampling(self):
        roles = assign_parent_roles([str(x) for x in range(66)])
        self.assertEqual([list(roles.values()).count(r) for r in ['train','validation','test']], [46,13,7])
        self.assertEqual(roles['59'], 'test')
        with self.assertRaises(ValueError):assign_parent_roles(['one'])

    def test_load_energy_and_delta_do_not_count_initialization_twice(self):
        s = summarize_load(np.array([100., 700., 800.]))
        self.assertEqual(s['duration_s'], 2)
        self.assertAlmostEqual(s['energy_kwh'], 1500/3600)
        self.assertAlmostEqual(s['above600_deficit_energy_kwh'], 300/3600)
        self.assertEqual(s['above600_duration_s'], 2)

    def test_ledger_accounts_for_every_row_and_breaks_real_gaps(self):
        d = pd.DataFrame({'timestamp':pd.to_datetime(['2025-01-01 00:00:00','2025-01-01 00:00:30','2025-01-01 00:02:00']),
                          'reason':['bad','bad','bad'],'parent':['p']*3})
        rows = ledger_intervals(d, 45., 30.)
        self.assertEqual(len(rows), 2)
        self.assertEqual(sum(r['point_count'] for r in rows), 3)

    def test_validation_rejects_nan_negative_duplicate_or_gap(self):
        src = pd.DataFrame({'timestamp':pd.date_range('2025-01-01', periods=3, freq='30s'), 'load_total_kw':[30.,500.,40.]})
        out = pd.DataFrame({'timestamp':pd.date_range('2025-01-01', periods=61, freq='s'), 'time_s':np.arange(61), 'load_total_kw':np.full(61,30.)})
        validate_sample(src, out, 45.)
        for value in [np.nan, -1., np.inf]:
            bad = out.copy(); bad.loc[10,'load_total_kw']=value
            with self.assertRaises(ValueError):validate_sample(src,bad,45.)
        with self.assertRaises(ValueError):validate_sample(src.iloc[[0,2]],out,45.)

if __name__ == '__main__':unittest.main()
