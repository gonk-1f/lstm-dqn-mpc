import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from utils.final_dataset_source import unique_nearest, collapse_channel, align_ais_vectorized
from utils.rebuilt_operating_dataset import align_ais_to_power


class SourceTests(unittest.TestCase):
    def test_vectorized_ais_preserves_bounded_existing_policy(self):
        origin=pd.Timestamp('2024-01-01')
        power=pd.DataFrame({'timestamp':origin+pd.to_timedelta([0,5,25,55,125,150],unit='s')})
        ais=pd.DataFrame({'timestamp':origin+pd.to_timedelta([0,20,40,120],unit='s'),'ais_speed_kn':[0.,3.,5.,1.]})
        expected=align_ais_to_power(power,ais,max_normal_gap_s=40,max_nearest_s=20)
        actual=align_ais_vectorized(power,ais,40,20)
        np.testing.assert_allclose(actual.speed_aligned_kn,expected.speed_aligned_kn,equal_nan=True)
        self.assertEqual(actual.speed_source.tolist(),expected.speed_source.tolist())

    def test_async_match_preserves_source_and_does_not_bridge_gap(self):
        ref = pd.to_datetime(['2024-01-01 00:00:00', '2024-01-01 00:00:30', '2024-01-01 00:01:00'])
        src = pd.DataFrame({'timestamp': ref[:2] + pd.Timedelta(seconds=3), 'power_kw': [5., 7.]})
        result = unique_nearest(ref, src, 4.)
        np.testing.assert_allclose(result.power_kw.iloc[:2], [5., 7.])
        self.assertTrue(np.isnan(result.power_kw.iloc[2]))
        self.assertEqual(result.offset_s.iloc[0], 3.)

    def test_ties_and_reuse_are_rejected(self):
        t = pd.Timestamp('2024-01-01')
        src = pd.DataFrame({'timestamp': [t - pd.Timedelta(seconds=2), t + pd.Timedelta(seconds=2)], 'power_kw': [1., 2.]})
        self.assertTrue(np.isnan(unique_nearest(pd.DatetimeIndex([t]), src, 3.).power_kw.iloc[0]))
        src = src.iloc[1:]
        got = unique_nearest(pd.DatetimeIndex([t, t + pd.Timedelta(seconds=3)]), src, 3.)
        self.assertEqual(got.power_kw.notna().sum(), 1)

    def test_conflicting_duplicate_not_averaged(self):
        df = pd.DataFrame({'timestamp': ['2024-01-01'] * 3, 'power_kw': [1., 1., 9.]})
        clean, qa = collapse_channel(df, ['power_kw'])
        self.assertTrue(np.isnan(clean.power_kw.iloc[0]))
        self.assertEqual(qa['conflicting_timestamps'], 1)


if __name__ == '__main__':
    unittest.main()
