import unittest

import numpy as np

from v3.test_diagnostic import aligned_forecasts, change_lag_steps


class TestDiagnosticAlignmentTests(unittest.TestCase):
    def test_each_horizon_is_aligned_to_its_future_target_time(self):
        actual = np.array([1, 2, 3, 4, 5], dtype=float)
        predicted = aligned_forecasts(
            actual, ("onboard",) * 5, history_steps=3,
            forecaster=lambda history: (history[-1] + 10, history[-1] + 20),
            horizon=2,
        )
        self.assertTrue(np.isnan(predicted[0, :3]).all())
        self.assertEqual(predicted[0, 3:].tolist(), [13.0, 14.0])
        self.assertTrue(np.isnan(predicted[1, :4]).all())
        self.assertEqual(predicted[1, 4], 23.0)

    def test_positive_change_lag_means_prediction_is_to_the_right(self):
        actual = np.zeros(60)
        actual[10:15] = 1
        actual[25:32] = 2
        actual[42:48] = 1
        predicted = np.r_[np.zeros(2), actual[:-2]]
        self.assertEqual(change_lag_steps(actual, predicted, max_lag_steps=5), 2)


if __name__ == "__main__":
    unittest.main()
