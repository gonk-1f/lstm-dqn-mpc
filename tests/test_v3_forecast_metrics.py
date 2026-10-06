import math
import unittest

import numpy as np

from v3.metrics import forecast_metrics, persistence_forecast
from v3.training import fit_and_select_history


class ForecastMetricTests(unittest.TestCase):
    def test_persistence_repeats_only_last_observed_load(self):
        histories = np.array([[[1.0], [2.0]], [[3.0], [4.0]]])
        np.testing.assert_array_equal(
            persistence_forecast(histories, horizon=3),
            [[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]],
        )

    def test_wape_uses_ratio_of_sums_and_mape_is_undefined_at_zero(self):
        actual = np.array([[0.0, 100.0], [200.0, 100.0]])
        predicted = np.array([[10.0, 90.0], [190.0, 110.0]])
        result = forecast_metrics(actual, predicted)
        self.assertEqual(result["mae_kw"], 10.0)
        self.assertEqual(result["rmse_kw"], 10.0)
        self.assertEqual(result["wape_percent"], 10.0)
        self.assertIsNone(result["mape_percent"])
        self.assertEqual(result["zero_actual_count"], 1)
        self.assertEqual(result["horizon"][0]["wape_percent"], 10.0)

    def test_mape_and_bias_when_all_actuals_are_positive(self):
        actual = np.array([[100.0, 200.0]])
        predicted = np.array([[110.0, 180.0]])
        result = forecast_metrics(actual, predicted)
        self.assertEqual(result["mape_percent"], 10.0)
        self.assertEqual(result["bias_kw"], 5.0)
        self.assertEqual(result["wape_percent"], 10.0)

    def test_wape_is_undefined_when_all_actual_loads_are_zero(self):
        result = forecast_metrics(np.zeros((2, 2)), np.ones((2, 2)))
        self.assertIsNone(result["wape_percent"])
        self.assertEqual(result["mae_kw"], 1.0)
        self.assertTrue(math.isfinite(result["rmse_kw"]))

    def test_selection_report_includes_percentage_and_persistence_metrics(self):
        from types import SimpleNamespace

        episode = SimpleNamespace(
            load_kw=np.arange(1, 25, dtype=float),
            operating_mode=("onboard",) * 24,
        )
        _, report = fit_and_select_history(
            (episode,), (episode,), candidates=(3,), epochs=1,
            patience=1, batch_size=8, hidden_size=4,
        )
        self.assertEqual(report["selection_rule"], "minimum equal-origin Validation five-step WAPE")
        self.assertIn("validation_metrics", report["results"][0])
        self.assertIn("wape_percent", report["persistence_metrics"])
        self.assertIn("wape_skill_percent", report)


if __name__ == "__main__":
    unittest.main()
