from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
FORECASTING = ROOT / "src" / "forecasting"
if str(FORECASTING) not in sys.path:
    sys.path.insert(0, str(FORECASTING))

from millisecond_multistep_lstm import (  # noqa: E402
    ModelConfig,
    SequenceToVectorLSTM,
    baseline_forecasts,
    build_windows,
    fit_standard_scaler,
    metrics_by_horizon,
)


class TestWindowConstruction(unittest.TestCase):
    def test_windows_do_not_cross_sequence_boundaries(self) -> None:
        sequences = {
            "a": np.arange(40, dtype=np.float32),
            "b": np.arange(100, 140, dtype=np.float32),
        }
        windows = build_windows(sequences, history_steps=30, prediction_steps=6)
        self.assertEqual(windows.x.shape, (10, 30, 1))
        self.assertEqual(windows.y.shape, (10, 6))
        self.assertEqual(set(windows.sequence_ids.tolist()), {"a", "b"})
        self.assertFalse(np.any((windows.x[:, -1, 0] < 50) & (windows.y[:, 0] > 50)))

    def test_scaler_uses_only_training_values(self) -> None:
        scaler = fit_standard_scaler({"train_a": np.array([0.0, 2.0])})
        self.assertEqual(scaler.mean, 1.0)
        self.assertEqual(scaler.std, 1.0)
        np.testing.assert_allclose(scaler.transform(np.array([101.0])), [100.0])

    def test_empty_windows_have_stable_shapes(self) -> None:
        windows = build_windows({"short": np.arange(10)}, history_steps=30, prediction_steps=6)
        self.assertEqual(windows.x.shape, (0, 30, 1))
        self.assertEqual(windows.y.shape, (0, 6))


class TestBaselinesAndMetrics(unittest.TestCase):
    def test_baselines_follow_declared_formulas(self) -> None:
        history = np.arange(30, dtype=np.float64)[None, :, None]
        forecasts = baseline_forecasts(history, prediction_steps=6)
        np.testing.assert_allclose(forecasts["current_hold"], np.full((1, 6), 29.0))
        np.testing.assert_allclose(forecasts["last_slope"], [[30, 31, 32, 33, 34, 35]])
        np.testing.assert_allclose(
            forecasts["local_linear_trend"], [[30, 31, 32, 33, 34, 35]], atol=1e-10
        )

    def test_horizon_metrics_use_wape_not_row_mape(self) -> None:
        truth = np.array([[0.0, 2.0], [2.0, 2.0]])
        pred = np.array([[1.0, 1.0], [1.0, 3.0]])
        table = metrics_by_horizon(truth, pred)
        self.assertAlmostEqual(float(table.loc[table.horizon == 1, "wape_pct"].iloc[0]), 100.0)
        self.assertAlmostEqual(float(table.loc[table.horizon == 2, "wape_pct"].iloc[0]), 50.0)
        self.assertIn("aggregate", table["horizon"].astype(str).tolist())

    def test_zero_denominator_wape_is_nan(self) -> None:
        table = metrics_by_horizon(np.zeros((2, 1)), np.ones((2, 1)))
        self.assertTrue(np.isnan(float(table.loc[table.horizon == 1, "wape_pct"].iloc[0])))


class TestModelShape(unittest.TestCase):
    def test_model_maps_30_by_1_to_six_outputs(self) -> None:
        config = ModelConfig(hidden_size=32, num_layers=2, dropout=0.1, mlp_head=(64,))
        model = SequenceToVectorLSTM(config=config, prediction_steps=6)
        output = model(torch.zeros(4, 30, 1))
        self.assertEqual(tuple(output.shape), (4, 6))


if __name__ == "__main__":
    unittest.main()
