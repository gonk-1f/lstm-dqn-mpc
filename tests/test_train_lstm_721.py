from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
MAIN_ROOT = SRC_ROOT / "main"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

from run_train_lstm_721 import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    DEFAULT_REFERENCE_META,
    _windows_from_frame,
    apply_training_overrides,
    build_parser,
    asym_weighted_huber_loss,
    best_checkpoint_score,
    config_from_reference_meta,
    compute_train_loss_thresholds,
    detailed_horizon_metrics,
    make_training_dataset,
)
from forecasting.lstm_load_predictor import LSTMForecastConfig  # noqa: E402
from forecasting.feature_pipeline import (  # noqa: E402
    clean_total_load_feature_columns,
    clean_total_load_feature_columns_1s,
    clean_total_load_speed_feature_columns,
    prepare_lstm_features,
)
from run_train_lstm_total_load_721 import build_parser as build_total_load_parser  # noqa: E402


class TestTrainLstm721(unittest.TestCase):
    def test_defaults_do_not_overwrite_current_checkpoint(self) -> None:
        args = build_parser().parse_args([])

        self.assertEqual(Path(args.output_dir), DEFAULT_OUTPUT_DIR)
        self.assertEqual(Path(args.reference_meta), DEFAULT_REFERENCE_META)
        self.assertFalse(args.overwrite_current)
        self.assertIn("lstm_721_retrain", str(DEFAULT_OUTPUT_DIR))

    def test_total_load_training_defaults_use_separate_outputs(self) -> None:
        args = build_total_load_parser().parse_args([])

        self.assertIn("total_load_66_segments.csv", str(args.source_csv))
        self.assertIn("voyage_split_total_load_721.json", str(args.split_json))
        self.assertIn("outputs\\lstm_total_load_721", str(Path(args.output_dir)))
        self.assertFalse(args.overwrite_current)
        self.assertEqual(args.candidate, "candidate_A_loss_recalib")
        self.assertEqual(args.history_len, 18)
        self.assertAlmostEqual(args.lr, 2.0e-4)
        self.assertEqual(args.epochs, 60)
        self.assertEqual(args.patience, 10)
        self.assertAlmostEqual(args.huber_delta_kw, 20.0)
        self.assertAlmostEqual(args.asym_under_weight, 1.5)
        self.assertEqual(args.selection_metric, "validation_weighted_MAE_h1_h3")
        self.assertEqual(args.horizon_weight, "1.5,1.3,1.1,1.0,0.8,0.6")
        self.assertEqual(args.feature_mode, "load_only")
        self.assertTrue(args.auto_loss_thresholds)

    def test_linear_interp_1s_training_entrypoint_is_deprecated(self) -> None:
        self.assertFalse((MAIN_ROOT / "run_train_lstm_total_load_1s_721.py").exists())
        self.assertTrue(
            (MAIN_ROOT / "_deprecated_run_train_lstm_total_load_1s_linear_interp_721_DO_NOT_USE.py").exists()
        )
        self.assertNotIn("speed_knots", clean_total_load_feature_columns_1s())
        self.assertNotIn("delta_speed", clean_total_load_feature_columns_1s())

    def test_clean_total_load_features_exclude_missing_side_speed_and_soc_inputs(self) -> None:
        features = clean_total_load_feature_columns()

        self.assertIn("load_total_kw", features)
        self.assertIn("time_sin", features)
        self.assertIn("time_cos", features)
        self.assertIn("delta_load_total", features)
        self.assertIn("ramp_6_load_total", features)
        self.assertIn("rolling_mean_load_total_w18", features)
        for removed in [
            "load_left_kw",
            "load_right_kw",
            "speed_knots",
            "delta_speed",
            "soc_left",
            "soc_right",
            "delta_load_left",
            "delta_load_right",
        ]:
            self.assertNotIn(removed, features)

    def test_clean_total_load_1s_features_use_second_scale_windows(self) -> None:
        features = clean_total_load_feature_columns_1s()

        self.assertIn("load_total_kw", features)
        self.assertIn("ramp_60_load_total", features)
        self.assertIn("rolling_mean_load_total_w180", features)
        self.assertNotIn("ramp_6_load_total", features)
        self.assertNotIn("rolling_mean_load_total_w18", features)

    def test_total_load_speed_features_add_only_ais_speed_inputs(self) -> None:
        load_only = clean_total_load_feature_columns()
        with_speed = clean_total_load_speed_feature_columns()

        self.assertIn("speed_knots", with_speed)
        self.assertIn("delta_speed", with_speed)
        self.assertNotIn("speed_knots", load_only)
        self.assertNotIn("delta_speed", load_only)
        for feature in load_only:
            self.assertIn(feature, with_speed)

    def test_config_from_reference_meta_preserves_current_delta10_hyperparameters(self) -> None:
        config, feature_set, loss_meta = config_from_reference_meta(DEFAULT_REFERENCE_META)

        self.assertEqual(config.history_len, 18)
        self.assertEqual(config.pred_horizon, 6)
        self.assertEqual(config.hidden_size, 192)
        self.assertEqual(config.num_layers, 2)
        self.assertAlmostEqual(config.dropout, 0.2)
        self.assertAlmostEqual(config.lr, 1.0e-4)
        self.assertEqual(config.epochs, 30)
        self.assertEqual(config.patience, 5)
        self.assertEqual(feature_set, "rolling")
        self.assertEqual(loss_meta["loss"], "asym_weighted_huber")
        self.assertAlmostEqual(loss_meta["huber_delta_kw"], 10.0)
        self.assertAlmostEqual(loss_meta["asym_under_weight"], 3.0)

    def test_asym_weighted_huber_penalizes_underprediction_more_than_overprediction(self) -> None:
        y_true = torch.tensor([[10.0, 10.0]])
        y_under = torch.tensor([[8.0, 8.0]])
        y_over = torch.tensor([[12.0, 12.0]])

        under = asym_weighted_huber_loss(
            y_pred_norm=y_under,
            y_true_norm=y_true,
            y_true_kw=y_true,
            target_std_kw=1.0,
            huber_delta_kw=10.0,
            asym_under_weight=3.0,
            high_load_threshold_kw=100.0,
            ramp_threshold_kw=100.0,
        )
        over = asym_weighted_huber_loss(
            y_pred_norm=y_over,
            y_true_norm=y_true,
            y_true_kw=y_true,
            target_std_kw=1.0,
            huber_delta_kw=10.0,
            asym_under_weight=3.0,
            high_load_threshold_kw=100.0,
            ramp_threshold_kw=100.0,
        )

        self.assertTrue(torch.isfinite(under))
        self.assertGreater(float(under), float(over))

    def test_make_training_dataset_returns_norm_and_kw_targets_together(self) -> None:
        x = torch.zeros((2, 3, 1)).numpy()
        y_norm = torch.tensor([[1.0], [2.0]]).numpy()
        y_kw = torch.tensor([[10.0], [20.0]]).numpy()

        dataset = make_training_dataset(x, y_norm, y_kw)

        self.assertEqual(len(dataset), 2)
        _, first_norm, first_kw = dataset[0]
        _, second_norm, second_kw = dataset[1]
        self.assertAlmostEqual(float(first_norm[0]), 1.0)
        self.assertAlmostEqual(float(first_kw[0]), 10.0)
        self.assertAlmostEqual(float(second_norm[0]), 2.0)
        self.assertAlmostEqual(float(second_kw[0]), 20.0)

    def test_best_checkpoint_score_uses_validation_mae_not_huber_loss(self) -> None:
        lower_loss_but_worse_mae = {"val_loss": 0.01, "val_MAE": 5.0}
        higher_loss_but_better_mae = {"val_loss": 0.02, "val_MAE": 3.0}

        self.assertGreater(
            best_checkpoint_score(lower_loss_but_worse_mae),
            best_checkpoint_score(higher_loss_but_better_mae),
        )

    def test_best_checkpoint_score_can_use_weighted_h1_h3_metric(self) -> None:
        worse_near_term = {
            "val_loss": 0.01,
            "val_MAE": 3.0,
            "val_weighted_MAE_h1_h3": 5.0,
        }
        better_near_term = {
            "val_loss": 0.02,
            "val_MAE": 4.0,
            "val_weighted_MAE_h1_h3": 2.0,
        }

        self.assertGreater(
            best_checkpoint_score(worse_near_term, "validation_weighted_MAE_h1_h3"),
            best_checkpoint_score(better_near_term, "validation_weighted_MAE_h1_h3"),
        )

    def test_compute_train_loss_thresholds_uses_train_quantiles_with_voyage_groups(self) -> None:
        import pandas as pd

        train = pd.DataFrame(
            {
                "voyage_name": ["a", "a", "a", "b", "b", "b"],
                "load_total_kw": [0.0, 10.0, 20.0, 100.0, 120.0, 140.0],
            }
        )

        high_load, ramp = compute_train_loss_thresholds(train, quantile=0.75)

        self.assertAlmostEqual(high_load, 115.0)
        self.assertAlmostEqual(ramp, 20.0)

    def test_detailed_horizon_metrics_reports_bias_and_direction_rates(self) -> None:
        import numpy as np

        true = np.array([[10.0, 20.0, 30.0], [20.0, 10.0, 40.0]])
        pred = np.array([[8.0, 25.0, 30.0], [22.0, 9.0, 50.0]])

        metrics = detailed_horizon_metrics(true, pred)

        self.assertAlmostEqual(metrics["Bias_h1"], 0.0)
        self.assertAlmostEqual(metrics["under_prediction_rate_h1"], 0.5)
        self.assertAlmostEqual(metrics["over_prediction_rate_h1"], 0.5)
        self.assertAlmostEqual(metrics["Bias_h2"], 2.0)
        self.assertAlmostEqual(metrics["under_prediction_rate_h3"], 0.0)
        self.assertAlmostEqual(metrics["over_prediction_rate_h3"], 0.5)

    def test_apply_training_overrides_sets_recalibrated_candidate_values(self) -> None:
        import argparse

        config = LSTMForecastConfig(history_len=18, pred_horizon=6, hidden_size=192, num_layers=2)
        loss_meta = {
            "loss": "asym_weighted_huber",
            "huber_delta_kw": 10.0,
            "asym_under_weight": 3.0,
            "asym_high_load_bonus": 0.5,
            "asym_ramp_bonus": 0.2,
            "horizon_weight": [1.0] * 6,
            "features": ["load_total_kw"],
        }
        args = argparse.Namespace(
            history_len=36,
            pred_horizon=None,
            hidden_size=None,
            num_layers=None,
            dropout=None,
            batch_size=None,
            lr=2e-4,
            epochs=60,
            patience=10,
            seed=None,
            grad_clip=None,
            huber_delta_kw=20.0,
            asym_under_weight=1.5,
            asym_high_load_bonus=0.2,
            asym_ramp_bonus=0.1,
            horizon_weight="1.5,1.3,1.1,1.0,0.8,0.6",
            selection_metric="validation_weighted_MAE_h1_h3",
        )

        selection_metric = apply_training_overrides(config, loss_meta, args)

        self.assertEqual(config.history_len, 36)
        self.assertAlmostEqual(config.lr, 2e-4)
        self.assertEqual(config.epochs, 60)
        self.assertEqual(config.patience, 10)
        self.assertAlmostEqual(loss_meta["huber_delta_kw"], 20.0)
        self.assertAlmostEqual(loss_meta["asym_under_weight"], 1.5)
        self.assertEqual(loss_meta["horizon_weight"], [1.5, 1.3, 1.1, 1.0, 0.8, 0.6])
        self.assertEqual(selection_metric, "validation_weighted_MAE_h1_h3")

    def test_windows_group_by_voyage_id_when_voyage_name_is_absent(self) -> None:
        import pandas as pd

        prepared = pd.DataFrame(
            {
                "voyage_id": ["a", "a", "a", "b", "b", "b"],
                "load_total_kw": [1.0, 2.0, 3.0, 100.0, 101.0, 102.0],
            }
        )
        config = LSTMForecastConfig(history_len=2, pred_horizon=2)

        with self.assertRaises(ValueError):
            _windows_from_frame(
                prepared,
                features=["load_total_kw"],
                config=config,
                feature_scaler={"mean": [0.0], "std": [1.0]},
                target_scaler={"mean": 0.0, "std": 1.0},
            )

    def test_feature_pipeline_groups_by_voyage_id_when_voyage_name_is_absent(self) -> None:
        import pandas as pd

        raw = pd.DataFrame(
            {
                "voyage_id": ["a", "a", "b", "b"],
                "load_total_kw": [1.0, 3.0, 100.0, 101.0],
            }
        )

        prepared = prepare_lstm_features(raw, "rolling")

        self.assertAlmostEqual(float(prepared.loc[0, "delta_load_total"]), 0.0)
        self.assertAlmostEqual(float(prepared.loc[1, "delta_load_total"]), 2.0)
        self.assertAlmostEqual(float(prepared.loc[2, "delta_load_total"]), 0.0)
        self.assertAlmostEqual(float(prepared.loc[3, "delta_load_total"]), 1.0)

    def test_rolling_1s_feature_pipeline_uses_past_only_windows(self) -> None:
        import pandas as pd

        raw = pd.DataFrame(
            {
                "voyage_id": ["a"] * 181,
                "timestamp": pd.date_range("2024-01-01", periods=181, freq="s"),
                "load_total_kw": [float(i) for i in range(181)],
            }
        )

        prepared = prepare_lstm_features(raw, "rolling_1s")

        self.assertAlmostEqual(float(prepared.loc[180, "rolling_mean_load_total_w180"]), 90.5)
        self.assertAlmostEqual(float(prepared.loc[180, "ramp_60_load_total"]), 60.0)
        self.assertAlmostEqual(float(prepared.loc[180, "slope_60_load_total"]), 1.0)

    def test_windows_from_frame_supports_stride_for_1s_training(self) -> None:
        import pandas as pd

        prepared = pd.DataFrame(
            {
                "voyage_id": ["a"] * 12,
                "load_total_kw": [float(i) for i in range(12)],
            }
        )
        config = LSTMForecastConfig(history_len=3, pred_horizon=2)

        x, y_norm, y_kw = _windows_from_frame(
            prepared,
            features=["load_total_kw"],
            config=config,
            feature_scaler={"mean": [0.0], "std": [1.0]},
            target_scaler={"mean": 0.0, "std": 1.0},
            window_stride=2,
        )

        self.assertEqual(x.dtype.name, "float32")
        self.assertEqual(y_norm.dtype.name, "float32")
        self.assertEqual(x.shape[0], 4)
        self.assertEqual(y_kw[0].tolist(), [3.0, 4.0])
        self.assertEqual(y_kw[1].tolist(), [5.0, 6.0])


if __name__ == "__main__":
    unittest.main()
