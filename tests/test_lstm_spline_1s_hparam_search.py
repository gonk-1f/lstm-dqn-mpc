from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import optuna

MAIN_ROOT = Path(__file__).resolve().parents[1] / "src" / "main"
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

from run_lstm_spline_1s_hparam_search import (  # noqa: E402
    SearchTask,
    TrialConfig,
    build_parser,
    build_default_tasks,
    build_fixed_taskC_config,
    build_horizon_steps,
    build_tasks_from_args,
    build_windows_for_series,
    config_fingerprint,
    current_hold_forecast,
    last_slope_forecast,
    moving_average_hold_forecast,
    primary_score_from_metrics,
    sample_trial_config,
    test_voyage_figure_path,
    write_one_voyage_prediction_figure,
    write_trial_snapshot,
)


class TestLstmSpline1sHparamSearch(unittest.TestCase):
    def test_default_tasks_report_required_horizons(self) -> None:
        tasks = build_default_tasks()

        self.assertEqual(build_horizon_steps(tasks["taskA"]), [1, 6, 30, 60])
        self.assertEqual(build_horizon_steps(tasks["taskB"]), [1, 6, 30, 60, 120, 180])

    def test_window_builder_uses_one_voyage_series_without_crossing_boundaries(self) -> None:
        load = np.arange(10, dtype=float)
        x, y = build_windows_for_series(load, history_len=3, pred_horizon=2, stride=1)

        self.assertEqual(x.shape, (6, 3, 1))
        self.assertEqual(y.shape, (6, 2))
        self.assertEqual(x[0, :, 0].tolist(), [0.0, 1.0, 2.0])
        self.assertEqual(y[0].tolist(), [3.0, 4.0])
        self.assertEqual(x[-1, :, 0].tolist(), [5.0, 6.0, 7.0])
        self.assertEqual(y[-1].tolist(), [8.0, 9.0])

    def test_baselines_are_causal(self) -> None:
        history = np.array([10.0, 12.0, 15.0], dtype=float)

        self.assertEqual(current_hold_forecast(history, 3).tolist(), [15.0, 15.0, 15.0])
        self.assertEqual(last_slope_forecast(history, 3).tolist(), [18.0, 21.0, 24.0])
        self.assertEqual(moving_average_hold_forecast(history, 3, window=2).tolist(), [13.5, 13.5, 13.5])

    def test_primary_score_uses_full_task_horizon_not_h1_only(self) -> None:
        task_a = SearchTask(name="taskA", pred_horizon=60, n_trials=40)
        task_b = SearchTask(name="taskB", pred_horizon=180, n_trials=30)
        metrics = {f"WAPE_h{idx}": float(idx) for idx in range(1, 181)}

        self.assertAlmostEqual(primary_score_from_metrics(metrics, task_a), 30.5)
        self.assertAlmostEqual(primary_score_from_metrics(metrics, task_b), 90.5)

    def test_config_fingerprint_detects_duplicate_trials(self) -> None:
        config_a = TrialConfig(
            history_len=540,
            pred_horizon=60,
            hidden_size=256,
            num_layers=3,
            dropout=0.0,
            mlp_head=(256, 128),
            loss="Huber",
            learning_rate=1e-3,
            batch_size=128,
            gradient_clip=0.5,
            weight_decay=1e-5,
            seed=123,
            epochs_max=50,
            early_stopping_patience=7,
        )
        config_b = TrialConfig(**{**config_a.__dict__, "epochs_max": 20, "early_stopping_patience": 3})

        self.assertEqual(config_fingerprint(config_a), config_fingerprint(config_b))

    def test_trial_snapshot_is_written_incrementally(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            row = {"task": "taskA", "trial_number": 0, "primary_score": 1.25}

            path = write_trial_snapshot(output_dir, "taskA", [row])

            self.assertEqual(path.name, "hparam_trials_taskA.partial.csv")
            self.assertTrue(path.exists())
            self.assertIn("primary_score", path.read_text(encoding="utf-8-sig"))

    def test_parser_defaults_to_bounded_trial_runtime(self) -> None:
        args = build_parser().parse_args([])

        self.assertEqual(args.trial_time_limit_sec, 1800)

    def test_task_c_cli_builds_fixed_30_step_input_6_step_output_task(self) -> None:
        args = build_parser().parse_args(["--run_taskC_30_to_6", "--n_trials_taskC", "10"])

        tasks = build_tasks_from_args(args)

        self.assertEqual(list(tasks.keys()), ["taskC_30_to_6"])
        task = tasks["taskC_30_to_6"]
        self.assertEqual(task.pred_horizon, 6)
        self.assertEqual(task.n_trials, 10)
        self.assertEqual(task.fixed_history_len, 30)
        self.assertEqual(build_horizon_steps(task), [1, 6])

    def test_task_c_trial_config_uses_fixed_history_len(self) -> None:
        args = build_parser().parse_args(["--run_taskC_30_to_6", "--n_trials_taskC", "1"])
        task = build_tasks_from_args(args)["taskC_30_to_6"]
        trial = optuna.create_study(direction="minimize").ask()

        config = sample_trial_config(trial, task, args)

        self.assertEqual(config.history_len, 30)
        self.assertEqual(config.pred_horizon, 6)

    def test_fixed_task_c_config_matches_selected_30_to_6_hyperparameters(self) -> None:
        args = build_parser().parse_args(["--run_fixed_taskC_30_to_6"])

        config = build_fixed_taskC_config(args)

        self.assertEqual(config.history_len, 30)
        self.assertEqual(config.pred_horizon, 6)
        self.assertEqual(config.hidden_size, 128)
        self.assertEqual(config.num_layers, 3)
        self.assertEqual(config.dropout, 0.0)
        self.assertEqual(config.mlp_head, (128,))
        self.assertEqual(config.loss, "Huber")
        self.assertEqual(config.learning_rate, 0.0001)
        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.gradient_clip, 1.0)
        self.assertEqual(config.weight_decay, 1e-5)
        self.assertEqual(config.seed, 123)

    def test_per_voyage_prediction_figure_is_written_under_task_folder(self) -> None:
        task = SearchTask(name="taskC_30_to_6", pred_horizon=6, n_trials=1, fixed_history_len=30)
        y_true = np.column_stack([np.arange(8, dtype=float) + idx for idx in range(6)])
        y_pred = y_true + 0.25
        time_s = np.arange(8, dtype=float)

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)

            path = write_one_voyage_prediction_figure(
                task=task,
                output_dir=output_dir,
                voyage_id="voyage_060",
                decision_time_s=time_s,
                y_true=y_true,
                y_pred=y_pred,
            )

            self.assertEqual(path, test_voyage_figure_path(output_dir, task, "voyage_060"))
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
