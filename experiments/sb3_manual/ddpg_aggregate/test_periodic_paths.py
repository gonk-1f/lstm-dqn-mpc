"""Regressions for Windows MAX_PATH in periodic child evaluation outputs."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from .test_periodic_validation import FakeModel, validation_subprocess
from .periodic_validation import EpisodeValidationCallback
from .fixed_train_eval import FIXED_TRAIN_ID, FixedTrainEvaluationCallback


class PeriodicPathTests(unittest.TestCase):
    def test_real_checkpoint_children_write_under_deep_windows_paths(self):
        from .fixed_train_eval import PROJECT_ROOT
        checkpoint = PROJECT_ROOT / (
            "outputs/ddpg_fixed_parent_005_seg01_500ep_economic_v1/ddpg/"
            "AggregateShipDDPG-v0_1/fixed_train_evaluation/episode_0020/model.zip"
        )
        if not checkpoint.is_file():
            self.skipTest("Requires the saved checkpoint from the reported path failure")
        before_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        child_env = os.environ.copy()
        child_env["PYTHONPATH"] = os.pathsep.join((str(PROJECT_ROOT), str(PROJECT_ROOT / "src")))
        with tempfile.TemporaryDirectory(prefix="ddpg_path_smoke_") as tmp:
            base = Path(tmp)
            parent = base / ("r" * (200 - len(str(base)) - 1))
            for module, metrics_name, csv_name, limit in (
                ("fixed_train_eval", "fixed_train_metrics.json", "fixed_train_step_data.csv", ["--max-steps", "2"]),
                ("validate", "validation_metrics.json", "validation_step_data.csv", ["--episodes", "1", "--max-steps-per-episode", "2"]),
            ):
                with self.subTest(module=module):
                    node = parent / module[:1]
                    node.mkdir(parents=True)
                    result = subprocess.run([
                        sys.executable, "-m", f"experiments.sb3_manual.ddpg_aggregate.{module}",
                        "--model-path", str(checkpoint), "--output-dir", str(node), *limit,
                    ], cwd=PROJECT_ROOT, env=child_env, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, text=True, timeout=90)
                    self.assertEqual(result.returncode, 0, result.stdout)
                    metrics = json.loads((node / metrics_name).read_text(encoding="utf-8"))
                    self.assertEqual(Path(metrics["output_dir"]), node)
                    self.assertTrue(all(metrics["integrity"].values()))
                    self.assertTrue(metrics["deterministic"])
                    self.assertIsNone(metrics["action_noise"])
                    with (node / csv_name).open(encoding="utf-8-sig", newline="") as handle:
                        self.assertEqual(len(list(csv.DictReader(handle))), 2)
                    for path in node.iterdir():
                        self.assertLess(len(str(path)), 240)
                        if path.suffix == ".png":
                            self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(), before_hash)

    def test_both_callbacks_use_flat_output_with_actual_windows_run_length(self):
        with tempfile.TemporaryDirectory(prefix="ddpg_paths_") as tmp:
            # The reported Windows run directory has 166 characters.
            base = Path(tmp)
            run = base / ("r" * (166 - len(str(base)) - 1))
            run.mkdir()
            model = FakeModel(run)
            original = model.get_attr
            model.get_attr = lambda key: [FIXED_TRAIN_ID] if key == "fixed_train_id" else original(key)
            for callback, episode, directory, filename in (
                (FixedTrainEvaluationCallback(), 20, "fixed_train_evaluation", "fixed_train_step_data.csv"),
                (EpisodeValidationCallback(), 100, "periodic_validation", "validation_power_curves.png"),
            ):
                with self.subTest(callback=type(callback).__name__):
                    model._ddpg_completed_episodes = episode
                    callback.init_callback(model)
                    callback.on_training_start({}, {})
                    def child(command, **kwargs):
                        self.assertIn("--output-dir", command)
                        node = Path(command[command.index("--output-dir") + 1])
                        self.assertEqual(node, run / directory / f"episode_{episode:04d}")
                        self.assertLess(len(str(node / filename)), 240)
                        if isinstance(callback, EpisodeValidationCallback):
                            return validation_subprocess(command, **kwargs)
                        summary = dict(episode=episode, sample_id=FIXED_TRAIN_ID,
                                       evaluation_truncated=False, completed=False,
                                       coverage=0.5, mean_reward_per_step=0.8,
                                       episode_reward=10.0, total_cost_cny=3.0, soc_final=0.5)
                        (node / "fixed_train_metrics.json").write_text(json.dumps({
                            "split": "train", "summary": summary, "deterministic": True,
                            "action_noise": None, "integrity": {"parameters_unchanged": True},
                        }), encoding="utf-8")
                    with patch("subprocess.run", side_effect=child):
                        if isinstance(callback, EpisodeValidationCallback):
                            callback._save_and_validate()
                        else:
                            callback._save_and_evaluate()


if __name__ == "__main__":
    unittest.main(verbosity=2)
