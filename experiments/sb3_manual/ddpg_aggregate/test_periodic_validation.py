"""Trigger checks only: the Validation subprocess is always mocked."""
from __future__ import annotations

import csv
import importlib
import importlib.util
import json
from pathlib import Path
import random
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch


class FakeLogger:
    def __init__(self):
        self.records = {}

    def record(self, key, value, **kwargs):
        self.records[key] = value

    def info(self, message):
        pass


class FakeModel:
    def __init__(self, directory):
        self.num_timesteps = 0
        self._episode_num = 0
        self._n_updates = 17
        self._ddpg_reward_identity = {"reward_mode": "economic", "lambda_soc": 12.5}
        self.seed = 42
        self.logger = FakeLogger()
        self.replay_buffer = [1, 2, 3]
        self.weights = np.array([0.5, 1.0])
        self.directory = directory
        self.saved = []

    def get_env(self):
        return self

    def get_attr(self, name):
        values = {
            "data_root": self.directory / "data",
            "reward_identity": {"reward_mode": "economic", "lambda_soc": 12.5},
            "results_writer": SimpleNamespace(
                file_handler=SimpleNamespace(name=str(self.directory / "0.monitor.csv"))
            ),
        }
        return [values[name]]

    def save(self, path):
        path = Path(path)
        path.write_bytes(b"mock checkpoint")
        self.saved.append(path)


def validation_subprocess(command, **kwargs):
    output = Path(command[command.index("--output-dir") + 1])
    output.mkdir(parents=True, exist_ok=True)
    rows = [{
        "sample_id": f"validation_{index}", "split": "validation", "completed": True,
        "soc_initial": 0.6, "soc_final": 0.55, "soc_min": 0.4, "soc_max": 0.6,
        "h2_cost_cny": 1.0, "fuel_cell_degradation_cost_cny": 2.0,
        "battery_degradation_cost_cny": 3.0, "c_bat_eq_cny": 0.25,
    } for index in range(8)]
    metrics = {
        "validation_voyages": 8, "validation_set_complete": True,
        "completion_rate": 1.0, "mean_coverage": 1.0, "mean_episode_reward": 0.8,
        "per_voyage": rows, "integrity": {"parameters_unchanged": True},
    }
    (output / "validation_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    return SimpleNamespace(returncode=0)


class EpisodeValidationTests(unittest.TestCase):
    def test_unversioned_replay_cannot_resume_after_step_counter_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = FakeModel(Path(tmp))
            del model._ddpg_reward_identity
            model.num_timesteps = model._n_updates = 0
            model.replay_buffer = SimpleNamespace(size=lambda: 1)
            with self.assertRaisesRegex(ValueError, "Legacy reward checkpoint"):
                self.start(model, output_root=Path(tmp) / "periodic")

    def test_old_reward_checkpoint_cannot_resume_even_after_counter_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = FakeModel(Path(tmp))
            model.num_timesteps = 0
            model._ddpg_reward_identity = {"reward_version": "ddpg_economic_tanh_v1"}
            with self.assertRaisesRegex(ValueError, "another reward definition"):
                self.start(model, output_root=Path(tmp) / "periodic")

    def test_legacy_resume_is_rejected_after_sb3_resets_step_counter(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = FakeModel(Path(tmp))
            del model._ddpg_reward_identity
            model.num_timesteps = 0
            with self.assertRaisesRegex(ValueError, "Legacy reward checkpoint"):
                self.start(model, output_root=Path(tmp) / "periodic")

    def callback_type(self):
        name = "experiments.sb3_manual.ddpg_aggregate.periodic_validation"
        self.assertIsNotNone(importlib.util.find_spec(name), "Episode validation callback is missing")
        return importlib.import_module(name).EpisodeValidationCallback

    def start(self, model, **kwargs):
        callback = self.callback_type()(**kwargs)
        callback.init_callback(model)
        callback.on_training_start({}, {})
        return callback

    def step(self, callback, model, done):
        model.num_timesteps += 1
        callback.update_locals({"dones": np.array([done])})
        return callback.on_step()

    def test_actual_episode_triggers_100_to_500_without_training_side_effects(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = FakeModel(Path(tmp))
            callback = self.start(model, output_root=Path(tmp) / "periodic")
            python_rng = random.getstate()
            numpy_rng = np.random.get_state()
            torch_rng = torch.get_rng_state().clone()
            with patch("subprocess.run", side_effect=validation_subprocess) as child:
                for episode in range(1, 501):
                    for _ in range(episode % 3 + 1):
                        self.assertTrue(self.step(callback, model, False))
                    self.assertEqual(self.step(callback, model, True), episode < 500)
                self.assertEqual(child.call_count, 5)
                for call in child.call_args_list:
                    command, options = call.args[0], call.kwargs
                    self.assertEqual(command[command.index("-m") + 1],
                                     "experiments.sb3_manual.ddpg_aggregate.validate")
                    self.assertEqual(command[command.index("--episodes") + 1], "8")
                    self.assertEqual(command[command.index("--device") + 1], "cpu")
                    self.assertEqual(command[command.index("--seed") + 1], "42")
                    self.assertIn("PYTHONPATH", options["env"])
                    self.assertNotIn("test4", " ".join(command))
            self.assertEqual(model._ddpg_completed_episodes, 500)
            self.assertEqual(model._episode_num, 0)
            self.assertEqual(model._n_updates, 17)
            self.assertEqual(model.replay_buffer, [1, 2, 3])
            np.testing.assert_array_equal(model.weights, [0.5, 1.0])
            self.assertEqual(random.getstate(), python_rng)
            after_numpy = np.random.get_state()
            self.assertEqual(after_numpy[0], numpy_rng[0])
            np.testing.assert_array_equal(after_numpy[1], numpy_rng[1])
            self.assertEqual(after_numpy[2:], numpy_rng[2:])
            self.assertTrue(torch.equal(torch.get_rng_state(), torch_rng))
            nodes = json.loads((Path(tmp) / "periodic/periodic_validation.json").read_text())
            self.assertEqual([row["episode"] for row in nodes], [100, 200, 300, 400, 500])
            self.assertEqual(nodes[0]["total_economic_cost_cny"], 50.0)
            self.assertEqual(len(nodes[0]["per_voyage"]), 8)
            with (Path(tmp) / "periodic/periodic_validation.csv").open(encoding="utf-8-sig", newline="") as handle:
                csv_rows = list(csv.DictReader(handle))
            self.assertEqual([int(row["episode"]) for row in csv_rows], [100, 200, 300, 400, 500])

    def test_resume_and_monitor_run_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = FakeModel(Path(tmp))
            model._ddpg_completed_episodes = 199
            callback = self.start(model)
            with patch("subprocess.run", side_effect=validation_subprocess) as child:
                self.assertTrue(self.step(callback, model, True))
            self.assertEqual(child.call_count, 1)
            self.assertTrue((Path(tmp) / "periodic_validation/episode_0200/model.zip").is_file())
            metadata = json.loads((Path(tmp) / "periodic_validation/episode_0200/checkpoint_metadata.json").read_text())
            self.assertEqual(metadata["completed_episodes"], 200)
            self.assertIn("before_replay", metadata["clock_phase"])

    def test_existing_checkpoint_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = FakeModel(Path(tmp))
            model._ddpg_completed_episodes = 99
            root = Path(tmp) / "periodic"
            (root / "episode_0100").mkdir(parents=True)
            marker = root / "episode_0100/model.zip"
            marker.write_bytes(b"keep old model")
            callback = self.start(model, output_root=root)
            with patch("subprocess.run") as child, self.assertRaises(FileExistsError):
                self.step(callback, model, True)
            child.assert_not_called()
            self.assertEqual(marker.read_bytes(), b"keep old model")


if __name__ == "__main__":
    unittest.main()
