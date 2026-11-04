"""Focused fixed-voyage checks; no formal training or Validation8/Test4 inference."""
from __future__ import annotations

import csv
import importlib
import importlib.util
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[3]
import sys
sys.path.insert(0, str(ROOT / "src"))
PACKAGE = "experiments.sb3_manual.ddpg_aggregate"
DATA = ROOT / "data/processed/operating_dataset_v6_1s"
FIXED_ID = "parent_005_seg01"
RUN = "outputs/ddpg_fixed_parent_005_seg01_500ep_linear_v2"
TRAIN_RUN = RUN + "_tau001"


class FixedEnvironmentTests(unittest.TestCase):
    def make_env(self, **kwargs):
        from .env import AggregateDDPGEnv
        env = AggregateDDPGEnv(DATA, **kwargs)
        self.addCleanup(env.close)
        return env

    def test_fixed_reset_restarts_real_voyage_and_physics(self):
        env = self.make_env(fixed_train_id=FIXED_ID)
        for seed in (42, None, 7):
            obs, info = env.reset(seed=seed)
            self.assertEqual(info, {"sample_id": FIXED_ID, "split": "train"})
            self.assertEqual((env.index, env.episode.time_s[0]), (0, 0))
            self.assertEqual((env.physical.soc, env.physical.previous_fc_kw), (0.6, 0))
            self.assertEqual(len(env.episode.load_kw), 9258)
            self.assertAlmostEqual(float(obs[0]), 0.6, places=6)
            env.step([0.0])
            self.assertEqual(env.index, 1)

    def test_unknown_and_nontrain_fixed_ids_are_rejected(self):
        from .env import AggregateDDPGEnv
        for kwargs in ({"fixed_train_id": "missing"},
                       {"fixed_train_id": "parent_001_seg01"},
                       {"split": "validation", "fixed_train_id": FIXED_ID}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                AggregateDDPGEnv(DATA, **kwargs)

    def test_duplicate_fixed_id_is_rejected(self):
        from .env import AggregateDDPGEnv, V6TrainingDataset
        from .env import REWARD_CONFIG
        self.assertEqual(REWARD_CONFIG["lambda_soc"], 8.0)
        dataset = V6TrainingDataset.open(DATA)
        voyages = dataset.load_train()
        fixed = next(ep for ep in voyages if ep.sample_id == FIXED_ID)
        with patch.object(V6TrainingDataset, "load_train", return_value=[fixed, fixed]), \
                self.assertRaises(ValueError):
            AggregateDDPGEnv(DATA, fixed_train_id=FIXED_ID)

    def test_none_preserves_seeded_random_train35(self):
        env = self.make_env(fixed_train_id=None)
        first = [env.reset(seed=42 if i == 0 else None)[1]["sample_id"] for i in range(12)]
        second = [env.reset(seed=42 if i == 0 else None)[1]["sample_id"] for i in range(12)]
        self.assertEqual(first, second)
        self.assertGreater(len(set(first)), 1)
        self.assertEqual(len(env.episodes), 35)


class FixedConfigurationTests(unittest.TestCase):
    def test_native_zoo_loads_formal_config_and_runs_32_steps(self):
        import argparse
        from rl_zoo3.exp_manager import ExperimentManager
        from . import zoo_register
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(prefix="ddpg_fixed_zoo_") as tmp:
            manager = ExperimentManager(
                argparse.Namespace(), "ddpg", "AggregateShipDDPG-v0", log_folder=tmp,
                config=str(Path(__file__).with_name("zoo_ddpg.yml")),
                n_timesteps=32, eval_freq=-1, device="cpu", seed=42, verbose=0,
            )
            model, _ = manager.setup_experiment()
            try:
                self.assertEqual(model.tau, 0.01)
                self.assertFalse(manager.continue_training)
                self.assertEqual((model.num_timesteps, model._n_updates, model.replay_buffer.size()), (0, 0, 0))
                np.testing.assert_array_equal(model.action_noise._sigma, [0.1])
                self.assertEqual(model.get_env().get_attr("fixed_train_id"), [FIXED_ID])
                self.assertEqual(model.get_env().get_attr("reward_identity")[0]["lambda_soc"], 8.0)
                manager.learn(model)
                self.assertEqual((model.num_timesteps, model._n_updates, model.replay_buffer.size()), (32, 0, 32))
            finally:
                model.get_env().close()

    def test_formal_parameters_and_budgets(self):
        config = yaml.safe_load((Path(__file__).with_name("zoo_ddpg.yml")).read_text())["AggregateShipDDPG-v0"]
        expected = dict(tau=0.01, noise_std=0.1, n_envs=1, train_freq=32,
                        gradient_steps=1, learning_starts=5000, batch_size=64,
                        gamma=0.999, learning_rate=0.001, buffer_size=1000000,
                        policy_kwargs="dict(net_arch=[256, 256])", noise_type="normal")
        for name, value in expected.items():
            self.assertEqual(config[name], value, name)
        self.assertEqual(config["env_kwargs"]["fixed_train_id"], FIXED_ID)
        self.assertEqual(config["env_kwargs"]["split"], "train")
        self.assertGreaterEqual(config["n_timesteps"], 9258 * 500)
        callbacks = config["callback"]
        self.assertEqual(callbacks[0][PACKAGE + ".fixed_train_eval.FixedTrainEvaluationCallback"]["every_episodes"], 20)
        self.assertEqual(callbacks[1][PACKAGE + ".periodic_validation.EpisodeValidationCallback"]["every_episodes"], 100)
        self.assertEqual(callbacks[2]["stable_baselines3.common.callbacks.StopTrainingOnMaxEpisodes"]["max_episodes"], 500)

    def test_root_launchers_use_new_model_and_independent_splits(self):
        model_suffix = RUN + "/ddpg/AggregateShipDDPG-v0_1/AggregateShipDDPG-v0.zip"
        for name in ("train", "validate", "test", "fixed_eval"):
            path = ROOT / f"run_ddpg_{name}.py"
            self.assertTrue(path.is_file(), path)
            spec = importlib.util.spec_from_file_location(f"launcher_{name}", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            with patch("subprocess.run", return_value=SimpleNamespace(returncode=0)) as child:
                self.assertEqual(module.main(), 0)
            command = child.call_args.args[0]
            if name == "train":
                self.assertEqual(command[command.index("-m") + 1], "rl_zoo3.train")
                self.assertEqual(command[command.index("--log-folder") + 1], TRAIN_RUN)
                self.assertEqual(command[command.index("--conf-file") + 1], "experiments/sb3_manual/ddpg_aggregate/zoo_ddpg.yml")
                self.assertEqual(command[command.index("--device") + 1], "cuda")
                self.assertNotIn("--trained-agent", command)
                self.assertNotIn("-i", command)
                self.assertIn("--save-replay-buffer", command)
            else:
                self.assertTrue(command[command.index("--model-path") + 1].replace("\\", "/").endswith(model_suffix))
                expected_module = {"validate": "validate", "test": "test4", "fixed_eval": "fixed_train_eval"}[name]
                self.assertEqual(command[command.index("-m") + 1], PACKAGE + "." + expected_module)


class FixedEvaluationTests(unittest.TestCase):
    def module(self):
        return importlib.import_module(PACKAGE + ".fixed_train_eval")

    def test_real_subprocess_inference_is_bounded_and_does_not_mutate_training(self):
        from stable_baselines3 import DDPG
        from .env import AggregateDDPGEnv
        import os
        import subprocess
        from .validate import _model_snapshot
        env = AggregateDDPGEnv(DATA, fixed_train_id=FIXED_ID)
        self.addCleanup(env.close)
        torch.set_num_threads(1)
        model = DDPG("MlpPolicy", env, buffer_size=32, learning_starts=0,
                     train_freq=1, gradient_steps=1,
                     policy_kwargs=dict(net_arch=[16, 16]), seed=42, device="cpu")
        model._ddpg_reward_identity = dict(env.reward_identity)
        # A bounded real Train smoke performs an Adam update and creates nonempty Replay.
        model.learn(total_timesteps=4)
        with tempfile.TemporaryDirectory(prefix="ddpg_fixed_smoke_") as tmp:
            root = Path(tmp)
            checkpoint = root / "model.zip"
            model.save(checkpoint)
            before = _model_snapshot(model)
            replay = model.replay_buffer
            replay_before = tuple(array.copy() for array in (
                replay.observations, replay.next_observations, replay.actions, replay.rewards, replay.dones
            ))
            python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
            child_env = os.environ.copy()
            child_env["PYTHONPATH"] = os.pathsep.join((str(ROOT), str(ROOT / "src")))
            subprocess.run([
                sys.executable, "-m", PACKAGE + ".fixed_train_eval", "--model-path", str(checkpoint),
                "--output-root", str(root / "results"), "--episode", "20", "--max-steps", "3",
            ], cwd=ROOT, env=child_env, check=True, timeout=60)
            self.assertEqual(_model_snapshot(model), before)
            for array, saved in zip((replay.observations, replay.next_observations, replay.actions,
                                     replay.rewards, replay.dones), replay_before):
                np.testing.assert_array_equal(array, saved)
            self.assertEqual(random.getstate(), python_rng)
            np.testing.assert_array_equal(np.random.get_state()[1], numpy_rng[1])
            self.assertEqual(np.random.get_state()[2:], numpy_rng[2:])
            self.assertTrue(torch.equal(torch.get_rng_state(), torch_rng))
            files = list((root / "results").glob("fixed_train_*/fixed_train_metrics.json"))
            self.assertEqual(len(files), 1)
            metrics = json.loads(files[0].read_text(encoding="utf-8"))
            self.assertTrue(all(metrics["integrity"].values()))
            self.assertTrue(metrics["deterministic"])
            self.assertIsNone(metrics["action_noise"])
            self.assertEqual(metrics["summary"]["sample_id"], FIXED_ID)
            self.assertEqual(metrics["summary"]["episode_length"], 3)
            self.assertFalse(metrics["summary"]["completed"])
            self.assertTrue(metrics["summary"]["evaluation_truncated"])
            self.assertGreater(metrics["summary"]["total_cost_cny"], 0)
            with (files[0].parent / "fixed_train_step_data.csv").open(encoding="utf-8-sig") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 3)

    def test_combined_callback_triggers_include_failed_episodes_and_500(self):
        from stable_baselines3.common.callbacks import CallbackList, StopTrainingOnMaxEpisodes
        from .periodic_validation import EpisodeValidationCallback
        from .test_periodic_validation import FakeModel, validation_subprocess
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = FakeModel(root)
            original_get_attr = model.get_attr
            model.get_attr = lambda key: [FIXED_ID] if key == "fixed_train_id" else original_get_attr(key)
            model.num_envs = 1
            fixed = module.FixedTrainEvaluationCallback(output_root=root / "fixed")
            validation = EpisodeValidationCallback(output_root=root / "validation")
            callbacks = CallbackList([fixed, validation, StopTrainingOnMaxEpisodes(500)])
            callbacks.init_callback(model)
            callbacks.on_training_start({}, {})
            fixed_nodes = []
            def evaluate():
                fixed_nodes.append(fixed.completed_episodes)
            with patch.object(fixed, "_save_and_evaluate", side_effect=evaluate), \
                    patch("subprocess.run", side_effect=validation_subprocess) as child:
                for episode in range(1, 501):
                    model.num_timesteps += 2
                    info = {"vessel_episode": dict(
                        sample_id=FIXED_ID, completed=episode % 7 != 0, executed_steps=1,
                        total_steps=9258, coverage=1/9258, reward_sum=0.5,
                        terminal_reason="completed" if episode % 7 else "no_feasible_physical_power",
                    ), "episode": {"r": 0.5, "l": 2, "t": 0.01}}
                    callbacks.update_locals({"dones": np.array([True]), "infos": [info]})
                    self.assertEqual(callbacks.on_step(), episode < 500)
                self.assertEqual(child.call_count, 5)
            self.assertEqual(fixed_nodes, list(range(20, 501, 20)))
            with (root / "fixed/training_episodes.csv").open(encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 500)
            self.assertEqual(rows[6]["completed"], "False")
            self.assertEqual(rows[6]["executed_steps"], "1")

    def test_checkpoint_child_command_and_unique_node(self):
        from .test_periodic_validation import FakeModel
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = FakeModel(root)
            original = model.get_attr
            model.get_attr = lambda key: [FIXED_ID] if key == "fixed_train_id" else original(key)
            model._ddpg_completed_episodes = 20
            callback = module.FixedTrainEvaluationCallback(output_root=root / "fixed")
            callback.init_callback(model)
            callback.on_training_start({}, {})
            def child_run(command, **kwargs):
                self.assertEqual(command[command.index("-m") + 1], PACKAGE + ".fixed_train_eval")
                self.assertEqual(command[command.index("--sample-id") + 1], FIXED_ID)
                self.assertEqual(command[command.index("--device") + 1], "cpu")
                self.assertNotIn("--max-steps", command)
                directory = Path(command[command.index("--output-dir") + 1])
                directory.mkdir(parents=True, exist_ok=True)
                summary = dict(episode=20, sample_id=FIXED_ID, evaluation_truncated=False,
                               completed=False, coverage=0.5, mean_reward_per_step=0.8,
                               episode_reward=10.0, total_cost_cny=3.0, soc_final=0.5)
                (directory / "fixed_train_metrics.json").write_text(json.dumps({
                    "split": "train", "summary": summary, "deterministic": True,
                    "action_noise": None, "integrity": {"parameters_unchanged": True},
                }), encoding="utf-8")
                return SimpleNamespace(returncode=0)
            before = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
            with patch("subprocess.run", side_effect=child_run):
                callback._save_and_evaluate()
            self.assertEqual(random.getstate(), before[0])
            np.testing.assert_array_equal(np.random.get_state()[1], before[1][1])
            self.assertTrue(torch.equal(torch.get_rng_state(), before[2]))
            self.assertEqual(model.replay_buffer, [1, 2, 3])
            metadata = json.loads((root / "fixed/episode_0020/checkpoint_metadata.json").read_text())
            self.assertIn("before_replay", metadata["clock_phase"])
            self.assertEqual(metadata["completed_episodes"], 20)
            with patch("subprocess.run") as child, self.assertRaises(FileExistsError):
                callback._save_and_evaluate()
            child.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
