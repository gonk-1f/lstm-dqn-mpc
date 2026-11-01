"""Configuration-only checks: no learn(), Study creation, or Test payloads."""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from optuna.trial import FixedTrial
import torch
from torch import nn
import yaml
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.logger import configure
from stable_baselines3.common.noise import NormalActionNoise

PACKAGE = "experiments.sb3_manual.ddpg_aggregate"
HERE = Path(__file__).parent
DATA = Path("data/processed/operating_dataset_v6_1s")


class SamplerTests(unittest.TestCase):
    def test_sampler_only_two_variables(self):
        self.assertIsNotNone(importlib.util.find_spec(PACKAGE + ".zoo_optuna"))
        adapter = importlib.import_module(PACKAGE + ".zoo_optuna")
        for tau, sigma, n_actions in ((0.005, 0.03, 1), (0.15, 0.12, 3)):
            trial = FixedTrial({"tau": tau, "noise_std": sigma})
            params = adapter.sample_ddpg_params(trial, n_actions, 1, {})
            self.assertEqual(set(params), {"tau", "action_noise"})
            self.assertEqual(set(trial.params), {"tau", "noise_std"})
            self.assertTrue(trial.distributions["tau"].log)
            self.assertEqual(trial.distributions["tau"].low, 0.005)
            self.assertEqual(trial.distributions["tau"].high, 0.15)
            self.assertFalse(trial.distributions["noise_std"].log)
            self.assertEqual(trial.distributions["noise_std"].low, 0.03)
            self.assertEqual(trial.distributions["noise_std"].high, 0.12)
            self.assertIsInstance(params["action_noise"], NormalActionNoise)
            np.testing.assert_array_equal(params["action_noise"]._mu, np.zeros(n_actions))
            np.testing.assert_array_equal(params["action_noise"]._sigma, sigma * np.ones(n_actions))

    def test_native_registry_and_entrypoint(self):
        adapter = importlib.import_module(PACKAGE + ".zoo_optuna")
        from rl_zoo3.exp_manager import ExperimentManager
        from rl_zoo3.train import train
        registry = ExperimentManager.objective.__globals__["HYPERPARAMS_SAMPLER"]
        self.assertIs(registry["ddpg"], adapter.sample_ddpg_params)
        self.assertIs(adapter.train, train)


class NativeConfigurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        importlib.import_module(PACKAGE + ".zoo_optuna")
        importlib.import_module(PACKAGE + ".zoo_register")
        from rl_zoo3.exp_manager import ExperimentManager
        cls.temp = tempfile.TemporaryDirectory(prefix="ddpg_config_")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.manager = ExperimentManager(
            argparse.Namespace(), "ddpg", "AggregateShipDDPG-v0",
            log_folder=cls.temp.name, config=str(HERE / "zoo_ddpg_optuna.yml"),
            optimize_hyperparameters=True, n_timesteps=77760,
            n_eval_episodes=8, n_eval_envs=1, n_evaluations=2,
            eval_env_kwargs={"data_root": str(DATA), "split": "validation"},
            n_startup_trials=10, max_total_trials=20, sampler="tpe", pruner="none",
            seed=42, device="cpu", verbose=0,
        )
        if cls.manager.setup_experiment() is not None:
            raise AssertionError("Expected native optimization setup, not formal training")
        cls.train_env = cls.manager.create_envs(1, no_log=True)
        cls.eval_env = cls.manager.create_envs(1, eval_env=True)
        cls.addClassCleanup(cls.train_env.close)
        cls.addClassCleanup(cls.eval_env.close)
        adapter = importlib.import_module(PACKAGE + ".zoo_optuna")
        cls.trial = FixedTrial({"tau": 0.02, "noise_std": 0.06})
        params = dict(cls.manager._hyperparams)
        params.update(adapter.sample_ddpg_params(cls.trial, cls.manager.n_actions, 1, {}))
        cls.model = DDPG(env=cls.train_env, seed=None, device="cpu", **params)
        cls.model.set_logger(configure(format_strings=[]))

    def test_fixed_configs_and_budget(self):
        formal = yaml.safe_load((HERE / "zoo_ddpg.yml").read_text())["AggregateShipDDPG-v0"]
        tuning = yaml.safe_load((HERE / "zoo_ddpg_optuna.yml").read_text())["AggregateShipDDPG-v0"]
        expected = dict(learning_rate=0.001, gamma=0.999, batch_size=64,
                        buffer_size=1000000, learning_starts=5000, train_freq=32,
                        gradient_steps=1, tau=0.005, policy="MlpPolicy",
                        noise_type="normal", noise_std=0.1, n_envs=1,
                        policy_kwargs="dict(net_arch=[256, 256])")
        for config in (formal, tuning):
            for key, value in expected.items():
                self.assertEqual(config[key], value, key)
            self.assertEqual(config["env_kwargs"]["split"], "train")
        self.assertEqual(formal["n_timesteps"], 500000)
        stop = formal["callback"][0]["stable_baselines3.common.callbacks.StopTrainingOnMaxEpisodes"]
        self.assertEqual(stop["max_episodes"], 484)
        self.assertEqual(tuning["n_timesteps"], 77760)
        self.assertEqual(self.manager.n_timesteps, 77760)
        self.assertEqual(self.manager.n_timesteps // self.manager.n_evaluations, 38880)
        lengths = sorted(len(e.load_kw) for e in self.train_env.envs[0].unwrapped.episodes)
        self.assertEqual((len(lengths), lengths[0], lengths[17], lengths[-1]),
                         (35, 3272, 9720, 36660))

    def test_actual_model_and_all_networks(self):
        m = self.model
        self.assertEqual(m.train_freq.frequency, 32)
        self.assertEqual(m.train_freq.unit.value, "step")
        self.assertEqual(m.gradient_steps, 1)
        self.assertEqual(m.learning_starts, 5000)
        self.assertEqual(m.batch_size, 64)
        self.assertEqual(m.buffer_size, 1000000)
        self.assertEqual(m.gamma, 0.999)
        self.assertEqual(m.lr_schedule(1), 0.001)
        self.assertEqual(m.tau, 0.02)
        self.assertEqual(m.n_envs, 1)
        for actor in (m.actor, m.actor_target):
            self.assertEqual([(x.in_features, x.out_features) for x in actor.mu if isinstance(x, nn.Linear)],
                             [(3, 256), (256, 256), (256, 1)])
            self.assertIsInstance(actor.mu[1], nn.ReLU)
            self.assertIsInstance(actor.mu[3], nn.ReLU)
            self.assertIsInstance(actor.mu[-1], nn.Tanh)
        for critic in (m.critic, m.critic_target):
            self.assertEqual(len(critic.q_networks), 1)
            self.assertEqual([(x.in_features, x.out_features) for x in critic.qf0 if isinstance(x, nn.Linear)],
                             [(4, 256), (256, 256), (256, 1)])
            self.assertIsInstance(critic.qf0[1], nn.ReLU)
            self.assertIsInstance(critic.qf0[3], nn.ReLU)
        self.assertIsInstance(m.actor.optimizer, torch.optim.Adam)
        self.assertIsInstance(m.critic.optimizer, torch.optim.Adam)
        self.assertEqual((m.num_timesteps, m._n_updates, m.replay_buffer.size()), (0, 0, 0))
        print("Actor / target:", m.actor.mu)
        print("Critic / target:", m.critic.qf0)
        print("Actual train_freq=32 steps; gradient_steps=1; Adam updates=0")

    def test_validation8_round_robin_and_split_isolation(self):
        train = self.train_env.envs[0].unwrapped
        validation = self.eval_env.envs[0].unwrapped
        train_ids = {e.sample_id for e in train.episodes}
        validation_ids = {e.sample_id for e in validation.episodes}
        self.assertEqual((train.split, validation.split), ("train", "validation"))
        self.assertEqual((len(train_ids), len(validation_ids)), (35, 8))
        self.assertFalse(train_ids & validation_ids)
        first = [train.reset(seed=42 if i == 0 else None)[1]["sample_id"] for i in range(8)]
        # Also check an offset cycle, as VecEnv auto-reset advances the cursor.
        ids = [validation.reset()[1]["sample_id"] for _ in range(17)]
        self.assertEqual(set(ids[:8]), validation_ids)
        self.assertEqual(set(ids[9:17]), validation_ids)
        second = [train.reset(seed=42 if i == 0 else None)[1]["sample_id"] for i in range(8)]
        self.assertEqual(first, second)
        self.assertTrue(self.manager.deterministic_eval)
        self.assertEqual((self.manager.n_eval_episodes, self.manager.n_eval_envs), (8, 1))
        print("Validation8 IDs:", ids[:8])

    def test_native_sampler_and_pruner(self):
        from optuna.pruners import NopPruner
        from optuna.samplers import TPESampler
        self.assertIsInstance(self.manager._create_sampler("tpe"), TPESampler)
        self.assertIsInstance(self.manager._create_pruner("none"), NopPruner)
        self.assertEqual((self.manager.n_startup_trials, self.manager.max_total_trials), (10, 20))

    def test_trial_parameter_logging_without_learning(self):
        from rl_zoo3.callbacks import TrialEvalCallback
        from rl_zoo3.utils import get_callback_list
        path = Path(self.temp.name) / "trial_0"
        eval_callback = TrialEvalCallback(self.eval_env, self.trial, n_eval_episodes=8,
                                         eval_freq=38880, deterministic=True,
                                         best_model_save_path=str(path), log_path=str(path))
        callbacks = get_callback_list({"callback": self.manager.specified_callbacks})
        self.assertEqual(len(callbacks), 1)
        callback_list = CallbackList([*callbacks, eval_callback])
        callback_list.init_callback(self.model)
        callback_list.on_training_start({"callback": callback_list}, {})
        saved = json.loads((path / "effective_parameters.json").read_text())
        self.assertEqual(saved["train_freq"], 32)
        self.assertEqual(saved["gradient_steps"], 1)
        self.assertEqual(saved["policy_kwargs"]["net_arch"], [256, 256])
        self.assertEqual(saved["tau"], 0.02)
        self.assertEqual(saved["noise_std"], [0.06])
        self.assertEqual(saved["evaluation_split"], ["validation"])
        self.assertEqual(saved, self.trial.user_attrs["effective_parameters"])
        self.assertEqual(set(self.trial.params), {"tau", "noise_std"})
        self.assertEqual((self.model.num_timesteps, self.model._n_updates), (0, 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
