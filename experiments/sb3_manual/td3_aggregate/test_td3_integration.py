"""Bounded TD3 acceptance checks; never run full voyages or access Test4."""
from __future__ import annotations

import importlib
import importlib.util
import csv
import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
PACKAGE = "experiments.sb3_manual.td3_aggregate"
DATA = ROOT / "data/processed/operating_dataset_v6_1s"
FIXED_ID = "parent_005_seg01"
_TRAIN_DATASET = None


class CachedTrainCase(unittest.TestCase):
    """Reuse loaded real Train35 arrays to keep acceptance I/O bounded."""

    @classmethod
    def setUpClass(cls):
        global _TRAIN_DATASET
        from v4.v6_dataset import V6TrainingDataset
        if _TRAIN_DATASET is None:
            _TRAIN_DATASET = V6TrainingDataset.open(DATA)
            _TRAIN_DATASET.load_train()
        cls.dataset = _TRAIN_DATASET

    def setUp(self):
        from v4.v6_dataset import V6TrainingDataset
        self.dataset_patch = patch.object(V6TrainingDataset, "open", return_value=self.dataset)
        self.dataset_patch.start()
        self.addCleanup(self.dataset_patch.stop)

    def make_env(self):
        env = importlib.import_module(PACKAGE + ".reward_wrapper").make_env(
            DATA, fixed_train_id=FIXED_ID)
        self.addCleanup(env.close)
        return env


class ScaledEconomicRewardTests(CachedTrainCase):
    def wrapper_module(self):
        name = PACKAGE + ".reward_wrapper"
        self.assertIsNotNone(importlib.util.find_spec(name), "TD3 reward wrapper is missing")
        return importlib.import_module(name)

    def pair(self):
        from experiments.sb3_manual.ddpg_aggregate.env import AggregateDDPGEnv
        module = self.wrapper_module()
        raw = AggregateDDPGEnv(DATA, fixed_train_id=FIXED_ID)
        scaled = module.make_env(DATA, fixed_train_id=FIXED_ID)
        self.addCleanup(raw.close)
        self.addCleanup(scaled.close)
        return raw, scaled

    def test_train_sequence_scales_only_reward_preserving_physics_and_cny(self):
        raw, scaled = self.pair()
        raw_obs, raw_info = raw.reset(seed=42)
        obs, info = scaled.reset(seed=42)
        np.testing.assert_array_equal(obs, raw_obs)
        self.assertEqual(info["sample_id"], raw_info["sample_id"])
        for action in (-1.0, 0.0, 1.0, -0.3, 0.5):
            with self.subTest(action=action):
                raw_obs, raw_reward, raw_done, raw_truncated, raw_info = raw.step([action])
                obs, reward, done, truncated, info = scaled.step([action])
                np.testing.assert_array_equal(obs, raw_obs)
                self.assertEqual((done, truncated), (raw_done, raw_truncated))
                self.assertEqual(reward, raw_reward / 10000.0)
                self.assertEqual(info["raw_reward"], raw_reward)
                self.assertEqual(info["scaled_reward"], reward)
                self.assertEqual(info["reward_scale"], 0.0001)
                for key, value in raw_info.items():
                    self.assertEqual(info[key], value, key)
                self.assertEqual(scaled.unwrapped.physical, raw.physical)
                self.assertEqual(scaled.unwrapped.economic_totals, raw.economic_totals)

    def test_infeasible_physics_has_scaled_failure_and_separate_episode_returns(self):
        from v3.control import AccountState, SOC_MIN
        raw, scaled = self.pair()
        for env in (raw, scaled):
            env.reset(seed=42)
            base = env.unwrapped
            base.index = int(np.flatnonzero(base.episode.load_kw > 600.0)[0])
            base.physical = AccountState(soc=SOC_MIN, previous_fc_kw=0.0)
        raw_obs, raw_reward, _, _, raw_info = raw.step([0.0])
        obs, reward, done, truncated, info = scaled.step([0.0])
        np.testing.assert_array_equal(obs, raw_obs)
        self.assertEqual((raw_reward, reward), (-20000.0, -2.0))
        self.assertTrue(done)
        self.assertFalse(truncated)
        self.assertFalse(info["step_executed"])
        self.assertEqual(info["terminal_reason"], "no_feasible_physical_power")
        self.assertEqual(info["vessel_episode"]["reward_sum"], -20000.0)
        self.assertEqual(info["vessel_episode"]["algorithm_reward_sum"], -2.0)
        for key, value in raw_info["vessel_episode"].items():
            self.assertEqual(info["vessel_episode"][key], value, key)
        self.assertEqual(scaled.unwrapped.economic_totals, raw.economic_totals)

    def test_terminal_shutdown_scales_once_and_reset_clears_algorithm_return(self):
        from v3.control import AccountState
        raw, scaled = self.pair()
        raw.reset(seed=42)
        scaled.reset(seed=42)
        rewards = []
        for action in (-1.0, 0.25):
            _, raw_reward, _, _, _ = raw.step([action])
            _, reward, _, _, _ = scaled.step([action])
            self.assertEqual(reward, raw_reward / 10000.0)
            rewards.append(reward)
        for env in (raw, scaled):
            env.unwrapped.index = len(env.unwrapped.episode.load_kw) - 1
            env.unwrapped.physical = AccountState(soc=0.6, previous_fc_kw=300.0)
        _, raw_reward, raw_done, _, raw_info = raw.step([0.0])
        _, reward, done, truncated, info = scaled.step([0.0])
        rewards.append(reward)
        self.assertTrue(raw_done and done)
        self.assertFalse(truncated)
        self.assertTrue(info["shutdown_executed"])
        self.assertEqual(reward, raw_reward / 10000.0)
        self.assertEqual(info["vessel_episode"]["reward_sum"], raw.reward_sum)
        self.assertAlmostEqual(info["vessel_episode"]["algorithm_reward_sum"], sum(rewards), places=15)
        self.assertEqual(info["shutdown_economic_cost_cny"], raw_info["shutdown_economic_cost_cny"])
        self.assertEqual(scaled.unwrapped.economic_totals, raw.economic_totals)
        scaled.reset(seed=42)
        scaled.unwrapped.index = len(scaled.unwrapped.episode.load_kw) - 1
        _, reward, _, _, info = scaled.step([-1.0])
        self.assertEqual(info["vessel_episode"]["algorithm_reward_sum"], reward)

    def test_wrapper_identity_does_not_relabel_underlying_ddpg_reward(self):
        module = self.wrapper_module()
        raw, scaled = self.pair()
        self.assertIsInstance(scaled, gym.RewardWrapper)
        self.assertEqual(module.REWARD_SCALE, 0.0001)
        self.assertEqual(scaled.reward_identity["reward_version"], module.TD3_REWARD_VERSION)
        self.assertEqual(scaled.reward_identity["base_reward_version"], raw.reward_identity["reward_version"])
        self.assertEqual(scaled.reward_identity["algorithm"], "TD3")
        self.assertEqual(scaled.reward_identity["critic_loss"], "sb3_native_mse")
        self.assertEqual(scaled.reward_identity["reward_scale"], 0.0001)
        self.assertEqual(scaled.unwrapped.reward_identity, raw.reward_identity)


class NativeTD3IntegrationTests(CachedTrainCase):
    def test_official_zoo_builds_native_td3_from_formal_config(self):
        import argparse
        from rl_zoo3.exp_manager import ExperimentManager
        from rl_zoo3.utils import ALGOS
        from stable_baselines3 import TD3
        importlib.import_module(PACKAGE + ".env_registration")
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        with tempfile.TemporaryDirectory(prefix="td3_zoo_setup_") as tmp:
            manager = ExperimentManager(
                argparse.Namespace(), "td3", "AggregateShipTD3-v0", log_folder=tmp,
                config=str(Path(__file__).with_name("zoo_td3.yml")),
                n_timesteps=64, eval_freq=-1, device="cpu", seed=42, verbose=0)
            model, _ = manager.setup_experiment()
            try:
                self.assertIs(ALGOS["td3"], TD3)
                self.assertIs(type(model), TD3)
                self.assertEqual((model.learning_starts, model.batch_size, model.buffer_size), (1000, 32, 1_000_000))
                self.assertEqual((model.policy_delay, model.target_policy_noise, model.target_noise_clip), (2, 0.2, 0.5))
                self.assertEqual(model.policy.net_arch, {"pi": [256, 256], "qf": [256, 256]})
                self.assertEqual(model.train_freq.frequency, 32)
                self.assertEqual(model.gradient_steps, 1)
                self.assertEqual(model.get_env().get_attr("fixed_train_id"), [FIXED_ID])
                self.assertEqual(model.get_env().get_attr("split"), ["train"])
                self.assertEqual((model.num_timesteps, model._n_updates, model.replay_buffer.size()), (0, 0, 0))
                np.testing.assert_allclose(model.action_noise._sigma, [0.1])
            finally:
                model.get_env().close()

    def test_registered_env_has_original_spaces_and_passes_native_gym_checks(self):
        from stable_baselines3.common.env_checker import check_env
        importlib.import_module(PACKAGE + ".env_registration")
        env = gym.make("AggregateShipTD3-v0", data_root=DATA, fixed_train_id=FIXED_ID)
        self.addCleanup(env.close)
        self.assertEqual(env.observation_space.shape, (5,))
        self.assertEqual(env.action_space.shape, (1,))
        self.assertEqual(env.action_space.dtype, np.float32)
        np.testing.assert_array_equal(env.action_space.low, [-1.0])
        np.testing.assert_array_equal(env.action_space.high, [1.0])
        observation, info = env.reset(seed=42)
        self.assertTrue(env.observation_space.contains(observation))
        self.assertEqual((info["sample_id"], info["split"]), (FIXED_ID, "train"))
        check_env(env, warn=True)
        with self.assertRaises(ValueError):
            importlib.import_module(PACKAGE + ".reward_wrapper").make_env(DATA, split="test")

    def test_formal_yaml_keeps_requested_td3_parameters_and_train_only(self):
        config = yaml.safe_load(Path(__file__).with_name("zoo_td3.yml").read_text(encoding="utf-8"))[
            "AggregateShipTD3-v0"]
        expected = dict(policy="MlpPolicy", n_envs=1, learning_rate=0.001, gamma=1.0,
                        buffer_size=1_000_000, batch_size=32, tau=0.001,
                        learning_starts=1000, train_freq=32, gradient_steps=1,
                        policy_delay=2, target_policy_noise=0.2, target_noise_clip=0.5,
                        noise_type="normal", noise_std=0.1)
        for name, value in expected.items():
            self.assertEqual(config[name], value, name)
        self.assertEqual(config["env_kwargs"], dict(data_root="data/processed/operating_dataset_v6_1s",
                                                  split="train", fixed_train_id=FIXED_ID))
        self.assertGreaterEqual(config["n_timesteps"], 500 * 9258)
        self.assertFalse(config.get("normalize", False))
        self.assertEqual(config["policy_kwargs"], {"net_arch": {"pi": [256, 256], "qf": [256, 256]}})
        callbacks = {name: kwargs for row in config["callback"] for name, kwargs in row.items()}
        diagnostics = callbacks[PACKAGE + ".callbacks.TD3TrainingCallback"]
        self.assertEqual(diagnostics["fixed_eval_every_episodes"], 20)
        self.assertEqual(diagnostics["validation_every_episodes"], 100)
        self.assertEqual(diagnostics["max_episodes"], 500)
        self.assertEqual(callbacks["stable_baselines3.common.callbacks.StopTrainingOnMaxEpisodes"]["max_episodes"], 500)

    def test_real_train_64_steps_update_native_critics_actor_targets_and_roundtrip(self):
        from stable_baselines3 import TD3
        from stable_baselines3.common.callbacks import ConvertCallback
        from stable_baselines3.common.noise import NormalActionNoise
        from v4.v6_dataset import V6TrainingDataset
        module = importlib.import_module(PACKAGE + ".reward_wrapper")
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        env = self.make_env()
        train_ids = set(self.dataset.split_episode_ids("train"))
        validation_ids = set(self.dataset.split_episode_ids("validation"))
        self.assertEqual((len(train_ids), len(validation_ids)), (35, 8))
        self.assertFalse(train_ids & validation_ids)
        self.assertTrue(all(ep.split == "train" for ep in env.unwrapped.episodes))
        model = TD3("MlpPolicy", env, learning_rate=0.001, gamma=1.0, buffer_size=512,
                    batch_size=32, tau=0.001, learning_starts=32, train_freq=1,
                    gradient_steps=1, policy_delay=2, target_policy_noise=0.2,
                    target_noise_clip=0.5, action_noise=NormalActionNoise(np.zeros(1), np.full(1, 0.1)),
                    policy_kwargs=dict(net_arch=[256, 256]), seed=42, device="cpu", verbose=0)
        temporary = tempfile.TemporaryDirectory(prefix="td3_acceptance_")
        self.addCleanup(temporary.cleanup)
        self.addCleanup(lambda: model.logger.close())
        directory = Path(temporary.name)
        from .callbacks import TD3TrainingCallback
        diagnostics = TD3TrainingCallback(output_root=directory / "training", verbose=0)
        self.assertIs(type(model), TD3)
        self.assertIs(model.train.__func__, TD3.train)
        self.assertEqual(len(model.critic.q_networks), 2)
        self.assertEqual(len(model.critic_target.q_networks), 2)
        actor_layers = [layer.out_features for layer in model.actor.mu if isinstance(layer, torch.nn.Linear)]
        self.assertEqual(actor_layers, [256, 256, 1])
        for critic in model.critic.q_networks:
            self.assertEqual([layer.out_features for layer in critic if isinstance(layer, torch.nn.Linear)],
                             [256, 256, 1])
        modules = (model.actor, model.critic, model.actor_target, model.critic_target)
        before = [[p.detach().clone() for p in network.parameters()] for network in modules]
        observed_ids = []

        def observe(callback_locals, _globals):
            observed_ids.extend(info["sample_id"] for info in callback_locals["infos"])
            return True

        from stable_baselines3.td3 import td3 as native_td3
        with patch.object(V6TrainingDataset, "load_validation", side_effect=AssertionError("Validation cannot train")), \
                patch.object(model.actor.optimizer, "step", wraps=model.actor.optimizer.step) as actor_step, \
                patch.object(model.critic.optimizer, "step", wraps=model.critic.optimizer.step) as critic_step, \
                patch.object(native_td3.F, "mse_loss", wraps=native_td3.F.mse_loss) as mse:
            model.learn(total_timesteps=64, callback=[ConvertCallback(observe), diagnostics], log_interval=-2)
            self.assertEqual((model.num_timesteps, model._n_updates, model.replay_buffer.size()), (64, 32, 64))
            self.assertEqual((critic_step.call_count, actor_step.call_count), (32, 16))
            self.assertEqual(mse.call_count, 64)
        self.assertEqual(len(observed_ids), 64)
        self.assertTrue(set(observed_ids) <= train_ids)
        self.assertFalse(set(observed_ids) & validation_ids)
        for network, snapshots in zip(modules, before):
            self.assertTrue(any(not torch.equal(p, old) for p, old in zip(network.parameters(), snapshots)))
        with (directory / "training/progress.csv").open(encoding="utf-8") as handle:
            records = list(csv.DictReader(handle))
        for key in ("train/actor_loss", "train/critic_loss", "train/n_updates",
                    "actor/action_mean", "actor/action_std", "actor/action_saturation_fraction",
                    "critic/q1_mean", "critic/q2_mean", "time/fps"):
            self.assertTrue(np.isfinite(float(records[-1][key])), key)
        self.assertEqual(float(records[-1]["replay/buffer_size"]), 64.0)
        self.assertEqual(float(records[-1]["vessel/completion_rate"]), 0.0)
        self.assertEqual(model._td3_successful_episodes, 0)
        metadata = json.loads((directory / "training/td3_training_metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["reward_identity"], env.reward_identity)
        self.assertEqual(metadata["reward_scale"], 0.0001)
        module.assert_model_identity(model, env.reward_identity)
        observation, _ = env.reset(seed=42)
        action, _ = model.predict(observation, deterministic=True)
        path = directory / "model.zip"
        model.save(path)
        loaded = TD3.load(path, device="cpu", action_noise=None)
        module.assert_model_identity(loaded, env.reward_identity)
        self.assertEqual(loaded._td3_reward_identity, env.reward_identity)
        self.assertEqual(loaded.reward_scale, 0.0001)
        self.assertIsNone(loaded.action_noise)
        loaded_action, _ = loaded.predict(observation, deterministic=True)
        np.testing.assert_array_equal(loaded_action, action)

    def test_incompatible_checkpoint_reward_identity_and_scale_are_rejected(self):
        env = self.make_env()
        module = importlib.import_module(PACKAGE + ".reward_wrapper")
        identity = dict(env.reward_identity)
        cases = (
            SimpleNamespace(_ddpg_reward_identity=identity, reward_scale=0.0001),
            SimpleNamespace(_td3_reward_identity={**identity, "reward_version": "old"}, reward_scale=0.0001),
            SimpleNamespace(_td3_reward_identity={**identity, "reward_scale": 1.0}, reward_scale=0.0001),
            SimpleNamespace(_td3_reward_identity=identity, reward_scale=1.0),
            SimpleNamespace(_td3_reward_identity=identity),
        )
        for model in cases:
            with self.subTest(model=model), self.assertRaises(ValueError):
                module.assert_model_identity(model, identity)


class CallbackProtocolTests(CachedTrainCase):
    def test_episode_20_100_500_protocol_is_cpu_isolated_and_counts_failures(self):
        from stable_baselines3.common.callbacks import CallbackList, StopTrainingOnMaxEpisodes
        from stable_baselines3.common.logger import Logger
        from .callbacks import ECONOMIC_FIELDS, TD3TrainingCallback
        env = self.make_env()
        with tempfile.TemporaryDirectory(prefix="td3_callback_") as tmp:
            directory = Path(tmp)
            replay = SimpleNamespace(size=lambda: 0, storage=np.array([1.0, 2.0]))
            model = SimpleNamespace(num_timesteps=0, _episode_num=0, _n_updates=0,
                                    _last_obs=None, replay_buffer=replay, seed=42,
                                    logger=Logger(None, []), weights=np.array([0.5, 1.0]))
            attributes = dict(fixed_train_id=FIXED_ID, split="train",
                              reward_identity=env.reward_identity, data_root=DATA)
            vector = SimpleNamespace(num_envs=1, get_attr=lambda name: [attributes[name]])
            model.get_env = lambda: vector
            model.set_logger = lambda logger: setattr(model, "logger", logger)
            model.save = lambda path: Path(path).write_bytes(b"mock TD3 checkpoint")
            diagnostics = TD3TrainingCallback(output_root=directory, verbose=0)
            callbacks = CallbackList([diagnostics, StopTrainingOnMaxEpisodes(500)])
            callbacks.init_callback(model)
            with patch(PACKAGE + ".callbacks.configure", return_value=Logger(None, [])):
                callbacks.on_training_start({}, {})
            self.assertEqual(model._td3_reward_identity, env.reward_identity)
            self.assertEqual(model.reward_scale, 0.0001)
            model._n_updates = 17
            calls = {"fixed_train_eval": [], "validate": []}

            def subprocess_result(command, **options):
                module = command[command.index("-m") + 1]
                entry = module.rsplit(".", 1)[-1]
                self.assertIn(entry, calls)
                self.assertEqual(module, PACKAGE + "." + entry)
                self.assertEqual(command[command.index("--device") + 1], "cpu")
                self.assertNotIn("--max-steps", command)
                self.assertNotIn("test4", " ".join(command).lower())
                self.assertTrue(options["check"])
                node = Path(command[command.index("--output-dir") + 1])
                metadata = json.loads((node / "checkpoint_metadata.json").read_text(encoding="utf-8"))
                self.assertFalse(metadata["replay_buffer_saved"])
                self.assertEqual(metadata["reward_scale"], 0.0001)
                self.assertEqual(metadata["critic_updates"], 17)
                self.assertIn("before_replay", metadata["clock_phase"])
                self.assertTrue(Path(command[command.index("--model-path") + 1]).is_file())
                episode = metadata["completed_episodes"]
                calls[entry].append(episode)
                summary = dict(sample_id=FIXED_ID, split="train", completed=False,
                               total_steps=9258, evaluation_truncated=False, coverage=0.5,
                               soc_min=0.4, soc_final=0.5, reward_sum=-20000.0,
                               algorithm_reward_sum=-2.0, **dict.fromkeys(ECONOMIC_FIELDS, 1.0))
                metrics = dict(deterministic=True, action_noise=None, reward_scale=0.0001,
                               integrity={"parameters_unchanged": True}, split="train")
                if entry == "fixed_train_eval":
                    metrics["summary"] = summary
                    filename = "fixed_train_metrics.json"
                    self.assertEqual(command[command.index("--sample-id") + 1], FIXED_ID)
                else:
                    self.assertEqual(command[command.index("--episodes") + 1], "8")
                    metrics.update(split="validation", validation_set_complete=True,
                                   completion_rate=0.0, mean_coverage=0.5,
                                   per_voyage=[{**summary, "sample_id": sample_id, "split": "validation"}
                                               for sample_id in self.dataset.split_episode_ids("validation")])
                    filename = "validation_metrics.json"
                (node / filename).write_text(json.dumps(metrics), encoding="utf-8")
                return SimpleNamespace(returncode=0)

            python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
            with patch(PACKAGE + ".callbacks.subprocess.run", side_effect=subprocess_result):
                for episode in range(1, 501):
                    for _ in range(episode % 3 + 1):
                        model.num_timesteps += 1
                        info = dict(raw_reward=0.0, scaled_reward=0.0, requested_fc_kw=300.0,
                                    step_executed=False, soc=0.5)
                        callbacks.update_locals({"dones": np.array([False]), "infos": [info]})
                        self.assertTrue(callbacks.on_step())
                    model.num_timesteps += 1
                    failed = episode % 7 == 0
                    summary = dict(sample_id=FIXED_ID, split="train", reward_sum=-20000.0,
                                   algorithm_reward_sum=-2.0, executed_steps=1, total_steps=9258,
                                   completed=not failed, coverage=1/9258, soc_min=0.4, soc_final=0.5,
                                   power_projection_fraction=0.0,
                                   terminal_reason="no_feasible_physical_power" if failed else "completed")
                    info = dict(raw_reward=-20000.0, scaled_reward=-2.0, requested_fc_kw=300.0,
                                step_executed=False, soc=0.5, vessel_episode=summary)
                    callbacks.update_locals({"dones": np.array([True]), "infos": [info]})
                    self.assertEqual(callbacks.on_step(), episode < 500)
            self.assertEqual(calls["fixed_train_eval"], list(range(20, 501, 20)))
            self.assertEqual(calls["validate"], list(range(100, 501, 100)))
            self.assertEqual(model._td3_completed_episodes, 500)
            self.assertEqual(model._td3_successful_episodes, 500 - 500 // 7)
            self.assertIs(model.replay_buffer, replay)
            np.testing.assert_array_equal(replay.storage, [1.0, 2.0])
            np.testing.assert_array_equal(model.weights, [0.5, 1.0])
            self.assertEqual(model._n_updates, 17)
            self.assertEqual(random.getstate(), python_rng)
            np.testing.assert_array_equal(np.random.get_state()[1], numpy_rng[1])
            self.assertEqual(np.random.get_state()[2:], numpy_rng[2:])
            self.assertTrue(torch.equal(torch.get_rng_state(), torch_rng))
            with (directory / "training_episodes.csv").open(encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 500)
            self.assertEqual(rows[6]["completed"], "False")
            self.assertEqual(float(rows[-1]["cumulative_completion_rate"]), (500 - 500 // 7) / 500)


class OfficialZooResumeGuardTests(unittest.TestCase):
    def test_registration_rejects_unmarked_warmup_checkpoint_before_loading(self):
        from zipfile import ZipFile
        registration = importlib.import_module(PACKAGE + ".env_registration")
        with tempfile.TemporaryDirectory(prefix="td3_resume_guard_") as temp:
            checkpoint = Path(temp) / "old_warmup.zip"
            with ZipFile(checkpoint, "w") as archive:
                archive.writestr("data", json.dumps({"num_timesteps": 32, "_n_updates": 0}))
            with patch.object(sys, "argv", ["rl_zoo3.train", "--trained-agent", str(checkpoint)]):
                with self.assertRaisesRegex(ValueError, "reward identity"):
                    importlib.reload(registration)


if __name__ == "__main__":
    unittest.main(verbosity=2)
