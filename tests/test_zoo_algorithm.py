"""Native SB3 learn/Logger contracts for the Zoo vessel adapter; synthetic only."""
import importlib
import importlib.util

import gymnasium as gym
import numpy as np
import pytest
import torch

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Logger, configure
from stable_baselines3.common.vec_env import DummyVecEnv

from v6_sb3.masked_dqn import MaskedDQN


def adapter_class():
    name = 'experiments.sb3_manual.zoo_vessel.algorithm'
    assert importlib.util.find_spec(name) is not None, 'the ZooMaskedDQN adapter must exist'
    return importlib.import_module(name).ZooMaskedDQN


class MaskToyEnv(gym.Env):
    observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(5,), dtype=np.float32)
    action_space = gym.spaces.Discrete(13)

    def __init__(self, legal=2, length=4, failure=False):
        super().__init__()
        self.legal = legal
        self.length = length
        self.failure = failure
        self.total_steps = 0
        self.index = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.index = 0
        return np.zeros(5, np.float32), {}

    def action_masks(self):
        mask = np.zeros(13, dtype=bool)
        mask[self.legal] = True
        return mask

    def step(self, action):
        assert int(action) == self.legal, 'collection/evaluation selected an illegal action'
        self.index += 1
        self.total_steps += 1
        done = self.index == self.length
        info = dict(next_action_mask=np.zeros(13, dtype=bool) if done else self.action_masks(),
                    terminal_reason=('failure_soc_limited' if self.failure else 'completed') if done else None,
                    failure_penalty_equivalent_cny=0.)
        observation = np.zeros(5, np.float32)
        observation[4] = self.index
        return observation, -1. if self.failure and done else 1., done, False, info


def agent(**overrides):
    torch.set_num_threads(1)
    kwargs = dict(env=MaskToyEnv(), buffer_size=256, learning_starts=5, batch_size=4,
        train_freq=32, gradient_steps=1, target_update_interval=16,
        td_diagnostics=False, policy_kwargs=dict(net_arch=[8, 8]),
        exploration_initial_eps=1., exploration_final_eps=.1,
        exploration_fraction=1., seed=13, device='cpu')
    kwargs.update(overrides)
    return adapter_class()(**kwargs)


def test_native_learn_splits_nonmultiple_warmup_before_formal_32_to_1_updates():
    updates = []
    model = agent(after_update_observer=lambda m: updates.append(
        (m.num_timesteps, m.formal_env_steps, m._n_updates)))
    assert isinstance(model, MaskedDQN) and isinstance(model, DQN)
    model.learn(total_timesteps=69, log_interval=None)
    assert model.target_sync_unit == 'formal_env_steps'
    assert model.num_timesteps == model.replay_buffer.insertions == 69
    assert model.formal_env_steps == 64 and model._n_updates == 2
    assert model._n_calls == 69
    assert updates == [(37, 32, 1), (69, 64, 2)]
    assert model.target_sync_history == [dict(formal_env_steps=16, adam_updates=0),
        dict(formal_env_steps=32, adam_updates=0), dict(formal_env_steps=48, adam_updates=1),
        dict(formal_env_steps=64, adam_updates=1)]
    assert model.target_sync_calls == 5  # one initial copy plus four formal copies


def test_warmup_collection_is_masked_and_never_advances_formal_target_clock():
    model = agent(learning_starts=5, target_update_interval=1)
    model.learn(total_timesteps=5, log_interval=None)
    assert model.num_timesteps == model.replay_buffer.insertions == 5
    assert model.formal_env_steps == model._n_updates == 0
    assert model.target_sync_calls == 1 and model.target_sync_history == []
    assert np.all(model.replay_buffer.actions[:5] == 2)


class ExplorationTrace(BaseCallback):
    def __init__(self):
        super().__init__()
        self.values = []

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        self.values.append((self.model.num_timesteps,
            self.model._current_progress_remaining, self.model.exploration_rate))


def test_epsilon_uses_native_total_step_schedule_including_warmup():
    model = agent()
    trace = ExplorationTrace()
    model.learn(total_timesteps=69, callback=trace, log_interval=None)
    assert [row[0] for row in trace.values] == [5, 37, 69]
    assert trace.values[0][2] > trace.values[1][2] > trace.values[2][2]
    for _, progress, epsilon in trace.values:
        assert epsilon == model.exploration_schedule(progress)
    assert model.exploration_rate == pytest.approx(.1)


def test_native_logger_and_td_mae_dump_do_not_change_learning_counts(tmp_path):
    targets = []
    def observe(model, batch, q, target):
        targets.append(float((target - q).abs().mean().item()))
    model = agent(train_step_observer=observe)
    logger = configure(str(tmp_path), ['csv'])
    model.set_logger(logger)
    model.learn(total_timesteps=69, log_interval=None)
    assert isinstance(model.logger, Logger) and model.logger is logger
    assert model.train_step_observer is observe
    assert model.last_td_mae == targets[-1]
    assert logger.name_to_value['train/td_mae'] == targets[-1]
    assert logger.name_to_value['train/n_updates'] == 2
    assert 'tensorboard' not in logger.name_to_excluded['train/n_updates']
    counts = (model.num_timesteps, model.formal_env_steps, model._n_updates,
              model.replay_buffer.insertions, model.target_sync_calls)
    model.dump_logs()
    assert counts == (model.num_timesteps, model.formal_env_steps, model._n_updates,
                      model.replay_buffer.insertions, model.target_sync_calls)
    contents = (tmp_path / 'progress.csv').read_text(encoding='utf-8')
    assert 'train/td_mae' in contents and 'time/total_timesteps' in contents


def test_after_update_observer_sees_the_current_completed_minibatch_td_mae():
    actual = []
    observed_after_update = []
    model = agent(train_step_observer=lambda m, batch, q, target: actual.append(
        float((target - q).abs().mean().item())),
        after_update_observer=lambda m: observed_after_update.append(m.last_td_mae))
    model.learn(total_timesteps=69, log_interval=None)
    assert observed_after_update == actual


def test_eval_mask_context_uses_its_own_env_and_restores_training_source():
    train = DummyVecEnv([lambda: MaskToyEnv(legal=1)])
    evaluation = DummyVecEnv([lambda: MaskToyEnv(legal=9)])
    nested = DummyVecEnv([lambda: MaskToyEnv(legal=4)])
    model = agent(env=train)
    observation = train.reset()
    evaluation.reset()
    nested.reset()
    try:
        assert int(model.predict(observation, deterministic=True)[0][0]) == 1
        with model.evaluation_mask_source(evaluation):
            assert int(model.predict(observation, deterministic=True)[0][0]) == 9
            with model.evaluation_mask_source(nested):
                assert int(model.predict(observation, deterministic=True)[0][0]) == 4
            assert int(model.predict(observation, deterministic=True)[0][0]) == 9
        assert int(model.predict(observation, deterministic=True)[0][0]) == 1
        mask = np.zeros(13, bool)
        mask[7] = True
        assert int(model.predict(observation, deterministic=True, action_masks=mask)[0][0]) == 7
    finally:
        train.close()
        evaluation.close()
        nested.close()


def test_native_evaluate_policy_cannot_read_training_mask_or_change_training_state():
    train_toy = MaskToyEnv(legal=1)
    train = DummyVecEnv([lambda: train_toy])
    evaluation = DummyVecEnv([lambda: MaskToyEnv(legal=9, length=2)])
    model = agent(env=train)
    before = (model.num_timesteps, model.formal_env_steps, model._n_updates,
              model.replay_buffer.insertions, model.target_sync_calls)
    try:
        with model.evaluation_mask_source(evaluation):
            rewards, lengths = evaluate_policy(model, evaluation, n_eval_episodes=3,
                deterministic=True, return_episode_rewards=True, warn=False)
        assert rewards == [2., 2., 2.] and lengths == [2, 2, 2]
        assert train_toy.total_steps == 0
        assert before == (model.num_timesteps, model.formal_env_steps, model._n_updates,
                          model.replay_buffer.insertions, model.target_sync_calls)
    finally:
        train.close()
        evaluation.close()


def test_failure_replay_keeps_terminal_successor_mask_and_no_bootstrap():
    batches = []
    model = agent(env=MaskToyEnv(length=1, failure=True), learning_starts=0,
                  train_step_observer=lambda m, batch, q, target: batches.append(target.clone()))
    model.learn(total_timesteps=32, log_interval=None)
    assert model._n_updates == 1
    assert np.all(model.replay_buffer.dones[:32] == 1.)
    assert not model.replay_buffer.next_action_masks[:32].any()
    assert np.all(model.replay_buffer.rewards[:32] == -1.)
    assert np.all(model.replay_buffer.failure_penalties[:32] == 0.)
    assert torch.equal(batches[-1], torch.full_like(batches[-1], -1.))


def test_loaded_model_requires_explicit_mask_without_env_and_excludes_eval_source(tmp_path):
    model = agent()
    evaluation = DummyVecEnv([lambda: MaskToyEnv(legal=9)])
    try:
        evaluation.reset()
        with model.evaluation_mask_source(evaluation):
            model.save(tmp_path / 'model')
        loaded = adapter_class().load(tmp_path / 'model.zip', device='cpu')
        observation = np.zeros(5, np.float32)
        with pytest.raises(ValueError, match='mask|environment'):
            loaded.predict(observation, deterministic=True)
        mask = np.zeros(13, bool)
        mask[7] = True
        assert int(loaded.predict(observation, deterministic=True, action_masks=mask)[0]) == 7
    finally:
        evaluation.close()


def test_target_clock_cannot_be_silently_switched_to_optimizer_updates():
    with pytest.raises(ValueError, match='formal'):
        agent(target_sync_unit='adam_updates')


def test_episode_collection_frequency_is_rejected_in_step_clock_adapter():
    model = agent(train_freq=(1, 'episode'))
    with pytest.raises(ValueError, match='step'):
        model.learn(total_timesteps=5, log_interval=None)
