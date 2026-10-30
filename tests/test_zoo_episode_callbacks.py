"""Post-Replay, post-Adam Episode evaluation and native Logger contracts."""
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np
import optuna
import pandas as pd
import pytest
import torch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from experiments.sb3_manual.zoo_vessel import callbacks as module
from test_zoo_callbacks import TinyVoyages


class BoundaryModel:
    def __init__(self, folder, *, mode='episode', target=500, previous=None):
        self.logger = configure(str(folder), ['csv', 'tensorboard'])
        self.budget_mode = mode
        self.n_episodes = target if mode == 'episode' else None
        self.exploration_schedule_mode = 'episode'
        self.completed_formal_episodes = 0
        self.formal_successes = self.formal_failures = 0
        self.warmup_env_steps = 64
        self.warmup_bridge_env_steps = 3
        self.num_timesteps = 64
        self.formal_env_steps = self._n_updates = 0
        self.replay_buffer = SimpleNamespace(insertions=64, size=lambda:self.num_timesteps)
        self.exploration_rate = 1.
        self.last_training_loss = .2
        self.last_td_mae = .1
        self._total_timesteps = 10000000
        self.episode_boundary_observer = previous
        self.saved = []

    def boundary(self, completed):
        self.completed_formal_episodes = completed
        self.formal_successes = completed // 2
        self.formal_failures = completed - self.formal_successes
        self.formal_env_steps = completed * 32
        self.num_timesteps = self.replay_buffer.insertions = 64 + self.formal_env_steps
        self._n_updates = completed
        self.episode_boundary_observer(self)

    def save(self, path):
        self.saved.append((str(path), self.completed_formal_episodes, self._n_updates))

    def dump_logs(self):
        self.logger.record('time/total_timesteps', self.num_timesteps)
        self.logger.dump(step=self.num_timesteps)


def incomplete_summary(*args):
    return dict(episodes=8, completed=1, completion_rate=.125,
        mean_duration_coverage=.5, fixed_complete_set_cost_cny=None,
        score=1.25, environment_steps=16, samples=[], selection_split='validation')


def start_callback(callback, model):
    callback.model = model
    callback.locals = {}
    callback._on_training_start()


def test_every_50_and_final_500_evaluation_observes_stored_terminal_and_due_adam(tmp_path, monkeypatch):
    calls, previous_calls = [], []
    previous = lambda model:previous_calls.append(model.completed_formal_episodes)
    model = BoundaryModel(tmp_path / 'native', previous=previous)
    def evaluate(agent, env):
        calls.append((agent.completed_formal_episodes, agent.num_timesteps,
                      agent.replay_buffer.insertions, agent._n_updates))
        return incomplete_summary()
    monkeypatch.setattr(module, 'evaluate_all_validation', evaluate)
    callback = module.VesselEvalCallback(object(), eval_freq=1,
        best_model_save_path=tmp_path / 'best', log_path=tmp_path / 'validation')
    try:
        start_callback(callback, model)
        assert model.episode_boundary_observer is not previous
        for completed in range(1, 501):
            model.boundary(completed)
        model.boundary(500)  # Duplicate notifications cannot reevaluate terminal.
        assert [row[0] for row in calls] == list(range(50, 501, 50))
        assert all(stored == total and adam == count for count,total,stored,adam in calls)
        assert callback.last_summary['completed_formal_episodes'] == 500
        assert callback.last_summary['adam_updates'] == 500
        assert callback.last_summary['evaluation_position'] == 'after terminal Replay insertion and due Adam update'
        assert model.last_validation_summary == callback.last_summary
        callback._on_training_end()
        assert model.episode_boundary_observer is previous
        assert previous_calls == list(range(1,501)) + [500]
        assert model.saved[0][1:] == (50,50)
    finally:
        model.logger.close()


def test_episode_native_csv_tensorboard_and_final_metrics_keep_unknown_cost_blank(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(module, 'evaluate_all_validation', incomplete_summary)
    model = BoundaryModel(tmp_path / 'native', target=3)
    callback = module.VesselEvalCallback(object(), eval_every_episodes=50,
        log_path=tmp_path / 'validation', best_model_save_path=tmp_path / 'best')
    try:
        start_callback(callback, model)
        for count in (1,2,3):
            model.boundary(count)
        callback._on_training_end()
        frame = pd.read_csv(tmp_path / 'native' / 'progress.csv')
        for key in ('time/total_timesteps','train/completed_formal_episodes','train/formal_env_steps',
                    'train/adam_updates','train/warmup_env_steps','train/warmup_bridge_env_steps',
                    'train/formal_successes','train/formal_failures','train/success_rate',
                    'train/failure_rate','train/target_episodes','train/budget_mode',
                    'train/exploration_schedule_mode','rollout/exploration_rate',
                    'validation/completion_rate','validation/evaluated_formal_episodes'):
            assert key in frame.columns, key
        assert frame['train/completed_formal_episodes'].iloc[-1] == 3
        assert frame['train/budget_mode'].iloc[-1] == 'episode'
        assert frame['train/formal_successes'].iloc[-1] == 1
        assert frame['train/formal_failures'].iloc[-1] == 2
        assert frame['train/success_rate'].iloc[-1] == pytest.approx(1/3)
        assert frame['validation/economic_cost_cny'].isna().all()
        assert '| train/' in capsys.readouterr().out
        accumulator = EventAccumulator(str(tmp_path / 'native'))
        accumulator.Reload()
        scalars = accumulator.Tags()['scalars']
        assert 'train/completed_formal_episodes' in scalars
        assert 'validation/evaluated_formal_episodes' in scalars
        assert 'validation/economic_cost_cny' not in scalars
    finally:
        model.logger.close()


def test_timestep_budget_with_episode_exploration_keeps_timestep_validation_trigger(tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'evaluate_all_validation', incomplete_summary)
    previous = lambda model:None
    model = BoundaryModel(tmp_path / 'native', mode='timestep', previous=previous)
    callback = module.VesselEvalCallback(object(), eval_freq=2)
    try:
        start_callback(callback, model)
        assert model.episode_boundary_observer is previous
        callback.num_timesteps = 1
        callback.n_calls = 1
        assert callback._on_step()
        assert callback.last_summary is None
        callback.num_timesteps = 2
        callback.n_calls = 2
        assert callback._on_step()
        assert callback.last_summary['evaluation_position'] == 'native callback before current transition storage and next Adam'
        callback._on_training_end()
    finally:
        model.logger.close()


@pytest.mark.parametrize('target', [1,2])
def test_trial_prunes_only_at_post_storage_post_update_episode_boundary(tmp_path, monkeypatch, target):
    monkeypatch.setattr(module, 'evaluate_all_validation', incomplete_summary)
    study = optuna.create_study(pruner=optuna.pruners.ThresholdPruner(upper=0.))
    trial = study.ask()
    model = BoundaryModel(tmp_path / 'native', target=target)
    callback = module.VesselTrialEvalCallback(object(), trial=trial, eval_every_episodes=1)
    try:
        start_callback(callback, model)
        callback.num_timesteps = 95  # Callback executes before current storage.
        callback.n_calls = 95
        assert callback._on_step()  # Must not discard a terminal transition.
        with pytest.raises(optuna.TrialPruned):
            model.boundary(1)
        assert model.num_timesteps == model.replay_buffer.insertions == 96
        assert model._n_updates == 1
        assert callback.is_pruned
        assert trial.user_attrs['validation']['completed_formal_episodes'] == 1
        callback._on_training_end()
        frame = pd.read_csv(tmp_path / 'native' / 'progress.csv')
        assert frame['time/total_timesteps'].tolist() == [96]
        assert frame['summary/final_validation_completion_rate'].iloc[-1] == .125
    finally:
        model.logger.close()


def real_model(folder):
    from experiments.sb3_manual.zoo_vessel.algorithm import ZooMaskedDQN
    torch.set_num_threads(1)
    env = DummyVecEnv([lambda:Monitor(TinyVoyages())])
    model = ZooMaskedDQN(env=env, n_episodes=16, exploration_schedule_mode='episode',
        learning_starts=0, buffer_size=256, batch_size=8, train_freq=32,
        gradient_steps=1, target_update_interval=16, exploration_fraction=.8,
        seed=17, policy_kwargs=dict(net_arch=[8,8]),
        tensorboard_log=str(folder / 'tb'), verbose=1)
    return model, env


def test_native_learn_final_validation_follows_last_replay_and_full_32_adam_without_duplicate_dump(tmp_path, capsys):
    model, env = real_model(tmp_path)
    evaluation = DummyVecEnv([lambda:Monitor(TinyVoyages('validation'))])
    callback = module.VesselEvalCallback(evaluation, eval_every_episodes=8,
        log_every=32, best_model_save_path=tmp_path / 'best', log_path=tmp_path / 'validation')
    model.final_model_path = str(tmp_path / 'final_model.zip')
    model.best_model_path = str(tmp_path / 'best' / 'best_model.zip')
    try:
        model.learn(total_timesteps=256, callback=callback, log_interval=None)
        assert model.completed_formal_episodes == 16
        assert model.num_timesteps == model.replay_buffer.insertions == 32
        assert model._n_updates == 1
        assert [(row['completed_formal_episodes'], row['training_env_steps'], row['adam_updates'])
                for row in callback.evaluations] == [(8,16,0),(16,32,1)]
        assert callback.last_summary['completed'] == callback.last_summary['episodes'] == 8
        assert callback.last_summary['replay_insertions'] == 32
        assert model.episode_boundary_observer is None
        csv = next((tmp_path / 'tb').rglob('progress.csv'))
        frame = pd.read_csv(csv)
        assert frame['time/total_timesteps'].tolist() == [16,32]
        assert frame['train/adam_updates'].tolist() == [0,1]
        assert frame['replay/pending_transition'].tolist() == [0,0]
        assert frame['summary/target_sync_count'].iloc[-1] == 2
        assert frame['summary/target_sync_calls_including_initial'].iloc[-1] == 3
        assert frame['summary/final_epsilon'].iloc[-1] == pytest.approx(.05)
        assert frame['summary/final_validation_completion_rate'].iloc[-1] == 1.
        assert frame['summary/final_model_path'].iloc[-1] == model.final_model_path
        assert frame['summary/best_model_path'].iloc[-1] == model.best_model_path
        assert pd.isna(frame['summary/target_sync_count'].iloc[0])
        accumulator = EventAccumulator(str(csv.parent))
        accumulator.Reload()
        assert accumulator.Scalars('summary/target_sync_count')[-1].value == 2
        assert accumulator.Scalars('summary/target_sync_calls_including_initial')[-1].value == 3
        assert accumulator.Scalars('summary/final_epsilon')[-1].value == pytest.approx(.05)
        assert accumulator.Scalars('summary/final_validation_completion_rate')[-1].value == 1.
        assert 'summary/final_model_path/text_summary' in accumulator.Tags()['tensors']
        assert 'summary/best_model_path/text_summary' in accumulator.Tags()['tensors']
        output = capsys.readouterr().out
        assert '| time/' in output and '| train/' in output
        model.save(tmp_path / 'final_model.zip')
        assert (tmp_path / 'best' / 'best_model.zip').exists()
        assert (tmp_path / 'final_model.zip').exists()
        assert model.target_sync_history == [dict(formal_env_steps=16, adam_updates=0),
                                              dict(formal_env_steps=32, adam_updates=0)]
    finally:
        model.logger.close(); env.close(); evaluation.close()


def test_episode_validation_and_native_logging_preserve_learning_and_rng(tmp_path):
    baseline, baseline_env = real_model(tmp_path / 'baseline')
    baseline.learn(total_timesteps=256, log_interval=None)
    baseline_rng = pickle.dumps(np.random.get_state())
    baseline_torch = torch.get_rng_state().clone()
    observed, env = real_model(tmp_path / 'observed')
    evaluation = DummyVecEnv([lambda:Monitor(TinyVoyages('validation'))])
    callback = module.VesselEvalCallback(evaluation, eval_every_episodes=8, log_every=8)
    try:
        observed.learn(total_timesteps=256, callback=callback, log_interval=None)
        assert baseline_rng == pickle.dumps(np.random.get_state())
        assert torch.equal(baseline_torch, torch.get_rng_state())
        assert baseline.num_timesteps == observed.num_timesteps == 32
        assert baseline._n_updates == observed._n_updates == 1
        assert baseline.completed_formal_episodes == observed.completed_formal_episodes == 16
        assert baseline.target_sync_history == observed.target_sync_history
        for name in ('actions','rewards','dones','next_action_masks'):
            np.testing.assert_array_equal(getattr(baseline.replay_buffer, name)[:32],
                getattr(observed.replay_buffer, name)[:32])
        for name, value in baseline.policy.state_dict().items():
            assert torch.equal(value, observed.policy.state_dict()[name]), name
    finally:
        baseline.logger.close(); observed.logger.close()
        baseline_env.close(); env.close(); evaluation.close()


def test_final_progress_without_validation_keeps_unknown_rate_none(tmp_path):
    model = BoundaryModel(tmp_path / 'native')
    callback = module.VesselProgressCallback()
    try:
        start_callback(callback, model)
        callback._on_training_end()
        frame = pd.read_csv(tmp_path / 'native' / 'progress.csv')
        assert frame['summary/final_validation_completion_rate'].isna().all()
        accumulator = EventAccumulator(str(tmp_path / 'native'))
        accumulator.Reload()
        assert 'summary/final_validation_completion_rate' not in accumulator.Tags()['scalars']
    finally:
        model.logger.close()
