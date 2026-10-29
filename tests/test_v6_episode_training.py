from pathlib import Path
from types import SimpleNamespace
import json

import gymnasium as gym
import numpy as np
import pytest
import torch


class ToyVesselEnv(gym.Env):
    """Real Gym transitions with deterministic costs and stochastic legal actions."""
    observation_space = gym.spaces.Box(-np.inf, np.inf, (5,), dtype=np.float32)
    action_space = gym.spaces.Discrete(13)

    def __init__(self, **kwargs):
        self.episode = None
        self.index = 0
        self.episode_status = None

    def observation(self):
        return np.array([.6-self.index*.0001, self.index/100, 0, 0, 0], np.float32)

    def reset(self, *, seed=None, options=None):
        self.episode = options['episode']
        self.index = 0
        self.episode_status = None
        self.current_decision_state = self.observation()
        return self.observation(), {}

    def action_masks(self):
        return np.full(13, self.episode_status is None)

    def step(self, action):
        self.index += 1
        done = self.index == self.episode.step_count
        self.episode_status = 'completed' if done else None
        cost = 1. + int(action)*.01
        self.last_transition = SimpleNamespace(duration_s=1., actual_soc=.6-self.index*.0001,
            soc_equivalent_hydrogen_cost_cny=.2, executed_ledger=SimpleNamespace(h2_cost_cny=cost-.2,
                fuel_cell_degradation_cost_cny=0., battery_degradation_cost_cny=0., shore_cost_cny=0.))
        self.current_decision_state = self.observation()
        return self.observation(), -cost, done, False, dict(next_action_mask=self.action_masks(),
            terminal_reason='completed' if done else None, failure_penalty_equivalent_cny=0.)

    def snapshot(self):
        return dict(index=self.index, episode_status=self.episode_status)

    def restore(self, snapshot, episode):
        self.episode = episode
        self.index = snapshot['index']
        self.episode_status = snapshot['episode_status']
        self.current_decision_state = self.observation()
        return self.observation(), {}


@pytest.fixture
def setup_runner(monkeypatch):
    from v6_sb3 import episode_training as runner
    monkeypatch.setattr(runner, '_make_env', lambda config: ToyVesselEnv())
    train = tuple(SimpleNamespace(sample_id=f'train{i}', split='train', step_count=n,
        duration_s=float(n), load_kw=np.ones(n), dt_s=np.ones(n)) for i, n in enumerate((17, 21, 33)))
    validation = (SimpleNamespace(sample_id='validation', split='validation', step_count=7,
        duration_s=7., load_kw=np.ones(7), dt_s=np.ones(7)),)
    config = dict(episodes=5, learning_starts=64, buffer_size=512, network_width=8,
        batch_size=8, target_update_interval=40, validation_every=2, checkpoint_keep=2,
        seed=7, order_seed=19, smoke=True)
    return runner, train, validation, config


def test_cadence_complete_episodes_validation_and_bounded_logs(setup_runner, tmp_path):
    runner, train, validation, config = setup_runner
    result = runner.run_episode_training(config, train, validation, tmp_path/'run')
    assert result['status'] == 'complete'
    assert result['completed_episodes'] == 5
    assert result['warmup_env_steps'] == 64
    assert result['adam_updates'] == result['formal_env_steps']//32
    assert result['total_env_steps'] == 64+result['formal_env_steps']
    assert result['replay_insertions'] == result['total_env_steps']
    assert result['validation_history'][-1]['episode'] == 5
    assert result['validation_history'][-1]['completed'] == 1
    assert len(list((tmp_path/'run'/'checkpoints').glob('checkpoint_*'))) == 2
    rows = [json.loads(line) for line in (tmp_path/'run'/'episodes.jsonl').read_text().splitlines()]
    assert len(rows) == 5
    assert any(row['loss_mean'] is None and row['td_mae'] is None for row in rows)
    assert sum(row['episode_env_steps'] for row in rows) == result['formal_env_steps']
    assert len(result['sample_order']) == 5
    assert sum(result['sample_counts'].values()) == 5
    assert result['target_sync_history'] == [dict(formal_env_steps=i, adam_updates=(i-1)//32)
        for i in range(40, result['formal_env_steps']+1, 40)]


@pytest.mark.parametrize('pause_steps', [1, 64, 64+17, 64+32, 64+35])
def test_exact_resume_has_no_lost_or_repeated_transition_adam_or_episode(setup_runner, tmp_path, pause_steps):
    runner, train, validation, config = setup_runner
    complete = runner.run_episode_training(config, train, validation, tmp_path/'complete')
    pause = runner.run_episode_training(config, train, validation, tmp_path/'pause', stop_after_steps=pause_steps)
    assert pause['status'] == 'paused'
    result = runner.run_episode_training(config, train, validation, tmp_path/'pause', resume=True)
    for field in ('formal_env_steps', 'warmup_env_steps', 'adam_updates', 'sample_order', 'sample_counts',
                  'target_sync_history', 'validation_history', 'replay_insertions', 'best_score'):
        assert result[field] == complete[field], field
    from v6_sb3.masked_dqn import MaskedDQN
    a = MaskedDQN.load(Path(complete['latest_checkpoint'])/'model.zip')
    b = MaskedDQN.load(Path(result['latest_checkpoint'])/'model.zip')
    for key, value in a.policy.state_dict().items():
        assert torch.equal(value, b.policy.state_dict()[key]), key
    assert (tmp_path/'complete'/'episodes.jsonl').read_bytes() == (tmp_path/'pause'/'episodes.jsonl').read_bytes()


def test_formal_failure_acknowledgement_and_split_guards(setup_runner, tmp_path):
    runner, train, validation, config = setup_runner
    with pytest.raises(ValueError, match='acknowledge'):
        runner.run_episode_training({**config, 'smoke':False}, train, validation, tmp_path/'blocked')
    with pytest.raises(ValueError, match='split'):
        runner.run_episode_training(config, train, train, tmp_path/'wrong')


def test_validation_is_greedy_and_does_not_change_training_state(setup_runner, tmp_path):
    runner, train, validation, config = setup_runner
    first = runner.run_episode_training(config, train, validation, tmp_path/'validated')
    second = runner.run_episode_training({**config, 'validation_every':99}, train, validation, tmp_path/'less')
    assert first['sample_order'] == second['sample_order']
    assert first['replay_insertions'] == second['replay_insertions']
    assert first['adam_updates'] == second['adam_updates']
    from v6_sb3.masked_dqn import MaskedDQN
    a = MaskedDQN.load(Path(first['latest_checkpoint'])/'model.zip')
    b = MaskedDQN.load(Path(second['latest_checkpoint'])/'model.zip')
    for key, value in a.policy.state_dict().items():
        assert torch.equal(value, b.policy.state_dict()[key])


def test_soft_environment_budget_waits_for_the_true_episode_boundary(setup_runner, tmp_path):
    runner, train, validation, config = setup_runner
    result = runner.run_episode_training({**config, 'max_env_steps':34}, train, validation, tmp_path/'budget')
    assert result['status'] == 'budget_complete'
    assert result['formal_env_steps'] == 50
    assert result['completed_episodes'] == 2
    assert result['active_sample_id'] is None
    assert result['validation_history'][-1]['episode'] == 2


def test_resume_rejects_changed_configuration_and_data(setup_runner, tmp_path):
    runner, train, validation, config = setup_runner
    runner.run_episode_training(config, train, validation, tmp_path/'run', stop_after_steps=5)
    with pytest.raises(ValueError, match='identity'):
        runner.run_episode_training({**config, 'learning_rate':.001}, train, validation, tmp_path/'run', resume=True)
    train[0].load_kw[0] = 2.
    with pytest.raises(ValueError, match='identity'):
        runner.run_episode_training(config, train, validation, tmp_path/'run', resume=True)


@pytest.mark.parametrize('orphan', ['.writing_000002', 'checkpoint_000002'])
def test_resume_survives_a_killed_checkpoint_generation(setup_runner, tmp_path, orphan):
    runner, train, validation, config = setup_runner
    run = tmp_path/'run'
    runner.run_episode_training(config, train, validation, run, stop_after_steps=1)
    folder = run/'checkpoints'/orphan
    folder.mkdir()
    (folder/'incomplete_marker').write_text('process killed before latest pointer update')
    result = runner.run_episode_training(config, train, validation, run, resume=True)
    assert result['status'] == 'complete'
    assert not list((run/'checkpoints').glob('.writing_*'))
    assert len(list((run/'checkpoints').glob('checkpoint_*'))) == config['checkpoint_keep']


def test_pruning_callback_has_a_complete_recoverable_validation_checkpoint(setup_runner, tmp_path):
    runner, train, validation, config = setup_runner
    import optuna
    called = []
    def prune(summary, progress):
        called.append(summary['episode'])
        assert progress['completed_episodes'] == 2
        assert Path(progress['latest_checkpoint']).joinpath('replay.pkl').exists()
        raise optuna.TrialPruned('test')
    with pytest.raises(optuna.TrialPruned):
        runner.run_episode_training(config, train, validation, tmp_path/'run', on_validation=prune)
    assert called == [2]
    after = runner.run_episode_training(config, train, validation, tmp_path/'run', resume=True)
    assert after['completed_episodes'] == 5
    assert [row['episode'] for row in after['validation_history']] == [2,4,5]


def test_failed_episode_has_one_true_terminal_replay_item_and_no_complete_cost(setup_runner, monkeypatch, tmp_path):
    runner, train, validation, config = setup_runner
    class FailureEnv(ToyVesselEnv):
        def step(self, action):
            observation, reward, done, truncated, info = super().step(action)
            if done:
                self.episode_status = 'soc_limited'
                info['terminal_reason'] = 'failure_soc_limited'
                info['failure_penalty_equivalent_cny'] = 10.
                reward -= 10.
            return observation, reward, done, truncated, info
    monkeypatch.setattr(runner, '_make_env', lambda config:FailureEnv())
    result = runner.run_episode_training(config, train, validation, tmp_path/'run')
    assert result['completed_episodes'] == 5
    assert result['validation_history'][-1]['completed'] == 0
    assert result['validation_history'][-1]['fixed_complete_set_cost_cny'] is None
    assert result['validation_history'][-1]['failure_reasons'] == {'soc_limited':1}
    rows = [json.loads(line) for line in (tmp_path/'run'/'episodes.jsonl').read_text().splitlines()]
    assert all(not row['completed'] and row['complete_economic_cost_cny'] is None for row in rows)
    for row in rows:
        assert row['return_sum'] == pytest.approx(-row['executed_prefix_economic_cost_cny']-10.)
    import pickle
    with Path(result['latest_checkpoint']).joinpath('replay.pkl').open('rb') as source:
        replay = pickle.load(source)
    formal = slice(config['learning_starts'], result['total_env_steps'])
    assert np.count_nonzero(replay.dones[formal]) == 5
    assert np.count_nonzero(replay.failure_penalties[formal]) == 5
    assert not replay.next_action_masks[formal][replay.dones[formal].astype(bool)].any()


@pytest.mark.parametrize('reward_mode', ['zhou_economic_only', 'v6_energy_soc_ramp'])
def test_real_direct_environment_and_real_sb3_optimizer_smoke(tmp_path, reward_mode):
    from v6_sb3.episode_training import run_episode_training, _smoke_episodes
    train, validation = _smoke_episodes()
    result = run_episode_training(dict(smoke=True, episodes=2, learning_starts=64,
        buffer_size=512, batch_size=8, network_width=8, validation_every=2,
        target_update_interval=64, reward_mode=reward_mode), train, validation, tmp_path/'run')
    assert result['status'] == 'complete'
    assert result['formal_env_steps'] == 128
    assert result['adam_updates'] == 4
    assert result['replay_insertions'] == 192
    assert result['validation_history'][-1]['completed'] == 2
    rows = [json.loads(line) for line in (tmp_path/'run'/'episodes.jsonl').read_text().splitlines()]
    if reward_mode == 'zhou_economic_only':
        for row in rows:
            assert row['return_sum'] == pytest.approx(-row['complete_economic_cost_cny'], abs=1e-8)
