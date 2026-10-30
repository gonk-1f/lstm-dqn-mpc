"""Episode budgets exercise native SB3 collection/storage/learn, not a toy runner."""
import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

from experiments.sb3_manual.zoo_vessel.algorithm import ZooMaskedDQN
from v6_sb3.masked_dqn import masked_td_targets


class Voyages(gym.Env):
    observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(5,), dtype=np.float32)
    action_space = gym.spaces.Discrete(13)

    def __init__(self, length=4, failures=()):
        self.length = length
        self.failures = failures
        self.index = 0
        self.voyage = -1
        self.executed = 0
        self.actions = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.voyage += 1
        self.index = 0
        return np.zeros(5, np.float32), {}

    def action_masks(self):
        mask = np.zeros(13, dtype=bool)
        mask[2] = True
        return mask

    def step(self, action):
        assert int(action) == 2
        self.actions.append(int(action))
        self.index += 1
        self.executed += 1
        terminal = self.index == self.length
        failed = terminal and self.voyage in self.failures
        obs = np.zeros(5, np.float32)
        obs[4] = self.index
        info = dict(next_action_mask=np.zeros(13, bool) if terminal else self.action_masks(),
            terminal_reason=('failure_soc_limited' if failed else 'completed') if terminal else None)
        if terminal:
            info['vessel_episode'] = dict(sample_id=f'train_{self.voyage}', split='train',
                completed=not failed, terminal_reason='soc_limited' if failed else 'completed')
        return obs, -1. if failed else 1., terminal, False, info


def agent(tmp_path, *, length=4, **overrides):
    torch.set_num_threads(1)
    kwargs = dict(env=Voyages(length=length, failures=(2,)), n_episodes=3,
        exploration_schedule_mode='episode', learning_starts=5,
        buffer_size=1024, batch_size=4, train_freq=32, gradient_steps=1,
        target_update_interval=16, policy_kwargs=dict(net_arch=[8, 8]),
        seed=13, device='cpu', td_diagnostics=False,
        exploration_initial_eps=1., exploration_final_eps=.05, exploration_fraction=.8)
    kwargs.update(overrides)
    model = ZooMaskedDQN(**kwargs)
    model.set_logger(configure(str(tmp_path), ['csv']))
    return model


class Trace(BaseCallback):
    def __init__(self):
        super().__init__()
        self.steps = []
        self.rollouts = []
        self.ended = False

    def _on_step(self):
        self.steps.append((self.num_timesteps, self.model.exploration_rate,
            self.model.replay_buffer.insertions, bool(self.locals['dones'][0])))
        return True

    def _on_rollout_end(self):
        self.rollouts.append(self.model.num_timesteps)

    def _on_training_end(self):
        self.ended = True


def test_three_formal_episodes_exclude_warmup_and_its_bridge_and_keep_terminal(tmp_path):
    model = agent(tmp_path)
    trace = Trace()
    boundaries = []
    model.episode_boundary_observer = lambda m: boundaries.append((m.completed_formal_episodes,
        m.num_timesteps, m.replay_buffer.insertions, m.formal_env_steps, m._n_updates))
    model.learn(total_timesteps=1000, callback=trace, log_interval=None)
    assert model.budget_mode == 'episode'
    assert model.completed_formal_episodes == 3
    assert model.formal_successes == 2 and model.formal_failures == 1
    assert model.warmup_env_steps == 5 and model.warmup_bridge_env_steps == 3
    assert model.num_timesteps == model.replay_buffer.insertions == 20
    assert model.formal_env_steps == 15 and model._n_updates == 0
    assert model.get_env().get_attr('executed')[0] == 20
    assert model.get_env().get_attr('index')[0] == 0  # native auto-reset, no next action
    assert model._episode_num == 5
    assert trace.ended and boundaries == [(1, 12, 12, 7, 0), (2, 16, 16, 11, 0), (3, 20, 20, 15, 0)]
    assert model.replay_buffer.dones[19, 0] == 1
    assert not model.replay_buffer.next_action_masks[19, 0].any()
    assert model.replay_buffer.next_observations[19, 0, 4] == 4
    assert all(insertions == step - 1 for step, _, insertions, _ in trace.steps)
    assert [row['total_env_steps'] for row in model.formal_episode_history] == [12, 16, 20]
    assert [row['episode_env_steps'] for row in model.formal_episode_history] == [4, 4, 4]
    assert model.last_formal_episode_summary['sample_id'] == 'train_4'


def test_warmup_ending_at_terminal_makes_next_natural_reset_eligible(tmp_path):
    model = agent(tmp_path, length=5)
    model.learn(total_timesteps=1000, log_interval=None)
    assert model.num_timesteps == 20 and model.formal_env_steps == 15
    assert model.warmup_env_steps == 5 and model.warmup_bridge_env_steps == 0
    assert model.completed_formal_episodes == 3
    assert model.formal_episode_history[0]['total_env_steps'] == 10


@pytest.mark.parametrize('length,updates', [(31, 0), (32, 1), (33, 1), (64, 2), (65, 2)])
def test_last_partial_block_never_adds_adam_but_full_blocks_do(tmp_path, length, updates):
    times = []
    model = agent(tmp_path, length=length, n_episodes=1, learning_starts=0,
        after_update_observer=lambda m: times.append(m.formal_env_steps))
    events = []
    model.episode_boundary_observer = lambda m: events.append((m.num_timesteps, m._n_updates))
    model.learn(total_timesteps=10000, log_interval=None)
    assert model.num_timesteps == model.replay_buffer.insertions == length
    assert model._n_updates == updates
    assert times == list(range(32, length + 1, 32))
    assert events == [(length, updates)]
    assert model.last_formal_episode_summary['adam_updates'] == updates
    assert model.target_sync_calls == 1 + length // 16
    assert [row['formal_env_steps'] for row in model.target_sync_history] == list(range(16, length + 1, 16))
    assert all(row['adam_updates'] == (row['formal_env_steps'] - 1) // 32 for row in model.target_sync_history)


def test_partial_episode_boundaries_accumulate_to_same_32_step_update_clock(tmp_path):
    times = []
    model = agent(tmp_path, length=11, n_episodes=3, learning_starts=0,
        after_update_observer=lambda m: times.append(m.formal_env_steps))
    model.learn(total_timesteps=1000, log_interval=None)
    assert model.formal_env_steps == 33 and model._n_updates == 1
    assert times == [32]
    assert model.completed_formal_episodes == 3


def test_failed_terminal_reward_and_empty_mask_do_not_bootstrap(tmp_path):
    model = agent(tmp_path, env=Voyages(length=7, failures=(0,)), learning_starts=0, n_episodes=1)
    model.learn(total_timesteps=1000, log_interval=None)
    replay = model.replay_buffer
    batch = replay._get_samples(np.array([6]))
    target = masked_td_targets(batch.rewards, torch.full((1, 13), 1e12), batch.dones, batch.next_action_masks)
    assert batch.rewards.item() == target.item() == -1.
    assert batch.dones.item() == 1.
    assert not batch.next_action_masks.any()
    assert model.formal_failures == 1 and model.formal_successes == 0


def test_episode_epsilon_is_fixed_within_voyage_and_warmup_bridge_is_one(tmp_path):
    model = agent(tmp_path)
    trace = Trace()
    model.learn(total_timesteps=1000, callback=trace, log_interval=None)
    eps = [value for _, value, _, _ in trace.steps]
    assert eps[:12] == [1.] * 12  # all warmup, bridge and first full formal voyage
    assert eps[12:16] == pytest.approx([1. - .95 / 2.4] * 4)
    assert eps[16:20] == pytest.approx([1. - .95 * 2. / 2.4] * 4)
    assert model.exploration_rate == .05


@pytest.mark.parametrize('completed,expected', [(0, 1.), (100, .7625), (200, .525),
    (300, .2875), (400, .05), (401, .05), (499, .05), (500, .05)])
def test_500_episode_schedule_reaches_final_at_completed_400(tmp_path, completed, expected):
    model = agent(tmp_path, n_episodes=500)
    model.completed_formal_episodes = completed
    assert model.episode_epsilon() == pytest.approx(expected)


def test_timestep_budget_can_explicitly_choose_episode_schedule(tmp_path):
    model = agent(tmp_path, n_episodes=None, exploration_episode_target=500)
    trace = Trace()
    model.learn(total_timesteps=69, callback=trace, log_interval=None)
    assert model.budget_mode == 'timestep'
    assert model.num_timesteps == 69 and model._n_updates == 2
    assert model.completed_formal_episodes == 15
    assert all(value == pytest.approx(1.) for step, value, _, _ in trace.steps if step <= 12)


def test_episode_budget_can_explicitly_choose_native_timestep_schedule(tmp_path):
    model = agent(tmp_path, n_episodes=1, exploration_schedule_mode='timestep', learning_starts=0, length=9)
    trace = Trace()
    model.learn(total_timesteps=1000, callback=trace, log_interval=None)
    assert model.num_timesteps == 9
    assert trace.steps[1][1] != trace.steps[-1][1]
    assert model.exploration_rate == model.exploration_schedule(model._current_progress_remaining)


def test_safety_cap_before_episode_budget_fails_explicitly(tmp_path):
    model = agent(tmp_path, length=100, n_episodes=3, learning_starts=5)
    with pytest.raises(RuntimeError, match='safety|Episode|episode'):
        model.learn(total_timesteps=21, log_interval=None)
    assert model.completed_formal_episodes < 3
    assert model.num_timesteps == model.replay_buffer.insertions == 21
    assert not model.replay_buffer.dones[:21].any()


def test_episode_model_save_load_retains_budget_and_counters_without_callback_closure(tmp_path):
    model = agent(tmp_path)
    model.episode_boundary_observer = lambda m: None
    model.learn(total_timesteps=1000, log_interval=None)
    path = tmp_path / 'model.zip'
    model.save(path)
    loaded = ZooMaskedDQN.load(path, device='cpu')
    assert loaded.n_episodes == 3 and loaded.completed_formal_episodes == 3
    assert loaded.exploration_schedule_mode == 'episode'
    assert loaded.episode_boundary_observer is None
    mask = np.zeros(13, bool); mask[9] = True
    assert int(loaded.predict(np.zeros(5, np.float32), deterministic=True, action_masks=mask)[0]) == 9


def test_episode_budget_relearn_does_not_silently_reset_clock_with_old_replay(tmp_path):
    model = agent(tmp_path)
    model.learn(total_timesteps=1000, log_interval=None)
    with pytest.raises(RuntimeError, match='fresh|resume'):
        model.learn(total_timesteps=1000, log_interval=None)
    assert model.num_timesteps == model.replay_buffer.insertions == 20


def test_pruning_at_post_update_boundary_preserves_terminal_and_propagates(tmp_path):
    import optuna
    model = agent(tmp_path, length=32, learning_starts=0)
    def prune(agent):
        assert agent._n_updates == 1 and agent.replay_buffer.insertions == 32
        raise optuna.TrialPruned('post-update pruning')
    model.episode_boundary_observer = prune
    with pytest.raises(optuna.TrialPruned, match='post-update'):
        model.learn(total_timesteps=1000, log_interval=None)
    assert model.replay_buffer.dones[31, 0] == 1
    assert model.num_timesteps == 32


@pytest.mark.parametrize('overrides', [dict(n_episodes=0), dict(n_episodes=-1),
    dict(n_episodes=1.5), dict(exploration_schedule_mode='unknown'),
    dict(n_episodes=2, train_freq=1), dict(n_episodes=2, gradient_steps=2)])
def test_invalid_episode_contracts_are_rejected_before_collection(tmp_path, overrides):
    with pytest.raises(ValueError):
        agent(tmp_path, **overrides)
