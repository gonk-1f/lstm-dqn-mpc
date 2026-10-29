"""Target clock follows stored formal transitions, independently of Adam."""
import numpy as np
import pytest
import torch
import gymnasium as gym
from stable_baselines3.common.logger import configure
from v6_sb3.masked_dqn import MaskedDQN


class ToyEnv(gym.Env):
    observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(5,), dtype=np.float32)
    action_space = gym.spaces.Discrete(13)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(5, np.float32), {}

    def action_masks(self):
        return np.ones(13, dtype=bool)


def model(**overrides):
    torch.set_num_threads(1)
    agent = MaskedDQN(env=ToyEnv(), buffer_size=64, learning_starts=0,
        batch_size=4, train_freq=32, gradient_steps=1, td_diagnostics=False,
        policy_kwargs=dict(net_arch=[8, 8]), **overrides)
    agent.set_logger(configure(None, []))
    obs = np.zeros((1, 5), np.float32)
    agent.replay_buffer.add(obs, obs, np.array([6]), np.array([-1.]), np.array([False]),
        [dict(next_action_mask=np.ones(13, dtype=bool))])
    return agent


def test_target_sync_uses_exact_10000_formal_steps_at_32_to_1_cadence():
    agent = model(target_sync_unit='formal_env_steps', target_update_interval=10000)
    # Random warmup stores/collects steps but does not advance the formal clock.
    agent.num_timesteps = 50000
    assert agent.formal_env_steps == 0 and agent.target_sync_history == []
    for step in range(1, 20001):
        agent.num_timesteps += 1
        agent.record_formal_transition()
        if step % 32 == 0:
            agent.train(gradient_steps=1, batch_size=4)
    assert agent._n_updates == 625
    assert agent.target_sync_calls == 3  # initial copy plus 10000 and 20000
    assert agent.target_sync_history == [
        dict(formal_env_steps=10000, adam_updates=312),
        dict(formal_env_steps=20000, adam_updates=624)]


def test_formal_mode_disables_optimizer_count_copies_and_roundtrips(tmp_path):
    agent = model(target_sync_unit='formal_env_steps', target_update_interval=3)
    for _ in range(6):
        agent.train(gradient_steps=1, batch_size=4)
    assert agent.target_sync_calls == 1 and agent.target_sync_history == []
    for _ in range(3):
        agent.record_formal_transition()
    assert agent.target_sync_calls == 2
    agent.save(tmp_path/'model')
    restored = MaskedDQN.load(tmp_path/'model.zip', device='cpu')
    assert restored.target_sync_unit == 'formal_env_steps'
    assert restored.formal_env_steps == 3
    assert restored.target_sync_history == [dict(formal_env_steps=3, adam_updates=6)]


def test_historical_mode_keeps_existing_adam_schedule():
    agent = model(target_update_interval=2)
    assert agent.target_sync_unit == 'adam_updates'
    agent.train(gradient_steps=2, batch_size=4)
    assert agent.target_sync_calls == 2
    with pytest.raises(ValueError, match='formal'):
        agent.record_formal_transition()


def test_unknown_target_clock_is_rejected():
    with pytest.raises(ValueError, match='Target'):
        model(target_sync_unit='wall_time')
