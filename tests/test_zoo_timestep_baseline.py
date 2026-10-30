"""Compare the retained timestep algorithm with PR #5's pinned parent source.

The numerical comparison runs against an independent historical implementation,
not a second instance of the modified class. No dataset or experiment is read.
"""
from pathlib import Path
import subprocess

import gymnasium as gym
import numpy as np
import pytest
import torch


class ChangingMasks(gym.Env):
    observation_space = gym.spaces.Box(-np.inf, np.inf, (5,), np.float32)
    action_space = gym.spaces.Discrete(13)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.index = 0
        return np.zeros(5, np.float32), {}

    def action_masks(self):
        mask = np.zeros(13, bool)
        mask[[2, 6] if self.index % 2 == 0 else [4, 8]] = True
        return mask

    def step(self, action):
        assert self.action_masks()[int(action)]
        self.index += 1
        done = self.index == 7
        observation = np.array([self.index / 7, int(action) / 13, .6, 0., 0.], np.float32)
        info = dict(next_action_mask=np.zeros(13, bool) if done else self.action_masks(),
                    failure_penalty_equivalent_cny=0.,
                    terminal_reason='failure_soc_limited' if done else None)
        return observation, -1. if done else 1. / (1. + int(action)), done, False, info


def test_timestep_route_matches_pinned_7df880d_weights_replay_and_clocks():
    from experiments.sb3_manual.zoo_vessel.algorithm import ZooMaskedDQN
    root = Path(__file__).resolve().parents[1]
    try:
        previous = subprocess.run(
            ['git', 'show', '7df880d:experiments/sb3_manual/zoo_vessel/algorithm.py'],
            cwd=root, capture_output=True, text=True, encoding='utf-8', check=False,
        )
    except FileNotFoundError:
        pytest.skip('Git is unavailable for the pinned historical comparison')
    if previous.returncode:
        pytest.skip('The pinned PR #5 parent source is absent in this checkout')
    namespace = {'__name__': '_zoo_vessel_7df880d_reference'}
    exec(compile(previous.stdout, '7df880d/algorithm.py', 'exec'), namespace)
    old_class = namespace['ZooMaskedDQN']
    torch.set_num_threads(1)

    def run(algorithm):
        model = algorithm(env=ChangingMasks(), learning_starts=5, buffer_size=256,
            batch_size=4, train_freq=32, gradient_steps=1, target_update_interval=16,
            gamma=1., seed=13, device='cpu', td_diagnostics=False,
            policy_kwargs=dict(net_arch=[8, 8]), exploration_initial_eps=1.,
            exploration_final_eps=.05, exploration_fraction=.8)
        model.learn(total_timesteps=69, log_interval=None)
        return model

    old, new = run(old_class), run(ZooMaskedDQN)
    try:
        assert new.budget_mode == 'timestep'
        assert new.exploration_schedule_mode == 'timestep'
        for model in (old, new):
            assert model.num_timesteps == model.replay_buffer.insertions == 69
            assert model.formal_env_steps == 64 and model._n_updates == 2
        assert old._n_calls == new._n_calls == 69
        assert old.target_sync_history == new.target_sync_history
        assert old.target_sync_calls == new.target_sync_calls
        assert old.exploration_rate == new.exploration_rate
        assert old.last_training_loss == new.last_training_loss
        assert old.last_td_mae == new.last_td_mae
        for name, tensor in old.policy.state_dict().items():
            assert torch.equal(tensor, new.policy.state_dict()[name]), name
        for name in ('observations', 'next_observations', 'actions', 'rewards',
                     'dones', 'next_action_masks', 'failure_penalties', 'reasons'):
            np.testing.assert_array_equal(getattr(old.replay_buffer, name),
                                          getattr(new.replay_buffer, name))
        old_optimizer = old.policy.optimizer.state_dict()
        new_optimizer = new.policy.optimizer.state_dict()
        assert old_optimizer['param_groups'] == new_optimizer['param_groups']
        assert old_optimizer['state'].keys() == new_optimizer['state'].keys()
        for parameter, values in old_optimizer['state'].items():
            for key, value in values.items():
                other = new_optimizer['state'][parameter][key]
                if isinstance(value, torch.Tensor):
                    assert torch.equal(value, other), (parameter, key)
                else:
                    assert value == other, (parameter, key)
    finally:
        old.env.close()
        new.env.close()
        old.logger.close()
        new.logger.close()
