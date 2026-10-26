"""Focused checks for the frozen V6 one-step, uniform DQN configuration."""

import copy

import pytest
import torch

import v4.dqn as dqn


def test_dqn_uses_target_masked_max_and_terminal_has_zero_bootstrap():
    assert hasattr(dqn, 'masked_dqn_targets')
    online = torch.tensor([[100., 2., 1., 0.], [float('nan')] * 4])
    target = torch.tensor([[3., 7., 999., 5.], [float('nan')] * 4])
    mask = torch.tensor([[True, True, False, True], [False] * 4])
    rewards, done = torch.tensor([[-2.], [-50.]]), torch.tensor([[0.], [1.]])
    ordinary = dqn.masked_dqn_targets(rewards, target, done,
        next_action_masks=mask, gamma=1.)
    double = dqn.masked_double_dqn_targets(rewards, online, target, done,
        next_action_masks=mask, gamma=1.)
    assert ordinary.tolist() == [[5.], [-50.]]
    assert double.tolist() == [[1.], [-50.]]


def test_v6_frozen_dimensions_parameters_and_executed_action_coordinates():
    assert hasattr(dqn, 'DirectPowerDQN')
    agent = dqn.DirectPowerDQN(replay_capacity=4)
    assert [agent.online.layers[0].in_features,
            *(layer.out_features for layer in agent.online.layers)] == [5, 256, 256, 13]
    assert agent.action_values_kw == tuple(range(-60, 61, 10))
    assert agent.algorithm == 'dqn' and agent.n_step == 1
    assert agent.gamma == agent.reward_scale == 1.
    assert agent.replay_sampling == 'uniform' and agent.failure_terminal_quota == 0
    settings = agent.optimizer.param_groups[0]
    assert (settings['lr'], settings['betas'], settings['eps']) == (.0001, (.9, .999), 1e-8)
    state, successor = (.5, .5, 200/600, 0., .5), (.5, .5, 230/600, 0., .5)
    assert agent.action_power_kw(9, state) == 230
    assert agent.action_index_for_power(230, state) == 9
    agent.remember(state, 230, -20., successor, done=False,
        next_feasible_actions=(170, 230, 290))
    item = agent.replay[-1]
    assert item.action_index == 9
    assert item.next_feasible_indices == (0, 6, 12)
    assert item.reward_cny == -20.
    with pytest.raises(ValueError, match='nonterminal'):
        agent.remember(state, 230, -20., successor, done=False, next_feasible_actions=())
    legacy = dqn.DirectPowerDDQN()
    assert legacy.state_mode == 'history7' and legacy.action_mode == 'absolute61'
    assert [legacy.online.layers[0].in_features,
            *(layer.out_features for layer in legacy.online.layers)] == [7, 128, 128, 61]


def test_target_stays_fixed_until_one_thousand_actual_adam_updates():
    assert hasattr(dqn, 'DirectPowerDQN')
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        agent = dqn.DirectPowerDQN(replay_capacity=1)
        state = (.5, .5, 200/600, 0., .5)
        agent.remember(state, 200, -4., state, done=True, next_feasible_actions=())
        initial = copy.deepcopy(agent.target.state_dict())
        assert agent.learn(batch_size=2) is None
        assert agent.economic_optimizer_updates == 0
        for _ in range(999):
            assert agent.learn(batch_size=1) is not None
        assert agent.economic_optimizer_updates == 999
        assert agent.target_sync_calls == 1
        assert all(torch.equal(initial[k], value) for k, value in agent.target.state_dict().items())
        assert agent.learn(batch_size=1) is not None
        assert agent.economic_optimizer_updates == 1000
        assert agent.target_sync_calls == 2 and agent.target_soft_update_calls == 0
        assert all(torch.equal(agent.online.state_dict()[k], value)
                   for k, value in agent.target.state_dict().items())
    finally:
        torch.set_num_threads(previous_threads)
