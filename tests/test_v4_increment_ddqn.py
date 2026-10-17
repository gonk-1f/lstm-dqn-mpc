"""Bounded checks for the optional seven-increment DDQN experiment."""

import copy
from dataclasses import replace

import pytest
import torch

from v4.dqn import DirectPowerDDQN, masked_double_dqn_targets


def increment_agent(**kwargs):
    return DirectPowerDDQN(state_mode='minimal3', action_mode='increment7', **kwargs)


def test_increment_network_selection_and_index_are_executed_power_differences():
    agent = increment_agent(seed=7)
    state = (.6, .5, 100/600)
    assert [agent.online.layers[0].in_features,
            *(layer.out_features for layer in agent.online.layers)] == [3, 128, 128, 7]
    assert agent.action_values_kw == (-60, -30, -10, 0, 10, 30, 60)
    assert all(torch.equal(value, agent.target.state_dict()[key])
               for key, value in agent.online.state_dict().items())
    with torch.no_grad():
        for parameter in agent.online.parameters():
            parameter.zero_()
        agent.online.layers[-1].bias.copy_(torch.arange(7.))
    assert agent.select_power(state, (40, 90, 100, 130), epsilon=0.) == 130
    assert {agent.select_power(state, (40, 90, 100, 130), epsilon=1.)
            for _ in range(80)} == {40, 90, 100, 130}
    assert agent._indices((40, 90, 100, 130), state) == (0, 2, 3, 5)
    assert agent.action_power_kw(6, state) == 160
    assert agent.action_index_for_power(40, state) == 0
    with pytest.raises(ValueError, match='increment'):
        agent.select_power(state, (200,), epsilon=0.)
    with pytest.raises(ValueError, match='minimal3'):
        DirectPowerDDQN(action_mode='increment7')


def test_increment_candidate_mask_does_not_clip_or_fall_back_to_absolute_grid():
    from v4.actions import candidate_fc_actions

    assert candidate_fc_actions(tuple(range(0, 201, 10)), 0, 'increment7') == (0, 10, 30, 60)
    assert candidate_fc_actions((100, 110, 120), 0, 'increment7') == ()
    assert candidate_fc_actions((540, 570, 590, 600), 600, 'increment7') == (540, 570, 590, 600)
    assert candidate_fc_actions((100, 110, 120), 0) == (100, 110, 120)


def test_increment_replay_next_mask_is_relative_to_next_actual_fc_and_updates_once():
    agent = increment_agent(n_step=1, reward_scale=.001, replay_sampling='prioritized',
                            failure_terminal_quota=2)
    state, successor = (.6, .5, 100/600), (.59, .6, 130/600)
    agent.remember(state, 130, -20., successor, done=False,
                   next_feasible_actions=(70, 100, 130, 190))
    item = agent.replay[0]
    assert item.action_index == 5  # +30, not absolute FC 130 / 10.
    assert item.next_feasible_indices == (0, 1, 3, 6)
    assert item.reward_cny == pytest.approx(-.02)
    assert agent._pool_coverage['action_kw'] == {'130': 1}
    assert agent._pool_coverage['action_delta_kw'] == {'30': 1}
    agent.remember(successor, 100, -30., (.58, .6, 100/600), done=True,
                   next_feasible_actions=())
    assert agent.replay[-1].action_index == 1
    assert agent.replay[-1].next_feasible_indices == ()
    before = {key: value.clone() for key, value in agent.online.state_dict().items()}
    assert agent.learn(batch_size=2) is not None
    assert agent.economic_optimizer_updates == agent.target_soft_update_calls == 1
    assert any(not torch.equal(before[key], value) for key, value in agent.online.state_dict().items())


def test_increment_trajectory_checks_recorded_action_identity_before_insertion():
    from v2.economics import RawCnyIntervalLedger
    from v4.control import DirectTransition

    agent = increment_agent(n_step=1)
    transition = DirectTransition((.6, .5, 0.), 60, -1., (.59, .5, .1),
        RawCnyIntervalLedger(0., 0., 0., 0.), 240., .59, True, (),
        terminal_reason='failure_increment_action_limited',
        action_mode='increment7', action_index=6, action_delta_kw=60)
    with pytest.raises(ValueError, match='action index'):
        agent.remember_trajectory((replace(transition, action_index=5),))
    assert not agent.replay
    with pytest.raises(ValueError, match='delta'):
        agent.remember_trajectory((replace(transition, action_delta_kw=30),))
    assert not agent.replay
    assert agent.remember_trajectory((transition,)) == 1
    assert agent.replay[0].action_index == 6
    assert agent.replay[0].terminal_reason == 'failure_increment_action_limited'
    assert agent.failure_terminal_replay_count == 1


def test_seven_action_one_step_masked_double_target_and_terminal_no_bootstrap():
    online = torch.tensor([[0., 4., 100., 0., 0., 0., 9.], [float('nan')]*7])
    target = torch.tensor([[0., 3., 999., 0., 0., 0., 7.], [float('nan')]*7])
    masks = torch.tensor([[False, True, False, False, False, False, True], [False]*7])
    result = masked_double_dqn_targets(torch.tensor([[-2.], [-5.]]), online, target,
        torch.tensor([[0.], [1.]]), next_action_masks=masks, gamma=1.)
    assert result.tolist() == [[5.], [-5.]]


def test_increment_checkpoint_identity_roundtrip_and_old_absolute_compatibility():
    agent = increment_agent(n_step=1, replay_sampling='prioritized', failure_terminal_quota=2)
    state = (.6, .5, 0.)
    agent.remember(state, 60, -4., (.59, .5, .1), done=True, next_feasible_actions=())
    saved = copy.deepcopy(agent.training_state())
    assert saved['action_mode'] == 'increment7' and saved['action_dim'] == 7
    restored = increment_agent(n_step=1, replay_sampling='prioritized', failure_terminal_quota=2)
    restored.load_training_state(saved)
    assert tuple(restored.replay) == tuple(agent.replay)
    assert torch.equal(restored.online(torch.tensor([state])), agent.online(torch.tensor([state])))
    absolute = DirectPowerDDQN(state_mode='minimal3', replay_sampling='prioritized', failure_terminal_quota=2)
    before = {key: value.clone() for key, value in absolute.online.state_dict().items()}
    with pytest.raises(ValueError, match='action'):
        absolute.load_training_state(saved)
    assert all(torch.equal(before[key], value) for key, value in absolute.online.state_dict().items())
    old = copy.deepcopy(absolute.training_state())
    for key in ('action_mode', 'action_dim', 'action_values_kw'):
        old.pop(key)
    absolute.load_training_state(old)
    assert absolute.online(torch.tensor([state])).shape == (1, 61)
    with pytest.raises(ValueError, match='action'):
        agent.load_training_state(old)
