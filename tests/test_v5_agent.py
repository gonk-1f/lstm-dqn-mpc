"""Small V5 dimension, action, checkpoint and diagnostic checks; no datasets."""
import copy

import pytest
import torch

from v2.economics import RawCnyIntervalLedger
from v3.control import EconomicMPC
from v4.control import DirectTransition
from v4.decision_diagnostics import actual_greedy_q_records
from v4.diagnostics import executed_transition_statistics, fixed_q_diagnostics, trajectory_record
from v4.dqn import DirectPowerDDQN, masked_double_dqn_targets


def new_agent(**kwargs):
    return DirectPowerDDQN(state_mode='load_delta4', action_mode='increment7',
        n_step=1, gamma=1., reward_scale=.001, replay_capacity=300000,
        replay_sampling='prioritized', failure_terminal_quota=2, **kwargs)


def test_load_delta4_network_action_mapping_and_one_masked_per_update():
    torch.set_num_threads(1)
    agent = new_agent(seed=42)
    state = (.6, .25, .5, -.1)
    successor = (.6, .2, .6, -.05)
    assert agent.online.layers[0].in_features == 4
    assert agent.online(torch.tensor([state])).shape == (1, 7)
    assert agent.optimizer.param_groups[0]['lr'] == .001
    assert agent.target_tau == .001
    assert all(torch.equal(value, agent.target.state_dict()[key])
               for key, value in agent.online.state_dict().items())
    for index, delta in enumerate((-60, -30, -10, 0, 10, 30, 60)):
        assert agent.action_power_kw(index, state) == 300 + delta
        assert agent.action_index_for_power(300 + delta, state) == index
    with pytest.raises(ValueError, match='range'):
        agent.action_power_kw(0, (.6, .1, 0., 0.))
    for index in range(64):
        agent.remember(state, 360, -20., successor, done=index >= 62,
            next_feasible_actions=() if index >= 62 else (300, 330, 360),
            experience_outcome='failure' if index >= 62 else 'success',
            terminal_reason='failure_soc_limited' if index >= 62 else None,
            failure_penalty_equivalent_cny=10. if index >= 62 else 0.)
    assert agent.replay[0].reward_cny == -.02
    assert agent.replay[0].action_index == 6
    before_online = {key: value.clone() for key, value in agent.online.state_dict().items()}
    before_target = {key: value.clone() for key, value in agent.target.state_dict().items()}
    assert agent.learn(batch_size=64) is not None
    assert agent.economic_optimizer_updates == agent.target_soft_update_calls == 1
    assert agent.target_sync_calls == 1
    assert any(not torch.equal(value, agent.online.state_dict()[key]) for key, value in before_online.items())
    for key, value in before_target.items():
        assert torch.allclose(agent.target.state_dict()[key],
            value.lerp(agent.online.state_dict()[key], .001))
    assert agent.failure_terminal_replay_count == 2
    assert agent.td_statistics()['sample_outcomes']['failure_terminal'] == 2


def test_checkpoint_layout_identity_rejects_old3_before_mutating_new4():
    agent = new_agent(seed=9)
    original = {key: value.clone() for key, value in agent.online.state_dict().items()}
    old = DirectPowerDDQN(state_mode='minimal3', action_mode='increment7',
        replay_sampling='prioritized', failure_terminal_quota=2).training_state()
    with pytest.raises(ValueError, match='state mode/dimension'):
        agent.load_training_state(old)
    assert all(torch.equal(value, agent.online.state_dict()[key]) for key, value in original.items())
    saved = copy.deepcopy(agent.training_state())
    compatible = new_agent(seed=6)
    compatible.load_training_state(saved)
    assert all(torch.equal(value, compatible.online.state_dict()[key])
               for key, value in agent.online.state_dict().items())
    saved['state_features'] = tuple(reversed(saved['state_features']))
    with pytest.raises(ValueError, match='state mode/dimension'):
        compatible.load_training_state(saved)
    # The historical state layouts remain available without changing defaults.
    assert DirectPowerDDQN(state_mode='minimal3').online.layers[0].in_features == 3
    assert DirectPowerDDQN(state_mode='history7').online.layers[0].in_features == 7


def test_four_feature_q_probes_and_previous_fc_telemetry():
    agent = new_agent(seed=4)
    # Load is zero, previous FC is 300: reading index 1 would miss this stop.
    item = DirectTransition((.8, 0., .5, -.1), 0, -1., (.8, 0., 0., 0.),
        RawCnyIntervalLedger(0., 0., 0., 0.), 0., .8, True, (),
        physical_feasible_actions=(0,), policy_candidate_actions=(0,))
    # Absolute61 can represent this diagnostic shutdown, unlike a 60 kW increment.
    absolute_agent = DirectPowerDDQN(state_mode='load_delta4')
    record = actual_greedy_q_records('synthetic', (item,), absolute_agent)[0]
    assert 'forced_stop' in record['priority_tags']
    assert record['hold_action_legal'] is False
    assert trajectory_record('synthetic', (item,), completed=True)['transitions'][0][
        'fc_stop_reason'] == 'forced_no_positive_action'
    assert executed_transition_statistics((item,))['fc_change_statistics_kw']['max_absolute'] == 300.
    probes = fixed_q_diagnostics(agent, EconomicMPC(nominal_cost_cny=1.))
    assert all(len(probe['state']) == 4 and probe['state'][3] == 0. for probe in probes)
    assert all(len(probe['q_values']) == 7 for probe in probes)


def test_reversal_counts_do_not_connect_samples_or_bridge_hold_actions():
    def item(previous, action, *, done=False):
        return DirectTransition((.6, .25, previous/600., 0.), action, -1.,
            (.6, .25, action/600., 0.), RawCnyIntervalLedger(1., 0., 0., 0.),
            150.-action, .6, done, () if done else (action,))
    values = (item(300, 360), item(360, 300, done=True),
              item(0, 60), item(60, 60), item(60, 0, done=True))
    statistics = executed_transition_statistics(values, sample_transition_counts=(2, 3))
    assert statistics['fc_direction_reversals'] == 1
    assert statistics['fc_direction_reversal_eligible_pairs'] == 1
    assert statistics['fc_direction_comparison_pairs'] == 3
    assert executed_transition_statistics(values)['fc_direction_reversals'] is None


def test_weighted_td_metrics_describe_loss_samples_without_changing_update(monkeypatch):
    agent = new_agent(seed=3)
    state = (.6, .25, .5, 0.)
    agent.remember(state, 300, -4., state, done=True, next_feasible_actions=())
    agent.remember(state, 300, -20., state, done=True, next_feasible_actions=())
    batch = list(agent.replay)
    weights = [.2, .8]
    with torch.no_grad():
        predictions = agent.online(torch.tensor([state, state])).gather(1, torch.tensor([[3], [3]]))[:, 0]
    errors = torch.tensor([item.reward_cny for item in batch]) - predictions
    expected_sum = sum(weight * abs(float(error)) for weight, error in zip(weights, errors))
    monkeypatch.setattr(agent, '_sample_replay_batch_with_weights', lambda size: (batch, weights, [.5, .5]))
    agent.learn(batch_size=2)
    stats = agent.td_statistics()['replay_sampling']
    assert stats['mean_is_weighted_absolute_td_error'] == pytest.approx(expected_sum / 2)
    assert stats['is_normalized_absolute_td_error'] == pytest.approx(expected_sum / sum(weights))
    agent.reset_td_statistics()
    assert agent.td_statistics()['replay_sampling']['mean_is_weighted_absolute_td_error'] is None


def test_one_step_target_is_masked_online_choice_target_value_no_terminal_bootstrap():
    target = masked_double_dqn_targets(torch.tensor([[-.01], [-10.]]),
        torch.tensor([[5., 1., 2.], [1., 2., 3.]]),
        torch.tensor([[50., 10., 20.], [float('nan')]*3]),
        torch.tensor([[0.], [1.]]), gamma=1.,
        next_action_masks=torch.tensor([[False, True, True], [False]*3]))
    assert target[:, 0].tolist() == pytest.approx([19.99, -10.])
