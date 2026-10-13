import pytest
import torch

from dqn.networks.mlp_qnet import MLPQNetwork
from v2.economics import RawCnyIntervalLedger
from v4.control import ACTION_KW, DirectTransition
from v4.dqn import DirectPowerDDQN, masked_double_dqn_targets


def test_mlp_dimensions_and_feasible_epsilon_selection():
    agent = DirectPowerDDQN(seed=7)
    assert agent.online(torch.zeros(2, 7)).shape == (2, 61)
    state = (0.6,) + (0.0,) * 6
    for _ in range(20):
        assert agent.select_power(state, (100, 110), epsilon=1.0) in (100, 110)
    with pytest.raises(ValueError, match="feasible"):
        agent.select_power(state, (), epsilon=0.0)


def test_current_economic_network_optimizer_and_initial_target_copy():
    agent = DirectPowerDDQN(seed=7)
    def layer_sizes(network):
        return [network.layers[0].in_features, *(layer.out_features for layer in network.layers)]
    assert layer_sizes(agent.online) == [7, 128, 128, 61]
    assert layer_sizes(agent.target) == [7, 128, 128, 61]
    assert all(torch.equal(value, agent.target.state_dict()[key])
               for key, value in agent.online.state_dict().items())
    assert agent.optimizer.param_groups[0]['lr'] == pytest.approx(.001)
    for name in ('outcome_model', 'outcome_optimizer', 'outcome_replay',
                 'remember_outcome_trajectory', 'learn_outcome'):
        assert not hasattr(agent, name)
    assert agent.online(torch.zeros(2, 7)).shape == (2, 61)
    before = {key: value.clone() for key, value in agent.online.state_dict().items()}
    state = (0.6,) + (0.0,) * 6
    agent.remember(state, 100, -2., state, done=False, next_feasible_actions=(100, 200))
    assert agent.learn(batch_size=1) is not None
    assert any(not torch.equal(value, agent.online.state_dict()[key])
               for key, value in before.items())


@pytest.mark.parametrize('legacy_hidden', [(128, 64), (128, 128)])
def test_legacy_eight_dimensional_checkpoint_cannot_silently_load(legacy_hidden):
    legacy = MLPQNetwork(8, 61, legacy_hidden)
    current = DirectPowerDDQN(seed=7)
    with pytest.raises(RuntimeError, match='size mismatch'):
        current.online.load_state_dict(legacy.state_dict())


def test_masked_double_dqn_target_uses_online_choice_target_value_and_terminal():
    rewards = torch.tensor([[-5.0], [-7.0]])
    online = torch.tensor([[100.0, 4.0, 3.0], [1.0, 2.0, 3.0]])
    target = torch.tensor([[1000.0, 11.0, 22.0], [4.0, 5.0, 6.0]])
    done = torch.tensor([[0.0], [1.0]])
    result = masked_double_dqn_targets(
        rewards, online, target, done,
        next_action_masks=torch.tensor([[False, True, True], [False, False, False]]),
        gamma=0.9,
    )
    assert result[:, 0].tolist() == pytest.approx([-5.0 + 0.9 * 11.0, -7.0])


def test_replay_stores_raw_cny_reward_and_learns_without_cost_scale():
    agent = DirectPowerDDQN(seed=1)
    state = (0.6,) + (0.0,) * 6
    agent.remember(state, 100, -123.45, state, done=True, next_feasible_actions=())
    item = agent.replay[-1]
    assert item.reward_cny == -123.45
    assert item.action_index == ACTION_KW.index(100)
    assert agent.learn(batch_size=1) is not None


def test_unterminated_failure_requires_explicit_failure_conversion():
    agent = DirectPowerDDQN(seed=1)
    state = (0.215,) + (0.0,) * 6
    ledger = RawCnyIntervalLedger(1.0, 2.0, 3.0, 0.0)
    failed = DirectTransition(state, 0, -6.0, state, ledger, 1000.0, 0.201, False, ())
    with pytest.raises(ValueError, match="explicitly failure-terminated"):
        agent.remember_trajectory((failed,))
    assert not agent.replay


def test_only_independently_completed_prefix_can_be_retained():
    agent = DirectPowerDDQN(seed=1)
    state = (0.5,) + (0.0,) * 6
    ledger = RawCnyIntervalLedger(1.0, 0.0, 0.0, 0.0)
    finished = DirectTransition(state, 100, -1.0, state, ledger, 0.0, 0.5, True, (),
                                terminal_reason='completed')
    failed_later = DirectTransition(state, 0, -1.0, state, ledger, 1000.0, 0.201, False, ())
    assert agent.remember_completed_prefix((finished, failed_later)) == 1
    assert [item.action_index for item in agent.replay] == [ACTION_KW.index(100)]


def _insert_sampling_example(agent, index, kind):
    state = (0.6, index / 1000.0) + (0.0,) * 5
    if kind == 'terminal':
        agent.remember(state, 0, -10132.66, state, done=True,
                       next_feasible_actions=(), experience_outcome='failure',
                       terminal_reason='failure_soc_limited',
                       failure_penalty_equivalent_cny=10132.66)
    else:
        agent.remember(state, 0, -1.0, state, done=kind == 'success',
                       next_feasible_actions=() if kind == 'success' else (0,),
                       experience_outcome='failure' if kind == 'prefix' else 'success')


def test_failure_terminal_quota_draws_two_tagged_items_without_replacement():
    agent = DirectPowerDDQN(seed=17, failure_terminal_quota=2)
    for i in range(70):
        _insert_sampling_example(agent, i, 'success')
    for i in range(70, 74):
        _insert_sampling_example(agent, i, 'prefix')
    for i in range(74, 77):
        _insert_sampling_example(agent, i, 'terminal')
    before = tuple(agent.replay)
    batch = agent._sample_replay_batch(64)
    assert len(batch) == len({item.state for item in batch}) == 64
    assert sum(item.terminal_reason == 'failure_soc_limited' for item in batch) == 2
    assert agent.failure_terminal_replay_count == 3
    assert tuple(agent.replay) == before


def test_failed_prefixes_remain_ordinary_sampling_candidates():
    agent = DirectPowerDDQN(seed=17, failure_terminal_quota=2)
    for i in range(62):
        _insert_sampling_example(agent, i, 'prefix')
    for i in range(62, 65):
        _insert_sampling_example(agent, i, 'terminal')
    batch = agent._sample_replay_batch(64)
    assert agent.failure_terminal_replay_count == 3
    assert sum(item.terminal_reason == 'failure_soc_limited' for item in batch) == 2
    assert sum(item.experience_outcome == 'failure' and not item.done for item in batch) == 62


@pytest.mark.parametrize('terminal_count,expected', [(0, 0), (1, 1), (3, 2)])
def test_failure_terminal_quota_uses_available_items(terminal_count, expected):
    agent = DirectPowerDDQN(seed=17, failure_terminal_quota=2)
    for i in range(70):
        _insert_sampling_example(agent, i, 'success')
    for i in range(70, 70 + terminal_count):
        _insert_sampling_example(agent, i, 'terminal')
    batch = agent._sample_replay_batch(64)
    assert len(batch) == len({item.state for item in batch}) == 64
    assert sum(item.terminal_reason == 'failure_soc_limited' for item in batch) == expected


def test_failure_terminal_quota_handles_ordinary_shortage_and_fifo_eviction():
    agent = DirectPowerDDQN(seed=17, replay_capacity=8, failure_terminal_quota=2)
    for i in range(3):
        _insert_sampling_example(agent, i, 'success')
    for i in range(3, 8):
        _insert_sampling_example(agent, i, 'terminal')
    assert agent.failure_terminal_replay_count == 5
    assert sum(item.terminal_reason == 'failure_soc_limited'
               for item in agent._sample_replay_batch(7)) == 4
    for i in range(8, 16):
        _insert_sampling_example(agent, i, 'success')
    assert agent.failure_terminal_replay_count == 0
    batch = agent._sample_replay_batch(7)
    assert len(batch) == len({item.state for item in batch}) == 7
    assert all(item.state[1] >= .005 for item in batch)


def test_failure_terminal_quota_is_seeded_and_zero_keeps_uniform_sampling():
    agents = [DirectPowerDDQN(seed=17, failure_terminal_quota=2) for _ in range(2)]
    for agent in agents:
        for i in range(70):
            _insert_sampling_example(agent, i, 'success')
        for i in range(70, 74):
            _insert_sampling_example(agent, i, 'terminal')
    for _ in range(2):
        assert [item.state for item in agents[0]._sample_replay_batch(64)] == [
            item.state for item in agents[1]._sample_replay_batch(64)]

    uniform = DirectPowerDDQN(seed=17, failure_terminal_quota=0)
    for i in range(70):
        _insert_sampling_example(uniform, i, 'success')
    for i in range(70, 74):
        _insert_sampling_example(uniform, i, 'terminal')
    state = uniform.random.getstate()
    expected = uniform.random.sample(tuple(uniform.replay), 64)
    uniform.random.setstate(state)
    assert uniform._sample_replay_batch(64) == expected


def test_failure_terminal_quota_reports_actual_td_draws_without_mutating_replay():
    agent = DirectPowerDDQN(seed=17, reward_scale=.001, failure_terminal_quota=2)
    for i in range(70):
        _insert_sampling_example(agent, i, 'success')
    for i in range(70, 73):
        _insert_sampling_example(agent, i, 'terminal')
    before = tuple(agent.replay)
    assert agent.learn(batch_size=64) is not None
    assert agent.learn(batch_size=64) is not None
    stats = agent.td_statistics()
    assert tuple(agent.replay) == before
    assert stats['optimizer_updates'] == 2
    assert stats['sample_outcomes']['failure_terminal'] == 4
    assert stats['sample_outcomes']['failure_terminal_fraction'] == 4 / 128
    assert stats['failure_terminal']['original_units']['mean_absolute_td_error'] == pytest.approx(
        stats['failure_terminal']['mean_absolute_td_error'] / .001)


@pytest.mark.parametrize('quota', (-1, True, 1.5))
def test_failure_terminal_quota_rejects_invalid_values(quota):
    with pytest.raises(ValueError, match='failure_terminal_quota'):
        DirectPowerDDQN(failure_terminal_quota=quota)


def _unknown_prefix(length, *, shore=False):
    from types import SimpleNamespace
    from v3.control import EconomicMPC
    from v4.control import ReplayExecutionError, replay_episode
    modes = ['onboard'] * length
    if shore:
        modes[5:5] = ['shore_pending', 'shore_charging']
    modes.append('unknown')
    episode = SimpleNamespace(sample_id='prefix', split='train', operating_mode=tuple(modes),
        load_kw=tuple(100. if mode == 'onboard' else 0. for mode in modes),
        battery_bus_kw=(0.,) * len(modes))
    with pytest.raises(ReplayExecutionError) as caught:
        replay_episode(episode, lambda state, feasible: 100, accountant=EconomicMPC(nominal_cost_cny=1.))
    assert caught.value.failure_kind == 'data_truncation'
    return caught.value.executed_transitions


@pytest.mark.parametrize('n,length,expected', [(8, 0, 0), (8, 8, 0), (8, 9, 1), (8, 20, 12), (1, 3, 2)])
def test_observed_unknown_prefix_requires_real_successor_and_full_window(n, length, expected):
    a = DirectPowerDDQN(seed=42, n_step=n, gamma=.9, reward_scale=.001)
    values = _unknown_prefix(length)
    assert a.remember_observed_prefix(values, source_sample_id='prefix', source_round=1,
                                     source_rows=tuple(range(length))) == expected
    assert a.economic_unresolved_replay_insertions == expected
    assert a.economic_success_replay_insertions == a.economic_failure_replay_insertions == 0
    for index, item in enumerate(a.replay):
        assert not item.done and item.terminal_reason is None and item.next_feasible_indices
        assert item.experience_outcome == 'unresolved' and item.bootstrap_steps == n
        assert item.next_state == values[index+n].state
        assert item.reward_cny == pytest.approx(.001 * sum(.9**j * values[index+j].reward_cny for j in range(n)))
        assert item.failure_penalty_equivalent_cny == 0
        assert item.source_sample_id == 'prefix' and item.source_round == 1 and item.source_row_index == index


def test_observed_prefix_can_cross_confirmed_shore_but_never_unknown():
    a = DirectPowerDDQN(n_step=8, reward_scale=.001)
    values = _unknown_prefix(12, shore=True)
    assert values[4].shore_ledger is not None and not values[4].done
    assert a.remember_observed_prefix(values) == 4
    assert a.replay[0].reward_cny == pytest.approx(.001 * sum(t.reward_cny for t in values[:8]))
    assert a.replay[0].next_state == values[8].state
    assert all(not item.done for item in a.replay)
    # The unobserved final successor cannot be passed to the ordinary terminal API.
    with pytest.raises(ValueError, match='completed or explicitly failure'):
        a.remember_trajectory(values)


def test_observed_prefix_keeps_independently_closed_segment_without_crossing_its_terminal():
    from types import SimpleNamespace
    from v3.control import EconomicMPC
    from v4.control import replay_episode
    complete = replay_episode(SimpleNamespace(sample_id='closed', split='train',
        operating_mode=('onboard',)*3, load_kw=(100.,)*3, battery_bus_kw=(0.,)*3),
        lambda state, feasible: 100, accountant=EconomicMPC(nominal_cost_cny=1.))
    partial = _unknown_prefix(12)
    a = DirectPowerDDQN(n_step=8, reward_scale=.001)
    assert a.remember_observed_prefix((*complete.transitions, *partial)) == 7
    assert a.economic_success_replay_insertions == 3
    assert a.economic_unresolved_replay_insertions == 4
    assert [t.bootstrap_steps for t in list(a.replay)[:3]] == [3, 2, 1]
    assert all(t.done for t in list(a.replay)[:3])
    assert all(not t.done and t.bootstrap_steps == 8 for t in list(a.replay)[3:])
    assert a.replay[3].reward_cny == pytest.approx(.001*sum(t.reward_cny for t in partial[:8]))


def test_observed_prefix_rejects_discontinuity_or_physical_failure_atomically():
    from dataclasses import replace
    a = DirectPowerDDQN(n_step=8)
    values = list(_unknown_prefix(12))
    values[5] = replace(values[5], state=(.7,) + values[5].state[1:])
    with pytest.raises(ValueError, match='contiguous'):
        a.remember_observed_prefix(values)
    assert not a.replay
    failed = replace(_unknown_prefix(12)[-1], done=True, terminal_reason='failure_soc_limited',
                     next_feasible_actions=(), failure_penalty_equivalent_cny=100.)
    with pytest.raises(ValueError, match='truncation'):
        a.remember_observed_prefix((failed,))
    with pytest.raises(ValueError, match='outcome'):
        a.remember((.6,)*7, 100, -1., (.6,)*7, done=True, next_feasible_actions=(),
                   experience_outcome='unresolved')
    assert not a.replay


def test_replay_diagnostics_record_actual_categories_without_changing_updates_or_rng():
    from copy import deepcopy
    agents = [DirectPowerDDQN(seed=42, reward_scale=.001, failure_terminal_quota=2,
                             replay_diagnostics=enabled) for enabled in (True, False)]
    for a in agents:
        state = (.6,) + (0.,)*6
        a.remember(state, 100, -1., state, done=True, next_feasible_actions=(),
                   source_sample_id='zero_boundary_015', source_round=100, source_row_index=97)
        a.remember(state, 100, -100., state, done=True, next_feasible_actions=(),
                   experience_outcome='failure', terminal_reason='failure_soc_limited')
        a.remember(state, 100, -1., state, done=False, next_feasible_actions=(100,),
                   experience_outcome='unresolved')
        a.learn(batch_size=3)
    a, b = agents
    assert a.random.getstate() == b.random.getstate()
    assert all(torch.equal(v, b.online.state_dict()[k]) for k, v in a.online.state_dict().items())
    assert all(torch.equal(v, b.target.state_dict()[k]) for k, v in a.target.state_dict().items())
    before_rng = a.random.getstate(), torch.random.get_rng_state().clone()
    before = deepcopy(a.online.state_dict())
    stats = a.td_statistics()
    assert stats['normal_terminal']['sample_count'] == 1
    assert stats['failure_terminal']['sample_count'] == 1
    assert stats['sample_outcomes']['unresolved'] == 1
    assert stats['replay_coverage']['pool_categories']['normal_terminal'] == 1
    probe = stats['watched_replay_targets'][0]
    assert probe['source_row_index'] == 97 and probe['draw_count_cumulative'] == 1
    assert probe['done'] and probe['target_scaled'] == pytest.approx(-.001)
    assert stats['normal_terminal']['original_units']['mean_absolute_td_error'] == pytest.approx(
        stats['normal_terminal']['mean_absolute_td_error']/.001)
    assert a.random.getstate() == before_rng[0] and torch.equal(torch.random.get_rng_state(), before_rng[1])
    assert all(torch.equal(v, a.online.state_dict()[k]) for k, v in before.items())
    restored = DirectPowerDDQN(seed=999, reward_scale=.001, failure_terminal_quota=2)
    restored.load_training_state(a.training_state())
    assert restored.economic_unresolved_replay_insertions == 1
    assert restored.td_statistics()['watched_replay_targets'] == stats['watched_replay_targets']
    with pytest.raises(ValueError, match='format'):
        restored.load_training_state({'format_version': 1})


@pytest.mark.parametrize('enabled', [True, False])
def test_unique_replay_id_coverage_counts_optimizer_draws_and_evicted_available_ids(enabled):
    agent = DirectPowerDDQN(seed=17, replay_capacity=3, replay_diagnostics=enabled)
    state = (.6,) + (0.,) * 6
    empty = agent.td_statistics()['replay_coverage']['unique_replay_ids_since_reset']
    if enabled:
        assert empty['sampled_count'] == empty['denominator'] == 0
        assert empty['coverage_rate'] is None
    else:
        assert empty is None
    for _ in range(3):
        agent.remember(state, 100, -1., state, done=True, next_feasible_actions=())
    agent.reset_td_statistics()
    initial_rng = agent.random.getstate()
    assert agent.learn(batch_size=4) is None
    agent._sample_replay_batch(1)
    agent.random.setstate(initial_rng)
    if enabled:
        assert agent.td_statistics()['replay_coverage']['unique_replay_ids_since_reset']['sampled_count'] == 0
    assert agent.learn(batch_size=1) is not None
    agent.random.setstate(initial_rng)
    assert agent.learn(batch_size=1) is not None
    coverage = agent.td_statistics()['replay_coverage']['unique_replay_ids_since_reset']
    if enabled:
        assert coverage['sampled_count'] == 1
        assert coverage['initial_pool_size'] == coverage['denominator'] == 3
        assert coverage['insertions_since_reset'] == 0
        assert coverage['coverage_rate'] == pytest.approx(1 / 3)
    else:
        assert coverage is None
    for _ in range(3):
        agent.remember(state, 100, -1., state, done=True, next_feasible_actions=())
    assert agent.learn(batch_size=3) is not None
    stats = agent.td_statistics()
    coverage = stats['replay_coverage']['unique_replay_ids_since_reset']
    assert stats['sample_count'] == 5
    if enabled:
        assert coverage['sampled_count'] == 4
        assert coverage['initial_pool_size'] == coverage['insertions_since_reset'] == 3
        assert coverage['denominator'] == 6
        assert coverage['coverage_rate'] == pytest.approx(4 / 6)
        assert 'evicted' in coverage['denominator_definition']
    else:
        assert coverage is None
    restored = DirectPowerDDQN(seed=999, replay_capacity=3)
    restored.load_training_state(agent.training_state())
    restored.reset_td_statistics()
    restored_coverage = restored.td_statistics()['replay_coverage']['unique_replay_ids_since_reset']
    if enabled:
        assert restored_coverage['sampled_count'] == restored_coverage['insertions_since_reset'] == 0
        assert restored_coverage['denominator'] == 3
        assert restored_coverage['coverage_rate'] == 0
    else:
        assert restored_coverage is None
