import pytest
import torch

from v2.economics import RawCnyIntervalLedger
from v4.control import ACTION_KW, DirectTransition
from v4.dqn import DirectPowerDDQN, masked_double_dqn_targets


def test_mlp_dimensions_and_feasible_epsilon_selection():
    agent = DirectPowerDDQN(seed=7)
    assert agent.online(torch.zeros(2, 8)).shape == (2, 61)
    state = (0.6,) + (0.0,) * 7
    for _ in range(20):
        assert agent.select_power(state, (100, 110), epsilon=1.0) in (100, 110)
    with pytest.raises(ValueError, match="feasible"):
        agent.select_power(state, (), epsilon=0.0)


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
    state = (0.6,) + (0.0,) * 7
    agent.remember(state, 100, -123.45, state, done=True, next_feasible_actions=())
    item = agent.replay[-1]
    assert item.reward_cny == -123.45
    assert item.action_index == ACTION_KW.index(100)
    assert agent.learn(batch_size=1) is not None


def test_failure_outcome_learns_executed_prefix_without_creating_economic_terminal():
    agent = DirectPowerDDQN(seed=1)
    state = (0.215,) + (0.0,) * 7
    ledger = RawCnyIntervalLedger(1.0, 2.0, 3.0, 0.0)
    failed = DirectTransition(
        state, 0, -6.0, state, ledger, 1000.0, 0.201,
        False, (),
    )
    agent.remember_outcome_trajectory((failed,), failed=True)
    assert len(agent.outcome_replay) == 1
    assert agent.outcome_replay[-1].failed is True
    assert agent.outcome_replay[-1].actual_reward_cny == -6.0
    assert not agent.replay
    assert agent.learn_outcome(batch_size=1) is not None


def test_failure_label_applies_only_to_unfinished_voyage_after_shore():
    agent = DirectPowerDDQN(seed=1)
    state = (0.5,) + (0.0,) * 7
    ledger = RawCnyIntervalLedger(1.0, 0.0, 0.0, 0.0)
    finished = DirectTransition(state, 100, -1.0, state, ledger, 0.0, 0.5, True, ())
    failed_later = DirectTransition(state, 0, -1.0, state, ledger, 1000.0, 0.201, False, ())
    agent.remember_outcome_trajectory((finished, failed_later), failed=True)
    assert [item.failed for item in agent.outcome_replay] == [False, True]
    assert agent.remember_completed_prefix((finished, failed_later)) == 1
    assert [item.action_index for item in agent.replay] == [ACTION_KW.index(100)]


def _insert_sampling_example(agent, index, kind):
    state = (0.6, index / 1000.0) + (0.0,) * 6
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
