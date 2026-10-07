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
