import pytest
import torch

from v4.control import ACTION_KW
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
