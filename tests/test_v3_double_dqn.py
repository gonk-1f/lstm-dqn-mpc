import unittest

import torch

from v3.control import MPCWeights
from v3.dqn import DoubleDQNAgent, double_dqn_targets


class DoubleDqnContractTests(unittest.TestCase):
    def test_online_network_selects_action_target_network_values_it(self):
        rewards = torch.tensor([[-2.0], [-3.0]])
        online_next = torch.tensor([[1.0, 5.0], [4.0, 2.0]])
        target_next = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
        done = torch.tensor([[0.0], [1.0]])
        targets = double_dqn_targets(rewards, online_next, target_next, done, gamma=1.0)
        self.assertEqual(targets.flatten().tolist(), [18.0, -3.0])

    def test_actual_transition_is_added_before_next_weight_selection(self):
        actions = (MPCWeights(0.2, 0.1), MPCWeights(1.0, 0.5))
        agent = DoubleDQNAgent(actions, state_dim=15, reward_scale_cny=1.0, seed=2)
        state = (0.0,) * 15
        action = agent.select_weights(state, epsilon=0.0)
        agent.remember(state, action, -12.5, (1.0,) * 15, done=False)
        self.assertEqual(len(agent.replay), 1)
        self.assertEqual(agent.replay[0].reward_cny, -12.5)
        self.assertIn(agent.select_weights((1.0,) * 15, epsilon=0.0), actions)


if __name__ == "__main__":
    unittest.main()
