from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
MAIN_ROOT = SRC / "main"

for path in (SRC, MAIN_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


from dqn.agents.dqn_agent import (  # noqa: E402
    DQNAgent,
    DQNTrainConfig,
)
from dqn.memory.replay_buffer import ReplayBuffer  # noqa: E402
from dqn.networks.mlp_qnet import MLPQNetwork  # noqa: E402
from dqn.policies.epsilon_greedy import (  # noqa: E402
    EpsilonGreedyPolicy,
)
from dqn.utils.action_mapper import (  # noqa: E402
    DQN_MPC_WEIGHT_ACTIONS,
)
from envs.dqn_mpc_weight_env import (  # noqa: E402
    DqnMpcWeightEnv,
)
from run_mpc_1s_n6_four_objective_sensitivity import (  # noqa: E402
    build_sensitivity_cases,
    four_objective_config,
)


SEED = 123
STATE_DIM = 11
ACTION_DIM = 7


def make_agent() -> DQNAgent:
    config = DQNTrainConfig(
        network_type="mlp",
        state_normalization_enabled=False,
        device="cpu",
    )
    return DQNAgent(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        config=config,
    )


def state_dict_copy(
    module: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().clone()
        for name, value in module.state_dict().items()
    }


def parameter_dict_copy(
    module: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().clone()
        for name, value in module.named_parameters()
    }


def state_dicts_equal(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
) -> bool:
    return (
        left.keys() == right.keys()
        and all(
            torch.equal(left[name], right[name])
            for name in left
        )
    )


class TestDqnMpcMlpSmoke(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(SEED)
        torch.manual_seed(SEED)

    def test_mlp_network_interface(self) -> None:
        agent = make_agent()
        state = np.zeros(STATE_DIM, dtype=np.float32)
        state_tensor = torch.as_tensor(
            state,
            dtype=agent.tensor_dtype,
            device=agent.device,
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = agent.q_net(state_tensor)

        self.assertEqual(
            ACTION_DIM,
            len(DQN_MPC_WEIGHT_ACTIONS),
        )
        self.assertIsInstance(agent.q_net, MLPQNetwork)
        self.assertIsInstance(
            agent.target_q_net,
            MLPQNetwork,
        )
        layer_dimensions = [
            (layer.in_features, layer.out_features)
            for layer in agent.q_net.layers
        ]
        self.assertEqual(
            layer_dimensions,
            [(11, 128), (128, 64), (64, ACTION_DIM)],
        )
        self.assertEqual(q_values.shape, (1, ACTION_DIM))
        self.assertEqual(
            tuple(agent.config.mlp_hidden_dims),
            (128, 64),
        )
        self.assertFalse(
            agent.config.state_normalization_enabled
        )
        self.assertFalse(
            agent.q_net.input_normalization_enabled
        )
        self.assertFalse(
            agent.target_q_net.input_normalization_enabled
        )
        self.assertFalse(agent.config.double_dqn)
        self.assertFalse(agent.config.dueling_dqn)

        action = agent.greedy_action(state)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, ACTION_DIM)

    def test_environment_replay_update_and_target_sync(
        self,
    ) -> None:
        agent = make_agent()
        policy = EpsilonGreedyPolicy(
            agent.config.epsilon_start,
            agent.config.epsilon_min,
            agent.config.epsilon_decay,
        )
        replay = ReplayBuffer(agent.config.buffer_size)

        sample_count = agent.config.batch_size + 1
        phase = np.linspace(
            0.0,
            2.0 * np.pi,
            sample_count,
            dtype=np.float64,
        )
        loads_kw = 250.0 + 35.0 * np.sin(phase)
        mpc_config = four_objective_config(
            build_sensitivity_cases()[0]
        )
        env = DqnMpcWeightEnv(
            loads_kw=loads_kw,
            base_config=mpc_config,
            initial_soc=0.55,
        )

        state = env.reset()
        self.assertEqual(state.shape, (STATE_DIM,))

        done = False
        transition_count = 0
        while not done:
            greedy_action = agent.greedy_action(state)
            action = policy.select_action(
                greedy_action=greedy_action,
                action_dim=ACTION_DIM,
            )
            next_state, reward, done, _ = env.step(action)

            self.assertEqual(next_state.shape, (STATE_DIM,))
            self.assertGreaterEqual(action, 0)
            self.assertLess(action, ACTION_DIM)
            self.assertTrue(np.isfinite(reward))

            replay.push(
                state,
                action,
                reward,
                done,
                next_state,
            )
            state = next_state
            policy.step()
            transition_count += 1

        self.assertEqual(
            transition_count,
            loads_kw.size - 1,
        )
        self.assertEqual(len(replay), transition_count)
        self.assertTrue(replay.dones[-1])
        terminal_next_state = np.asarray(
            replay.next_states[-1],
            dtype=np.float32,
        ).copy()

        reset_state = env.reset()
        self.assertEqual(reset_state.shape, (STATE_DIM,))
        self.assertEqual(env.decision_index, 0)
        self.assertFalse(env.done)
        self.assertEqual(len(replay), transition_count)
        np.testing.assert_array_equal(
            replay.next_states[-1],
            terminal_next_state,
        )
        self.assertFalse(
            np.array_equal(
                terminal_next_state,
                reset_state,
            )
        )

        batch = replay.sample(agent.config.batch_size)
        states, actions, rewards, dones, next_states = batch
        batch_size = agent.config.batch_size
        self.assertEqual(states.shape, (batch_size, STATE_DIM))
        self.assertEqual(actions.shape, (batch_size,))
        self.assertEqual(rewards.shape, (batch_size,))
        self.assertEqual(dones.shape, (batch_size,))
        self.assertEqual(
            next_states.shape,
            (batch_size, STATE_DIM),
        )

        initial_online = parameter_dict_copy(agent.q_net)
        initial_target = state_dict_copy(agent.target_q_net)
        self.assertTrue(
            state_dicts_equal(
                state_dict_copy(agent.q_net),
                initial_target,
            )
        )

        loss = agent.update(batch)
        self.assertTrue(np.isfinite(loss))

        updated_online = parameter_dict_copy(agent.q_net)
        target_before_sync = state_dict_copy(
            agent.target_q_net
        )
        self.assertFalse(
            state_dicts_equal(initial_online, updated_online)
        )
        self.assertTrue(
            state_dicts_equal(
                initial_target,
                target_before_sync,
            )
        )
        self.assertFalse(
            state_dicts_equal(
                state_dict_copy(agent.q_net),
                target_before_sync,
            )
        )

        agent.sync_target_network()

        synced_target = parameter_dict_copy(
            agent.target_q_net
        )
        self.assertTrue(
            state_dicts_equal(updated_online, synced_target)
        )

    def test_terminal_bellman_target_equals_reward(
        self,
    ) -> None:
        agent = make_agent()
        rewards = torch.tensor(
            [1.25, -0.75],
            dtype=agent.tensor_dtype,
            device=agent.device,
        )
        dones = torch.tensor(
            [1.0, 0.0],
            dtype=agent.tensor_dtype,
            device=agent.device,
        )
        next_states = torch.randn(
            2,
            STATE_DIM,
            dtype=agent.tensor_dtype,
            device=agent.device,
        )

        with torch.no_grad():
            for parameter in agent.q_net.parameters():
                parameter.zero_()
            for parameter in agent.target_q_net.parameters():
                parameter.zero_()
            agent.target_q_net.layers[-1].bias.copy_(
                torch.arange(
                    1,
                    ACTION_DIM + 1,
                    dtype=agent.tensor_dtype,
                    device=agent.device,
                )
            )

        targets = agent.bellman_target(
            rewards,
            dones,
            next_states,
        )

        expected = torch.tensor(
            [
                float(rewards[0]),
                float(rewards[1])
                + agent.discount * float(ACTION_DIM),
            ],
            dtype=agent.tensor_dtype,
            device=agent.device,
        )
        torch.testing.assert_close(
            targets,
            expected,
            rtol=1.0e-6,
            atol=1.0e-6,
        )


if __name__ == "__main__":
    unittest.main()
