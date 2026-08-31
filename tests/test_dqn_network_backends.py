from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from dqn.agents.dqn_agent import DQNAgent, DQNTrainConfig  # noqa: E402
from dqn.networks import KANNetworkConfig, build_q_network  # noqa: E402


class DqnNetworkBackendTests(unittest.TestCase):
    def _build(self, network_type: str):
        return build_q_network(
            state_dim=7,
            action_dim=4,
            network_type=network_type,
            mlp_hidden_dims=(128, 64),
            kan_config=KANNetworkConfig(),
            device_name="cpu",
        )

    def test_mlp_and_kan_have_identical_io_contract(self) -> None:
        states = torch.zeros((3, 7), dtype=torch.float32)
        for network_type in ("mlp", "kan"):
            with self.subTest(network_type=network_type):
                network = self._build(network_type)
                self.assertEqual(tuple(network(states).shape), (3, 4))

    def test_factory_exposes_only_mlp_and_kan_backends(self) -> None:
        self._build("mlp")
        self._build("kan")
        for legacy_name in ("kan_v2", "sine_kan"):
            with self.subTest(network_type=legacy_name):
                with self.assertRaises(ValueError):
                    self._build(legacy_name)

    def test_agent_greedy_action_accepts_both_backends(self) -> None:
        state = np.zeros(7, dtype=np.float32)
        for network_type in ("mlp", "kan"):
            with self.subTest(network_type=network_type):
                agent = DQNAgent(
                    state_dim=7,
                    action_dim=4,
                    config=DQNTrainConfig(network_type=network_type, device="cpu"),
                )
                action = agent.greedy_action(state)
                self.assertIn(action, range(4))

    def test_network_switch_does_not_change_action_count(self) -> None:
        for network_type in ("mlp", "kan"):
            network = self._build(network_type)
            self.assertEqual(network(torch.zeros((1, 7))).shape[-1], 4)


if __name__ == "__main__":
    unittest.main()
