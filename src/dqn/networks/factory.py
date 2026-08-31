from __future__ import annotations

from dataclasses import asdict

from dqn.networks.kan_qnet import KANNetworkConfig, KANQNetwork
from dqn.networks.mlp_qnet import MLPQNetwork


def build_q_network(
    state_dim: int,
    action_dim: int,
    *,
    network_type: str,
    mlp_hidden_dims: tuple[int, ...],
    kan_config: KANNetworkConfig,
    device_name: str,
):
    network_name = str(network_type).strip().lower()
    if network_name == "mlp":
        return MLPQNetwork(state_dim=state_dim, action_dim=action_dim, hidden_dims=mlp_hidden_dims)
    if network_name == "kan":
        return KANQNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            config=kan_config,
        )
    raise ValueError(f"Unsupported DQN network_type: {network_type}")


def describe_q_network_config(
    *,
    network_type: str,
    mlp_hidden_dims: tuple[int, ...],
    kan_config: KANNetworkConfig,
) -> dict[str, object]:
    data: dict[str, object] = {
        "network_type": str(network_type).strip().lower(),
        "mlp_hidden_dims": [int(v) for v in mlp_hidden_dims],
    }
    data.update({f"kan_{key}": value for key, value in asdict(kan_config).items()})
    return data
