from __future__ import annotations

from dataclasses import asdict

from dqn.networks.kan_qnet import KANNetworkConfig, KANQNetwork
from dqn.networks.kan_v2_qnet import KANV2NetworkConfig, KANV2QNetwork
from dqn.networks.mlp_qnet import MLPQNetwork
from dqn.networks.sine_kan_qnet import SineKANNetworkConfig, SineKANQNetwork


def build_q_network(
    state_dim: int,
    action_dim: int,
    *,
    network_type: str,
    mlp_hidden_dims: tuple[int, ...],
    kan_config: KANNetworkConfig,
    kan_v2_config: KANV2NetworkConfig | None = None,
    sine_kan_config: SineKANNetworkConfig | None = None,
    device_name: str,
):
    network_name = str(network_type).strip().lower()
    if network_name == "mlp":
        return MLPQNetwork(state_dim=state_dim, action_dim=action_dim, hidden_dims=mlp_hidden_dims)
    if network_name == "kan":
        return KANQNetwork(state_dim=state_dim, action_dim=action_dim, config=kan_config, device_name=device_name)
    if network_name == "kan_v2":
        if kan_v2_config is None:
            kan_v2_config = KANV2NetworkConfig()
        return KANV2QNetwork(state_dim=state_dim, action_dim=action_dim, config=kan_v2_config)
    if network_name == "sine_kan":
        if sine_kan_config is None:
            sine_kan_config = SineKANNetworkConfig()
        return SineKANQNetwork(state_dim=state_dim, action_dim=action_dim, config=sine_kan_config)
    raise ValueError(f"Unsupported DQN network_type: {network_type}")


def describe_q_network_config(
    *,
    network_type: str,
    mlp_hidden_dims: tuple[int, ...],
    kan_config: KANNetworkConfig,
    kan_v2_config: KANV2NetworkConfig | None = None,
    sine_kan_config: SineKANNetworkConfig | None = None,
) -> dict[str, object]:
    data: dict[str, object] = {
        "network_type": str(network_type).strip().lower(),
        "mlp_hidden_dims": [int(v) for v in mlp_hidden_dims],
    }
    data.update({f"kan_{key}": value for key, value in asdict(kan_config).items()})
    if kan_v2_config is not None:
        data.update({f"kan_v2_{key}": value for key, value in asdict(kan_v2_config).items()})
    if sine_kan_config is not None:
        data.update({f"sine_kan_{key}": value for key, value in asdict(sine_kan_config).items()})
    return data
