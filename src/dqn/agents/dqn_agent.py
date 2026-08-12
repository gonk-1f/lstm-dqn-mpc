from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dqn.networks import KANNetworkConfig, KANV2NetworkConfig, build_q_network, describe_q_network_config
from dqn.networks.sine_kan_qnet import SineKANNetworkConfig


@dataclass
class DQNTrainConfig:
    seed: int = 42
    discount: float = 0.99
    lr: float = 5e-4
    batch_size: int = 64
    max_steps: int = 10000
    warmup_steps: int = 5000
    buffer_size: int = 100000
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.99999813
    target_sync_interval: int = 500    # hard-sync target network every N steps
    device: str = "auto"
    log_window_steps: int = 1000
    grad_clip_norm: float = 10.0
    solver_failure_reward: float = -620.0
    loss_type: str = "mse"
    network_type: str = "mlp"
    mlp_hidden_dims: tuple[int, ...] = (128, 64)
    kan_hidden_dims: tuple[int, ...] = (32,)
    kan_grid: int = 4
    kan_order: int = 3
    kan_symbolic_enabled: bool = False
    kan_bias_trainable: bool = False
    kan_sp_trainable: bool = False
    kan_sb_trainable: bool = False
    double_dqn: bool = False
    kan_grid_update_enabled: bool = False
    kan_grid_update_interval_steps: int = 0
    kan_grid_update_until_step: int = 0
    kan_grid_update_samples: int = 256
    kan_backend: str = "efficient_kan"
    kan_v2_latent_dim: int = 12
    kan_v2_width: int = 16
    kan_v2_depth: int = 1
    kan_v2_grid: int = 4
    kan_v2_basis_min: float = -2.5
    kan_v2_basis_max: float = 2.5
    kan_v2_freeze_base_scale: bool = False
    kan_v2_freeze_spline_scale: bool = True
    kan_v2_freeze_spline_weight: bool = False
    kan_v2_spline_init_scale: float = 0.02
    kan_v2_dropout: float = 0.0
    dueling_dqn: bool = False
    state_normalization_enabled: bool = False
    state_normalization_min_count: int = 32
    # SineKAN
    sine_kan_latent_dim: int = 12
    sine_kan_width: int = 16
    sine_kan_grid_size: int = 8
    sine_kan_dropout: float = 0.0
    sine_kan_dueling: bool = False


def resolve_torch_device(device: str) -> str:
    requested = device.lower().strip()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return requested


def _remap_legacy_mlp_state_dict_keys(state_dict: dict) -> dict:
    if any(str(key).startswith("layers.") for key in state_dict):
        return state_dict

    remapped: dict = {}
    remapped_any = False
    for key, value in state_dict.items():
        key_str = str(key)
        if key_str.startswith("fc"):
            layer_part, sep, suffix = key_str.partition(".")
            if sep and layer_part[2:].isdigit():
                layer_idx = int(layer_part[2:]) - 1
                remapped[f"layers.{layer_idx}.{suffix}"] = value
                remapped_any = True
                continue
        remapped[key] = value
    return remapped if remapped_any else state_dict


class DQNAgent:
    """DQN agent for selecting complete MPC four-weight actions.

    Physical interpretation of `Q(s, a)` in this project:
    `Q(s, a)` estimates the expected discounted cumulative control value when
    the vessel microgrid is currently in state `s` and action `a` selects one
    complete MPC weight tuple, then follows the learned policy afterwards.
    A larger Q-value means that the action is expected to yield better long-run
    tracking, load-balance, SOC-safety, and smoothness performance.
    """

    def __init__(self, state_dim: int, action_dim: int, config: DQNTrainConfig):
        self.config = config
        self.device_name = resolve_torch_device(config.device)
        if str(config.network_type).strip().lower() == "kan" and self.device_name.startswith("cuda"):
            # pykan 0.0.2 keeps some spline-layer device markers on CPU even
            # after module.to("cuda"), so the KAN backend is kept on CPU.
            self.device_name = "cpu"
        self.device = torch.device(self.device_name)
        self.kan_config = KANNetworkConfig(
            hidden_dims=tuple(int(v) for v in config.kan_hidden_dims),
            grid=int(config.kan_grid),
            order=int(config.kan_order),
            symbolic_enabled=bool(config.kan_symbolic_enabled),
            bias_trainable=bool(config.kan_bias_trainable),
            sp_trainable=bool(config.kan_sp_trainable),
            sb_trainable=bool(config.kan_sb_trainable),
        )
        self.kan_v2_config = KANV2NetworkConfig(
            backend=str(config.kan_backend),
            latent_dim=int(config.kan_v2_latent_dim),
            width=int(config.kan_v2_width),
            depth=int(config.kan_v2_depth),
            grid=int(config.kan_v2_grid),
            basis_min=float(config.kan_v2_basis_min),
            basis_max=float(config.kan_v2_basis_max),
            freeze_base_scale=bool(config.kan_v2_freeze_base_scale),
            freeze_spline_scale=bool(config.kan_v2_freeze_spline_scale),
            freeze_spline_weight=bool(config.kan_v2_freeze_spline_weight),
            spline_init_scale=float(config.kan_v2_spline_init_scale),
            dropout=float(config.kan_v2_dropout),
            dueling=bool(config.dueling_dqn),
        )
        self.sine_kan_config = SineKANNetworkConfig(
            latent_dim=int(config.sine_kan_latent_dim),
            width=int(config.sine_kan_width),
            grid_size=int(config.sine_kan_grid_size),
            dropout=float(config.sine_kan_dropout),
            dueling=bool(config.sine_kan_dueling),
        )
        self.q_net = build_q_network(
            state_dim,
            action_dim,
            network_type=config.network_type,
            mlp_hidden_dims=tuple(int(v) for v in config.mlp_hidden_dims),
            kan_config=self.kan_config,
            kan_v2_config=self.kan_v2_config,
            sine_kan_config=self.sine_kan_config,
            device_name=self.device_name,
        ).to(self.device)
        self.target_q_net = build_q_network(
            state_dim,
            action_dim,
            network_type=config.network_type,
            mlp_hidden_dims=tuple(int(v) for v in config.mlp_hidden_dims),
            kan_config=self.kan_config,
            kan_v2_config=self.kan_v2_config,
            sine_kan_config=self.sine_kan_config,
            device_name=self.device_name,
        ).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        trainable_parameters = [param for param in self.q_net.parameters() if param.requires_grad]
        self.optimizer = torch.optim.Adam(trainable_parameters, lr=config.lr)
        self.discount = config.discount
        self.action_dim = action_dim
        self.tensor_dtype = next(self.q_net.parameters()).dtype
        self.latest_update_diagnostics: dict[str, float] = {}
        self._state_norm_count = 0
        self._state_norm_mean: np.ndarray | None = None
        self._state_norm_m2: np.ndarray | None = None
        self.network_info = describe_q_network_config(
            network_type=config.network_type,
            mlp_hidden_dims=tuple(int(v) for v in config.mlp_hidden_dims),
            kan_config=self.kan_config,
            kan_v2_config=self.kan_v2_config,
            sine_kan_config=self.sine_kan_config,
        )
        self.network_info.update(
            {
                "state_normalization_enabled": bool(config.state_normalization_enabled),
                "state_normalization_min_count": int(config.state_normalization_min_count),
                "dueling_dqn": bool(config.dueling_dqn),
            }
        )

    def greedy_action(self, state: np.ndarray) -> int:
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=self.tensor_dtype, device=self.device).unsqueeze(0)
            action = self.q_net(state_tensor).argmax(dim=1).item()
        return int(action)

    def bellman_target(self, rewards: torch.Tensor, dones: torch.Tensor, next_states: torch.Tensor) -> torch.Tensor:
        """Compute the one-step Bellman target:

        y_t = r_t + gamma * max_a' Q_target(s_{t+1}, a') * (1 - done_t)
        """
        if self.config.double_dqn:
            next_actions = self.q_net(next_states).detach().argmax(dim=1, keepdim=True)
            next_q_values = self.target_q_net(next_states).detach().gather(1, next_actions).squeeze(1)
        else:
            next_q_values = self.target_q_net(next_states).detach().max(dim=1).values
        return rewards + self.discount * next_q_values * (1 - dones)

    def compute_loss(self, batch) -> torch.Tensor:
        states, actions, rewards, dones, next_states = batch
        states = torch.tensor(states, dtype=self.tensor_dtype, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=self.tensor_dtype, device=self.device)
        dones = torch.tensor(dones, dtype=self.tensor_dtype, device=self.device)
        next_states = torch.tensor(next_states, dtype=self.tensor_dtype, device=self.device)

        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        target = self.bellman_target(rewards, dones, next_states)
        with torch.no_grad():
            self.latest_update_diagnostics = {
                "q_value_mean": float(q_values.detach().mean().item()),
                "q_value_std": float(q_values.detach().std(unbiased=False).item()),
                "target_q_mean": float(target.detach().mean().item()),
                "target_q_std": float(target.detach().std(unbiased=False).item()),
            }
        if self.config.loss_type.lower() == "mse":
            return F.mse_loss(q_values, target)
        return F.smooth_l1_loss(q_values, target)

    def update(self, batch) -> float:
        loss = self.compute_loss(batch)
        self.optimizer.zero_grad()
        loss.backward()
        if self.config.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.config.grad_clip_norm)
        self.optimizer.step()
        return float(loss.item())

    def sync_target_network(self) -> None:
        self.target_q_net.load_state_dict(self.q_net.state_dict())

    def observe_states_for_normalization(self, states: np.ndarray) -> bool:
        if not bool(self.config.state_normalization_enabled):
            return False
        if not hasattr(self.q_net, "set_input_normalization"):
            return False
        arr = np.asarray(states, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.size == 0:
            return False
        if self._state_norm_mean is None:
            self._state_norm_mean = np.zeros(arr.shape[1], dtype=np.float64)
            self._state_norm_m2 = np.zeros(arr.shape[1], dtype=np.float64)
        assert self._state_norm_m2 is not None
        for row in arr:
            self._state_norm_count += 1
            delta = row - self._state_norm_mean
            self._state_norm_mean += delta / float(self._state_norm_count)
            delta2 = row - self._state_norm_mean
            self._state_norm_m2 += delta * delta2
        if self._state_norm_count < max(1, int(self.config.state_normalization_min_count)):
            return False
        variance = self._state_norm_m2 / max(1.0, float(self._state_norm_count - 1))
        std = np.sqrt(np.maximum(variance, 1e-6))
        mean_tensor = torch.tensor(self._state_norm_mean, dtype=torch.float32, device=self.device)
        std_tensor = torch.tensor(std, dtype=torch.float32, device=self.device)
        self.q_net.set_input_normalization(mean_tensor, std_tensor)
        if hasattr(self.target_q_net, "set_input_normalization"):
            self.target_q_net.set_input_normalization(mean_tensor, std_tensor)
        return True

    def update_kan_grid_from_samples(self, states: np.ndarray, update_target: bool = True) -> bool:
        if not hasattr(self.q_net, "update_grid_from_samples"):
            return False
        state_tensor = torch.tensor(states, dtype=self.tensor_dtype, device=self.device)
        self.q_net.update_grid_from_samples(state_tensor)
        if update_target and hasattr(self.target_q_net, "update_grid_from_samples"):
            self.target_q_net.update_grid_from_samples(state_tensor)
        return True

    def save(self, model_path: str | Path) -> Path:
        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.q_net.state_dict(), path)
        return path

    def input_normalization_stats(self) -> dict[str, object] | None:
        normalizer = getattr(self.q_net, "normalizer", None)
        if normalizer is None:
            return None
        return {
            "count": int(self._state_norm_count),
            "mean": [float(v) for v in normalizer.mean.detach().cpu().numpy().tolist()],
            "std": [float(v) for v in normalizer.std.detach().cpu().numpy().tolist()],
            "stored_in_model_state_dict": True,
        }

    def load(self, model_path: str | Path) -> None:
        state_dict = torch.load(model_path, map_location=self.device)
        if isinstance(state_dict, dict):
            state_dict = _remap_legacy_mlp_state_dict_keys(state_dict)
        self.q_net.load_state_dict(state_dict)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        if bool(self.config.state_normalization_enabled) and hasattr(self.q_net, "set_input_normalization"):
            mean = getattr(self.q_net, "input_norm_mean", None)
            std = getattr(self.q_net, "input_norm_std", None)
            if mean is not None and std is not None:
                self.q_net.set_input_normalization(mean.detach(), std.detach())
                if hasattr(self.target_q_net, "set_input_normalization"):
                    self.target_q_net.set_input_normalization(mean.detach(), std.detach())
