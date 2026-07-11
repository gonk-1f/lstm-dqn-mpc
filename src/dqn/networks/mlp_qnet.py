from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPQNetwork(nn.Module):
    """Lightweight multilayer perceptron used as the default DQN backend."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: tuple[int, ...] = (128, 64)):
        super().__init__()
        if len(hidden_dims) < 1:
            raise ValueError("MLPQNetwork requires at least one hidden layer.")
        self.register_buffer("input_norm_mean", torch.zeros(int(state_dim), dtype=torch.float32))
        self.register_buffer("input_norm_std", torch.ones(int(state_dim), dtype=torch.float32))
        self.input_normalization_enabled = False
        dims = [int(state_dim), *[int(v) for v in hidden_dims], int(action_dim)]
        self.layers = nn.ModuleList(
            [nn.Linear(dims[idx], dims[idx + 1]) for idx in range(len(dims) - 1)]
        )

    @property
    def normalizer(self):
        return self if bool(self.input_normalization_enabled) else None

    @property
    def mean(self) -> torch.Tensor:
        return self.input_norm_mean

    @property
    def std(self) -> torch.Tensor:
        return self.input_norm_std

    def set_input_normalization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.input_norm_mean.copy_(mean.detach().to(device=self.input_norm_mean.device, dtype=torch.float32))
        self.input_norm_std.copy_(
            torch.clamp(std.detach().to(device=self.input_norm_std.device, dtype=torch.float32), min=1e-6)
        )
        self.input_normalization_enabled = True

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = state
        if bool(self.input_normalization_enabled):
            x = (x - self.input_norm_mean.to(dtype=x.dtype)) / self.input_norm_std.to(dtype=x.dtype)
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)
