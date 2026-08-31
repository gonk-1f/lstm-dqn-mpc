from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class KANNetworkConfig:
    backend: str = "efficient_kan"
    latent_dim: int = 12
    width: int = 16
    depth: int = 1
    grid: int = 4
    basis_min: float = -2.5
    basis_max: float = 2.5
    freeze_base_scale: bool = False
    freeze_spline_scale: bool = True
    freeze_spline_weight: bool = False
    spline_init_scale: float = 0.02
    dropout: float = 0.0
    dueling: bool = False


class _StateNormalizer(nn.Module):
    def __init__(self, state_dim: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(int(state_dim), dtype=torch.float32))
        self.register_buffer("std", torch.ones(int(state_dim), dtype=torch.float32))

    def set_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.mean.copy_(mean.to(torch.float32))
        self.std.copy_(torch.clamp(std.to(torch.float32), min=1.0e-6))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return (state.to(torch.float32) - self.mean) / self.std


class _Scale(nn.Module):
    def __init__(self, value: float, trainable: bool):
        super().__init__()
        tensor = torch.tensor(float(value), dtype=torch.float32)
        if trainable:
            self.value = nn.Parameter(tensor)
        else:
            self.register_buffer("value", tensor)

    def forward(self) -> torch.Tensor:
        return self.value


class _CompactKANLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        config: KANNetworkConfig,
    ) -> None:
        super().__init__()
        backend = str(config.backend).strip().lower()
        if backend not in {"efficient_kan", "fast_kan"}:
            raise ValueError(f"Unsupported KAN backend: {config.backend}")
        if int(config.grid) < 2:
            raise ValueError("KAN grid must be at least 2")

        self.backend = backend
        self.grid = int(config.grid)
        self.base = nn.Linear(int(in_dim), int(out_dim))
        self.spline_weight = nn.Parameter(
            torch.empty(int(out_dim), int(in_dim), self.grid)
        )
        centers = torch.linspace(
            float(config.basis_min),
            float(config.basis_max),
            self.grid,
            dtype=torch.float32,
        )
        spacing = (
            float(config.basis_max) - float(config.basis_min)
        ) / max(1, self.grid - 1)
        self.register_buffer("centers", centers)
        self.register_buffer(
            "spacing",
            torch.tensor(max(spacing, 1.0e-6), dtype=torch.float32),
        )
        self.base_scale = _Scale(
            1.0,
            trainable=not bool(config.freeze_base_scale),
        )
        self.spline_scale = _Scale(
            1.0,
            trainable=not bool(config.freeze_spline_scale),
        )

        nn.init.kaiming_uniform_(self.base.weight, a=5**0.5)
        nn.init.zeros_(self.base.bias)
        nn.init.normal_(
            self.spline_weight,
            mean=0.0,
            std=float(config.spline_init_scale),
        )
        if bool(config.freeze_spline_weight):
            self.spline_weight.requires_grad_(False)

    def _basis(self, values: torch.Tensor) -> torch.Tensor:
        distance = values.unsqueeze(-1) - self.centers
        if self.backend == "fast_kan":
            return torch.exp(-0.5 * torch.square(distance / self.spacing))
        return torch.relu(1.0 - torch.abs(distance) / self.spacing)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        base = F.silu(self.base(values))
        spline = torch.einsum(
            "big,oig->bo",
            self._basis(values),
            self.spline_weight,
        )
        return self.base_scale() * base + self.spline_scale() * spline


class KANQNetwork(nn.Module):
    """Self-contained Torch KAN backend with the same 7-to-4 contract as MLP."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: KANNetworkConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.normalizer = _StateNormalizer(state_dim)
        self.embedding = nn.Linear(int(state_dim), int(config.latent_dim))
        self.embedding_norm = nn.LayerNorm(int(config.latent_dim))
        self.dropout = (
            nn.Dropout(float(config.dropout))
            if float(config.dropout) > 0.0
            else nn.Identity()
        )

        layers: list[nn.Module] = []
        in_dim = int(config.latent_dim)
        for _ in range(max(1, int(config.depth))):
            layers.append(
                _CompactKANLayer(
                    in_dim,
                    int(config.width),
                    config=config,
                )
            )
            in_dim = int(config.width)
        self.kan_layers = nn.ModuleList(layers)
        self.dueling = bool(config.dueling)
        if self.dueling:
            self.value_head = nn.Linear(in_dim, 1)
            self.advantage_head = nn.Linear(in_dim, int(action_dim))
            self.head = None
        else:
            self.head = nn.Linear(in_dim, int(action_dim))

        nn.init.kaiming_uniform_(self.embedding.weight, a=5**0.5)
        nn.init.zeros_(self.embedding.bias)
        if self.dueling:
            nn.init.kaiming_uniform_(self.value_head.weight, a=5**0.5)
            nn.init.zeros_(self.value_head.bias)
            nn.init.kaiming_uniform_(self.advantage_head.weight, a=5**0.5)
            nn.init.zeros_(self.advantage_head.bias)
        else:
            assert self.head is not None
            nn.init.kaiming_uniform_(self.head.weight, a=5**0.5)
            nn.init.zeros_(self.head.bias)
        self.float()

    def set_input_normalization(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> None:
        self.normalizer.set_stats(mean, std)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        values = self.normalizer(state)
        values = F.silu(self.embedding_norm(self.embedding(values)))
        values = self.dropout(values)
        for layer in self.kan_layers:
            values = self.dropout(layer(values))
        if self.dueling:
            value = self.value_head(values)
            advantage = self.advantage_head(values)
            return value + advantage - advantage.mean(dim=1, keepdim=True)
        assert self.head is not None
        return self.head(values)
