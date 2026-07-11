from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class KANV2NetworkConfig:
    backend: str = "efficient_kan"
    latent_dim: int = 12
    width: int = 16
    depth: int = 1
    grid: int = 5
    basis_min: float = -2.5
    basis_max: float = 2.5
    freeze_base_scale: bool = False
    freeze_spline_scale: bool = True
    freeze_spline_weight: bool = False
    spline_init_scale: float = 0.02
    dropout: float = 0.0
    dueling: bool = False


class StateNormalizer(nn.Module):
    def __init__(self, state_dim: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(int(state_dim), dtype=torch.float32))
        self.register_buffer("std", torch.ones(int(state_dim), dtype=torch.float32))

    def set_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        std = torch.clamp(std.to(torch.float32), min=1e-6)
        self.mean.copy_(mean.to(torch.float32))
        self.std.copy_(std)

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


class CompactKANLayer(nn.Module):
    """Small KAN-style layer optimized for DQN value approximation.

    This local layer avoids the slow pykan symbolic/path bookkeeping. It uses a
    base linear term plus an edge-wise basis expansion. `efficient_kan` uses a
    compact triangular basis; `fast_kan` uses Gaussian RBF features.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        backend: str,
        grid: int,
        basis_min: float,
        basis_max: float,
        freeze_base_scale: bool,
        freeze_spline_scale: bool,
        freeze_spline_weight: bool,
        spline_init_scale: float,
    ):
        super().__init__()
        backend_name = str(backend).strip().lower()
        if backend_name not in {"efficient_kan", "fast_kan"}:
            raise ValueError(f"Unsupported KAN v2 backend: {backend}")
        self.backend = backend_name
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.grid = int(grid)
        if self.grid < 2:
            raise ValueError("KAN v2 grid must be at least 2.")

        centers = torch.linspace(float(basis_min), float(basis_max), self.grid, dtype=torch.float32)
        spacing = float((basis_max - basis_min) / max(1, self.grid - 1))
        self.register_buffer("centers", centers)
        self.register_buffer("spacing", torch.tensor(max(spacing, 1e-6), dtype=torch.float32))

        self.base = nn.Linear(self.in_dim, self.out_dim)
        self.spline_weight = nn.Parameter(torch.empty(self.out_dim, self.in_dim, self.grid))
        self.base_scale = _Scale(1.0, trainable=not freeze_base_scale)
        self.spline_scale = _Scale(1.0, trainable=not freeze_spline_scale)

        nn.init.kaiming_uniform_(self.base.weight, a=5**0.5)
        nn.init.zeros_(self.base.bias)
        nn.init.normal_(self.spline_weight, mean=0.0, std=float(spline_init_scale))
        if freeze_spline_weight:
            self.spline_weight.requires_grad_(False)

    def _basis(self, x: torch.Tensor) -> torch.Tensor:
        distance = x.unsqueeze(-1) - self.centers
        if self.backend == "fast_kan":
            return torch.exp(-0.5 * torch.square(distance / self.spacing))
        return torch.relu(1.0 - torch.abs(distance) / self.spacing)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.silu(self.base(x))
        basis = self._basis(x)
        spline = torch.einsum("big,oig->bo", basis, self.spline_weight)
        return self.base_scale() * base + self.spline_scale() * spline


class KANV2QNetwork(nn.Module):
    """Stabilized KAN Q-network for the ship DQN branch."""

    def __init__(self, state_dim: int, action_dim: int, config: KANV2NetworkConfig):
        super().__init__()
        self.config = config
        self.normalizer = StateNormalizer(state_dim)
        self.embedding = nn.Linear(int(state_dim), int(config.latent_dim))
        self.embedding_norm = nn.LayerNorm(int(config.latent_dim))
        self.dropout = nn.Dropout(float(config.dropout)) if config.dropout > 0 else nn.Identity()

        layers: list[nn.Module] = []
        in_dim = int(config.latent_dim)
        for _ in range(max(1, int(config.depth))):
            layers.append(
                CompactKANLayer(
                    in_dim,
                    int(config.width),
                    backend=str(config.backend),
                    grid=int(config.grid),
                    basis_min=float(config.basis_min),
                    basis_max=float(config.basis_max),
                    freeze_base_scale=bool(config.freeze_base_scale),
                    freeze_spline_scale=bool(config.freeze_spline_scale),
                    freeze_spline_weight=bool(config.freeze_spline_weight),
                    spline_init_scale=float(config.spline_init_scale),
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

    def set_input_normalization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.normalizer.set_stats(mean, std)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = self.normalizer(state)
        x = F.silu(self.embedding_norm(self.embedding(x)))
        x = self.dropout(x)
        for layer in self.kan_layers:
            x = layer(x)
            x = self.dropout(x)
        if self.dueling:
            value = self.value_head(x)
            advantage = self.advantage_head(x)
            return value + advantage - advantage.mean(dim=1, keepdim=True)
        assert self.head is not None
        return self.head(x)
