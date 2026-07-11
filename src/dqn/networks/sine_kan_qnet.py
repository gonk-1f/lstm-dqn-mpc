"""SineKAN Q-network for DQN — sinusoidal basis functions replace B-splines.

Integrates SineKANLayer (from SineKAN-main/sine_kan.py by ereinha) into the
proven DQN pipeline: Normalizer → Embedding → SineKANLayer → Dueling heads.
Retains dropout, gradient clipping, and state normalisation from the existing
mature DQN modules.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Import SineKANLayer from the SineKAN-main folder ──────────────────
_SINEKAN_ROOT = Path(__file__).resolve().parents[3] / "SineKAN-main"
if str(_SINEKAN_ROOT) not in sys.path:
    sys.path.insert(0, str(_SINEKAN_ROOT))
from sine_kan import SineKANLayer  # noqa: E402


@dataclass(frozen=True)
class SineKANNetworkConfig:
    latent_dim: int = 12
    width: int = 16
    grid_size: int = 8
    dropout: float = 0.0
    dueling: bool = False
    add_bias: bool = True
    norm_freq: bool = True


class StateNormalizer(nn.Module):
    """Online-input normaliser — critical for SineKAN whose sin(·) basis
    becomes insensitive if inputs drift outside [-1, 1]."""

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


class SineKANQNetwork(nn.Module):
    """Q-network backed by a single SineKANLayer.

    Pipeline
    --------
    state (23) → Normalizer → Linear(latent) → LayerNorm → SiLU →
    Dropout → SineKANLayer(latent → width) → Dropout →
    [Dueling] value + advantage heads → Q(s, a)
    """

    def __init__(self, state_dim: int, action_dim: int, config: SineKANNetworkConfig):
        super().__init__()
        self.config = config
        self.normalizer = StateNormalizer(state_dim)

        self.embedding = nn.Linear(int(state_dim), int(config.latent_dim))
        self.embedding_norm = nn.LayerNorm(int(config.latent_dim))
        self.dropout = nn.Dropout(float(config.dropout)) if config.dropout > 0 else nn.Identity()

        self.sine_kan = SineKANLayer(
            input_dim=int(config.latent_dim),
            output_dim=int(config.width),
            device='cpu',  # let parent .to() handle the actual device
            grid_size=int(config.grid_size),
            is_first=False,
            add_bias=bool(config.add_bias),
            norm_freq=bool(config.norm_freq),
        )

        self.dueling = bool(config.dueling)
        if self.dueling:
            self.value_head = nn.Linear(int(config.width), 1)
            self.advantage_head = nn.Linear(int(config.width), int(action_dim))
            self.head = None
        else:
            self.head = nn.Linear(int(config.width), int(action_dim))

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
        """Called by the DQN agent during observation collection."""
        self.normalizer.set_stats(mean, std)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = self.normalizer(state)
        x = F.silu(self.embedding_norm(self.embedding(x)))
        x = self.dropout(x)
        x = self.sine_kan(x)
        x = self.dropout(x)
        if self.dueling:
            value = self.value_head(x)
            advantage = self.advantage_head(x)
            return value + advantage - advantage.mean(dim=1, keepdim=True)
        assert self.head is not None
        return self.head(x)
