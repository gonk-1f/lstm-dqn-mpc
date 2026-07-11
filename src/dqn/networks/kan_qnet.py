from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class KANNetworkConfig:
    hidden_dims: tuple[int, ...] = (32,)
    grid: int = 4
    order: int = 3
    symbolic_enabled: bool = False
    bias_trainable: bool = False
    sp_trainable: bool = False
    sb_trainable: bool = False


class KANQNetwork(nn.Module):
    """KAN-based Q network with the same input/output contract as the MLP backend."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: KANNetworkConfig,
        device_name: str = "cpu",
    ):
        super().__init__()
        try:
            from kan import KAN
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "KAN backend requires the `pykan` package. Install it with "
                "`python -m pip install pykan==0.0.2 tqdm` inside the project environment."
            ) from exc

        width = [int(state_dim), *[int(v) for v in config.hidden_dims], int(action_dim)]
        self.model = KAN(
            width=width,
            grid=int(config.grid),
            k=int(config.order),
            symbolic_enabled=bool(config.symbolic_enabled),
            bias_trainable=bool(config.bias_trainable),
            sp_trainable=bool(config.sp_trainable),
            sb_trainable=bool(config.sb_trainable),
            device=str(device_name),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # pykan returns float64 tensors by default; cast the output back so the
        # rest of the DQN stack can keep using the existing float32 losses.
        return self.model(state.to(torch.float64)).to(torch.float32)

    def update_grid_from_samples(self, states: torch.Tensor) -> None:
        self.model.update_grid_from_samples(states.to(torch.float64))
