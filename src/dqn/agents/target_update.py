from __future__ import annotations

import torch.nn as nn


def soft_update(target: nn.Module, source: nn.Module, tau: float = 0.01) -> None:
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + source_param.data * tau)
