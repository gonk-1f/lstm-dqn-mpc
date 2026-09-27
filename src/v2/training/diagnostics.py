"""Deterministic DQN learning and action-distribution diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch

from ..dqn.action_space import FINAL_DQN_ACTION_CATALOG


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class QValueDiagnostics:
    common_mode_mean: float
    centered_advantage_std: float
    top_two_margin_p50: float

    def __post_init__(self) -> None:
        for name in (
            "common_mode_mean",
            "centered_advantage_std",
            "top_two_margin_p50",
        ):
            _finite_float(getattr(self, name), name)


@dataclass(frozen=True)
class ActionDistributionDiagnostics:
    unique_action_count: int
    max_action_share: float
    shannon_entropy: float
    action_counts: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        if type(self.unique_action_count) is not int or self.unique_action_count <= 0:
            raise ValueError("unique_action_count must be a positive exact integer")
        share = _finite_float(self.max_action_share, "max_action_share")
        entropy = _finite_float(self.shannon_entropy, "shannon_entropy")
        if not 0.0 < share <= 1.0 or entropy < 0.0:
            raise ValueError("action distribution values are outside their domains")
        if type(self.action_counts) is not tuple or len(self.action_counts) != self.unique_action_count:
            raise ValueError("action_counts must match unique_action_count")


@dataclass(frozen=True)
class DqnOptimizationDiagnostics:
    loss: float
    td_abs_p50: float
    td_abs_p95: float
    td_abs_max: float
    q_common_mean: float
    q_advantage_std: float
    q_margin_p50: float
    gradient_norm_preclip: float

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = _finite_float(getattr(self, name), name)
            if name in {
                "loss",
                "td_abs_p50",
                "td_abs_p95",
                "td_abs_max",
                "q_advantage_std",
                "q_margin_p50",
                "gradient_norm_preclip",
            } and value < 0.0:
                raise ValueError(f"{name} must be nonnegative")

    @classmethod
    def from_tensors(
        cls,
        *,
        loss: torch.Tensor,
        td_error: torch.Tensor,
        q_values: torch.Tensor,
        gradient_norm: torch.Tensor | float,
    ) -> "DqnOptimizationDiagnostics":
        if type(loss) is not torch.Tensor or loss.numel() != 1:
            raise TypeError("loss must be one scalar torch.Tensor")
        if type(td_error) is not torch.Tensor or td_error.numel() == 0:
            raise TypeError("td_error must be a nonempty torch.Tensor")
        if type(q_values) is not torch.Tensor:
            raise TypeError("q_values must be a torch.Tensor")
        td = td_error.detach().cpu().numpy().astype(float, copy=False).reshape(-1)
        if not np.isfinite(td).all():
            raise ValueError("td_error must be finite")
        q_summary = summarize_q_values(
            q_values.detach().cpu().numpy().astype(float, copy=False)
        )
        absolute = np.abs(td)
        return cls(
            loss=_finite_float(loss.detach().cpu().item(), "loss"),
            td_abs_p50=float(np.percentile(absolute, 50)),
            td_abs_p95=float(np.percentile(absolute, 95)),
            td_abs_max=float(np.max(absolute)),
            q_common_mean=q_summary.common_mode_mean,
            q_advantage_std=q_summary.centered_advantage_std,
            q_margin_p50=q_summary.top_two_margin_p50,
            gradient_norm_preclip=_finite_float(
                gradient_norm.detach().cpu().item()
                if type(gradient_norm) is torch.Tensor
                else gradient_norm,
                "gradient_norm",
            ),
        )


def summarize_q_values(q_values: object) -> QValueDiagnostics:
    values = np.asarray(q_values, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 2:
        raise ValueError("q_values must be a nonempty 2D array with two actions")
    if not np.isfinite(values).all():
        raise ValueError("q_values must be finite")
    row_means = np.mean(values, axis=1)
    centered = values - row_means[:, None]
    ordered = np.sort(values, axis=1)
    margins = ordered[:, -1] - ordered[:, -2]
    return QValueDiagnostics(
        common_mode_mean=float(np.mean(row_means)),
        centered_advantage_std=float(np.std(centered, ddof=0)),
        top_two_margin_p50=float(np.percentile(margins, 50)),
    )


def summarize_actions(
    action_indices: tuple[int, ...],
    *,
    action_dim: int,
) -> ActionDistributionDiagnostics:
    if type(action_indices) is not tuple or not action_indices:
        raise ValueError("action_indices must be a nonempty exact tuple")
    if (
        type(action_dim) is not int
        or action_dim <= 0
        or action_dim > len(FINAL_DQN_ACTION_CATALOG)
    ):
        raise ValueError("action_dim is outside the canonical catalog")
    if any(type(index) is not int for index in action_indices):
        raise TypeError("action_indices must contain exact integers")
    if any(not 0 <= index < action_dim for index in action_indices):
        raise ValueError("action index lies outside action_dim")
    counts = np.bincount(np.asarray(action_indices, dtype=np.int64), minlength=action_dim)
    present = np.flatnonzero(counts)
    probabilities = counts[present].astype(float) / len(action_indices)
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    action_counts = tuple(
        (FINAL_DQN_ACTION_CATALOG[int(index)].action_id, int(counts[index]))
        for index in present
    )
    return ActionDistributionDiagnostics(
        unique_action_count=len(present),
        max_action_share=float(np.max(probabilities)),
        shannon_entropy=entropy,
        action_counts=action_counts,
    )


__all__ = [
    "ActionDistributionDiagnostics",
    "DqnOptimizationDiagnostics",
    "QValueDiagnostics",
    "summarize_actions",
    "summarize_q_values",
]
