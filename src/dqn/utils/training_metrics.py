"""Bounded diagnostic statistics, independent of the learning algorithm."""
from __future__ import annotations

from collections import deque

import numpy as np


class LossAccumulator:
    """Exact all-update mean/extrema, with an explicitly recent-only median."""

    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.minimum = float('inf')
        self.maximum = float('-inf')
        self.recent = deque(maxlen=1000)

    def add(self, values) -> None:
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if not np.isfinite(values).all():
            raise RuntimeError('training loss contains NaN or Inf')
        if not values.size:
            return
        self.count += int(values.size)
        self.total += float(values.sum())
        self.minimum = min(self.minimum, float(values.min()))
        self.maximum = max(self.maximum, float(values.max()))
        self.recent.extend(float(value) for value in values)

    def summary(self) -> dict:
        recent = np.asarray(self.recent)
        return dict(count=self.count, mean=self.total / self.count if self.count else None,
            min=self.minimum if self.count else None, max=self.maximum if self.count else None,
            median=float(np.median(recent)) if self.count else None,
            median_scope='all_updates' if self.count <= 1000 else 'recent_1000',
            last=float(recent[-1]) if self.count else None,
            recent_1000_mean=float(recent.mean()) if self.count else None)

    def state_dict(self) -> dict:
        return dict(count=self.count, total=self.total, minimum=self.minimum,
                    maximum=self.maximum, recent=list(self.recent))

    def load_state_dict(self, state: dict) -> None:
        self.count = int(state['count'])
        self.total = float(state['total'])
        self.minimum = float(state['minimum'])
        self.maximum = float(state['maximum'])
        self.recent = deque(state['recent'], maxlen=1000)
