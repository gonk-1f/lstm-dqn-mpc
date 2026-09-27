"""Bounded causal history encoder for the formal DQN state."""

from __future__ import annotations

from collections import deque
import math

from .state import (
    FORMAL_FRAME_DIMENSION,
    FORMAL_STATE_DIMENSION,
    FORMAL_STATE_HISTORY_LENGTH,
)


def _validate_frame(frame: object) -> tuple[float, ...]:
    if type(frame) is not tuple:
        raise TypeError("formal frame must be an exact tuple")
    if len(frame) != FORMAL_FRAME_DIMENSION:
        raise ValueError(f"formal frame must contain {FORMAL_FRAME_DIMENSION} values")
    for value in frame:
        if type(value) is not float:
            raise TypeError("formal frame values must be exact floats")
        if not math.isfinite(value):
            raise ValueError("formal frame values must be finite")
    return frame


class FormalStateHistory:
    """Store nine completed frames and encode them with the current frame."""

    def __init__(self) -> None:
        self._frames: deque[tuple[float, ...]] = deque(
            maxlen=FORMAL_STATE_HISTORY_LENGTH - 1
        )

    @property
    def committed_count(self) -> int:
        return len(self._frames)

    def reset(self) -> None:
        self._frames.clear()

    def commit(self, frame: tuple[float, ...]) -> None:
        self._frames.append(_validate_frame(frame))

    def encode(self, current_frame: tuple[float, ...]) -> tuple[float, ...]:
        current = _validate_frame(current_frame)
        frames = (tuple(self._frames) + (current,))[-FORMAL_STATE_HISTORY_LENGTH:]
        missing = FORMAL_STATE_HISTORY_LENGTH - len(frames)
        values = (0.0,) * (missing * FORMAL_FRAME_DIMENSION)
        values += tuple(value for frame in frames for value in frame)
        mask = (0.0,) * missing + (1.0,) * len(frames)
        encoded = values + mask
        if len(encoded) != FORMAL_STATE_DIMENSION:
            raise RuntimeError("formal history encoder produced the wrong dimension")
        return encoded


__all__ = ["FormalStateHistory"]
