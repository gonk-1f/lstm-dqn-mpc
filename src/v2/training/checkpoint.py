"""Atomic, identity-bound v2 formal DQN checkpoints."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path

import torch

from ..contracts import control_semantics
from ..dqn.action_space import ACTION_CATALOG_DIGEST, FINAL_DQN_ACTION_CATALOG
from ..dqn.state import FORMAL_STATE_SCHEMA_DIGEST, FORMAL_STATE_SCHEMA_VERSION
from .dqn import DqnAgent
from .schedule import EpisodeShuffleSchedule


CHECKPOINT_VERSION = "v2_formal_dqn_checkpoint_v1"


class IncompatibleCheckpointError(ValueError):
    pass


@dataclass(frozen=True)
class ResumeMetadata:
    global_macro_step: int
    round_index: int
    episode_position: int


def save_checkpoint(
    path: Path,
    *,
    agent: DqnAgent,
    schedule: EpisodeShuffleSchedule,
    global_macro_step: int,
    round_index: int,
    episode_position: int,
) -> None:
    if type(agent) is not DqnAgent or type(schedule) is not EpisodeShuffleSchedule:
        raise TypeError("agent and schedule must use exact v2 types")
    for name, value in (
        ("global_macro_step", global_macro_step),
        ("round_index", round_index),
        ("episode_position", episode_position),
    ):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a nonnegative exact integer")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)
    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "state_schema_version": FORMAL_STATE_SCHEMA_VERSION,
        "state_schema_digest": FORMAL_STATE_SCHEMA_DIGEST,
        "action_catalog_digest": ACTION_CATALOG_DIGEST,
        "action_dim": len(FINAL_DQN_ACTION_CATALOG),
        "semantics": control_semantics(),
        "training_config": asdict(agent.config),
        "agent": agent.state_dict(),
        "schedule": schedule.state_dict(),
        "global_macro_step": global_macro_step,
        "round_index": round_index,
        "episode_position": episode_position,
    }
    try:
        torch.save(payload, temporary)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_checkpoint(
    path: Path,
    *,
    agent: DqnAgent,
    schedule: EpisodeShuffleSchedule,
) -> ResumeMetadata:
    try:
        payload = torch.load(Path(path), map_location=agent.device, weights_only=False)
    except Exception as exc:
        raise IncompatibleCheckpointError(f"checkpoint cannot be read: {exc}") from exc
    if type(payload) is not dict:
        raise IncompatibleCheckpointError("checkpoint payload must be a dict")
    expected = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "state_schema_version": FORMAL_STATE_SCHEMA_VERSION,
        "state_schema_digest": FORMAL_STATE_SCHEMA_DIGEST,
        "action_catalog_digest": ACTION_CATALOG_DIGEST,
        "action_dim": len(FINAL_DQN_ACTION_CATALOG),
        "semantics": control_semantics(),
        "training_config": asdict(agent.config),
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise IncompatibleCheckpointError("checkpoint state/action/config identity differs")
    try:
        agent.load_state_dict(payload["agent"])
        schedule.load_state_dict(payload["schedule"])
        values = tuple(payload[name] for name in (
            "global_macro_step", "round_index", "episode_position"
        ))
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("checkpoint counters are invalid")
    except Exception as exc:
        raise IncompatibleCheckpointError(f"checkpoint runtime state differs: {exc}") from exc
    return ResumeMetadata(*values)


__all__ = [
    "CHECKPOINT_VERSION", "IncompatibleCheckpointError", "ResumeMetadata",
    "load_checkpoint", "save_checkpoint",
]
