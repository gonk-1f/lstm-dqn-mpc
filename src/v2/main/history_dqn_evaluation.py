"""Load one frozen H1-H4 identity for Validation or Test evaluation."""

from __future__ import annotations

import json
from pathlib import Path

from ..training.dqn import DqnTrainingConfig
from ..training.experiments import DqnExperimentProfile, history_study_profile
from ..training.reward_scaling import load_reward_scale_document


def load_evaluation_profile(
    *,
    experiment_id: str,
    reward_scale_path: Path | None,
    expected_manifest_hashes: dict[str, str],
    rounds: int,
) -> tuple[DqnExperimentProfile, DqnTrainingConfig, dict[str, object]]:
    """Reconstruct and bind the exact training identity used by a checkpoint bank."""

    if experiment_id == "H1":
        if reward_scale_path is not None:
            raise ValueError("H1 raw evaluation forbids --reward-scale")
        profile = history_study_profile("H1", None)
    else:
        if experiment_id not in {"H2", "H3", "H4"}:
            raise ValueError("experiment_id must be one of H1, H2, H3, H4")
        if reward_scale_path is None:
            raise FileNotFoundError("scaled evaluation requires --reward-scale")
        scale_path = Path(reward_scale_path)
        if not scale_path.is_file():
            raise FileNotFoundError(scale_path)
        calibration = load_reward_scale_document(
            json.loads(scale_path.read_text(encoding="utf-8")),
            expected_manifest_hashes=expected_manifest_hashes,
        )
        profile = history_study_profile(experiment_id, calibration)
    config = profile.dqn_config(rounds=rounds)
    identity = {
        "experiment_id": profile.experiment_id,
        "reward_mode": profile.reward_mode,
        "reward_scaling_identity": profile.reward_scaling_identity,
    }
    return profile, config, identity


__all__ = ["load_evaluation_profile"]
