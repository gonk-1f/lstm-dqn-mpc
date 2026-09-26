"""Generate the authenticated Train-only terminal-failure penalty audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Sequence

from ..data.formal_training_dataset import FormalTrainingDataset
from ..dqn.action_space import ACTION_CATALOG_DIGEST
from ..failure_policy import FORMAL_FAILURE_POLICY
from .train_formal_dqn import (
    DEFAULT_AIS_ROOT,
    DEFAULT_MODE_ROOT,
    DEFAULT_POWER_ROOT,
    REPOSITORY_ROOT,
    _environment,
)


AUDIT_SCHEMA_VERSION = "v2_terminal_failure_audit_v1"
REFERENCE_ACTION_ID = "w_8_1_1"
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT / "outputs" / "v2_failure_penalty_audit" / "audit_summary.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def _result_digest(payload_without_digest: dict[str, object]) -> str:
    return hashlib.sha256(_canonical_bytes(payload_without_digest)).hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train-only v2 failure-penalty audit")
    parser.add_argument("--power-root", type=Path, default=DEFAULT_POWER_ROOT)
    parser.add_argument("--ais-root", type=Path, default=DEFAULT_AIS_ROOT)
    parser.add_argument("--mode-root", type=Path, default=DEFAULT_MODE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def _manifest_hashes(power_root: Path, ais_root: Path, mode_root: Path) -> dict[str, str]:
    return {
        "power": _sha256(Path(power_root) / "metadata" / "sample_manifest.csv"),
        "ais": _sha256(Path(ais_root) / "metadata" / "sample_manifest.csv"),
        "modes": _sha256(Path(mode_root) / "metadata" / "sample_manifest.csv"),
    }


def generate_audit(args: argparse.Namespace) -> dict[str, object]:
    dataset = FormalTrainingDataset.open(args.power_root, args.ais_root, args.mode_root)
    train = dataset.load_train()
    episode_results: list[dict[str, object]] = []
    for position, episode in enumerate(train, start=1):
        _, environment = _environment(episode)
        environment.reset()
        raw_cost = 0.0
        penalty = 0.0
        learning_reward = 0.0
        transitions = 0
        while True:
            transition = environment.step(REFERENCE_ACTION_ID)
            raw_cost += transition.raw_economic_cost_cny
            penalty += transition.failure_penalty_score
            learning_reward += transition.learning_reward
            transitions += 1
            if transition.done:
                break
        episode_results.append(
            {
                "sample_id": episode.sample_id,
                "raw_economic_cost_cny": raw_cost,
                "failure_penalty_score": penalty,
                "learning_reward": learning_reward,
                "transition_count": transitions,
                "episode_completed": transition.episode_completed,
                "failure_kind": transition.failure_kind,
            }
        )
        print(
            f"audit_episode={position}/{len(train)} sample_id={episode.sample_id} "
            f"episode_completed={'YES' if transition.episode_completed else 'NO'} "
            f"raw_economic_cost_cny={raw_cost:.9f}",
            flush=True,
        )

    completed = [row for row in episode_results if row["episode_completed"] is True]
    failed = [row for row in episode_results if row["episode_completed"] is False]
    maximum = max(completed, key=lambda row: float(row["raw_economic_cost_cny"]))
    payload: dict[str, object] = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "dataset_version": "operating_dataset_zero_boundary_v2",
        "split": "train",
        "input_manifest_sha256": _manifest_hashes(
            args.power_root, args.ais_root, args.mode_root
        ),
        "train_segment_ids": [episode.sample_id for episode in train],
        "train_segment_count": len(train),
        "reference_action_id": REFERENCE_ACTION_ID,
        "action_catalog_digest": ACTION_CATALOG_DIGEST,
        "failure_penalty_score": FORMAL_FAILURE_POLICY.penalty_score,
        "failure_kind": FORMAL_FAILURE_POLICY.failure_kind,
        "evidence_status": FORMAL_FAILURE_POLICY.evidence_status,
        "calibration_id": FORMAL_FAILURE_POLICY.calibration_id,
        "completed_episode_count": len(completed),
        "failed_episode_count": len(failed),
        "failed_episode_ids": [str(row["sample_id"]) for row in failed],
        "maximum_completed_raw_economic_cost_cny": float(
            maximum["raw_economic_cost_cny"]
        ),
        "maximum_completed_cost_episode_id": str(maximum["sample_id"]),
        "test_payloads_opened": dataset.opened_test_payloads,
        "formal_training_started": False,
        "episode_results": episode_results,
    }
    payload["result_digest"] = _result_digest(payload)
    return payload


def _write_atomic(path: Path, payload: dict[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = generate_audit(args)
    _write_atomic(args.output, payload)
    print(
        f"FAILURE_PENALTY_AUDIT=PASS output={args.output} "
        f"completed={payload['completed_episode_count']} "
        f"failed={payload['failed_episode_count']} "
        f"result_digest={payload['result_digest']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
