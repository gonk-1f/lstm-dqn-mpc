"""Generate the authenticated Train-only 30 s interval reward scale."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Sequence

from ..analysis.action_screening import DataSplit, DatasetProvenance
from ..analysis.train_state_audit import ACTIVE_DATASET_VERSION
from ..config import TimeScaleConfig
from ..data.formal_training_dataset import FormalTrainingDataset
from ..economics import calibrate_reward_scale
from ..evaluation.checkpoint_selection import canonical_result_digest
from ..training.reward_scaling import (
    REFERENCE_ACTION_ID,
    REWARD_SCALE_DOCUMENT_VERSION,
    build_reward_scale_document,
)
from .train_formal_dqn import (
    DEFAULT_AIS_ROOT,
    DEFAULT_MODE_ROOT,
    DEFAULT_POWER_ROOT,
    REPOSITORY_ROOT,
    _environment,
)


DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "outputs"
    / "v2_history_dqn_study"
    / "reward_scale_calibration.json"
)
CALIBRATION_REASON = (
    "arithmetic mean of all raw 30 s Train interval ledger costs under fixed "
    f"{REFERENCE_ACTION_ID}"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_hashes(
    power_root: Path,
    ais_root: Path,
    mode_root: Path,
) -> dict[str, str]:
    return {
        "power": _sha256(Path(power_root) / "metadata" / "sample_manifest.csv"),
        "ais": _sha256(Path(ais_root) / "metadata" / "sample_manifest.csv"),
        "modes": _sha256(Path(mode_root) / "metadata" / "sample_manifest.csv"),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train-only v2 reward-scale calibration")
    parser.add_argument("--power-root", type=Path, default=DEFAULT_POWER_ROOT)
    parser.add_argument("--ais-root", type=Path, default=DEFAULT_AIS_ROOT)
    parser.add_argument("--mode-root", type=Path, default=DEFAULT_MODE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def generate_calibration(
    args: argparse.Namespace,
    *,
    timescale: TimeScaleConfig | None = None,
) -> dict[str, object]:
    dataset = FormalTrainingDataset.open(args.power_root, args.ais_root, args.mode_root)
    train = dataset.load_train()
    costs: list[float] = []
    for position, episode in enumerate(train, start=1):
        if timescale is None:
            backend, environment = _environment(episode)
        else:
            backend, environment = _environment(episode, timescale=timescale)
        environment.reset()
        while True:
            transition = environment.step(REFERENCE_ACTION_ID)
            if transition.done:
                break
        episode_costs = tuple(
            float(ledger.total_cost_cny) for ledger in backend.interval_ledgers
        )
        costs.extend(episode_costs)
        print(
            f"reward_scale_episode={position}/{len(train)} "
            f"sample_id={episode.sample_id} intervals={len(episode_costs)}",
            flush=True,
        )
    if dataset.opened_test_payloads != 0:
        raise PermissionError("reward-scale calibration opened Test payloads")
    hashes = _manifest_hashes(args.power_root, args.ais_root, args.mode_root)
    provenance_id = "sha256:" + canonical_result_digest(
        {"input_manifest_sha256": hashes}
    )
    calibration = calibrate_reward_scale(
        tuple(costs),
        provenance=DatasetProvenance(
            ACTIVE_DATASET_VERSION,
            provenance_id,
            DataSplit.TRAIN,
        ),
        audit_id=REWARD_SCALE_DOCUMENT_VERSION,
        reason=CALIBRATION_REASON,
    )
    return build_reward_scale_document(
        calibration=calibration,
        reference_action_id=REFERENCE_ACTION_ID,
        manifest_hashes=hashes,
        train_segment_ids=tuple(episode.sample_id for episode in train),
        test_payloads_opened=dataset.opened_test_payloads,
        timescale=timescale,
    )


def _write_atomic(path: Path, payload: dict[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        temporary.write_text(
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    document = generate_calibration(args)
    _write_atomic(args.output, document)
    print(
        f"REWARD_SCALE_CALIBRATION=PASS output={args.output} "
        f"sample_count={document['sample_count']} "
        f"scale_cny={document['scale_cny']:.12g} "
        f"test_payloads_opened={document['test_payloads_opened']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
