"""Collect frozen-baseline Train35 transitions without selecting a cost reference."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

from .validate import DEFAULT_DATA_ROOT, PROJECT_ROOT, _model_snapshot
from . import env as environment
import numpy as np
from stable_baselines3 import DDPG
import torch

BASELINE_SHA256 = "caf67c28eea863c01c8d6f9a8f7e38ea27b2f3ce924fe1d12dc4a843551f100f"
DEFAULT_MODEL = PROJECT_ROOT / "outputs/ddpg_train_500ep/ddpg/AggregateShipDDPG-v0_1/AggregateShipDDPG-v0.zip"
FIELDS = ("soc_before", "soc_after", "fc_kw", "battery_kw", "load_kw",
          "h2_cost_cny", "fcdeg_cny", "lbdeg_cny")
_MODEL = _ENV = _EPISODES = _BEFORE = _LEDGER = _SOURCES = None


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source_hashes():
    return {str(path.relative_to(PROJECT_ROOT)): _sha256(path) for path in (
        Path(__file__), Path(environment.__file__),
        PROJECT_ROOT / "experiments/sb3_manual/ddpg_aggregate/validate.py",
        PROJECT_ROOT / "src/v3/control.py",
        PROJECT_ROOT / "src/v2/models/fuel_cell_efficiency.py",
        PROJECT_ROOT / "src/v2/models/battery_energy.py",
        PROJECT_ROOT / "src/v2/models/battery_degradation.py",
        PROJECT_ROOT / "src/v2/models/fuel_cell_degradation.py",
        PROJECT_ROOT / "src/v2/economics.py",
    )}


def _initialize(model_path, data_root):
    global _MODEL, _ENV, _EPISODES, _BEFORE, _SOURCES
    torch.set_num_threads(1)
    if _sha256(model_path) != BASELINE_SHA256:
        raise ValueError("This collector requires the identified formal baseline checkpoint")
    _MODEL = DDPG.load(model_path, device="cpu", action_noise=None)
    _MODEL.policy.set_training_mode(False)
    _BEFORE = _model_snapshot(_MODEL)
    _ENV = environment.AggregateDDPGEnv(data_root, split="train")
    _EPISODES = tuple(_ENV.episodes)
    if len(_EPISODES) != 35 or any(ep.split != "train" for ep in _EPISODES):
        raise RuntimeError("Collection requires exactly the frozen Train35 voyages")
    _SOURCES = _source_hashes()
    interval = _ENV.physics.interval

    def capture_interval(*args, **kwargs):
        global _LEDGER
        result = interval(*args, **kwargs)
        _LEDGER = result[1]
        return result

    _ENV.physics.interval = capture_interval
    print(f"Worker {os.getpid()} ready: Train35 only; model/Replay frozen", flush=True)


def _collect(index, one_step=False):
    global _LEDGER
    episode = _EPISODES[index]
    _ENV.episodes = (episode,)
    obs, reset_info = _ENV.reset(seed=42)
    if reset_info != {"sample_id": episode.sample_id, "split": "train"}:
        raise RuntimeError("Fixed Train voyage selection failed")
    values = []
    calls = 0
    with torch.inference_mode():
        while True:
            before_soc, before_index = _ENV.physical.soc, _ENV.index
            _LEDGER = None
            action, _ = _MODEL.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = _ENV.step(action)
            calls += 1
            if _ENV.index > before_index:
                if _LEDGER is None:
                    raise RuntimeError("Executed interval did not return the original ledger")
                values.append((
                    before_soc, _ENV.physical.soc, info["fc_actual_kw"],
                    info["battery_actual_kw"], info["load_kw"], _LEDGER.h2_cost_cny,
                    _LEDGER.fuel_cell_degradation_cost_cny, _LEDGER.battery_degradation_cost_cny,
                ))
            if terminated or truncated:
                summary = dict(info["vessel_episode"])
                break
            if one_step:
                summary = _ENV._episode_summary(False, "collection_one_step_check")
                break
    after = _model_snapshot(_MODEL)
    if after != _BEFORE:
        raise RuntimeError("Deterministic collection modified model/optimizer/counters/Replay")
    summary.update(sample_index=index, environment_calls=calls, worker_pid=os.getpid())
    integrity = {"model_before": _BEFORE, "model_after": after, "unchanged": True}
    return index, summary, np.asarray(values, dtype=np.float64).reshape(-1, len(FIELDS)), integrity, _SOURCES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/ddpg_reward_calibration")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--check-one-step", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    model_path, data_root = args.model_path.resolve(), args.data_root.resolve()
    if _sha256(model_path) != BASELINE_SHA256:
        raise ValueError("The supplied model is not the identified formal DDPG baseline")
    if args.check_one_step:
        _initialize(str(model_path), str(data_root))
        _, summary, values, _, _ = _collect(0, one_step=True)
        assert len(values) == 1 and summary["executed_steps"] == 1
        print(json.dumps({"single_step_check": "passed", "summary": summary}, indent=2))
        _ENV.close()
        return
    started = time.perf_counter()
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    output = args.output_root.resolve() / f"collection_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    sources = _source_hashes()
    metadata = json.loads((data_root / "metadata/active_split.json").read_text(encoding="utf-8"))
    identity = {
        "model_path": str(model_path), "model_sha256": BASELINE_SHA256,
        "data_root": str(data_root), "source_sha256": sources,
        "split": "train", "workers": args.workers, "deterministic": True,
        "action_noise": None, "training": False, "phase": "raw_collection_only",
        "command": [sys.executable, "-m", "experiments.sb3_manual.ddpg_aggregate.calibrate_reward", *sys.argv[1:]],
        "economic_equivalence_status": "raw baseline collection only; no cost reference selected",
        "data_active_split_sha256": _sha256(data_root / "metadata/active_split.json"),
        "data_manifest_sha256": _sha256(data_root / metadata["manifest_relative_path"]),
    }
    (output / "collection_identity.json").write_text(json.dumps(identity, indent=2), encoding="utf-8")
    print(f"Collection output: {output}", flush=True)
    summaries, arrays, worker_states = {}, {}, {}
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=_initialize, initargs=(str(model_path), str(data_root))) as pool:
        futures = [pool.submit(_collect, index) for index in range(35)]
        for future in as_completed(futures):
            index, summary, values, snapshot, worker_sources = future.result()
            if worker_sources != sources:
                raise RuntimeError("Collection source files changed before worker initialization")
            summaries[index], arrays[index] = summary, values
            worker_states[str(summary["worker_pid"])] = snapshot
            np.savez_compressed(output / f"raw_{summary['sample_id']}.npz",
                                sample_index=np.full(len(values), index, dtype=np.int16),
                                **{name: values[:, column] for column, name in enumerate(FIELDS)})
            (output / "collection_progress.json").write_text(json.dumps({
                "completed_voyages": len(summaries),
                "per_voyage": [summaries[key] for key in sorted(summaries)],
            }, indent=2), encoding="utf-8")
            print(f"{len(summaries)}/35 {summary['sample_id']}: {summary['terminal_reason']}, "
                  f"steps={len(values)}/{summary['total_steps']}, coverage={summary['coverage']:.4f}", flush=True)
    ordered = [summaries[index] for index in range(35)]
    if len({row["sample_id"] for row in ordered}) != 35:
        raise RuntimeError("Collection did not cover 35 distinct Train voyages")
    matrix = np.concatenate([arrays[index] for index in range(35)])
    sample_index = np.concatenate([np.full(len(arrays[index]), index, dtype=np.int16) for index in range(35)])
    np.savez_compressed(output / "train35_raw_costs.npz", sample_index=sample_index,
                        **{name: matrix[:, column] for column, name in enumerate(FIELDS)})
    if _sha256(model_path) != BASELINE_SHA256 or _source_hashes() != sources:
        raise RuntimeError("Frozen model or source files changed during collection")
    result = {
        **identity, "train_ids": [row["sample_id"] for row in ordered],
        "planned_steps": sum(row["total_steps"] for row in ordered),
        "executed_steps": len(matrix), "completed_voyages": sum(row["completed"] for row in ordered),
        "per_voyage": ordered, "per_worker_integrity": worker_states,
        "elapsed_s": time.perf_counter() - started,
        "unexecuted_steps": "No costs fabricated for failure calls or unexecuted suffixes.",
        "raw_npz": str(output / "train35_raw_costs.npz"),
    }
    (output / "calibration_collection.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Completed raw Train35 collection: {len(matrix)} physical steps; {output}", flush=True)


if __name__ == "__main__":
    main()
