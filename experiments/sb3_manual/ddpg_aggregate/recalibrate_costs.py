"""Reprice recorded Train35 baseline physics; never execute a policy or learn."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from v2.models.battery_degradation import (
    BATTERY_LIFETIME_THROUGHPUT_FACTOR, BATTERY_NOMINAL_CHARGE_CAPACITY_AH,
    BATTERY_REPLACEMENT_COST_CNY,
)
from v2.models.battery_energy import (
    BATTERY_CURRENT_ACCOUNTING_VERSION, battery_current_from_soc_change,
)
from v2.models.fuel_cell_degradation import (
    FC_AGGREGATE_REPLACEMENT_COST_CNY, FC_DEGRADATION_MODEL_VERSION,
    FC_EOL_VOLTAGE_LOSS_UV,
)
from v3.control import AccountState, EconomicMPC

from .economic_reward import (
    FAILURE_REWARD, REWARD_VERSION, interval_economic_components, load_reward_config,
)

DEFAULT_COLLECTION = PROJECT_ROOT / (
    "outputs/ddpg_reward_calibration/collection_20261103_133754_684927/"
    "calibration_collection.json"
)
DEFAULT_DATA = PROJECT_ROOT / "data/processed/operating_dataset_v6_1s"
DEFAULT_CONFIG = Path(__file__).with_name("economic_reward_config.json")
SOURCE_FIELDS = (
    "soc_before", "soc_after", "fc_kw", "battery_kw", "load_kw",
    "h2_cost_cny", "fcdeg_cny", "lbdeg_cny",
)
RESULT_FIELDS = (
    "soc_before", "soc_after", "fc_kw", "battery_kw", "battery_current_a",
    "h2_cost_cny", "fcdeg_cny", "lbdeg_cny", "c_bat_eq_cny", "total_cost_cny",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes() -> dict:
    paths = (
        Path(__file__), Path(__file__).with_name("economic_reward.py"),
        PROJECT_ROOT / "src/v3/control.py", PROJECT_ROOT / "src/v2/economics.py",
        PROJECT_ROOT / "src/v2/models/battery_energy.py",
        PROJECT_ROOT / "src/v2/models/battery_degradation.py",
        PROJECT_ROOT / "src/v2/models/fuel_cell_degradation.py",
        PROJECT_ROOT / "src/v2/models/fuel_cell_efficiency.py",
    )
    return {path.relative_to(PROJECT_ROOT).as_posix(): sha256(path) for path in paths}


def verify_train_source(collection_path: Path, data_root: Path) -> tuple[dict, dict]:
    """Check frozen source identities and per-voyage arrays against the master NPZ."""
    report = json.loads(collection_path.read_text(encoding="utf-8"))
    metadata_path = data_root / "metadata/active_split.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    manifest_path = data_root / metadata["manifest_relative_path"]
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        train_ids = sorted(row["candidate_id"] for row in csv.DictReader(handle) if row["split"] == "train")
    if report["split"] != "train" or len(train_ids) != 35 or sorted(report["train_ids"]) != train_ids:
        raise ValueError("Recalibration requires exactly the frozen Train35 baseline voyages")
    baseline_path = Path(report["model_path"])
    identities = {
        "baseline_model_path": str(baseline_path),
        "baseline_model_sha256": sha256(baseline_path),
        "collection_report_sha256": sha256(collection_path),
        "raw_npz_sha256": sha256(collection_path.parent / "train35_raw_costs.npz"),
        "data_active_split_sha256": sha256(metadata_path),
        "data_manifest_sha256": sha256(manifest_path),
    }
    for actual, recorded in (
        (identities["baseline_model_sha256"], report["model_sha256"]),
        (identities["data_active_split_sha256"], report["data_active_split_sha256"]),
        (identities["data_manifest_sha256"], report["data_manifest_sha256"]),
    ):
        if actual != recorded:
            raise ValueError("Baseline model or frozen Train35 metadata differs from the recorded collection")
    lengths = [int(row["executed_steps"]) for row in report["per_voyage"]]
    offsets = np.r_[0, np.cumsum(lengths)]
    if int(offsets[-1]) != report["executed_steps"]:
        raise ValueError("Recorded physical step counts disagree")
    per_voyage_paths = [collection_path.parent / f"raw_{sample_id}.npz" for sample_id in report["train_ids"]]
    with np.load(collection_path.parent / "train35_raw_costs.npz", allow_pickle=False) as master:
        indices = master["sample_index"]
        for index, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
            if not np.all(indices[start:end] == index):
                raise ValueError("Master baseline NPZ voyage boundaries disagree")
        # Load one master column at a time to keep calibration memory bounded.
        for name in SOURCE_FIELDS:
            column = master[name]
            if not np.isfinite(column).all():
                raise ValueError("Recorded baseline has nonfinite physical/cost fields")
            for path, start, end in zip(per_voyage_paths, offsets[:-1], offsets[1:]):
                with np.load(path, allow_pickle=False) as voyage:
                    if not np.array_equal(column[start:end], voyage[name]):
                        raise ValueError("Per-voyage baseline differs from the recorded master NPZ")
    identities["per_voyage_raw_sha256"] = {path.name: sha256(path) for path in per_voyage_paths}
    return report, identities


def replay_voyage(physics, source: dict) -> tuple[np.ndarray, dict]:
    """Re-execute recorded physical setpoints with the current accounting only."""
    count = len(source["fc_kw"])
    result = np.empty((count, len(RESULT_FIELDS)), dtype=np.float64)
    state = AccountState()
    max_soc_error = max_battery_error = max_h2_error = 0.0
    for index in range(count):
        if abs(state.soc - float(source["soc_before"][index])) > 1e-12:
            raise ValueError("Recorded baseline SOC does not match unchanged physical dynamics")
        before = state
        fc_kw = float(source["fc_kw"][index])
        state, ledger, battery_kw = physics.interval(
            before, fc_kw, float(source["load_kw"][index]), duration_s=1.0
        )
        soc_error = abs(state.soc - float(source["soc_after"][index]))
        battery_error = abs(battery_kw - float(source["battery_kw"][index]))
        h2_error = abs(ledger.h2_cost_cny - float(source["h2_cost_cny"][index]))
        if soc_error > 1e-12 or battery_error > 1e-9 or h2_error > 1e-12:
            raise ValueError("Repricing changed recorded physics or hydrogen accounting")
        max_soc_error = max(max_soc_error, soc_error)
        max_battery_error = max(max_battery_error, battery_error)
        max_h2_error = max(max_h2_error, h2_error)
        costs = interval_economic_components(physics, before, state, fc_kw, ledger)
        current = battery_current_from_soc_change(
            before.soc, state.soc, 1.0, BATTERY_NOMINAL_CHARGE_CAPACITY_AH
        )
        result[index] = (
            before.soc, state.soc, fc_kw, battery_kw, current,
            costs["h2_cost_cny"], costs["fuel_cell_degradation_cost_cny"],
            costs["battery_degradation_cost_cny"], costs["c_bat_eq_cny"], costs["total_cost_cny"],
        )
    return result, {
        "max_soc_replay_error": max_soc_error,
        "max_battery_power_replay_error_kw": max_battery_error,
        "max_h2_cost_replay_error_cny": max_h2_error,
        "fc_cumulative_voltage_loss_uv": state.fc_loss_uv,
        "battery_cumulative_weighted_ah": state.battery_weighted_ah,
    }


def recalculate(collection_path: Path, data_root: Path, output_root: Path) -> tuple[dict, dict]:
    started = time.perf_counter()
    report, identities = verify_train_source(collection_path, data_root)
    sources = source_hashes()
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    output = output_root / f"recalibration_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    print(f"Recalibration output: {output}", flush=True)
    # nominal_cost_cny belongs only to unused MPC optimization; no solve/learn is called.
    physics = EconomicMPC(nominal_cost_cny=1.0)
    costs, per_voyage, result_hashes = [], [], {}
    for index, original in enumerate(report["per_voyage"]):
        sample_id = original["sample_id"]
        if sample_id != report["train_ids"][index]:
            raise ValueError("Recorded baseline voyage order is inconsistent")
        source_path = collection_path.parent / f"raw_{sample_id}.npz"
        with np.load(source_path, allow_pickle=False) as raw:
            source = {name: raw[name] for name in SOURCE_FIELDS}
        matrix, integrity = replay_voyage(physics, source)
        step_costs = matrix[:, -1].copy()
        costs.append(step_costs)
        result_path = output / f"recomputed_{sample_id}.npz"
        np.savez_compressed(result_path, **{name: matrix[:, column] for column, name in enumerate(RESULT_FIELDS)})
        result_hashes[result_path.name] = sha256(result_path)
        row = {name: original[name] for name in ("sample_id", "completed", "executed_steps", "total_steps", "coverage")}
        row.update(integrity)
        row["economic_cost_sum_cny"] = float(np.sum(step_costs))
        row["economic_cost_median_cny"] = float(np.median(step_costs))
        per_voyage.append(row)
        print(f"{index + 1}/35 {sample_id}: {len(step_costs)} executed seconds; physics unchanged", flush=True)
    combined = np.concatenate(costs)
    negative = int(np.sum(combined < 0.0))
    zero = int(np.sum(combined == 0.0))
    nonfinite = int(np.sum(~np.isfinite(combined)))
    if negative or nonfinite:
        raise ValueError(f"Recalibration cannot proceed: negative={negative}, nonfinite={nonfinite}")
    reference = float(np.median(combined))
    if not np.isfinite(reference) or reference <= 0.0:
        raise ValueError("Train35 median C_ref must be finite and strictly positive")
    aggregate_path = output / "train35_economic_costs.npz"
    np.savez_compressed(aggregate_path, total_cost_cny=combined)
    if source_hashes() != sources or sha256(Path(identities["baseline_model_path"])) != identities["baseline_model_sha256"]:
        raise RuntimeError("Accounting source or frozen baseline changed during recalibration")
    result = {
        "reward_version": REWARD_VERSION, "c_ref": reference, "split": "train",
        "collection_report": collection_path.relative_to(PROJECT_ROOT).as_posix(),
        **identities, "source_sha256": sources,
        "train_ids": report["train_ids"], "selected_voyages": 35,
        "completed_voyages": sum(row["completed"] for row in per_voyage),
        "executed_physical_steps": len(combined), "planned_physical_steps": report["planned_steps"],
        "sampling": "median of every recorded actually executed physical second across Train35 exactly once; failure calls and unexecuted suffixes excluded",
        "method": "replay recorded FC/load setpoints through current EconomicMPC.interval and four-component accounting; no policy inference or training",
        "negative_economic_steps": negative, "zero_economic_steps": zero,
        "nonfinite_economic_steps": nonfinite,
        "economic_cost_quantiles_cny": {str(q): float(np.quantile(combined, q)) for q in (0, .25, .5, .75, 1)},
        "fc_degradation_model_version": FC_DEGRADATION_MODEL_VERSION,
        "fc_eol_voltage_loss_uv": FC_EOL_VOLTAGE_LOSS_UV,
        "fc_aggregate_replacement_cost_cny": FC_AGGREGATE_REPLACEMENT_COST_CNY,
        "battery_current_accounting_version": BATTERY_CURRENT_ACCOUNTING_VERSION,
        "battery_current_rule": "(SOC_before-SOC_after)*nominal_capacity_Ah*3600/dt; actual modeled cell current; charge/discharge efficiency already included once in SOC dynamics",
        "battery_lifetime_throughput_factor": BATTERY_LIFETIME_THROUGHPUT_FACTOR,
        "battery_replacement_cost_cny": BATTERY_REPLACEMENT_COST_CNY,
        "per_voyage": per_voyage, "recomputed_raw_sha256": result_hashes,
        "recomputed_economic_costs_sha256": sha256(aggregate_path),
        "validation_payloads_used": 0, "test_payloads_used": 0,
        "elapsed_s": time.perf_counter() - started,
    }
    result_path = output / "recalibration_result.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    reference_config = load_reward_config()
    config = {
        "reward_version": REWARD_VERSION, "c_ref": reference, "failure_reward": FAILURE_REWARD,
        "eta_fc_avg": reference_config["eta_fc_avg"],
        "eta_fc_avg_calibration": reference_config["eta_fc_avg_calibration"],
        "calibration": {
            **result, "recalibration_report": result_path.relative_to(PROJECT_ROOT).as_posix(),
            "recalibration_report_sha256": sha256(result_path),
        },
    }
    return config, result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-report", type=Path, default=DEFAULT_COLLECTION)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/ddpg_reward_recalibration_v3")
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--write-config", action="store_true")
    args = parser.parse_args()
    config, result = recalculate(args.collection_report.resolve(), args.data_root.resolve(), args.output_root.resolve())
    if args.write_config:
        args.config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "c_ref", "executed_physical_steps", "negative_economic_steps", "zero_economic_steps",
        "nonfinite_economic_steps", "elapsed_s",
    )}, indent=2), flush=True)


if __name__ == "__main__":
    main()
