"""Bounded Train-only closed-loop screen for calibrated v3 MPC actions."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Sequence

from v2.data.formal_training_dataset import FormalTrainingDataset
from v2.main.train_formal_dqn import (
    DEFAULT_AIS_ROOT, DEFAULT_MODE_ROOT, DEFAULT_POWER_ROOT, REPOSITORY_ROOT,
)

from .control import AccountState, DT_SECONDS, EconomicMPC, MPCWeights, SOC_MAX, SOC_MIN
from .episode_replay import EpisodeReplay, ReplayExecutionError, replay_episode


DEFAULT_CATALOG = REPOSITORY_ROOT / "outputs" / "v3_persistence_calibration" / "action_calibration_response16.json"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "outputs" / "v3_persistence_calibration" / "closed_loop_train_pilot.json"
DEFAULT_CASES = (("zero_boundary_017", 621), ("zero_boundary_046", 344))
SHORE_VALUES = frozenset(("shore_pending", "shore_charging"))


@dataclass(frozen=True)
class CalibratedCatalog:
    nominal_cost_cny: float
    actions: tuple[MPCWeights, ...]


@dataclass(frozen=True)
class SelectedVoyage:
    source_sample_id: str
    start_index: int
    end_index_exclusive: int
    episode: object

    @property
    def case_id(self) -> str:
        return f"{self.source_sample_id}:{self.start_index}"


def catalog_from_document(
    payload: dict[str, object], *, expected_manifest_sha256: dict[str, str],
) -> CalibratedCatalog:
    if payload.get("split_used") != "train" or payload.get("test_payloads_opened") != 0:
        raise PermissionError("catalog must come from Train without Test payload access")
    if payload.get("manifest_sha256") != expected_manifest_sha256:
        raise PermissionError("catalog input manifest hashes differ from the current dataset")
    if payload.get("schema_version") != "v3_persistence_action_calibration_v1":
        raise ValueError("unsupported calibration artifact version")
    nominal = payload.get("nominal_cost_cny")
    if isinstance(nominal, bool) or not isinstance(nominal, (int, float)) or not math.isfinite(nominal) or nominal <= 0:
        raise ValueError("catalog needs a finite positive nominal cost")
    rows = payload.get("actions")
    if not isinstance(rows, list) or len(rows) != 16:
        raise ValueError("catalog needs 16 unique actions with consecutive IDs")
    try:
        ids = tuple(row["action_id"] for row in rows)
        actions = tuple(MPCWeights(float(row["lambda_ref"]), float(row["lambda_soc"])) for row in rows)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("catalog action rows are malformed") from exc
    if ids != tuple(range(16)) or len(set(actions)) != 16:
        raise ValueError("catalog needs 16 unique actions with consecutive IDs")
    return CalibratedCatalog(float(nominal), actions)


def select_train_voyage(
    train_episodes: Sequence[object], sample_id: str, start_index: int,
) -> SelectedVoyage:
    if type(start_index) is not int or start_index < 0:
        raise ValueError("voyage start index must be a nonnegative integer")
    matches = [episode for episode in train_episodes if episode.sample_id == sample_id]
    if len(matches) != 1:
        raise ValueError("Train sample ID must identify one episode")
    source = matches[0]
    if source.split != "train":
        raise PermissionError("voyage selection requires a Train episode")
    modes = tuple(source.operating_mode)
    if start_index >= len(modes) or modes[start_index] != "onboard" or (
        start_index > 0 and modes[start_index - 1] == "onboard"
    ):
        raise ValueError("voyage start must be the first ONBOARD row of a run")
    end = start_index
    while end < len(modes) and modes[end] == "onboard":
        end += 1
    while end < len(modes) and modes[end] in SHORE_VALUES:
        end += 1
    selected = SimpleNamespace(
        sample_id=f"{sample_id}:{start_index}", split="train",
        operating_mode=modes[start_index:end],
        load_kw=tuple(float(value) for value in source.load_kw[start_index:end]),
        battery_bus_kw=tuple(float(value) for value in source.battery_bus_kw[start_index:end]),
    )
    return SelectedVoyage(sample_id, start_index, end, selected)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_hashes(dataset: FormalTrainingDataset) -> dict[str, str]:
    return {
        "power": _sha256(dataset._power_root / "metadata" / "sample_manifest.csv"),
        "ais": _sha256(dataset._ais_root / "metadata" / "sample_manifest.csv"),
        "modes": _sha256(dataset._mode_root / "metadata" / "sample_manifest.csv"),
    }


def _success_metrics(result: EpisodeReplay) -> dict[str, object]:
    costs = result.total_ledger
    soc = (result.start_soc, *result.soc_by_row)
    fc = result.fc_power_kw_by_row
    fc_changes = [abs(right - left) for left, right in zip((0.0, *fc[:-1]), fc)]
    return {
        "status": "complete",
        "onboard_steps": result.onboard_steps,
        "shore_steps": result.shore_steps,
        "transition_count": len(result.transitions),
        "terminal_transition_count": sum(item.done for item in result.transitions),
        "cost_cny": {
            "hydrogen": costs.h2_cost_cny,
            "fuel_cell_degradation": costs.fuel_cell_degradation_cost_cny,
            "battery_degradation": costs.battery_degradation_cost_cny,
            "shore_electricity": costs.shore_cost_cny,
            "total": costs.total_cost_cny,
        },
        "soc_start": result.start_soc,
        "soc_end": result.final_state.soc,
        "soc_min": min(soc),
        "soc_max": max(soc),
        "fc_power_max_kw": max(fc),
        "fc_total_absolute_change_kw": math.fsum(fc_changes),
        "shore_requested_battery_kwh": math.fsum(
            -power * DT_SECONDS / 3600.0
            for block in result.shore_blocks for power in block.requested_battery_bus_kw
        ),
        "shore_accepted_battery_kwh": math.fsum(
            -power * DT_SECONDS / 3600.0
            for block in result.shore_blocks for power in block.accepted_battery_bus_kw
        ),
        "fc_power_kw_by_row": list(fc),
        "battery_bus_kw_by_row": list(result.battery_bus_kw_by_row),
        "soc_by_row": list(result.soc_by_row),
    }


def generate_pilot(
    dataset: FormalTrainingDataset, catalog: CalibratedCatalog, *,
    cases: Sequence[tuple[str, int]] = DEFAULT_CASES,
    action_ids: Sequence[int] = tuple(range(16)),
    initial_soc: float = 0.6,
) -> dict[str, object]:
    if dataset.opened_test_payloads != 0:
        raise PermissionError("Test payloads were opened before Train pilot")
    if not SOC_MIN <= initial_soc <= SOC_MAX:
        raise ValueError("pilot initial SOC is outside physical bounds")
    chosen_ids = tuple(action_ids)
    if not chosen_ids or len(chosen_ids) != len(set(chosen_ids)) or any(
        type(index) is not int or index < 0 or index >= len(catalog.actions) for index in chosen_ids
    ):
        raise ValueError("action IDs must be distinct catalog indices")
    train = dataset.load_train()
    voyages = tuple(select_train_voyage(train, sample_id, start) for sample_id, start in cases)
    if not voyages or len({voyage.case_id for voyage in voyages}) != len(voyages):
        raise ValueError("pilot cases must identify distinct Train voyages")
    mpc = EconomicMPC(nominal_cost_cny=catalog.nominal_cost_cny)
    records: list[dict[str, object]] = []
    for case_number, voyage in enumerate(voyages, start=1):
        onboard_count = sum(mode == "onboard" for mode in voyage.episode.operating_mode)
        shore_count = len(voyage.episode.operating_mode) - onboard_count
        action_records = []
        for action_id in chosen_ids:
            action = catalog.actions[action_id]
            started = perf_counter()
            try:
                result = replay_episode(
                    voyage.episode, lambda _state, fixed=action: fixed,
                    mpc=mpc, initial_state=AccountState(soc=float(initial_soc)),
                )
            except ReplayExecutionError as exc:
                metrics = {
                    "status": "physical_failure", "row_index": exc.row_index,
                    "mode": exc.mode.value, "reason": str(exc.cause),
                }
            else:
                metrics = _success_metrics(result)
            elapsed = perf_counter() - started
            action_records.append({
                "action_id": action_id,
                "lambda_ref": action.lambda_ref,
                "lambda_soc": action.lambda_soc,
                "elapsed_seconds": elapsed,
                **metrics,
            })
            print(
                f"pilot_case={case_number}/{len(voyages)} id={voyage.case_id} "
                f"action={action_id} status={metrics['status']} elapsed_s={elapsed:.2f}",
                flush=True,
            )
        records.append({
            "case_id": voyage.case_id,
            "source_sample_id": voyage.source_sample_id,
            "source_start_index": voyage.start_index,
            "source_end_index_exclusive": voyage.end_index_exclusive,
            "onboard_steps": onboard_count,
            "shore_steps": shore_count,
            "actions": action_records,
        })
    if dataset.opened_test_payloads != 0:
        raise PermissionError("Train pilot opened Test payloads")
    return {
        "schema_version": "v3_persistence_closed_loop_train_pilot_v1",
        "split_used": "train",
        "test_payloads_opened": dataset.opened_test_payloads,
        "manifest_sha256": _manifest_hashes(dataset),
        "nominal_cost_cny": catalog.nominal_cost_cny,
        "initial_soc": float(initial_soc),
        "mode_route": {
            "onboard": "policy_then_mpc_then_actual_load",
            "shore_pending": "shore_settlement_without_policy_or_mpc",
            "shore_charging": "shore_settlement_without_policy_or_mpc",
            "unresolved": "fail_closed",
        },
        "case_assumption": "each complete selected ONBOARD run starts at the stated initial SOC, independent of earlier rows in its source episode",
        "action_ids": list(chosen_ids),
        "cases": records,
    }


def _parse_case(value: str) -> tuple[str, int]:
    sample_id, separator, index = value.rpartition(":")
    if not separator or not sample_id:
        raise argparse.ArgumentTypeError("case must use sample_id:start_index")
    try:
        return sample_id, int(index)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("case start_index must be an integer") from exc


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Mode-gated v3 closed-loop Train action pilot")
    parser.add_argument("--power-root", type=Path, default=DEFAULT_POWER_ROOT)
    parser.add_argument("--ais-root", type=Path, default=DEFAULT_AIS_ROOT)
    parser.add_argument("--mode-root", type=Path, default=DEFAULT_MODE_ROOT)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--case", type=_parse_case, action="append")
    parser.add_argument("--action-id", type=int, action="append")
    parser.add_argument("--initial-soc", type=float, default=0.6)
    args = parser.parse_args(argv)
    dataset = FormalTrainingDataset.open(args.power_root, args.ais_root, args.mode_root)
    payload = json.loads(args.catalog.read_text(encoding="utf-8"))
    catalog = catalog_from_document(payload, expected_manifest_sha256=_manifest_hashes(dataset))
    report = generate_pilot(
        dataset, catalog,
        cases=DEFAULT_CASES if args.case is None else args.case,
        action_ids=tuple(range(16)) if args.action_id is None else args.action_id,
        initial_soc=args.initial_soc,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f"{args.output.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({
        "output": str(args.output), "case_count": len(report["cases"]),
        "action_count": len(report["action_ids"]),
        "test_payloads_opened": report["test_payloads_opened"],
    }, indent=2))


if __name__ == "__main__":
    main()
