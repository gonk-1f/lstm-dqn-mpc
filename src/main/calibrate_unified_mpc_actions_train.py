"""Train-only calibration for the unified four-term DQN-MPC objective.

This script never constructs a DQN agent and never loads Validation or Test.
It scans predeclared sum-one MPC actions on deterministic Train windows and
full Train segments, then writes auditable physical metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
MAIN_ROOT = SRC_ROOT / "main"
for path in (SRC_ROOT, MAIN_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dqn.utils.action_mapper import (  # noqa: E402
    LEGACY_FOUR_WEIGHT_ACTIONS as DQN_MPC_WEIGHT_ACTIONS,
    MPCWeightAction,
)
from envs.dqn_mpc_weight_env import DqnMpcWeightEnv, MpcSolveFailure  # noqa: E402
from mpc.solvers.fc_dp0_curve import h2_kg_step_dp0_quadratic  # noqa: E402
from mpc_solvers.formal_config import build_formal_mpc_config  # noqa: E402
from mpc_solvers.dqn_mpc_solver_bank import MpcWeightSolverBank  # noqa: E402
from utils.formal_operating_dataset import (  # noqa: E402
    load_formal_operating_split,
    load_operating_segment_loads,
)


OUTPUT_DIR = REPO_ROOT / "outputs" / "unified_objective_action_redesign_20260911"

# The catalog is intentionally small and declared before observing results.
# Every tuple is (H2, battery, SOC, FC variation), is nonnegative, and sums to 1.
CANDIDATES: tuple[MPCWeightAction, ...] = (
    MPCWeightAction(0, 0.25, 0.25, 0.25, 0.25, "B0_balanced_equal"),
    MPCWeightAction(1, 0.20, 0.30, 0.35, 0.15, "B1_balanced_soc"),
    MPCWeightAction(2, 0.20, 0.25, 0.40, 0.15, "B2_balanced_soc_high"),
    MPCWeightAction(3, 0.25, 0.30, 0.30, 0.15, "B3_balanced_batt"),
    MPCWeightAction(4, 0.50, 0.25, 0.15, 0.10, "E0_h2_economy"),
    MPCWeightAction(5, 0.45, 0.30, 0.15, 0.10, "E1_h2_economy_batt"),
    MPCWeightAction(6, 0.50, 0.20, 0.20, 0.10, "E2_h2_economy_soc"),
    MPCWeightAction(7, 0.40, 0.30, 0.20, 0.10, "E3_h2_economy_moderate"),
    MPCWeightAction(8, 0.15, 0.20, 0.15, 0.50, "F0_fc_smoothing"),
    MPCWeightAction(9, 0.15, 0.25, 0.15, 0.45, "F1_fc_smoothing_batt"),
    MPCWeightAction(10, 0.15, 0.20, 0.20, 0.45, "F2_fc_smoothing_soc"),
    MPCWeightAction(11, 0.20, 0.20, 0.15, 0.45, "F3_fc_smoothing_h2"),
    MPCWeightAction(12, 0.10, 0.20, 0.60, 0.10, "S0_soc_protection"),
    MPCWeightAction(13, 0.10, 0.25, 0.55, 0.10, "S1_soc_protection_batt"),
    MPCWeightAction(14, 0.15, 0.20, 0.55, 0.10, "S2_soc_protection_h2"),
    MPCWeightAction(15, 0.10, 0.20, 0.55, 0.15, "S3_soc_protection_smooth"),
    MPCWeightAction(16, 0.05, 0.15, 0.75, 0.05, "S4_soc_protection_strong"),
    MPCWeightAction(17, 0.05, 0.20, 0.70, 0.05, "S5_soc_protection_batt_strong"),
    MPCWeightAction(18, 0.05, 0.15, 0.70, 0.10, "S6_soc_protection_smooth_strong"),
    MPCWeightAction(19, 0.10, 0.15, 0.70, 0.05, "S7_soc_protection_h2_strong"),
    MPCWeightAction(20, 0.15, 0.20, 0.50, 0.15, "B4_balanced_safe"),
    MPCWeightAction(21, 0.15, 0.25, 0.45, 0.15, "B5_balanced_batt_safe"),
    MPCWeightAction(22, 0.25, 0.15, 0.50, 0.10, "E4_h2_economy_safe"),
    MPCWeightAction(23, 0.30, 0.15, 0.45, 0.10, "E5_h2_economy_safe_high"),
    MPCWeightAction(24, 0.25, 0.20, 0.45, 0.10, "E6_h2_economy_safe_batt"),
    MPCWeightAction(25, 0.10, 0.15, 0.45, 0.30, "F4_fc_smoothing_safe"),
    MPCWeightAction(26, 0.10, 0.15, 0.50, 0.25, "F5_fc_smoothing_safe_soc"),
    MPCWeightAction(27, 0.10, 0.20, 0.40, 0.30, "F6_fc_smoothing_safe_batt"),
    MPCWeightAction(28, 0.20, 0.10, 0.65, 0.05, "E7_h2_economy_protected"),
    MPCWeightAction(29, 0.20, 0.15, 0.60, 0.05, "E8_h2_economy_protected_batt"),
    MPCWeightAction(30, 0.15, 0.15, 0.65, 0.05, "E9_h2_economy_moderate_protected"),
    MPCWeightAction(31, 0.05, 0.10, 0.60, 0.25, "F7_fc_smoothing_protected"),
    MPCWeightAction(32, 0.05, 0.15, 0.60, 0.20, "F8_fc_smoothing_protected_batt"),
    MPCWeightAction(33, 0.05, 0.10, 0.65, 0.20, "F9_fc_smoothing_protected_soc"),
    MPCWeightAction(34, 0.10, 0.15, 0.70, 0.05, "E10_h2_economy_feasible"),
    MPCWeightAction(35, 0.15, 0.10, 0.70, 0.05, "E11_h2_economy_feasible_high"),
    MPCWeightAction(36, 0.10, 0.10, 0.75, 0.05, "E12_h2_economy_feasible_soc"),
)

WINDOWS = (
    ("low_stable", "train_parent_037_02", 0, 600, 0.55),
    ("ordinary_stable", "train_parent_061_01", 1710, 600, 0.55),
    ("fluctuating", "train_parent_063_01", 5010, 600, 0.55),
    ("high_load", "train_parent_013_02", 2400, 600, 0.55),
    ("sustained_high", "train_parent_021_01", 2820, 600, 0.55),
    ("rapid_rise", "train_parent_021_01", 2310, 600, 0.55),
    ("soc_pressure", "train_parent_061_01", 1710, 600, 0.35),
)

REFERENCE_STATE_PROBE = (
    REPO_ROOT
    / "outputs"
    / "mpc_action_redesign_20260911"
    / "common_reward_state_winners"
    / "state_probe.csv"
)
SCALE_REFERENCE_ACTION_IDS = (0, 4, 8, 12)
FULL_SCAN_ACTION_IDS = (13, 17, 31, 34, 35, 36)
FULL_SCAN_SEGMENTS = (
    ("calibration", "low_load", "train_parent_037_02"),
    ("calibration", "ordinary", "train_parent_061_01"),
    ("calibration", "fluctuating", "train_parent_063_01"),
    ("calibration", "sustained_high", "train_parent_013_02"),
    ("calibration", "soc_pressure", "train_parent_021_01"),
    ("parent_holdout", "low_load", "train_parent_032_04"),
    ("parent_holdout", "ordinary", "train_parent_009_01"),
    ("parent_holdout", "fluctuating", "train_parent_058_01"),
    ("parent_holdout", "high_load", "train_parent_023_01"),
    ("parent_holdout", "high_volatile", "train_parent_003_01"),
)


def _validate_actions(actions: Iterable[MPCWeightAction]) -> None:
    for action in actions:
        weights = np.asarray(action.as_tuple(), dtype=float)
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError(f"invalid candidate weights: {action}")
        if not np.isclose(float(weights.sum()), 1.0, rtol=0.0, atol=1.0e-12):
            raise ValueError(f"candidate weights do not sum to one: {action}")


def rollout(
    loads: np.ndarray,
    action: MPCWeightAction,
    config: Any,
    *,
    initial_soc: float,
) -> dict[str, Any]:
    env = DqnMpcWeightEnv(
        loads_kw=loads,
        base_config=config,
        initial_soc=initial_soc,
        actions=(action,),
    )
    env.reset()
    previous_fc = float(env.previous_fc_kw)
    rows: list[tuple[float, float, float, float, float, float]] = []
    failure_step: int | None = None
    failure_status = ""
    while not env.done:
        try:
            _, reward, _, info = env.step(action.action_id)
        except MpcSolveFailure as error:
            failure_step = int(error.decision_index)
            failure_status = str(error.solver_status)
            break
        fc = float(info["p_fc_kw"])
        batt = float(info["p_batt_kw"])
        rows.append(
            (
                float(reward),
                float(info["raw_mpc_objective"]),
                fc,
                batt,
                float(info["soc_after"]),
                abs(fc - previous_fc),
            )
        )
        previous_fc = fc

    values = np.asarray(rows, dtype=float)
    steps = int(values.shape[0])
    hours = steps * float(config.dt_seconds) / 3600.0
    if steps:
        h2_kg = float(
            np.sum(
                h2_kg_step_dp0_quadratic(
                    values[:, 2],
                    dt_seconds=config.dt_seconds,
                    p_rated_total_kw=config.fuel_cell_max_kw,
                )
            )
        )
        net_kwh = float(np.sum(values[:, 3]) * config.dt_seconds / 3600.0)
        throughput = float(np.sum(np.abs(values[:, 3])) * config.dt_seconds / 3600.0)
        fc_tv = float(np.sum(values[:, 5]))
    else:
        h2_kg = net_kwh = throughput = fc_tv = float("nan")
    return {
        "completed": failure_step is None,
        "failure_step": failure_step,
        "failure_status": failure_status,
        "steps": steps,
        "hours_executed": hours,
        "reward_per_step": float(np.mean(values[:, 0])) if steps else float("nan"),
        "mpc_objective_per_step": float(np.mean(values[:, 1])) if steps else float("nan"),
        "min_soc": float(np.min(values[:, 4])) if steps else initial_soc,
        "mean_soc": float(np.mean(values[:, 4])) if steps else initial_soc,
        "final_soc": float(env.current_soc),
        "soc_mean_abs_deviation": float(np.mean(np.abs(values[:, 4] - 0.55))) if steps else abs(initial_soc - 0.55),
        "h2_kg_h": h2_kg / hours if steps else float("nan"),
        "battery_net_discharge_kwh_h": net_kwh / hours if steps else float("nan"),
        "battery_throughput_kwh_h": throughput / hours if steps else float("nan"),
        "fc_tv_kw_h": fc_tv / hours if steps else float("nan"),
        "mean_fc_kw": float(np.mean(values[:, 2])) if steps else float("nan"),
        "fc_ge_590_fraction": float(np.mean(values[:, 2] >= 590.0)) if steps else float("nan"),
        "first_p_fc_kw": float(values[0, 2]) if steps else float("nan"),
        "first_p_batt_kw": float(values[0, 3]) if steps else float("nan"),
    }


def run_window_scan(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    _validate_actions(CANDIDATES)
    split = load_formal_operating_split()
    config = build_formal_mpc_config()
    rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    for regime, segment_id, start, steps, initial_soc in WINDOWS:
        all_loads = load_operating_segment_loads("train", segment_id, split=split)
        loads = all_loads[start : start + steps + 1]
        if loads.size != steps + 1:
            raise RuntimeError(f"short calibration window: {segment_id}")
        window_rows.append(
            {
                "regime": regime,
                "segment_id": segment_id,
                "start": start,
                "steps": steps,
                "initial_soc": initial_soc,
                "mean_load_kw": float(np.mean(loads[:-1])),
                "std_load_kw": float(np.std(loads[:-1])),
                "max_load_kw": float(np.max(loads[:-1])),
                "mean_abs_delta_kw": float(np.mean(np.abs(np.diff(loads)))),
            }
        )
        for action in CANDIDATES:
            record = {
                "candidate": action.name,
                "action_id": action.action_id,
                **{f"q_{key}": value for key, value in zip(("h2", "batt", "soc", "fcvar"), action.as_tuple())},
                "regime": regime,
                "segment_id": segment_id,
                "initial_soc": initial_soc,
            }
            record.update(rollout(loads, action, config, initial_soc=initial_soc))
            rows.append(record)
        print(f"WINDOW_DONE {regime} {segment_id}", flush=True)

    details = pd.DataFrame(rows)
    summary = details.groupby(
        ["candidate", "action_id", "q_h2", "q_batt", "q_soc", "q_fcvar"],
        as_index=False,
    ).agg(
        completed=("completed", "sum"),
        failures=("completed", lambda values: int((~values).sum())),
        reward_per_step=("reward_per_step", "mean"),
        mpc_objective_per_step=("mpc_objective_per_step", "mean"),
        worst_min_soc=("min_soc", "min"),
        mean_min_soc=("min_soc", "mean"),
        mean_final_soc=("final_soc", "mean"),
        mean_soc_abs_deviation=("soc_mean_abs_deviation", "mean"),
        h2_kg_h=("h2_kg_h", "mean"),
        battery_net_discharge_kwh_h=("battery_net_discharge_kwh_h", "mean"),
        battery_throughput_kwh_h=("battery_throughput_kwh_h", "mean"),
        fc_tv_kw_h=("fc_tv_kw_h", "mean"),
        mean_fc_kw=("mean_fc_kw", "mean"),
        fc_ge_590_fraction=("fc_ge_590_fraction", "mean"),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(window_rows).to_csv(output_dir / "calibration_windows.csv", index=False)
    details.to_csv(output_dir / "candidate_window_results.csv", index=False)
    summary.to_csv(output_dir / "candidate_window_summary.csv", index=False)
    (output_dir / "candidate_catalog.json").write_text(
        json.dumps([asdict(action) for action in CANDIDATES], indent=2) + "\n",
        encoding="utf-8",
    )
    return details, summary


def _distribution(values: pd.Series) -> dict[str, float]:
    numeric = values.astype(float)
    return {
        "min": float(numeric.min()),
        "p01": float(numeric.quantile(0.01)),
        "p05": float(numeric.quantile(0.05)),
        "p25": float(numeric.quantile(0.25)),
        "median": float(numeric.median()),
        "mean": float(numeric.mean()),
        "std": float(numeric.std(ddof=0)),
        "p75": float(numeric.quantile(0.75)),
        "p95": float(numeric.quantile(0.95)),
        "p99": float(numeric.quantile(0.99)),
        "max": float(numeric.max()),
    }


def run_scale_audit(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audit H/B/S/F on the frozen 840 Train states and four semantic endpoints."""

    if not REFERENCE_STATE_PROBE.exists():
        raise FileNotFoundError(REFERENCE_STATE_PROBE)
    source = pd.read_csv(REFERENCE_STATE_PROBE).drop_duplicates("state_id")
    if len(source) != 840:
        raise RuntimeError(f"expected 840 frozen Train states, found {len(source)}")
    actions = tuple(CANDIDATES[index] for index in SCALE_REFERENCE_ACTION_IDS)
    _validate_actions(actions)
    config = build_formal_mpc_config()
    bank = MpcWeightSolverBank(config, actions=actions)
    rows: list[dict[str, Any]] = []
    for row in source.itertuples(index=False):
        for action in actions:
            result, solve_ms = bank.solve(
                action_id=action.action_id,
                load_forecast_kw=np.full(config.horizon, float(row.current_load_kw)),
                current_soc=float(row.soc),
                prev_fc_kw=float(row.previous_fc_kw),
                soc_reference=float(config.soc_reference),
            )
            solved = bool(result.x is not None and str(result.info.status).lower().startswith("solved"))
            record: dict[str, Any] = {
                "state_id": int(row.state_id),
                "segment_id": str(row.segment_id),
                "soc": float(row.soc),
                "current_load_kw": float(row.current_load_kw),
                "load_delta_kw": float(row.load_delta_kw),
                "previous_fc_kw": float(row.previous_fc_kw),
                "candidate": action.name,
                "action_id": action.action_id,
                "solved": solved,
                "solver_status": str(result.info.status),
                "solve_ms": float(solve_ms),
            }
            if solved:
                terms = result.mpc_objective_terms
                solution = np.asarray(result.x, dtype=float)
                record.update(
                    {
                        "H": float(terms["h2_norm"]),
                        "B": float(terms["battery_power_sq_norm"]),
                        "S": float(terms["soc_reference_sq_norm"]),
                        "F": float(terms["fc_variation_sq_norm"]),
                        "J": float(result.raw_mpc_objective),
                        "reward": float(1.0 / (1.0 + result.raw_mpc_objective)),
                        "p_fc0_kw": float(solution[0]),
                        "p_batt0_kw": float(solution[config.horizon]),
                    }
                )
            rows.append(record)
    probes = pd.DataFrame(rows)
    if not bool(probes["solved"].all()):
        raise RuntimeError("scale audit contains an unsolved state-action probe")
    load_low, load_high = source["current_load_kw"].quantile([1 / 3, 2 / 3])
    probes["soc_group"] = pd.cut(
        probes["soc"],
        bins=[-np.inf, 0.50, 0.55, 0.60, np.inf],
        labels=["soc_lt_050", "soc_050_055", "soc_055_060", "soc_gt_060"],
        right=False,
    ).astype(str)
    probes["load_group"] = pd.cut(
        probes["current_load_kw"],
        bins=[-np.inf, load_low, load_high, np.inf],
        labels=["load_low", "load_medium", "load_high"],
        include_lowest=True,
    ).astype(str)
    delta_threshold = max(float(source["load_delta_kw"].abs().quantile(0.70)), 1.0)
    probes["transition_group"] = np.select(
        [probes["load_delta_kw"] >= delta_threshold, probes["load_delta_kw"] <= -delta_threshold],
        ["load_rise", "load_fall"],
        default="load_stable",
    )
    probes["fc_group"] = pd.cut(
        probes["previous_fc_kw"],
        bins=[-np.inf, 200.0, 400.0, 590.0, np.inf],
        labels=["fc_lt_200", "fc_200_400", "fc_400_590", "fc_ge_590"],
        right=False,
    ).astype(str)

    summary_rows: list[dict[str, Any]] = []
    for term in ("H", "B", "S", "F"):
        summary_rows.append({"scope": "overall", "group": "all", "term": term, "count": len(probes), **_distribution(probes[term])})
    for group_column in ("soc_group", "load_group", "transition_group", "fc_group"):
        for group_name, group in probes.groupby(group_column):
            for term in ("H", "B", "S", "F"):
                summary_rows.append(
                    {
                        "scope": group_column,
                        "group": str(group_name),
                        "term": term,
                        "count": len(group),
                        **_distribution(group[term]),
                    }
                )
    summary = pd.DataFrame(summary_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    probes.to_csv(output_dir / "base_term_state_action_probe.csv", index=False)
    summary.to_csv(output_dir / "base_term_scale_summary.csv", index=False)
    pd.DataFrame(
        [
            {"term": "H", "engineering_unit": "FC at 600 kW for one step", "per_step_value": 1.0, "six_step_value": 6.0},
            {"term": "B", "engineering_unit": "|battery power| at 624 kW for one step", "per_step_value": 1.0, "six_step_value": 6.0},
            {"term": "S", "engineering_unit": "|SOC-0.55| at 0.05 for one predicted state", "per_step_value": 1.0, "six_step_value": 6.0},
            {"term": "F", "engineering_unit": "|FC delta| at 48 kW for one move", "per_step_value": 1.0, "six_step_value": 6.0},
        ]
    ).to_csv(output_dir / "base_term_engineering_references.csv", index=False)
    return probes, summary


def run_full_segment_scan(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare finalists on calibration and parent-disjoint Train segments."""

    split = load_formal_operating_split()
    config = build_formal_mpc_config()
    actions = tuple(CANDIDATES[index] for index in FULL_SCAN_ACTION_IDS)
    _validate_actions(actions)
    rows: list[dict[str, Any]] = []
    for subset, regime, segment_id in FULL_SCAN_SEGMENTS:
        loads = load_operating_segment_loads("train", segment_id, split=split)
        for action in actions:
            record = {
                "subset": subset,
                "regime": regime,
                "segment_id": segment_id,
                "candidate": action.name,
                "action_id": action.action_id,
                **{f"q_{key}": value for key, value in zip(("h2", "batt", "soc", "fcvar"), action.as_tuple())},
            }
            record.update(rollout(loads, action, config, initial_soc=0.55))
            rows.append(record)
        print(f"SEGMENT_DONE {subset} {regime} {segment_id}", flush=True)
    details = pd.DataFrame(rows)
    summary = details.groupby(
        ["subset", "candidate", "action_id", "q_h2", "q_batt", "q_soc", "q_fcvar"],
        as_index=False,
    ).agg(
        completed=("completed", "sum"),
        failures=("completed", lambda values: int((~values).sum())),
        executed_hours=("hours_executed", "sum"),
        reward_per_step=("reward_per_step", "mean"),
        mpc_objective_per_step=("mpc_objective_per_step", "mean"),
        worst_min_soc=("min_soc", "min"),
        mean_min_soc=("min_soc", "mean"),
        mean_final_soc=("final_soc", "mean"),
        mean_soc_abs_deviation=("soc_mean_abs_deviation", "mean"),
        h2_kg_h=("h2_kg_h", "mean"),
        battery_net_discharge_kwh_h=("battery_net_discharge_kwh_h", "mean"),
        battery_throughput_kwh_h=("battery_throughput_kwh_h", "mean"),
        fc_tv_kw_h=("fc_tv_kw_h", "mean"),
        mean_fc_kw=("mean_fc_kw", "mean"),
        fc_ge_590_fraction=("fc_ge_590_fraction", "mean"),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    details.to_csv(output_dir / "finalist_full_train_segment_results.csv", index=False)
    summary.to_csv(output_dir / "finalist_full_train_summary.csv", index=False)
    return details, summary


def _add_state_groups(frame: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    load_low, load_high = source["current_load_kw"].quantile([1 / 3, 2 / 3])
    result["soc_group"] = pd.cut(
        result["soc"],
        bins=[-np.inf, 0.50, 0.55, 0.60, np.inf],
        labels=["soc_lt_050", "soc_050_055", "soc_055_060", "soc_gt_060"],
        right=False,
    ).astype(str)
    result["load_group"] = pd.cut(
        result["current_load_kw"],
        bins=[-np.inf, load_low, load_high, np.inf],
        labels=["load_low", "load_medium", "load_high"],
        include_lowest=True,
    ).astype(str)
    delta_threshold = max(float(source["load_delta_kw"].abs().quantile(0.70)), 1.0)
    result["transition_group"] = np.select(
        [result["load_delta_kw"] >= delta_threshold, result["load_delta_kw"] <= -delta_threshold],
        ["load_rise", "load_fall"],
        default="load_stable",
    )
    result["fc_group"] = pd.cut(
        result["previous_fc_kw"],
        bins=[-np.inf, 200.0, 400.0, 590.0, np.inf],
        labels=["fc_lt_200", "fc_200_400", "fc_400_590", "fc_ge_590"],
        right=False,
    ).astype(str)
    return result


def run_final_reward_audit(output_dir: Path) -> dict[str, pd.DataFrame]:
    """Run the final 840-state x four-action objective/reward audit."""

    source = pd.read_csv(REFERENCE_STATE_PROBE).drop_duplicates("state_id")
    if len(source) != 840:
        raise RuntimeError(f"expected 840 frozen Train states, found {len(source)}")
    _validate_actions(DQN_MPC_WEIGHT_ACTIONS)
    config = build_formal_mpc_config()
    bank = MpcWeightSolverBank(config, actions=DQN_MPC_WEIGHT_ACTIONS)
    rows: list[dict[str, Any]] = []
    for row in source.itertuples(index=False):
        for action in DQN_MPC_WEIGHT_ACTIONS:
            result, solve_ms = bank.solve(
                action_id=action.action_id,
                load_forecast_kw=np.full(config.horizon, float(row.current_load_kw)),
                current_soc=float(row.soc),
                prev_fc_kw=float(row.previous_fc_kw),
                soc_reference=float(config.soc_reference),
            )
            solved = bool(result.x is not None and str(result.info.status).lower().startswith("solved"))
            record: dict[str, Any] = {
                "state_id": int(row.state_id),
                "segment_id": str(row.segment_id),
                "soc": float(row.soc),
                "current_load_kw": float(row.current_load_kw),
                "load_delta_kw": float(row.load_delta_kw),
                "previous_fc_kw": float(row.previous_fc_kw),
                "action_id": action.action_id,
                "action_name": action.name,
                "solved": solved,
                "solver_status": str(result.info.status),
                "solve_ms": float(solve_ms),
            }
            if solved:
                terms = result.mpc_objective_terms
                solution = np.asarray(result.x, dtype=float)
                component_values = np.asarray(
                    [
                        terms["h2_norm"],
                        terms["battery_power_sq_norm"],
                        terms["soc_reference_sq_norm"],
                        terms["fc_variation_sq_norm"],
                    ],
                    dtype=float,
                )
                weights = np.asarray(action.as_tuple(), dtype=float)
                contributions = weights * component_values
                record.update(
                    {
                        "H": component_values[0],
                        "B": component_values[1],
                        "S": component_values[2],
                        "F": component_values[3],
                        "qH": contributions[0],
                        "qB": contributions[1],
                        "qS": contributions[2],
                        "qF": contributions[3],
                        "J": float(result.raw_mpc_objective),
                        "reward": float(1.0 / (1.0 + result.raw_mpc_objective)),
                        "p_fc0_kw": float(solution[0]),
                        "p_batt0_kw": float(solution[config.horizon]),
                        "p_fc_plan": json.dumps(solution[: config.horizon].tolist()),
                        "p_batt_plan": json.dumps(solution[config.horizon : 2 * config.horizon].tolist()),
                    }
                )
            rows.append(record)
    probes = pd.DataFrame(rows)
    if not bool(probes["solved"].all()):
        raise RuntimeError("final reward audit contains an unsolved probe")
    probes = _add_state_groups(probes, source)

    winners: list[dict[str, Any]] = []
    for state_id, group in probes.groupby("state_id"):
        ranked = group.sort_values(["reward", "action_id"], ascending=[False, True])
        best, second = ranked.iloc[0], ranked.iloc[1]
        winners.append(
            {
                "state_id": int(state_id),
                "winner_action_id": int(best["action_id"]),
                "winner_action_name": str(best["action_name"]),
                "best_reward": float(best["reward"]),
                "second_reward": float(second["reward"]),
                "reward_gap": float(best["reward"] - second["reward"]),
                "relative_reward_gap": float((best["reward"] - second["reward"]) / max(abs(best["reward"]), 1.0e-12)),
                "soc_group": str(best["soc_group"]),
                "load_group": str(best["load_group"]),
                "transition_group": str(best["transition_group"]),
                "fc_group": str(best["fc_group"]),
            }
        )
    winner_frame = pd.DataFrame(winners)

    distribution_rows: list[dict[str, Any]] = []
    for action, group in probes.groupby(["action_id", "action_name"]):
        action_id, action_name = action
        for metric in ("J", "reward", "qH", "qB", "qS", "qF"):
            distribution_rows.append(
                {
                    "action_id": int(action_id),
                    "action_name": str(action_name),
                    "metric": metric,
                    "count": len(group),
                    **_distribution(group[metric]),
                }
            )
    distributions = pd.DataFrame(distribution_rows)

    winner_rows: list[dict[str, Any]] = []
    for group_column in (None, "soc_group", "load_group", "transition_group", "fc_group"):
        grouped = [("all", winner_frame)] if group_column is None else winner_frame.groupby(group_column)
        for group_name, group in grouped:
            for action in DQN_MPC_WEIGHT_ACTIONS:
                count = int(group["winner_action_id"].eq(action.action_id).sum())
                winner_rows.append(
                    {
                        "scope": "overall" if group_column is None else group_column,
                        "group": str(group_name),
                        "action_id": action.action_id,
                        "action_name": action.name,
                        "state_count": len(group),
                        "winner_count": count,
                        "winner_share": count / len(group),
                    }
                )
    winner_summary = pd.DataFrame(winner_rows)

    pairwise_rows: list[dict[str, Any]] = []
    for left in DQN_MPC_WEIGHT_ACTIONS:
        for right in DQN_MPC_WEIGHT_ACTIONS[left.action_id + 1 :]:
            a = probes.loc[probes["action_id"].eq(left.action_id)].set_index("state_id")
            b = probes.loc[probes["action_id"].eq(right.action_id)].set_index("state_id")
            joined = a.join(b, lsuffix="_a", rsuffix="_b")
            fc0 = (joined["p_fc0_kw_a"] - joined["p_fc0_kw_b"]).abs()
            batt0 = (joined["p_batt0_kw_a"] - joined["p_batt0_kw_b"]).abs()
            fc_rms = joined.apply(
                lambda item: float(np.sqrt(np.mean((np.asarray(json.loads(item["p_fc_plan_a"])) - np.asarray(json.loads(item["p_fc_plan_b"]))) ** 2))),
                axis=1,
            )
            pairwise_rows.append(
                {
                    "pair": f"A{left.action_id}-A{right.action_id}",
                    "median_abs_p_fc0_kw": float(fc0.median()),
                    "p95_abs_p_fc0_kw": float(fc0.quantile(0.95)),
                    "max_abs_p_fc0_kw": float(fc0.max()),
                    "median_abs_p_batt0_kw": float(batt0.median()),
                    "median_fc_plan_rms_kw": float(fc_rms.median()),
                    "p95_fc_plan_rms_kw": float(fc_rms.quantile(0.95)),
                    "share_fc_plan_rms_lt_1kw": float((fc_rms < 1.0).mean()),
                }
            )
    pairwise = pd.DataFrame(pairwise_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    probes.to_csv(output_dir / "final_state_action_probe.csv", index=False)
    winner_frame.to_csv(output_dir / "final_state_winners.csv", index=False)
    distributions.to_csv(output_dir / "final_objective_reward_distribution.csv", index=False)
    winner_summary.to_csv(output_dir / "final_winner_summary.csv", index=False)
    pairwise.to_csv(output_dir / "final_pairwise_action_differences.csv", index=False)
    return {
        "probes": probes,
        "winners": winner_frame,
        "distributions": distributions,
        "winner_summary": winner_summary,
        "pairwise": pairwise,
    }


def run_final_closed_loop(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    split = load_formal_operating_split()
    config = build_formal_mpc_config()
    _validate_actions(DQN_MPC_WEIGHT_ACTIONS)
    rows: list[dict[str, Any]] = []
    for subset, regime, segment_id in FULL_SCAN_SEGMENTS:
        loads = load_operating_segment_loads("train", segment_id, split=split)
        for action in DQN_MPC_WEIGHT_ACTIONS:
            record = {
                "subset": subset,
                "regime": regime,
                "segment_id": segment_id,
                "action_id": action.action_id,
                "action_name": action.name,
                **{f"q_{key}": value for key, value in zip(("h2", "batt", "soc", "fcvar"), action.as_tuple())},
            }
            record.update(rollout(loads, action, config, initial_soc=0.55))
            rows.append(record)
        print(f"FINAL_SEGMENT_DONE {subset} {regime} {segment_id}", flush=True)
    details = pd.DataFrame(rows)
    summary = details.groupby(
        ["subset", "action_id", "action_name", "q_h2", "q_batt", "q_soc", "q_fcvar"],
        as_index=False,
    ).agg(
        completed=("completed", "sum"),
        failures=("completed", lambda values: int((~values).sum())),
        executed_hours=("hours_executed", "sum"),
        reward_per_step=("reward_per_step", "mean"),
        mpc_objective_per_step=("mpc_objective_per_step", "mean"),
        worst_min_soc=("min_soc", "min"),
        mean_min_soc=("min_soc", "mean"),
        mean_final_soc=("final_soc", "mean"),
        mean_soc_abs_deviation=("soc_mean_abs_deviation", "mean"),
        h2_kg_h=("h2_kg_h", "mean"),
        battery_net_discharge_kwh_h=("battery_net_discharge_kwh_h", "mean"),
        battery_throughput_kwh_h=("battery_throughput_kwh_h", "mean"),
        fc_tv_kw_h=("fc_tv_kw_h", "mean"),
        mean_fc_kw=("mean_fc_kw", "mean"),
        fc_ge_590_fraction=("fc_ge_590_fraction", "mean"),
    )
    details.to_csv(output_dir / "final_action_closed_loop_results.csv", index=False)
    summary.to_csv(output_dir / "final_action_closed_loop_summary.csv", index=False)
    return details, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--mode", choices=("scale", "scan", "full", "final", "all"), default="all")
    args = parser.parse_args()
    if args.mode in ("scale", "all"):
        _, scale_summary = run_scale_audit(args.output_dir)
        print(scale_summary.loc[scale_summary["scope"].eq("overall")].to_string(index=False), flush=True)
    if args.mode in ("scan", "all"):
        _, candidate_summary = run_window_scan(args.output_dir)
        print(candidate_summary.to_string(index=False), flush=True)
    if args.mode in ("full", "all"):
        _, full_summary = run_full_segment_scan(args.output_dir)
        print(full_summary.to_string(index=False), flush=True)
    if args.mode in ("final", "all"):
        final = run_final_reward_audit(args.output_dir)
        print(final["winner_summary"].loc[final["winner_summary"]["scope"].eq("overall")].to_string(index=False), flush=True)
        print(final["pairwise"].to_string(index=False), flush=True)
        _, closed_loop = run_final_closed_loop(args.output_dir)
        print(closed_loop.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
