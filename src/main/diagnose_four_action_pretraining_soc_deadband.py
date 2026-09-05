"""Read-only, train-split diagnostic of the four SOC-deadband MPC actions.

This entry point never creates an agent, replay buffer, checkpoint, or training
update.  It uses fixed A0 rollouts solely to obtain physically reachable
controller states, then probes every action with the same causal persistence
forecast.  Validation and test splits are never opened.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dqn.utils.action_mapper import DQN_MPC_WEIGHT_ACTIONS
from dqn.utils.reward import calculate_mpc_weight_reward
from mpc.solvers.fc_dp0_curve import h2_kg_step_dp0_quadratic
from mpc_solvers.dqn_mpc_solver_bank import MpcWeightSolverBank
from mpc_solvers.formal_config import (
    FIXED_SOC_REFERENCE,
    SOC_SOFT_MAX,
    SOC_SOFT_MIN,
    SOC_SOFT_SCALE,
    build_formal_mpc_config,
)
from utils.formal_operating_dataset import (
    load_formal_operating_split,
    load_operating_segment_loads,
)


OUTPUT_DIR = (
    REPO_ROOT / "outputs" / "action_space" / "four_action_pretraining_soc_deadband"
)
REQUIRED_OUTPUT_FILENAMES = {
    "summary": "summary.json",
    "state_probe": "state_probe.csv",
    "pairwise": "pairwise_action_differences.csv",
    "regime": "regime_summary.csv",
    "rollout": "fixed_action_rollout_summary.csv",
}
SOC_STRESS_VALUES = (0.45, 0.49, 0.50, 0.55, 0.60, 0.61, 0.65)
ACTION_IDS = tuple(action.action_id for action in DQN_MPC_WEIGHT_ACTIONS)


def soc_deadband_violation(soc: float) -> float:
    """Distance from the closed formal SOC soft working range."""

    value = float(soc)
    return max(0.0, SOC_SOFT_MIN - value, value - SOC_SOFT_MAX)


def _quantile_label(value: float, low: float, high: float, *, prefix: str) -> str:
    if value <= low:
        return f"{prefix}_low"
    if value >= high:
        return f"{prefix}_high"
    return f"{prefix}_medium"


def _status_is_solved(result: Any) -> bool:
    return bool(
        result.x is not None
        and str(result.info.status).lower().startswith("solved")
    )


def _initial_controller_state(
    loads_kw: np.ndarray,
    config: Any,
    *,
    initial_soc: float = FIXED_SOC_REFERENCE,
) -> tuple[float, float, float]:
    first_load = float(loads_kw[0])
    previous_fc = float(
        np.clip(first_load, config.fuel_cell_min_kw, config.fuel_cell_max_kw)
    )
    return float(initial_soc), previous_fc, first_load - previous_fc


def _step_physics(
    *,
    result: Any,
    load_actual_kw: float,
    soc_before: float,
    config: Any,
) -> tuple[float, float, float]:
    solution = np.asarray(result.x, dtype=np.float64).reshape(-1)
    p_fc = float(solution[0])
    p_batt = float(load_actual_kw) - p_fc
    next_soc = float(
        soc_before
        - p_batt * float(config.dt_seconds) / 3600.0 / float(config.battery_capacity_kwh)
    )
    return p_fc, p_batt, next_soc


def _summarize_train_segments(split: Any) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for segment_id in split.train_segments:
        loads = load_operating_segment_loads("train", segment_id, split=split)
        deltas = np.diff(loads)
        rows.append(
            {
                "segment_id": segment_id,
                "points": int(loads.size),
                "mean_load_kw": float(np.mean(loads)),
                "max_load_kw": float(np.max(loads)),
                "mean_abs_delta_kw": float(np.mean(np.abs(deltas))),
                "p95_delta_kw": float(np.quantile(deltas, 0.95)),
                "p05_delta_kw": float(np.quantile(deltas, 0.05)),
            }
        )
    return pd.DataFrame(rows)


def _choose_segments(segment_summary: pd.DataFrame, count: int = 8) -> list[str]:
    """Select distinct train-only segments spanning load and transition regimes."""

    median_mean = float(segment_summary["mean_load_kw"].median())
    targets = (
        ("mean_load_kw", "min"),
        ("mean_load_kw", "max"),
        ("max_load_kw", "max"),
        ("mean_abs_delta_kw", "max"),
        ("p95_delta_kw", "max"),
        ("p05_delta_kw", "min"),
        ("mean_load_kw", "nearest_median"),
    )
    chosen: list[str] = []
    for column, selector in targets:
        if selector == "min":
            row = segment_summary.loc[segment_summary[column].idxmin()]
        elif selector == "max":
            row = segment_summary.loc[segment_summary[column].idxmax()]
        else:
            row = segment_summary.loc[(segment_summary[column] - median_mean).abs().idxmin()]
        identifier = str(row["segment_id"])
        if identifier not in chosen:
            chosen.append(identifier)
    for identifier in segment_summary.sort_values("mean_abs_delta_kw", ascending=False)["segment_id"]:
        if str(identifier) not in chosen:
            chosen.append(str(identifier))
        if len(chosen) >= count:
            break
    return chosen[:count]


def _collect_a0_reachable_states(
    *,
    segment_ids: Iterable[str],
    split: Any,
    config: Any,
    steps_per_segment: int,
) -> list[dict[str, Any]]:
    bank = MpcWeightSolverBank(config)
    rows: list[dict[str, Any]] = []
    for segment_id in segment_ids:
        loads = load_operating_segment_loads("train", segment_id, split=split)
        soc, previous_fc, previous_batt = _initial_controller_state(loads, config)
        max_steps = min(int(steps_per_segment), int(loads.size - 1))
        for index in range(max_steps):
            current_load = float(loads[index])
            previous_load = float(loads[index - 1]) if index else current_load
            result, solve_ms = bank.solve(
                action_id=0,
                load_forecast_kw=np.full(config.horizon, current_load),
                current_soc=soc,
                prev_fc_kw=previous_fc,
                soc_reference=FIXED_SOC_REFERENCE,
            )
            if not _status_is_solved(result):
                break
            rows.append(
                {
                    "segment_id": str(segment_id),
                    "decision_index": int(index),
                    "soc_reachable": float(soc),
                    "previous_fc_kw": float(previous_fc),
                    "previous_batt_kw": float(previous_batt),
                    "current_load_kw": current_load,
                    "previous_load_kw": previous_load,
                    "load_delta_kw": current_load - previous_load,
                    "mean_load_10s_kw": float(np.mean(loads[max(0, index - 9) : index + 1])),
                    "mean_load_60s_kw": float(np.mean(loads[max(0, index - 59) : index + 1])),
                    "a0_nominal_solve_ms": float(solve_ms),
                }
            )
            next_load = float(loads[index + 1])
            previous_fc, previous_batt, soc = _step_physics(
                result=result,
                load_actual_kw=next_load,
                soc_before=soc,
                config=config,
            )
    if not rows:
        raise RuntimeError("fixed A0 train rollouts produced no reachable states")
    return rows


def _stratified_anchor_states(reachable: list[dict[str, Any]], anchors: int = 30) -> list[dict[str, Any]]:
    frame = pd.DataFrame(reachable).copy()
    load_low, load_high = frame["current_load_kw"].quantile([1 / 3, 2 / 3])
    delta_scale = max(float(frame["load_delta_kw"].abs().quantile(0.70)), 1.0)
    frame["load_regime"] = [
        _quantile_label(float(value), float(load_low), float(load_high), prefix="load")
        for value in frame["current_load_kw"]
    ]
    frame["transition_regime"] = np.select(
        [frame["load_delta_kw"] >= delta_scale, frame["load_delta_kw"] <= -delta_scale],
        ["rapid_rise", "rapid_fall"],
        default="stable",
    )
    frame["stratum"] = frame["load_regime"] + "__" + frame["transition_regime"]
    selected: list[int] = []
    groups = [group.index.tolist() for _, group in frame.groupby("stratum", sort=True)]
    cursor = 0
    while len(selected) < anchors and any(groups):
        group = groups[cursor % len(groups)]
        if group:
            selected.append(group.pop(len(group) // 2))
        groups = [item for item in groups if item]
        cursor += 1
    if len(selected) < anchors:
        for index in frame.index:
            if index not in selected:
                selected.append(int(index))
            if len(selected) >= anchors:
                break
    return frame.loc[selected[:anchors]].to_dict(orient="records")


def _objective_components(solution: np.ndarray, config: Any, action_id: int, previous_fc_kw: float) -> dict[str, float]:
    horizon = int(config.horizon)
    p_fc = solution[:horizon]
    p_batt = solution[horizon : 2 * horizon]
    soc = solution[2 * horizon : 3 * horizon + 1]
    action = DQN_MPC_WEIGHT_ACTIONS[action_id]
    h2_reference = float(
        h2_kg_step_dp0_quadratic(config.fuel_cell_max_kw, dt_seconds=config.dt_seconds, p_rated_total_kw=config.fuel_cell_max_kw)
    )
    h2_norm = float(np.sum(h2_kg_step_dp0_quadratic(p_fc, dt_seconds=config.dt_seconds, p_rated_total_kw=config.fuel_cell_max_kw)) / h2_reference)
    batt_norm = float(np.sum((p_batt / config.battery_power_ref_kw) ** 2))
    violation = np.maximum.reduce((np.zeros(horizon), config.soc_soft_min - soc[1:], soc[1:] - config.soc_soft_max))
    soc_norm = float(np.sum((violation / config.soc_band) ** 2))
    fc_delta = np.concatenate(([p_fc[0] - previous_fc_kw], np.diff(p_fc)))
    fc_var_norm = float(np.sum((fc_delta / config.fuel_cell_ramp_rate_kw_per_s) ** 2))
    weighted_h2 = float(action.q_h2 * h2_norm)
    weighted_batt = float(action.q_batt * batt_norm)
    weighted_soc = float(action.q_soc * soc_norm)
    weighted_fc = float(action.q_fc_var * fc_var_norm)
    return {
        "h2_objective_component": h2_norm,
        "battery_objective_component": batt_norm,
        "soc_deadband_objective_component": soc_norm,
        "fc_variation_objective_component": fc_var_norm,
        "weighted_h2": weighted_h2,
        "weighted_batt": weighted_batt,
        "weighted_soc": weighted_soc,
        "weighted_fcvar": weighted_fc,
        "mpc_weighted_total_objective": weighted_h2 + weighted_batt + weighted_soc + weighted_fc,
    }


def _probe_states(*, anchors: list[dict[str, Any]], config: Any) -> pd.DataFrame:
    bank = MpcWeightSolverBank(config)
    rows: list[dict[str, Any]] = []
    state_id = 0
    for anchor_id, anchor in enumerate(anchors):
        for stress_soc in SOC_STRESS_VALUES:
            state_id += 1
            load_regime = str(anchor["load_regime"])
            transition_regime = str(anchor["transition_regime"])
            for action_id in ACTION_IDS:
                result, solve_ms = bank.solve(
                    action_id=action_id,
                    load_forecast_kw=np.full(config.horizon, float(anchor["current_load_kw"])),
                    current_soc=float(stress_soc),
                    prev_fc_kw=float(anchor["previous_fc_kw"]),
                    soc_reference=FIXED_SOC_REFERENCE,
                )
                solved = _status_is_solved(result)
                row: dict[str, Any] = {
                    "state_id": state_id,
                    "anchor_id": anchor_id,
                    "segment_id": anchor["segment_id"],
                    "decision_index": int(anchor["decision_index"]),
                    "soc": float(stress_soc),
                    "soc_regime": (
                        "below_soft_range" if stress_soc < SOC_SOFT_MIN else "above_soft_range" if stress_soc > SOC_SOFT_MAX else "inside_soft_range"
                    ),
                    "load_regime": load_regime,
                    "transition_regime": transition_regime,
                    "previous_fc_kw": float(anchor["previous_fc_kw"]),
                    "previous_batt_kw": float(anchor["previous_batt_kw"]),
                    "current_load_kw": float(anchor["current_load_kw"]),
                    "previous_load_kw": float(anchor["previous_load_kw"]),
                    "load_delta_kw": float(anchor["load_delta_kw"]),
                    "mean_load_10s_kw": float(anchor["mean_load_10s_kw"]),
                    "mean_load_60s_kw": float(anchor["mean_load_60s_kw"]),
                    "action_id": int(action_id),
                    "action_name": DQN_MPC_WEIGHT_ACTIONS[action_id].name,
                    "solved": bool(solved),
                    "solver_status": str(result.info.status),
                    "solve_ms": float(solve_ms),
                }
                if solved:
                    solution = np.asarray(result.x, dtype=np.float64).reshape(-1)
                    horizon = int(config.horizon)
                    p_fc = solution[:horizon]
                    p_batt = solution[horizon : 2 * horizon]
                    soc_plan = solution[2 * horizon : 3 * horizon + 1]
                    reward, reward_info = calculate_mpc_weight_reward(
                        p_fc_kw=float(p_fc[0]),
                        p_batt_kw=float(p_batt[0]),
                        next_soc=float(soc_plan[1]),
                        previous_fc_kw=float(anchor["previous_fc_kw"]),
                    )
                    row.update(
                        {
                            "p_fc0_kw": float(p_fc[0]),
                            "p_batt0_kw": float(p_batt[0]),
                            "soc1": float(soc_plan[1]),
                            "p_fc_trajectory_kw": json.dumps(p_fc.tolist()),
                            "p_batt_trajectory_kw": json.dumps(p_batt.tolist()),
                            "soc_trajectory": json.dumps(soc_plan.tolist()),
                            "common_reward": float(reward),
                            "common_cost": float(reward_info["total_cost"]),
                            **_objective_components(solution, config, action_id, float(anchor["previous_fc_kw"])),
                        }
                    )
                rows.append(row)
    frame = pd.DataFrame(rows)
    for state, indexes in frame.groupby("state_id").groups.items():
        subset = frame.loc[indexes]
        solved = subset.loc[subset["solved"]]
        if solved.empty:
            continue
        ranked = solved.sort_values(["common_cost", "action_id"])
        best = ranked.iloc[0]
        second_cost = float(ranked.iloc[1]["common_cost"]) if len(ranked) > 1 else float("nan")
        gap = second_cost - float(best["common_cost"])
        relative_gap = gap / max(abs(float(best["common_cost"])), 1.0e-8)
        frame.loc[indexes, "winner_action_id"] = int(best["action_id"])
        frame.loc[indexes, "winner_action_name"] = str(best["action_name"])
        frame.loc[indexes, "best_common_reward"] = -float(best["common_cost"])
        frame.loc[indexes, "second_best_common_reward"] = -second_cost
        frame.loc[indexes, "absolute_common_cost_gap"] = gap
        frame.loc[indexes, "relative_common_cost_gap"] = relative_gap
        frame.loc[indexes, "near_tie_le_1pct"] = bool(relative_gap <= 0.01)
        frame.loc[indexes, "near_tie_le_5pct"] = bool(relative_gap <= 0.05)
    return frame


def _summary_stats(values: pd.Series) -> dict[str, float]:
    numeric = values.dropna().astype(float)
    return {
        "mean": float(numeric.mean()),
        "median": float(numeric.median()),
        "p90": float(numeric.quantile(0.90)),
        "max": float(numeric.max()),
    }


def _pairwise_differences(probes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    solved = probes.loc[probes["solved"]].copy()
    for left, right in itertools.combinations(ACTION_IDS, 2):
        a = solved.loc[solved["action_id"].eq(left)].set_index("state_id")
        b = solved.loc[solved["action_id"].eq(right)].set_index("state_id")
        joined = a.join(b, lsuffix="_left", rsuffix="_right", how="inner")
        fc0 = (joined["p_fc0_kw_left"] - joined["p_fc0_kw_right"]).abs()
        batt0 = (joined["p_batt0_kw_left"] - joined["p_batt0_kw_right"]).abs()
        soc1 = (joined["soc1_left"] - joined["soc1_right"]).abs()
        fc_rms = joined.apply(lambda row: float(np.sqrt(np.mean((np.asarray(json.loads(row["p_fc_trajectory_kw_left"])) - np.asarray(json.loads(row["p_fc_trajectory_kw_right"]))) ** 2))), axis=1)
        batt_rms = joined.apply(lambda row: float(np.sqrt(np.mean((np.asarray(json.loads(row["p_batt_trajectory_kw_left"])) - np.asarray(json.loads(row["p_batt_trajectory_kw_right"]))) ** 2))), axis=1)
        row = {
            "pair": f"A{left}-A{right}",
            "left_action_id": left,
            "right_action_id": right,
            "state_count": int(len(joined)),
            **{f"abs_delta_p_fc0_{key}_kw": value for key, value in _summary_stats(fc0).items()},
            **{f"abs_delta_p_batt0_{key}_kw": value for key, value in _summary_stats(batt0).items()},
            **{f"abs_delta_soc1_{key}": value for key, value in _summary_stats(soc1).items()},
            **{f"fc_trajectory_rms_{key}_kw": value for key, value in _summary_stats(fc_rms).items()},
            **{f"batt_trajectory_rms_{key}_kw": value for key, value in _summary_stats(batt_rms).items()},
            "share_abs_delta_p_fc0_gt_1kw": float((fc0 > 1.0).mean()),
            "share_abs_delta_p_fc0_gt_5kw": float((fc0 > 5.0).mean()),
            "share_abs_delta_p_fc0_gt_10kw": float((fc0 > 10.0).mean()),
            "potentially_redundant_pair": bool((fc0 < 1.0).mean() > 0.95 and (fc_rms < 1.0).mean() > 0.95),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _regime_summary(probes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    solved = probes.loc[probes["solved"]]
    for keys, group in solved.groupby(["soc_regime", "load_regime", "transition_regime", "action_id", "action_name"], dropna=False):
        soc_regime, load_regime, transition_regime, action_id, action_name = keys
        rows.append(
            {
                "soc_regime": soc_regime,
                "load_regime": load_regime,
                "transition_regime": transition_regime,
                "action_id": int(action_id),
                "action_name": action_name,
                "state_action_count": int(len(group)),
                "mean_p_fc0_kw": float(group["p_fc0_kw"].mean()),
                "mean_p_batt0_kw": float(group["p_batt0_kw"].mean()),
                "mean_soc1": float(group["soc1"].mean()),
                "mean_common_cost": float(group["common_cost"].mean()),
                "mean_soc_deadband_component": float(group["soc_deadband_objective_component"].mean()),
                "winner_count": int(group["winner_action_id"].eq(action_id).sum()),
            }
        )
    return pd.DataFrame(rows)


def _fixed_action_rollout(
    *,
    loads: np.ndarray,
    start_index: int,
    window_steps: int,
    action_id: int,
    config: Any,
    initial_soc: float,
) -> dict[str, Any]:
    bank = MpcWeightSolverBank(config)
    local_loads = loads[start_index : start_index + window_steps + 1]
    if local_loads.size != window_steps + 1:
        raise ValueError("fixed-action window must contain the requested steps plus one execution sample")
    soc, previous_fc, previous_batt = _initial_controller_state(
        local_loads,
        config,
        initial_soc=initial_soc,
    )
    h2_total = 0.0
    battery_throughput = 0.0
    fc_variation = 0.0
    peak_discharge = 0.0
    statuses: list[str] = []
    fc_values: list[float] = []
    min_soc = soc
    steps = int(window_steps)
    for index in range(steps):
        result, _ = bank.solve(
            action_id=action_id,
            load_forecast_kw=np.full(config.horizon, float(local_loads[index])),
            current_soc=soc,
            prev_fc_kw=previous_fc,
            soc_reference=FIXED_SOC_REFERENCE,
        )
        statuses.append(str(result.info.status))
        if not _status_is_solved(result):
            break
        next_fc, next_batt, next_soc = _step_physics(
            result=result,
            load_actual_kw=float(local_loads[index + 1]),
            soc_before=soc,
            config=config,
        )
        h2_total += float(h2_kg_step_dp0_quadratic(next_fc, dt_seconds=config.dt_seconds, p_rated_total_kw=config.fuel_cell_max_kw))
        battery_throughput += abs(next_batt) * float(config.dt_seconds) / 3600.0
        fc_variation += abs(next_fc - previous_fc)
        peak_discharge = max(peak_discharge, next_batt)
        fc_values.append(next_fc)
        min_soc = min(min_soc, next_soc)
        previous_fc, previous_batt, soc = next_fc, next_batt, next_soc
    return {
        "completed": bool(len(statuses) == steps and all(status.lower().startswith("solved") for status in statuses)),
        "steps": int(len(statuses)),
        "mpc_solves": int(len(statuses)),
        "hydrogen_kg": float(h2_total),
        "battery_throughput_kwh": float(battery_throughput),
        "fc_total_variation_kw": float(fc_variation),
        "min_soc": float(min_soc),
        "final_soc": float(soc),
        "mean_fc_kw": float(np.mean(fc_values)) if fc_values else float("nan"),
        "peak_battery_discharge_kw": float(peak_discharge),
        "last_solver_status": statuses[-1] if statuses else "not_run",
    }


def _select_window_start(loads: np.ndarray, *, window_steps: int, mode: str, target_mean: float | None = None) -> int:
    """Choose a real train-only 300 s window without altering its samples."""

    starts = np.arange(0, loads.size - window_steps, 30, dtype=int)
    if starts.size == 0:
        raise ValueError("segment is too short for fixed-action window")
    means = np.asarray([np.mean(loads[start : start + window_steps]) for start in starts])
    changes = np.asarray([loads[start + window_steps - 1] - loads[start] for start in starts])
    variability = np.asarray([np.mean(np.abs(np.diff(loads[start : start + window_steps]))) for start in starts])
    if mode == "low_load_stable":
        score = means + variability
        chosen = int(np.argmin(score))
    elif mode == "medium_load_stable":
        if target_mean is None:
            raise ValueError("medium-load window needs target_mean")
        chosen = int(np.argmin(np.abs(means - target_mean) + variability))
    elif mode == "high_load":
        chosen = int(np.argmax(means))
    elif mode == "rapid_load_rise":
        chosen = int(np.argmax(changes))
    elif mode == "rapid_load_fluctuation":
        chosen = int(np.argmax(variability))
    elif mode == "rapid_load_fall":
        chosen = int(np.argmin(changes))
    else:
        raise ValueError(f"unsupported fixed-action window mode: {mode}")
    return int(starts[chosen])


def _fixed_rollout_summary(segment_summary: pd.DataFrame, split: Any, config: Any, window_steps: int = 300) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    eligible = segment_summary.loc[segment_summary["points"] > window_steps].copy()
    if len(eligible) < 6:
        raise RuntimeError("fewer than six train segments support a 300-second fixed-action window")
    median_load = float(eligible["mean_load_kw"].median())
    stable = eligible.loc[
        eligible["mean_abs_delta_kw"]
        <= eligible["mean_abs_delta_kw"].quantile(0.40)
    ]
    windows = [
        ("low_load_stable", stable.loc[stable["mean_load_kw"].idxmin()]),
        ("medium_load_stable", stable.loc[(stable["mean_load_kw"] - median_load).abs().idxmin()]),
        ("high_load", eligible.loc[eligible["mean_load_kw"].idxmax()]),
        ("rapid_load_rise", eligible.loc[eligible["p95_delta_kw"].idxmax()]),
        ("rapid_load_fluctuation", eligible.loc[eligible["mean_abs_delta_kw"].idxmax()]),
        ("rapid_load_fall", eligible.loc[eligible["p05_delta_kw"].idxmin()]),
    ]
    # This seventh window intentionally shares a real load profile with the
    # medium-load window but starts from a low physically valid SOC.
    windows.append(("soc_low_counterfactual", windows[1][1]))
    for window_number, (window_regime, source) in enumerate(windows, start=1):
        segment_id = str(source["segment_id"])
        loads = load_operating_segment_loads("train", segment_id, split=split)
        initial_soc = 0.45 if window_regime == "soc_low_counterfactual" else FIXED_SOC_REFERENCE
        source_mode = "medium_load_stable" if window_regime == "soc_low_counterfactual" else window_regime
        start_index = _select_window_start(
            loads,
            window_steps=window_steps,
            mode=source_mode,
            target_mean=median_load,
        )
        local = loads[start_index : start_index + window_steps]
        deltas = np.diff(local)
        for action in DQN_MPC_WEIGHT_ACTIONS:
            record = _fixed_action_rollout(
                loads=loads,
                start_index=start_index,
                window_steps=window_steps,
                action_id=action.action_id,
                config=config,
                initial_soc=initial_soc,
            )
            rows.append(
                {
                    "window_id": f"window_{window_number:02d}",
                    "segment_id": segment_id,
                    "window_regime": window_regime,
                    "window_start_index": start_index,
                    "window_steps_requested": window_steps,
                    "initial_soc": initial_soc,
                    "window_mean_load_kw": float(np.mean(local)),
                    "window_max_load_kw": float(np.max(local)),
                    "window_mean_abs_delta_kw": float(np.mean(np.abs(deltas))) if deltas.size else 0.0,
                    "action_id": action.action_id,
                    "action_name": action.name,
                    **record,
                }
            )
    return pd.DataFrame(rows)


def _semantic_conclusions(probes: pd.DataFrame, pairwise: pd.DataFrame, rollouts: pd.DataFrame) -> tuple[dict[str, Any], str, list[str]]:
    solved = probes.loc[probes["solved"]]
    pivot = solved.pivot(index="state_id", columns="action_id", values=["p_fc0_kw", "p_batt0_kw", "soc1", "soc_deadband_objective_component"])
    normal = solved.loc[solved["soc_regime"].eq("inside_soft_range"), "state_id"].unique()
    low_soc = solved.loc[solved["soc_regime"].eq("below_soft_range"), "state_id"].unique()
    rapid = solved.loc[(solved["transition_regime"].eq("rapid_rise")) & (solved["current_load_kw"] - solved["previous_fc_kw"] > 48.0), "state_id"].unique()

    def _median_delta(metric: str, left: int, right: int, ids: np.ndarray) -> float | None:
        common = [index for index in ids if index in pivot.index]
        if not common:
            return None
        values = pivot.loc[common, (metric, left)] - pivot.loc[common, (metric, right)]
        return float(values.median())

    a1_fc = _median_delta("p_fc0_kw", 1, 0, normal)
    a1_batt = _median_delta("p_batt0_kw", 1, 0, normal)
    a2_fc = _median_delta("p_fc0_kw", 2, 0, low_soc)
    a2_batt = _median_delta("p_batt0_kw", 2, 0, low_soc)
    a2_soc = _median_delta("soc1", 2, 0, low_soc)
    a3_fc = _median_delta("p_fc0_kw", 3, 0, rapid)
    a3_batt = _median_delta("p_batt0_kw", 3, 0, rapid)
    inside = solved.loc[solved["soc_regime"].eq("inside_soft_range")]
    initial_deadband_max = float(max(soc_deadband_violation(value) for value in inside["soc"]))
    future_soc_component_max = float(inside["soc_deadband_objective_component"].max())
    semantics = {
        "a1_hydrogen_economy": {
            "normal_soc_state_count": int(len(normal)),
            "median_delta_p_fc0_vs_a0_kw": a1_fc,
            "median_delta_p_batt0_vs_a0_kw": a1_batt,
            "expected_direction_observed": bool(a1_fc is not None and a1_fc < 0.0 and a1_batt is not None and a1_batt > 0.0),
        },
        "a2_soc_recovery": {
            "low_soc_state_count": int(len(low_soc)),
            "median_delta_p_fc0_vs_a0_kw": a2_fc,
            "median_delta_p_batt0_vs_a0_kw": a2_batt,
            "median_delta_soc1_vs_a0": a2_soc,
            "expected_direction_observed": bool(a2_fc is not None and a2_fc > 0.0 and a2_batt is not None and a2_batt < 0.0 and a2_soc is not None and a2_soc > 0.0),
        },
        "a2_inside_deadband": {
            "inside_soc_state_count": int(len(inside) / len(ACTION_IDS)),
            "max_initial_state_deadband_violation": initial_deadband_max,
            "deadband_is_inactive_at_current_state": bool(initial_deadband_max <= 1.0e-12),
            "max_future_soc_deadband_component": future_soc_component_max,
            "note": "A nonzero future component only means the persistence-plan SOC exits the band; the formal objective has no 0.55 tracking term.",
        },
        "a3_fast_fc_response": {
            "rapid_rise_gap_state_count": int(len(rapid)),
            "median_delta_p_fc0_vs_a0_kw": a3_fc,
            "median_delta_p_batt0_vs_a0_kw": a3_batt,
            "expected_direction_observed": bool(a3_fc is not None and a3_fc > 0.0 and a3_batt is not None and a3_batt < 0.0),
        },
        "fixed_action_rollouts_differ": bool(rollouts.groupby("action_id")["hydrogen_kg"].mean().nunique() > 1),
    }
    redundant = pairwise.loc[pairwise["potentially_redundant_pair"], "pair"].tolist()
    winner_counts = solved.drop_duplicates("state_id")["winner_action_id"].value_counts()
    dominant_share = float(winner_counts.max() / max(len(winner_counts) and int(winner_counts.sum()), 1))
    reasons: list[str] = []
    if redundant:
        reasons.append(f"potentially redundant pairs: {', '.join(redundant)}")
    if not semantics["a1_hydrogen_economy"]["expected_direction_observed"]:
        reasons.append("A1 did not show the expected hydrogen-economy first-step direction in normal-SOC probes")
    if not semantics["a2_soc_recovery"]["expected_direction_observed"]:
        reasons.append("A2 did not show the expected low-SOC recovery first-step direction")
    if len(rapid) and not semantics["a3_fast_fc_response"]["expected_direction_observed"]:
        reasons.append("A3 did not show the expected rapid-rise FC response")
    near_tie_share = float(solved.drop_duplicates("state_id")["near_tie_le_1pct"].mean())
    if dominant_share > 0.90:
        reasons.append("common-reward winner share exceeds 90%; inspect near-tie rates before interpreting dominance")
    if near_tie_share > 0.90:
        reasons.append("more than 90% of common-reward comparisons are within 1%; winner concentration is not strong evidence of an absolute action advantage")
    verdict = "FAIL" if redundant or any("did not show" in reason for reason in reasons) else "WARNING" if reasons else "PASS"
    return semantics, verdict, reasons


def run_diagnostic(*, output_dir: Path = OUTPUT_DIR, anchors: int = 30, rollout_steps: int = 120) -> dict[str, Any]:
    """Run the frozen-train, no-learning action-space diversity diagnostic."""

    if anchors <= 0:
        raise ValueError("anchors must be positive")
    split = load_formal_operating_split()
    config = build_formal_mpc_config()
    segment_summary = _summarize_train_segments(split)
    selected_segments = _choose_segments(segment_summary)
    reachable = _collect_a0_reachable_states(segment_ids=selected_segments, split=split, config=config, steps_per_segment=rollout_steps)
    anchor_rows = _stratified_anchor_states(reachable, anchors=anchors)
    probes = _probe_states(anchors=anchor_rows, config=config)
    pairwise = _pairwise_differences(probes)
    regimes = _regime_summary(probes)
    rollouts = _fixed_rollout_summary(segment_summary, split, config)
    semantics, verdict, verdict_reasons = _semantic_conclusions(probes, pairwise, rollouts)

    states = probes.drop_duplicates("state_id")
    winner_counts = {f"A{action_id}": int((states["winner_action_id"] == action_id).sum()) for action_id in ACTION_IDS}
    solved_count = int(probes["solved"].sum())
    failed_count = int((~probes["solved"]).sum())
    dominant_share = float(max(winner_counts.values()) / max(len(states), 1))
    pairwise_records = pairwise.to_dict(orient="records")
    summary = {
        "scope": {"split": "train", "validation_or_test_accessed": False, "dqn_training": False, "agent_update": False, "forecast": "current-load persistence [current_load] * 6"},
        "representative_state_count": int(len(states)),
        "reachable_a0_state_count": int(len(reachable)),
        "selected_train_segments": selected_segments,
        "soc_stress_values": list(SOC_STRESS_VALUES),
        "total_mpc_solves": int(len(probes) + rollouts["mpc_solves"].sum()),
        "probe_mpc_solves": int(len(probes)),
        "probe_solved": solved_count,
        "probe_failed": failed_count,
        "winner_counts": winner_counts,
        "dominant_winner_share": dominant_share,
        "near_tie_le_1pct_share": float(states["near_tie_le_1pct"].mean()),
        "near_tie_le_5pct_share": float(states["near_tie_le_5pct"].mean()),
        "pairwise_action_differences": pairwise_records,
        "potentially_redundant_pairs": pairwise.loc[pairwise["potentially_redundant_pair"], "pair"].tolist(),
        "semantic_checks": semantics,
        "fixed_action_rollout_summary": {"window_count": int(rollouts["window_id"].nunique()), "all_completed": bool(rollouts["completed"].all())},
        "overall_verdict": verdict,
        "verdict_reasons": verdict_reasons,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    probes.to_csv(output_dir / REQUIRED_OUTPUT_FILENAMES["state_probe"], index=False)
    pairwise.to_csv(output_dir / REQUIRED_OUTPUT_FILENAMES["pairwise"], index=False)
    regimes.to_csv(output_dir / REQUIRED_OUTPUT_FILENAMES["regime"], index=False)
    rollouts.to_csv(output_dir / REQUIRED_OUTPUT_FILENAMES["rollout"], index=False)
    (output_dir / REQUIRED_OUTPUT_FILENAMES["summary"]).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--anchors", type=int, default=30)
    parser.add_argument("--reachable-steps-per-segment", type=int, default=120)
    args = parser.parse_args()
    summary = run_diagnostic(output_dir=args.output_dir, anchors=args.anchors, rollout_steps=args.reachable_steps_per_segment)
    print(json.dumps({key: summary[key] for key in ("representative_state_count", "probe_solved", "probe_failed", "overall_verdict")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
