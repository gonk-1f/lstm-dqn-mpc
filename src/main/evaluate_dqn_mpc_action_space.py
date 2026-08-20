from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from dqn.utils.action_mapper import (
    DQN_MPC_WEIGHT_ACTIONS,
    MPCWeightAction,
)
from dqn.utils.reward import (
    LOAD_DELTA_RISE_REFERENCE_KW,
    calculate_mpc_weight_reward,
)
from envs.dqn_mpc_weight_env import DqnMpcWeightEnv
from mpc_solvers.dqn_mpc_solver_bank import MpcWeightSolverBank
from mpc_solvers.mpc_qp_formulation import (
    QpMpcConfig,
    resolved_ramp_kw_per_step,
)
from run_mpc_1s_n6_four_objective_sensitivity import (
    N6_STATE_COMMIT_TOLERANCES,
    physical_h2_kg_step,
)
from train_dqn_mpc_mlp import (
    DEFAULT_SPLIT_JSON,
    DEFAULT_VOYAGE_DATA_DIR,
    VoyageSplit,
    build_formal_mpc_config,
    load_voyage_loads,
    load_voyage_split,
)


LOCKED_TEST_VOYAGES = frozenset(
    f"voyage_{index:03d}" for index in range(60, 67)
)
ALLOWED_EVALUATION_SPLITS = ("train", "validation")


@dataclass(frozen=True)
class RepresentativeState:
    state_id: str
    source_split: str
    source_voyage_id: str
    decision_index: int
    current_soc: float
    previous_fc_kw: float
    previous_batt_kw: float
    current_load_kw: float
    previous_load_kw: float
    future_load_kw: tuple[float, ...]
    load_regime: str
    load_delta_regime: str
    soc_regime: str
    previous_fc_regime: str


def classify_solver_status(status: str) -> str:
    normalized = str(status).strip().lower()
    if "solved inaccurate" in normalized:
        return "solved_inaccurate"
    if normalized.startswith("solved"):
        return "solved"
    if "primal infeasible" in normalized:
        return "primal_infeasible"
    if (
        "maximum iterations" in normalized
        or "max_iter" in normalized
    ):
        return "maximum_iterations"
    return "other_failure"


def _validate_voyage_loads(
    voyage_loads: Mapping[
        tuple[str, str],
        Sequence[float] | np.ndarray,
    ],
) -> tuple[tuple[str, str, np.ndarray], ...]:
    if not voyage_loads:
        raise ValueError("action-space evaluation requires voyage data")

    rows: list[tuple[str, str, np.ndarray]] = []
    for (raw_split, raw_voyage), raw_loads in sorted(
        voyage_loads.items()
    ):
        split_name = str(raw_split).strip().lower()
        voyage_id = str(raw_voyage)
        if split_name not in ALLOWED_EVALUATION_SPLITS:
            raise ValueError(
                "action-space evaluation only permits train and "
                f"validation data, not {split_name!r}"
            )
        if voyage_id in LOCKED_TEST_VOYAGES:
            raise ValueError(
                f"test voyage access is prohibited: {voyage_id}"
            )

        loads = np.asarray(raw_loads, dtype=np.float64).reshape(-1)
        if loads.size < 3 or not np.all(np.isfinite(loads)):
            raise ValueError(
                f"{voyage_id} must contain at least three finite loads"
            )
        rows.append((split_name, voyage_id, loads))

    return tuple(rows)


def _future_window(
    loads_kw: np.ndarray,
    decision_index: int,
    horizon: int,
) -> tuple[float, ...]:
    preview = loads_kw[
        decision_index + 1 : decision_index + 1 + horizon
    ]
    if preview.size < horizon:
        preview = np.pad(
            preview,
            (0, horizon - preview.size),
            mode="edge",
        )
    return tuple(float(value) for value in preview)


def build_representative_states(
    voyage_loads: Mapping[
        tuple[str, str],
        Sequence[float] | np.ndarray,
    ],
    *,
    base_config: QpMpcConfig,
) -> tuple[RepresentativeState, ...]:
    """Select nine real load windows, then apply physical SOC/FC strata."""

    voyages = _validate_voyage_loads(voyage_loads)
    horizon = int(base_config.horizon)
    candidates: list[
        tuple[str, str, int, float, float, tuple[float, ...]]
    ] = []

    for split_name, voyage_id, loads in voyages:
        for decision_index in range(1, loads.size - 1):
            current_load = float(loads[decision_index])
            previous_load = float(loads[decision_index - 1])
            candidates.append(
                (
                    split_name,
                    voyage_id,
                    decision_index,
                    current_load,
                    current_load - previous_load,
                    tuple(current_load for _ in range(horizon)),
                )
            )

    if not candidates:
        raise ValueError("no representative-state candidates exist")

    loads = np.asarray(
        [candidate[3] for candidate in candidates],
        dtype=float,
    )
    deltas = np.asarray(
        [candidate[4] for candidate in candidates],
        dtype=float,
    )
    load_targets = np.quantile(loads, [0.2, 0.5, 0.8])
    delta_targets = np.quantile(deltas, [0.1, 0.5, 0.9])
    load_scale = max(float(np.ptp(loads)), 1.0)
    delta_scale = max(float(np.ptp(deltas)), 1.0)

    load_regimes = ("low", "medium", "high")
    delta_regimes = (
        "rapidly_falling",
        "near_steady",
        "rapidly_rising",
    )
    required_windows = len(load_regimes) * len(delta_regimes)
    if len(candidates) < required_windows:
        raise ValueError(
            "at least nine distinct load windows are required "
            "for representative-state construction"
        )
    selected: list[
        tuple[
            str,
            str,
            tuple[str, str, int, float, float, tuple[float, ...]],
        ]
    ] = []
    selected_window_keys: set[tuple[str, str, int]] = set()

    for load_index, load_regime in enumerate(load_regimes):
        for delta_index, delta_regime in enumerate(delta_regimes):
            load_target = float(load_targets[load_index])
            delta_target = (
                LOAD_DELTA_RISE_REFERENCE_KW
                if delta_regime == "rapidly_rising"
                else float(delta_targets[delta_index])
            )
            candidate = min(
                (
                    item
                    for item in candidates
                    if (item[0], item[1], item[2])
                    not in selected_window_keys
                    and (
                        delta_regime != "rapidly_rising"
                        or (
                            item[0] == "train"
                            and item[4]
                            >= LOAD_DELTA_RISE_REFERENCE_KW
                        )
                    )
                ),
                key=lambda item: (
                    abs(item[3] - load_target) / load_scale
                    + abs(item[4] - delta_target) / delta_scale,
                    item[0],
                    item[1],
                    item[2],
                ),
            )
            selected_window_keys.add(
                (candidate[0], candidate[1], candidate[2])
            )
            selected.append(
                (load_regime, delta_regime, candidate)
            )

    soc_values = (
        (
            "low",
            float(base_config.soc_min + base_config.soc_band),
        ),
        ("reference", 0.55),
        (
            "high",
            float(base_config.soc_max - base_config.soc_band),
        ),
    )
    previous_fc_values = (
        ("low", float(base_config.fuel_cell_min_kw)),
        (
            "medium",
            float(
                (
                    base_config.fuel_cell_min_kw
                    + base_config.fuel_cell_max_kw
                )
                / 2.0
            ),
        ),
        ("high", float(base_config.fuel_cell_max_kw)),
    )

    states: list[RepresentativeState] = []
    for load_regime, delta_regime, candidate in selected:
        (
            split_name,
            voyage_id,
            decision_index,
            current_load,
            _,
            future_load,
        ) = candidate
        previous_load = float(
            voyage_loads[(split_name, voyage_id)][decision_index - 1]
        )
        for soc_regime, current_soc in soc_values:
            for previous_fc_regime, previous_fc in previous_fc_values:
                states.append(
                    RepresentativeState(
                        state_id=f"probe_{len(states) + 1:03d}",
                        source_split=split_name,
                        source_voyage_id=voyage_id,
                        decision_index=int(decision_index),
                        current_soc=float(current_soc),
                        previous_fc_kw=float(previous_fc),
                        previous_batt_kw=float(
                            current_load - previous_fc
                        ),
                        current_load_kw=float(current_load),
                        previous_load_kw=previous_load,
                        future_load_kw=future_load,
                        load_regime=load_regime,
                        load_delta_regime=delta_regime,
                        soc_regime=soc_regime,
                        previous_fc_regime=previous_fc_regime,
                    )
                )

    return tuple(states)


def _solver_info_value(
    info: Any,
    name: str,
) -> float | int | None:
    value = getattr(info, name, None)
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def evaluate_state_probes(
    *,
    states: Sequence[RepresentativeState],
    actions: Sequence[MPCWeightAction],
    base_config: QpMpcConfig,
) -> list[dict[str, Any]]:
    resolved_actions = tuple(actions)
    rows: list[dict[str, Any]] = []
    horizon = int(base_config.horizon)

    for state in states:
        bank = MpcWeightSolverBank(
            base_config,
            actions=resolved_actions,
        )
        for action in resolved_actions:
            causal_forecast_kw = tuple(
                float(state.current_load_kw)
                for _ in range(horizon)
            )
            result, solve_ms = bank.solve(
                action_id=action.action_id,
                load_forecast_kw=causal_forecast_kw,
                current_soc=state.current_soc,
                prev_fc_kw=state.previous_fc_kw,
                soc_reference=0.55,
            )
            status = str(result.info.status)
            status_class = classify_solver_status(status)
            expected_solution_size = 3 * horizon + 1
            solution: np.ndarray | None = None
            solution_validation_error: str | None = None
            if result.x is None:
                solution_validation_error = "solution vector is missing"
            else:
                candidate_solution = np.asarray(
                    result.x,
                    dtype=float,
                ).reshape(-1)
                if candidate_solution.size != expected_solution_size:
                    solution_validation_error = (
                        "solution vector has size "
                        f"{candidate_solution.size}, expected "
                        f"{expected_solution_size}"
                    )
                elif not np.all(np.isfinite(candidate_solution)):
                    solution_validation_error = (
                        "solution vector contains non-finite values"
                    )
                else:
                    solution = candidate_solution

            if (
                status_class in {"solved", "solved_inaccurate"}
                and solution_validation_error is not None
            ):
                status_class = "other_failure"
            row: dict[str, Any] = {
                **asdict(state),
                "action_id": action.action_id,
                "action_name": action.name,
                "q_h2": action.q_h2,
                "q_batt": action.q_batt,
                "q_soc": action.q_soc,
                "q_fc_var": action.q_fc_var,
                "solver_status": status,
                "solver_status_class": status_class,
                "solve_ms": float(solve_ms),
                "solver_iterations": _solver_info_value(
                    result.info, "iter"
                ),
                "primal_residual": _solver_info_value(
                    result.info, "prim_res"
                ),
                "dual_residual": _solver_info_value(
                    result.info, "dual_res"
                ),
                "solution_validation_error": (
                    solution_validation_error
                ),
            }

            if (
                status_class not in {"solved", "solved_inaccurate"}
                or solution is None
            ):
                row.update(
                    {
                        "p_fc_first_kw": None,
                        "p_batt_first_kw": None,
                        "delta_p_fc_kw": None,
                        "soc_predicted": None,
                        "soc_next": None,
                        "h2_kg": None,
                        "reward": None,
                        "weighted_h2": None,
                        "weighted_batt": None,
                        "weighted_soc": None,
                        "weighted_fc_var": None,
                        "p_fc_horizon_kw": [],
                        "p_batt_horizon_kw": [],
                        "soc_horizon": [],
                    }
                )
                rows.append(row)
                continue

            p_fc_horizon = solution[:horizon]
            p_batt_horizon = solution[horizon : 2 * horizon]
            soc_horizon = solution[2 * horizon :]
            p_fc_first = float(p_fc_horizon[0])
            p_batt_first = float(
                state.current_load_kw - p_fc_first
            )
            soc_next = float(
                state.current_soc
                - p_batt_first
                * float(base_config.dt_seconds)
                / 3600.0
                / float(base_config.battery_capacity_kwh)
            )
            reward, reward_terms = calculate_mpc_weight_reward(
                p_fc_kw=p_fc_first,
                p_batt_kw=p_batt_first,
                next_soc=soc_next,
                previous_fc_kw=state.previous_fc_kw,
                soc_before=state.current_soc,
                load_delta_kw=(
                    float(state.current_load_kw)
                    - float(state.previous_load_kw)
                ),
            )
            row.update(
                {
                    "p_fc_first_kw": p_fc_first,
                    "p_batt_first_kw": p_batt_first,
                    "delta_p_fc_kw": float(
                        p_fc_first - state.previous_fc_kw
                    ),
                    "soc_predicted": float(soc_horizon[1]),
                    "soc_next": soc_next,
                    "h2_kg": physical_h2_kg_step(
                        base_config, p_fc_first
                    ),
                    "reward": float(reward),
                    "weighted_h2": reward_terms["weighted_h2"],
                    "weighted_batt": reward_terms[
                        "weighted_batt"
                    ],
                    "weighted_soc": reward_terms["weighted_soc"],
                    "weighted_fc_var": reward_terms[
                        "weighted_fc_var"
                    ],
                    "p_fc_horizon_kw": [
                        float(value) for value in p_fc_horizon
                    ],
                    "p_batt_horizon_kw": [
                        float(value) for value in p_batt_horizon
                    ],
                    "soc_horizon": [
                        float(value) for value in soc_horizon
                    ],
                }
            )
            rows.append(row)

    return rows


def _exception_status(error: RuntimeError) -> str:
    status = getattr(error, "solver_status", None)
    if status is not None:
        return str(status)
    message = str(error)
    marker = "status="
    return (
        message.split(marker, 1)[1].strip()
        if marker in message
        else message
    )


def evaluate_fixed_action_coverage(
    *,
    voyage_loads: Mapping[
        tuple[str, str],
        Sequence[float] | np.ndarray,
    ],
    actions: Sequence[MPCWeightAction],
    base_config: QpMpcConfig,
) -> list[dict[str, Any]]:
    voyages = _validate_voyage_loads(voyage_loads)
    rows: list[dict[str, Any]] = []

    for split_name, voyage_id, loads in voyages:
        for action in actions:
            env = DqnMpcWeightEnv(
                loads_kw=loads,
                base_config=base_config,
                initial_soc=0.55,
                actions=(action,),
            )
            env.reset()
            successful_steps = 0
            rewards: list[float] = []
            solve_times: list[float] = []
            p_fc_values: list[float] = []
            p_batt_values: list[float] = []
            soc_values: list[float] = []
            delta_fc_values: list[float] = []
            h2_total = 0.0
            reward_component_totals = {
                "weighted_h2": 0.0,
                "weighted_batt": 0.0,
                "weighted_soc": 0.0,
                "weighted_fc_var": 0.0,
            }
            solved_inaccurate_steps = 0
            failure: RuntimeError | None = None
            done = False

            while not done:
                try:
                    _, reward, done, info = env.step(
                        action.action_id
                    )
                except RuntimeError as error:
                    failure = error
                    break

                successful_steps += 1
                rewards.append(float(reward))
                solve_times.append(float(info["solve_ms"]))
                p_fc = float(info["p_fc_kw"])
                p_batt = float(info["p_batt_kw"])
                p_fc_values.append(p_fc)
                p_batt_values.append(p_batt)
                soc_values.append(float(info["soc_after"]))
                delta_fc_values.append(
                    p_fc - float(info["p_fc_prev_kw"])
                )
                h2_total += physical_h2_kg_step(
                    base_config, p_fc
                )
                if classify_solver_status(
                    str(info["solver_status"])
                ) == "solved_inaccurate":
                    solved_inaccurate_steps += 1
                terms = info["reward_terms"]
                for key in reward_component_totals:
                    reward_component_totals[key] += float(
                        terms[key]
                    )

            completed = failure is None and bool(done)
            if failure is None:
                status = (
                    "solved inaccurate"
                    if solved_inaccurate_steps
                    else "solved"
                )
                failure_decision_index = None
                failure_solve_ms = None
            else:
                status = _exception_status(failure)
                failure_decision_index = int(
                    getattr(
                        failure,
                        "decision_index",
                        env.decision_index,
                    )
                )
                failure_solve_ms = float(
                    getattr(failure, "solve_ms", np.nan)
                )
            status_class = classify_solver_status(status)
            if (
                failure is not None
                and status_class in {"solved", "solved_inaccurate"}
            ):
                status = (
                    "invalid result after solver status "
                    f"{status}"
                )
                status_class = "other_failure"

            row: dict[str, Any] = {
                "split": split_name,
                "voyage_id": voyage_id,
                "action_id": action.action_id,
                "action_name": action.name,
                "q_h2": action.q_h2,
                "q_batt": action.q_batt,
                "q_soc": action.q_soc,
                "q_fc_var": action.q_fc_var,
                "completed": completed,
                "successful_steps": successful_steps,
                "expected_steps": int(loads.size - 1),
                "solver_status": status,
                "solver_status_class": status_class,
                "solved_inaccurate_steps": solved_inaccurate_steps,
                "first_failure_decision_index": (
                    failure_decision_index
                ),
                "first_failure_solve_ms": failure_solve_ms,
                "mean_solve_ms": (
                    float(np.mean(solve_times))
                    if solve_times
                    else None
                ),
                "p99_solve_ms": (
                    float(np.quantile(solve_times, 0.99))
                    if solve_times
                    else None
                ),
                "episode_reward": float(np.sum(rewards)),
                "mean_reward_per_step": (
                    float(np.mean(rewards)) if rewards else None
                ),
                "total_h2_kg": float(h2_total),
                "final_soc": (
                    float(env.current_soc)
                    if successful_steps
                    else 0.55
                ),
                "minimum_soc": (
                    float(np.min(soc_values))
                    if soc_values
                    else 0.55
                ),
                "battery_charge_energy_kwh": float(
                    np.sum(
                        np.maximum(
                            -np.asarray(p_batt_values), 0.0
                        )
                    )
                    / 3600.0
                ),
                "battery_discharge_energy_kwh": float(
                    np.sum(
                        np.maximum(
                            np.asarray(p_batt_values), 0.0
                        )
                    )
                    / 3600.0
                ),
                "battery_throughput_kwh": float(
                    np.sum(np.abs(p_batt_values)) / 3600.0
                ),
                "fc_variation_rms_kw": (
                    float(
                        np.sqrt(
                            np.mean(
                                np.square(delta_fc_values)
                            )
                        )
                    )
                    if delta_fc_values
                    else None
                ),
                "fc_max_step_kw": (
                    float(np.max(np.abs(delta_fc_values)))
                    if delta_fc_values
                    else None
                ),
                **reward_component_totals,
            }
            rows.append(row)

    return rows


def _split_voyage_ids(
    split: VoyageSplit,
    split_name: str,
) -> tuple[str, ...]:
    return (
        split.train_voyages
        if split_name == "train"
        else split.validation_voyages
    )


def evaluate_action_space(
    *,
    actions: Sequence[MPCWeightAction] = DQN_MPC_WEIGHT_ACTIONS,
    split_names: Sequence[str] = ALLOWED_EVALUATION_SPLITS,
    split_path: str | Path = DEFAULT_SPLIT_JSON,
    data_dir: str | Path = DEFAULT_VOYAGE_DATA_DIR,
) -> dict[str, Any]:
    resolved_splits = tuple(
        str(value).strip().lower() for value in split_names
    )
    if not resolved_splits:
        raise ValueError("at least one split is required")
    if (
        len(resolved_splits) != len(set(resolved_splits))
        or any(
            split_name not in ALLOWED_EVALUATION_SPLITS
            for split_name in resolved_splits
        )
    ):
        raise ValueError(
            "action-space evaluation permits train and validation "
            "only; test trajectory access is prohibited"
        )

    split = load_voyage_split(split_path)
    voyage_loads: dict[tuple[str, str], np.ndarray] = {}
    for split_name in resolved_splits:
        for voyage_id in _split_voyage_ids(split, split_name):
            if (
                voyage_id in LOCKED_TEST_VOYAGES
                or voyage_id in split.test_voyages
            ):
                raise ValueError(
                    f"test voyage access is prohibited: {voyage_id}"
                )
            voyage_loads[(split_name, voyage_id)] = (
                load_voyage_loads(
                    split_name,
                    voyage_id,
                    split=split,
                    data_dir=data_dir,
                )
            )

    base_config = build_formal_mpc_config()
    states = build_representative_states(
        voyage_loads,
        base_config=base_config,
    )
    probes = evaluate_state_probes(
        states=states,
        actions=actions,
        base_config=base_config,
    )
    coverage = evaluate_fixed_action_coverage(
        voyage_loads=voyage_loads,
        actions=actions,
        base_config=base_config,
    )
    return {
        "actions": [
            {
                "action_id": action.action_id,
                "name": action.name,
                "weights": list(action.as_tuple()),
            }
            for action in actions
        ],
        "representative_states": [
            asdict(state) for state in states
        ],
        "probe_rows": probes,
        "coverage_rows": coverage,
        "data_access": {
            "loaded_train_voyages": (
                list(split.train_voyages)
                if "train" in resolved_splits
                else []
            ),
            "loaded_validation_voyages": (
                list(split.validation_voyages)
                if "validation" in resolved_splits
                else []
            ),
            "test_voyages_locked": list(split.test_voyages),
            "fixed_test_voyage_ids": sorted(LOCKED_TEST_VOYAGES),
            "test_trajectories_accessed": [],
        },
        "physical_tolerances": dict(
            N6_STATE_COMMIT_TOLERANCES
        ),
        "ramp_limit_kw": resolved_ramp_kw_per_step(
            base_config
        ),
    }


def _finite_mean(values: Sequence[Any]) -> float | None:
    finite = np.asarray(
        [
            float(value)
            for value in values
            if value is not None and np.isfinite(float(value))
        ],
        dtype=float,
    )
    return float(np.mean(finite)) if finite.size else None


def _pairwise_probe_metrics(
    probe_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_action: dict[int, dict[str, Mapping[str, Any]]] = (
        defaultdict(dict)
    )
    action_names: dict[int, str] = {}
    for row in probe_rows:
        if row.get("p_fc_first_kw") is None:
            continue
        action_id = int(row["action_id"])
        by_action[action_id][str(row["state_id"])] = row
        action_names[action_id] = str(row["action_name"])

    metrics: list[dict[str, Any]] = []
    action_ids = sorted(by_action)
    for index, left_id in enumerate(action_ids):
        for right_id in action_ids[index + 1 :]:
            common = sorted(
                set(by_action[left_id])
                & set(by_action[right_id])
            )
            if not common:
                continue
            first_differences: list[float] = []
            horizon_rms: list[float] = []
            soc_endpoint_differences: list[float] = []
            for state_id in common:
                left = by_action[left_id][state_id]
                right = by_action[right_id][state_id]
                first_differences.append(
                    abs(
                        float(left["p_fc_first_kw"])
                        - float(right["p_fc_first_kw"])
                    )
                )
                left_horizon = np.asarray(
                    left["p_fc_horizon_kw"], dtype=float
                )
                right_horizon = np.asarray(
                    right["p_fc_horizon_kw"], dtype=float
                )
                horizon_rms.append(
                    float(
                        np.sqrt(
                            np.mean(
                                np.square(
                                    left_horizon - right_horizon
                                )
                            )
                        )
                    )
                )
                soc_endpoint_differences.append(
                    abs(
                        float(left["soc_horizon"][-1])
                        - float(right["soc_horizon"][-1])
                    )
                )
            metrics.append(
                {
                    "left_action_id": left_id,
                    "left_action_name": action_names[left_id],
                    "right_action_id": right_id,
                    "right_action_name": action_names[right_id],
                    "common_states": len(common),
                    "mean_abs_first_fc_difference_kw": float(
                        np.mean(first_differences)
                    ),
                    "max_abs_first_fc_difference_kw": float(
                        np.max(first_differences)
                    ),
                    "mean_horizon_fc_rms_difference_kw": float(
                        np.mean(horizon_rms)
                    ),
                    "mean_abs_horizon_end_soc_difference": float(
                        np.mean(soc_endpoint_differences)
                    ),
                }
            )
    return metrics


def summarize_action_space(
    result: Mapping[str, Any],
) -> dict[str, Any]:
    probe_rows = list(result.get("probe_rows", []))
    coverage_rows = list(result.get("coverage_rows", []))
    probe_status_counts = dict(
        Counter(
            str(row["solver_status_class"])
            for row in probe_rows
        )
    )
    coverage_status_counts = dict(
        Counter(
            str(row["solver_status_class"])
            for row in coverage_rows
        )
    )

    probe_winners: Counter[int] = Counter()
    probe_regime_winners: dict[
        str, dict[str, Counter[int]]
    ] = {
        "load": defaultdict(Counter),
        "load_delta": defaultdict(Counter),
        "soc": defaultdict(Counter),
        "previous_fc": defaultdict(Counter),
    }
    by_state: dict[str, list[Mapping[str, Any]]] = defaultdict(
        list
    )
    for row in probe_rows:
        if row.get("reward") is not None:
            by_state[str(row["state_id"])].append(row)
    for rows in by_state.values():
        winner = max(rows, key=lambda row: float(row["reward"]))
        action_id = int(winner["action_id"])
        probe_winners[action_id] += 1
        probe_regime_winners["load"][
            str(winner["load_regime"])
        ][action_id] += 1
        probe_regime_winners["load_delta"][
            str(winner["load_delta_regime"])
        ][action_id] += 1
        probe_regime_winners["soc"][
            str(winner["soc_regime"])
        ][action_id] += 1
        probe_regime_winners["previous_fc"][
            str(winner["previous_fc_regime"])
        ][action_id] += 1

    per_action: dict[str, Any] = {}
    for action in result.get("actions", []):
        action_id = int(action["action_id"])
        rows = [
            row
            for row in coverage_rows
            if int(row["action_id"]) == action_id
        ]
        status_counts = Counter(
            str(row["solver_status_class"]) for row in rows
        )
        success = sum(bool(row["completed"]) for row in rows)
        per_action[f"A{action_id}"] = {
            "name": str(action["name"]),
            "weights": list(action["weights"]),
            "total": len(rows),
            "success": int(success),
            "primal_infeasible": int(
                status_counts["primal_infeasible"]
            ),
            "maximum_iterations": int(
                status_counts["maximum_iterations"]
            ),
            "solved_inaccurate": int(
                status_counts["solved_inaccurate"]
            ),
            "other_failure": int(
                status_counts["other_failure"]
            ),
            "mean_solve_ms": _finite_mean(
                [row.get("mean_solve_ms") for row in rows]
            ),
            "mean_reward_per_step": _finite_mean(
                [
                    row.get("mean_reward_per_step")
                    for row in rows
                    if row.get("completed")
                ]
            ),
        }

    all_action_failure_voyages: list[str] = []
    coverage_by_voyage: dict[
        str, list[Mapping[str, Any]]
    ] = defaultdict(list)
    for row in coverage_rows:
        coverage_by_voyage[str(row["voyage_id"])].append(row)
    for voyage_id, rows in sorted(coverage_by_voyage.items()):
        if rows and not any(bool(row["completed"]) for row in rows):
            all_action_failure_voyages.append(voyage_id)

    pairwise = _pairwise_probe_metrics(probe_rows)
    redundant_pairs = [
        f"A{metric['left_action_id']}-A{metric['right_action_id']}"
        for metric in pairwise
        if (
            metric["max_abs_first_fc_difference_kw"] <= 1.0
            and metric["mean_horizon_fc_rms_difference_kw"] <= 1.0
        )
    ]
    probe_failure_count = sum(
        count
        for status, count in probe_status_counts.items()
        if status not in {"solved", "solved_inaccurate"}
    )
    maximum_iterations = int(
        coverage_status_counts.get("maximum_iterations", 0)
    )
    winner_total = sum(probe_winners.values())
    dominant_winner_fraction = (
        max(probe_winners.values()) / winner_total
        if winner_total and probe_winners
        else None
    )
    baseline_success = per_action.get("A0", {}).get("success")
    lower_coverage_actions = (
        [
            action_id
            for action_id, action in per_action.items()
            if (
                baseline_success is not None
                and action["success"] < baseline_success
            )
        ]
        if coverage_rows
        else []
    )
    acceptance_reasons: list[str] = []
    if probe_failure_count:
        acceptance_reasons.append(
            f"{probe_failure_count} representative-state solves failed"
        )
    if redundant_pairs:
        acceptance_reasons.append(
            "behaviorally redundant pairs: "
            + ", ".join(redundant_pairs)
        )
    if maximum_iterations:
        acceptance_reasons.append(
            f"{maximum_iterations} maximum-iterations coverage failures"
        )
    if (
        dominant_winner_fraction is not None
        and dominant_winner_fraction >= 0.9
    ):
        acceptance_reasons.append(
            "one action wins at least 90% of representative states"
        )
    if lower_coverage_actions:
        acceptance_reasons.append(
            "actions below Candidate C coverage: "
            + ", ".join(lower_coverage_actions)
        )
    if all_action_failure_voyages:
        acceptance_reasons.append(
            "all-action-failure voyages: "
            + ", ".join(all_action_failure_voyages)
        )

    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "actions": list(result.get("actions", [])),
        "data_access": dict(result.get("data_access", {})),
        "probe": {
            "representative_state_count": len(by_state),
            "solve_count": len(probe_rows),
            "status_counts": probe_status_counts,
            "reward_winner_counts": {
                f"A{action_id}": int(count)
                for action_id, count in sorted(
                    probe_winners.items()
                )
            },
            "conditional_reward_winner_counts": {
                axis: {
                    regime: {
                        f"A{action_id}": int(count)
                        for action_id, count in sorted(
                            counts.items()
                        )
                    }
                    for regime, counts in sorted(regimes.items())
                }
                for axis, regimes in probe_regime_winners.items()
            },
            "dominant_winner_fraction": (
                dominant_winner_fraction
            ),
            "pairwise": pairwise,
            "redundant_pairs_at_1kw": redundant_pairs,
        },
        "coverage": {
            "voyage_action_pairs": len(coverage_rows),
            "status_counts": coverage_status_counts,
            "per_action": per_action,
            "all_action_failure_voyages": (
                all_action_failure_voyages
            ),
        },
        "acceptance": {
            "decision": (
                "PASS" if not acceptance_reasons else "FAIL"
            ),
            "reasons": acceptance_reasons,
            "rules": {
                "probe_failures_allowed": 0,
                "redundant_pair_rule": (
                    "max first-FC difference <=1 kW and mean "
                    "horizon FC RMS difference <=1 kW"
                ),
                "maximum_iterations_allowed": 0,
                "dominant_probe_winner_fraction_limit": 0.9,
                "minimum_per_action_coverage": (
                    "Candidate C completed-voyage count"
                ),
                "all_action_failure_voyages_allowed": 0,
            },
        },
    }


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    return value


def _write_rows(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    field_names = sorted(
        {
            str(key)
            for row in rows
            for key in row
        }
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=field_names,
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: _csv_value(row.get(key))
                    for key in field_names
                }
            )


def write_action_space_evaluation(
    result: Mapping[str, Any],
    *,
    output_dir: str | Path,
    prefix: str = "final",
) -> dict[str, Path]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    safe_prefix = str(prefix).strip()
    if not safe_prefix or any(
        character in safe_prefix for character in "\\/:"
    ):
        raise ValueError("output prefix must be a simple name")

    paths = {
        "probes": directory / f"{safe_prefix}_state_probes.csv",
        "coverage": directory / f"{safe_prefix}_coverage.csv",
        "summary": directory / f"{safe_prefix}_summary.json",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "action-space output already exists: "
            + ", ".join(str(path) for path in existing)
        )

    _write_rows(paths["probes"], list(result.get("probe_rows", [])))
    _write_rows(
        paths["coverage"],
        list(result.get("coverage_rows", [])),
    )
    paths["summary"].write_text(
        json.dumps(
            summarize_action_space(result),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return paths


def load_action_specification(
    path: str | Path,
) -> tuple[MPCWeightAction, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("actions") if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows:
        raise ValueError("action specification must contain a non-empty list")

    actions: list[MPCWeightAction] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("each action specification must be an object")
        if "weights" in row:
            weights = row["weights"]
            if not isinstance(weights, list) or len(weights) != 4:
                raise ValueError(
                    "action weights must use "
                    "[q_h2, q_batt, q_soc, q_fc_var]"
                )
        else:
            weights = [
                row["q_h2"],
                row["q_batt"],
                row["q_soc"],
                row["q_fc_var"],
            ]
        actions.append(
            MPCWeightAction(
                action_id=int(row["action_id"]),
                q_h2=float(weights[0]),
                q_batt=float(weights[1]),
                q_soc=float(weights[2]),
                q_fc_var=float(weights[3]),
                name=str(row["name"]),
            )
        )
    return tuple(actions)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate DQN-MPC weight actions on locked "
            "train/validation data only."
        )
    )
    parser.add_argument(
        "--actions-json",
        type=Path,
        default=None,
        help="optional explicit candidate action table",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--prefix",
        default="final",
    )
    args = parser.parse_args()

    actions = (
        load_action_specification(args.actions_json)
        if args.actions_json is not None
        else DQN_MPC_WEIGHT_ACTIONS
    )
    result = evaluate_action_space(actions=actions)
    paths = write_action_space_evaluation(
        result,
        output_dir=args.output_dir,
        prefix=args.prefix,
    )
    summary = summarize_action_space(result)
    print(
        json.dumps(
            {
                "decision": summary["acceptance"]["decision"],
                "paths": {
                    key: str(value)
                    for key, value in paths.items()
                },
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
