"""Train-only scale calibration and preliminary 16-action MPC response audit."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Sequence

import numpy as np

from v2.control.causal_base_load import CausalBaseLoadFilter
from v2.data.formal_training_dataset import FormalTrainingDataset
from v2.data.supervisory_rules import normalize_onboard_load_kw
from v2.main.train_formal_dqn import (
    DEFAULT_AIS_ROOT, DEFAULT_MODE_ROOT, DEFAULT_POWER_ROOT, REPOSITORY_ROOT,
)

from .control import AccountState, DT_SECONDS, EconomicMPC, HORIZON, MPCWeights


DEFAULT_OUTPUT = REPOSITORY_ROOT / "outputs" / "v3_persistence_calibration" / "action_calibration.json"
SOC_SCENARIOS = (0.4, 0.5, 0.6)
AXIS_MULTIPLIERS = (0.0, 0.25, 1.0, 4.0)


@dataclass(frozen=True)
class CalibrationOrigin:
    sample_id: str
    step_index: int
    load_kw: float
    base_kw: float
    previous_base_kw: float
    load_change_kw: float

    @property
    def case_id(self) -> str:
        return f"{self.sample_id}:{self.step_index}"


def four_level_axis(base: float, *, positive_multiplier: float = 1.0) -> tuple[float, ...]:
    if not math.isfinite(base) or base <= 0.0:
        raise ValueError("action-axis base must be finite and positive")
    if not math.isfinite(positive_multiplier) or positive_multiplier <= 0.0:
        raise ValueError("positive action-axis multiplier must be finite and positive")
    return tuple(float(base * factor * positive_multiplier) for factor in AXIS_MULTIPLIERS)


def cartesian_actions(
    ref_axis: Sequence[float], soc_axis: Sequence[float],
) -> tuple[MPCWeights, ...]:
    actions = tuple(MPCWeights(float(ref), float(soc)) for ref in ref_axis for soc in soc_axis)
    if len(actions) != len(ref_axis) * len(soc_axis) or len(set(actions)) != len(actions):
        raise ValueError("action axes must form distinct finite weight pairs")
    return actions


def calibrate_weight_bases(
    economic_costs_cny: Sequence[float], nominal_cost_cny: float,
    material_reference_penalty: float, material_soc_penalty: float,
) -> tuple[float, float]:
    costs = np.asarray(tuple(economic_costs_cny), dtype=float)
    scales = (nominal_cost_cny, material_reference_penalty, material_soc_penalty)
    if (
        costs.size == 0 or not np.isfinite(costs).all() or np.any(costs < 0)
        or not any(costs > 0) or not np.isfinite(scales).all() or min(scales) <= 0
    ):
        raise ValueError("calibration requires positive finite Train costs and penalty scales")
    typical_economics = float(np.median(costs[costs > 0]) / nominal_cost_cny)
    return typical_economics / material_reference_penalty, typical_economics / material_soc_penalty


def _origins(train_episodes: Sequence[object]) -> tuple[CalibrationOrigin, ...]:
    result: list[CalibrationOrigin] = []
    for episode in train_episodes:
        load_filter = CausalBaseLoadFilter(sample_seconds=DT_SECONDS, tau_seconds=180.0)
        prior_onboard = False
        prior_load = 0.0
        for step_index, (source_load, mode) in enumerate(zip(episode.load_kw, episode.operating_mode)):
            if mode != "onboard":
                prior_onboard = False
                continue
            if not prior_onboard:
                load_filter = CausalBaseLoadFilter(sample_seconds=DT_SECONDS, tau_seconds=180.0)
                load_filter.commit(0.0)
                prior_load = 0.0
            previous_base = load_filter.observed_base_kw
            assert previous_base is not None
            load = normalize_onboard_load_kw(float(source_load))
            load_filter.commit(load)
            base = load_filter.observed_base_kw
            assert base is not None
            result.append(CalibrationOrigin(
                str(episode.sample_id), step_index, load, base,
                previous_base, load - prior_load,
            ))
            prior_load = load
            prior_onboard = True
    if not result:
        raise ValueError("Train contains no ONBOARD decision origins")
    return tuple(result)


def _quantile_origins(origins: Sequence[CalibrationOrigin], count: int) -> tuple[CalibrationOrigin, ...]:
    if count <= 0:
        raise ValueError("sample count must be positive")
    ordered = sorted(origins, key=lambda value: (value.load_kw, value.sample_id, value.step_index))
    positions = np.rint(np.linspace(0, len(ordered) - 1, min(count, len(ordered)))).astype(int)
    return tuple(ordered[index] for index in positions)


def _reference(origin: CalibrationOrigin) -> tuple[float, ...]:
    alpha = math.exp(-DT_SECONDS / 180.0)
    base = origin.base_kw
    result = []
    for _ in range(HORIZON):
        base = alpha * base + (1.0 - alpha) * origin.load_kw
        result.append(base)
    return tuple(result)


def _baseline_terms(mpc: EconomicMPC, origin: CalibrationOrigin, soc: float):
    reference = _reference(origin)
    powers = tuple(float(np.clip(value, 0.0, mpc.plant.fuel_cell_rated_total_kw)) for value in reference)
    state = _state_for_origin(mpc, origin, soc)
    return mpc.objective_terms(powers, (origin.load_kw,) * HORIZON, reference, state)


def _state_for_origin(mpc: EconomicMPC, origin: CalibrationOrigin, soc: float) -> AccountState:
    return AccountState(
        soc=soc,
        previous_fc_kw=float(np.clip(origin.previous_base_kw, 0.0, mpc.plant.fuel_cell_rated_total_kw)),
    )


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    data = np.asarray(tuple(values), dtype=float)
    if data.size == 0 or not np.isfinite(data).all():
        raise ValueError("summary needs finite observations")
    return {
        "count": int(data.size), "min": float(np.min(data)),
        "p10": float(np.quantile(data, 0.10)),
        "p50": float(np.median(data)),
        "p90": float(np.quantile(data, 0.90)),
        "p95": float(np.quantile(data, 0.95)),
        "max": float(np.max(data)),
    }


def _pilot_cases(origins: Sequence[CalibrationOrigin]) -> tuple[tuple[CalibrationOrigin, float], ...]:
    quantile = _quantile_origins(origins, 6)
    cases = [(origin, soc) for origin in quantile for soc in SOC_SCENARIOS]
    biggest_increase = max(origins, key=lambda item: item.load_change_kw)
    biggest_decrease = min(origins, key=lambda item: item.load_change_kw)
    for origin in (biggest_increase, biggest_decrease):
        if (origin, 0.5) not in cases:
            cases.append((origin, 0.5))
    return tuple(cases)


def _screen_actions(
    mpc: EconomicMPC, origins: Sequence[CalibrationOrigin], actions: Sequence[MPCWeights],
) -> dict[str, object]:
    observations: list[dict[str, object]] = []
    pilot_cases = _pilot_cases(origins)
    for case_index, (origin, soc) in enumerate(pilot_cases, start=1):
        forecast = (origin.load_kw,) * HORIZON
        reference = _reference(origin)
        state = _state_for_origin(mpc, origin, soc)
        case = {"case_id": origin.case_id, "sample_id": origin.sample_id,
                "load_kw": origin.load_kw, "soc": soc,
                "load_change_kw": origin.load_change_kw, "actions": []}
        for action_id, action in enumerate(actions):
            try:
                plan = mpc.solve(forecast, reference, state, action)
            except (RuntimeError, ValueError) as exc:
                case["actions"].append({"action_id": action_id, "error": str(exc)})
                continue
            terms = mpc.objective_terms(plan.fc_power_kw, forecast, reference, state)
            case["actions"].append({
                "action_id": action_id,
                "first_fc_kw": plan.fc_power_kw[0],
                "soc_path": list(plan.soc_path),
                "economic_normalized": terms.economic_normalized,
                "weighted_reference": action.lambda_ref * terms.reference_penalty,
                "weighted_soc": action.lambda_soc * terms.soc_penalty,
            })
        observations.append(case)
        print(f"pilot_case={case_index}/{len(pilot_cases)} id={origin.case_id} soc={soc:.2f}", flush=True)
    action_successes = [sum("error" not in case["actions"][index] for case in observations)
                        for index in range(len(actions))]
    near_identical: list[tuple[int, int]] = []
    for left in range(len(actions)):
        for right in range(left + 1, len(actions)):
            if action_successes[left] != len(observations) or action_successes[right] != len(observations):
                continue
            fc_difference = max(abs(case["actions"][left]["first_fc_kw"]
                                    - case["actions"][right]["first_fc_kw"])
                                for case in observations)
            soc_difference = max(
                abs(a - b)
                for case in observations
                for a, b in zip(case["actions"][left]["soc_path"], case["actions"][right]["soc_path"])
            )
            if fc_difference <= 5.0 and soc_difference <= 0.005:
                near_identical.append((left, right))
    first_fc_spreads = []
    for case in observations:
        powers = [item["first_fc_kw"] for item in case["actions"] if "error" not in item]
        first_fc_spreads.append(max(powers) - min(powers) if powers else 0.0)
    return {
        "case_count": len(observations),
        "solver_successes_per_action": action_successes,
        "first_fc_spread_kw": _summary(first_fc_spreads),
        "near_identical_action_pairs_5kw_0p005soc": [list(pair) for pair in near_identical],
        "cases": observations,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generate_calibration(
    dataset: FormalTrainingDataset, *, sample_count: int = 240,
    soc_response_factor: float = 1.0,
) -> dict[str, object]:
    train = dataset.load_train()
    if dataset.opened_test_payloads != 0:
        raise PermissionError("Test payloads were opened before Train calibration")
    origins = _origins(train)
    selected = _quantile_origins(origins, sample_count)
    unscaled_mpc = EconomicMPC(nominal_cost_cny=1.0)
    cost_observations = [_baseline_terms(unscaled_mpc, origin, 0.5).economic_cost_cny
                         for origin in selected]
    positive_costs = [value for value in cost_observations if value > 0.0]
    if not positive_costs:
        raise ValueError("Train reference policy produced no positive five-step costs")
    nominal_cost = float(np.median(positive_costs))
    load_change_p95_kw = float(np.quantile(
        [abs(origin.load_change_kw) for origin in origins], 0.95,
    ))
    material_ref = HORIZON * (load_change_p95_kw / unscaled_mpc.plant.fuel_cell_rated_total_kw) ** 2
    soc_observations = [
        _baseline_terms(unscaled_mpc, origin, soc).soc_penalty
        for origin in selected for soc in (0.4, 0.6)
    ]
    material_soc = float(np.median([value for value in soc_observations if value > 0.0]))
    base_ref, base_soc = calibrate_weight_bases(
        positive_costs, nominal_cost, material_ref, material_soc,
    )
    ref_axis = four_level_axis(base_ref)
    balanced_soc_axis = four_level_axis(base_soc)
    soc_axis = four_level_axis(base_soc, positive_multiplier=soc_response_factor)
    actions = cartesian_actions(ref_axis, soc_axis)
    scaled_mpc = EconomicMPC(nominal_cost_cny=nominal_cost)
    screen = _screen_actions(scaled_mpc, origins, actions)
    if dataset.opened_test_payloads != 0:
        raise PermissionError("Train calibration opened Test payloads")
    return {
        "schema_version": "v3_persistence_action_calibration_v1",
        "split_used": "train",
        "train_episode_count": len(train),
        "train_onboard_origin_count": len(origins),
        "test_payloads_opened": dataset.opened_test_payloads,
        "manifest_sha256": {
            "power": _sha256(dataset._power_root / "metadata" / "sample_manifest.csv"),
            "ais": _sha256(dataset._ais_root / "metadata" / "sample_manifest.csv"),
            "modes": _sha256(dataset._mode_root / "metadata" / "sample_manifest.csv"),
        },
        "calibration_assumptions": {
            "forecast": "hold current ONBOARD load for five 30 s steps",
            "reference": "causal 180 s LPF preview; reset at each ONBOARD run",
            "reference_policy": "clip each LPF reference to 0..600 kW FC",
            "reference_previous_fc": "previous observed LPF value; not a replayed MPC command",
            "nominal_soc": 0.5,
            "nominal_cost_rule": "median positive five-step optimizer economic numerator over quantile-sampled Train origins",
            "material_reference_rule": "five times squared Train p95 absolute one-step load change divided by 600 kW",
            "material_soc_rule": "median positive five-step SOC penalty at SOC 0.4 and 0.6 under reference policy",
            "axis_multipliers": list(AXIS_MULTIPLIERS),
            "pilot_soc_scenarios": list(SOC_SCENARIOS),
            "status": "candidate calibration and open-loop response screen; no DQN training or Validation selection",
        },
        "nominal_cost_cny": nominal_cost,
        "economic_cost_cny_distribution": _summary(cost_observations),
        "positive_economic_cost_cny_distribution": _summary(positive_costs),
        "zero_economic_cost_count": len(cost_observations) - len(positive_costs),
        "calibration_origin_count": len(selected),
        "calibration_origin_ids": [origin.case_id for origin in selected],
        "train_abs_load_change_p95_kw": load_change_p95_kw,
        "material_reference_penalty": material_ref,
        "material_soc_penalty": material_soc,
        "soc_penalty_distribution": _summary(soc_observations),
        "base_lambda_ref": base_ref,
        "base_lambda_soc": base_soc,
        "soc_response_factor": soc_response_factor,
        "lambda_soc_axis_balanced": list(balanced_soc_axis),
        "lambda_ref_axis": list(ref_axis),
        "lambda_soc_axis": list(soc_axis),
        "actions": [
            {"action_id": index, "lambda_ref": action.lambda_ref, "lambda_soc": action.lambda_soc}
            for index, action in enumerate(actions)
        ],
        "pilot_screen": screen,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train-only v3 persistence MPC action calibration")
    parser.add_argument("--power-root", type=Path, default=DEFAULT_POWER_ROOT)
    parser.add_argument("--ais-root", type=Path, default=DEFAULT_AIS_ROOT)
    parser.add_argument("--mode-root", type=Path, default=DEFAULT_MODE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-count", type=int, default=240)
    parser.add_argument("--soc-response-factor", type=float, default=1.0)
    args = parser.parse_args(argv)
    dataset = FormalTrainingDataset.open(args.power_root, args.ais_root, args.mode_root)
    report = generate_calibration(
        dataset, sample_count=args.sample_count,
        soc_response_factor=args.soc_response_factor,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f"{args.output.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({
        "output": str(args.output), "nominal_cost_cny": report["nominal_cost_cny"],
        "lambda_ref_axis": report["lambda_ref_axis"],
        "lambda_soc_axis": report["lambda_soc_axis"],
        "pilot_solver_successes_per_action": report["pilot_screen"]["solver_successes_per_action"],
        "test_payloads_opened": report["test_payloads_opened"],
    }, indent=2))


if __name__ == "__main__":
    main()
