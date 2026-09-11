from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"C:\Users\20883\OneDrive\Desktop\lstm-dqn-mpc\lstm-dqn-mpc")
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "src" / "main")]

import train_dqn_mpc_mlp as training
from dqn.utils.action_mapper import MPCWeightAction
from envs.dqn_mpc_weight_env import DqnMpcWeightEnv, MpcSolveFailure
from mpc.solvers.fc_dp0_curve import h2_kg_step_dp0_quadratic


OUT = ROOT / "outputs" / "mpc_action_redesign_20260911"
SEGMENTS = (
    ("low_load", "train_parent_037_02"),
    ("ordinary", "train_parent_061_01"),
    ("fluctuating", "train_parent_063_01"),
    ("sustained_high", "train_parent_013_02"),
    ("soc_pressure", "train_parent_021_01"),
)
CANDIDATES = (
    ("E0", 0.40, 0.25, 8.0, 8.0, "economy_extreme"),
    ("E2", 0.40, 0.35, 20.0, 12.0, "economy_moderate"),
    ("B2", 0.25, 0.50, 50.0, 20.0, "balanced"),
    ("B4", 0.20, 0.50, 40.0, 16.0, "balanced"),
    ("S1", 0.25, 0.50, 30.0, 40.0, "smooth"),
    ("S3", 0.30, 0.40, 30.0, 40.0, "smooth"),
    ("S5", 0.20, 0.65, 60.0, 30.0, "smooth_protective"),
    ("R2", 0.15, 0.65, 80.0, 12.0, "soc"),
    ("R4", 0.15, 0.80, 120.0, 8.0, "soc_extreme"),
    ("R6", 0.15, 0.65, 120.0, 12.0, "soc_refined"),
    ("R7", 0.15, 0.70, 160.0, 12.0, "soc_refined"),
)


def rollout(loads: np.ndarray, action: MPCWeightAction, config: object) -> dict[str, object]:
    env = DqnMpcWeightEnv(
        loads_kw=loads,
        base_config=config,
        initial_soc=0.55,
        actions=(action,),
    )
    env.reset()
    previous_fc = float(env.previous_fc_kw)
    records: list[tuple[float, float, float, float, float]] = []
    failure_step: int | None = None
    failure_status = ""
    while not env.done:
        try:
            _, reward, _, info = env.step(int(action.action_id))
        except MpcSolveFailure as exc:
            failure_step = int(exc.decision_index)
            failure_status = str(exc.solver_status)
            break
        fc = float(info["p_fc_kw"])
        batt = float(info["p_batt_kw"])
        records.append((float(reward), fc, batt, float(info["soc_after"]), abs(fc - previous_fc)))
        previous_fc = fc

    values = np.asarray(records, dtype=float)
    steps = int(values.shape[0])
    hours = steps / 3600.0
    h2_kg = (
        float(np.sum(h2_kg_step_dp0_quadratic(
            values[:, 1], dt_seconds=1.0, p_rated_total_kw=config.fuel_cell_max_kw
        )))
        if steps else float("nan")
    )
    net_kwh = float(np.sum(values[:, 2]) / 3600.0) if steps else float("nan")
    throughput_kwh = float(np.sum(np.abs(values[:, 2])) / 3600.0) if steps else float("nan")
    fc_tv_kw = float(np.sum(values[:, 4])) if steps else float("nan")
    return {
        "completed": failure_step is None,
        "failure_step": failure_step,
        "failure_status": failure_status,
        "steps": steps,
        "hours_executed": hours,
        "reward_per_step": float(np.mean(values[:, 0])) if steps else float("nan"),
        "min_soc": float(np.min(values[:, 3])) if steps else 0.55,
        "final_soc": float(env.current_soc),
        "h2_kg_h": h2_kg / hours if steps else float("nan"),
        "battery_net_discharge_kwh_h": net_kwh / hours if steps else float("nan"),
        "battery_throughput_kwh_h": throughput_kwh / hours if steps else float("nan"),
        "fc_tv_kw_h": fc_tv_kw / hours if steps else float("nan"),
        "mean_fc_kw": float(np.mean(values[:, 1])) if steps else float("nan"),
        "fc_ge_590_fraction": float(np.mean(values[:, 1] >= 590.0)) if steps else float("nan"),
        "first_p_fc_kw": float(values[0, 1]) if steps else float("nan"),
        "first_p_batt_kw": float(values[0, 2]) if steps else float("nan"),
    }


def main() -> None:
    split = training.load_voyage_split()
    config = training.build_formal_mpc_config()
    actions = [
        MPCWeightAction(index, row[1], row[2], row[3], row[4], row[0])
        for index, row in enumerate(CANDIDATES)
    ]
    rows: list[dict[str, object]] = []
    segment_catalog: list[dict[str, object]] = []
    for regime, segment_id in SEGMENTS:
        loads = training.load_operating_segment_loads("train", segment_id, split=split).astype(float)
        segment_catalog.append({
            "regime": regime,
            "segment_id": segment_id,
            "points": int(loads.size),
            "hours": float((loads.size - 1) / 3600.0),
            "mean_load_kw": float(np.mean(loads)),
            "std_load_kw": float(np.std(loads)),
            "p95_load_kw": float(np.quantile(loads, 0.95)),
            "max_load_kw": float(np.max(loads)),
            "load_ge_590_fraction": float(np.mean(loads >= 590.0)),
        })
        for action, meta in zip(actions, CANDIDATES):
            row = {
                "candidate": meta[0], "family": meta[5],
                "q_h2": meta[1], "q_batt": meta[2], "q_soc": meta[3], "q_fcvar": meta[4],
                "regime": regime, "segment_id": segment_id,
            }
            row.update(rollout(loads, action, config))
            rows.append(row)
        print("SEGMENT_DONE", regime, segment_id, flush=True)

    details = pd.DataFrame(rows)
    details.to_csv(OUT / "stage2_candidate_segment_results.csv", index=False)
    segments = pd.DataFrame(segment_catalog)
    segments.to_csv(OUT / "stage2_train_segments.csv", index=False)
    summary = details.groupby(
        ["candidate", "family", "q_h2", "q_batt", "q_soc", "q_fcvar"], as_index=False
    ).agg(
        completed=("completed", "sum"),
        failures=("completed", lambda values: int((~values).sum())),
        reward_per_step=("reward_per_step", "mean"),
        worst_min_soc=("min_soc", "min"),
        mean_min_soc=("min_soc", "mean"),
        mean_final_soc=("final_soc", "mean"),
        h2_kg_h=("h2_kg_h", "mean"),
        battery_net_discharge_kwh_h=("battery_net_discharge_kwh_h", "mean"),
        battery_throughput_kwh_h=("battery_throughput_kwh_h", "mean"),
        fc_tv_kw_h=("fc_tv_kw_h", "mean"),
        mean_fc_kw=("mean_fc_kw", "mean"),
        fc_ge_590_fraction=("fc_ge_590_fraction", "mean"),
    )
    summary.to_csv(OUT / "stage2_candidate_summary.csv", index=False)
    (OUT / "stage2_candidates.json").write_text(
        json.dumps([
            {"candidate": row[0], "family": row[5], "weights": list(row[1:5])}
            for row in CANDIDATES
        ], indent=2) + "\n", encoding="utf-8"
    )
    print(segments.to_string(index=False), flush=True)
    print(summary.to_string(index=False), flush=True)
    print("STAGE2_COMPLETE", OUT, flush=True)


if __name__ == "__main__":
    main()
