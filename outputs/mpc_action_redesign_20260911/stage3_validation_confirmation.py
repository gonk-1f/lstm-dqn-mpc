from __future__ import annotations

import concurrent.futures
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"C:\Users\20883\OneDrive\Desktop\lstm-dqn-mpc\lstm-dqn-mpc")
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "src" / "main")]

import train_dqn_mpc_mlp as training
from dqn.utils.action_mapper import DQN_MPC_WEIGHT_ACTIONS
from envs.dqn_mpc_weight_env import DqnMpcWeightEnv, MpcSolveFailure
from mpc.solvers.fc_dp0_curve import h2_kg_step_dp0_quadratic


OUT = ROOT / "outputs" / "mpc_action_redesign_20260911"
TRACE_DIR = OUT / "validation_extreme_traces"
EXTREMES = {"validation_parent_016_01", "validation_parent_053_01"}


def run_action(action_id: int) -> list[dict[str, object]]:
    split = training.load_voyage_split()
    config = training.build_formal_mpc_config()
    action = DQN_MPC_WEIGHT_ACTIONS[action_id]
    rows: list[dict[str, object]] = []
    for segment_id in split.validation_segments:
        loads = training.load_operating_segment_loads("validation", segment_id, split=split).astype(float)
        env = DqnMpcWeightEnv(
            loads_kw=loads, base_config=config, initial_soc=0.55, actions=(action,)
        )
        env.reset()
        previous_fc = float(env.previous_fc_kw)
        records: list[dict[str, float | int]] = []
        failure_step: int | None = None
        failure_status = ""
        while not env.done:
            try:
                _, reward, _, info = env.step(action_id)
            except MpcSolveFailure as exc:
                failure_step = int(exc.decision_index)
                failure_status = str(exc.solver_status)
                break
            fc = float(info["p_fc_kw"])
            batt = float(info["p_batt_kw"])
            records.append({
                "step": int(env.decision_index - 1),
                "reward": float(reward),
                "p_fc_kw": fc,
                "p_batt_kw": batt,
                "soc": float(info["soc_after"]),
                "fc_delta_abs_kw": abs(fc - previous_fc),
            })
            previous_fc = fc
        frame = pd.DataFrame(records)
        steps = int(len(frame))
        hours = steps / 3600.0
        h2_kg = (
            float(np.sum(h2_kg_step_dp0_quadratic(
                frame["p_fc_kw"].to_numpy(float),
                dt_seconds=1.0,
                p_rated_total_kw=config.fuel_cell_max_kw,
            ))) if steps else float("nan")
        )
        net_kwh = float(frame["p_batt_kw"].sum() / 3600.0) if steps else float("nan")
        throughput_kwh = float(frame["p_batt_kw"].abs().sum() / 3600.0) if steps else float("nan")
        fc_tv_kw = float(frame["fc_delta_abs_kw"].sum()) if steps else float("nan")
        rows.append({
            "action_id": action_id,
            "action_name": action.name,
            "q_h2": action.q_h2,
            "q_batt": action.q_batt,
            "q_soc": action.q_soc,
            "q_fcvar": action.q_fc_var,
            "segment_id": segment_id,
            "completed": failure_step is None,
            "failure_step": failure_step,
            "failure_status": failure_status,
            "steps": steps,
            "hours_executed": hours,
            "reward_sum": float(frame["reward"].sum()) if steps else float("nan"),
            "reward_per_step": float(frame["reward"].mean()) if steps else float("nan"),
            "min_soc": float(frame["soc"].min()) if steps else 0.55,
            "final_soc": float(env.current_soc),
            "h2_kg": h2_kg,
            "h2_kg_h": h2_kg / hours if steps else float("nan"),
            "battery_net_discharge_kwh": net_kwh,
            "battery_net_discharge_kwh_h": net_kwh / hours if steps else float("nan"),
            "battery_throughput_kwh": throughput_kwh,
            "battery_throughput_kwh_h": throughput_kwh / hours if steps else float("nan"),
            "fc_tv_kw": fc_tv_kw,
            "fc_tv_kw_h": fc_tv_kw / hours if steps else float("nan"),
            "mean_fc_kw": float(frame["p_fc_kw"].mean()) if steps else float("nan"),
            "fc_ge_590_fraction": float((frame["p_fc_kw"] >= 590.0).mean()) if steps else float("nan"),
            "first_p_fc_kw": float(frame.iloc[0]["p_fc_kw"]) if steps else float("nan"),
            "first_p_batt_kw": float(frame.iloc[0]["p_batt_kw"]) if steps else float("nan"),
        })
        if segment_id in EXTREMES:
            trace = frame.copy()
            trace.insert(0, "segment_id", segment_id)
            trace.insert(1, "action_id", action_id)
            trace.insert(2, "action_name", action.name)
            trace.to_csv(TRACE_DIR / f"{segment_id}_A{action_id}.csv", index=False)
    print("ACTION_DONE", action_id, action.name, flush=True)
    return rows


def main() -> None:
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(run_action, range(4)))
    details = pd.DataFrame([row for group in results for row in group])
    details.to_csv(OUT / "stage3_validation_segment_results.csv", index=False)
    summary_rows: list[dict[str, object]] = []
    for action_id, group in details.groupby("action_id", sort=True):
        steps = int(group["steps"].sum())
        hours = float(group["hours_executed"].sum())
        summary_rows.append({
            "action_id": int(action_id),
            "action_name": group.iloc[0]["action_name"],
            "weights": f"({group.iloc[0]['q_h2']},{group.iloc[0]['q_batt']},{group.iloc[0]['q_soc']},{group.iloc[0]['q_fcvar']})",
            "completed": int(group["completed"].sum()),
            "failures": int((~group["completed"]).sum()),
            "steps": steps,
            "reward_per_step": float(group["reward_sum"].sum() / steps),
            "worst_min_soc": float(group["min_soc"].min()),
            "mean_min_soc": float(group["min_soc"].mean()),
            "mean_final_soc": float(group["final_soc"].mean()),
            "h2_kg_h": float(group["h2_kg"].sum() / hours),
            "battery_net_discharge_kwh_h": float(group["battery_net_discharge_kwh"].sum() / hours),
            "battery_throughput_kwh_h": float(group["battery_throughput_kwh"].sum() / hours),
            "fc_tv_kw_h": float(group["fc_tv_kw"].sum() / hours),
            "mean_fc_kw": float(np.average(group["mean_fc_kw"], weights=group["steps"])),
            "fc_ge_590_fraction": float(np.average(group["fc_ge_590_fraction"], weights=group["steps"])),
            "first_p_fc_range_kw": float(group["first_p_fc_kw"].max() - group["first_p_fc_kw"].min()),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "stage3_validation_summary.csv", index=False)
    extremes = details[details["segment_id"].isin(EXTREMES)].copy()
    extremes.to_csv(OUT / "stage3_validation_extremes.csv", index=False)
    print(summary.to_string(index=False), flush=True)
    print(extremes.to_string(index=False), flush=True)
    print("STAGE3_COMPLETE", OUT, flush=True)


if __name__ == "__main__":
    main()
