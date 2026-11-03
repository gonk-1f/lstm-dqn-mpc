"""Greedy evaluation over exactly the existing eight validation voyages."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from stable_baselines3 import DDPG
from .env import AggregateDDPGEnv
from .validate import ACCOUNTING_FIELDS, step_accounting
from .economic_reward import evaluation_identity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    env = AggregateDDPGEnv(args.data_root, split="validation")
    model = DDPG.load(str(args.model), device="cpu", action_noise=None)
    reward_metadata = evaluation_identity(model, env.reward_identity)
    rows = []
    trajectories = []
    trajectory_fields = (
        "sample_id", "step_index", "interval_start_s", "interval_end_s", "step_executed",
        "load_kw", "requested_fc_kw", "fc_actual_kw", "battery_actual_kw", "soc",
        "delta_fc_kw", "soc_penalty", "fc_ramp_penalty", "projected",
        "feasible_fc_lower_kw", "feasible_fc_upper_kw", "reward", "terminal_reason",
        "fc_efficiency_actual", "equivalent_fc_efficiency",
    ) + ACCOUNTING_FIELDS
    try:
        for _ in range(len(env.episodes)):
            obs, _ = env.reset()
            step_index = 0
            while True:
                act, _ = model.predict(obs, deterministic=True)
                before = env.physical
                obs, reward, terminated, truncated, info = env.step(act)
                step_index += 1
                step_executed = info.get("terminal_reason") != "no_feasible_physical_power"
                trajectory = {field: info.get(field) for field in trajectory_fields}
                trajectory.update(step_accounting(env, before, info))
                trajectory.update(
                    step_index=step_index,
                    interval_start_s=step_index - 1,
                    interval_end_s=step_index if step_executed else None,
                    step_executed=step_executed,
                    reward=reward,
                )
                trajectories.append(trajectory)
                if terminated or truncated:
                    rows.append(info["vessel_episode"])
                    print(f"{rows[-1]['sample_id']}: {rows[-1]['terminal_reason']} "
                          f"coverage={rows[-1]['coverage']:.3f} "
                          f"ramp_rms={rows[-1]['fc_ramp_rms_kw_per_step']:.2f}kW")
                    break
        if len(rows) != 8 or len({r["sample_id"] for r in rows}) != 8:
            raise RuntimeError("Validation must cover exactly eight distinct voyages")
        with (args.output_dir / "validation.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summary = {
            **reward_metadata,
            "validation_voyages": len(rows),
            "completed": sum(r["completed"] for r in rows),
            "completion_rate": sum(r["completed"] for r in rows) / len(rows),
            "mean_coverage": sum(r["coverage"] for r in rows) / len(rows),
            "per_voyage": rows,
        }
        (args.output_dir / "validation.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in summary.items() if k != "per_voyage"},
                         ensure_ascii=False, indent=2))
        with (args.output_dir / "validation_trajectories.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as f:
            writer = csv.DictWriter(f, fieldnames=trajectory_fields)
            writer.writeheader()
            writer.writerows(trajectories)
    finally:
        env.close()


if __name__ == "__main__":
    main()
