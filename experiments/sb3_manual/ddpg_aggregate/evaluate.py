"""Greedy evaluation over exactly the existing eight validation voyages."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from stable_baselines3 import DDPG
from .env import AggregateDDPGEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = AggregateDDPGEnv(args.data_root, split="validation")
    model = DDPG.load(str(args.model), device="cpu")
    rows = []
    try:
        for _ in range(len(env.episodes)):
            obs, _ = env.reset()
            while True:
                act, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(act)
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
    finally:
        env.close()


if __name__ == "__main__":
    main()
