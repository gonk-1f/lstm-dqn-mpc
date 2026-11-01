"""Train native Stable-Baselines3 DDPG on Train35, with an independent model file."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise

from .env import AggregateDDPGEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--total-timesteps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.total_timesteps < 1:
        parser.error("total-timesteps must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = Monitor(AggregateDDPGEnv(args.data_root, split="train"),
                  filename=str(args.output_dir / "train_monitor.csv"))
    try:
        model = DDPG(
            "MlpPolicy", env,
            learning_rate=0.001,
            gamma=0.999,
            batch_size=64,
            buffer_size=1_000_000,
            learning_starts=5000,
            train_freq=32,
            gradient_steps=1,
            tau=0.005,
            policy_kwargs=dict(net_arch=[256, 256]),
            action_noise=NormalActionNoise(mean=np.zeros(1), sigma=0.1 * np.ones(1)),
            seed=args.seed,
            device=args.device,
            verbose=1,
            tensorboard_log=str(args.output_dir / "tensorboard"),
        )
        model.learn(total_timesteps=args.total_timesteps, log_interval=1)
        model_path = args.output_dir / "ddpg_aggregate"
        model.save(str(model_path))
        meta = {"algorithm": "SB3.DDPG", "states": 3, "actions": 1,
                "seed": args.seed, "total_timesteps": int(model.num_timesteps),
                "data_root": str(args.data_root.resolve()),
                "model": str(model_path) + ".zip"}
        (args.output_dir / "run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(json.dumps(meta, ensure_ascii=False, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
