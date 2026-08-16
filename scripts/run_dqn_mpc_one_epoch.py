from __future__ import annotations

import importlib.util
import json
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = (
    ROOT
    / "src"
    / "main"
    / "train_dqn_mpc_mlp.py"
)

ONE_EPOCH_STEPS = 1_236_214


def load_training_module():
    spec = importlib.util.spec_from_file_location(
        "train_dqn_mpc_one_epoch",
        TRAIN_SCRIPT,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Cannot load training script: {TRAIN_SCRIPT}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module


def main() -> None:
    module = load_training_module()

    config = module.DQNTrainConfig(
        seed=42,
        max_steps=ONE_EPOCH_STEPS,
        warmup_steps=5000,
        batch_size=64,
        buffer_size=100000,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.99999813,
        target_sync_interval=500,
        log_window_steps=1000,
        solver_failure_reward=-620.0,
        device="cpu",
    )

    stamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_dir = (
        ROOT
        / "outputs"
        / f"dqn_mpc_mlp_one_epoch_{stamp}"
    )

    print(
        "Starting one-epoch DQN-MPC training...",
        flush=True,
    )
    print(
        f"Requested steps: {config.max_steps}",
        flush=True,
    )
    print(
        f"Output directory: {output_dir}",
        flush=True,
    )

    started = time.perf_counter()

    runtime, summary = module.train_dqn_mpc_mlp(
        config=config,
    )

    elapsed = time.perf_counter() - started

    paths = module.write_baseline_outputs(
        runtime=runtime,
        summary=summary,
        output_dir=output_dir,
    )

    validation = summary["validation"]

    result = {
        "requested_steps": config.max_steps,
        "actual_global_step": int(
            runtime.global_step
        ),
        "elapsed_seconds": float(elapsed),
        "elapsed_hours": float(
            elapsed / 3600.0
        ),
        "training_episode_count": len(
            summary["training_episodes"]
        ),
        "final_epsilon": float(
            runtime.policy.epsilon
        ),
        "replay_buffer_size": len(
            runtime.replay_buffer
        ),
        "gradient_updates": len(
            runtime.losses
        ),
        "loss_statistics": summary[
            "loss_statistics"
        ],
        "validation_mean_reward_per_step":
            validation[
                "mean_reward_per_step"
            ],
        "output_files": {
            key: str(path)
            for key, path in paths.items()
        },
    }

    print()
    print("=== ONE-EPOCH TRAINING RESULT ===")
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()