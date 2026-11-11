
from __future__ import annotations

from datetime import datetime

import numpy as np
import torch

from stable_baselines3 import TD3
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

from callbacks import ShipEMSTrainingCallback

from td3_config import (
    DATA_ROOT,
    DEVICE,
    MAX_EPISODES,
    OUTPUT_ROOT,
    REWARD_PLOT_INTERVAL,
    SEED,
    TRAIN_DIAGNOSTIC_INTERVAL,
    VALIDATION_INTERVAL,
    load_dataset_bundle,
    make_env,
    save_json,
)


# ============================================================
# TD3 fixed configuration
# ============================================================

LEARNING_RATE = 0.001
BUFFER_SIZE = 1_000_000
LEARNING_STARTS = 5000
BATCH_SIZE = 64

GAMMA = 1.0
TAU = 0.001

TRAIN_FREQ = (1, "step")
GRADIENT_STEPS = 1

N_STEPS = 1
POLICY_DELAY = 2

ACTION_NOISE_SIGMA = 0.05
TARGET_POLICY_NOISE = 0.2
TARGET_NOISE_CLIP = 0.5

NET_ARCH = {
    "pi": [256, 256],
    "qf": [256, 256],
}


def main():

    torch.set_num_threads(1)
    set_random_seed(SEED)

    # --------------------------------------------------------
    # 1. Load the official dataset split
    # --------------------------------------------------------

    bundle = load_dataset_bundle(DATA_ROOT)

    train_profile = bundle.train_profile
    train_id = bundle.train_id

    validation_profiles = bundle.validation_profiles
    validation_ids = bundle.validation_ids

    train_steps_per_episode = len(train_profile) - 1

    if train_steps_per_episode <= 0:
        raise RuntimeError("固定训练航段没有有效环境步。")

    # --------------------------------------------------------
    # 2. Create independent output directory
    # --------------------------------------------------------

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir = OUTPUT_ROOT / f"run_{run_stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    log_dir = run_dir / "sb3_logs"
    tensorboard_dir = run_dir / "tensorboard"

    log_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 3. Save configuration
    # --------------------------------------------------------

    config = {
        "algorithm": "Stable-Baselines3 TD3",

        "seed": SEED,
        "device": DEVICE,

        "data_root": str(DATA_ROOT.resolve()),
        "train_voyage": train_id,
        "train_sample_count": len(train_profile),
        "train_steps_per_episode": train_steps_per_episode,

        "validation_ids": validation_ids,
        "validation_count": len(validation_ids),

        "max_episodes": MAX_EPISODES,

        "learning_rate": LEARNING_RATE,
        "buffer_size": BUFFER_SIZE,
        "learning_starts": LEARNING_STARTS,
        "batch_size": BATCH_SIZE,

        "gamma": GAMMA,
        "tau": TAU,

        "train_freq": TRAIN_FREQ,
        "gradient_steps": GRADIENT_STEPS,

        "n_steps": N_STEPS,
        "policy_delay": POLICY_DELAY,

        "action_noise_sigma": ACTION_NOISE_SIGMA,
        "target_policy_noise": TARGET_POLICY_NOISE,
        "target_noise_clip": TARGET_NOISE_CLIP,

        "net_arch": NET_ARCH,

        "critic_loss": "SB3 native TD3 implementation",

        "reward_plot_interval": REWARD_PLOT_INTERVAL,
        "train_diagnostic_interval": TRAIN_DIAGNOSTIC_INTERVAL,
        "validation_interval": VALIDATION_INTERVAL,
    }

    save_json(run_dir / "training_config.json", config)

    # --------------------------------------------------------
    # 4. Create fixed-voyage training environment
    # --------------------------------------------------------

    def make_training_env():

        env = make_env(
            [train_profile],
            [train_id],
        )

        return Monitor(
            env,
            filename=str(run_dir / "monitor"),
        )

    train_env = DummyVecEnv([make_training_env])

    # --------------------------------------------------------
    # 5. Normal exploration noise
    # --------------------------------------------------------

    action_noise = NormalActionNoise(
        mean=np.zeros(1, dtype=np.float32),
        sigma=np.full(
            1,
            ACTION_NOISE_SIGMA,
            dtype=np.float32,
        ),
    )

    # --------------------------------------------------------
    # 6. Create official SB3 TD3
    # --------------------------------------------------------

    model = TD3(
        policy="MlpPolicy",
        env=train_env,

        learning_rate=LEARNING_RATE,

        buffer_size=BUFFER_SIZE,
        learning_starts=LEARNING_STARTS,
        batch_size=BATCH_SIZE,

        gamma=GAMMA,
        tau=TAU,

        train_freq=TRAIN_FREQ,
        gradient_steps=GRADIENT_STEPS,

        n_steps=N_STEPS,
        policy_delay=POLICY_DELAY,

        action_noise=action_noise,

        target_policy_noise=TARGET_POLICY_NOISE,
        target_noise_clip=TARGET_NOISE_CLIP,

        policy_kwargs={
            "net_arch": NET_ARCH,
        },

        seed=SEED,
        device=DEVICE,

        verbose=1,

        tensorboard_log=str(tensorboard_dir),
    )

    # --------------------------------------------------------
    # 7. SB3 logger
    # --------------------------------------------------------

    model.set_logger(
        configure(
            str(log_dir),
            ["stdout", "csv", "tensorboard"],
        )
    )

    # --------------------------------------------------------
    # 8. Training callback
    # --------------------------------------------------------

    callback = ShipEMSTrainingCallback(
        train_profile=train_profile,
        train_id=train_id,

        validation_profiles=validation_profiles,
        validation_ids=validation_ids,

        output_dir=run_dir,

        max_episodes=MAX_EPISODES,

        reward_plot_interval=REWARD_PLOT_INTERVAL,
        train_diagnostic_interval=TRAIN_DIAGNOSTIC_INTERVAL,
        validation_interval=VALIDATION_INTERVAL,

        seed=SEED,
        verbose=1,
    )

    # --------------------------------------------------------
    # 9. Training
    # --------------------------------------------------------

    total_timesteps_upper_bound = (
        MAX_EPISODES * train_steps_per_episode
    )

    print("=" * 72)
    print("Ship EMS TD3 training")
    print("=" * 72)

    print(f"Algorithm: SB3 TD3")
    print(f"Training voyage: {train_id}")
    print(f"Training episodes: {MAX_EPISODES}")

    print(f"Learning starts: {LEARNING_STARTS}")
    print(f"N-step return: {N_STEPS}")
    print(f"Train frequency: {TRAIN_FREQ}")
    print(f"Gradient steps: {GRADIENT_STEPS}")

    print(f"Batch size: {BATCH_SIZE}")
    print(f"Gamma: {GAMMA}")
    print(f"Tau: {TAU}")

    print(f"Action noise: {ACTION_NOISE_SIGMA}")
    print(f"Device: {model.device}")

    print(f"Output directory: {run_dir}")

    print("=" * 72)

    try:

        model.learn(
            total_timesteps=total_timesteps_upper_bound,

            callback=callback,

            reset_num_timesteps=True,

            progress_bar=False,

            log_interval=10,
        )

    finally:

        train_env.close()

    # --------------------------------------------------------
    # 10. Save final model and summary
    # --------------------------------------------------------

    model.save(str(run_dir / "last_model"))

    save_json(
        run_dir / "training_summary.json",
        {
            "episodes_completed": len(callback.episode_rows),

            "total_timesteps": int(model.num_timesteps),

            "critic_updates": int(
                getattr(model, "_n_updates", 0)
            ),

            "best_validation_episode": callback.best_episode,

            "best_validation_score": (
                list(callback.best_score)
                if callback.best_score is not None
                else None
            ),

            "last_model": str(
                run_dir / "last_model.zip"
            ),

            "best_model": (
                str(run_dir / "best_model.zip")
                if (run_dir / "best_model.zip").is_file()
                else None
            ),
        },
    )

    print("=" * 72)
    print("TD3 training finished")

    print(
        f"Episodes completed: "
        f"{len(callback.episode_rows)}"
    )

    print(
        f"Total environment steps: "
        f"{model.num_timesteps}"
    )

    print(f"Output: {run_dir}")

    print("=" * 72)


if __name__ == "__main__":
    main()
