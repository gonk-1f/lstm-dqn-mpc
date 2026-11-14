from __future__ import annotations

from datetime import datetime
from math import isfinite

import numpy as np
import torch
from torch.nn import functional as F

from stable_baselines3 import TD3
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.utils import (
    polyak_update,
    set_random_seed,
    update_learning_rate,
)
from stable_baselines3.common.vec_env import DummyVecEnv

from callbacks import ShipEMSTrainingCallback

from td3_config import (
    ACTION_MAPPING,
    DATA_ROOT,
    DEVICE,
    DIAGNOSTIC_VOYAGE_ID,
    DT_S,
    FAILURE_PENALTY,
    MAX_EPISODES,
    OUTPUT_ROOT,
    REWARD_VERSION,
    REWARD_PLOT_INTERVAL,
    SEED,
    SOC_STEP_WEIGHT,
    TRAIN_DIAGNOSTIC_INTERVAL,
    TRAIN_WINDOW_STEPS,
    VALIDATION_INTERVAL,
    load_dataset_bundle,
    make_env,
    save_json,
)


# ============================================================
# 1. TD3 fixed configuration
# ============================================================

# Separate learning rates
ACTOR_LEARNING_RATE = 0.0001
CRITIC_LEARNING_RATE = 0.001

BUFFER_SIZE = 200_000
LEARNING_STARTS = 25_000
BATCH_SIZE = 64
ADAM_BETAS = (0.9, 0.999)
HUBER_DELTA = 1.0

GAMMA = 1.0
TAU = 0.005

TRAIN_FREQ = (1, "step")
GRADIENT_STEPS = 1

N_STEPS = 1
POLICY_DELAY = 2

ACTION_NOISE_SIGMA = 0.05
TARGET_POLICY_NOISE = 0.1
TARGET_NOISE_CLIP = 0.2

NET_ARCH = {
    "pi": [256, 256],
    "qf": [256, 256],
}


# ============================================================
# 2. TD3 with separate Actor and Critic learning rates
# ============================================================

class SeparateLearningRateTD3(TD3):
    """SB3 TD3 with a Huber critic loss and independent optimizer rates."""

    def _setup_model(self) -> None:
        super()._setup_model()

        # Set initial optimizer learning rates.
        update_learning_rate(
            self.actor.optimizer,
            ACTOR_LEARNING_RATE,
        )

        update_learning_rate(
            self.critic.optimizer,
            CRITIC_LEARNING_RATE,
        )

    def _update_learning_rate(self, optimizers) -> None:
        """
        Called by SB3 TD3 during gradient training.

        Do not call super()._update_learning_rate(),
        because it would assign the same learning rate
        schedule to both optimizers.
        """

        update_learning_rate(
            self.actor.optimizer,
            ACTOR_LEARNING_RATE,
        )

        update_learning_rate(
            self.critic.optimizer,
            CRITIC_LEARNING_RATE,
        )

        self.logger.record(
            "train/learning_rate",
            CRITIC_LEARNING_RATE,
        )

        self.logger.record(
            "train/actor_learning_rate",
            ACTOR_LEARNING_RATE,
        )

        self.logger.record(
            "train/critic_learning_rate",
            CRITIC_LEARNING_RATE,
        )

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Follow SB3 TD3's update schedule with Huber regression for both critics.
        self.policy.set_training_mode(True)
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses, critic_losses = [], []
        for _ in range(gradient_steps):
            self._n_updates += 1
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )
            discounts = replay_data.discounts
            if discounts is None:
                discounts = self.gamma

            with torch.no_grad():
                noise = torch.empty_like(replay_data.actions).normal_(
                    0, self.target_policy_noise
                )
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = (
                    self.actor_target(replay_data.next_observations) + noise
                ).clamp(-1, 1)
                next_q_values = torch.cat(
                    self.critic_target(replay_data.next_observations, next_actions),
                    dim=1,
                ).min(dim=1, keepdim=True).values
                target_q_values = (
                    replay_data.rewards
                    + (1 - replay_data.dones) * discounts * next_q_values
                )

            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = sum(
                F.huber_loss(current_q, target_q_values, delta=HUBER_DELTA)
                for current_q in current_q_values
            )
            critic_loss_value = critic_loss.item()
            if not isfinite(critic_loss_value):
                raise FloatingPointError(f"Non-finite Critic loss at step {self.num_timesteps}")
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()
            critic_losses.append(critic_loss_value)

            if self._n_updates % self.policy_delay == 0:
                actor_actions = self.actor(replay_data.observations)
                actor_loss = -self.critic.q1_forward(
                    replay_data.observations, actor_actions
                ).mean()
                actor_loss_value = actor_loss.item()
                if not isfinite(actor_loss_value):
                    raise FloatingPointError(f"Non-finite Actor loss at step {self.num_timesteps}")
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()
                actor_losses.append(actor_loss_value)

                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
                polyak_update(self.critic_batch_norm_stats, self.critic_batch_norm_stats_target, 1.0)
                polyak_update(self.actor_batch_norm_stats, self.actor_batch_norm_stats_target, 1.0)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/target_updates", self._n_updates // self.policy_delay)
        if actor_losses:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))


# ============================================================
# 3. Main training procedure
# ============================================================

def main() -> None:

    torch.set_num_threads(1)
    set_random_seed(SEED)

    # --------------------------------------------------------
    # 3.1 Load official dataset split
    # --------------------------------------------------------

    bundle = load_dataset_bundle(DATA_ROOT)

    train_id = DIAGNOSTIC_VOYAGE_ID
    train_profile = bundle.train_profiles[bundle.train_ids.index(train_id)]

    validation_profiles = bundle.validation_profiles
    validation_ids = bundle.validation_ids

    train_voyage_steps = len(train_profile) - 1

    # --------------------------------------------------------
    # 3.2 Create independent output directory
    # --------------------------------------------------------

    run_stamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    run_dir = OUTPUT_ROOT / f"run_{run_stamp}"

    run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    log_dir = run_dir / "sb3_logs"
    tensorboard_dir = run_dir / "tensorboard"

    log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    tensorboard_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 3.3 Save training configuration
    # --------------------------------------------------------

    config = {
        "algorithm": "Stable-Baselines3 TD3",
        "action_mapping": ACTION_MAPPING,
        "learning_rate_mode": "separate_actor_critic",

        "seed": SEED,
        "device": DEVICE,

        "data_root": str(DATA_ROOT.resolve()),
        "split_manifest": bundle.manifest_path,
        "step_duration_s": DT_S,
        "train_voyage_sampling": "random_window_on_fixed_voyage",
        "train_voyage_ids": [train_id],
        "train_voyage_count": 1,
        "max_train_voyage_steps": train_voyage_steps,
        "train_window_steps": TRAIN_WINDOW_STEPS,
        "train_window_samples": TRAIN_WINDOW_STEPS + 1,
        "max_train_window_start_step": train_voyage_steps - TRAIN_WINDOW_STEPS,
        "diagnostic_voyage": train_id,

        "validation_ids": validation_ids,
        "validation_count": len(validation_ids),

        "max_episodes": MAX_EPISODES,

        # Keep the legacy field for compatibility.
        "learning_rate": CRITIC_LEARNING_RATE,

        "actor_learning_rate": ACTOR_LEARNING_RATE,
        "critic_learning_rate": CRITIC_LEARNING_RATE,

        "buffer_size": BUFFER_SIZE,
        "learning_starts": LEARNING_STARTS,
        "batch_size": BATCH_SIZE,

        "gamma": GAMMA,
        "tau": TAU,
        "target_update_interval_env_steps": TRAIN_FREQ[0] * POLICY_DELAY,

        "train_freq": TRAIN_FREQ,
        "gradient_steps": GRADIENT_STEPS,

        "n_steps": N_STEPS,
        "policy_delay": POLICY_DELAY,

        "action_noise_type": "gaussian",
        "action_noise_sigma": ACTION_NOISE_SIGMA,
        "target_policy_noise": TARGET_POLICY_NOISE,
        "target_noise_clip": TARGET_NOISE_CLIP,

        "net_arch": NET_ARCH,

        "critic_loss": "huber",
        "huber_delta": HUBER_DELTA,
        "optimizer": "Adam",
        "adam_betas": ADAM_BETAS,

        # New reward configuration
        "reward_version": REWARD_VERSION,
        "reward_alpha": 1.0,
        "soc_step_weight": SOC_STEP_WEIGHT,
        "soc_target": 0.5,
        "terminal_soc_penalty_enabled": False,
        "failure_penalty": FAILURE_PENALTY,

        "reward_plot_interval": REWARD_PLOT_INTERVAL,
        "train_diagnostic_interval": TRAIN_DIAGNOSTIC_INTERVAL,
        "validation_interval": VALIDATION_INTERVAL,
    }

    save_json(
        run_dir / "training_config.json",
        config,
    )

    # --------------------------------------------------------
    # 3.4 Sample training windows from the fixed voyage
    # --------------------------------------------------------

    def make_training_env():

        env = make_env(
            [train_profile],
            [train_id],
            train_window_steps=TRAIN_WINDOW_STEPS,
        )

        return Monitor(
            env,
            filename=str(run_dir / "monitor"),
            info_keywords=("window_start_step", "window_end_step"),
        )

    train_env = DummyVecEnv([
        make_training_env
    ])

    # --------------------------------------------------------
    # 3.5 Exploration noise
    # --------------------------------------------------------

    action_noise = NormalActionNoise(
        mean=np.zeros(
            1,
            dtype=np.float32,
        ),
        sigma=np.full(
            1,
            ACTION_NOISE_SIGMA,
            dtype=np.float32,
        ),
    )

    # --------------------------------------------------------
    # 3.6 Create TD3 model
    # --------------------------------------------------------

    model = SeparateLearningRateTD3(
        policy="MlpPolicy",
        env=train_env,

        # SB3 initial/default schedule.
        # The subclass separately assigns both optimizers.
        learning_rate=CRITIC_LEARNING_RATE,

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
            "optimizer_class": torch.optim.Adam,
            "optimizer_kwargs": {"betas": ADAM_BETAS},
        },

        seed=SEED,
        device=DEVICE,

        verbose=1,

        tensorboard_log=str(tensorboard_dir),
    )

    # --------------------------------------------------------
    # 3.7 Verify actual optimizer learning rates
    # --------------------------------------------------------

    actor_lr_actual = float(
        model.actor.optimizer.param_groups[0]["lr"]
    )

    critic_lr_actual = float(
        model.critic.optimizer.param_groups[0]["lr"]
    )

    if not np.isclose(
        actor_lr_actual,
        ACTOR_LEARNING_RATE,
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            "Actor学习率配置不正确。"
        )

    if not np.isclose(
        critic_lr_actual,
        CRITIC_LEARNING_RATE,
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            "Critic学习率配置不正确。"
        )

    # --------------------------------------------------------
    # 3.8 SB3 logger
    # --------------------------------------------------------

    model.set_logger(
        configure(
            str(log_dir),
            ["stdout", "csv", "tensorboard"],
        )
    )

    # --------------------------------------------------------
    # 3.9 Training callback
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
    # 3.10 Training
    # --------------------------------------------------------

    total_timesteps_upper_bound = MAX_EPISODES * TRAIN_WINDOW_STEPS

    print("=" * 72)
    print("Ship EMS TD3 training")
    print("=" * 72)

    print("Algorithm: SB3 TD3")
    print(f"Fixed training voyage: {train_id} ({train_voyage_steps} steps)")
    print(f"Random training window: {TRAIN_WINDOW_STEPS} steps")
    print(f"Training episodes: {MAX_EPISODES}")
    print(f"Step duration: {DT_S:g} s")

    print(
        f"Actor learning rate: "
        f"{actor_lr_actual}"
    )

    print(
        f"Critic learning rate: "
        f"{critic_lr_actual}"
    )

    print(
        f"SOC step penalty weight: "
        f"{SOC_STEP_WEIGHT}"
    )

    print("Terminal SOC penalty: disabled")

    print(f"Learning starts: {LEARNING_STARTS}")
    print(f"N-step return: {N_STEPS}")
    print(f"Train frequency: {TRAIN_FREQ}")
    print(f"Gradient steps: {GRADIENT_STEPS}")

    print(f"Batch size: {BATCH_SIZE}")
    print(f"Critic loss: Huber delta={HUBER_DELTA:g}")
    print(f"Adam betas: {model.critic.optimizer.param_groups[0]['betas']}")
    print(f"Gamma: {GAMMA}")
    print(f"Tau: {TAU}")
    print(f"Target network update interval: {TRAIN_FREQ[0] * POLICY_DELAY} environment steps")

    print(f"Action noise: Gaussian sigma={ACTION_NOISE_SIGMA}")
    print(f"Target policy noise: {TARGET_POLICY_NOISE}")
    print(f"Target noise clip: {TARGET_NOISE_CLIP}")
    print(f"Action mapping: {ACTION_MAPPING}")
    print(f"Reward version: {REWARD_VERSION}")
    print(f"Physical failure reward: {FAILURE_PENALTY:g}")
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

    except (FloatingPointError, ValueError):
        callback._save_training_tables()
        raise

    finally:

        train_env.close()

    # --------------------------------------------------------
    # 3.11 Save final model and summary
    # --------------------------------------------------------

    model.save(
        str(run_dir / "last_model")
    )

    completed_windows = sum(row["completed"] for row in callback.episode_rows)
    save_json(
        run_dir / "training_summary.json",
        {
            "episodes_completed": len(
                callback.episode_rows
            ),
            "completed_training_windows": completed_windows,
            "physical_failure_episodes": sum(
                row["failure"] for row in callback.episode_rows
            ),

            "total_timesteps": int(
                model.num_timesteps
            ),

            "critic_updates": int(
                getattr(model, "_n_updates", 0)
            ),
            "target_updates": model._n_updates // model.policy_delay,

            "actor_learning_rate": ACTOR_LEARNING_RATE,
            "critic_learning_rate": CRITIC_LEARNING_RATE,

            "soc_step_weight": SOC_STEP_WEIGHT,
            "terminal_soc_penalty_enabled": False,

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
    print(f"TD3 finished with {completed_windows}/{MAX_EPISODES} complete windows")

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
