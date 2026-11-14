"""Run native TD3 on the project's existing finite ship EMS windows."""

from __future__ import annotations

from datetime import datetime
from importlib.metadata import version
from pathlib import Path
import sys

import numpy as np
import torch
from stable_baselines3 import TD3
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.utils import set_random_seed

from callbacks import ShipEMSTrainingCallback
from td3_config import (
    ACTION_MAPPING,
    DEVICE,
    DIAGNOSTIC_VOYAGE_ID,
    DT_S,
    REWARD_VERSION,
    SEED,
    SOC_STEP_WEIGHT,
    TRAIN_WINDOW_STEPS,
    VALIDATION_INTERVAL,
    evaluate_profiles,
    load_dataset_bundle,
    make_env,
    save_json,
    write_csv,
)
from td3_reference_config import (
    ACTION_NOISE_SIGMA,
    DATA_ROOT,
    DIAGNOSTIC_INTERVAL,
    EPISODES,
    OUTPUT_ROOT,
    TD3_KWARGS,
    TOTAL_TIMESTEPS,
)


REWARD_COST_FIELDS = (
    "hydrogen_cost_rmb",
    "fc_degradation_cost_rmb",
    "battery_degradation_cost_rmb",
    "step_battery_equivalent_hydrogen_cost_rmb",
    "step_soc_penalty_unscaled",
)
TRACE_FIELDS = (
    "sample_id", "step", "load_kw", "actor_action_fraction", "requested_fc_kw",
    "fc_actual_kw", "battery_actual_kw", "soc_before", "soc", "delta_fc_kw",
    "power_balance_error_kw", *REWARD_COST_FIELDS, "step_reward_cost_rmb",
    "reward", "terminal_reason",
)


def runtime_versions() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        **{name: version(name) for name in ("stable-baselines3", "gymnasium", "torch", "numpy")},
    }


def build_model(env, *, seed: int = SEED, device: str = DEVICE) -> TD3:
    return TD3(
        "MlpPolicy",
        env,
        **TD3_KWARGS,
        action_noise=NormalActionNoise(
            np.zeros(env.action_space.shape),
            np.full(env.action_space.shape, ACTION_NOISE_SIGMA),
        ),
        seed=seed,
        device=device,
        verbose=1,
    )


class ReferenceDiagnostics(ShipEMSTrainingCallback):
    """Evaluate the saved policy after SB3 finishes each scheduled update batch."""

    def _on_training_start(self) -> None:
        super()._on_training_start()
        self._episode_reward_costs = dict.fromkeys(REWARD_COST_FIELDS, 0.0)

    def _on_step(self) -> bool:
        info = self.locals["infos"][0]
        for key in REWARD_COST_FIELDS:
            if info.get(key) is not None:
                self._episode_reward_costs[key] += info[key]
        keep_training = super()._on_step()
        if self.locals["dones"][0]:
            self.episode_rows[-1].update({
                f"sum_{key}": value for key, value in self._episode_reward_costs.items()
            })
            self._episode_reward_costs = dict.fromkeys(REWARD_COST_FIELDS, 0.0)
        return keep_training

    def _run_fixed_train_diagnostic(self, episode_number: int) -> None:
        was_training = self.model.policy.training
        try:
            results, steps = evaluate_profiles(
                self.model, [self.train_profile], [self.train_id],
                seed=self.seed, collect_step_rows=True,
            )
        finally:
            self.model.policy.set_training_mode(was_training)

        result = results[0]
        result["training_episode"] = episode_number
        result["critic_updates"] = self.model._n_updates
        result["critic_loss"] = self.logger.name_to_value.get("train/critic_loss")
        result.update(self._replay_q_metrics())
        actions = np.array([row["actor_action_fraction"] for row in steps])
        counts, edges = np.histogram(actions, bins=20, range=(0.0, 1.0))
        result["actor_action_mean"] = float(actions.mean())
        result["actor_action_std"] = float(actions.std())
        result["actor_action_saturation_fraction"] = float(np.mean(np.abs(2 * actions - 1) >= 0.99))
        for key in REWARD_COST_FIELDS:
            result[f"sum_{key}"] = sum(row[key] for row in steps if row.get(key) is not None)

        self.train_diagnostic_rows.append(result)
        write_csv(
            self.output_dir / f"power_episode_{episode_number:04d}.csv",
            [{key: row.get(key) for key in TRACE_FIELDS} for row in steps],
        )
        write_csv(
            self.output_dir / f"actor_actions_episode_{episode_number:04d}.csv",
            [
                {"action_lower": lo, "action_upper": hi, "count": int(count)}
                for lo, hi, count in zip(edges[:-1], edges[1:], counts)
            ],
        )
        self._save_power_plot(episode_number, result, steps)
        print(
            f"[Fixed Train] episode={episode_number}, completed={result['completed']}, "
            f"coverage={result['coverage']:.1%}, final SOC={result['final_soc']}, "
            f"Critic loss={result['critic_loss']}"
        )

    def _replay_q_metrics(self) -> dict[str, float]:
        replay = self.model.replay_buffer
        width = min(replay.size(), TRAIN_WINDOW_STEPS)
        # Fixed positions preserve the RNG state used by SB3's replay sampling.
        offsets = np.linspace(-width, -1, min(width, 256), dtype=int)
        indices = (replay.pos + offsets) % replay.buffer_size
        observations = replay.to_torch(replay.observations[indices, 0])
        actions = replay.to_torch(replay.actions[indices, 0])
        with torch.no_grad():
            q1, q2 = self.model.critic(observations, actions)
        return {
            "replay_q1_mean": q1.mean().item(),
            "replay_q1_std": q1.std(unbiased=False).item(),
            "replay_q2_mean": q2.mean().item(),
            "replay_q2_std": q2.std(unbiased=False).item(),
        }

    def _save_updated_checkpoint(self) -> None:
        episode = len(self.episode_rows)
        if (
            episode == 0 or not self.episode_rows[-1]["completed"]
            or self.episode_rows[-1]["global_timesteps"] != self.num_timesteps
            or (self.train_diagnostic_rows and self.train_diagnostic_rows[-1]["training_episode"] == episode)
        ):
            return
        if episode % DIAGNOSTIC_INTERVAL == 0:
            row = self.episode_rows[-1]
            self.logger.record("train/mean_reward_per_step", row["mean_reward_per_step"])
            self.logger.record("train/mean_reward_per_step_ma10", row["mean_reward_per_step_ma10"])
            self.logger.record("time/episodes", episode, exclude="tensorboard")
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.model.save(self.output_dir / f"checkpoint_episode_{episode:04d}")
            self.model.save(self.output_dir / "last_model")
            self._save_reward_plot(episode)
            self._run_fixed_train_diagnostic(episode)
            self.logger.dump(self.num_timesteps)
        if episode % VALIDATION_INTERVAL == 0:
            self._run_validation(episode)
        if episode % DIAGNOSTIC_INTERVAL == 0 or episode % VALIDATION_INTERVAL == 0:
            self._save_training_tables()

    def _on_rollout_start(self) -> None:
        self._save_updated_checkpoint()

    def _on_training_end(self) -> None:
        # A requested stop can interrupt SB3 before it stores a window's last transition.
        if not self.stop_file.exists():
            self._save_updated_checkpoint()
        super()._on_training_end()


def main() -> None:
    versions = runtime_versions()
    if versions["stable-baselines3"] != "2.2.1" or versions["gymnasium"] != "0.29.1":
        raise RuntimeError("Use the isolated environment from requirements-td3-reference.txt.")

    torch.set_num_threads(1)
    set_random_seed(SEED, using_cuda=torch.cuda.is_available())
    bundle = load_dataset_bundle(DATA_ROOT)
    train_id = DIAGNOSTIC_VOYAGE_ID
    train_profile = bundle.train_profiles[bundle.train_ids.index(train_id)]
    run_dir = OUTPUT_ROOT / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    env = Monitor(
        make_env([train_profile], [train_id], train_window_steps=TRAIN_WINDOW_STEPS),
        filename=str(run_dir / "monitor"),
        info_keywords=("window_start_step", "window_end_step"),
    )
    try:
        model = build_model(env)
        model.set_logger(configure(str(run_dir / "sb3_logs"), ["stdout", "csv"]))
        callback = ReferenceDiagnostics(
            train_profile, train_id, bundle.validation_profiles, bundle.validation_ids,
            run_dir, max_episodes=EPISODES,
            # Periodic saves and evaluations run after SB3 finishes the updates.
            reward_plot_interval=EPISODES + 1,
            train_diagnostic_interval=EPISODES + 1, validation_interval=EPISODES + 1,
            seed=SEED,
        )
        save_json(run_dir / "training_config.json", {
            **TD3_KWARGS,
            "algorithm": "native SB3 TD3",
            "versions": versions,
            "device": str(model.device),
            "torch_cuda": torch.version.cuda,
            "data_root": str(DATA_ROOT.resolve()),
            "split_manifest": bundle.manifest_path,
            "seed": SEED,
            "train_voyage_ids": [train_id],
            "train_voyage_sampling": "random_window_on_fixed_voyage",
            "train_window_steps": TRAIN_WINDOW_STEPS,
            "step_duration_s": DT_S,
            "max_episodes": EPISODES,
            "total_timesteps": TOTAL_TIMESTEPS,
            "action_noise_type": "gaussian",
            "action_noise_sigma": ACTION_NOISE_SIGMA,
            "action_mapping": ACTION_MAPPING,
            "reward_version": REWARD_VERSION,
            "soc_step_weight": SOC_STEP_WEIGHT,
            "critic_loss": "mse",
            "diagnostic_interval": DIAGNOSTIC_INTERVAL,
            "validation_interval": VALIDATION_INTERVAL,
            "validation_ids": bundle.validation_ids,
            "q_diagnostics": "256 evenly spaced stored observation-action pairs from the latest window",
        })
        print(f"Native TD3 {versions['stable-baselines3']} on {model.device}. Output: {run_dir}")
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback, log_interval=5)
        save_json(run_dir / "training_summary.json", {
            "completed_training_windows": sum(row["completed"] for row in callback.episode_rows),
            "physical_failure_episodes": sum(row["failure"] for row in callback.episode_rows),
            "total_timesteps": model.num_timesteps,
            "critic_updates": model._n_updates,
            "actor_and_target_updates": model._n_updates // model.policy_delay,
            "best_validation_episode": callback.best_episode,
            "last_model": str(run_dir / "last_model.zip"),
        })
    finally:
        env.close()


if __name__ == "__main__":
    main()
