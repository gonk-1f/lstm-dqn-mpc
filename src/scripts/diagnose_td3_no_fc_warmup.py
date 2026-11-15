"""Collect the native SB3 pre-learning rollouts for the no-FC-dynamics experiment."""

from datetime import datetime

import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from td3_config import (
    DIAGNOSTIC_VOYAGE_ID, SEED, TRAIN_WINDOW_STEPS, load_dataset_bundle, save_json, write_csv,
)
from td3_reference_config import DATA_ROOT, TD3_KWARGS
from td3_reference_model_config import (
    MODEL_PARAMETERS, MODEL_VERSION, OUTPUT_ROOT, make_env, observation_scales,
)
from train_td3_reference_model import TRACE_FIELDS, build_model, runtime_versions


class WarmupTrace(BaseCallback):
    def __init__(self, budget: int):
        super().__init__()
        self.budget = budget
        self.steps: list[dict] = []
        self.episodes: list[dict] = []
        self.episode_step = 0
        self.episode_reward = 0.0

    def _on_step(self) -> bool:
        info = self.locals["infos"][0]
        done = bool(self.locals["dones"][0])
        reward = float(self.locals["rewards"][0])
        self.episode_step += 1
        self.episode_reward += reward
        row = {
            "episode": len(self.episodes) + 1, "step": self.episode_step,
            "global_step": self.num_timesteps, "sample_id": info["voyage_id"],
            "window_start_step": info["window_start_step"],
            "window_end_step": info["window_end_step"],
            **{key: info.get(key) for key in TRACE_FIELDS},
            "reward": reward, "terminated": info["terminated"], "truncated": info["truncated"],
        }
        self.steps.append(row)
        if done or self.num_timesteps == self.budget:
            self.episodes.append({
                "episode": row["episode"], "window_start_step": row["window_start_step"],
                "window_end_step": row["window_end_step"],
                "attempted_steps": self.episode_step,
                "executed_steps": self.episode_step - int(info["failure"]),
                "initial_soc": self.steps[-self.episode_step]["soc_before"], "final_soc": info["soc"],
                "total_reward": self.episode_reward, "failure": info["failure"],
                "terminal_reason": info["terminal_reason"] or "diagnostic_step_budget",
            })
            self.episode_step = 0
            self.episode_reward = 0.0
        return not info["failure"] and (done or self.num_timesteps < self.budget)


def main() -> None:
    versions = runtime_versions()
    if (versions["stable-baselines3"], versions["gymnasium"]) != ("2.2.1", "0.29.1"):
        raise RuntimeError("Use the isolated environment from requirements-td3-reference.txt.")
    torch.set_num_threads(1)
    set_random_seed(SEED, using_cuda=torch.cuda.is_available())
    bundle = load_dataset_bundle(DATA_ROOT)
    train_id = DIAGNOSTIC_VOYAGE_ID
    profile = bundle.train_profiles[bundle.train_ids.index(train_id)]
    load_scale, delta_scale = observation_scales(bundle.train_profiles)
    env = Monitor(make_env(
        [profile], [train_id], load_scale_kw=load_scale, load_delta_scale_kw=delta_scale,
        train_window_steps=TRAIN_WINDOW_STEPS,
    ))
    run_dir = OUTPUT_ROOT / datetime.now().strftime("warmup_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        model = build_model(env)
        budget = min(25_000, model.learning_starts)
        model.set_logger(configure(str(run_dir), []))
        trace = WarmupTrace(budget)
        # Use the same rollout setup as learn(), then collect without calling train().
        _, callback = model._setup_learn(budget, trace)
        callback.on_training_start(locals(), globals())
        vec_env = model.get_env()
        while model.num_timesteps < budget:
            rollout = model.collect_rollouts(
                vec_env, callback, model.train_freq, model.replay_buffer,
                action_noise=model.action_noise, learning_starts=model.learning_starts,
            )
            if not rollout.continue_training:
                break
        callback.on_training_end()
        write_csv(run_dir / "warmup_trajectory.csv", trace.steps)
        write_csv(run_dir / "warmup_episodes.csv", trace.episodes)
        failure = trace.steps[-1] if trace.steps and trace.steps[-1]["failure"] else None
        executed = [row for row in trace.steps if not row["failure"]]
        save_json(run_dir / "warmup_summary.json", {
            "model_version": MODEL_VERSION, "model_parameters": MODEL_PARAMETERS,
            "versions": versions, "device": str(model.device), "seed": SEED,
            "voyage_id": train_id, "data_root": str(DATA_ROOT.resolve()), "td3_parameters": TD3_KWARGS,
            "step_budget": budget, "attempted_steps": model.num_timesteps,
            "executed_steps": len(executed),
            "completed_windows": sum(row["terminal_reason"] == "training_window_end" for row in trace.episodes),
            "replay_size": model.replay_buffer.size(), "gradient_updates": model._n_updates,
            "largest_normal_step_cost": max((-row["reward"] for row in executed), default=None),
            "fc_zero_output_steps": sum(row["fc_actual_kw"] == 0.0 for row in executed),
            "first_failure": failure,
        })
        print(f"Warm-up diagnostic complete after {model.num_timesteps} steps: {run_dir}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
