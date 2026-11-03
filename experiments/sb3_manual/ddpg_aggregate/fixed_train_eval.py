"""Deterministic fixed-Train-voyage diagnostics, separate from Validation8/Test4."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import torch
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import check_for_correct_spaces

from .env import AggregateDDPGEnv
from .economic_reward import evaluation_identity
from .validate import DEFAULT_DATA_ROOT, STEP_FIELDS, step_accounting, _model_snapshot

FIXED_TRAIN_ID = "parent_005_seg01"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs/ddpg_fixed_parent_005_evaluation_linear_v2"


def _append_csv(path: Path, row: dict) -> None:
    new = not path.exists()
    with path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)


def run_fixed_train_evaluation(
    model_path: Path, data_root: Path = DEFAULT_DATA_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT, episode: int = 500,
    sample_id: str = FIXED_TRAIN_ID, device: str = "cpu", seed: int = 42,
    max_steps: int | None = None, output_dir: Path | None = None,
) -> dict:
    """Load a model copy; one full Train voyage unless explicitly smoke-limited."""
    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps must be positive")
    torch.set_num_threads(1)
    model_path = Path(model_path).resolve()
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    model = DDPG.load(str(model_path), device=device, action_noise=None)
    model.policy.set_training_mode(False)
    before_model = _model_snapshot(model)
    env = AggregateDDPGEnv(data_root, split="train", fixed_train_id=sample_id)
    try:
        check_for_correct_spaces(env, model.observation_space, model.action_space)
        reward_metadata = evaluation_identity(model, env.reward_identity)
        if output_dir is None:
            stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
            directory = Path(output_root).resolve() / f"fixed_train_{stamp}"
            directory.mkdir(parents=True, exist_ok=False)
        else:
            # Periodic checkpoints already own a unique directory. Avoid nesting
            # another timestamp under it, which exceeds Windows MAX_PATH.
            directory = Path(output_dir).resolve()
            directory.mkdir(parents=True, exist_ok=True)
        obs, reset_info = env.reset(seed=seed)
        if reset_info["sample_id"] != sample_id:
            raise RuntimeError("Fixed training voyage selection changed")
        initial_soc = env.physical.soc
        calls, reward_sum = 0, 0.0
        with (directory / "fixed_train_step_data.csv").open(
            "x", encoding="utf-8-sig", newline=""
        ) as handle, torch.inference_mode():
            writer = csv.DictWriter(handle, fieldnames=STEP_FIELDS)
            writer.writeheader()
            while True:
                before, start = env.physical, env.index
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                calls += 1
                reward_sum += float(reward)
                executed = bool(info["step_executed"])
                limited = bool(not (terminated or truncated) and max_steps is not None and calls >= max_steps)
                row = {key: info.get(key) for key in STEP_FIELDS}
                row.update(
                    episode=episode, step_index=calls, interval_start_s=start,
                    interval_end_s=env.index if executed else None,
                    step_executed=executed, actor_action=float(np.asarray(action).item()),
                    soc_before=before.soc, reward=float(reward), terminated=bool(terminated),
                    truncated=bool(truncated), evaluation_truncated=limited,
                    terminal_reason="fixed_eval_step_limit" if limited else info.get("terminal_reason"),
                    **step_accounting(env, before, info),
                )
                writer.writerow(row)
                if terminated or truncated or limited:
                    summary = dict(info["vessel_episode"]) if not limited else env._episode_summary(
                        False, "fixed_eval_step_limit"
                    )
                    summary.update(
                        episode=episode, episode_reward=reward_sum,
                        mean_reward_per_step=reward_sum / calls, episode_length=calls,
                        soc_initial=initial_soc, soc_delta=env.physical.soc - initial_soc,
                        evaluation_truncated=limited,
                        full_voyage_economic_cost_cny=summary["total_cost_cny"] if summary["completed"] else None,
                    )
                    break
        after_model = _model_snapshot(model)
        integrity = {
            "parameters_unchanged": before_model["policy_sha256"] == after_model["policy_sha256"],
            "optimizer_state_unchanged": before_model["optimizer_sha256"] == after_model["optimizer_sha256"],
            "counters_unchanged": all(before_model[key] == after_model[key] for key in ("num_timesteps", "n_updates")),
            "replay_unchanged": before_model["replay"] == after_model["replay"],
            "no_gradients": before_model["no_gradients"] and after_model["no_gradients"],
            "model_file_unchanged": model_hash == hashlib.sha256(model_path.read_bytes()).hexdigest(),
        }
        if not all(integrity.values()):
            raise RuntimeError(f"Fixed evaluation modified model state: {integrity}")
        metrics = {
            "output_dir": str(directory), "model_path": str(model_path), "model_sha256": model_hash,
            "split": "train", "evaluation_scope": "fixed_train_learning_diagnostic_not_generalization",
            "deterministic": True, "action_noise": None, **reward_metadata,
            "summary": summary, "integrity": integrity,
        }
        _append_csv(directory / "fixed_train_summary.csv", summary)
        (directory / "fixed_train_metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
        )
        print(f"Fixed Train diagnostic: {sample_id}, {summary['terminal_reason']}, "
              f"coverage={summary['coverage']:.4f}; {directory}", flush=True)
        return metrics
    finally:
        env.close()


class FixedTrainEvaluationCallback(BaseCallback):
    """Every 20 real episode endings: checkpoint, then isolated CPU inference.

    Must precede EpisodeValidationCallback/StopTrainingOnMaxEpisodes in the list,
    so diagnostics at episode 500 run before the stopping callbacks return False.
    """

    def __init__(self, every_episodes: int = 20, output_root: str | Path | None = None, verbose: int = 1):
        super().__init__(verbose=verbose)
        if every_episodes < 1:
            raise ValueError("every_episodes must be positive")
        self.every_episodes = every_episodes
        self.output_root = None if output_root is None else Path(output_root)
        self.completed_episodes = 0

    def _on_training_start(self) -> None:
        ids = self.training_env.get_attr("fixed_train_id")
        if len(ids) != 1 or ids[0] != FIXED_TRAIN_ID:
            raise ValueError("Fixed diagnostic requires one parent_005_seg01 training environment")
        self.completed_episodes = int(getattr(self.model, "_ddpg_completed_episodes", self.model._episode_num))
        self.data_root = Path(self.training_env.get_attr("data_root")[0]).resolve()
        if self.output_root is None:
            writer = self.training_env.get_attr("results_writer")[0]
            self.directory = Path(writer.file_handler.name).resolve().parent / "fixed_train_evaluation"
        else:
            self.directory = self.output_root.resolve()
        self.directory.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals.get("infos", [])):
            if not done:
                continue
            self.completed_episodes += 1
            self.model._ddpg_completed_episodes = self.completed_episodes
            summary = info["vessel_episode"]
            monitor = info["episode"]
            _append_csv(self.directory / "training_episodes.csv", {
                "episode": self.completed_episodes,
                "environment_steps": int(self.model.num_timesteps),
                "adam_updates": int(self.model._n_updates),
                "sample_id": summary["sample_id"], "episode_reward": summary["reward_sum"],
                "mean_reward_per_step": summary["reward_sum"] / monitor["l"],
                "episode_length": monitor["l"], "executed_steps": summary["executed_steps"],
                "total_steps": summary["total_steps"], "completed": summary["completed"],
                "coverage": summary["coverage"], "terminal_reason": summary["terminal_reason"],
            })
            if self.completed_episodes % self.every_episodes == 0:
                self._save_and_evaluate()
        return True

    def _save_and_evaluate(self) -> None:
        episode = self.completed_episodes
        node = self.directory / f"episode_{episode:04d}"
        node.mkdir(exist_ok=False)
        checkpoint = node / "model.zip"
        self.model.save(str(checkpoint))
        (node / "checkpoint_metadata.json").write_text(json.dumps({
            "completed_episodes": episode, "environment_steps": int(self.model.num_timesteps),
            "adam_updates": int(self.model._n_updates),
            "clock_phase": "after_env_step_before_replay_insert_and_rollout_adam_update",
            "sample_id": FIXED_TRAIN_ID, "reward_identity": self.model._ddpg_reward_identity,
        }, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        child_env = os.environ.copy()
        paths = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src")]
        if child_env.get("PYTHONPATH"):
            paths.append(child_env["PYTHONPATH"])
        child_env["PYTHONPATH"] = os.pathsep.join(paths)
        subprocess.run([
            sys.executable, "-u", "-m", "experiments.sb3_manual.ddpg_aggregate.fixed_train_eval",
            "--model-path", str(checkpoint), "--data-root", str(self.data_root),
            "--output-dir", str(node), "--episode", str(episode),
            "--sample-id", FIXED_TRAIN_ID, "--device", "cpu",
            "--seed", str(self.model.seed if self.model.seed is not None else 42),
        ], cwd=PROJECT_ROOT, env=child_env, check=True)
        metrics_path = node / "fixed_train_metrics.json"
        if not metrics_path.is_file():
            raise RuntimeError("Fixed evaluation must produce exactly one metrics file")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        summary = metrics["summary"]
        if (metrics["split"] != "train" or summary["sample_id"] != FIXED_TRAIN_ID
                or summary["evaluation_truncated"] or not metrics["deterministic"]
                or metrics["action_noise"] is not None or not all(metrics["integrity"].values())):
            raise RuntimeError("Fixed evaluation did not follow the deterministic full-voyage protocol")
        _append_csv(self.directory / "fixed_train_evaluation.csv", summary)
        path = self.directory / "fixed_train_evaluation.json"
        history = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        history.append(metrics)
        path.write_text(json.dumps(history, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        for key in ("mean_reward_per_step", "episode_reward", "coverage", "completed", "total_cost_cny", "soc_final"):
            self.logger.record(f"fixed_train/{key}", summary[key])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, help="Exact periodic checkpoint directory; existing results are not overwritten")
    parser.add_argument("--episode", type=int, default=500)
    parser.add_argument("--sample-id", default=FIXED_TRAIN_ID)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, help="Bounded smoke only; marks a partial diagnostic")
    return run_fixed_train_evaluation(**vars(parser.parse_args(argv)))


if __name__ == "__main__":
    main()
