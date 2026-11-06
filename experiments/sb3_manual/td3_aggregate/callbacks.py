"""TD3 diagnostics and isolated evaluation, without changing SB3's update loop.

The rollout callback runs before the current transition enters replay. Checkpoint
metadata makes that phase explicit. Evaluations load frozen model files in a
separate CPU process; no evaluator or validation environment lives in training.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ..ddpg_aggregate.economic_reward import FC_DEGRADATION_COST_FIELDS

FIXED_TRAIN_ID = "parent_005_seg01"
FIXED_TRAIN_STEPS = 9258
REWARD_SCALE = 0.0001
ENV_ID = "AggregateShipTD3-v0"
ECONOMIC_FIELDS = (
    "hydrogen_mass_kg", "h2_cost_cny", "c_bat_eq_cny",
    "battery_equivalent_hydrogen_kg", "battery_degradation_cost_cny",
    "fuel_cell_degradation_cost_cny", *FC_DEGRADATION_COST_FIELDS,
    "shore_cost_cny", "physical_economic_cost_cny", "total_cost_cny",
    "reward_economic_cost_cny", "soc_deviation_penalty_cny", "reward_total_cost_cny",
)


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def _append_csv(path: Path, row: dict[str, Any]) -> None:
    exists = path.exists()
    fields = list(row)
    if exists:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            if next(csv.reader(handle), None) != fields:
                raise ValueError(f"Cannot append an incompatible CSV schema: {path}")
    with path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


class TD3TrainingCallback(BaseCallback):
    """Count real endings, log diagnostics and evaluate every 20/100 episodes.

    Place before ``StopTrainingOnMaxEpisodes``. The native stopping callback
    counts only the current learn call; this callback additionally enforces the
    saved cumulative budget for ``--trained-agent`` runs. Neither callback edits
    replay or the native TD3 optimizer/target update schedule.

    Use Zoo ``--log-interval -2`` to let this callback flush at rollout end every
    1000 steps and at episode endings. Native ``train/*`` entries are retained;
    they describe the latest completed update, before this rollout's update.
    """

    def __init__(
        self, fixed_eval_every_episodes: int = 20,
        validation_every_episodes: int = 100, max_episodes: int = 500,
        log_every_steps: int = 1000, output_root: str | Path | None = None,
        verbose: int = 1,
    ):
        super().__init__(verbose=verbose)
        for name, value in (
            ("fixed_eval_every_episodes", fixed_eval_every_episodes),
            ("validation_every_episodes", validation_every_episodes),
            ("max_episodes", max_episodes), ("log_every_steps", log_every_steps),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.fixed_eval_every_episodes = fixed_eval_every_episodes
        self.validation_every_episodes = validation_every_episodes
        self.max_episodes = max_episodes
        self.log_every_steps = log_every_steps
        self.output_root = None if output_root is None else Path(output_root)
        self.completed_episodes = 0
        self.successful_episodes = 0
        self._window_sums: dict[str, float] = defaultdict(float)
        self._window_counts: dict[str, int] = defaultdict(int)
        self._episode_sums: dict[str, float] = defaultdict(float)
        self._episode_counts: dict[str, int] = defaultdict(int)
        self._observations: list[np.ndarray] = []
        self._pending_dump = False

    def _on_training_start(self) -> None:
        if self.training_env.num_envs != 1:
            raise ValueError("TD3 fixed-voyage diagnostics require n_envs=1")
        if self.training_env.get_attr("fixed_train_id") != [FIXED_TRAIN_ID]:
            raise ValueError("TD3 training must use parent_005_seg01")
        if self.training_env.get_attr("split") != ["train"]:
            raise ValueError("Validation/Test environments cannot enter training")
        identity = deepcopy(self.training_env.get_attr("reward_identity")[0])
        if identity.get("algorithm") != "TD3" or identity.get("reward_scale") != REWARD_SCALE:
            raise ValueError("Training requires the TD3 economic reward wrapper at scale 0.0001")
        if getattr(self.model, "_ddpg_reward_identity", None) is not None:
            raise ValueError("A DDPG checkpoint cannot be relabeled as scaled TD3")
        saved_identity = getattr(self.model, "_td3_reward_identity", None)
        if saved_identity is not None:
            if saved_identity != identity or getattr(self.model, "reward_scale", None) != REWARD_SCALE:
                raise ValueError("Checkpoint TD3 reward identity/scale differs from this environment")
            if not hasattr(self.model, "_td3_completed_episodes"):
                raise ValueError("TD3 checkpoint lacks its cumulative episode counter")
        elif (
            self.model.num_timesteps > 0 or self.model._n_updates > 0
            or (self.model.replay_buffer is not None and self.model.replay_buffer.size() > 0)
        ):
            raise ValueError("Cannot attach TD3 reward metadata to an unmarked trained model")
        count = getattr(self.model, "_td3_completed_episodes", 0)
        if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 0:
            raise ValueError("Checkpoint episode counter must be a nonnegative integer")
        self.completed_episodes = int(count)
        if self.completed_episodes >= self.max_episodes:
            raise ValueError("This checkpoint has already reached the cumulative episode budget")
        successes = getattr(self.model, "_td3_successful_episodes", 0 if count == 0 else None)
        if (
            isinstance(successes, bool) or not isinstance(successes, (int, np.integer))
            or not 0 <= successes <= self.completed_episodes
        ):
            raise ValueError("Checkpoint needs its valid cumulative successful-episode counter")
        self.successful_episodes = int(successes)
        self.model._td3_reward_identity = identity
        self.model.reward_scale = REWARD_SCALE
        self.model._td3_completed_episodes = self.completed_episodes
        self.model._td3_successful_episodes = self.successful_episodes
        self.data_root = Path(self.training_env.get_attr("data_root")[0]).resolve()
        if self.output_root is None:
            writer = self.training_env.get_attr("results_writer")[0]
            if writer is None:
                raise ValueError("TD3 diagnostics require a Monitor output file or output_root")
            self.directory = Path(writer.file_handler.name).resolve().parent
        else:
            self.directory = self.output_root.resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        # Zoo has no YAML log-format option. Configure SB3's native Logger here.
        # A new Zoo run has a unique monitor directory; explicit reuse also keeps
        # the previous progress.csv and TensorBoard files intact.
        log_directory = self.directory
        if (log_directory / "progress.csv").exists():
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
            log_directory = self.directory / f"resume_logs_{stamp}"
        previous_logger = self.logger
        logger = configure(str(log_directory), ["stdout", "csv", "tensorboard"])
        logger.name_to_value.update(previous_logger.name_to_value)
        logger.name_to_count.update(previous_logger.name_to_count)
        logger.name_to_excluded.update(previous_logger.name_to_excluded)
        self.model.set_logger(logger)
        previous_logger.close()
        self._last_dump_step = int(self.num_timesteps)
        self._window_sums.clear()
        self._window_counts.clear()
        self._episode_sums.clear()
        self._episode_counts.clear()
        self._observations.clear()
        self._pending_dump = False
        _write_json(log_directory / "td3_training_metadata.json", {
            "algorithm": "stable_baselines3.TD3", "env_id": ENV_ID,
            "reward_scale": REWARD_SCALE, "reward_identity": identity,
            "starting_completed_episodes": self.completed_episodes,
            "starting_successful_episodes": self.successful_episodes,
            "max_episodes": self.max_episodes, "fixed_train_id": FIXED_TRAIN_ID,
            "fixed_train_steps": FIXED_TRAIN_STEPS,
            "fixed_eval_every_episodes": self.fixed_eval_every_episodes,
            "validation_every_episodes": self.validation_every_episodes,
            "log_every_steps": self.log_every_steps,
            "logger_formats": ["stdout", "csv", "tensorboard"],
            "network_diagnostics": "frozen current actor/critics on recent observed training states; no replay sampling",
            "actor_saturation_threshold": 0.99,
            "train_metric_clock": "latest completed SB3 update before this rollout update",
            "checkpoint_clock": "after_env_step_before_replay_insert_and_rollout_update",
            "checkpoint_replay_saved": False,
            "raw_reward_units": "negative raw CNY reward cost, except fixed physical failure override",
            "algorithm_reward_units": "raw_reward * 0.0001",
        })
        self.logger.record("reward/reward_scale", REWARD_SCALE)
        self.logger.record("time/completed_episodes", self.completed_episodes)

    def _accumulate(self, name: str, value: float) -> None:
        for sums, counts in (
            (self._window_sums, self._window_counts),
            (self._episode_sums, self._episode_counts),
        ):
            sums[name] += float(value)
            counts[name] += 1

    def _observe_step(self, info: dict[str, Any]) -> None:
        obs = self.model._last_obs
        if obs is not None:
            self._observations.append(np.array(obs, dtype=np.float32, copy=True))
            # Bound memory even if a caller uses a multi-step rollout.
            if len(self._observations) > self.log_every_steps:
                del self._observations[0]
        for name in ("raw_reward", "scaled_reward"):
            self._accumulate(name, info[name])
        self._accumulate("requested_fc_kw", info["requested_fc_kw"])
        self._accumulate("requested_action", info["requested_fc_kw"] / 300.0 - 1.0)
        if info["step_executed"]:
            for name in ("fc_actual_kw", "battery_actual_kw", "load_kw"):
                self._accumulate(name, info[name])
            self._accumulate("executed_action", info["fc_actual_kw"] / 300.0 - 1.0)
            self._accumulate("projected", float(info.get("projected", False)))
        for name in ECONOMIC_FIELDS:
            if name in info:
                self._accumulate(name, info[name])
        self.logger.record("vessel/soc", info["soc"])

    def _on_step(self) -> bool:
        infos, dones = self.locals["infos"], self.locals["dones"]
        if len(infos) != 1 or len(dones) != 1:
            raise ValueError("TD3 diagnostics require exactly one training environment")
        info = infos[0]
        self._observe_step(info)
        if bool(dones[0]):
            self.completed_episodes += 1
            self.model._td3_completed_episodes = self.completed_episodes
            self._record_episode(info["vessel_episode"])
            if self.completed_episodes % self.fixed_eval_every_episodes == 0:
                self._evaluate("fixed_train")
            if self.completed_episodes % self.validation_every_episodes == 0:
                self._evaluate("validation")
            self._pending_dump = True
        if self.num_timesteps - self._last_dump_step >= self.log_every_steps:
            self._pending_dump = True
        return self.completed_episodes < self.max_episodes

    def _record_episode(self, summary: dict[str, Any]) -> None:
        if summary["split"] != "train" or summary["sample_id"] != FIXED_TRAIN_ID:
            raise ValueError("Only the fixed Train35 voyage may produce TD3 training episodes")
        raw = float(summary["reward_sum"])
        scaled = float(summary["algorithm_reward_sum"])
        if not math.isclose(scaled, raw * REWARD_SCALE, rel_tol=1e-10, abs_tol=1e-9):
            raise ValueError("Episode raw/algorithm reward sums violate TD3 reward scaling")
        self.successful_episodes += int(bool(summary["completed"]))
        self.model._td3_successful_episodes = self.successful_episodes
        counts, sums = self._episode_counts, self._episode_sums
        mean = lambda key: sums[key] / counts[key] if counts[key] else None
        row = {
            "episode": self.completed_episodes,
            "environment_steps": int(self.num_timesteps),
            "critic_updates": int(self.model._n_updates),
            "sample_id": summary["sample_id"], "split": summary["split"],
            "episode_length": counts["raw_reward"],
            "executed_steps": summary["executed_steps"], "total_steps": summary["total_steps"],
            "completed": summary["completed"], "coverage": summary["coverage"],
            "successful_episodes": self.successful_episodes,
            "cumulative_completion_rate": self.successful_episodes / self.completed_episodes,
            "terminal_reason": summary["terminal_reason"],
            "raw_reward_sum": raw, "algorithm_reward_sum": scaled,
            "reward_scale": REWARD_SCALE,
            "soc_min": summary["soc_min"], "soc_final": summary["soc_final"],
            "mean_requested_fc_kw": mean("requested_fc_kw"),
            "mean_fc_kw": mean("fc_actual_kw"),
            "mean_battery_kw": mean("battery_actual_kw"), "mean_load_kw": mean("load_kw"),
            "power_projection_fraction": summary["power_projection_fraction"],
            **{name: summary.get(name) for name in ECONOMIC_FIELDS},
        }
        _append_csv(self.directory / "training_episodes.csv", row)
        for name, value in row.items():
            if name not in {"sample_id", "split", "terminal_reason"} and value is not None:
                self.logger.record(f"episode/{name}", value)
        self.logger.record("episode/terminal_reason", summary["terminal_reason"])
        self._episode_sums.clear()
        self._episode_counts.clear()

    def _network_diagnostics(self) -> None:
        if not self._observations:
            return
        observations = np.concatenate(self._observations, axis=0)
        actor, critic = self.model.actor, self.model.critic
        modules = {module: module.training for network in (actor, critic) for module in network.modules()}
        device = torch.device(self.model.device)
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        try:
            actor.eval()
            critic.eval()
            # No replay draw, action noise, predict(), optimizer, target network or
            # gradient writes. Preserve CPU/CUDA RNG and every module's mode.
            with torch.random.fork_rng(devices=devices), torch.inference_mode():
                obs_tensor, _ = self.model.policy.obs_to_tensor(observations)
                actions = actor(obs_tensor)
                q_values = critic(obs_tensor, actions)
                self.logger.record("actor/action_mean", float(actions.mean().item()))
                self.logger.record("actor/action_std", float(actions.std(unbiased=False).item()))
                self.logger.record("actor/action_saturation_fraction", float((actions.abs() >= 0.99).float().mean().item()))
                self.logger.record("actor/observed_state_count", int(len(observations)))
                for index, q in enumerate(q_values, start=1):
                    self.logger.record(f"critic/q{index}_mean", float(q.mean().item()))
                    self.logger.record(f"critic/q{index}_std", float(q.std(unbiased=False).item()))
                    self.logger.record(f"critic/q{index}_min", float(q.min().item()))
                    self.logger.record(f"critic/q{index}_max", float(q.max().item()))
        finally:
            for module, training in modules.items():
                module.training = training
        self._observations.clear()

    def _flush(self) -> None:
        self._network_diagnostics()
        for name, total in self._window_sums.items():
            if name in ECONOMIC_FIELDS:
                self.logger.record(f"economics/window_sum_{name}", total)
            elif name in {"raw_reward", "scaled_reward"}:
                key = "raw_reward_sum" if name == "raw_reward" else "algorithm_reward_sum"
                self.logger.record(f"reward/window_{key}", total)
            elif name == "projected":
                self.logger.record("vessel/power_projection_fraction", total / self._window_counts[name])
            else:
                self.logger.record(f"vessel/{name}_mean", total / self._window_counts[name])
        self.logger.record("time/total_timesteps", int(self.num_timesteps))
        self.logger.record("time/completed_episodes", self.completed_episodes)
        self.logger.record("vessel/successful_episodes", self.successful_episodes)
        self.logger.record("vessel/completion_rate", self.successful_episodes / self.completed_episodes if self.completed_episodes else 0.0)
        replay = getattr(self.model, "replay_buffer", None)
        self.logger.record("replay/buffer_size", int(replay.size()) if replay is not None else 0)
        self.logger.record("diagnostics/critic_updates_completed", int(self.model._n_updates))
        # Reuse SB3's rollout/time reporting and its single native logger dump.
        # Pending train/* values remain intact until that dump; diagnostics do
        # not replace the latest native actor/critic losses or update count.
        self.model.dump_logs()
        self._window_sums.clear()
        self._window_counts.clear()
        self._last_dump_step = int(self.num_timesteps)
        self._pending_dump = False

    def _on_rollout_end(self) -> None:
        if self._pending_dump:
            self._flush()

    def _on_training_end(self) -> None:
        # SB3 skips on_rollout_end when a stopping callback returns False.
        # Also flush the final native train/* records from the preceding update.
        if self._window_counts or self.logger.name_to_value:
            self._flush()

    def _evaluate(self, split: str) -> None:
        fixed = split == "fixed_train"
        directory = self.directory / ("fixed_train_evaluation" if fixed else "periodic_validation")
        directory.mkdir(exist_ok=True)
        node = directory / f"episode_{self.completed_episodes:04d}"
        node.mkdir(exist_ok=False)
        checkpoint = node / "model.zip"
        self.model.save(str(checkpoint))
        _write_json(node / "checkpoint_metadata.json", {
            "algorithm": "stable_baselines3.TD3", "env_id": ENV_ID,
            "completed_episodes": self.completed_episodes,
            "successful_episodes": self.successful_episodes,
            "environment_steps": int(self.num_timesteps),
            "critic_updates": int(self.model._n_updates),
            "sb3_episode_counter": int(self.model._episode_num),
            "clock_phase": "after_env_step_before_replay_insert_and_rollout_update",
            "reward_identity": self.model._td3_reward_identity,
            "reward_scale": REWARD_SCALE, "checkpoint": str(checkpoint),
            "replay_buffer_saved": False,
            "evaluation_split": split, "evaluation_device": "cpu",
        })
        command = [
            sys.executable, "-u", "-m",
            f"experiments.sb3_manual.td3_aggregate.{'fixed_train_eval' if fixed else 'validate'}",
            "--model-path", str(checkpoint), "--data-root", str(self.data_root),
            "--output-dir", str(node), "--device", "cpu",
            "--seed", str(self.model.seed if self.model.seed is not None else 42),
        ]
        if fixed:
            command += ["--episode", str(self.completed_episodes), "--sample-id", FIXED_TRAIN_ID]
        else:
            command += ["--episodes", "8"]
        child_env = os.environ.copy()
        paths = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src")]
        if child_env.get("PYTHONPATH"):
            paths.append(child_env["PYTHONPATH"])
        child_env["PYTHONPATH"] = os.pathsep.join(paths)
        subprocess.run(command, cwd=PROJECT_ROOT, env=child_env, check=True)
        metrics_path = node / ("fixed_train_metrics.json" if fixed else "validation_metrics.json")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if not metrics.get("deterministic") or metrics.get("action_noise") is not None:
            raise RuntimeError("Periodic TD3 evaluation must use deterministic actions without noise")
        if not metrics.get("integrity") or not all(metrics["integrity"].values()):
            raise RuntimeError("Periodic TD3 evaluation failed frozen-model integrity checks")
        if metrics.get("reward_scale") != REWARD_SCALE:
            raise RuntimeError("Periodic TD3 evaluation used a different reward scale")
        if fixed:
            summary = metrics["summary"]
            if (
                metrics["split"] != "train" or summary["sample_id"] != FIXED_TRAIN_ID
                or summary["total_steps"] != FIXED_TRAIN_STEPS
                or summary["evaluation_truncated"]
            ):
                raise RuntimeError("Fixed diagnostic did not follow the full 9258s voyage protocol")
            row = {"training_episode": self.completed_episodes, **summary}
            _append_csv(directory / "fixed_train_evaluation.csv", row)
            for name in ("coverage", "completed", "total_cost_cny", "soc_min", "soc_final", "reward_sum", "algorithm_reward_sum"):
                self.logger.record(f"fixed_train/{'raw_reward_sum' if name == 'reward_sum' else name}", summary[name])
        else:
            voyages = metrics["per_voyage"]
            if (
                metrics["split"] != "validation" or not metrics["validation_set_complete"]
                or len(voyages) != 8 or len({row["sample_id"] for row in voyages}) != 8
                or any(row["split"] != "validation" or row["evaluation_truncated"] for row in voyages)
            ):
                raise RuntimeError("Periodic TD3 evaluation did not cover independent Validation8")
            row = {
                "training_episode": self.completed_episodes,
                "environment_steps": int(self.num_timesteps),
                "completion_rate": metrics["completion_rate"], "mean_coverage": metrics["mean_coverage"],
                "mean_raw_reward_sum": float(np.mean([v["reward_sum"] for v in voyages])),
                "mean_algorithm_reward_sum": float(np.mean([v["algorithm_reward_sum"] for v in voyages])),
                "mean_soc_final": float(np.mean([v["soc_final"] for v in voyages])),
                "min_soc": min(v["soc_min"] for v in voyages),
                **{f"total_{name}": sum(v[name] for v in voyages) for name in ECONOMIC_FIELDS},
            }
            _append_csv(directory / "periodic_validation.csv", row)
            for name, value in row.items():
                self.logger.record(f"validation/{name}", value)
        if self.verbose:
            self.logger.info(f"TD3 episode {self.completed_episodes}: independent {split} results: {metrics_path}")
