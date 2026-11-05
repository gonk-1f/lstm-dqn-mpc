"""Episode-based validation in a separate process, without touching training RNG."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from .economic_reward import FC_DEGRADATION_COST_FIELDS

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class EpisodeValidationCallback(BaseCallback):
    """Save/evaluate after real episode endings; preserve native SB3 update clocks."""

    def __init__(
        self, every_episodes: int = 100, max_episodes: int = 500,
        output_root: str | Path | None = None, verbose: int = 1,
    ):
        super().__init__(verbose=verbose)
        if every_episodes < 1 or max_episodes < 1:
            raise ValueError("Episode budgets must be positive")
        self.every_episodes = every_episodes
        self.max_episodes = max_episodes
        self.output_root = None if output_root is None else Path(output_root)
        self.completed_episodes = 0

    def _on_training_start(self) -> None:
        self.completed_episodes = int(getattr(
            self.model, "_ddpg_completed_episodes", self.model._episode_num
        ))
        if self.completed_episodes >= self.max_episodes:
            raise ValueError("This checkpoint has already reached the episode budget")
        identity = self.training_env.get_attr("reward_identity")[0]
        saved_identity = getattr(self.model, "_ddpg_reward_identity", None)
        if saved_identity is not None and saved_identity != identity:
            raise ValueError("Cannot resume a checkpoint from another reward definition")
        # SB3 may reset num_timesteps before this hook; optimizer updates survive.
        if saved_identity is None and (
            self.model.num_timesteps > 0 or self.model._n_updates > 0
            or (self.model.replay_buffer is not None and self.model.replay_buffer.size() > 0)
        ):
            raise ValueError("Legacy reward checkpoint cannot resume economic-reward training")
        self.model._ddpg_completed_episodes = self.completed_episodes
        self.model._ddpg_reward_identity = identity
        for name in ("eta_fc_avg", "soc_deviation_weight", "soc_reference"):
            if name in identity:
                self.logger.record(f"reward/{name}", identity[name])
        if self.verbose:
            self.logger.info(f"Economic reward calibration: {identity}")
        self.data_root = Path(self.training_env.get_attr("data_root")[0]).resolve()
        if self.output_root is None:
            writer = self.training_env.get_attr("results_writer")[0]
            if writer is None:
                raise ValueError("Periodic validation requires a Monitor file or output_root")
            self.directory = Path(writer.file_handler.name).resolve().parent / "periodic_validation"
        else:
            self.directory = self.output_root.resolve()
        self.directory.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            for name in (
                "equivalent_fc_efficiency", "fc_efficiency_actual", "c_bat_eq_cny",
                "h2_cost_cny", "fuel_cell_degradation_cost_cny",
                "battery_degradation_cost_cny", "battery_current_a", "total_cost_cny",
                "reward_total_cost_cny",
                "reward_economic_cost_cny", "soc_deviation_penalty_cny",
                "shutdown_fc_transient_cost_cny", "shutdown_fc_start_stop_cost_cny",
                "shutdown_economic_cost_cny",
            ) + FC_DEGRADATION_COST_FIELDS:
                if name in info:
                    self.logger.record(f"economics/{name}", info[name])
            for name in ("physical_fc_on", "fc_start_stop_cycles"):
                if name in info:
                    self.logger.record(f"vessel/{name}", info[name])
        ended = int(np.count_nonzero(self.locals["dones"]))
        if ended:
            self.completed_episodes += ended
            self.model._ddpg_completed_episodes = self.completed_episodes
            if self.completed_episodes % self.every_episodes == 0:
                self._save_and_validate()
        return self.completed_episodes < self.max_episodes

    def _save_and_validate(self) -> None:
        episode = self.completed_episodes
        node = self.directory / f"episode_{episode:04d}"
        node.mkdir(exist_ok=False)
        checkpoint = node / "model.zip"
        self.model.save(str(checkpoint))
        checkpoint_metadata = {
            "completed_episodes": episode,
            "environment_steps": int(self.model.num_timesteps),
            "adam_updates": int(self.model._n_updates),
            "sb3_episode_counter": int(self.model._episode_num),
            "clock_phase": "after_env_step_before_replay_insert_and_rollout_adam_update",
            "reward_identity": self.model._ddpg_reward_identity,
            "checkpoint": str(checkpoint),
        }
        with (node / "checkpoint_metadata.json").open("x", encoding="utf-8") as handle:
            json.dump(checkpoint_metadata, handle, ensure_ascii=False, indent=2, allow_nan=False)
        child_env = os.environ.copy()
        paths = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src")]
        if child_env.get("PYTHONPATH"):
            paths.append(child_env["PYTHONPATH"])
        child_env["PYTHONPATH"] = os.pathsep.join(paths)
        command = [
            sys.executable, "-u", "-m", "experiments.sb3_manual.ddpg_aggregate.validate",
            "--model-path", str(checkpoint), "--data-root", str(self.data_root),
            "--output-dir", str(node), "--episodes", "8",
            "--device", "cpu", "--seed", str(self.model.seed if self.model.seed is not None else 42),
        ]
        # Loading a second SB3 model in this process would reset global RNG seeds.
        subprocess.run(command, cwd=PROJECT_ROOT, env=child_env, check=True)
        metrics_path = node / "validation_metrics.json"
        if not metrics_path.is_file():
            raise RuntimeError("Periodic validation must produce exactly one complete metrics file")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        voyages = metrics["per_voyage"]
        if (
            metrics["validation_voyages"] != 8 or not metrics["validation_set_complete"]
            or len(voyages) != 8 or len({row["sample_id"] for row in voyages}) != 8
            or any(row["split"] != "validation" for row in voyages)
            or not all(metrics["integrity"].values())
        ):
            raise RuntimeError("Periodic evaluation did not complete the independent Validation8 protocol")
        components = {
            "total_h2_cost_cny": sum(row["h2_cost_cny"] for row in voyages),
            "total_fc_degradation_cost_cny": sum(row["fuel_cell_degradation_cost_cny"] for row in voyages),
            "total_battery_degradation_cost_cny": sum(row["battery_degradation_cost_cny"] for row in voyages),
            "total_battery_equivalent_hydrogen_cost_cny": sum(row["c_bat_eq_cny"] for row in voyages),
        }
        total_economic_cost = sum(row["total_cost_cny"] for row in voyages)
        summary = {
            "episode": episode,
            "environment_steps": int(self.model.num_timesteps),
            "adam_updates": int(self.model._n_updates),
            "completion_rate": metrics["completion_rate"],
            "mean_duration_coverage": metrics["mean_coverage"],
            "mean_reward": metrics["mean_episode_reward"],
            **components,
            **{f"total_{name}": sum(row[name] for row in voyages) for name in FC_DEGRADATION_COST_FIELDS},
            "total_reward_cost_cny": sum(row["reward_total_cost_cny"] for row in voyages),
            "total_reward_economic_cost_cny": sum(row["reward_economic_cost_cny"] for row in voyages),
            "total_soc_deviation_penalty_cny": sum(row["soc_deviation_penalty_cny"] for row in voyages),
            "total_fc_shutdowns": sum(row["fc_shutdowns"] for row in voyages),
            "total_fc_start_stop_cycles": sum(row["fc_start_stop_cycles"] for row in voyages),
            "total_economic_cost_cny": total_economic_cost,
            "mean_full_voyage_economic_cost_cny": (
                total_economic_cost / 8 if all(row["completed"] for row in voyages) else None
            ),
            "mean_soc_initial": sum(row["soc_initial"] for row in voyages) / 8,
            "mean_soc_final": sum(row["soc_final"] for row in voyages) / 8,
            "min_soc": min(row["soc_min"] for row in voyages),
            "max_soc": max(row["soc_max"] for row in voyages),
            "checkpoint": str(checkpoint),
            "validation_metrics_path": str(metrics_path),
        }
        csv_path = self.directory / "periodic_validation.csv"
        new_csv = not csv_path.exists()
        with csv_path.open("a", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary))
            if new_csv:
                writer.writeheader()
            writer.writerow(summary)
        json_path = self.directory / "periodic_validation.json"
        history = json.loads(json_path.read_text(encoding="utf-8")) if json_path.exists() else []
        history.append({**summary, "per_voyage": voyages})
        json_path.write_text(
            json.dumps(history, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
        )
        for name, value in summary.items():
            if name not in {"checkpoint", "validation_metrics_path"}:
                self.logger.record(f"validation/{name}", value)
        if self.verbose:
            print(
                f"Episode {episode} Validation8: completion={metrics['completion_rate']:.3f}, "
                f"mean_reward={metrics['mean_episode_reward']:.6g}, "
                f"executed_economic_cost={total_economic_cost:.6g} CNY; {metrics_path}",
                flush=True,
            )
