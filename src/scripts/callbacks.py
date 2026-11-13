from __future__ import annotations

from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from td3_config import (
    DT_S,
    evaluate_profiles,
    save_json,
    summarize_validation,
    write_csv,
)


class ShipEMSTrainingCallback(BaseCallback):
    def __init__(
        self,
        train_profile,
        train_id: str,
        validation_profiles,
        validation_ids,
        output_dir: Path,
        max_episodes: int = 500,
        reward_plot_interval: int = 10,
        train_diagnostic_interval: int = 10,
        validation_interval: int = 100,
        seed: int = 42,
        verbose: int = 1,
    ):
        super().__init__(verbose=verbose)
        self.train_profile = train_profile
        self.train_id = train_id
        self.validation_profiles = validation_profiles
        self.validation_ids = list(validation_ids)
        self.output_dir = Path(output_dir)
        self.stop_file = self.output_dir / "stop.request"
        self.max_episodes = int(max_episodes)
        self.reward_plot_interval = int(reward_plot_interval)
        self.train_diagnostic_interval = int(train_diagnostic_interval)
        self.validation_interval = int(validation_interval)
        self.seed = int(seed)

        self.episode_rows = []
        self.train_diagnostic_rows = []
        self.validation_rows = []
        self.best_score = None
        self.best_episode = None

    def _on_training_start(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _save_training_tables(self):
        write_csv(self.output_dir / "training_episodes.csv", self.episode_rows)
        write_csv(
            self.output_dir / "fixed_train_diagnostics.csv",
            self.train_diagnostic_rows,
        )
        write_csv(
            self.output_dir / "validation_history.csv",
            self.validation_rows,
        )

    def _save_reward_plot(self, episode_number: int | None = None):
        if not self.episode_rows:
            return

        import matplotlib.pyplot as plt
        import pandas as pd

        rewards = pd.Series(
            [row["mean_reward_per_step"] for row in self.episode_rows],
            dtype="float64",
        )
        rolling = rewards.rolling(window=10, min_periods=1).mean()

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(
            np.arange(1, len(rewards) + 1),
            rewards.to_numpy(),
            linewidth=0.8,
            marker=".",
            alpha=0.65,
            label="Mean reward per step",
        )
        failed = [row for row in self.episode_rows if row["failure"]]
        if failed:
            ax.scatter(
                [row["episode"] for row in failed],
                [row["mean_reward_per_step"] for row in failed],
                color="tab:red",
                marker="x",
                label="Physical failure",
            )
        ax.plot(
            np.arange(1, len(rolling) + 1),
            rolling.to_numpy(),
            linewidth=1.8,
            label="10-episode moving average",
        )
        ax.set_xlabel("Episode")
        ax.set_ylabel("Episode return / episode steps")
        ax.set_title("TD3 training reward")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.output_dir / "reward_curve.png", dpi=180)
        if episode_number is not None:
            fig.savefig(
                self.output_dir / f"reward_episode_{episode_number:04d}.png",
                dpi=180,
            )
        plt.close(fig)

    def _save_power_plot(self, episode_number: int, result: dict, steps: list[dict]):
        import matplotlib.pyplot as plt

        executed = [row for row in steps if row["fc_actual_kw"] is not None]
        fig, (power_ax, soc_ax) = plt.subplots(
            2, 1, figsize=(11, 6), sharex=True, height_ratios=(2, 1)
        )
        time_s = [(row["step"] - 1) * DT_S for row in executed]
        for key, label in (
            ("load_kw", "Ship load"),
            ("fc_actual_kw", "Fuel cell (net)"),
            ("battery_actual_kw", "Battery (+ discharge)"),
        ):
            power_ax.step(time_s, [row[key] for row in executed], where="post", label=label)

        if executed:
            soc_ax.plot(
                [0, *[row["step"] * DT_S for row in executed]],
                [executed[0]["soc_before"], *[row["soc"] for row in executed]],
                color="tab:purple",
            )
        if not result["completed"]:
            for ax in (power_ax, soc_ax):
                ax.axvline(len(executed) * DT_S, color="tab:red", linestyle="--")
        power_ax.set_ylabel("Power (kW)")
        power_ax.legend(loc="best")
        soc_ax.set_ylabel("SOC")
        soc_ax.set_xlabel("Elapsed time (s)")
        soc_ax.set_ylim(0.18, 0.82)
        for ax in (power_ax, soc_ax):
            ax.grid(True, alpha=0.25)
        fig.suptitle(
            f"Episode {episode_number} | deterministic {self.train_id} | "
            f"coverage {result['coverage']:.1%}"
        )
        fig.tight_layout()
        fig.savefig(
            self.output_dir / f"power_episode_{episode_number:04d}.png", dpi=180
        )
        plt.close(fig)

    def _run_fixed_train_diagnostic(self, episode_number: int):
        was_training = bool(self.model.policy.training)
        self.model.policy.set_training_mode(False)
        try:
            results, steps = evaluate_profiles(
                self.model,
                [self.train_profile],
                [self.train_id],
                seed=self.seed,
                collect_step_rows=True,
            )
        finally:
            self.model.policy.set_training_mode(was_training)

        for row in results:
            row["training_episode"] = episode_number
            self.train_diagnostic_rows.append(row)

        write_csv(
            self.output_dir / "fixed_train_diagnostics.csv",
            self.train_diagnostic_rows,
        )
        fields = (
            "sample_id", "step", "load_kw", "actor_action_fraction",
            "requested_fc_kw", "fc_actual_kw", "battery_actual_kw",
            "feasible_fc_lower_kw", "feasible_fc_upper_kw",
            "step_battery_equivalent_hydrogen_cost_rmb", "step_soc_penalty_unscaled",
            "soc_before", "soc", "reward", "terminal_reason",
        )
        write_csv(
            self.output_dir / f"power_episode_{episode_number:04d}.csv",
            [{field: row.get(field) for field in fields} for row in steps],
        )
        self._save_power_plot(episode_number, results[0], steps)

        if self.verbose:
            row = results[0]
            print(
                f"[Fixed Train diagnostic] episode={episode_number}, "
                f"completed={row['completed']}, coverage={row['coverage']:.4f}, "
                f"final_SOC={row['final_soc']}"
            )

    def _run_validation(self, episode_number: int):
        was_training = bool(self.model.policy.training)
        self.model.policy.set_training_mode(False)
        try:
            results, _ = evaluate_profiles(
                self.model,
                self.validation_profiles,
                self.validation_ids,
                seed=self.seed,
                collect_step_rows=False,
            )
        finally:
            self.model.policy.set_training_mode(was_training)

        for row in results:
            row["training_episode"] = episode_number
            self.validation_rows.append(row)

        metrics = summarize_validation(results)
        metrics["training_episode"] = episode_number
        metrics["mean_reward_for_model_selection"] = metrics["mean_episode_reward"]

        write_csv(
            self.output_dir / "validation_history.csv",
            self.validation_rows,
        )
        save_json(
            self.output_dir / f"validation_episode_{episode_number:04d}.json",
            {"metrics": metrics, "voyages": results},
        )

        # Lexicographic selection: more completed Validation voyages first;
        # then higher mean episode reward (the reward actually optimized).
        score = (
            int(metrics["completed_voyages"]),
            float(metrics["mean_episode_reward"])
            if metrics["mean_episode_reward"] is not None
            else float("-inf"),
        )

        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.best_episode = episode_number
            self.model.save(str(self.output_dir / "best_model"))
            save_json(
                self.output_dir / "best_validation.json",
                {
                    "training_episode": episode_number,
                    "selection_score": list(score),
                    "metrics": metrics,
                    "voyages": results,
                },
            )

        if self.verbose:
            print(
                f"[Validation8] episode={episode_number}, "
                f"completed={metrics['completed_voyages']}/8, "
                f"mean_reward={metrics['mean_episode_reward']}, "
                f"mean_cost_completed_only="
                f"{metrics['mean_total_economic_cost_completed_only_rmb']}"
            )

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")

        if dones is None or infos is None:
            return True

        for done, info in zip(dones, infos):
            if not bool(done):
                continue

            episode_info = info.get("episode", {})
            episode_number = len(self.episode_rows) + 1
            episode_reward = float(
                episode_info.get("r", info.get("episode_reward", 0.0))
            )
            episode_length = int(
                episode_info.get("l", info.get("episode_length", 0))
            )
            mean_reward = episode_reward / episode_length
            completed = (
                info.get("terminal_reason") == "voyage_end"
                and not info.get("failure", False)
                and bool(info.get("economic_cost_ready", False))
            )

            self.episode_rows.append({
                "episode": episode_number,
                "global_timesteps": int(self.num_timesteps),
                "sample_id": info.get("voyage_id", self.train_id),
                "episode_reward": episode_reward,
                "mean_reward_per_step": mean_reward,
                "episode_length": episode_length,
                "completed": bool(completed),
                "failure": bool(info.get("failure", False)),
                "terminal_reason": info.get("terminal_reason"),
                "final_soc": info.get("final_soc", info.get("soc")),
                "total_economic_cost_rmb": info.get("total_economic_cost_rmb"),
                "voyage_hydrogen_kg": info.get("voyage_hydrogen_kg"),
                "fc_total_degradation_cost_rmb": info.get(
                    "fc_total_degradation_cost_rmb"
                ),
                "voyage_battery_degradation_cost_rmb": info.get(
                    "voyage_battery_degradation_cost_rmb"
                ),
            })

            if episode_number % self.reward_plot_interval == 0:
                self._save_training_tables()
                self._save_reward_plot(episode_number)
                self.model.save(str(self.output_dir / "last_model"))
                if self.verbose:
                    print(
                        f"[Training] episode={episode_number}/{self.max_episodes}, "
                        f"steps={self.num_timesteps}, mean_reward={mean_reward:.6f}"
                    )

            if episode_number % self.train_diagnostic_interval == 0:
                # Keep the checkpoint used for this deterministic diagnostic.
                self.model.save(
                    str(
                        self.output_dir
                        / f"checkpoint_episode_{episode_number:04d}"
                    )
                )

                self._run_fixed_train_diagnostic(episode_number)

            if episode_number % self.validation_interval == 0:
                self._run_validation(episode_number)

            if episode_number >= self.max_episodes:
                self._save_training_tables()
                self._save_reward_plot()
                self.model.save(str(self.output_dir / "last_model"))
                if self.verbose:
                    print(f"[Training] reached {self.max_episodes} episodes; stopping.")
                return False

        if self.num_timesteps % 100 == 0 and self.stop_file.is_file():
            if self.verbose:
                print(f"[Training] stop requested at step {self.num_timesteps}.")
            return False

        return True

    def _on_training_end(self) -> None:
        self._save_training_tables()
        self._save_reward_plot()
        self.model.save(str(self.output_dir / "last_model"))
