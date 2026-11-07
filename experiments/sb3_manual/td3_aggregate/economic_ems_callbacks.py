"""TD3 economic EMS callback with random Train35 episodes and fixed diagnostics."""

import json
import shutil
import subprocess
import sys

from .callbacks import ECONOMIC_FIELDS, FIXED_TRAIN_ID, PROJECT_ROOT, TD3TrainingCallback
from .economic_ems import REWARD_COMPONENT_FIELDS


class EconomicEMSTrainingCallback(TD3TrainingCallback):
    economic_fields = ECONOMIC_FIELDS + REWARD_COMPONENT_FIELDS
    reward_component_fields = REWARD_COMPONENT_FIELDS
    environment_id = "AggregateShipTD3Economic-v0"
    algorithm_name = "experiments.sb3_manual.td3_aggregate.huber_td3.HuberTD3"
    evaluation_extra_args = ("--reward-mode", "economic_ems")
    reward_scale = 0.001
    require_fixed_train = False

    def _additional_episode_fields(self, summary):
        return {
            "soc_initial": summary["soc_initial"],
            "soc_delta": summary["soc_delta"],
            "terminal_soc_target": summary["terminal_soc_target"],
            "terminal_soc_error": summary["terminal_soc_error"],
        }

    def _on_step(self):
        keep_training = super()._on_step()
        dones = self.locals.get("dones", [])
        if dones and bool(dones[0]) and self.completed_episodes % 20 == 0:
            subprocess.run(
                [sys.executable, "-m", "experiments.sb3_manual.td3_aggregate.economic_ems_plots",
                 "--run-dir", str(self.directory)],
                cwd=PROJECT_ROOT,
                check=True,
            )
        return keep_training

    def _evaluate(self, split: str) -> None:
        super()._evaluate(split)
        if split != "validation":
            return
        metrics_path = (
            self.directory / "periodic_validation"
            / f"episode_{self.completed_episodes:04d}" / "validation_metrics.json"
        )
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        voyages = metrics["per_voyage"]
        # A candidate must complete all Validation8 voyages and keep every
        # terminal SOC within 0.05 of the specified 0.5 target.
        eligible = (
            metrics.get("validation_set_complete") is True
            and len(voyages) == 8
            and all(voyage.get("success") is True for voyage in voyages)
            and max(abs(float(voyage["soc_final"]) - 0.5) for voyage in voyages) <= 0.05
            and min(float(voyage["soc_min"]) for voyage in voyages) >= 0.2 - 1e-9
        )
        if not eligible:
            return
        score = float(metrics["mean_full_voyage_economic_cost_cny"])
        best_path = self.directory / "best_validation_model.zip"
        current = getattr(self, "best_validation_cost_cny", float("inf"))
        if score < current:
            node = metrics_path.parent
            shutil.copy2(node / "model.zip", best_path)
            self.best_validation_cost_cny = score
            (self.directory / "best_validation_metadata.json").write_text(
                json.dumps({
                    "training_episode": self.completed_episodes,
                    "mean_validation_total_cost_cny": score,
                    "max_abs_terminal_soc_error": max(
                        abs(float(voyage["soc_final"]) - 0.5) for voyage in voyages
                    ),
                    "validation_voyages": [voyage["sample_id"] for voyage in voyages],
                    "checkpoint": str(best_path),
                    "selection_rule": "complete Validation8; every terminal SOC within 0.05 of 0.5; minimize mean full-voyage physical CNY cost",
                }, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
