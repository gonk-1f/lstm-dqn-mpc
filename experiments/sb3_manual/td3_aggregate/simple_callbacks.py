"""Reuse TD3 diagnostics/evaluation for the isolated simple-reward experiment."""
import subprocess
import sys

from .callbacks import ECONOMIC_FIELDS, PROJECT_ROOT, TD3TrainingCallback
from .simple_reward import REWARD_COMPONENT_FIELDS


class SimpleRewardTrainingCallback(TD3TrainingCallback):
    economic_fields = ECONOMIC_FIELDS + REWARD_COMPONENT_FIELDS
    reward_component_fields = REWARD_COMPONENT_FIELDS
    environment_id = "AggregateShipTD3Simple-v0"
    evaluation_extra_args = ("--reward-mode", "simple")

    def _additional_episode_fields(self, summary):
        return {name: summary[name] for name in ("soc_initial", "soc_delta")}

    def _on_training_start(self):
        super()._on_training_start()
        self.pause_request = self.directory / "pause_after_episode.request"
        self.logger.info(f"Safe pause after the current episode: create {self.pause_request}")

    def _on_step(self):
        keep_training = super()._on_step()
        if self.pause_request.exists() and bool(self.locals["dones"][0]):
            self.logger.info("Safe pause requested: Zoo will save the model and Replay Buffer.")
            return False
        return keep_training

    def _plot(self):
        if (self.directory / "training_episodes.csv").exists():
            subprocess.run([
                sys.executable, "-m", "experiments.sb3_manual.td3_aggregate.simple_reward_plots",
                "--run-dir", str(self.directory),
            ], cwd=PROJECT_ROOT, check=True)

    def _evaluate(self, split):
        super()._evaluate(split)
        if split == "fixed_train":
            self._plot()

    def _on_training_end(self):
        super()._on_training_end()
        self._plot()


class ControlRewardTrainingCallback(SimpleRewardTrainingCallback):
    environment_id = "AggregateShipTD3Control-v0"
    evaluation_extra_args = ("--reward-mode", "control")
