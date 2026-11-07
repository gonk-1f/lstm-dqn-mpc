"""TD3 economic EMS callback with random Train35 episodes and fixed diagnostics."""

from .callbacks import ECONOMIC_FIELDS, FIXED_TRAIN_ID, TD3TrainingCallback
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
