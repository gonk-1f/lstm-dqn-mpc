"""The pure-control reward and diagnostics for the SOC-12 Huber TD3 run."""
from pathlib import Path

from .huber_td3 import CRITIC_LOSS_ID
from .simple_callbacks import ControlRewardTrainingCallback
from .simple_reward import SimpleHydrogenReward, make_control_env


HUBER_CONTROL_ENV_ID = "AggregateShipTD3ControlHuber-v0"
HUBER_CONTROL_REWARD_VERSION = "td3_control_soc12_huber_scaled_1e4_v1"
SOC_DEVIATION_WEIGHT = 12.0


def make_huber_control_env(
    data_root: str | Path, split: str = "train", fixed_train_id: str | None = None,
) -> SimpleHydrogenReward:
    """Apply the new SOC weight through the existing pure-control wrapper."""
    env = make_control_env(data_root=data_root, split=split, fixed_train_id=fixed_train_id)
    env.lambda_soc = SOC_DEVIATION_WEIGHT
    env.reward_identity.update(
        reward_version=HUBER_CONTROL_REWARD_VERSION,
        env_id=HUBER_CONTROL_ENV_ID,
        soc_deviation_weight=SOC_DEVIATION_WEIGHT,
        critic_loss=CRITIC_LOSS_ID,
        critic_huber_delta=1.0,
    )
    return env


class HuberControlTrainingCallback(ControlRewardTrainingCallback):
    environment_id = HUBER_CONTROL_ENV_ID
    evaluation_extra_args = ("--reward-mode", "control_huber")
    algorithm_name = "experiments.sb3_manual.td3_aggregate.huber_td3.HuberTD3"
