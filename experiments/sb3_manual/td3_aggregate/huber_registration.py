"""Register the Huber TD3 variant inside its dedicated Zoo process."""
import sys

from gymnasium.envs.registration import register
from rl_zoo3.utils import ALGOS

from .huber_control import HUBER_CONTROL_ENV_ID
from .huber_td3 import HuberTD3
from .simple_training import validate_resume_metadata


validate_resume_metadata(sys.argv[1:])
ALGOS["td3"] = HuberTD3
register(
    id=HUBER_CONTROL_ENV_ID,
    entry_point="experiments.sb3_manual.td3_aggregate.huber_control:make_huber_control_env",
)
