"""Register the isolated economic Huber-TD3 EMS environment in RL Zoo."""

from gymnasium.envs.registration import register
from rl_zoo3.utils import ALGOS

from .economic_ems import make_economic_env
from .huber_td3 import HuberTD3

ALGOS["td3"] = HuberTD3

register(
    id="AggregateShipTD3Economic-v0",
    entry_point="experiments.sb3_manual.td3_aggregate.economic_ems:make_economic_env",
)
