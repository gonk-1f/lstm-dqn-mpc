from gymnasium.envs.registration import register

from rl_zoo3.utils import ALGOS

from .huber_ddpg import HuberDDPG

# Zoo imports this --gym-packages module before model creation/loading.
ALGOS["ddpg"] = HuberDDPG

register(
    id="AggregateShipDDPG-v0",
    entry_point="experiments.sb3_manual.ddpg_aggregate.env:AggregateDDPGEnv",
)
