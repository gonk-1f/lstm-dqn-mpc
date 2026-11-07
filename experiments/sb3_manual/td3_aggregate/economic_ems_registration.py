"""Register the isolated economic TD3 EMS environment in RL Zoo."""

from gymnasium.envs.registration import register

from .economic_ems import make_economic_env

register(
    id="AggregateShipTD3Economic-v0",
    entry_point="experiments.sb3_manual.td3_aggregate.economic_ems:make_economic_env",
)
