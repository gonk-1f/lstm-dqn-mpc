"""Separate Gym ID; native Zoo/SB3 TD3 remains unchanged."""
import sys
from gymnasium.envs.registration import register
from .simple_training import validate_resume_metadata

validate_resume_metadata(sys.argv[1:])
register(
    id="AggregateShipTD3Simple-v0",
    entry_point="experiments.sb3_manual.td3_aggregate.simple_reward:make_simple_env",
)
register(
    id="AggregateShipTD3Control-v0",
    entry_point="experiments.sb3_manual.td3_aggregate.simple_reward:make_control_env",
)
