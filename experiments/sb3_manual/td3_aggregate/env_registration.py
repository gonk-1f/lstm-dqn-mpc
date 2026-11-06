"""Gym registration only: leave RL Zoo ALGOS['td3'] as native SB3 TD3."""
from gymnasium.envs.registration import register
import sys

from .run_td3_train import validate_resume_metadata

# Zoo imports --gym-packages before loading --trained-agent. A callback alone
# cannot detect an unmarked warmup model after learn() resets num_timesteps.
validate_resume_metadata(sys.argv[1:])

register(
    id="AggregateShipTD3-v0",
    entry_point="experiments.sb3_manual.td3_aggregate.reward_wrapper:make_env",
)
