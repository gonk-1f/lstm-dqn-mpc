"""Register the isolated economic Huber-TD3 EMS environment in RL Zoo."""

import argparse
import json
from pathlib import Path
import sys
from zipfile import ZipFile

from gymnasium.envs.registration import register
from rl_zoo3.utils import ALGOS

from .economic_ems import (
    ECONOMIC_EMS_ENV_ID, ECONOMIC_EMS_REWARD_VERSION, N_STEP_RETURN, REWARD_SCALE,
    V6_ENV_ID, V6_REWARD_VERSION,
)
from .economic_action_wrappers import PREVIOUS_FC_ACTION_MAPPING_VERSION
from .huber_td3 import HuberTD3


def validate_resume_metadata(argv):
    """Reject a different reward/action protocol before Zoo loads its Replay."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-i", "--trained-agent", type=Path)
    parser.add_argument("--env", default=ECONOMIC_EMS_ENV_ID)
    options, _ = parser.parse_known_args(argv)
    if options.trained_agent is None:
        return
    with ZipFile(options.trained_agent) as archive:
        metadata = json.loads(archive.read("data"))
    identity = metadata.get("_td3_reward_identity") or {}
    v6 = options.env == V6_ENV_ID
    reward_version = V6_REWARD_VERSION if v6 else ECONOMIC_EMS_REWARD_VERSION
    if (
        identity.get("env_id") != options.env
        or identity.get("reward_version") != reward_version
        or (v6 and identity.get("action_mapping") != PREVIOUS_FC_ACTION_MAPPING_VERSION)
        or identity.get("n_step_return") != N_STEP_RETURN
        or identity.get("reward_scale") != REWARD_SCALE
        or metadata.get("n_steps") != N_STEP_RETURN
        or metadata.get("reward_scale") != REWARD_SCALE
        or any(key not in metadata for key in ("_td3_completed_episodes", "_td3_successful_episodes"))
    ):
        raise ValueError(f"Economic TD3 resume requires a matching {reward_version} checkpoint with a 32-step return")


validate_resume_metadata(sys.argv[1:])
ALGOS["td3"] = HuberTD3

register(
    id=ECONOMIC_EMS_ENV_ID,
    entry_point="experiments.sb3_manual.td3_aggregate.economic_ems:make_economic_env",
)
register(
    id=V6_ENV_ID,
    entry_point="experiments.sb3_manual.td3_aggregate.economic_ems:make_v6_env",
)
