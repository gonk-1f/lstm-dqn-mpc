"""Register the isolated economic Huber-TD3 EMS environment in RL Zoo."""

import argparse
import json
from pathlib import Path
import sys
from zipfile import ZipFile

from gymnasium.envs.registration import register
from rl_zoo3.utils import ALGOS

from .economic_ems import ECONOMIC_EMS_ENV_ID, ECONOMIC_EMS_REWARD_VERSION, REWARD_SCALE
from .huber_td3 import HuberTD3


def validate_resume_metadata(argv):
    """Reject a different reward/action protocol before Zoo loads its Replay."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-i", "--trained-agent", type=Path)
    options, _ = parser.parse_known_args(argv)
    if options.trained_agent is None:
        return
    with ZipFile(options.trained_agent) as archive:
        metadata = json.loads(archive.read("data"))
    identity = metadata.get("_td3_reward_identity") or {}
    if (
        identity.get("env_id") != ECONOMIC_EMS_ENV_ID
        or identity.get("reward_version") != ECONOMIC_EMS_REWARD_VERSION
        or identity.get("reward_scale") != REWARD_SCALE
        or metadata.get("reward_scale") != REWARD_SCALE
        or any(key not in metadata for key in ("_td3_completed_episodes", "_td3_successful_episodes"))
    ):
        raise ValueError("Economic TD3 resume requires a checkpoint from the fixed-voyage reward protocol")


validate_resume_metadata(sys.argv[1:])
ALGOS["td3"] = HuberTD3

register(
    id=ECONOMIC_EMS_ENV_ID,
    entry_point="experiments.sb3_manual.td3_aggregate.economic_ems:make_economic_env",
)
