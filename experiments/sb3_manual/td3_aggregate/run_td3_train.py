"""Launch the official RL Zoo trainer with this experiment's fixed defaults."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

from . import PROJECT_ROOT


def validate_resume_metadata(argv) -> None:
    """Zoo's gym-package hook and the launcher check before SB3 resets clocks."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-i", "--trained-agent", type=Path)
    resume, _ = parser.parse_known_args(argv)
    if resume.trained_agent is not None:
        from .reward_wrapper import REWARD_SCALE, TD3_REWARD_VERSION
        path = resume.trained_agent
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        with ZipFile(path) as archive:
            metadata = json.loads(archive.read("data"))
        identity = metadata.get("_td3_reward_identity")
        if (not isinstance(identity, dict) or identity.get("algorithm") != "TD3"
                or identity.get("reward_version") != TD3_REWARD_VERSION
                or identity.get("reward_scale") != REWARD_SCALE
                or metadata.get("reward_scale") != REWARD_SCALE
                or metadata.get("_ddpg_reward_identity") is not None):
            raise ValueError("Pretrained model has an incompatible TD3 reward identity/scale")
        if any(key not in metadata for key in ("_td3_completed_episodes", "_td3_successful_episodes")):
            raise ValueError("TD3 checkpoint lacks cumulative episode counters")
        # Full identity equality with the actual physical environment is checked
        # again by TD3TrainingCallback before collecting any transition.


def main(argv=None) -> int:
    extra_args = list(sys.argv[1:] if argv is None else argv)
    validate_resume_metadata(extra_args)

    child_env = os.environ.copy()
    paths = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src")]
    if child_env.get("PYTHONPATH"):
        paths.append(child_env["PYTHONPATH"])
    child_env["PYTHONPATH"] = os.pathsep.join(paths)
    command = [
        sys.executable, "-u", "-m", "rl_zoo3.train",
        "--algo", "td3", "--env", "AggregateShipTD3-v0",
        "--gym-packages", "experiments.sb3_manual.td3_aggregate.env_registration",
        "--conf-file", "experiments/sb3_manual/td3_aggregate/zoo_td3.yml",
        "--device", "cuda", "--eval-freq", "-1", "--seed", "42",
        "--log-folder", "outputs/td3_train_500ep",
        "--tensorboard-log", "outputs/td3_train_500ep/tensorboard",
        "--log-interval", "-2", "--save-replay-buffer",
        *extra_args,
    ]
    # Keep the child's native stdout/stderr and KeyboardInterrupt save path.
    # subprocess.run() would kill the child if this launcher also received Ctrl+C.
    child = subprocess.Popen(command, cwd=PROJECT_ROOT, env=child_env)
    try:
        return child.wait()
    except KeyboardInterrupt:
        print("Waiting for RL Zoo to finish saving the model and Replay Buffer...", flush=True)
        return child.wait()


if __name__ == "__main__":
    raise SystemExit(main())
