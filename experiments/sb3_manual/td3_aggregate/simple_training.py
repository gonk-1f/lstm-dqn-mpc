"""Independent native Zoo launch/resume entry for the simple reward."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

from . import PROJECT_ROOT


def validate_resume_metadata(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-i", "--trained-agent", type=Path)
    parser.add_argument("--control-only", action="store_true")
    parser.add_argument("--control-huber", action="store_true")
    parser.add_argument("--env", default="AggregateShipTD3Simple-v0")
    args, _ = parser.parse_known_args(argv)
    if args.trained_agent is None:
        return
    from .simple_reward import SIMPLE_REWARD_VERSION, CONTROL_REWARD_VERSION
    control_only = args.control_only or args.env == "AggregateShipTD3Control-v0"
    expected_version = CONTROL_REWARD_VERSION if control_only else SIMPLE_REWARD_VERSION
    if args.control_huber or args.env == "AggregateShipTD3ControlHuber-v0":
        from .huber_control import HUBER_CONTROL_REWARD_VERSION

        expected_version = HUBER_CONTROL_REWARD_VERSION
    path = args.trained_agent
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with ZipFile(path) as archive:
        metadata = json.loads(archive.read("data"))
    identity = metadata.get("_td3_reward_identity", {})
    if (identity.get("reward_version") != expected_version
            or identity.get("algorithm") != "TD3"
            or identity.get("reward_scale") != 0.0001
            or metadata.get("reward_scale") != 0.0001
            or any(key not in metadata for key in ("_td3_completed_episodes", "_td3_successful_episodes"))):
        raise ValueError("Resume requires a simple-reward TD3 checkpoint with matching Replay; the selected reward identity is incompatible")


def main(argv=None):
    extra = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--control-only", action="store_true")
    selection.add_argument("--control-huber", action="store_true")
    options, extra = parser.parse_known_args(extra)
    registration = "simple_registration"
    if options.control_huber:
        env_id = "AggregateShipTD3ControlHuber-v0"
        config_file = "zoo_td3_control_huber.yml"
        output_root = "outputs/td3_control_soc12_huber_freq1_500ep"
        registration = "huber_registration"
    elif options.control_only:
        env_id = "AggregateShipTD3Control-v0"
        config_file = "zoo_td3_control.yml"
        output_root = "outputs/td3_control_fc_soc_500ep"
    else:
        env_id = "AggregateShipTD3Simple-v0"
        config_file = "zoo_td3_simple.yml"
        output_root = "outputs/td3_simple_h2_fc_soc_500ep"
    validate_resume_metadata(["--env", env_id, *extra])
    child_env = os.environ.copy()
    paths = [str(PROJECT_ROOT), str(PROJECT_ROOT / "src")]
    if child_env.get("PYTHONPATH"):
        paths.append(child_env["PYTHONPATH"])
    child_env["PYTHONPATH"] = os.pathsep.join(paths)
    command = [
        sys.executable, "-u", "-m", "rl_zoo3.train",
        "--algo", "td3", "--env", env_id,
        "--gym-packages", f"experiments.sb3_manual.td3_aggregate.{registration}",
        "--conf-file", f"experiments/sb3_manual/td3_aggregate/{config_file}",
        "--device", "cuda", "--eval-freq", "-1", "--seed", "42",
        "--log-folder", output_root,
        "--tensorboard-log", output_root + "/tensorboard",
        "--log-interval", "-2", "--save-replay-buffer", *extra,
    ]
    child = subprocess.Popen(command, cwd=PROJECT_ROOT, env=child_env)
    try:
        return child.wait()
    except KeyboardInterrupt:
        print("Waiting for Zoo to save the simple-reward model and Replay Buffer...", flush=True)
        return child.wait()


if __name__ == "__main__":
    raise SystemExit(main())
