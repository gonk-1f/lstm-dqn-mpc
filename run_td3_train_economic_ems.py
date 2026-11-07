"""PyCharm entry point for the versioned economic TD3 EMS experiment."""

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main() -> int:
    env = os.environ.copy()
    paths = [str(ROOT), str(ROOT / "src")]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    command = [
        sys.executable, "-u", "-m", "rl_zoo3.train",
        "--algo", "td3",
        "--env", "AggregateShipTD3Economic-v0",
        "--gym-packages", "experiments.sb3_manual.td3_aggregate.economic_ems_registration",
        "--conf-file", "experiments/sb3_manual/td3_aggregate/zoo_td3_economic_ems.yml",
        "--device", "cuda",
        "--num-threads", "1",
        "--eval-freq", "-1",
        "--seed", "42",
        "--log-folder", "outputs/td3_economic_ems_v3_soc5_500ep",
        "--tensorboard-log", "outputs/td3_economic_ems_v3_soc5_500ep/tensorboard",
        "--log-interval", "-2",
        "--save-replay-buffer",
    ]
    child = subprocess.Popen(command, cwd=ROOT, env=env)
    try:
        return child.wait()
    except KeyboardInterrupt:
        print("Waiting for RL Zoo to save the economic TD3 model and Replay Buffer...", flush=True)
        return child.wait()


if __name__ == "__main__":
    raise SystemExit(main())
