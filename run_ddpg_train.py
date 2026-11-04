"""Run formal DDPG training with the interpreter selected in PyCharm."""
import os
from pathlib import Path
import subprocess
import sys


def main() -> int:
    project_root = Path(__file__).resolve().parent
    env = os.environ.copy()
    python_paths = [str(project_root), str(project_root / "src")]
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_paths)

    command = [
        sys.executable, "-u", "-m", "rl_zoo3.train",
        "--algo", "ddpg",
        "--env", "AggregateShipDDPG-v0",
        "--gym-packages", "experiments.sb3_manual.ddpg_aggregate.zoo_register",
        "--conf-file", "experiments/sb3_manual/ddpg_aggregate/zoo_ddpg.yml",
        "--device", "cuda",
        "--eval-freq", "-1",
        "--seed", "42",
        "--log-folder", "outputs/ddpg_fixed_parent_005_seg01_500ep_linear_v2_tau001",
        "--save-replay-buffer",
    ]
    # Inherit stdin/stdout/stderr so training stays in PyCharm's Run console.
    return subprocess.run(command, cwd=project_root, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
