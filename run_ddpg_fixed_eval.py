"""Run fixed-Train-voyage diagnostics with PyCharm's selected interpreter."""
import os
from pathlib import Path
import subprocess
import sys


def main() -> int:
    project_root = Path(__file__).resolve().parent
    model_path = project_root / "outputs/ddpg_fixed_parent_005_seg01_500ep_economic_v3/ddpg/AggregateShipDDPG-v0_1/AggregateShipDDPG-v0.zip"
    env = os.environ.copy()
    paths = [str(project_root), str(project_root / "src")]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    command = [
        sys.executable, "-u", "-m", "experiments.sb3_manual.ddpg_aggregate.fixed_train_eval",
        "--model-path", str(model_path),
        "--output-root", str(project_root / "outputs/ddpg_fixed_parent_005_evaluation_economic_v3"),
        "--sample-id", "parent_005_seg01", "--episode", "500", "--device", "cpu", "--seed", "42",
    ]
    return subprocess.run(command, cwd=project_root, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
