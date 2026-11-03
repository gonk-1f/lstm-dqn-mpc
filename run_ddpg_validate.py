"""Run DDPG Validation8 with the interpreter selected in PyCharm."""
import os
from pathlib import Path
import subprocess
import sys


def main() -> int:
    project_root = Path(__file__).resolve().parent
    model_path = project_root / "outputs/ddpg_train_500ep_economic_v1/ddpg/AggregateShipDDPG-v0_1/AggregateShipDDPG-v0.zip"
    env = os.environ.copy()
    python_paths = [str(project_root), str(project_root / "src")]
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_paths)

    command = [
        sys.executable, "-u", "-m", "experiments.sb3_manual.ddpg_aggregate.validate",
        "--model-path", str(model_path),
        "--output-root", str(project_root / "outputs/ddpg_validation_economic_v1"),
        "--device", "cpu",
        "--seed", "42",
        "--episodes", "8",
    ]
    # Inherit stdin/stdout/stderr so progress appears in PyCharm's Run console.
    return subprocess.run(command, cwd=project_root, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
