"""Start the economic-reward native RL Zoo Optuna search from PyCharm."""
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
        sys.executable, "-u", "-m", "experiments.sb3_manual.ddpg_aggregate.zoo_optuna",
        "--algo", "ddpg",
        "--env", "AggregateShipDDPG-v0",
        "--gym-packages", "experiments.sb3_manual.ddpg_aggregate.zoo_register",
        "--conf-file", "experiments/sb3_manual/ddpg_aggregate/zoo_ddpg_optuna.yml",
        "--optimize-hyperparameters",
        "--n-timesteps", "77760",
        "--n-trials", "20",
        "--max-total-trials", "20",
        "--n-jobs", "1",
        "--sampler", "tpe",
        "--pruner", "none",
        "--n-startup-trials", "10",
        "--seed", "42",
        "--eval-episodes", "8",
        "--n-eval-envs", "1",
        "--n-evaluations", "2",
        "--eval-env-kwargs", "data_root:'data/processed/operating_dataset_v6_1s'", "split:'validation'",
        "--storage", "sqlite:///outputs/ddpg_optuna_economic_v3/study.db",
        "--study-name", "ddpg_tau_noise_economic_negative_no_start_stop_v5_obs5_huber1",
        "--optimization-log-path", "outputs/ddpg_optuna_economic_v3/evaluations",
        "--log-folder", "outputs/ddpg_optuna_economic_v3/logs",
        "--device", "cuda",
        "--no-optim-plots",
    ]
    return subprocess.run(command, cwd=project_root, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
