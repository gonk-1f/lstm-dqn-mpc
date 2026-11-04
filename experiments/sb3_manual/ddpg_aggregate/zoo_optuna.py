"""Limit DDPG sampling, then delegate CLI, learning and Optuna to RL Zoo."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import numpy as np
from stable_baselines3.common.noise import NormalActionNoise
from rl_zoo3.hyperparams_opt import HYPERPARAMS_SAMPLER

from .economic_reward import REWARD_VERSION

DEFAULT_STUDY_NAME = "ddpg_tau_noise_" + REWARD_VERSION.removeprefix("ddpg_")
DEFAULT_OPTUNA_ROOT = "outputs/ddpg_optuna_economic_v3"


def prepare_optimization_args(argv):
    """Isolate the reward's native Zoo study before starting any optimization."""
    args = list(argv)
    if "--help" in args or "-h" in args:
        return args
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-optimize", "--optimize-hyperparameters", action="store_true")
    parser.add_argument("--study-name")
    parser.add_argument("--storage")
    parser.add_argument("-f", "--log-folder")
    parser.add_argument("--optimization-log-path")
    options, _ = parser.parse_known_args(args)
    if options.study_name is not None and not options.study_name.startswith(DEFAULT_STUDY_NAME):
        raise ValueError(
            f"New economic reward requires a Study name beginning with {DEFAULT_STUDY_NAME!r}; "
            "old reward studies cannot be resumed"
        )
    if not options.optimize_hyperparameters:
        return args
    defaults = (
        ("--study-name", options.study_name, DEFAULT_STUDY_NAME),
        ("--storage", options.storage, f"sqlite:///{DEFAULT_OPTUNA_ROOT}/study.db"),
        ("--log-folder", options.log_folder, f"{DEFAULT_OPTUNA_ROOT}/logs"),
        ("--optimization-log-path", options.optimization_log_path, f"{DEFAULT_OPTUNA_ROOT}/evaluations"),
    )
    for flag, supplied, value in defaults:
        if supplied is None:
            args.extend((flag, value))
    return args


def sample_ddpg_params(trial, n_actions, n_envs, additional_args):
    tau = trial.suggest_float("tau", 0.005, 0.06, log=True)
    noise_std = trial.suggest_float("noise_std", 0.07, 0.15)
    return {
        "tau": tau,
        "action_noise": NormalActionNoise(
            mean=np.zeros(n_actions), sigma=noise_std * np.ones(n_actions),
        ),
    }


# Register before importing train (which imports ExperimentManager).
HYPERPARAMS_SAMPLER["ddpg"] = sample_ddpg_params

from rl_zoo3.train import train  # noqa: E402


def main():
    sys.argv[1:] = prepare_optimization_args(sys.argv[1:])
    train()


if __name__ == "__main__":
    main()
