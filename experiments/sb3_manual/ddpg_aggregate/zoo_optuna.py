"""Limit DDPG sampling, then delegate CLI, learning and Optuna to RL Zoo."""
import numpy as np
from stable_baselines3.common.noise import NormalActionNoise
from rl_zoo3.hyperparams_opt import HYPERPARAMS_SAMPLER


def sample_ddpg_params(trial, n_actions, n_envs, additional_args):
    tau = trial.suggest_float("tau", 0.005, 0.15, log=True)
    noise_std = trial.suggest_float("noise_std", 0.03, 0.12)
    return {
        "tau": tau,
        "action_noise": NormalActionNoise(
            mean=np.zeros(n_actions), sigma=noise_std * np.ones(n_actions),
        ),
    }


# Register before importing train (which imports ExperimentManager).
HYPERPARAMS_SAMPLER["ddpg"] = sample_ddpg_params

from rl_zoo3.train import train  # noqa: E402


if __name__ == "__main__":
    train()
