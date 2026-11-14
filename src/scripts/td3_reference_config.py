"""Training settings for the native SB3 2.2.1 comparison."""

import os

from td3_config import DATA_ROOT as PROJECT_DATA_ROOT
from td3_config import PROJECT_ROOT, TRAIN_WINDOW_STEPS

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "td3_sb3_221_reference_window5000_100ep"
DATA_ROOT = PROJECT_DATA_ROOT
if "SHIP_EMS_DATA_ROOT" not in os.environ and not DATA_ROOT.is_dir():
    # Worktrees do not check out the ignored canonical 1 s dataset.
    shared_data_root = PROJECT_ROOT.parent / "v6-sb3-baseline" / "data" / "processed" / DATA_ROOT.name
    if shared_data_root.is_dir():
        DATA_ROOT = shared_data_root

EPISODES = 100
TOTAL_TIMESTEPS = EPISODES * TRAIN_WINDOW_STEPS
DIAGNOSTIC_INTERVAL = 20
ACTION_NOISE_SIGMA = 0.05

TD3_KWARGS = {
    "learning_rate": 1e-4,
    "buffer_size": 200_000,
    "learning_starts": 25_000,
    "batch_size": 64,
    "gamma": 1.0,
    "tau": 0.005,
    "policy_delay": 2,
    "target_policy_noise": 0.1,
    "target_noise_clip": 0.2,
    "policy_kwargs": {"net_arch": [256, 256]},
    "train_freq": (1, "episode"),
    "gradient_steps": -1,
}
