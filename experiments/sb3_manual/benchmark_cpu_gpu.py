
from pathlib import Path
from time import perf_counter
from statistics import median

import numpy as np
import torch

from stable_baselines3.common.logger import configure
from v4.v6_dataset import V6TrainingDataset
from v6_sb3.direct_environment import DirectVesselEnv
from v6_sb3.masked_dqn import MaskedDQN

torch.set_num_threads(1)

# Use real transitions only to fill the Replay Buffer.
# This is a speed test, not a learning experiment.
dataset = V6TrainingDataset.open(
    Path("data/processed/operating_dataset_v6_1s")
)
episode = dataset.load_train()[0]
data = []

env = DirectVesselEnv(
    reward_mode="inverse_one_plus_economic"
)

try:
    obs, _ = env.reset(options={"episode": episode})

    for _ in range(128):
        mask = env.action_masks()
        action = int(np.flatnonzero(mask)[0])

        next_obs, reward, done, truncated, info = (
            env.step(action)
        )

        data.append((
            obs.copy(),
            mask.copy(),
            action,
            next_obs.copy(),
            reward,
            done,
            info["next_action_mask"].copy(),
            info["terminal_reason"],
        ))

        if done or truncated:
            obs, _ = env.reset(options={"episode": episode})
        else:
            obs = next_obs
finally:
    env.close()


def synchronize(device):
    if device == "cuda":
        torch.cuda.synchronize()


def benchmark(device):
    env = DirectVesselEnv(
        reward_mode="inverse_one_plus_economic"
    )

    model = MaskedDQN(
        env=env,
        learning_rate=1e-4,
        buffer_size=512,
        batch_size=64,
        learning_starts=0,
        train_freq=32,
        gradient_steps=1,
        target_update_interval=10000,
        gamma=1.0,
        policy_kwargs={"net_arch": [256, 256]},
        td_diagnostics=False,
        device=device,
        seed=42,
    )

    model.set_logger(
        configure(folder=None, format_strings=[])
    )

    try:
        # Fill both CPU and GPU Replay Buffers
        # with identical real transitions.
        for obs, mask, action, next_obs, reward, done, next_mask, reason in data:
            model.replay_buffer.add(
                obs[None, :],
                next_obs[None, :],
                np.array([[action]]),
                np.array([reward], dtype=np.float32),
                np.array([done], dtype=np.float32),
                [{
                    "next_action_mask": next_mask,
                    "terminal_reason": reason,
                }],
            )

        first_obs, first_mask = data[0][:2]

        # Warm up inference
        for _ in range(20):
            model.masked_predict(
                first_obs, first_mask,
                deterministic=True
            )
        synchronize(device)

        # Measure 500 inference calls
        start = perf_counter()
        for _ in range(500):
            model.masked_predict(
                first_obs, first_mask,
                deterministic=True
            )
        synchronize(device)
        inference_ms = (perf_counter() - start) * 1000 / 500

        # Warm up Adam
        model.train(gradient_steps=3, batch_size=64)
        synchronize(device)

        # Three repeated measurements,
        # each with 30 Adam updates
        results = []

        for _ in range(3):
            synchronize(device)
            start = perf_counter()

            model.train(
                gradient_steps=30,
                batch_size=64
            )

            synchronize(device)
            results.append(
                (perf_counter() - start) * 1000 / 30
            )

        adam_ms = median(results)

        print(f"\n===== {device.upper()} =====")
        print(f"Inference: {inference_ms:.4f} ms/call")
        print(f"Adam: {adam_ms:.4f} ms/update")
        print(f"Loss: {model.last_training_loss:.6f}")
        print(f"Adam updates: {model._n_updates}")

        return inference_ms, adam_ms

    finally:
        env.close()


cpu_infer, cpu_adam = benchmark("cpu")

if torch.cuda.is_available():
    gpu_infer, gpu_adam = benchmark("cuda")

    print("\n===== Speed comparison =====")
    print(f"Inference CPU/GPU: {cpu_infer/gpu_infer:.2f}x")
    print(f"Adam CPU/GPU: {cpu_adam/gpu_adam:.2f}x")
else:
    print("CUDA unavailable")
