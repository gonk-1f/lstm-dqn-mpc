
from pathlib import Path

import numpy as np

from v4.v6_dataset import V6TrainingDataset
from v6_sb3.direct_environment import DirectVesselEnv
from v6_sb3.masked_dqn import MaskedDQN


# 1. Load one real Train voyage
dataset = V6TrainingDataset.open(
    Path("data/processed/operating_dataset_v6_1s")
)
episode = dataset.load_train()[0]

# 2. Create the V6 environment
env = DirectVesselEnv(
    reward_mode="inverse_one_plus_economic"
)

# 3. Initialize an untrained DQN neural network
model = MaskedDQN(
    env=env,
    learning_rate=0.0001,
    buffer_size=256,
    batch_size=8,
    learning_starts=10,
    policy_kwargs={"net_arch": [32, 32]},
    td_diagnostics=False,
    device="cpu",
    seed=42,
)
print("DQN device:", model.device)
try:
    # 4. Reset to the beginning of the voyage
    obs, info = env.reset(options={"episode": episode})
    mask = env.action_masks()

    # 5. Let the DQN select a legal action
    action, _ = model.masked_predict(
        obs,
        mask,
        deterministic=True,
    )
    action = int(action)

    print("Voyage:", episode.sample_id)
    print("Observation:", obs)
    print("Legal actions:", np.flatnonzero(mask))
    print("DQN selected:", action)
    print("Action delta kW:", (action - 6) * 10)
    print("Action legal:", bool(mask[action]))

    # 6. Execute one real physical transition
    next_obs, reward, terminated, truncated, info = (
        env.step(action)
    )

    print("Next observation:", next_obs)
    print("Reward:", reward)
    print("Cost CNY:", info["total_cost_cny"])

    # Inspect the actual economic ledger
    transition = info["transition"]
    ledger = transition.executed_ledger

    print("\n===== Economic cost breakdown =====")
    print("Hydrogen:", ledger.h2_cost_cny)
    print("FC degradation:", ledger.fuel_cell_degradation_cost_cny)
    print("Battery degradation:", ledger.battery_degradation_cost_cny)
    print("Shore electricity:", ledger.shore_cost_cny)
    print(
        "SOC equivalent:",
        transition.soc_equivalent_hydrogen_cost_cny
    )
    print("Total cost:", info["total_cost_cny"])

    print("Terminated:", terminated)

    # 7. Confirm no training occurred
    print("Adam updates:", model._n_updates)
    print("Replay insertions:", model.replay_buffer.insertions)

    # Continue the same voyage for up to 30 seconds
    total_cost = float(info["total_cost_cny"])
    total_reward = float(reward)
    starts = int(info["transition"].action_kw > 0)
    previous_fc = float(info["transition"].action_kw)

    print("\n===== 30-second rollout =====")

    for step in range(2, 31):
        if terminated or truncated:
            break

        mask = env.action_masks()
        action, _ = model.masked_predict(
            next_obs, mask, deterministic=True
        )

        next_obs, reward, terminated, truncated, info = (
            env.step(int(action))
        )

        fc = float(info["transition"].action_kw)
        starts += int(previous_fc == 0 and fc > 0)
        previous_fc = fc

        total_cost += float(info["total_cost_cny"])
        total_reward += float(reward)

        if step % 5 == 0 or terminated or truncated:
            print(
                f"Step {step}: "
                f"FC={fc:.0f} kW, "
                f"SOC={next_obs[0]:.6f}, "
                f"Cost={total_cost:.4f} CNY, "
                f"Status={info.get('terminal_reason')}"
            )

    print("\nTotal cost:", total_cost)
    print("Total reward:", total_reward)
    print("FC startup count:", starts)
    print("Episode status:", info.get("terminal_reason"))
    print("Adam updates:", model._n_updates)


finally:
    env.close()
