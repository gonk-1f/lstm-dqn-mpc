from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dqn.agents.dqn_agent import DQNTrainConfig
import train_dqn_mpc_mlp as training


MODEL_DIR = (
    REPO_ROOT
    / "outputs"
    / "dqn_mpc_mlp_causal_1epoch_20260820"
)

MODEL_PATH = MODEL_DIR / "model_final.pt"


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    split = training.load_voyage_split()

    runtime = training.create_training_runtime(
        DQNTrainConfig()
    )

    runtime.agent.load(MODEL_PATH)
    runtime.agent.q_net.eval()

    base_config = training.build_formal_mpc_config()

    def load_validation(voyage_id: str):
        return training.load_voyage_loads(
            "validation",
            voyage_id,
            split=split,
        )

    result = training.validate_voyages(
        voyage_ids=split.validation_voyages,
        load_voyage=load_validation,
        base_config=base_config,
        agent=runtime.agent,
    )

    frame = pd.DataFrame(result["voyages"])

    columns = [
        "voyage_id",
        "completed",
        "solver_failure_count",
        "episode_steps",
        "episode_reward",
        "mean_reward_per_step",
        "final_soc",
        "action_count_A0",
        "action_count_A1",
        "action_count_A2",
        "action_count_A3",
    ]

    frame = frame[columns]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_path = (
        MODEL_DIR
        / f"validation_recheck_{timestamp}.csv"
    )

    frame.to_csv(output_path, index=False)

    completed = int(frame["completed"].sum())

    print("\nValidation finished.")
    print(f"Completed: {completed}/{len(frame)}")
    print(
        "Failed:",
        frame.loc[
            ~frame["completed"],
            "voyage_id",
        ].tolist(),
    )
    print("Saved:", output_path)


if __name__ == "__main__":
    main()