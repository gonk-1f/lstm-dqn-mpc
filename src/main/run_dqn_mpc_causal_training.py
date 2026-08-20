from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dqn.agents.dqn_agent import DQNTrainConfig
import train_dqn_mpc_mlp as training


MAX_STEPS = 1_245_456


def main() -> None:
    config = DQNTrainConfig(
        max_steps=MAX_STEPS,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = (
        REPO_ROOT
        / "outputs"
        / f"dqn_mpc_causal_1epoch_{timestamp}"
    )

    runtime, summary = training.train_dqn_mpc_mlp(
        config=config
    )

    output_paths = training.write_baseline_outputs(
        runtime=runtime,
        summary=summary,
        output_dir=output_dir,
    )

    print("\nTraining finished.")
    print("Output directory:", output_dir)

    for name, path in output_paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()