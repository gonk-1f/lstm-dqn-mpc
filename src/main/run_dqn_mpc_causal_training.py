from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dqn.agents.dqn_agent import DQNTrainConfig
import train_dqn_mpc_mlp as training


NUM_TRAINING_ROUNDS = 1


def main() -> None:
    config = DQNTrainConfig()
    split = training.load_voyage_split()
    runtime = training.create_training_runtime(config)
    base_config = training.build_formal_mpc_config()
    output_dir = REPO_ROOT / "outputs" / "dqn_mpc_mlp_causal_formal_rounds"
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)

    def load_train(voyage_id: str):
        return training.load_voyage_loads("train", voyage_id, split=split)

    def load_validation(voyage_id: str):
        return training.load_voyage_loads("validation", voyage_id, split=split)

    rounds = training.train_complete_voyage_rounds(
        num_training_rounds=NUM_TRAINING_ROUNDS,
        voyage_ids=split.train_voyages,
        load_voyage=load_train,
        base_config=base_config,
        runtime=runtime,
    )
    for round_summary in rounds:
        round_dir = output_dir / f"round_{round_summary['round_id']}"
        round_dir.mkdir()
        runtime.agent.save(round_dir / f"model_round{round_summary['round_id']}.pt")
        validation = training.validate_voyages(voyage_ids=split.validation_voyages, load_voyage=load_validation, base_config=base_config, agent=runtime.agent)
        pd.DataFrame(validation["voyages"]).to_csv(round_dir / "validation_by_voyage.csv", index=False)
        (round_dir / "validation_summary.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        round_summary["validation"] = validation
    (output_dir / "training_summary.json").write_text(json.dumps({"num_training_rounds": NUM_TRAINING_ROUNDS, "rounds": rounds, "test_voyages": []}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
