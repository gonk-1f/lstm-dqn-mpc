
from __future__ import annotations

from datetime import datetime
import hashlib

import torch

from stable_baselines3 import TD3

from td3_config import (
    DATA_ROOT,
    DEVICE,
    OUTPUT_ROOT,
    SEED,
    evaluate_profiles,
    load_dataset_bundle,
    save_json,
    summarize_validation,
    write_csv,
)


def find_latest_model():

    if not OUTPUT_ROOT.is_dir():
        raise FileNotFoundError(
            f"训练输出目录不存在：{OUTPUT_ROOT}"
        )

    run_dirs = sorted(
        [
            p for p in OUTPUT_ROOT.glob("run_*")
            if p.is_dir()
        ],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    for run_dir in run_dirs:

        best_model = run_dir / "best_model.zip"
        last_model = run_dir / "last_model.zip"

        if best_model.is_file():
            return run_dir, best_model

        if last_model.is_file():
            return run_dir, last_model

    raise FileNotFoundError(
        "没有找到训练模型，请先运行 train_td3.py。"
    )


def main():

    torch.set_num_threads(1)

    # --------------------------------------------------------
    # 1. Load Validation8
    # --------------------------------------------------------

    bundle = load_dataset_bundle(DATA_ROOT)

    # --------------------------------------------------------
    # 2. Load trained official SB3 TD3
    # --------------------------------------------------------

    run_dir, model_path = find_latest_model()

    model = TD3.load(
        str(model_path),
        device=DEVICE,
    )

    model.policy.set_training_mode(False)

    print("=" * 72)
    print("TD3 deterministic evaluation")
    print(f"Model: {model_path}")
    print(f"Validation voyages: {len(bundle.validation_ids)}")
    print("=" * 72)

    # --------------------------------------------------------
    # 3. Deterministic Validation8
    # --------------------------------------------------------

    results, step_rows = evaluate_profiles(
        model=model,

        profiles=bundle.validation_profiles,

        sample_ids=bundle.validation_ids,

        seed=SEED,

        collect_step_rows=True,
    )

    metrics = summarize_validation(results)

    metrics.update({

        "model_path": str(model_path.resolve()),

        "model_sha256": hashlib.sha256(
            model_path.read_bytes()
        ).hexdigest(),

        "data_root": str(DATA_ROOT.resolve()),

        "split": "validation",

        "deterministic": True,

        "exploration_noise": None,

        "gradient_updates_during_evaluation": 0,

        "replay_insertions_during_evaluation": 0,

        "test_payloads_loaded": False,

        "validation_ids": bundle.validation_ids,
    })

    # --------------------------------------------------------
    # 4. Save results
    # --------------------------------------------------------

    stamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_dir = (
        run_dir
        / "final_validation"
        / f"validation_{stamp}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    write_csv(
        output_dir / "validation_summary.csv",
        results,
    )

    write_csv(
        output_dir / "validation_step_data.csv",
        step_rows,
    )

    save_json(
        output_dir / "validation_metrics.json",
        {
            "metrics": metrics,
            "voyages": results,
        },
    )

    # --------------------------------------------------------
    # 5. Print results
    # --------------------------------------------------------

    print("=" * 72)
    print("TD3 Validation8 evaluation completed")

    print(
        f"Completed: "
        f"{metrics['completed_voyages']}/8"
    )

    print(
        f"Completion rate: "
        f"{metrics['completion_rate']:.3f}"
    )

    print(
        f"Mean episode reward: "
        f"{metrics['mean_episode_reward']}"
    )

    print(
        "Mean economic cost of completed voyages: "
        f"{metrics['mean_total_economic_cost_completed_only_rmb']}"
    )

    print(f"Results: {output_dir}")

    print(
        "No gradient updates or replay buffer insertions "
        "during evaluation."
    )

    print("=" * 72)


if __name__ == "__main__":
    main()
