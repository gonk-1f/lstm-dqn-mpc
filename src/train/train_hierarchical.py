from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from dqn.agents.dqn_agent import DQNTrainConfig
from mpc.controllers.upper_mpc import UpperMPCController, build_casadi_mpc_config
from train.train_dqn import train_dual_side_dqn, train_simple_dqn
from utils.feature_engineering import split_train_eval
from utils.config_loader import get_project_root
from utils.multirate import MultiRateConfig, expand_multirate_dataset, summarize_multirate_feasibility


def prepare_phase1_hierarchy_data(
    feature_csv: str | Path,
    output_dir: str | Path,
    eval_ratio: float = 0.2,
    multirate_config: MultiRateConfig | None = None,
    solve_stride: int = 1,
) -> dict:
    feature_path = Path(feature_csv)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    feature_df = pd.read_csv(feature_path)
    multirate_config = multirate_config or MultiRateConfig()
    upper_controller = UpperMPCController(
        casadi_config=build_casadi_mpc_config(
            project_root=get_project_root(__file__),
            dt_hours=multirate_config.mpc_update_seconds / 3600.0,
        )
    )
    hierarchical_df = upper_controller.annotate_dataset(feature_df, solve_stride=solve_stride)
    hierarchical_dqn_df = expand_multirate_dataset(hierarchical_df, multirate_config)
    train_df, eval_df = split_train_eval(hierarchical_dqn_df, eval_ratio=eval_ratio)

    hierarchical_csv = output_root / "hierarchical_features.csv"
    hierarchical_dqn_csv = output_root / "hierarchical_features_dqn_timescale.csv"
    train_csv = output_root / "hierarchical_train.csv"
    eval_csv = output_root / "hierarchical_eval.csv"
    hierarchical_df.to_csv(hierarchical_csv, index=False, encoding="utf-8-sig")
    hierarchical_dqn_df.to_csv(hierarchical_dqn_csv, index=False, encoding="utf-8-sig")
    train_df.to_csv(train_csv, index=False, encoding="utf-8-sig")
    eval_df.to_csv(eval_csv, index=False, encoding="utf-8-sig")

    summary = {
        "rows": len(hierarchical_df),
        "train_rows": len(train_df),
        "eval_rows": len(eval_df),
        "upper_mpc": upper_controller.describe(),
        "hierarchical_csv": str(hierarchical_csv),
        "hierarchical_dqn_csv": str(hierarchical_dqn_csv),
        "train_csv": str(train_csv),
        "eval_csv": str(eval_csv),
        "solve_stride": solve_stride,
        "timing": summarize_multirate_feasibility(multirate_config),
    }
    with (output_root / "hierarchy_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def train_phase1_hierarchy(
    feature_csv: str | Path,
    output_dir: str | Path,
    dqn_config: DQNTrainConfig | None = None,
    eval_ratio: float = 0.2,
    multirate_config: MultiRateConfig | None = None,
    env_type: str = "simple",
    solve_stride: int = 1,
) -> dict:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    multirate_config = multirate_config or MultiRateConfig()
    data_summary = prepare_phase1_hierarchy_data(
        feature_csv=feature_csv,
        output_dir=output_root,
        eval_ratio=eval_ratio,
        multirate_config=multirate_config,
        solve_stride=solve_stride,
    )
    train_csv = Path(data_summary["train_csv"])
    eval_csv = Path(data_summary["eval_csv"])

    env_type = env_type.lower()
    if env_type == "dual_side":
        dqn_dir = output_root / "dqn_dual_side"
        dqn_summary = train_dual_side_dqn(
            train_csv=train_csv,
            output_dir=dqn_dir,
            config=dqn_config or DQNTrainConfig(),
        )
    else:
        dqn_dir = output_root / "dqn_simple"
        dqn_summary = train_simple_dqn(
            train_csv=train_csv,
            output_dir=dqn_dir,
            config=dqn_config or DQNTrainConfig(),
        )

    summary = {
        "env_type": env_type,
        **data_summary,
        "dqn_dir": str(dqn_dir),
        "dqn": dqn_summary,
    }
    with (output_root / "hierarchy_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary
