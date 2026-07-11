from __future__ import annotations

import json
from pathlib import Path

from eval.evaluate_dual_side import evaluate_dual_side_dqn, rollout_dual_side_reference
from eval.evaluate_tracking import evaluate_simple_tracking
from plots.plot_hierarchical_comparison import plot_hierarchical_comparison


def evaluate_phase1_hierarchy(
    eval_csv: str | Path,
    model_path: str | Path,
    output_path: str | Path,
    env_type: str = "simple",
) -> dict:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    env_type = env_type.lower()

    if env_type == "dual_side":
        dqn_output = output.with_name(f"{output.stem}_dqn{output.suffix}")
        baseline_output = output.with_name(f"{output.stem}_baseline{output.suffix}")
        dqn_summary = evaluate_dual_side_dqn(eval_csv=eval_csv, model_path=model_path, output_path=dqn_output)
        baseline_summary = rollout_dual_side_reference(eval_csv=eval_csv, output_path=baseline_output)
        summary = {
            "env_type": env_type,
            "dqn": dqn_summary,
            "baseline_center_split": baseline_summary,
            "improvement": {
                "tracking_error_mae_kw": float(
                    baseline_summary["tracking_error_mae_kw"] - dqn_summary["tracking_error_mae_kw"]
                ),
                "left_tracking_error_mae_kw": float(
                    baseline_summary["left_tracking_error_mae_kw"] - dqn_summary["left_tracking_error_mae_kw"]
                ),
                "right_tracking_error_mae_kw": float(
                    baseline_summary["right_tracking_error_mae_kw"] - dqn_summary["right_tracking_error_mae_kw"]
                ),
                "total_balance_error_mae_kw": float(
                    baseline_summary["total_balance_error_mae_kw"] - dqn_summary["total_balance_error_mae_kw"]
                ),
            },
            "artifacts": {
                "dqn_json": str(dqn_output),
                "dqn_csv": str(dqn_output.with_suffix(".csv")),
                "baseline_json": str(baseline_output),
                "baseline_csv": str(baseline_output.with_suffix(".csv")),
            },
        }
        comparison_png = output.with_name(f"{output.stem}_comparison.png")
        plot_hierarchical_comparison(
            dqn_csv=dqn_output.with_suffix(".csv"),
            baseline_csv=baseline_output.with_suffix(".csv"),
            output_png=comparison_png,
        )
        summary["artifacts"]["comparison_png"] = str(comparison_png)
    else:
        dqn_summary = evaluate_simple_tracking(eval_csv=eval_csv, model_path=model_path, output_path=output)
        summary = {
            "env_type": env_type,
            "dqn": dqn_summary,
            "artifacts": {
                "dqn_json": str(output),
                "dqn_csv": str(output.with_suffix(".csv")),
            },
        }

    with output.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary
