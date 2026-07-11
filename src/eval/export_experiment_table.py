from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "| experiment |\n| --- |\n"
    columns = list(df.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for _, row in df.iterrows():
        values = [str(row[col]) for col in columns]
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *body]) + "\n"


def export_experiment_table(summary_dir: str | Path, output_csv: str | Path, output_md: str | Path) -> pd.DataFrame:
    summary_root = Path(summary_dir)
    rows: list[dict] = []
    for json_path in sorted(summary_root.glob("phase1_eval_summary_v*.json")):
        if json_path.stem.endswith("_dqn") or json_path.stem.endswith("_baseline") or json_path.stem.endswith("_best"):
            continue
        with json_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        rows.append(
            {
                "experiment": json_path.stem.replace("phase1_eval_summary_", ""),
                "reward_mean": payload.get("dqn", {}).get("reward_mean"),
                "tracking_error_mae_kw": payload.get("dqn", {}).get("tracking_error_mae_kw"),
                "total_balance_error_mae_kw": payload.get("dqn", {}).get("total_balance_error_mae_kw"),
                "tracking_improvement_kw": payload.get("improvement", {}).get("tracking_error_mae_kw"),
                "balance_improvement_kw": payload.get("improvement", {}).get("total_balance_error_mae_kw"),
                "model_path": payload.get("model_path"),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by=["tracking_improvement_kw", "reward_mean"], ascending=[False, False]).reset_index(drop=True)

    output_csv_path = Path(output_csv)
    output_md_path = Path(output_md)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    output_md_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    output_md_path.write_text(_dataframe_to_markdown(df), encoding="utf-8")
    return df
