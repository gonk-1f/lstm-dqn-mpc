from __future__ import annotations

import pandas as pd


def build_learning_features(aligned_df: pd.DataFrame) -> pd.DataFrame:
    df = aligned_df.copy().sort_values(["voyage_name", "timestamp"]).reset_index(drop=True)
    if "load_left_kw" not in df.columns:
        df["load_left_kw"] = df["load_total_kw"] * 0.5
    if "load_right_kw" not in df.columns:
        df["load_right_kw"] = df["load_total_kw"] * 0.5
    if "fuel_cell_power_left_kw" not in df.columns:
        df["fuel_cell_power_left_kw"] = df["fuel_cell_power_total_kw"] * 0.5
    if "fuel_cell_power_right_kw" not in df.columns:
        df["fuel_cell_power_right_kw"] = df["fuel_cell_power_total_kw"] * 0.5
    if "battery_power_left_kw" not in df.columns:
        df["battery_power_left_kw"] = df["battery_power_total_kw"] * 0.5
    if "battery_power_right_kw" not in df.columns:
        df["battery_power_right_kw"] = df["battery_power_total_kw"] * 0.5
    df["next_load_total_kw"] = df.groupby("voyage_name")["load_total_kw"].shift(-1)
    df["prev_load_total_kw"] = df.groupby("voyage_name")["load_total_kw"].shift(1)
    df["load_ma_5"] = (
        df.groupby("voyage_name")["load_total_kw"]
        .rolling(window=5, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["speed_ma_5"] = (
        df.groupby("voyage_name")["speed_knots"]
        .rolling(window=5, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["load_ramp_kw"] = df["load_total_kw"] - df["prev_load_total_kw"].fillna(df["load_total_kw"])
    df["load_left_ratio"] = df["load_left_kw"] / df["load_total_kw"].replace(0, pd.NA)
    df["load_right_ratio"] = df["load_right_kw"] / df["load_total_kw"].replace(0, pd.NA)
    df["load_left_ratio"] = df["load_left_ratio"].fillna(0.5)
    df["load_right_ratio"] = df["load_right_ratio"].fillna(0.5)
    df["mpc_fuel_cell_ref_kw"] = df["fuel_cell_power_total_kw"]
    df["mpc_battery_ref_kw"] = df["load_total_kw"] - df["mpc_fuel_cell_ref_kw"]
    df["mpc_fuel_cell_ref_left_kw"] = df["mpc_fuel_cell_ref_kw"] * df["load_left_ratio"]
    df["mpc_fuel_cell_ref_right_kw"] = df["mpc_fuel_cell_ref_kw"] * df["load_right_ratio"]
    df["mpc_battery_ref_left_kw"] = df["mpc_battery_ref_kw"] * df["load_left_ratio"]
    df["mpc_battery_ref_right_kw"] = df["mpc_battery_ref_kw"] * df["load_right_ratio"]
    df["tracking_error_baseline_kw"] = (
        df["fuel_cell_power_total_kw"] + df["battery_power_total_kw"] - df["load_total_kw"]
    )
    df["tracking_error_left_baseline_kw"] = (
        df["fuel_cell_power_left_kw"] + df["battery_power_left_kw"] - df["load_left_kw"]
    )
    df["tracking_error_right_baseline_kw"] = (
        df["fuel_cell_power_right_kw"] + df["battery_power_right_kw"] - df["load_right_kw"]
    )
    return df.dropna(subset=["next_load_total_kw"]).reset_index(drop=True)


def split_train_eval(feature_df: pd.DataFrame, eval_ratio: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    voyages = sorted(feature_df["voyage_name"].unique())
    split_index = max(1, int(round(len(voyages) * (1 - eval_ratio))))
    train_names = set(voyages[:split_index])
    train_df = feature_df[feature_df["voyage_name"].isin(train_names)].reset_index(drop=True)
    eval_df = feature_df[~feature_df["voyage_name"].isin(train_names)].reset_index(drop=True)
    if eval_df.empty:
        eval_df = train_df.copy()
    return train_df, eval_df
