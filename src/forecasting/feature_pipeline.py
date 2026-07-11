from __future__ import annotations

import pandas as pd

from forecasting.lstm_load_predictor import add_time_features, base_rows


ROLLING_WINDOWS = [3, 6, 12, 18]
EXPLICIT_RAMP_LAGS = [1, 3, 6]
RELATIVE_ROLLING_WINDOWS = [6, 18]
ROLLING_1S_WINDOWS = [15, 60, 180]
EXPLICIT_1S_RAMP_LAGS = [1, 5, 30, 60]
RELATIVE_1S_ROLLING_WINDOWS = [60, 180]
DELTA_FEATURE_MAP = [
    ("load_total_kw", "delta_load_total"),
    ("load_left_kw", "delta_load_left"),
    ("load_right_kw", "delta_load_right"),
    ("speed_knots", "delta_speed"),
]


def _feature_window_spec(feature_set: str) -> tuple[list[int], list[int], list[int], int]:
    if feature_set == "rolling_1s":
        return ROLLING_1S_WINDOWS, EXPLICIT_1S_RAMP_LAGS, RELATIVE_1S_ROLLING_WINDOWS, 60
    return ROLLING_WINDOWS, EXPLICIT_RAMP_LAGS, RELATIVE_ROLLING_WINDOWS, 6


def clean_total_load_feature_columns() -> list[str]:
    """Feature set for the 66-voyage total-load LSTM dataset."""
    features = [
        "load_total_kw",
        "time_sin",
        "time_cos",
        "delta_load_total",
        "ramp_1_load_total",
        "ramp_3_load_total",
        "ramp_6_load_total",
        "slope_6_load_total",
        "load_minus_rolling_mean_w6",
        "load_minus_rolling_mean_w18",
    ]
    for window in ROLLING_WINDOWS:
        for stat in ["mean", "std", "min", "max", "range", "ramp"]:
            features.append(f"rolling_{stat}_load_total_w{window}")
    return features


def clean_total_load_feature_columns_1s() -> list[str]:
    """Feature set for the 1 s total-load LSTM dataset.

    The windows are second-scale physical windows, not the 30 s point windows
    used by ``clean_total_load_feature_columns``.
    """
    features = [
        "load_total_kw",
        "time_sin",
        "time_cos",
        "delta_load_total",
    ]
    for lag in EXPLICIT_1S_RAMP_LAGS:
        features.append(f"ramp_{lag}_load_total")
    features.extend(
        [
            "slope_60_load_total",
            "load_minus_rolling_mean_w60",
            "load_minus_rolling_mean_w180",
        ]
    )
    for window in ROLLING_1S_WINDOWS:
        for stat in ["mean", "std", "min", "max", "range", "ramp"]:
            features.append(f"rolling_{stat}_load_total_w{window}")
    return features


def clean_total_load_speed_feature_columns() -> list[str]:
    """Total-load feature set plus AIS speed inputs."""
    features = clean_total_load_feature_columns()
    insert_at = features.index("time_sin") if "time_sin" in features else 1
    return features[:insert_at] + ["speed_knots", "delta_speed"] + features[insert_at:]


def prepare_lstm_features(df: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    out = add_time_features(base_rows(df))
    group_key = "voyage_name" if "voyage_name" in out.columns else "voyage_id" if "voyage_id" in out.columns else None
    group_indices = [list(idx) for idx in out.groupby(group_key, sort=False).groups.values()] if group_key else [list(out.index)]
    for source_col, delta_col in DELTA_FEATURE_MAP:
        if source_col not in out.columns:
            continue
        out[delta_col] = 0.0
        if feature_set in {"delta", "rolling", "rolling_1s"}:
            for idx_list in group_indices:
                out.loc[idx_list, delta_col] = out.loc[idx_list, source_col].astype(float).diff().fillna(0.0)
    if feature_set in {"rolling", "rolling_1s"} and "load_total_kw" in out.columns:
        rolling_windows, ramp_lags, relative_windows, slope_lag = _feature_window_spec(feature_set)
        for window in rolling_windows:
            for stat in ["mean", "std", "min", "max", "range", "ramp"]:
                out[f"rolling_{stat}_load_total_w{window}"] = 0.0
        for lag in ramp_lags:
            out[f"ramp_{lag}_load_total"] = 0.0
        out[f"slope_{slope_lag}_load_total"] = 0.0
        for window in relative_windows:
            out[f"load_minus_rolling_mean_w{window}"] = 0.0
        for idx_list in group_indices:
            series = out.loc[idx_list, "load_total_kw"].astype(float)
            ramp_by_lag: dict[int, pd.Series] = {}
            for lag in ramp_lags:
                ramp = (series - series.shift(int(lag))).fillna(0.0)
                ramp_by_lag[int(lag)] = ramp
                out.loc[idx_list, f"ramp_{lag}_load_total"] = ramp
            out.loc[idx_list, f"slope_{slope_lag}_load_total"] = ramp_by_lag[slope_lag] / float(slope_lag)
            for window in rolling_windows:
                rolling = series.rolling(window, min_periods=1)
                roll_min = rolling.min()
                roll_max = rolling.max()
                roll_mean = rolling.mean()
                first_in_window = rolling.apply(lambda values: float(values[0]), raw=True)
                out.loc[idx_list, f"rolling_mean_load_total_w{window}"] = roll_mean
                out.loc[idx_list, f"rolling_std_load_total_w{window}"] = rolling.std().fillna(0.0)
                out.loc[idx_list, f"rolling_min_load_total_w{window}"] = roll_min
                out.loc[idx_list, f"rolling_max_load_total_w{window}"] = roll_max
                out.loc[idx_list, f"rolling_range_load_total_w{window}"] = roll_max - roll_min
                out.loc[idx_list, f"rolling_ramp_load_total_w{window}"] = series - first_in_window
                if window in relative_windows:
                    out.loc[idx_list, f"load_minus_rolling_mean_w{window}"] = series - roll_mean
    return out.reset_index(drop=True)
