
"""
Ship EMS data loader.

Supports:
1. 1-second load dataset
2. 30-second operational dataset

Required columns:
- timestamp
- load_total_kw

Original columns are preserved.
Actual time intervals are preserved.
No interpolation or clipping is performed.
"""

from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "timestamp",
    "load_total_kw",
}


def load_voyage_csv(
    file_path: str | Path,
    expected_interval_s: float | None = None,
) -> pd.DataFrame:
    """
    Load one voyage CSV file.

    Parameters
    ----------
    file_path:
        CSV file path.

    expected_interval_s:
        Expected nominal sampling interval.
        Use 1 for 1-second data.
        Use 30 for 30-second data.
        None means automatic detection.

    Returns
    -------
    pd.DataFrame
        Original columns plus:

        elapsed_s:
            Elapsed time from the first sample.

        dt_to_next_s:
            Actual time interval from the current
            sample to the next sample.

            The last row is NaN because there
            is no next sample.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}"
        )

    # -------------------------------------------------
    # 1. Read original data
    # -------------------------------------------------

    df = pd.read_csv(path)

    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if len(df) < 2:
        raise ValueError(
            "A voyage must contain at least two samples."
        )

    # -------------------------------------------------
    # 2. Parse timestamp
    # -------------------------------------------------

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    if df["timestamp"].isna().any():
        raise ValueError(
            "Invalid timestamp values detected."
        )

    if df["timestamp"].duplicated().any():
        raise ValueError(
            "Duplicate timestamps detected."
        )

    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError(
            "Timestamps must be strictly increasing."
        )

    # -------------------------------------------------
    # 3. Validate load power
    # -------------------------------------------------

    df["load_total_kw"] = pd.to_numeric(
        df["load_total_kw"],
        errors="coerce",
    )

    load_values = df["load_total_kw"].to_numpy(
        dtype=np.float64
    )

    if not np.isfinite(load_values).all():
        raise ValueError(
            "Invalid or missing load power values."
        )

    if (load_values < 0).any():
        raise ValueError(
            "Negative ship load detected."
        )

    # -------------------------------------------------
    # 4. Calculate actual time intervals
    # -------------------------------------------------

    dt = (
        df["timestamp"]
        .diff()
        .dt.total_seconds()
        .dropna()
        .to_numpy(dtype=np.float64)
    )

    if not np.isfinite(dt).all() or (dt <= 0).any():
        raise ValueError(
            "Invalid time intervals detected."
        )

    nominal_interval = float(np.median(dt))

    if expected_interval_s is not None:

        if not np.isclose(
            nominal_interval,
            expected_interval_s,
            atol=0.5,
        ):
            raise ValueError(
                f"Expected approximately "
                f"{expected_interval_s}s sampling, "
                f"but detected {nominal_interval}s."
            )

    # -------------------------------------------------
    # 5. Elapsed time
    # -------------------------------------------------

    elapsed = (
        df["timestamp"] - df["timestamp"].iloc[0]
    ).dt.total_seconds()

    df["elapsed_s"] = elapsed.astype(np.float64)

    # Preserve original time_s if available.
    # Otherwise create a unified relative time column.

    if "time_s" in df.columns:

        original_time = pd.to_numeric(
            df["time_s"],
            errors="coerce",
        )

        if original_time.isna().any():
            raise ValueError(
                "Invalid time_s values detected."
            )

        if not np.allclose(
            original_time.to_numpy(dtype=np.float64),
            elapsed.to_numpy(dtype=np.float64),
            atol=1e-6,
        ):
            raise ValueError(
                "time_s does not match timestamp."
            )

    else:
        df["time_s"] = df["elapsed_s"]

    # -------------------------------------------------
    # 6. Actual duration to the next sample
    # -------------------------------------------------

    dt_to_next = (
        df["timestamp"].shift(-1)
        - df["timestamp"]
    ).dt.total_seconds()

    df["dt_to_next_s"] = dt_to_next.astype(
        np.float64
    )

    # Last row intentionally remains NaN.
    # It has no next state.

    # -------------------------------------------------
    # 7. Store metadata
    # -------------------------------------------------

    df.attrs["source_file"] = str(path)

    df.attrs["nominal_interval_s"] = (
        nominal_interval
    )

    df.attrs["num_samples"] = len(df)

    df.attrs["duration_s"] = float(
        elapsed.iloc[-1]
    )

    return df
