"""User-requested, frozen-model Test load-forecast plots; not formal v2 evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Callable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from v2.data.supervisory_rules import normalize_onboard_load_kw
from v2.main.train_formal_dqn import DEFAULT_AIS_ROOT, DEFAULT_MODE_ROOT, DEFAULT_POWER_ROOT

from .metrics import forecast_metrics
from .training import load_fitted_forecaster


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def aligned_forecasts(
    actual_kw: Sequence[float], modes: Sequence[str], *, history_steps: int,
    forecaster: Callable[[tuple[float, ...]], Sequence[float]], horizon: int = 5,
) -> np.ndarray:
    """At origin k, place forecast h at target index k+h; mask shore gaps."""
    actual = np.asarray(actual_kw, dtype=float)
    if actual.ndim != 1 or len(actual) != len(modes) or not np.isfinite(actual).all():
        raise ValueError("actual and modes must be aligned, finite one-dimensional data")
    if history_steps <= 0 or horizon <= 0:
        raise ValueError("history and horizon must be positive")
    onboard = np.asarray([isinstance(mode, str) and mode.lower() == "onboard" for mode in modes])
    cleaned = actual.copy()
    for index in np.flatnonzero(onboard):
        cleaned[index] = normalize_onboard_load_kw(float(actual[index]))
    predicted = np.full((horizon, len(actual)), np.nan)
    for origin in range(history_steps - 1, len(actual) - 1):
        if not onboard[origin - history_steps + 1 : origin + 1].all():
            continue
        history = tuple(float(value) for value in cleaned[origin - history_steps + 1 : origin + 1])
        future = np.asarray(forecaster(history), dtype=float)
        if future.shape != (horizon,) or not np.isfinite(future).all() or (future < 0).any():
            raise ValueError("forecaster must return finite nonnegative horizon powers")
        for h in range(1, horizon + 1):
            target = origin + h
            if target >= len(actual) or not onboard[origin + 1 : target + 1].all():
                break
            predicted[h - 1, target] = future[h - 1]
    return predicted


def _change_corr(actual_kw: np.ndarray, predicted_kw: np.ndarray, lag: int) -> float | None:
    actual = np.asarray(actual_kw, dtype=float)
    predicted = np.asarray(predicted_kw, dtype=float)
    if actual.shape != predicted.shape or actual.ndim != 1:
        raise ValueError("lag inputs must be equal one-dimensional arrays")
    a = np.diff(actual)
    p = np.diff(predicted)
    if lag > 0:
        a, p = a[:-lag], p[lag:]
    elif lag < 0:
        a, p = a[-lag:], p[:lag]
    mask = np.isfinite(a) & np.isfinite(p)
    if mask.sum() < 10 or np.std(a[mask]) == 0 or np.std(p[mask]) == 0:
        return None
    return float(np.corrcoef(a[mask], p[mask])[0, 1])


def change_lag_steps(
    actual_kw: np.ndarray, predicted_kw: np.ndarray, *, max_lag_steps: int = 10
) -> int | None:
    """Positive best lag means forecast changes appear later than actual changes."""
    candidates = [
        (lag, _change_corr(actual_kw, predicted_kw, lag))
        for lag in range(-max_lag_steps, max_lag_steps + 1)
    ]
    usable = [(lag, score) for lag, score in candidates if score is not None and np.isfinite(score)]
    return max(usable, key=lambda pair: pair[1])[0] if usable else None


def _test_rows(root: Path) -> tuple[pd.DataFrame, str]:
    manifest = root / "metadata" / "sample_manifest.csv"
    rows = pd.read_csv(manifest, encoding="utf-8-sig")
    selected = rows.loc[rows["split"].eq("test")].sort_values("sample_id")
    if len(selected) != 5 or selected["sample_id"].duplicated().any():
        raise ValueError("frozen Test manifest must contain five unique episodes")
    return selected, _sha256(manifest)


def _verified_path(root: Path, relative: str, expected_hash: str) -> Path:
    root = root.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("manifest path escapes dataset root") from exc
    if _sha256(path) != expected_hash:
        raise ValueError(f"Test payload hash mismatch: {path.name}")
    return path


def _episode(power_root: Path, mode_root: Path, power_row: object, mode_row: object):
    power_path = _verified_path(power_root, str(power_row.relative_path), str(power_row.sha256))
    mode_path = _verified_path(mode_root, str(mode_row.relative_path), str(mode_row.sha256))
    power = pd.read_csv(power_path, encoding="utf-8-sig")
    modes = pd.read_csv(mode_path, encoding="utf-8-sig")
    duration = float(power_row.duration_s)
    time = pd.to_numeric(power["time_s"], errors="raise")
    selected = power.loc[
        time.mod(30.0).eq(0.0) & time.add(30.0).le(duration),
        ["time_s", "load_total_kw"],
    ].reset_index(drop=True)
    if len(selected) != len(modes) or not np.array_equal(
        selected["time_s"].to_numpy(dtype=float), modes["time_s"].to_numpy(dtype=float)
    ):
        raise ValueError("Test power and mode axes differ")
    return (
        selected["time_s"].to_numpy(dtype=float),
        selected["load_total_kw"].to_numpy(dtype=float),
        tuple(str(value) for value in modes["mode"]),
    )


def _plot(path: Path, sample_id: str, time_s: np.ndarray, actual: np.ndarray, modes: Sequence[str], predicted: np.ndarray) -> None:
    x = time_s / 60.0
    one = predicted[0]
    five = predicted[4]
    valid_change = np.isfinite(one[1:]) & np.isfinite(one[:-1])
    changes = np.where(valid_change, np.abs(np.diff(actual)), -1.0)
    center = int(np.argmax(changes) + 1) if (changes >= 0).any() else len(actual) // 2
    start, stop = max(0, center - 20), min(len(actual), center + 21)
    onboard = np.asarray([mode.lower() == "onboard" for mode in modes])
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), constrained_layout=True)
    for ax in axes:
        ax.plot(x, actual, color="#17223b", lw=1.0, label="Actual load (kW)")
        ax.plot(x, one, color="#db3a34", lw=1.0, label="LSTM 1-step (30 s)")
        ax.plot(x, five, color="#ee9b00", lw=0.9, alpha=0.85, label="LSTM 5-step (150 s)")
        ax.fill_between(x, 0, 1, where=~onboard, transform=ax.get_xaxis_transform(), color="gray", alpha=0.13)
        ax.grid(alpha=0.2)
        ax.set_ylabel("Power (kW)")
    axes[0].set_title(f"{sample_id}: actual load and forecasts aligned to target time")
    axes[0].legend(ncol=3, loc="upper right", fontsize=9)
    axes[1].set_xlim(x[start], x[stop - 1])
    axes[1].set_title("Largest observed ONBOARD ramp: alignment detail")
    axes[1].set_xlabel("Episode time (min)")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def generate_test_diagnostic(
    *, checkpoint: Path, selection: Path, output_dir: Path,
    power_root: Path = DEFAULT_POWER_ROOT,
    ais_root: Path = DEFAULT_AIS_ROOT,
    mode_root: Path = DEFAULT_MODE_ROOT,
) -> dict[str, object]:
    """Read Test only after the v3 LSTM checkpoint and Validation selection exist."""
    checkpoint = Path(checkpoint)
    selection_data = json.loads(Path(selection).read_text(encoding="utf-8"))
    forecaster = load_fitted_forecaster(checkpoint)
    if selection_data["selected_history_steps"] != forecaster.history_steps:
        raise ValueError("frozen LSTM checkpoint differs from Validation selection")
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    power_rows, power_hash = _test_rows(Path(power_root))
    mode_rows, mode_hash = _test_rows(Path(mode_root))
    _, ais_hash = _test_rows(Path(ais_root))
    by_mode = mode_rows.set_index("sample_id")
    if set(power_rows["sample_id"]) != set(mode_rows["sample_id"]):
        raise ValueError("power and mode Test sample identities differ")
    output_dir.mkdir(parents=True)
    episodes = []
    for row in power_rows.itertuples(index=False):
        time_s, actual, modes = _episode(Path(power_root), Path(mode_root), row, by_mode.loc[row.sample_id])
        predicted = aligned_forecasts(
            actual, modes, history_steps=forecaster.history_steps,
            forecaster=forecaster, horizon=5,
        )
        actual_onboard = actual.copy()
        for index, mode in enumerate(modes):
            if mode.lower() == "onboard":
                actual_onboard[index] = normalize_onboard_load_kw(float(actual[index]))
        horizon_metrics = []
        for h in range(1, 6):
            valid = np.isfinite(predicted[h - 1])
            if not valid.any():
                horizon_metrics.append(None)
                continue
            metrics = forecast_metrics(
                actual_onboard[valid, None], predicted[h - 1, valid, None]
            )
            metrics.pop("horizon")
            horizon_metrics.append(metrics)
        lag = change_lag_steps(actual_onboard, predicted[0], max_lag_steps=10)
        file_name = f"{row.sample_id}_load_forecast.png"
        _plot(output_dir / file_name, row.sample_id, time_s, actual, modes, predicted)
        episodes.append({
            "sample_id": row.sample_id,
            "plot": file_name,
            "onboard_points": sum(mode.lower() == "onboard" for mode in modes),
            "horizon_metrics": horizon_metrics,
            "one_step_change_lag_steps": lag,
            "one_step_change_lag_seconds": None if lag is None else lag * 30,
            "zero_lag_change_correlation": _change_corr(actual_onboard, predicted[0], 0),
            "best_lag_change_correlation": None if lag is None else _change_corr(actual_onboard, predicted[0], lag),
        })
    report = {
        "status": "user_requested_v3_test_diagnostic_not_formal_v2_final_evaluation",
        "checkpoint_sha256": _sha256(checkpoint),
        "selection_sha256": _sha256(Path(selection)),
        "manifest_sha256": {"power": power_hash, "ais": ais_hash, "modes": mode_hash},
        "history_steps": forecaster.history_steps,
        "horizon_steps": 5,
        "sample_seconds": 30,
        "lag_sign": "positive means forecast changes occur later than actual changes",
        "episodes": episodes,
    }
    (output_dir / "diagnostic.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/v3_lstm/selected_lstm.pt"))
    parser.add_argument("--selection", type=Path, default=Path("outputs/v3_lstm/selection.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/v3_lstm_test_diagnostic"))
    args = parser.parse_args(argv)
    report = generate_test_diagnostic(
        checkpoint=args.checkpoint, selection=args.selection, output_dir=args.output_dir
    )
    print(json.dumps({
        "checkpoint_sha256": report["checkpoint_sha256"],
        "episode_count": len(report["episodes"]),
        "output_dir": str(args.output_dir),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
