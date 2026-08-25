from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mpc.solvers.fc_dp0_curve import h2_kg_step_dp0_quadratic


VOYAGE_IDS = tuple(f"voyage_{index:03d}" for index in range(60, 67))
SOC_REFERENCE = 0.55
DT_SECONDS = 1.0
FUEL_CELL_RATED_KW = 600.0

DQN_OUTPUT_DIR = (
    REPO_ROOT
    / "outputs"
    / "dqn_mpc_mlp_causal_1epoch_20260820"
    / "formal_test"
)
FIXED_A0_OUTPUT_DIR = REPO_ROOT / "outputs" / "mpc_nominal_causal_test"
COMPARISON_OUTPUT_DIR = REPO_ROOT / "outputs" / "mpc_vs_dqn_comparison"
PLOT_OUTPUT_DIR = COMPARISON_OUTPUT_DIR / "plots"

TRACE_COLUMNS = (
    "execution_index",
    "load_kw",
    "p_fc_kw",
    "p_batt_kw",
    "soc_before",
    "soc_after",
)
SUMMARY_COLUMNS = (
    "voyage_id",
    "completed",
    "solver_failure_count",
    "episode_steps",
)
METRIC_COLUMNS = (
    "voyage_id",
    "controller",
    "completed",
    "solver_failures",
    "steps",
    "total_h2_kg",
    "battery_throughput_kwh",
    "fc_total_variation_kw",
    "soc_mae",
    "min_soc",
    "final_soc",
)
SUMMARY_OUTPUT_COLUMNS = (
    "controller",
    "completed_voyages",
    "solver_failures",
    "total_h2_kg",
    "total_battery_throughput_kwh",
    "total_fc_variation_kw",
    "mean_soc_mae",
    "mean_min_soc",
    "worst_min_soc",
    "mean_abs_final_soc_error",
)


def read_summary(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = set(SUMMARY_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing summary columns in {path}: {sorted(missing)}")

    frame = frame.loc[:, SUMMARY_COLUMNS].copy()
    frame["voyage_id"] = frame["voyage_id"].astype(str)
    if frame["voyage_id"].duplicated().any():
        raise ValueError(f"Duplicate voyage_id values in {path}")
    return frame.set_index("voyage_id")


def read_trace(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = set(TRACE_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing trace columns in {path}: {sorted(missing)}")

    frame = frame.loc[:, TRACE_COLUMNS].copy()
    for column in TRACE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="raise")

    if frame.empty:
        raise ValueError(f"Trace is empty: {path}")
    if not np.isfinite(frame.to_numpy(dtype=np.float64)).all():
        raise ValueError(f"Trace contains non-finite values: {path}")
    return frame


def as_bool(value: object, *, field_name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    raise ValueError(f"Invalid boolean value for {field_name}: {value!r}")


def validate_trace_pair(
    voyage_id: str,
    fixed_a0_trace: pd.DataFrame,
    dqn_trace: pd.DataFrame,
) -> None:
    if len(fixed_a0_trace) != len(dqn_trace):
        raise ValueError(
            f"Trace length mismatch for {voyage_id}: "
            f"Fixed_A0={len(fixed_a0_trace)}, DQN_MPC={len(dqn_trace)}"
        )

    if not np.array_equal(
        fixed_a0_trace["execution_index"].to_numpy(),
        dqn_trace["execution_index"].to_numpy(),
    ):
        raise ValueError(f"execution_index mismatch for {voyage_id}")

    if not np.array_equal(
        fixed_a0_trace["load_kw"].to_numpy(),
        dqn_trace["load_kw"].to_numpy(),
    ):
        raise ValueError(f"load_kw mismatch for {voyage_id}")


def calculate_metrics(
    *,
    voyage_id: str,
    controller: str,
    trace: pd.DataFrame,
    summary_row: pd.Series,
) -> dict[str, object]:
    p_fc_kw = trace["p_fc_kw"].to_numpy(dtype=np.float64)
    p_batt_kw = trace["p_batt_kw"].to_numpy(dtype=np.float64)
    soc_after = trace["soc_after"].to_numpy(dtype=np.float64)

    h2_steps_kg = h2_kg_step_dp0_quadratic(
        p_fc_kw,
        dt_seconds=DT_SECONDS,
        p_rated_total_kw=FUEL_CELL_RATED_KW,
    )

    return {
        "voyage_id": voyage_id,
        "controller": controller,
        "completed": as_bool(
            summary_row["completed"],
            field_name="completed",
        ),
        "solver_failures": int(summary_row["solver_failure_count"]),
        "steps": int(summary_row["episode_steps"]),
        "total_h2_kg": float(np.sum(h2_steps_kg)),
        "battery_throughput_kwh": float(
            np.sum(np.abs(p_batt_kw)) / 3600.0
        ),
        "fc_total_variation_kw": float(np.sum(np.abs(np.diff(p_fc_kw)))),
        "soc_mae": float(np.mean(np.abs(soc_after - SOC_REFERENCE))),
        "min_soc": float(np.min(soc_after)),
        "final_soc": float(soc_after[-1]),
    }


def plot_soc_comparison(
    voyage_id: str,
    fixed_a0_trace: pd.DataFrame,
    dqn_trace: pd.DataFrame,
) -> None:
    fixed_soc = fixed_a0_trace["soc_after"].to_numpy(dtype=np.float64)
    dqn_soc = dqn_trace["soc_after"].to_numpy(dtype=np.float64)
    all_soc = np.concatenate((fixed_soc, dqn_soc, [SOC_REFERENCE]))
    padding = max(0.01, 0.15 * (float(all_soc.max()) - float(all_soc.min())))
    y_min = max(0.18, float(all_soc.min()) - padding)
    y_max = min(0.82, float(all_soc.max()) + padding)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        fixed_a0_trace["execution_index"],
        fixed_soc,
        label="Fixed A0",
    )
    ax.plot(
        dqn_trace["execution_index"],
        dqn_soc,
        label="DQN MPC",
    )
    ax.axhline(
        SOC_REFERENCE,
        linewidth=1.0,
        label="SOC reference = 0.55",
    )
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("SOC")
    ax.set_title(f"{voyage_id} - SOC Comparison")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        PLOT_OUTPUT_DIR / f"{voyage_id}_soc_comparison.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_power_comparison(
    voyage_id: str,
    fixed_a0_trace: pd.DataFrame,
    dqn_trace: pd.DataFrame,
) -> None:
    fixed_time_s = fixed_a0_trace["execution_index"]
    dqn_time_s = dqn_trace["execution_index"]

    fig, (ax_top, ax_bottom) = plt.subplots(
        nrows=2,
        sharex=True,
        figsize=(12, 8),
    )
    ax_top.plot(fixed_time_s, fixed_a0_trace["load_kw"], label="Load")
    ax_top.plot(fixed_time_s, fixed_a0_trace["p_fc_kw"], label="Fixed A0 FC")
    ax_top.plot(dqn_time_s, dqn_trace["p_fc_kw"], label="DQN FC")
    ax_top.set_ylabel("Power (kW)")
    ax_top.set_title(f"{voyage_id} - Power Comparison")
    ax_top.grid(True, alpha=0.3)
    ax_top.legend()

    ax_bottom.plot(
        fixed_time_s,
        fixed_a0_trace["p_batt_kw"],
        label="Fixed A0 Battery",
    )
    ax_bottom.plot(
        dqn_time_s,
        dqn_trace["p_batt_kw"],
        label="DQN Battery",
    )
    ax_bottom.axhline(0.0, linewidth=1.0)
    ax_bottom.set_xlabel("Time (s)")
    ax_bottom.set_ylabel("Battery Power (kW; + discharge)")
    ax_bottom.grid(True, alpha=0.3)
    ax_bottom.legend()
    fig.tight_layout()
    fig.savefig(
        PLOT_OUTPUT_DIR / f"{voyage_id}_power_comparison.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def create_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for controller in ("Fixed_A0", "DQN_MPC"):
        controller_metrics = metrics.loc[
            metrics["controller"] == controller
        ]
        rows.append(
            {
                "controller": controller,
                "completed_voyages": int(controller_metrics["completed"].sum()),
                "solver_failures": int(controller_metrics["solver_failures"].sum()),
                "total_h2_kg": float(controller_metrics["total_h2_kg"].sum()),
                "total_battery_throughput_kwh": float(
                    controller_metrics["battery_throughput_kwh"].sum()
                ),
                "total_fc_variation_kw": float(
                    controller_metrics["fc_total_variation_kw"].sum()
                ),
                "mean_soc_mae": float(controller_metrics["soc_mae"].mean()),
                "mean_min_soc": float(controller_metrics["min_soc"].mean()),
                "worst_min_soc": float(controller_metrics["min_soc"].min()),
                "mean_abs_final_soc_error": float(
                    np.mean(
                        np.abs(
                            controller_metrics["final_soc"].to_numpy()
                            - SOC_REFERENCE
                        )
                    )
                ),
            }
        )
    return pd.DataFrame(rows, columns=SUMMARY_OUTPUT_COLUMNS)


def main() -> None:
    fixed_a0_summary = read_summary(FIXED_A0_OUTPUT_DIR / "test_by_voyage.csv")
    dqn_summary = read_summary(DQN_OUTPUT_DIR / "test_by_voyage.csv")

    COMPARISON_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict[str, object]] = []
    for voyage_id in VOYAGE_IDS:
        if voyage_id not in fixed_a0_summary.index:
            raise ValueError(f"Missing Fixed_A0 summary row for {voyage_id}")
        if voyage_id not in dqn_summary.index:
            raise ValueError(f"Missing DQN_MPC summary row for {voyage_id}")

        fixed_a0_trace = read_trace(
            FIXED_A0_OUTPUT_DIR / "traces" / f"{voyage_id}_trace.csv"
        )
        dqn_trace = read_trace(
            DQN_OUTPUT_DIR / "traces" / f"{voyage_id}_trace.csv"
        )
        validate_trace_pair(voyage_id, fixed_a0_trace, dqn_trace)

        metric_rows.append(
            calculate_metrics(
                voyage_id=voyage_id,
                controller="Fixed_A0",
                trace=fixed_a0_trace,
                summary_row=fixed_a0_summary.loc[voyage_id],
            )
        )
        metric_rows.append(
            calculate_metrics(
                voyage_id=voyage_id,
                controller="DQN_MPC",
                trace=dqn_trace,
                summary_row=dqn_summary.loc[voyage_id],
            )
        )
        plot_soc_comparison(voyage_id, fixed_a0_trace, dqn_trace)
        plot_power_comparison(voyage_id, fixed_a0_trace, dqn_trace)

    metrics = pd.DataFrame(metric_rows, columns=METRIC_COLUMNS)
    metrics.to_csv(
        COMPARISON_OUTPUT_DIR / "metrics_by_voyage.csv",
        index=False,
    )
    create_summary(metrics).to_csv(
        COMPARISON_OUTPUT_DIR / "summary.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
