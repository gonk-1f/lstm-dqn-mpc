"""Diagnostics for Validation-only LPF time-constant screening."""

from __future__ import annotations

import math

import numpy as np

from .formal_policy import EpisodePowerTrace, PolicyEvaluation


SCREENING_STATUS = "PRETRAIN_ENVIRONMENT_SCREEN_ONLY"


def _trace_row(
    trace: EpisodePowerTrace,
    evaluation: object,
) -> dict[str, object]:
    fc = np.asarray(trace.fuel_cell_power_kw, dtype=float)
    battery = np.asarray(trace.battery_bus_power_kw, dtype=float)
    if len(fc) == 0:
        fc_steps = np.asarray([], dtype=float)
    else:
        fc_steps = np.abs(np.diff(np.concatenate((np.asarray([0.0]), fc))))
    step_seconds = (
        trace.soc_time_s[1] - trace.soc_time_s[0]
        if len(trace.soc_time_s) > 1
        else 0.0
    )
    return {
        "sample_id": trace.sample_id,
        "completed": trace.completed,
        "failure_kind": trace.failure_kind or "",
        "interval_count": len(trace.time_s),
        "raw_economic_cost_cny": float(evaluation.raw_economic_cost_cny),
        "failure_penalty_score": float(evaluation.failure_penalty_score),
        "soc_min": float(evaluation.soc_min),
        "soc_max": float(evaluation.soc_max),
        "fc_total_variation_kw": float(math.fsum(fc_steps.tolist())),
        "fc_mean_abs_step_kw": float(np.mean(fc_steps)) if len(fc_steps) else 0.0,
        "fc_p95_abs_step_kw": float(np.percentile(fc_steps, 95)) if len(fc_steps) else 0.0,
        "fc_max_abs_step_kw": float(np.max(fc_steps)) if len(fc_steps) else 0.0,
        "battery_absolute_energy_kwh": float(
            math.fsum(np.abs(battery).tolist()) * step_seconds / 3600.0
        ),
        "battery_rms_kw": float(np.sqrt(np.mean(np.square(battery))))
        if len(battery)
        else 0.0,
    }


def build_tau_screen_summary(
    *,
    tau_lpf_seconds: float,
    evaluation: PolicyEvaluation,
    traces: tuple[EpisodePowerTrace, ...],
    checkpoint_trained_tau_seconds: float,
) -> tuple[dict[str, object], tuple[dict[str, object], ...]]:
    """Summarize one tau using only exact traces from the same evaluation."""

    tau = float(tau_lpf_seconds)
    trained_tau = float(checkpoint_trained_tau_seconds)
    if not math.isfinite(tau) or tau <= 0.0:
        raise ValueError("tau_lpf_seconds must be positive and finite")
    if not math.isfinite(trained_tau) or trained_tau <= 0.0:
        raise ValueError("checkpoint_trained_tau_seconds must be positive and finite")
    if type(evaluation) is not PolicyEvaluation:
        raise TypeError("evaluation must be an exact PolicyEvaluation")
    if type(traces) is not tuple or any(
        type(trace) is not EpisodePowerTrace for trace in traces
    ):
        raise TypeError("traces must contain exact EpisodePowerTrace values")
    if evaluation.episode_ids != tuple(trace.sample_id for trace in traces):
        raise ValueError("evaluation and trace episode order differs")

    rows = tuple(
        _trace_row(trace, episode)
        for trace, episode in zip(traces, evaluation.episodes)
    )
    summary = {
        "screening_status": SCREENING_STATUS,
        "tau_lpf_seconds": tau,
        "checkpoint_trained_tau_seconds": trained_tau,
        "policy_id": evaluation.policy_id,
        "episode_count": len(evaluation.episodes),
        "completed_episodes": evaluation.completed_episodes,
        "failed_episodes": evaluation.failed_episodes,
        "completion_rate": evaluation.completion_rate,
        "h2_cost_cny": evaluation.h2_cost_cny,
        "fc_degradation_cost_cny": evaluation.fc_degradation_cost_cny,
        "battery_degradation_cost_cny": evaluation.battery_degradation_cost_cny,
        "shore_cost_cny": evaluation.shore_cost_cny,
        "raw_economic_cost_cny": evaluation.raw_economic_cost_cny,
        "failure_penalty_score": evaluation.failure_penalty_score,
        "learning_reward": evaluation.learning_reward,
        "soc_min": min(row["soc_min"] for row in rows),
        "soc_max": max(row["soc_max"] for row in rows),
        "fc_total_variation_kw": math.fsum(
            float(row["fc_total_variation_kw"]) for row in rows
        ),
        "fc_mean_abs_step_kw": (
            math.fsum(
                float(row["fc_mean_abs_step_kw"]) * int(row["interval_count"])
                for row in rows
            )
            / sum(int(row["interval_count"]) for row in rows)
        ),
        "fc_max_abs_step_kw": max(float(row["fc_max_abs_step_kw"]) for row in rows),
        "battery_absolute_energy_kwh": math.fsum(
            float(row["battery_absolute_energy_kwh"]) for row in rows
        ),
        "battery_rms_kw": math.sqrt(
            math.fsum(
                float(row["battery_rms_kw"]) ** 2 * int(row["interval_count"])
                for row in rows
            )
            / sum(int(row["interval_count"]) for row in rows)
        ),
    }
    return summary, rows


__all__ = ["SCREENING_STATUS", "build_tau_screen_summary"]
