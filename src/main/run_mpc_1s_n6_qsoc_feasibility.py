from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from benchmark_mpc_qp_osqp_1s import default_config
from mpc_solvers.mpc_qp_formulation import QpMpcConfig
from run_mpc_1s_n6_weight_selection import (
    DEFAULT_INPUT_PATH,
    EXPECTED_TEST_VOYAGES,
    N6_HORIZON,
    N6_OSQP_SETTINGS,
    N6_TOLERANCES,
    REPO_ROOT,
    REQUIRED_N6_METRIC_KEYS,
    _format_report_value,
    _json_ready,
    run_candidate,
)

DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "mpc_1s_n6_qsoc_feasibility"
DEFAULT_REPORTS_DIR = REPO_ROOT / "reports"
DEFAULT_SUMMARY_REPORT = DEFAULT_REPORTS_DIR / "mpc_1s_n6_qsoc_feasibility_summary.md"
DEFAULT_TABLE_REPORT = DEFAULT_REPORTS_DIR / "mpc_1s_n6_qsoc_feasibility_table.csv"
PREVIOUS_A_SUMMARY = (
    REPO_ROOT
    / "outputs"
    / "mpc_1s_n6_weight_selection"
    / "candidate_A"
    / "summary_metrics.json"
)
N60_POINTER = (
    REPO_ROOT
    / "outputs"
    / "mpc_1s_n6_weight_selection"
    / "N60_HISTORICAL_BENCHMARK.md"
)

QSOC_CANDIDATES: tuple[dict[str, Any], ...] = (
    {"candidate_id": "QSOC_5", "q_h2": 0.5, "q_soc": 5.0, "q_batt": 0.05, "soc_band": 0.05},
    {"candidate_id": "QSOC_10", "q_h2": 0.5, "q_soc": 10.0, "q_batt": 0.05, "soc_band": 0.05},
    {"candidate_id": "QSOC_20", "q_h2": 0.5, "q_soc": 20.0, "q_batt": 0.05, "soc_band": 0.05},
)

FINITE_GATE_METRICS = frozenset(
    set(REQUIRED_N6_METRIC_KEYS)
    | {
        "closed_loop_coverage_fraction",
        "solver_failure_count",
        "physical_infeasible_point_count",
        "worst_voyage_soc_net_change",
        "max_actual_power_balance_residual_kw",
        "max_soc_bound_residual",
        "max_soc_prediction_residual",
        "primal_residual_max_abs",
        "dual_residual_max_abs",
    }
)
PHYSICAL_RESIDUAL_LIMITS: dict[str, float] = {
    "max_actual_power_balance_residual_kw": N6_TOLERANCES["actual_balance_kw"],
    "max_plan_power_balance_residual_kw": N6_TOLERANCES["qp_balance_kw"],
    "max_fc_bound_residual_kw": N6_TOLERANCES["power_bound_kw"],
    "max_battery_bound_residual_kw": N6_TOLERANCES["power_bound_kw"],
    "max_ramp_residual_kw": N6_TOLERANCES["ramp_kw"],
    "max_soc_bound_residual": N6_TOLERANCES["soc"],
    "max_soc_prediction_residual": N6_TOLERANCES["soc_prediction"],
}


def _candidate_spec(candidate_id: str) -> dict[str, Any]:
    normalized_id = str(candidate_id).upper()
    selected = next(
        (item for item in QSOC_CANDIDATES if item["candidate_id"] == normalized_id),
        None,
    )
    if selected is None:
        raise ValueError(
            f"candidate_id must be one of {[item['candidate_id'] for item in QSOC_CANDIDATES]}"
        )
    return selected


def qsoc_candidate_config(candidate_id: str) -> QpMpcConfig:
    selected = _candidate_spec(candidate_id)
    return default_config(
        horizon=N6_HORIZON,
        battery_capacity_kwh=693.0,
        q_h2=float(selected["q_h2"]),
        q_soc=float(selected["q_soc"]),
        q_batt=float(selected["q_batt"]),
        q_ramp=0.0,
        q_terminal_soc=0.0,
        fuel_cell_ramp_rate_kw_per_s=48.0,
        battery_power_max_kw=346.5,
        battery_power_ref_kw=346.5,
        soc_band=float(selected["soc_band"]),
    )


def _finite_number(summary: dict[str, Any], key: str) -> float | None:
    try:
        value = float(summary.get(key, float("nan")))
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def evaluate_candidate_gate(summary: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    nonfinite_metrics = sorted(
        key for key in FINITE_GATE_METRICS if _finite_number(summary, key) is None
    )
    if nonfinite_metrics:
        reasons.append(f"required metrics are missing or non-finite: {nonfinite_metrics}")
    if not bool(summary.get("closed_loop_complete", False)):
        reasons.append("not all seven voyage closed loops completed")
    solver_failures = _finite_number(summary, "solver_failure_count")
    if solver_failures is None or solver_failures != 0.0:
        reasons.append("final solver failure count is not zero")
    physical_failures = _finite_number(summary, "physical_infeasible_point_count")
    if physical_failures is None or physical_failures != 0.0:
        reasons.append("physical infeasible point count is not zero")
    if not bool(summary.get("aggregate_metrics_comparable", False)):
        reasons.append("aggregate metrics are incomplete or prefix-only")
    coverage = _finite_number(summary, "closed_loop_coverage_fraction")
    if coverage is None or coverage < 1.0:
        reasons.append("closed-loop coverage is below 100 percent")
    solver_success = _finite_number(summary, "solver_success_rate")
    if solver_success is None or solver_success < 1.0:
        reasons.append("solver success rate is below 100 percent")

    for key, limit in PHYSICAL_RESIDUAL_LIMITS.items():
        value = _finite_number(summary, key)
        if value is None or value > limit:
            reasons.append(f"{key} exceeds its recorded tolerance {limit}")

    soc_min = _finite_number(summary, "soc_min")
    if soc_min is None or soc_min < 0.2 - N6_TOLERANCES["soc"]:
        reasons.append("actual SOC minimum is below the physical lower bound")
    soc_max = _finite_number(summary, "soc_max")
    if soc_max is None or soc_max > 0.8 + N6_TOLERANCES["soc"]:
        reasons.append("actual SOC maximum is above the physical upper bound")
    worst_soc_change = _finite_number(summary, "worst_voyage_soc_net_change")
    if worst_soc_change is None or worst_soc_change < -0.03:
        reasons.append("worst-voyage SOC net change is below -0.03")
    solve_time_max = _finite_number(summary, "solve_time_ms_max")
    if solve_time_max is None or solve_time_max >= 1000.0:
        reasons.append("maximum solve time does not satisfy the 1 s control interval")

    return {
        "candidate_id": str(summary.get("candidate_id", "")).upper(),
        "passed": not reasons,
        "reasons": reasons,
    }


def build_diagnostic_decision(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    expected_ids = tuple(item["candidate_id"] for item in QSOC_CANDIDATES)
    normalized = [
        {**summary, "candidate_id": str(summary.get("candidate_id", "")).upper()}
        for summary in summaries
    ]
    by_candidate = {str(summary["candidate_id"]): summary for summary in normalized}
    if len(normalized) != len(expected_ids) or set(by_candidate) != set(expected_ids):
        raise ValueError(f"formal diagnosis requires exactly {list(expected_ids)}")
    if len(by_candidate) != len(normalized):
        raise ValueError("formal diagnosis cannot contain duplicate candidate summaries")

    evaluations: dict[str, dict[str, Any]] = {}
    for candidate_id in expected_ids:
        summary = by_candidate[candidate_id]
        if int(summary.get("voyage_count", -1)) != len(EXPECTED_TEST_VOYAGES):
            raise ValueError(f"candidate {candidate_id} must contain all seven formal voyages")
        if bool(summary.get("is_partial_debug_run", True)):
            raise ValueError(f"candidate {candidate_id} cannot be a partial debug run")
        evaluations[candidate_id] = evaluate_candidate_gate(summary)

    witnesses = [
        candidate_id for candidate_id in expected_ids if evaluations[candidate_id]["passed"]
    ]
    sufficient = bool(witnesses)
    status = (
        "weight_only_sufficient"
        if sufficient
        else "weight_only_insufficient_in_tested_range"
    )
    candidate_decisions = {
        candidate_id: {
            "passed": bool(evaluations[candidate_id]["passed"]),
            "reasons": list(evaluations[candidate_id]["reasons"]),
        }
        for candidate_id in expected_ids
    }
    if sufficient:
        conclusion = (
            "At least one preregistered q_soc value is a feasibility witness under the exact "
            "terminal-free N=6 ideal-foresight setup. This does not constitute an accepted paper weight."
        )
    else:
        conclusion = (
            "Increasing q_soc to 5, 10, or 20 alone was insufficient under the exact terminal-free "
            "N=6 ideal-foresight setup. This result does not prove that every value above 20 is impossible."
        )
    return {
        "experiment": "1 s offline ideal-foresight N=6 q_soc-only feasibility diagnosis",
        "status": status,
        "feasibility_witnesses": witnesses,
        "selected_candidate": None,
        "provisional_config_created": False,
        "accepted_config_created": False,
        "dqn_started": False,
        "selection_method": "fixed engineering gates without scoring or least-bad ranking",
        "gate_priority": [
            "physical feasibility",
            "long-term SOC",
            "power allocation",
            "economy and device use",
            "solver performance",
        ],
        "candidate_decisions": candidate_decisions,
        "conclusion": conclusion,
        "conclusion_scope": "only q_soc in {5, 10, 20} under the recorded N=6 setup",
    }


def run_qsoc_candidate(
    candidate_id: str,
    *,
    input_path: str | Path = DEFAULT_INPUT_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    make_plots: bool = True,
    max_steps_per_voyage: int | None = None,
    expected_voyage_count: int | None = len(EXPECTED_TEST_VOYAGES),
) -> dict[str, Any]:
    normalized_id = str(candidate_id).upper()
    return run_candidate(
        normalized_id,
        input_path=input_path,
        output_root=output_root,
        make_plots=make_plots,
        max_steps_per_voyage=max_steps_per_voyage,
        expected_voyage_count=expected_voyage_count,
        config=qsoc_candidate_config(normalized_id),
    )


def validate_candidate_fingerprint(
    candidate_id: str,
    metadata: dict[str, Any],
    *,
    expected_input_path: str | Path = DEFAULT_INPUT_PATH,
) -> None:
    normalized_id = str(candidate_id).upper()
    expected_timing = {
        "decision_interval_seconds": 1.0,
        "forecast_samples": "t+1..t+6",
        "prediction_horizon_steps": N6_HORIZON,
        "control_horizon_steps": N6_HORIZON,
        "applied_action": "first step only",
        "actual_battery_definition": "P_load(t+1) - P_fc(t+1)",
        "actual_soc_update": "SOC - P_batt/(3600*693)",
        "voyage_end_policy": "repeat the final sample within the same voyage; never cross voyages",
    }
    expected_initial_state = {
        "soc": 0.55,
        "fuel_cell_kw": "clip(first voyage load, 0, 560)",
        "reset_for_each_voyage": True,
    }
    expected_osqp = {
        "persistent_solver_per_voyage": True,
        **N6_OSQP_SETTINGS,
        "affine_variable_and_constraint_scaling": True,
        "max_iter_recovery": "one cold restart of the same QP; no control fallback",
    }
    expected_fields: dict[str, Any] = {
        "candidate_id": normalized_id,
        "status": "raw_candidate",
        "solver": "OSQP",
        "problem_class": "convex_qp",
        "forecast_source": "offline ideal foresight",
        "input_data": _portable_repo_path(Path(expected_input_path)),
        "input_data_note": "natural-clipped cubic-spline 1 s reconstruction; not measured 1 s data",
        "lstm_used": False,
        "timing": expected_timing,
        "initial_state": expected_initial_state,
        "soc_reference": 0.55,
        "model": _json_ready(asdict(qsoc_candidate_config(normalized_id))),
        "tolerances": dict(N6_TOLERANCES),
        "osqp_settings": expected_osqp,
    }
    mismatches = [
        key for key, expected in expected_fields.items() if metadata.get(key) != expected
    ]
    if mismatches:
        raise ValueError(
            f"candidate {normalized_id} config fingerprint mismatch: {sorted(mismatches)}"
        )


def load_formal_summaries(
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    *,
    expected_input_path: str | Path = DEFAULT_INPUT_PATH,
) -> list[dict[str, Any]]:
    root = Path(output_root)
    summaries: list[dict[str, Any]] = []
    for candidate in QSOC_CANDIDATES:
        candidate_id = str(candidate["candidate_id"])
        candidate_dir = root / f"candidate_{candidate_id}"
        summary_path = candidate_dir / "summary_metrics.json"
        config_path = candidate_dir / "config.json"
        if not summary_path.exists():
            raise ValueError(f"missing formal summary: {summary_path}")
        if not config_path.exists():
            raise ValueError(f"missing formal config: {config_path}")
        metadata = json.loads(config_path.read_text(encoding="utf-8"))
        validate_candidate_fingerprint(
            candidate_id,
            metadata,
            expected_input_path=expected_input_path,
        )
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        payload_id = str(payload.get("candidate_id", "")).upper()
        if payload_id != candidate_id:
            raise ValueError(
                f"candidate {candidate_id} summary identity mismatch: {payload_id or 'missing'}"
            )
        payload["candidate_id"] = payload_id
        summaries.append(payload)
    return summaries


def _enriched_summaries(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    specs = {str(item["candidate_id"]): item for item in QSOC_CANDIDATES}
    enriched: list[dict[str, Any]] = []
    for summary in summaries:
        candidate_id = str(summary.get("candidate_id", "")).upper()
        spec = specs[candidate_id]
        missing = sorted(REQUIRED_N6_METRIC_KEYS.difference(summary))
        if missing:
            raise ValueError(f"candidate {candidate_id} is missing required metrics: {missing}")
        enriched.append(
            {
                "candidate_id": candidate_id,
                "q_h2": float(spec["q_h2"]),
                "q_soc": float(spec["q_soc"]),
                "q_batt": float(spec["q_batt"]),
                "soc_band": float(spec["soc_band"]),
                **{key: value for key, value in summary.items() if key != "candidate_id"},
            }
        )
    return enriched


def _portable_repo_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _previous_anchor_text() -> str:
    if not PREVIOUS_A_SUMMARY.exists():
        return f"The retained q_soc=2 anchor summary is referenced at `{_portable_repo_path(PREVIOUS_A_SUMMARY)}`."
    payload = json.loads(PREVIOUS_A_SUMMARY.read_text(encoding="utf-8"))
    coverage = _format_report_value(payload.get("closed_loop_coverage_fraction"))
    worst_soc = _format_report_value(payload.get("worst_voyage_soc_net_change"))
    return (
        "The retained q_soc=2 A anchor had closed-loop coverage "
        f"{coverage} and worst-voyage SOC net change {worst_soc}; its source remains "
        f"`{_portable_repo_path(PREVIOUS_A_SUMMARY)}`."
    )


def invalidate_diagnostic_artifacts(
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    reports_dir: str | Path = DEFAULT_REPORTS_DIR,
) -> None:
    paths = (
        Path(output_root) / "diagnostic_decision.json",
        Path(reports_dir) / DEFAULT_SUMMARY_REPORT.name,
        Path(reports_dir) / DEFAULT_TABLE_REPORT.name,
    )
    for path in paths:
        if path.exists():
            if not path.is_file():
                raise ValueError(f"refusing to invalidate non-file diagnostic path: {path}")
            path.unlink()


def write_diagnostic_artifacts(
    summaries: list[dict[str, Any]],
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    reports_dir: str | Path = DEFAULT_REPORTS_DIR,
) -> tuple[Path, Path, Path]:
    decision = build_diagnostic_decision(summaries)
    enriched = _enriched_summaries(summaries)
    output_path = Path(output_root)
    report_path = Path(reports_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report_path.mkdir(parents=True, exist_ok=True)

    decision_path = output_path / "diagnostic_decision.json"
    table_path = report_path / DEFAULT_TABLE_REPORT.name
    markdown_path = report_path / DEFAULT_SUMMARY_REPORT.name
    decision_path.write_text(
        json.dumps(_json_ready(decision), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(enriched).to_csv(table_path, index=False)

    key_columns = (
        "candidate_id",
        "q_soc",
        "closed_loop_complete",
        "closed_loop_coverage_fraction",
        "solver_failure_count",
        "physical_infeasible_point_count",
        "final_soc",
        "worst_voyage_soc_net_change",
        "soc_min",
        "soc_max",
        "hydrogen_total_kg",
        "battery_throughput_kwh",
        "fc_above_load_fraction",
        "fc_surplus_energy_kwh",
        "solve_time_ms_p99",
        "solve_time_ms_max",
    )
    lines = [
        "# N=6 q_soc-only feasibility diagnosis",
        "",
        "## Experiment boundary",
        "",
        "This is a bounded structural diagnosis on the natural-clipped cubic-spline 1 s reconstruction. "
        "It is offline ideal foresight, not measured online 1 s data and not an LSTM forecast.",
        "At decision time `t`, the QP uses `t+1..t+6`, applies only the first action, computes actual "
        "battery power as actual load minus applied FC power, and updates actual SOC from that battery power.",
        "No DQN is trained or invoked. No terminal SOC term, slack, load shedding, soft ramp, model change, "
        "or N=60 rerun is introduced.",
        "",
        "## Preregistered candidates",
        "",
        "Only `q_soc` changes. Every candidate fixes `q_h2=0.5`, `q_batt=0.05`, `SOC_band=0.05`, "
        "`q_ramp=0`, and `q_terminal_soc=0`.",
        "",
        "| candidate_id | q_soc |",
        "|---|---:|",
    ]
    lines.extend(
        f"| {item['candidate_id']} | {_format_report_value(item['q_soc'])} |"
        for item in QSOC_CANDIDATES
    )
    lines.extend(
        [
            "",
            "## Candidate results",
            "",
            "| " + " | ".join(key_columns) + " |",
            "|" + "|".join("---:" for _ in key_columns) + "|",
        ]
    )
    for summary in enriched:
        lines.append(
            "| " + " | ".join(_format_report_value(summary.get(key)) for key in key_columns) + " |"
        )
    lines.extend(
        [
            "",
            "The companion CSV and per-candidate files retain all requested aggregate, per-voyage, and "
            "solver metrics. If a voyage terminates early, its hydrogen and energy values cover only the "
            "successfully applied prefix and are not valid full-voyage economic comparisons.",
            "",
            "## Fixed feasibility gate",
            "",
            "A feasibility witness must complete all seven voyages, have zero final solver failures and zero "
            "physical infeasible points, keep actual SOC within `[0.2,0.8]`, have worst-voyage SOC net change "
            "at least `-0.03`, retain complete aggregate metrics, and solve every QP in under 1 s.",
            "No aggregate score or least-bad ranking is used.",
            "",
            "## Diagnostic decision",
            "",
            f"- Status: `{decision['status']}`",
            "- Selected candidate: none (this diagnosis identifies feasibility witnesses; it does not accept a paper weight)",
            "- Feasibility witnesses: "
            + (
                ", ".join(f"`{item}`" for item in decision["feasibility_witnesses"])
                if decision["feasibility_witnesses"]
                else "none"
            ),
            "- Provisional config created: false",
            "- Accepted config created: false",
            "",
            str(decision["conclusion"]),
            "",
            "Candidate gate decisions:",
        ]
    )
    for candidate in QSOC_CANDIDATES:
        candidate_id = str(candidate["candidate_id"])
        evaluation = decision["candidate_decisions"][candidate_id]
        if evaluation["passed"]:
            text = "passed every fixed feasibility gate"
        else:
            text = "; ".join(str(reason) for reason in evaluation["reasons"])
        lines.append(f"- **{candidate_id}**: {text}.")
    lines.extend(
        [
            "",
            "## Historical boundaries",
            "",
            _previous_anchor_text(),
            f"N=60 remains a historical solver/performance benchmark only; its retained pointer is "
            f"`{_portable_repo_path(N60_POINTER)}` and its result tree was not modified or rerun.",
            "",
            "The structural conclusion is limited to `q_soc in {5,10,20}` under this exact terminal-free "
            "N=6 setup. Failure of all three does not prove that every larger q_soc is impossible; passing "
            "does not automatically authorize DQN training or promote a final paper configuration.",
            "",
        ]
    )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return decision_path, markdown_path, table_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose whether q_soc alone can make terminal-free N=6 MPC feasible."
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--candidate", choices=[item["candidate_id"] for item in QSOC_CANDIDATES])
    modes.add_argument("--all", action="store_true", dest="run_all")
    modes.add_argument("--report-only", action="store_true")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--max-steps-per-voyage", type=int)
    parser.add_argument("--expected-voyages", type=int, default=len(EXPECTED_TEST_VOYAGES))
    args = parser.parse_args(argv)
    invalidate_diagnostic_artifacts(args.output_root, args.reports_dir)

    if args.report_only:
        summaries = load_formal_summaries(
            args.output_root,
            expected_input_path=args.input,
        )
        decision_path, markdown_path, table_path = write_diagnostic_artifacts(
            summaries,
            output_root=args.output_root,
            reports_dir=args.reports_dir,
        )
        print(f"decision={decision_path}")
        print(f"report={markdown_path}")
        print(f"table={table_path}")
        return 0

    candidate_ids = (
        [item["candidate_id"] for item in QSOC_CANDIDATES]
        if args.run_all
        else [args.candidate]
    )
    for candidate_id in candidate_ids:
        result = run_qsoc_candidate(
            str(candidate_id),
            input_path=args.input,
            output_root=args.output_root,
            make_plots=not args.no_plots,
            max_steps_per_voyage=args.max_steps_per_voyage,
            expected_voyage_count=args.expected_voyages,
        )
        print(json.dumps(_json_ready(result["summary"]), ensure_ascii=False))

    if args.run_all and args.max_steps_per_voyage is None:
        summaries = load_formal_summaries(
            args.output_root,
            expected_input_path=args.input,
        )
        decision_path, markdown_path, table_path = write_diagnostic_artifacts(
            summaries,
            output_root=args.output_root,
            reports_dir=args.reports_dir,
        )
        print(f"decision={decision_path}")
        print(f"report={markdown_path}")
        print(f"table={table_path}")
    else:
        print("diagnosis=pending_remaining_formal_candidates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
