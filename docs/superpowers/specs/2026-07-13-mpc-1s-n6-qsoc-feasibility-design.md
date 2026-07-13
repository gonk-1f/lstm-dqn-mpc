# N=6 q_soc-Only Feasibility Diagnosis Design

**Date:** 2026-07-13

**Approval:** The user explicitly authorized the bounded `q_soc in {5, 10, 20}` diagnosis on 2026-07-13. The authorization fixes every other controller and experiment setting and forbids DQN work or deletion of the N=60 results.

## Purpose

The experiment answers one structural question: can the existing terminal-free, six-step MPC complete all seven test voyages and sustain SOC when only the stage SOC weight is increased by one order of magnitude? It is not a general hyperparameter search and it does not establish that values outside the preregistered set will or will not work.

The previous A-D experiment remains an immutable historical snapshot. Its `q_soc=2` anchor did not complete all voyages. This diagnosis uses a new runner, output root, report pair, and decision file so the earlier results cannot be overwritten or silently reinterpreted.

## Frozen experiment boundary

All three candidates use:

| setting | fixed value |
|---|---:|
| horizon | `N=6` |
| sample interval | `1 s` |
| `q_h2` | `0.5` |
| `q_batt` | `0.05` |
| `SOC_band` | `0.05` |
| `q_ramp` | `0` |
| `q_terminal_soc` | `0` |
| battery capacity | `693 kWh` |
| battery power bound | `+/-346.5 kW` |
| FC power bound | `0..560 kW` |
| SOC bounds/reference | `0.2..0.8`, reference `0.55` |
| FC ramp | `48 kW/s` |

The only candidate-dependent value is:

| candidate ID | `q_soc` |
|---|---:|
| `QSOC_5` | `5` |
| `QSOC_10` | `10` |
| `QSOC_20` | `20` |

The input remains the seven test voyages in `outputs/mpc_solver_benchmark_1s/data/test_voyages_spline_1s.parquet`. It is a natural-clipped cubic-spline 1 s reconstruction of 30 s vessel data, uses future endpoints, and is therefore offline ideal foresight rather than measured online 1 s data or an LSTM forecast.

At decision index `t`, the QP receives `load[t+1:t+7]`, applies only its first FC action at `t+1`, calculates actual battery power as `P_load(t+1)-P_fc(t+1)`, and updates actual SOC from that battery power. The final horizon is padded only with the last sample of the same voyage. Every voyage resets to `SOC=0.55` and the same initial FC rule. The QP, exact affine OSQP scaling, tolerances, cold-restart rule, failure termination rule, metric definitions, and plotting logic remain unchanged.

## Implementation boundary

The existing A-D runner keeps its candidate table, default paths, report names, and behavior. Its reusable execution layer gains an optional explicit `QpMpcConfig` parameter. Calls that omit it continue to resolve A-D exactly as before.

A new thin runner, `src/main/run_mpc_1s_n6_qsoc_feasibility.py`, owns the three diagnostic candidates and calls the shared N=6 execution and metric functions. It writes only to:

- `outputs/mpc_1s_n6_qsoc_feasibility/candidate_QSOC_<value>/`
- `outputs/mpc_1s_n6_qsoc_feasibility/diagnostic_decision.json`
- `reports/mpc_1s_n6_qsoc_feasibility_summary.md`
- `reports/mpc_1s_n6_qsoc_feasibility_table.csv`

No DQN, LSTM, 10 ms, historical N=60, or old A-D artifact is modified by the runner.

## Decision rule

Each candidate is checked in the existing engineering priority order: physical feasibility, long-term SOC, power allocation, economy/device use, then solver performance. A candidate is a feasibility witness only when all seven voyages are complete, final solver failures and physical-infeasible points are zero, actual SOC stays within `[0.2, 0.8]` within tolerance, the worst-voyage SOC net change is at least `-0.03`, aggregate metrics are complete, and maximum solve time remains below the 1 s control interval.

The combined diagnostic decision has one of two structural outcomes:

- `weight_only_sufficient`: at least one preregistered candidate is a feasibility witness.
- `weight_only_insufficient_in_tested_range`: none of the three candidates passes every gate.

Passing does not automatically make a weight accepted for the paper or trigger DQN training. No accepted configuration is written automatically. A provisional configuration may be considered only after the full engineering review and repository verification, and must remain explicitly provisional.

If no candidate passes, the report may conclude only that increasing `q_soc` to 5, 10, or 20 was insufficient under this exact terminal-free N=6 experiment. It must not claim that all larger `q_soc` values are mathematically impossible.

## Evidence and failure handling

Every candidate retains the existing compact artifacts: `config.json`, `summary_metrics.json`, `voyage_metrics.csv`, `solver_statistics.csv`, `constraint_audit.md`, and seven power/SOC plots when data exist. No full per-step CSV or solve-time log is retained.

The combined table preserves the original 24 physical and solver metrics plus coverage and failure state. Energy and hydrogen totals from an incomplete run describe only the applied prefix and are not valid full-voyage economic comparisons. Solver success alone never counts as physical acceptance.

The runner stops a voyage at its first final OSQP failure, records the status, and does not invent fallback controls or freeze SOC. `maximum iterations` receives the existing single cold restart of the same QP; no slack, terminal term, load shedding, soft ramp, or model change is introduced.

## Verification

Tests must prove the exact three-candidate table, that only `q_soc` changes, terminal weight remains zero, explicit-config injection leaves the old A-D contract unchanged, output/report paths are independent, all three formal summaries are required for a final diagnosis, and the feasibility gate is applied without least-bad selection.

Before push, run the focused tests, `python -m compileall src`, the complete unit-test suite, `git diff --check`, artifact-size/name checks, and an independent code/report review. Push is forbidden if tests fail, the experiment is incomplete, Git conflicts exist, or remote synchronization fails.
