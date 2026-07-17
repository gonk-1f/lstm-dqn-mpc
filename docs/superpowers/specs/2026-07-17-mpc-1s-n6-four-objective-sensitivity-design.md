# N=6 Four-Objective Normalized MPC Sensitivity Design

**Date:** 2026-07-17

**Status:** approved design; implementation not started

**Repository:** `C:\Users\20883\OneDrive\Desktop\lstm-dqn-mpc\lstm-dqn-mpc`

## 1. Objective and boundaries

Establish one formal `dt=1 s`, `N=6`, ideal-foresight OSQP-MPC workflow for
voyages `060` through `066`. The controller uses actual future spline load
points `t+1...t+6`, applies only the first optimized move, and advances SOC
from the applied battery power.

The only objective terms are normalized hydrogen consumption, squared battery
power, squared SOC tracking error, and squared fuel-cell power variation. This
work does not add or modify LSTM, DQN, interpolation, physical limits, fallback
control, slack variables, terminal SOC terms, degradation models, filters,
reward functions, or automatic weight selection.

## 2. Audited workspace state and preservation rule

The repository is on `main` at `30661f7`, tracking `origin/main`. The following
pre-existing worktree changes belong to the superseded unnormalized N=6
experiment:

- `src/main/mpc_solvers/mpc_qp_formulation.py`
- `src/main/run_mpc_1s_n6_weight_selection.py`
- `tests/test_mpc_1s_n6_qsoc_feasibility.py`
- `tests/test_mpc_1s_n6_weight_selection.py`
- `outputs/mpc_1s_n6_h2_fcvar_batt_unnormalized/`
- `reports/mpc_1s_n6_h2_fcvar_batt_unnormalized_summary.md`
- `reports/mpc_1s_n6_h2_fcvar_batt_unnormalized_table.csv`

The latest explicit cleanup instruction supersedes those old experiment
changes, but their verified solver scaling, rolling-state update, metrics, and
plotting logic must be migrated before the old entry and tests are removed.
No `git reset --hard`, `git clean`, bulk overwrite, or unrelated worktree
cleanup is permitted.

The tracked `.codex_tmp/millisecond_10ms_lstm_smoke/` tree is retained because
inspection identified it as an LSTM experiment, not an N=6 weight experiment.
`tmp/` is retained unless an item-by-item audit proves it is both disposable
and in scope. The independent N=60 OSQP benchmark outputs remain untouched.

## 3. Formal objective

The new objective variant is:

`n6_h2_batt_soc_fcvar_normalized_v1`

For horizon `N=6`, it implements:

```
J = q_h2 * J_h2_norm
  + q_batt * J_batt_norm
  + q_soc * J_soc_norm
  + q_fc_var * J_fc_var_norm
```

with:

```
J_h2_norm = sum[k=0..N-1] m_h2(P_fc[k]) / m_h2(560 kW, dt=1 s)

J_batt_norm = sum[k=0..N-1] (P_batt[k] / 346.5 kW)^2

J_soc_norm = sum[k=1..N] ((SOC[k] - 0.55) / 0.05)^2

J_fc_var_norm = ((P_fc[0] - P_fc_prev) / 48 kW)^2
              + sum[k=1..N-1]
                    ((P_fc[k] - P_fc[k-1]) / 48 kW)^2
```

The hydrogen denominator is calculated from the repository's existing Dp0
model at `P_fc=560 kW` and `dt=1 s`; it is not refitted or hard-coded from a
different experiment. `SOC_ref=0.55` remains fixed throughout each voyage.

The QP remains in OSQP form `0.5*x'P*x + q'x`. Small objective-construction
helpers inside the existing formulation module will add each term once so the
four-objective variant composes verified logic without copying four complete
branches. Its only active weight fields are `q_h2`, `q_batt`, `q_soc`, and
`q_fc_var`. Retained legacy fields `q_ramp` and `q_terminal_soc` have no effect
on `P`, `q`, or metadata objective terms for this variant.

Metadata must contain:

- `objective_variant`
- exactly four `objective_terms`, in this order:
  `H2_norm`, `Batt_power_sq_norm`, `SOC_tracking_sq_norm`, and
  `FC_variation_sq_norm`
- the four active weights
- `h2_reference_kg_per_step`
- `battery_power_ref_kw=346.5`
- `soc_reference=0.55`
- `soc_band=0.05`
- `fuel_cell_variation_ref_kw_per_step=48`
- exact mathematical descriptions of all four normalized terms
- explicit absence of terminal SOC, slack, and extra ramp-cost terms

## 4. Physical model invariants

The implementation preserves the current constraints and dynamics:

- `P_fc` range: `0...560 kW`
- `P_batt` range: current `-346.5...346.5 kW` convention
- battery capacity: `693 kWh`
- SOC range: `0.2...0.8`
- power balance: `P_fc + P_batt = P_load`
- SOC update: `SOC[k+1] = SOC[k] - P_batt[k] / (3600*693)`
- FC hard ramp: at most `48 kW` per 1 s step, including the first step
  relative to the previously applied FC power

Changing any objective weight must not change constraint matrix `A` or bounds
`l` and `u`.

## 5. Single formal runner

The only new executable entry is:

`src/main/run_mpc_1s_n6_four_objective_sensitivity.py`

The current weight-selection runner is used as the migration source and then
removed. The new runner retains only reusable data loading, equivalent QP
scaling, persistent OSQP setup/update, closed-loop execution, metrics, and
plotting. It removes the A/B/C calibration, manual selection, composite gates,
old report-only selection flow, and all q_soc-only or clamping logic.

The CLI supports only the required controls:

- `--baseline`
- `--one-factor`
- `--voyage`
- `--output-dir`
- `--overwrite`

Without `--overwrite`, an existing completed configuration is not silently
replaced. A voyage filter is for reproducible diagnosis through the same
formal entry; it is never presented as a complete seven-voyage result.
`--baseline` runs only `baseline_1_1_1_1`. `--one-factor` first runs that
baseline or verifies an existing complete baseline with matching input and
implementation provenance, then runs the remaining 16 unique configurations
in deterministic one-factor order.

Solver failures are recorded with their original OSQP status. A failed step
does not update SOC and terminates that voyage; no later points are fabricated.

## 6. Experiment matrix and sequencing

The baseline is run and audited first:

`baseline_1_1_1_1 = (q_h2, q_batt, q_soc, q_fc_var) = (1,1,1,1)`

After recording its feasibility and numerical behavior, the one-factor run
uses values `{0.25, 0.5, 1, 2, 4}`. Removing the four duplicate baseline
entries yields exactly 17 unique configurations: the baseline plus four
non-unit settings for each of the four weights. No fourth dimension grid,
second-stage tuning, composite score, or automatic winner is allowed.

The baseline audit reports all seven voyage completion states, primal
infeasibility, maximum-iteration failures, physical residuals and bounds, SOC
direction, FC load tracking, and battery charge/discharge extrema. A baseline
failure is diagnosed as physical infeasibility or numerical non-convergence;
it is not hidden or repaired by changing the specified model.

## 7. Metrics and artifacts

Every configuration-voyage pair records all required solver, physical, raw
objective, normalized objective, weighted-contribution, and total-weighted-
objective metrics. Aggregate rows must be mathematically traceable to voyage
rows and must not omit failed voyages.

The required columns are:

- solver: `completed`, `solver_failure_count`, `primal_infeasible_count`,
  `max_iter_count`, `first_failure_time_s`, `mean_solve_time_ms`,
  `p95_solve_time_ms`, and `max_solve_time_ms`;
- physical: `initial_soc`, `final_soc`, `delta_soc`, `min_soc`, `max_soc`,
  `max_power_balance_residual_kw`, `max_fc_ramp_kw_per_step`, `max_fc_kw`,
  `min_fc_kw`, `max_batt_discharge_kw`, and `max_batt_charge_kw`;
- unweighted raw: `total_h2_kg`, `sum_p_batt_sq_kw2`,
  `sum_soc_error_sq`, and `sum_fc_delta_sq_kw2`;
- unweighted normalized: `J_h2_norm`, `J_batt_norm`, `J_soc_norm`, and
  `J_fc_var_norm`;
- weighted: one contribution column for each of the four terms plus
  `total_weighted_objective`.

Default output root:

`outputs/mpc_1s_n6_four_objective_sensitivity/`

Each configuration contains a compact config record, one seven-row voyage
metrics table, and one four-panel image per voyage:

1. load and fuel-cell power;
2. battery power;
3. SOC and fixed SOC reference;
4. four cumulative unweighted normalized objective terms.

Summary figures show, as physical quantities rather than only weighted cost,
the effect of each varied weight on hydrogen, squared battery power, squared
SOC error, squared FC change, final/minimum SOC, and completion rate.

The only report-root products are:

- `reports/mpc_1s_n6_four_objective_sensitivity_summary.md`
- `reports/mpc_1s_n6_four_objective_sensitivity_table.csv`

The report describes feasibility, observed trends, trade-offs, clearly
unacceptable configurations, and evidence-based candidate intervals for a
future manually approved study. It does not declare a final optimum.

## 8. Cleanup boundary

After reusable logic is migrated and references are checked, remove the three
old N=6 entries, their three dedicated test files, the explicitly listed old
reports, and these old result trees:

- `outputs/mpc_1s_n6_weight_selection/`
- `outputs/mpc_1s_n6_qsoc_feasibility/`
- `outputs/mpc_1s_n6_soc_clamping_diagnostic/`
- `outputs/mpc_1s_n6_h2_fcvar_batt/`
- `outputs/mpc_1s_n6_h2_fcvar_batt_unnormalized/`

The two untracked unnormalized reports are also superseded and removed. The
N=60 benchmark data outside these trees remains intact. The small historical
pointer `N60_HISTORICAL_BENCHMARK.md` inside the obsolete N=6 tree is replaced
by an accurate active-document reference to the actual retained N=60 output,
not treated as the N=60 evidence itself.

Historical design/plan records remain as history. Active references are
updated in `README.md`, `STATUS.md`, `docs/DATA_PROVENANCE.md`,
`docs/PROJECT_MAP.md`, and `docs/UNFINISHED_TASKS.md`; the cleanup audit is
updated if needed. A final repository search must find no active import or
execution reference to the removed entries.

## 9. Test-driven implementation and verification

A replacement focused test file is written before production changes. It
verifies:

1. each weight changes only its intended QP coefficients;
2. all four exact normalization denominators;
3. first-step and subsequent FC variation construction;
4. complete baseline metadata;
5. invariant `A`, `l`, and `u` under weight changes;
6. positive-semidefinite Hessian;
7. unchanged power-balance and SOC dynamics;
8. absence of terminal SOC, slack, and extra ramp costs;
9. exact 17-configuration matrix and baseline-first ordering;
10. required metrics, failure termination, overwrite guard, and plot/report
    schema where practical without running the full experiment in unit tests.

Verification order:

1. focused new MPC tests;
2. baseline seven-voyage run and audit;
3. 17-configuration one-factor run;
4. generated metric/report consistency checks;
5. `python -m compileall src`;
6. `python -m unittest discover -s tests -v`;
7. `git diff --check`;
8. final `git grep` and `git status` audit.

No experiment number or trend is reported until it exists in actual output.
No push is part of this task unless separately authorized.

## 10. Final handoff

The final response follows the requested ten-part order: actual deletions;
retained suspicious files and reasons; core modified files; final objective;
numeric normalization denominators; new and full-suite test results; baseline
seven-voyage results; 17-configuration sensitivity summary; report/output
paths; and observed problems plus evidence-based next search intervals.
Unrun work is labeled `未运行`, and claims without evidence are labeled
`无法确认`.
