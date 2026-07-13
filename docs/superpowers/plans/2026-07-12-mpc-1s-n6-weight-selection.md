# 1 s Offline Perfect-Foresight N=6 MPC Weight Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate the four authorized fixed-weight N=6 OSQP-QP MPC candidates on seven offline 1 s spline test voyages using ideal six-step foresight, and select one only if it passes the physical-first gates.

**Architecture:** Add a standalone N=6 experiment runner that reuses the convex QP and supporting OSQP helpers without changing the historical N=60 workflow. Use an exact affine scaling for numerical conditioning, terminate a voyage on the first final solver failure, keep trajectories in memory, write only compact scientific artifacts, and require an explicit manual decision after all four candidates finish.

**Executed outcome (2026-07-13):** A-D all failed the physical/SOC gate. The bounded search stopped with `no_candidate_selected`; no provisional config was created.

**Tech Stack:** Python 3.11, NumPy, pandas/Parquet, SciPy sparse QP formulation, OSQP, matplotlib, unittest.

---

### Task 1: Lock the N=6 timing and candidate contracts

**Files:**
- Create: `tests/test_mpc_1s_n6_weight_selection.py`
- Create: `src/main/run_mpc_1s_n6_weight_selection.py`

- [ ] **Step 1: Write failing tests** for `N6_HORIZON == 6`, the exact A-D candidate table, `ideal_future_window(loads, t)` returning `t+1..t+6` with same-voyage edge padding, and rejection of invalid decision indices.
- [ ] **Step 2: Verify RED** with `python -m unittest tests.test_mpc_1s_n6_weight_selection -v`; expected failure is an import error because the runner does not exist.
- [ ] **Step 3: Implement the constants and helper** with an immutable four-entry candidate tuple and a six-value NumPy return.
- [ ] **Step 4: Verify GREEN** with the same focused command.

### Task 2: Prove first-step execution and actual state update

**Files:**
- Modify: `tests/test_mpc_1s_n6_weight_selection.py`
- Modify: `src/main/run_mpc_1s_n6_weight_selection.py`

- [ ] **Step 1: Add failing tests** that pass a synthetic 19-value N=6 solution and assert `P_fc_plan=x[0]`, `P_batt_plan=x[6]`, `SOC_predicted=x[13]`, `P_batt_actual=load_actual-x[0]`, and `SOC_actual=current_soc-P_batt_actual/(3600*693)`.
- [ ] **Step 2: Verify RED**; expected failure is a missing first-step extraction function.
- [ ] **Step 3: Implement the minimal extraction function** without clipping SOC and without applying stages 2-6.
- [ ] **Step 4: Verify GREEN** and assert the generic QP metadata has 19 variables, seven SOC states and a six-point load requirement.

### Task 3: Implement rolling N=6 OSQP execution

**Files:**
- Modify: `tests/test_mpc_1s_n6_weight_selection.py`
- Modify: `src/main/run_mpc_1s_n6_weight_selection.py`

- [ ] **Step 1: Add a failing synthetic-voyage integration test** with eight 1 s samples; assert seven executions, first execution uses sample 1, final window pads only the last sample, previous actual FC feeds the first ramp bound, and plan/actual fields are separate.
- [ ] **Step 2: Verify RED**.
- [ ] **Step 3: Implement one persistent OSQP workspace per voyage**, using exact affine QP scaling, the existing bound-update helper and status fields `solved`, `solved inaccurate`, `maximum iterations reached`, iterations, primal/dual residuals and update-plus-solve time.
- [ ] **Step 4: Verify GREEN** and terminate at the first final failed solve; do not continue a fictitious frozen-state closed loop or add fallback physics.

### Task 4: Implement physical and solver metrics

**Files:**
- Modify: `tests/test_mpc_1s_n6_weight_selection.py`
- Modify: `src/main/run_mpc_1s_n6_weight_selection.py`

- [ ] **Step 1: Add failing metric tests** using a small synthetic control frame with hand-calculated load MWh, charge/discharge/throughput kWh, FC surplus kWh, over-load fraction, limit fractions, SOC delta, solver status ratios and constraint maxima.
- [ ] **Step 2: Verify RED**.
- [ ] **Step 3: Implement per-voyage and overall aggregators**; calculate H2 from unweighted physical Dp0 mass and expose raw residual maxima plus tolerance-classified violation counts.
- [ ] **Step 4: Verify GREEN** and prove changing `q_h2` does not rescale physical H2 for identical applied FC power.

### Task 5: Implement lightweight artifact and report generation

**Files:**
- Modify: `tests/test_mpc_1s_n6_weight_selection.py`
- Modify: `src/main/run_mpc_1s_n6_weight_selection.py`
- Create only if a candidate passes: `configs/benchmarks/mpc_1s_n6_provisional.json`

- [ ] **Step 1: Add a failing temporary-directory test** for the five required per-candidate summary files and absence of full `control_timeseries.csv`/`solve_times.csv` outputs.
- [ ] **Step 2: Verify RED**.
- [ ] **Step 3: Implement JSON/CSV/Markdown writers and compact per-voyage power/SOC plots** using an Agg backend.
- [ ] **Step 4: Add combined report generation** that reads an optional explicit provisional selection config; never computes a least-bad winner.
- [ ] **Step 5: Verify GREEN**.

### Task 6: Run the bounded four-candidate experiment

**Files generated:**
- Create: `outputs/mpc_1s_n6_weight_selection/`
- Create: `reports/mpc_1s_n6_weight_selection_summary.md`
- Create: `reports/mpc_1s_n6_weight_selection_table.csv`

- [ ] **Step 1: Run candidate A alone** with `python src/main/run_mpc_1s_n6_weight_selection.py --candidate A` and inspect all seven voyages for FC surplus, battery charging and SOC behavior.
- [ ] **Step 2: Run B, C and D sequentially** with the same CLI and their candidate IDs; do not add a fifth case.
- [ ] **Step 3: Compare the four summary and voyage tables** in the user's physical/SOC/power/economy/solver order.
- [ ] **Step 4: Write the explicit manual decision** with candidate decisions and engineering rationale; write a provisional config only if one candidate passes. The executed outcome was `no_candidate_selected`.
- [ ] **Step 5: Run `--report-only`** to regenerate the final aggregate report/table from the explicit decision.

### Task 7: Update active status and historical N=60 labeling

**Files:**
- Modify: `README.md`
- Modify: `STATUS.md`
- Modify: `docs/PROJECT_MAP.md`
- Modify: `docs/DATA_PROVENANCE.md`
- Modify: `docs/UNFINISHED_TASKS.md`
- Create: `outputs/mpc_1s_n6_weight_selection/N60_HISTORICAL_BENCHMARK.md`

- [ ] **Step 1: Mark N=60 as historical solver/performance benchmark** and N=6 as the formal control horizon; retain links to one N=60 configuration, QP check, timing and candidate summary.
- [ ] **Step 2: Record the N=6 result accurately** as offline ideal foresight and provisional/accepted according to evidence; do not call spline data measured 1 s data or call the run LSTM-MPC.
- [ ] **Step 3: Record the completed-but-negative N=6 selection result in unfinished tasks** and leave the authorized redesign, reusable controller/fallback and real LSTM integration tasks open.

### Task 8: Verify, commit and push

**Files:** all task-scoped files only.

- [ ] **Step 1: Run** `python -m compileall src`.
- [ ] **Step 2: Run** `python -m unittest discover -s tests -v`; expected: all tests pass.
- [ ] **Step 3: Run** `git diff --check`, inspect `git status --short`, staged names and artifact sizes; ensure no DQN/10 ms files or large step logs changed.
- [ ] **Step 4: Commit** with `experiment: select N6 MPC weights on 1s spline data`.
- [ ] **Step 5: Run** `git pull --rebase origin main`, stop on conflict, then `git push origin main`.
- [ ] **Step 6: Confirm** local HEAD equals `origin/main` and report both hashes.
