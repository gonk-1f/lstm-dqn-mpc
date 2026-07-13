# N=6 q_soc-Only Feasibility Diagnosis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a preregistered N=6 ideal-foresight feasibility diagnosis for `q_soc={5,10,20}` while changing no other MPC setting and preserving every old A-D and N=60 artifact.

**Architecture:** Keep the existing N=6 QP, OSQP, closed-loop, metrics, and A-D interfaces unchanged by adding optional explicit-config dependency injection. Add a thin diagnostic runner that owns the new candidate set, isolated paths, feasibility gate, combined decision, and reports.

**Tech Stack:** Python 3, NumPy, pandas, SciPy sparse matrices, OSQP, matplotlib, `unittest`, Git.

**Executed outcome (2026-07-13):** `QSOC_5`, `QSOC_10`, and `QSOC_20` all completed 93,030 closed-loop steps. Only `QSOC_20` passed every fixed feasibility gate; the result remains a structural witness rather than a provisional/accepted paper configuration.

---

### Task 1: Lock the new experiment contract with failing tests

**Files:**
- Create: `tests/test_mpc_1s_n6_qsoc_feasibility.py`
- Test: `tests/test_mpc_1s_n6_weight_selection.py`

- [ ] **Step 1: Add a failing exact-candidate test** that imports `QSOC_CANDIDATES` and expects exactly:

```python
(
    {"candidate_id": "QSOC_5", "q_h2": 0.5, "q_soc": 5.0, "q_batt": 0.05, "soc_band": 0.05},
    {"candidate_id": "QSOC_10", "q_h2": 0.5, "q_soc": 10.0, "q_batt": 0.05, "soc_band": 0.05},
    {"candidate_id": "QSOC_20", "q_h2": 0.5, "q_soc": 20.0, "q_batt": 0.05, "soc_band": 0.05},
)
```

- [ ] **Step 2: Add failing configuration assertions** for `horizon=6`, `dt_seconds=1`, `q_ramp=0`, `q_terminal_soc=0`, `battery_capacity_kwh=693`, FC/battery/SOC/ramp boundaries, and that the three serialized configs differ only in `q_soc` plus their candidate identity.

- [ ] **Step 3: Add a failing explicit-config injection test** that passes `qsoc_candidate_config("QSOC_5")` into the shared candidate orchestration while patching the A-D `candidate_config()` lookup to raise. Assert the explicit config reaches voyage execution, metric construction, and metadata writing.

- [ ] **Step 4: Add failing path and decision tests** requiring `outputs/mpc_1s_n6_qsoc_feasibility`, `reports/mpc_1s_n6_qsoc_feasibility_*`, exact three-candidate completeness, no least-bad selection, and the two statuses `weight_only_sufficient` and `weight_only_insufficient_in_tested_range`.

- [ ] **Step 5: Run the new tests and confirm RED.**

Run:

```powershell
python -m unittest tests.test_mpc_1s_n6_qsoc_feasibility -v
```

Expected: import failure because `run_mpc_1s_n6_qsoc_feasibility.py` does not yet exist.

### Task 2: Add explicit-config reuse without changing A-D behavior

**Files:**
- Modify: `src/main/run_mpc_1s_n6_weight_selection.py`
- Test: `tests/test_mpc_1s_n6_weight_selection.py`
- Test: `tests/test_mpc_1s_n6_qsoc_feasibility.py`

- [ ] **Step 1: Add optional `config` parameters** to `run_voyage`, `_candidate_metadata`, `write_candidate_artifacts`, and `run_candidate`. Resolve once with this pattern so old callers are unchanged:

```python
resolved_config = config if config is not None else candidate_config(normalized_id)
```

- [ ] **Step 2: Pass `resolved_config` through all layers** and use it for QP building, metric calculation, hydrogen calculation, and the `model` section of `config.json`. Do not alter `CANDIDATES`, defaults, solver settings, timing, failure handling, or metric formulas.

- [ ] **Step 3: Run focused old and new tests.**

Run:

```powershell
python -m unittest tests.test_mpc_1s_n6_weight_selection tests.test_mpc_1s_n6_qsoc_feasibility -v
```

Expected: old A-D tests remain green; new tests advance to failures for the missing diagnostic runner behavior.

### Task 3: Implement the isolated q_soc diagnostic runner

**Files:**
- Create: `src/main/run_mpc_1s_n6_qsoc_feasibility.py`
- Modify: `tests/test_mpc_1s_n6_qsoc_feasibility.py`

- [ ] **Step 1: Define the exact candidates and independent defaults.**

```python
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "mpc_1s_n6_qsoc_feasibility"
DEFAULT_REPORTS_DIR = REPO_ROOT / "reports"
QSOC_CANDIDATES = (
    {"candidate_id": "QSOC_5", "q_h2": 0.5, "q_soc": 5.0, "q_batt": 0.05, "soc_band": 0.05},
    {"candidate_id": "QSOC_10", "q_h2": 0.5, "q_soc": 10.0, "q_batt": 0.05, "soc_band": 0.05},
    {"candidate_id": "QSOC_20", "q_h2": 0.5, "q_soc": 20.0, "q_batt": 0.05, "soc_band": 0.05},
)
```

- [ ] **Step 2: Implement `qsoc_candidate_config(candidate_id)`** with the same physical `default_config()` call as A, fixing `q_ramp=0` and `q_terminal_soc=0` and changing only `q_soc` across candidates.

- [ ] **Step 3: Implement the gate as explicit reasons, not a score.** A summary passes only when:

```python
closed_loop_complete is True
solver_failure_count == 0
physical_infeasible_point_count == 0
aggregate_metrics_comparable is True
soc_min >= 0.2 - N6_TOLERANCES["soc"]
soc_max <= 0.8 + N6_TOLERANCES["soc"]
worst_voyage_soc_net_change >= -0.03
solve_time_ms_max < 1000.0
```

- [ ] **Step 4: Implement `build_diagnostic_decision(summaries)`** to require exact IDs, seven non-debug voyages each, return all feasibility witnesses without ranking them, set `provisional_config_created=false`, and use only the two registered structural statuses.

- [ ] **Step 5: Implement combined artifacts** at `diagnostic_decision.json`, `mpc_1s_n6_qsoc_feasibility_table.csv`, and `mpc_1s_n6_qsoc_feasibility_summary.md`. Include the experiment boundary, fixed table, gate results and reasons, key physical/solver metrics, incomplete-prefix caveat, prior `q_soc=2` anchor, and historical N=60 pointer.

- [ ] **Step 6: Implement CLI modes** `--candidate <ID>`, `--all`, and `--report-only`. `--all` must execute `QSOC_5`, `QSOC_10`, then `QSOC_20`; final decision/report generation requires all three formal summaries. A single candidate run may write its candidate artifacts but must not invent a final diagnosis.

- [ ] **Step 7: Run focused tests until GREEN.**

Run:

```powershell
python -m unittest tests.test_mpc_1s_n6_qsoc_feasibility tests.test_mpc_1s_n6_weight_selection -v
```

Expected: all focused tests pass with no warnings or errors.

### Task 4: Run the formal experiment and audit results

**Files:**
- Create: `outputs/mpc_1s_n6_qsoc_feasibility/`
- Create: `reports/mpc_1s_n6_qsoc_feasibility_summary.md`
- Create: `reports/mpc_1s_n6_qsoc_feasibility_table.csv`

- [ ] **Step 1: Run a two-step-per-voyage smoke test outside the repository** and confirm the metadata records only the allowed change.

```powershell
$smoke = Join-Path $env:TEMP ("mpc_n6_qsoc_smoke_" + [guid]::NewGuid().ToString("N"))
python src/main/run_mpc_1s_n6_qsoc_feasibility.py --candidate QSOC_5 --output-root "$smoke/outputs" --reports-dir "$smoke/reports" --max-steps-per-voyage 2 --no-plots --expected-voyages 7
```

Expected: one partial `candidate_QSOC_5` artifact set under the temporary path and no final `diagnostic_decision.json`.

- [ ] **Step 2: Run all three formal candidates in preregistered order.**

Run:

```powershell
python src/main/run_mpc_1s_n6_qsoc_feasibility.py --all --expected-voyages 7
```

Expected: three candidate directories, 21 voyage plots, one decision JSON, one combined CSV, and one combined Markdown report; no full per-step CSV.

- [ ] **Step 3: Independently audit each candidate** for seven voyages, coverage, final failures/status counts, SOC extrema and worst net change, power/residual tolerances, FC surplus, battery throughput, and solver timing. Treat economy values as prefix-only when a candidate is incomplete.

- [ ] **Step 4: Verify preservation boundaries** using Git path lists and file hashes/status: no path under the old A-D output/report roots, N=60 tree, DQN, LSTM, or 10 ms subsystems changed.

### Task 5: Record current project status without overstating the result

**Files:**
- Modify: `README.md`
- Modify: `STATUS.md`
- Modify: `docs/DATA_PROVENANCE.md`
- Modify: `docs/PROJECT_MAP.md`
- Modify: `docs/UNFINISHED_TASKS.md`

- [ ] **Step 1: Add the new diagnostic as a separate result** after the old A-D result; preserve the old numbers and explicit `no_candidate_selected` decision.

- [ ] **Step 2: State the structural conclusion at the evidence level.** If all fail, use “insufficient for `q_soc={5,10,20}` under the tested terminal-free N=6 setup,” not “all larger SOC weights are impossible.” If any pass, call it a feasibility witness, not an accepted paper weight.

- [ ] **Step 3: Keep DQN blocked from formal training** until a fixed N=6 baseline is explicitly accepted, and retain N=60 as historical benchmark only.

- [ ] **Step 4: Check all documented commands, paths, dataset labels, candidate values, and status terms against generated JSON/CSV evidence.**

### Task 6: Verify, review, commit, and synchronize

**Files:**
- Test: complete repository
- Review: all task paths

- [ ] **Step 1: Run fresh verification.**

```powershell
python -m compileall src
python -m unittest discover -s tests -v
git diff --check
git status --short --branch
```

Expected: compile succeeds, all unit tests pass, diff check is clean, and only task-scoped paths are modified/untracked.

- [ ] **Step 2: Request an independent spec-compliance review and then a code-quality review.** Fix every Critical or Important finding and rerun the relevant tests.

- [ ] **Step 3: Stage only task files**, inspect `git diff --cached --name-status`, `git diff --cached --check`, total artifact size, and the largest new files. Confirm no full step logs, cache, environment, DQN, LSTM, 10 ms, old A-D, or N=60 path is staged.

- [ ] **Step 4: Commit with a new message** distinct from the prior A-D experiment:

```text
experiment: diagnose N6 MPC q_soc feasibility
```

- [ ] **Step 5: Synchronize only after every gate passes.**

```powershell
git pull --rebase origin main
git push origin main
```

If either command fails, do not force-push or bypass network restrictions; retain the local commits and report the exact blocker.
