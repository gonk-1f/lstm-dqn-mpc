# Spline 1s LSTM Optuna Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an isolated Optuna-based LSTM hyperparameter search for natural-clipped cubic-spline reconstructed 1 s load data.

**Architecture:** Add one new experiment entrypoint under `src/main/` that reads only `outputs/spline_1s_diagnostics/data/natural_clipped_by_voyage/` and the fixed 46/13/7 split metadata. The script owns data checks, per-voyage window generation, causal baseline evaluation, Optuna trial execution, seed checks, model saving, figures, and a caveated markdown report.

**Tech Stack:** Python, PyTorch, Optuna, pandas, numpy, matplotlib, unittest.

---

### Task 1: Guardrail Tests

**Files:**
- Create: `tests/test_lstm_spline_1s_hparam_search.py`
- Create later: `src/main/run_lstm_spline_1s_hparam_search.py`

- [ ] **Step 1: Write the failing tests**

```python
from run_lstm_spline_1s_hparam_search import (
    SearchTask,
    build_default_tasks,
    build_horizon_steps,
    build_windows_for_series,
    current_hold_forecast,
    last_slope_forecast,
    moving_average_hold_forecast,
    primary_score_from_metrics,
)
```

The tests assert:
- Task A reports horizons `[1, 6, 30, 60]`.
- Task B reports horizons `[1, 6, 30, 60, 120, 180]`.
- Window generation never crosses voyage boundaries because it works on one voyage series at a time.
- Baselines use only current and past values.
- Primary score for Task A averages WAPE through h60; Task B averages WAPE through h180.

- [ ] **Step 2: Run tests to verify RED**

Run: `D:\py\Python3\python.exe -m unittest tests.test_lstm_spline_1s_hparam_search`

Expected: import failure because `run_lstm_spline_1s_hparam_search` does not exist yet.

### Task 2: Minimal Search Utility Implementation

**Files:**
- Create: `src/main/run_lstm_spline_1s_hparam_search.py`

- [ ] **Step 1: Implement tested helpers**

Implement:
- `SearchTask`
- `build_default_tasks`
- `build_horizon_steps`
- `build_windows_for_series`
- `current_hold_forecast`
- `last_slope_forecast`
- `moving_average_hold_forecast`
- `primary_score_from_metrics`

These functions must not read MPC, DQN, or 30 s LSTM-MPC files.

- [ ] **Step 2: Run tests to verify GREEN**

Run: `D:\py\Python3\python.exe -m unittest tests.test_lstm_spline_1s_hparam_search`

Expected: tests pass.

### Task 3: Experiment Entrypoint

**Files:**
- Modify: `src/main/run_lstm_spline_1s_hparam_search.py`

- [ ] **Step 1: Add CLI and data checks**

Required CLI behavior:

```text
D:\py\Python3\python.exe src\main\run_lstm_spline_1s_hparam_search.py --smoke
```

The script must write `outputs/lstm_spline_1s_hparam_search/data_check.md` with source path, split counts, row counts, load stats, NaN flag, negative-load flag, duplicate `time_s` flag, scaler fit scope, `online_feasible=false`, and `uses_future_endpoint=true`.

- [ ] **Step 2: Add Optuna objective**

Use `optuna.create_study(direction="minimize")`. Search Task A and Task B separately. In smoke mode use one trial per task, reduced epochs, and limited windows. In full mode use 40 Task A trials and 30 Task B trials unless the user overrides them.

- [ ] **Step 3: Add outputs**

Write:
- `hparam_trials_taskA.csv`
- `hparam_trials_taskB.csv`
- `best_configs_taskA.json`
- `best_configs_taskB.json`
- `best_seed_check_taskA.csv`
- `best_seed_check_taskB.csv`
- `metrics_by_horizon_taskA.csv`
- `metrics_by_horizon_taskB.csv`
- `baseline_compare_taskA.csv`
- `baseline_compare_taskB.csv`
- `models/taskA_best.pt`
- `models/taskB_best.pt`
- required figures under `figures/`
- `REPORT_SPLINE_1S_LSTM_HPARAM_SEARCH.md`

All report text must state that these are best hyperparameters on spline-reconstructed 1 s data, not valid online 1 s forecasting evidence.

### Task 4: Verification

**Files:**
- Modify if needed: `src/main/run_lstm_spline_1s_hparam_search.py`
- Modify if needed: `tests/test_lstm_spline_1s_hparam_search.py`

- [ ] **Step 1: Run focused tests**

Run: `D:\py\Python3\python.exe -m unittest tests.test_lstm_spline_1s_hparam_search`

Expected: all tests pass.

- [ ] **Step 2: Run smoke experiment**

Run: `D:\py\Python3\python.exe src\main\run_lstm_spline_1s_hparam_search.py --smoke --no_reference_refresh`

Expected: output files exist for both tasks, one Optuna trial per task completes, and the report is written with the required caveat.

### Task 5: Status Update

**Files:**
- Modify: `project_status.md`
- Modify: `next_steps.md`
- Modify: `thread.md`

- [ ] **Step 1: Record what ran**

Record the script path, smoke command, generated output directory, and whether full 40/30 trial search has been started or remains next.

- [ ] **Step 2: Note git limitation**

This workspace is not a git repository, so plan and code commits cannot be made here.
