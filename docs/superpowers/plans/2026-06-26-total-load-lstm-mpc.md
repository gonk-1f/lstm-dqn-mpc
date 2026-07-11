# Total Load LSTM-MPC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a new 66-voyage energy-side equivalent total-load LSTM and LSTM-H2-MPC baseline without overwriting the previous propulsion-load baseline.

**Architecture:** Add a separate total-load data build path from `total_load_excels`, then point new LSTM training/evaluation and MPC execution entrypoints at the new segmented dataset and split. Keep the MPC controller structure unchanged except for the specified total-load target, 1806 kWh battery capacity, disabled fuel-cell hard ramp constraint, and one-step-ahead t+1..t+6 timing.

**Tech Stack:** Python, pandas, PyTorch, CasADi MPC, unittest, matplotlib.

---

### Task 1: Total-Load Dataset Builder

**Files:**
- Create: `src/main/build_total_load_dataset_721.py`
- Test: `tests/test_total_load_dataset_721.py`

- [ ] Write tests that create small Excel fixtures and verify column mapping, `load_total_kw = fuel_cell_total_kw + battery_total_kw`, voyage-level chronological 46/13/7 splitting, and no original file overwrite.
- [ ] Implement `build_total_load_dataset_721.py` with reusable `build_dataset()` and CLI defaults for `total_load_excels`, `outputs/total_load_dataset_build`, and `outputs/config`.
- [ ] Run `python -m unittest tests.test_total_load_dataset_721`.

### Task 2: Total-Load LSTM Training Entry

**Files:**
- Modify: `src/main/run_train_lstm_721.py`
- Create: `src/main/run_train_lstm_total_load_721.py`
- Test: `tests/test_train_lstm_721.py`

- [ ] Add or update tests so training windows group by `voyage_id` when `voyage_name` is absent and never cross voyage boundaries.
- [ ] Add `--target_col` support to the existing reusable training function while preserving old defaults.
- [ ] Add `run_train_lstm_total_load_721.py` that defaults to `outputs/total_load_dataset_build/total_load_66_segments.csv`, `outputs/config/voyage_split_total_load_721.json`, and `outputs/lstm_total_load_721`.
- [ ] Run `python -m unittest tests.test_train_lstm_721`.

### Task 3: Total-Load LSTM-MPC Entry

**Files:**
- Modify: `src/main/run_lstm_mpc_test.py`
- Create: `src/main/run_lstm_mpc_total_load_test.py`
- Test: `tests/test_lstm_mpc_nextstep_timing.py`, `tests/test_mpc_ramp_constraint_toggle.py`

- [ ] Add tests for the new total-load weight set, 1806 kWh capacity, disabled hard ramp constraint, and `mpc_load_ref = lstm_pred[:6]` when LSTM history is available.
- [ ] Parameterize the current runner enough to accept alternate checkpoint, split, source CSV, load metadata, output directory, and weight set.
- [ ] Add `run_lstm_mpc_total_load_test.py` with defaults for the new checkpoint and output directory.
- [ ] Run `python -m unittest tests.test_lstm_mpc_nextstep_timing tests.test_mpc_ramp_constraint_toggle`.

### Task 4: Execute Fixed Baseline

**Files:**
- Modify: `outputs/config/mpc_weight_sets.json`
- Generated: `outputs/total_load_dataset_build/*`
- Generated: `outputs/config/voyage_split_total_load_721.json`
- Generated: `outputs/config/SPLIT_TOTAL_LOAD_721.txt`
- Generated: `outputs/lstm_total_load_721/*`
- Generated: `outputs/lstm_mpc_total_load_test/*`

- [ ] Add weight set `dp0_total_load_raw_h2_soc_batt_ramp_nextstep_v1` with the requested raw physical weights.
- [ ] Run the dataset build command and verify exactly 66 voyages, split 46/13/7, target column metadata, and no major data-integrity failures.
- [ ] Train LSTM into `outputs/lstm_total_load_721`.
- [ ] Run fixed LSTM-MPC once into `outputs/lstm_mpc_total_load_test`.
- [ ] Run the required core tests.

### Task 5: Status and Report

**Files:**
- Modify: `project_status.md`
- Modify: `next_steps.md`
- Modify: `thread.md`

- [ ] Update status files with the new total-load dataset, split, LSTM checkpoint, MPC output directory, battery capacity, ramp toggle, and timing.
- [ ] Report the required 14 summary items and call out any failed checks or blocked dependency.
