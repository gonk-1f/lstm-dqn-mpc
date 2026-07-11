# Charge-Sustaining H2-MPC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish a physically consistent charge-sustaining fixed H2-MPC baseline without changing LSTM timing or training DQN.

**Architecture:** Extend the total-power CasADi MPC with explicit SOC reference modes and a solve-time battery charge bound. Pass the voyage-start SOC through the runner into each rolling solve, export the resulting diagnostics, and select a baseline using physical eligibility checks before objective ranking.

**Tech Stack:** Python, NumPy, pandas, CasADi/IPOPT, matplotlib, unittest.

---

### Task 1: SOC Reference Unit Tests

**Files:**
- Create: `tests/test_mpc_soc_reference_modes.py`
- Test: `tests/test_lstm_mpc_zero_delay.py`

- [ ] Add failing tests for `initial_soc`, `reserve_only`, `fixed_target`, the recovery charge limit, and voyage-level reference persistence.
- [ ] Run `python -m unittest tests.test_mpc_soc_reference_modes` and confirm failures are caused by missing configuration/functions.

### Task 2: CasADi SOC Objective And Bounds

**Files:**
- Modify: `src/mpc/solvers/casadi_solver.py`
- Modify: `src/mpc/controllers/reference_generator.py`

- [ ] Add validated configuration fields for SOC reference mode, reserve, terminal deadband, and recovery charge limit.
- [ ] Add helper functions that resolve the reference and calculate numeric stage/terminal penalties for reporting tests.
- [ ] Add a `soc_reference_value` CasADi parameter so voyage-start SOC remains constant across rolling solves.
- [ ] Replace fixed-target stage and terminal terms in the total-power solver with mode-specific expressions.
- [ ] Apply `P_batt >= -max_charge_power_kw` through solve-time bounds when recovery limiting is active.
- [ ] Export mode, reference, reserve, recovery-limit state, and `h2_cost_raw_kg`.
- [ ] Re-run `python -m unittest tests.test_mpc_soc_reference_modes` and confirm green.

### Task 3: Runner Configuration And Metrics

**Files:**
- Modify: `src/main/run_lstm_mpc_test.py`
- Test: `tests/test_lstm_mpc_zero_delay.py`

- [ ] Map nested SOC recovery configuration into `CasadiMPCConfig`.
- [ ] Pass `init_soc` as the constant voyage reference for every rolling solve.
- [ ] Preserve `build_mpc_load_ref(current, pred[:5])` and its existing regression test.
- [ ] Add timeseries and summary metrics for reference mode, charge-limit activation, FC/battery extrema, threshold durations, first-10-minute SOC rise, SOC delta, and adjusted H2 alias.
- [ ] Add failing tests first, then implement and rerun the targeted test modules.

### Task 4: Candidate Weight Sets

**Files:**
- Modify: `src/main/run_lstm_mpc_test.py`
- Modify: `outputs/config/mpc_weight_sets.json`

- [ ] Preserve the old aggressive group as `dp0_batt_penalty_v1_old_fixed_target_diag`.
- [ ] Add `dp0_cs_initial_v1`, `dp0_cs_initial_terminal_v1`, `dp0_reserve_only_v1`, and `dp0_cs_batt_protect_v1` with Dp0 mass/dimensionless objective flags and recovery limiting.
- [ ] Set the provisional default to `dp0_cs_initial_v1`; replace it with the sweep recommendation only after full verification.

### Task 5: Sweep Eligibility And Figures

**Files:**
- Modify: `src/main/run_lstm_mpc_weight_sweep.py`
- Test: `tests/test_lstm_mpc_zero_delay.py`

- [ ] Add failing recommendation tests for diagnostic exclusion and every abnormal-behavior threshold.
- [ ] Aggregate the new physical metrics and explicitly record eligibility/rejection reasons.
- [ ] Exclude diagnostic and `fixed_target` candidates from recommendation.
- [ ] Generate the required power/SOC, objective, and first-20-minute SOC-reference figures.
- [ ] Update the diagnosis report with root cause, H2 mass semantics, old/new comparison, and the selected future DQN baseline.

### Task 6: Full Verification

**Files:**
- Update: `project_status.md`
- Update: `next_steps.md`
- Update: `thread.md`

- [ ] Run `python -m unittest tests.test_mpc_soc_reference_modes tests.test_lstm_mpc_zero_delay tests.test_train_lstm_721`.
- [ ] Run `python src/main/run_lstm_mpc_weight_sweep.py --output_dir outputs/lstm_mpc_weight_sweep`.
- [ ] Verify all four voyages for the recommendation against every acceptance threshold.
- [ ] Run the recommended set into `outputs/lstm_mpc_test/` so the fixed-name outputs represent the accepted current baseline.
- [ ] Update the three status files with measured facts only.
