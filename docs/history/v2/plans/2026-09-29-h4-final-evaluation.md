# H4 Final Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Authenticate H4 round 20, evaluate it on the existing fixed Test split, and render one executed power-allocation plot per Test episode.

**Architecture:** Extend the existing formal checkpoint selector and Test entrypoint so both reconstruct the explicitly requested H1-H4 experiment profile and validate its reward-scale identity. Add an immutable power-trace result alongside the existing policy metrics, sourced from the same environment execution, then render traces to a sibling plot directory so the sealed five-file Test bundle remains unchanged.

**Tech Stack:** Python 3, PyTorch, NumPy, Matplotlib, unittest, existing v2 formal evaluation contracts.

---

### Task 1: Bind selection and Test loading to the H4 experiment identity

**Files:**
- Modify: `src/v2/main/select_formal_dqn_checkpoint.py`
- Modify: `src/v2/main/evaluate_formal_dqn_test.py`
- Test: `tests/test_v2_checkpoint_selection.py`
- Test: `tests/test_v2_final_test_evaluation.py`

- [ ] Write failing tests showing `--experiment H4 --reward-scale <calibration>` constructs the H4 configuration and rejects a checkpoint with a different training identity.
- [ ] Run the focused tests and verify failure is caused by the missing experiment-aware CLI behavior.
- [ ] Add the minimal shared profile loading needed by both entrypoints; keep H1 defaults backward compatible.
- [ ] Run the focused tests and verify they pass.

### Task 2: Capture exact executed power traces

**Files:**
- Modify: `src/v2/envs/formal_episode.py`
- Modify: `src/v2/evaluation/formal_policy.py`
- Test: `tests/test_v2_formal_episode.py`
- Test: `tests/test_v2_formal_policy_evaluation.py`

- [ ] Write failing tests requiring one battery-bus power value per executed interval and an immutable trace containing elapsed seconds, requested load, FC power, battery-bus power, SOC, operating mode, and terminal status.
- [ ] Run the focused tests and verify the trace API is absent.
- [ ] Add `executed_battery_bus_power_kw` to the backend and a trace-returning evaluation wrapper that reuses the existing episode execution path.
- [ ] Run the focused tests and verify all trace lengths, signs, and failure truncation semantics.

### Task 3: Render Test episode power plots without altering the sealed result bundle

**Files:**
- Create: `src/v2/evaluation/power_trace_plots.py`
- Modify: `src/v2/main/evaluate_formal_dqn_test.py`
- Test: `tests/test_v2_power_trace_plots.py`
- Test: `tests/test_v2_final_test_evaluation.py`

- [ ] Write failing tests requiring one PNG per Test episode, an HTML index, and a CSV manifest in a separate plot directory.
- [ ] Run the focused tests and verify the plot writer does not yet exist.
- [ ] Implement deterministic Matplotlib plots with elapsed time in seconds, load, FC power, battery-bus power, zero line, shore-mode shading, SOC secondary axis, and explicit incomplete-episode labeling.
- [ ] Wire the Test entrypoint to write sealed metrics first and plots second to the separate requested directory.
- [ ] Run focused tests and verify both the sealed five-file bundle and plot outputs.

### Task 4: Run the authenticated H4 selection and Test evaluation

**Files:**
- Generate: `outputs/v2_history_dqn_study/H4_formal_selection_40/`
- Generate: `outputs/v2_history_dqn_study/H4_round20_test/`
- Generate: `outputs/v2_history_dqn_study/H4_round20_test_power_plots/`

- [ ] Run focused tests, all v2 tests, compile/import checks, and `git diff --check`.
- [ ] Run Validation selection across H4 rounds 1-40 and verify round 20 is selected with zero Test payloads opened.
- [ ] Run the H4 Test entrypoint once against the selected bundle and fixed `w_8_1_1` baseline.
- [ ] Inspect every generated PNG and summarize per-episode completion, costs, SOC range, and action distribution.
