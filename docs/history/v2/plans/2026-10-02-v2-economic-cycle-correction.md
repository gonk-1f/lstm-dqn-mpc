# v2 Economic and Start-Cycle Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct fuel-cell start-cycle accounting, freeze the approved economic parameters, retire P1/P2, and regenerate the Train-only reward scale.

**Architecture:** Add a pure start-event counter in the fuel-cell degradation module and make the formal episode backend use it. Keep equipment price and replacement fraction explicit, split hydrogen and equipment provenance, and bind the changed economics through semantic versions. Remove only P1/P2-specific surfaces while preserving generic H1-H4 training configuration.

**Tech Stack:** Python 3, NumPy, PyTorch, unittest/pytest, PowerShell.

---

### Task 1: Establish failing economic and cycle tests

**Files:**
- Modify: `tests/test_v2_degradation_models.py`
- Modify: `tests/test_v2_economic_interval_semantics.py`
- Modify: `tests/test_v2_state_and_economics.py`
- Modify: `tests/test_v2_contracts.py`

- [ ] Add a test requiring only `OFF -> ON` to return one start cycle.
- [ ] Change price/provenance expectations to hydrogen `21.9` from Yang Table 6.
- [ ] Change the FC full-life replacement expectation to `1,050,000 CNY`.
- [ ] Change semantic-version expectations to the new reward and FC identities.
- [ ] Run focused tests and verify failures are caused by the old production behavior.

### Task 2: Implement the minimal production correction

**Files:**
- Modify: `src/v2/models/fuel_cell_degradation.py`
- Modify: `src/v2/envs/formal_episode.py`
- Modify: `src/v2/economics.py`
- Modify: `src/v2/contracts.py`

- [ ] Add the exact start-event counter and use it in the backend.
- [ ] Freeze the `0.5` replacement fraction and aggregate replacement charge.
- [ ] Split Yang hydrogen-price provenance from Zhou equipment-price provenance.
- [ ] Bump reward and FC degradation semantic versions.
- [ ] Run the focused tests and verify green.

### Task 3: Retire P1 and P2

**Files:**
- Modify: `src/v2/training/experiments.py`
- Modify: `src/v2/main/train_history_dqn_study.py`
- Modify: `tests/test_v2_dqn_experiments.py`
- Modify: `tests/test_v2_history_study_cli.py`
- Delete: `docs/superpowers/specs/2026-10-01-v2-p1-pilot-hyperparameters-design.md`
- Delete: `docs/superpowers/plans/2026-10-01-v2-p1-pilot-hyperparameters.md`
- Delete: `docs/superpowers/specs/2026-10-01-v2-p2-replay-only-pilot-design.md`
- Delete: `docs/superpowers/plans/2026-10-01-v2-p2-replay-only-pilot.md`
- Delete: `outputs/v2_p1_pilot/`
- Delete: `outputs/v2_p2_pilot/`

- [ ] Remove P1/P2 profiles, CLI choices, focused tests, and documents.
- [ ] Delete the two explicitly authorized output trees after verifying targets.
- [ ] Run experiment, CLI, checkpoint, and DQN focused tests.

### Task 4: Update documentation and regenerate reward scale

**Files:**
- Modify: `docs/v2_economic_parameters.md`
- Modify: `docs/v2_fc_degradation_model.md`
- Modify: `outputs/v2_history_dqn_study/reward_scale_calibration.json`

- [ ] Document the start-event rule, 21.9 price, 0.5 factor, provenance, and 1,050,000 CNY cap.
- [ ] Run the Train-only reward-scale generator and require `test_payloads_opened=0`.
- [ ] Authenticate the regenerated document under the new semantics.

### Task 5: Full verification

**Files:**
- Verify only.

- [ ] Run focused tests.
- [ ] Run every `test_v2*.py` test file.
- [ ] Run solver smoke and compile/import checks.
- [ ] Run `git diff --check` and inspect final status for retired P1/P2 artifacts.
