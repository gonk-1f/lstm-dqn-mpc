# v2 P1 Pilot Hyperparameters Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an isolated, checkpoint-bound P1 pilot configuration with replay capacity 75,000 and epsilon decay 100,000.

**Architecture:** Extend `DqnTrainingConfig` with exploration schedule fields and make training consume them. Add P1 as a separate experiment profile while preserving the H1-H4 matrix. Keep legacy checkpoint loading fail-closed by permitting missing schedule fields only for the historical schedule.

**Tech Stack:** Python, NumPy, PyTorch, unittest/pytest.

---

### Task 1: Specify P1 and schedule behavior

**Files:**
- Modify: `tests/test_v2_dqn_training.py`
- Modify: `tests/test_v2_dqn_experiments.py`
- Modify: `tests/test_v2_history_study_cli.py`

- [ ] Add failing tests for config-bound epsilon schedules, exact P1 values, unchanged H1-H4 values, and the `P1` CLI/output directory.
- [ ] Run the focused tests and confirm failures are caused by the absent P1/schedule behavior.

### Task 2: Bind exploration semantics to checkpoints

**Files:**
- Modify: `tests/test_v2_checkpoint_resume.py`
- Modify: `src/v2/training/dqn.py`
- Modify: `src/v2/training/checkpoint.py`

- [ ] Add a failing test proving schedule mismatch rejects resume and the historical schedule accepts a legacy identity.
- [ ] Add `epsilon_start`, `epsilon_end`, and `epsilon_decay_steps` to `DqnTrainingConfig`, validate them, and calculate epsilon from the active config.
- [ ] Compare checkpoint identities exactly, except for a missing legacy schedule that may map only to historical defaults.
- [ ] Run the focused checkpoint and DQN tests until green.

### Task 3: Add the isolated P1 entry path

**Files:**
- Modify: `src/v2/training/experiments.py`
- Modify: `src/v2/main/train_formal_dqn.py`
- Modify: `src/v2/main/train_history_dqn_study.py`

- [ ] Add P1 without changing the historical `history_study_profiles()` H1-H4 matrix.
- [ ] Make `_train` use the config-bound epsilon schedule.
- [ ] Allow `--experiment P1`, require scaled reward calibration, and default its output to a separate `P1` directory.
- [ ] Run focused CLI/experiment tests until green.

### Task 4: Verification

**Files:**
- No additional production files.

- [ ] Run focused P1, DQN, checkpoint, and CLI tests.
- [ ] Run all v2 tests affected by training/checkpoint behavior.
- [ ] Run compile/import checks and `git diff --check`.
- [ ] Report exact 15-round start and 30-round resume PowerShell commands without starting training or opening Test.
