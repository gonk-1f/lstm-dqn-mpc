# Direct-Power MLP Double-DQN Implementation Plan

> **For agentic workers:** Work task by task with tests before production edits.

**Goal:** Replace DQN-selected MPC weights with a 30 s, 61-action MLP Double-DQN that directly commands 0–600 kW FC power and learns from the unscaled actual four-component CNY ledger.

**Architecture:** Keep `operating_mode` as an external gate. ONBOARD decisions use eight causal features (SOC, latest load, two load changes, previous FC power, two cumulative life fractions, departure flag). A v4 environment executes the selected FC command against the next real load, checks battery/SOC feasibility, and settles subsequent shore requests without a DQN action. Its trainer uses the existing MLP network and Double-DQN target rule; archived v2/v3 code supplies the formal accounting model only.

**Tech Stack:** Python 3, NumPy, PyTorch, pytest, authenticated formal Train/Validation dataset.

---

### Task 1: Direct action and causal state

**Files:** `src/v4/control.py`, `tests/test_v4_control.py`.

- [x] Test `ACTION_KW == tuple(range(0, 601, 10))`, eight finite features at a virtual zero-load departure, and previous FC/load changes after one step.
- [x] Verify the new tests fail because `v4.control` is absent.
- [x] Implement exact action validation and a state builder using only prior measured rows and carried physical accounting state.
- [x] Verify these tests pass.

### Task 2: Physical episode replay

**Files:** `src/v4/control.py`, `tests/test_v4_control.py`.

- [x] Test a short ONBOARD → SHORE → ONBOARD episode: ONBOARD only invokes the policy; 0.6 SOC shore cap and post-shore SOC carry apply; FC is zero on shore; last ONBOARD reward includes shore actual cost once.
- [x] Test selected 10 kW-grid FC power equals executed FC power, actual battery power is `actual_load - selected_fc`, and infeasible actions fail closed with the source row index.
- [x] Verify red tests, then implement the replay using the shared `EconomicMPC.interval`/`shore_interval` accounting functions without calling `solve`.
- [x] Verify green tests and preserved four-component ledger totals.

### Task 3: MLP Double-DQN and Train-only pilot

**Files:** `src/v4/dqn.py`, `src/v4/train.py`, `tests/test_v4_dqn.py`, `tests/test_v4_training.py`.

- [x] Test 8-input/61-output MLP Q values, Double-DQN target, replay memory storing raw CNY rewards, and deterministic epsilon-greedy action selection.
- [x] Test the entrypoint loads Train for learning and Validation for selection while refusing Test access.
- [x] Verify red tests, then implement the agent and a bounded smoke runner with explicit failures rather than hidden power clipping.
- [x] Verify green tests; run a Train-only smoke and report costs, completions, FC starts, SOC range and Test-open count.

### Task 4: Cleanup and verification

**Files:** `docs/v4_direct_power.md`, obsolete generated `outputs/` artifacts, obsolete v3-specific tests that no longer guard shared accounting.

- [x] Document timing, action/state contract, actual CNY reward, mode routing, known infeasibility limit and Train/Validation/Test split.
- [x] Remove only reviewed superseded generated outputs and v3-specific tests from the new branch; retain v2 physics/data/ledger tests and the archived v2/v3 branch.
- [x] Run v4 tests, shared-accounting regression tests, compileall, `git diff --check` and a Git status audit.
- [x] Commit and push the new branch with the verified scope.
