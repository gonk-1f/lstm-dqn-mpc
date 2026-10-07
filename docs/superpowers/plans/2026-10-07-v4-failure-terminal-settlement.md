# v4 Failure Outcome and Terminal Settlement Implementation Plan

> **For agentic workers:** Implement inline with tests first. Do not change the main MLP, FC action grid, epsilon schedule, gamma, learning rate, or the 16 economic updates per completed episode.

**Goal:** Preserve executed failure prefixes for a separate completion outcome learner and add explicitly modeled terminal battery replenishment cost without conflating it with actual shore charging.

**Architecture:** `replay_episode` owns physical execution and separates actual ledgers from an optional modeled terminal ledger. A failed replay carries its executed prefix and reason. The existing cost DDQN trains only on complete, well-defined economic returns; a separate small outcome model trains on completed and failed prefixes, without acting as a safety filter. `train.py` reports both observed and modeled cost. Final greedy Train and Validation completion jointly gate checkpoint selection.

**Tech Stack:** Python, PyTorch, pytest, existing v2/v3 plant and shore models.

---

### Task 1: Modeled terminal settlement

**Files:** `src/v4/control.py`, `tests/test_v4_control.py`.

- [x] Add a failing test for a one-row ONBOARD episode ending below SOC 0.6: actual ledger unchanged, modeled ledger contains grid electricity and battery degradation, final physical SOC unchanged, last reward includes both.
- [x] Add a failing test that an actual following SHORE block never receives modeled terminal settlement.
- [x] Add a failing test that SOC at or above 0.6 produces no modeled settlement.
- [x] Implement an explicit `MODELED` terminal settlement using the existing `shore_interval` accountant and a fixed 624 kW charging request, capped exactly at 0.6. Simulate on a copy of the account state with FC already off; retain grid and battery degradation only, without modifying the executed path or adding rows.
- [x] Run focused and integrated control tests with `PYTHONPATH=src` and verify green.

### Task 2: Preserve executed failure prefixes

**Files:** `src/v4/control.py`, `tests/test_v4_control.py`.

- [x] Add a failing test in which an executed first row makes the next row have no feasible FC action; the raised error must contain exactly the executed transition and a distinct `no_feasible_action` outcome.
- [x] Extend `ReplayExecutionError` with immutable executed transitions and failure kind; attach those at every replay failure boundary without inventing an action on the failing row.
- [x] Run the focused control tests and verify green.

### Task 3: Learn failure outcomes without CNY penalties

**Files:** `src/v4/dqn.py`, `src/v4/train.py`, `tests/test_v4_dqn.py`, `tests/test_v4_training.py`.

- [x] Add failing tests that failed prefixes with actual CNY enter a separate outcome replay, completed trajectories enter with success labels, and no failed terminal is inserted into the economic replay as `done=True`.
- [x] Add a small binary outcome model and training method; keep economic `select_power` and `learn` unchanged. Classifier labels mean observed rollout failure under the sampled behavior, not proven per-action physical infeasibility. In multi-voyage samples, label earlier completed voyages successful.
- [x] Update train reporting to expose failed-prefix counts and distinct observed/modeled/comparable completed costs. Require final greedy Train and Validation completion for checkpoint eligibility and keep Test closed.
- [x] Run focused v4 tests with `PYTHONPATH=src` and verify green.

### Task 4: Documentation and verification

**Files:** `docs/v4_direct_power.md`, `docs/v4_status_2026-10-07.md`, this plan.

- [x] Document fixed modeled charging power as an assumption, the actual/modelled ledger split, failure outcome semantics, and the fact that the outcome model does not yet choose actions.
- [x] Run the relevant v2/v3/v4 tests and `git diff --check`.
- [x] Run a bounded Train/Validation smoke and inspect the report for Test-open count, modeled cost, prefix count, and checkpoint eligibility.
