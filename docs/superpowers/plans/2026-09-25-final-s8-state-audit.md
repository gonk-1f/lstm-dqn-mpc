# Final S8 State Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the Train-only state audit against the current 30/8/5 dataset, shore-aware formal episodes, and frozen S8 schema, then fail closed unless preflight can authenticate that evidence.

**Architecture:** The formal episode loader remains the authority for the exact 30 s power/AIS/mode axis. Raw BMS is used only to supply causally aligned measured SOC evidence. Audit rows are restricted to ONBOARD decisions, reset their LPF/history at shore boundaries exactly like the environment, and bind their output manifest to current dataset manifests and the frozen S8 schema digest.

**Tech Stack:** Python 3, NumPy, pandas, Matplotlib, `unittest`, SHA-256 authenticated CSV/JSON artifacts.

---

### Task 1: Close the ONBOARD negative-deadband execution gap

**Files:**
- Modify: `tests/test_v2_formal_episode.py`
- Modify: `tests/test_v2_supervisory_rules.py`
- Modify: `src/v2/data/supervisory_rules.py`
- Modify: `src/v2/envs/formal_episode.py`

- [ ] Add a failing test proving an ONBOARD load in `[-1, 0) kW` reaches the MPC as exactly `0 kW`, while a value below `-1 kW` is rejected.
- [ ] Run the focused test and confirm the current backend forwards the negative value.
- [ ] Add `normalize_onboard_load_kw`: preserve nonnegative loads, clamp only the frozen one-kW deadband to zero, and reject more-negative loads. Apply it before state construction and MPC execution.
- [ ] Run supervisory and formal-episode tests to green.

### Task 2: Audit the actual frozen S8 decision state

**Files:**
- Modify: `tests/test_v2_train_state_audit.py`
- Modify: `src/v2/analysis/train_state_audit.py`
- Modify: `src/v2/main/run_train_state_audit.py`

- [ ] Add failing tests for an audit row containing `speed_kn/speed_fraction`, exact frozen S8 feature order, shore-boundary history reset, one-to-six-sample causal windows, and held-out payload exclusion.
- [ ] Add causal twelve-cluster SOC alignment to the formal episode timestamps with the frozen 10 s freshness cap and no source-row reuse.
- [ ] Build rows from `FormalTrainingDataset.load_train()`: use authenticated load, FC, battery, speed, and mode payloads; include ONBOARD only; normalize deadband load; reset LPF/history on shore; never open Validation/Test payloads.
- [ ] Require the produced normalized S8 columns to equal `FORMAL_STATE_FEATURE_NAMES` and scales to equal the frozen state constants.
- [ ] Run the focused audit tests to green.

### Task 3: Regenerate and authenticate final audit evidence

**Files:**
- Modify: `tests/test_v2_formal_preflight.py`
- Modify: `src/v2/preflight.py`
- Regenerate: `outputs/v2_dqn_state_audit/**`
- Regenerate: `docs/v2_dqn_state_audit.md`

- [ ] Add failing preflight tests that reject a missing, stale, or tampered audit manifest and require exact power/AIS/mode manifest hashes plus `FORMAL_STATE_SCHEMA_DIGEST`.
- [ ] Extend the audit manifest with the three current manifest hashes, S8 schema version/digest, exact 30 Train IDs and hashes, ONBOARD row accounting, and zero held-out payload opens.
- [ ] Rewrite the report around the frozen S8 schema, including speed evidence, Markov inventory, degradation-clipping assumptions, and an explicit statement that shore intervals do not request DQN decisions.
- [ ] Generate into temporary destinations, verify every artifact hash and numeric invariant, visually inspect all figures, then replace the stale audit bundle.
- [ ] Make `final_dqn_state` and curated dataset release VERIFIED only when the new audit matches current manifests and schema; otherwise remain NO-GO.

### Task 4: Refresh downstream evidence and training gate

**Files:**
- Modify as required by tests: `src/main/run_v2_objective_scale_audit.py`
- Regenerate: `docs/v2_objective_scale_audit.md` or preserve NO-GO if the current audit cannot be reproduced.
- Modify: dataset QA release metadata only after every blocker is independently discharged.

- [ ] Run the Train-only objective audit against the current 30-segment manifest and current ONBOARD semantics; fix stale enum names through a failing regression test if encountered.
- [ ] Require objective evidence to carry current dataset provenance rather than the previous 38-segment digest.
- [ ] Run preflight and require `FORMAL_TRAINING=GO` only if state, actions, objective scale, mode sidecar, and curated manifests all authenticate successfully.
- [ ] Run focused tests, all v2 tests, all repository tests, solver smoke, compile/import, artifact-hash verification, and `git diff --check`.
- [ ] Do not start formal training. Report the exact PyCharm terminal command only after the gate is GO.
