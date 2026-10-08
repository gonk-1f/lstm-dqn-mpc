# V2 Shore-Event and S8 Finalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace blanket signed-load shore inference with an authenticated causal mode sidecar, freeze the ONBOARD-only S8 state, and make shore intervals pause DQN/MPC while remaining in the preceding transition's physical and economic accounting.

**Architecture:** A new immutable mode sidecar owns raw FC/BMS/AIS evidence and produces one row per frozen 30 s point. `FormalTrainingDataset` authenticates and loads that sidecar. `FormalEpisodeBackend` executes one physical interval and reports whether it used MPC and whether the next sample is a decision boundary; `MultiRateWeightEnvironment` counts only actual MPC solves and drains an intervening shore event into one macro ledger. S8/checkpoints, undiscounted gamma, preflight, and the independent runner consume those contracts.

**Tech Stack:** Python 3, NumPy, pandas, SciPy PCHIP, PyTorch, unittest/pytest, SHA-256 manifests.

---

### Task 1: Add causal operating-mode classification contracts

**Files:**
- Modify: `src/v2/data/supervisory_rules.py`
- Modify: `tests/test_v2_supervisory_rules.py`

- [ ] **Step 1: Write the failing classifier tests**

Add tests that construct exact 30 s `ModeSample` sequences and assert:

```python
self.assertEqual(SHORE_MIN_CONSECUTIVE_SAMPLES, 3)
self.assertEqual(
    classify_operating_modes(three_valid_candidates),
    (OperatingMode.SHORE_PENDING,
     OperatingMode.SHORE_PENDING,
     OperatingMode.SHORE_CHARGING),
)
self.assertTrue(all(
    mode is OperatingMode.UNRESOLVED
    for mode in classify_operating_modes(two_candidate_run)
))
self.assertIs(
    classify_operating_modes((moving_fc_charge_sample,))[0],
    OperatingMode.ONBOARD,
)
```

Also assert that zero-speed positive hotel load remains `ONBOARD`, invalid
quality is `UNRESOLVED`, and no classification reads a future value through its
public single-step causal detector.

- [ ] **Step 2: Run the classifier tests and verify RED**

Run:

```powershell
python -m pytest tests/test_v2_supervisory_rules.py -q
```

Expected: failures because the pending/onboard/unresolved enum and three-sample
contract do not exist.

- [ ] **Step 3: Implement the minimal mode state machine**

Freeze these values and semantics:

```python
BATTERY_CHARGE_THRESHOLD_KW = 1.0
SHORE_MIN_CONSECUTIVE_SAMPLES = 3

class OperatingMode(Enum):
    ONBOARD = "onboard"
    SHORE_PENDING = "shore_pending"
    SHORE_CHARGING = "shore_charging"
    UNRESOLVED = "unresolved"
```

Use only current/past samples in the online detector. The offline sequence
classifier may mark a candidate run shorter than three as `UNRESOLVED`, but it
must preserve `SHORE_PENDING` for the first two rows of every confirmed run.

- [ ] **Step 4: Run the classifier tests and verify GREEN**

Run the Step 2 command. Expected: all tests pass.

### Task 2: Build and authenticate the raw-evidence mode sidecar

**Files:**
- Create: `src/v2/data/shore_mode_sidecar.py`
- Create: `src/v2/main/build_shore_mode_sidecar.py`
- Create: `tests/test_v2_shore_mode_sidecar.py`
- Generate: `data/processed/operating_dataset_zero_boundary_v2_modes/`

- [ ] **Step 1: Write failing sidecar unit tests**

Use temporary raw fixtures containing eight FC, twelve BMS, and AIS streams.
Assert exact output columns, signs, causal provenance, mode reasons, row count,
timestamp equality with the power/AIS axes, and manifest SHA-256 validation.
Required row fields are:

```python
(
    "timestamp", "time_s", "p_fc_total_kw", "p_batt_bus_kw",
    "speed_kn", "speed_provenance", "channels_complete",
    "freshness_valid", "duplicate_conflict", "mode", "mode_reason",
)
```

Tests must reject an existing destination, missing channels, path escape,
tampered payloads, and identities/splits that differ from the frozen dataset.

- [ ] **Step 2: Run the sidecar tests and verify RED**

```powershell
python -m pytest tests/test_v2_shore_mode_sidecar.py -q
```

Expected: import failure for the new builder.

- [ ] **Step 3: Implement the minimal sidecar builder**

Reuse the audited raw readers and power conventions. Align raw component
evidence to the already authenticated 30 s timestamps without opening Test
payloads during later training. Persist a manifest with dataset version,
policy/detector version, relative path, row count, split, and payload SHA-256.
The builder CLI is:

```powershell
python -m v2.main.build_shore_mode_sidecar `
  --dataset-root data/processed/operating_dataset_zero_boundary_v2 `
  --ais-root data/processed/operating_dataset_zero_boundary_v2_ais `
  --raw-root 'C:\Users\20883\OneDrive\Desktop\氢舟一号' `
  --output-root data/processed/operating_dataset_zero_boundary_v2_modes
```

- [ ] **Step 4: Run the sidecar tests and generate the real sidecar**

Run the Step 2 command, then the CLI above. Expected: unit tests pass and the
builder prints counts for all 53 parents plus Train/Validation unresolved rows.

### Task 3: Authenticate the sidecar in the formal loader

**Files:**
- Modify: `src/v2/data/formal_training_dataset.py`
- Modify: `tests/test_v2_formal_training_dataset.py`

- [ ] **Step 1: Write failing loader tests**

Extend `FormalEpisode` with immutable arrays/tuples for FC power, battery bus
power, operating mode, and reason. Assert three manifests share exact
parent/sample/split identity, payload axes are identical, hashes are checked,
and training/model selection still cannot open Test payloads.

- [ ] **Step 2: Run loader tests and verify RED**

```powershell
python -m pytest tests/test_v2_formal_training_dataset.py -q
```

Expected: constructor/open signature and episode fields are missing.

- [ ] **Step 3: Implement fail-closed three-manifest loading**

Change the factory to:

```python
FormalTrainingDataset.open(power_root, ais_root, mode_root)
```

Reject `UNRESOLVED` in the formal-ready check while allowing the loader to
expose counts for diagnostics. Compute Train macro volume from actual ONBOARD
events rather than `ceil(all_physical_steps / 5)`.

- [ ] **Step 4: Run loader tests and verify GREEN**

Run the Step 2 command. Expected: all tests pass.

### Task 4: Freeze the ONBOARD-only S8 state and checkpoint contract

**Files:**
- Modify: `src/v2/dqn/state.py`
- Modify: `src/v2/dqn/__init__.py`
- Modify: `src/v2/training/checkpoint.py`
- Modify: `tests/test_v2_formal_state.py`
- Modify: `tests/test_v2_checkpoint_resume.py`

- [ ] **Step 1: Write failing S8 and leakage tests**

Assert exact version, dimension, and order:

```python
self.assertEqual(FORMAL_STATE_SCHEMA_VERSION, "v2_s8_onboard_ais_v1")
self.assertEqual(FORMAL_STATE_DIMENSION, 8)
self.assertEqual(FORMAL_STATE_FEATURE_NAMES, (
    "soc", "causal_base_load_fraction", "load_residual_fraction",
    "recent_load_population_std_fraction",
    "recent_load_window_trend_fraction", "fuel_cell_power_fraction",
    "fuel_cell_delta_fraction", "speed_fraction",
))
```

Build identical current/history prefixes with different future speed/load/SOC/
FC/battery/mode tails and assert identical current S8. Assert cold-start/re-entry
produces base=current load and zero residual/std/trend/delta-FC. Assert the old
S9 checkpoint schema is rejected.

- [ ] **Step 2: Run S8/checkpoint tests and verify RED**

```powershell
python -m pytest tests/test_v2_formal_state.py tests/test_v2_checkpoint_resume.py -q
```

Expected: old S9 version/order/dimension and `shore_connected` argument fail.

- [ ] **Step 3: Implement minimal S8/schema digest changes**

Remove `shore_connected`, order the eight normalized values exactly as frozen,
keep `/600`, `trend*150/600`, and `/20 kn` fixed, and update error messages and
checkpoint metadata validation.

- [ ] **Step 4: Run S8/checkpoint tests and verify GREEN**

Run the Step 2 command. Expected: all tests pass.

### Task 5: Implement shore physics and event-driven macro boundaries

**Files:**
- Modify: `src/v2/envs/formal_episode.py`
- Modify: `src/v2/envs/multirate_weight_env.py`
- Modify: `tests/test_v2_formal_episode.py`
- Modify: `tests/test_v2_multirate_env.py`
- Modify: `tests/test_v2_economic_interval_semantics.py`

- [ ] **Step 1: Write failing focused environment tests**

Cover all approved contracts:

```python
self.assertEqual(transition.executed_mpc_steps, 3)
self.assertEqual(solver.calls, 3)
self.assertEqual(transition.ledger.shore_cost_cny, expected_shore_cost)
self.assertEqual(agent_side_effects.global_steps, 1)
```

Also test uninterrupted `M=5`, shore after solve 5, intermediate shore and
ONBOARD re-entry, final shore/terminal done, FC stop degradation once, onboard
FC charging not shore, long shore causing no DQN counters, target truncation,
SOC continuity, one efficiency factor, actual accepted-current degradation,
and post-shore S8 history reset.

- [ ] **Step 2: Run environment tests and verify RED**

```powershell
python -m pytest tests/test_v2_formal_episode.py tests/test_v2_multirate_env.py tests/test_v2_economic_interval_semantics.py -q
```

Expected: current backend counts shore as MPC, emits S9 shore states, and closes
at a fixed physical-step count.

- [ ] **Step 3: Implement interval-result semantics and shore accounting**

Extend the exact backend result with:

```python
mpc_solve_executed: bool
next_decision_ready: bool
```

The macro loop increments only when `mpc_solve_executed`; it keeps draining
shore ledgers until re-entry/terminal. Formal shore power is battery-side
capacity, accepted against simulated SOC target/bounds. Divide accepted battery
energy by 0.95 exactly once for grid energy/cost. Do not append shore samples to
the onboard state history. Reset the LPF/history on re-entry while preserving
SOC and cumulative degradation accounts.

- [ ] **Step 4: Run environment tests and verify GREEN**

Run the Step 2 command. Expected: all tests pass.

### Task 6: Freeze undiscounted training and update the runner/preflight

**Files:**
- Modify: `src/v2/training/dqn.py`
- Modify: `src/v2/preflight.py`
- Modify: `src/v2/main/train_formal_dqn.py`
- Modify: `tests/test_v2_dqn_training.py`
- Modify: `tests/test_v2_formal_preflight.py`
- Modify: `tests/test_v2_formal_training_cli.py`

- [ ] **Step 1: Write failing runtime/preflight tests**

Assert `DqnTrainingConfig.formal_baseline().gamma == 1.0`, the runner requires
the authenticated mode root, reports ONBOARD MPC solve and paused shore counts,
increments global/epsilon/optimizer only per emitted transition, keeps
Validation deterministic and non-training, and reports `NO-GO` when mode rows
are unresolved or hashes fail.

- [ ] **Step 2: Run runtime/preflight tests and verify RED**

```powershell
python -m pytest tests/test_v2_dqn_training.py tests/test_v2_formal_preflight.py tests/test_v2_formal_training_cli.py -q
```

Expected: gamma remains 0.99 and the runner still uses blanket signed load/S9.

- [ ] **Step 3: Implement the minimal runtime changes**

Set formal gamma to `1.0`, add `--mode-root`, remove signed-load classification
from the runner, validate only ONBOARD S8 boundaries, and keep the existing
30-round seeded Train shuffle, epsilon-by-global-transition schedule, 36-action
catalog, checkpoint/resume, and progress logging.

- [ ] **Step 4: Run runtime/preflight tests and verify GREEN**

Run the Step 2 command. Expected: all tests pass.

### Task 7: Update formal documentation and perform final verification

**Files:**
- Modify: `README.md`
- Modify: `docs/v2_preflight_report.md`
- Modify: `docs/v2_formal_training_runbook.md`
- Modify: `docs/v2_action_space_design.md`

- [ ] **Step 1: Replace stale S9/blanket-negative semantics**

Document S8, the authenticated mode sidecar, ONBOARD-only `M`, `gamma=1.0`,
shore reward attachment, Test isolation, and the exact remaining preflight
status. Do not describe project thresholds as vessel-measured or globally
optimal.

- [ ] **Step 2: Run focused and complete verification**

```powershell
python -m pytest tests/test_v2_supervisory_rules.py tests/test_v2_shore_mode_sidecar.py tests/test_v2_formal_training_dataset.py tests/test_v2_formal_state.py tests/test_v2_checkpoint_resume.py tests/test_v2_formal_episode.py tests/test_v2_multirate_env.py tests/test_v2_economic_interval_semantics.py tests/test_v2_dqn_training.py tests/test_v2_formal_preflight.py tests/test_v2_formal_training_cli.py -q
python -m pytest tests/test_v2_*.py -q
python -m v2.main.train_formal_dqn --preflight-only
python -m v2.main.train_formal_dqn --smoke-only --output-dir outputs/v2_formal_dqn_smoke
python -m compileall -q src/v2 tests
python -X utf8 -c "import v2; import v2.main.train_formal_dqn"
git diff --check
```

Expected: every command exits zero only if the generated mode evidence has no
formal blocker. If unresolved rows remain, tests and smoke may pass but formal
preflight must truthfully remain `NO-GO`; do not override that gate.

- [ ] **Step 3: Review and commit scoped changes**

Inspect `git status`, `git diff --stat`, and the full relevant diff. Stage only
the v2 design/plan, production code, tests, generated authenticated sidecar,
and updated documentation. Use terse Conventional Commit messages and push the
current branch only after fresh verification.
