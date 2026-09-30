# v2 LPF Single-Commit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make one successful 30 s ONBOARD interval advance the causal base-load LPF exactly once, invalidate artifacts produced under the duplicate-update contract, and restore a clean H4 training handoff.

**Architecture:** `NonlinearMPC.solve()` becomes side-effect free with respect to `CausalBaseLoadFilter`; the episode backend owns the only commit after a command has passed validation and is executed. A new control-semantics identity field rejects old checkpoints and derived artifacts. TDD establishes the contract before production changes.

**Tech Stack:** Python 3, NumPy, SciPy SLSQP, PyTorch, unittest/pytest, PowerShell, Git.

---

## File map

- `tests/test_v2_nonlinear_mpc.py`: solver purity and failed-solve filter-state tests.
- `tests/test_v2_formal_episode.py`: one commit per executed ONBOARD interval and SHORE reset behavior.
- `tests/test_v2_contracts.py`: exact control-semantics identity and old-artifact rejection.
- `src/v2/control/nonlinear_mpc.py`: remove solver-owned filter commit.
- `src/v2/envs/formal_episode.py`: retain backend-owned commit after command validation.
- `src/v2/contracts.py`: publish the single-commit contract version.
- `src/v2/preflight.py`: require and report the new contract in current artifacts.
- `docs/method_v2_multiscale_dqn_wmpc.md`: document one update per real interval.
- `docs/v2_timescale_selection.md`: distinguish configured tau from its discrete update rule.
- `docs/v2_formal_training_runbook.md`: replace stale outputs and retain the clean retraining commands.
- `outputs/v2_dqn_state_audit/*`: regenerated Train-only state evidence.
- `outputs/v2_objective_scale_audit/audit_summary.json`: regenerated objective evidence.
- `outputs/v2_failure_penalty_audit/audit_summary.json`: regenerated failure evidence.
- `outputs/v2_history_dqn_study/reward_scale_calibration.json`: regenerated Train-only Cref.

### Task 1: Establish the failing single-commit contract

**Files:**
- Modify: `tests/test_v2_nonlinear_mpc.py`
- Modify: `tests/test_v2_formal_episode.py`

- [ ] **Step 1: Change the solver contract test to require no mutation**

Replace the success portion of `test_failed_solves_do_not_consume_filter_observation_and_success_commits_once` with a purity assertion and rename it:

```python
def test_solve_never_consumes_filter_observation(self) -> None:
    from v2.control.nonlinear_mpc import MPCWeights

    weights = MPCWeights(0.5, 0.25, 0.25)
    _, estimator, controller = self._objects()
    estimator.observe(100.0, horizon=1)
    before = estimator.observed_base_kw

    result = controller.solve(200.0, 0.5, 100.0, weights, estimator)

    self.assertEqual(estimator.observed_base_kw, before)
    self.assertEqual(len(result.p_fc_kw), 5)
```

Keep the existing invalid-input, optimizer-failure, and invalid-solution assertions; each must continue to assert that `observed_base_kw` remains equal to `before`.

- [ ] **Step 2: Add a backend regression test using the real controller**

Add a test that seeds the backend filter, executes one ONBOARD interval, and checks exactly one recurrence:

```python
def test_onboard_interval_commits_base_filter_exactly_once(self) -> None:
    from v2.dqn.action_space import FINAL_DQN_ACTION_CATALOG
    from v2.envs.formal_episode import (
        FormalEpisodeBackend,
        build_formal_nonlinear_mpc,
    )

    backend = FormalEpisodeBackend(
        load_kw=np.asarray([200.0, 250.0]),
        speed_kn=np.asarray([4.0, 4.0]),
        fc_power_kw=np.asarray([180.0, 220.0]),
        battery_bus_kw=np.asarray([20.0, 30.0]),
        operating_mode=("onboard", "onboard"),
        mpc=build_formal_nonlinear_mpc(),
    )
    backend._base_filter.commit(100.0)
    before = float(backend._base_filter.observed_base_kw)
    alpha = backend._base_filter.alpha

    backend.execute_mpc_step(FINAL_DQN_ACTION_CATALOG[0].to_mpc_weights())

    expected = alpha * before + (1.0 - alpha) * 200.0
    self.assertAlmostEqual(float(backend._base_filter.observed_base_kw), expected)

    backend.execute_mpc_step(FINAL_DQN_ACTION_CATALOG[0].to_mpc_weights())
    expected = alpha * expected + (1.0 - alpha) * 250.0
    self.assertAlmostEqual(float(backend._base_filter.observed_base_kw), expected)
```

Use the real nonlinear controller so the current solver-owned commit is present during the red phase. Do not add a production-only filter accessor.

- [ ] **Step 3: Run the red tests**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path "src").Path
python -m pytest tests/test_v2_nonlinear_mpc.py tests/test_v2_formal_episode.py -q
```

Expected: the solver-purity assertion fails because `solve()` commits once; the backend recurrence assertion reports the equivalent of two commits for one interval. All unrelated assertions should still pass.

- [ ] **Step 4: Record the red-stage evidence in the working notes**

Record the exact failing test names and observed-versus-expected filter states in the implementation handoff. Do not change production code before both failures are observed.

### Task 2: Implement the minimal ownership fix

**Files:**
- Modify: `src/v2/control/nonlinear_mpc.py:534-708`
- Verify: `src/v2/envs/formal_episode.py:270-300`
- Test: `tests/test_v2_nonlinear_mpc.py`
- Test: `tests/test_v2_formal_episode.py`

- [ ] **Step 1: Remove the solver-owned commit**

Delete only this mutation from `NonlinearMPC.solve()`:

```python
base_load_filter.commit(load)
```

Keep `preview()` and all plan diagnostics unchanged.

- [ ] **Step 2: Retain the backend commit at the execution boundary**

Keep the existing backend call after first-command validation:

```python
self._base_filter.commit(load)
```

Do not move it before `p_fc`, battery power, or next-SOC validation. This ensures an invalid plan does not consume the observation.

- [ ] **Step 3: Run focused tests to verify green**

Run:

```powershell
python -m pytest tests/test_v2_nonlinear_mpc.py tests/test_v2_formal_episode.py -q
```

Expected: all tests pass and the one/two-interval recurrence checks match `exp(-30/180)` exactly once per interval.

- [ ] **Step 4: Run direct solver call-site tests**

Run:

```powershell
python -m pytest tests/test_v2_objective_scale_audit.py tests/test_v2_train_state_audit.py tests/test_v2_reward_scale_calibration.py -q
```

Expected: all pass. The objective-scale runner performs independent candidate solves and therefore requires no commit; sequential episode execution remains owned by the backend.

### Task 3: Version the corrected control semantics

**Files:**
- Modify: `src/v2/contracts.py`
- Modify: `tests/test_v2_contracts.py`
- Modify: `tests/test_v2_parameter_freeze.py`

- [ ] **Step 1: Write the failing semantics tests**

Add exact assertions:

```python
expected["base_load_filter_update_version"] = (
    "causal_single_commit_per_executed_interval_v1"
)
```

Add a rejection case that removes this field from a copy of `control_semantics()` and expects `IncompatibleArtifactError` from `require_v2_semantics()`.

- [ ] **Step 2: Run the semantics tests red**

Run:

```powershell
python -m pytest tests/test_v2_contracts.py tests/test_v2_parameter_freeze.py -q
```

Expected: FAIL because the new field is absent.

- [ ] **Step 3: Add the production constant and field**

In `src/v2/contracts.py` add:

```python
BASE_LOAD_FILTER_UPDATE_VERSION = (
    "causal_single_commit_per_executed_interval_v1"
)
```

Return it from `control_semantics()`:

```python
"base_load_filter_update_version": BASE_LOAD_FILTER_UPDATE_VERSION,
```

The existing exact-key comparison in `require_v2_semantics()` then rejects all old artifacts automatically.

- [ ] **Step 4: Run the semantics tests green**

Run the same pytest command. Expected: PASS.

### Task 4: Update preflight and method documentation

**Files:**
- Modify: `src/v2/preflight.py`
- Modify: `tests/test_v2_parameter_freeze.py`
- Modify: `docs/method_v2_multiscale_dqn_wmpc.md`
- Modify: `docs/v2_timescale_selection.md`
- Modify: `docs/v2_formal_training_runbook.md`

- [ ] **Step 1: Add a preflight assertion for the update contract**

Extend the existing LPF/preflight test so VERIFIED requires both:

```python
TAU_LPF_SECONDS == 180.0
control_semantics()["base_load_filter_update_version"]
    == "causal_single_commit_per_executed_interval_v1"
```

- [ ] **Step 2: Run the preflight test red**

Run:

```powershell
python -m pytest tests/test_v2_parameter_freeze.py -q
```

Expected: FAIL until the preflight detail and gate include the new contract.

- [ ] **Step 3: Implement the preflight detail**

Update the LPF item detail to state that 180 s is applied once per executed 30 s interval. Preserve `FROZEN_PROJECT_DESIGN` and do not claim vessel calibration or unique optimality.

- [ ] **Step 4: Update the three method documents**

Document:

```text
alpha = exp(-Ts/tau) = exp(-30/180)
one commit per successfully executed ONBOARD interval
solver preview has no filter-state side effect
```

Mark prior duplicate-update H4 outputs invalid for formal evidence. Keep action, reward, economic, dataset, and training hyperparameters unchanged.

- [ ] **Step 5: Run focused preflight tests green**

Run:

```powershell
python -m pytest tests/test_v2_parameter_freeze.py tests/test_v2_contracts.py -q
```

Expected: PASS.

### Task 5: Remove invalid derived outputs safely

**Files:**
- Remove to Windows Recycle Bin: the approved paths in the design spec.

- [ ] **Step 1: Resolve and validate exact deletion targets**

Use `Resolve-Path -LiteralPath` on each existing target. Assert each resolved path starts with the repository `outputs` absolute path. Do not use globs or unresolved environment variables.

- [ ] **Step 2: Move the invalid H4 artifacts to Recycle Bin**

Recycle exactly:

```text
outputs/v2_history_dqn_study/H4_tau180
outputs/v2_history_dqn_study/H4_tau180_selection_40
outputs/v2_history_dqn_study/H4_tau180_test
outputs/v2_history_dqn_study/H4_tau180_test_power_plots
outputs/v2_history_dqn_study/tau_lpf_validation_screen
```

- [ ] **Step 3: Recycle stale generated audit/calibration files**

Recycle current generated files under:

```text
outputs/v2_dqn_state_audit
outputs/v2_objective_scale_audit/audit_summary.json
outputs/v2_failure_penalty_audit/audit_summary.json
outputs/v2_history_dqn_study/reward_scale_calibration.json
```

Preserve `H4_round20_test*` and `reward_scale_calibration_tau90_backup.json`.

- [ ] **Step 4: Verify cleanup**

Run `Test-Path -LiteralPath` for every removed and retained path. Expected: all approved invalid targets are absent; all retained historical targets are present. Report that deletion is recoverable through Windows Recycle Bin.

### Task 6: Regenerate training prerequisites

**Files:**
- Recreate: `outputs/v2_dqn_state_audit/*`
- Recreate: `outputs/v2_objective_scale_audit/audit_summary.json`
- Recreate: `outputs/v2_failure_penalty_audit/audit_summary.json`
- Recreate: `outputs/v2_history_dqn_study/reward_scale_calibration.json`

- [ ] **Step 1: Regenerate the Train-only state audit**

Run:

```powershell
python -X utf8 -u -m v2.main.run_train_state_audit `
  --output-root outputs/v2_dqn_state_audit `
  --report-path docs/v2_dqn_state_audit.md
```

Expected: current control semantics include the single-commit field and `test_payloads_opened=0`.

- [ ] **Step 2: Regenerate objective-scale evidence**

Run:

```powershell
python -X utf8 -u src/main/run_v2_objective_scale_audit.py
```

Expected: status GO/VERIFIED, finite objective components, and current control semantics.

- [ ] **Step 3: Regenerate failure-penalty evidence**

Run:

```powershell
python -X utf8 -u -m v2.main.run_failure_penalty_audit
```

Expected: Train-only result with `test_payloads_opened=0` and a valid maximum completed raw cost below the fixed penalty.

- [ ] **Step 4: Regenerate reward scaling**

Run:

```powershell
python -X utf8 -u -m v2.main.run_reward_scale_calibration `
  --output outputs/v2_history_dqn_study/reward_scale_calibration.json
```

Expected: a finite positive Cref, the new control-semantics identity, and `test_payloads_opened=0`.

### Task 7: Full verification and training handoff

**Files:**
- Verify all modified code, tests, docs, and regenerated outputs.

- [ ] **Step 1: Run focused tests**

```powershell
python -m pytest `
  tests/test_v2_nonlinear_mpc.py `
  tests/test_v2_formal_episode.py `
  tests/test_v2_contracts.py `
  tests/test_v2_parameter_freeze.py `
  tests/test_v2_reward_scale_calibration.py -q
```

Expected: all pass.

- [ ] **Step 2: Run all v2 tests**

```powershell
python -m pytest tests -q
```

Expected: all tests pass; no failure is waived.

- [ ] **Step 3: Run preflight and solver smoke**

```powershell
python -X utf8 -u -m v2.main.train_history_dqn_study `
  --preflight-only --experiment H4 `
  --reward-scale outputs/v2_history_dqn_study/reward_scale_calibration.json

python -X utf8 -u -m v2.main.train_history_dqn_study `
  --smoke-only --experiment H4 `
  --reward-scale outputs/v2_history_dqn_study/reward_scale_calibration.json
```

Expected: `FORMAL_TRAINING=GO`, smoke PASS, and `test_payloads_opened=0`.

- [ ] **Step 4: Compile/import and diff checks**

```powershell
python -m compileall -q src tests
python -X utf8 -c "import v2; from v2.contracts import control_semantics; print(control_semantics())"
git diff --check
```

Expected: exit code 0; semantics print `tau_lpf_seconds: 180.0` and the single-commit version.

- [ ] **Step 5: Provide the clean retraining commands**

New training:

```powershell
$env:PYTHONPATH=(Resolve-Path "src").Path
$env:OPENBLAS_NUM_THREADS="1"
$env:OMP_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"

python -X utf8 -u -m v2.main.train_history_dqn_study `
  --experiment H4 `
  --reward-scale outputs/v2_history_dqn_study/reward_scale_calibration.json `
  --rounds 40 `
  --seed 42 `
  --device cpu `
  --log-every 50 `
  --output-dir outputs/v2_history_dqn_study/H4_tau180
```

Resume:

```powershell
python -X utf8 -u -m v2.main.train_history_dqn_study `
  --experiment H4 `
  --reward-scale outputs/v2_history_dqn_study/reward_scale_calibration.json `
  --rounds 40 `
  --seed 42 `
  --device cpu `
  --log-every 50 `
  --output-dir outputs/v2_history_dqn_study/H4_tau180 `
  --resume outputs/v2_history_dqn_study/H4_tau180/latest.pt
```

Do not run Validation/Test or tune DQN hyperparameters until the new 40-round training completes.
