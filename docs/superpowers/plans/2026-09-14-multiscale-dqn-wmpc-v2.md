# Multi-scale DQN-WMPC v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the formal 1 s / 84-action v1 method with an evidence-gated, 30 s nonlinear receding-horizon MPC and a slower macro-step DQN weight selector.

**Architecture:** New formal code lives under `src/v2` and cannot import v1 Dp0, reward, action, solver, checkpoint, or processed 1 s dataset modules. The lower controller minimizes only normalized base-load tracking, FC smoothing, and SOC-deadband objectives; the upper DQN holds one three-weight action for `M` actual MPC periods and receives the negative accumulated economic cost. Unsupported physical/economic parameters remain `None` and block formal training through an explicit preflight report.

**Tech Stack:** Python 3, NumPy, SciPy `optimize.minimize` (SLSQP), pandas, PyTorch, `unittest`.

---

### Task 1: Freeze v2 contracts and archive v1 experiment surface

**Files:**
- Create: `src/v2/contracts.py`
- Create: `src/v2/config.py`
- Create: `tests/test_v2_contracts.py`
- Create: `docs/v2_cleanup_manifest.md`
- Move: v1 method documents to `docs/archive_v1/`
- Delete: tracked v1-only directories listed in `docs/v2_cleanup_manifest.md`

- [ ] **Step 1: Write failing contract tests**

```python
def test_provisional_timescales_have_separate_semantics():
    cfg = V2Config.provisional()
    assert cfg.ts_mpc_seconds == 30.0
    assert cfg.n_mpc == 5
    assert cfg.dqn_switch_steps == 5
    assert cfg.prediction_seconds == 150.0
    assert cfg.switch_seconds == 150.0
    assert cfg.n_mpc_field != cfg.dqn_switch_steps_field

def test_v1_semantics_are_rejected():
    with self.assertRaises(IncompatibleArtifactError):
        require_v2_semantics({"method_version": "executed_closed_loop_reward"})
```

- [ ] **Step 2: Run `python -X utf8 -m unittest tests.test_v2_contracts -v` and verify failure due to missing `src.v2`**

- [ ] **Step 3: Implement immutable version and time-scale contracts**

```python
METHOD_VERSION = "multiscale_dqn_wmpc_v2"
MPC_OBJECTIVE_VERSION = "fc_base_smooth_soc_deadband_v1"
ACTION_TABLE_VERSION = "three_weight_simplex_behavior_filtered_v1"
REWARD_VERSION = "macro_interval_real_economic_cost_v1"

@dataclass(frozen=True)
class TimeScaleConfig:
    ts_mpc_seconds: float = 30.0
    n_mpc: int = 5
    dqn_switch_steps: int = 5
```

- [ ] **Step 4: Record exact pre-cleanup HEAD/status/paths, then remove only the manifest-listed v1 outputs and archive v1 docs**

- [ ] **Step 5: Re-run the contract test and commit**

### Task 2: Establish raw-data and plant provenance gates

**Files:**
- Create: `src/v2/data/raw_inventory.py`
- Create: `src/v2/preflight.py`
- Create: `tests/test_v2_data_guards.py`
- Create: `docs/v2_raw_excel_inventory.md`
- Create: `docs/v2_data_provenance.md`
- Create: `docs/v2_plant_configuration.md`

- [ ] **Step 1: Write failing tests that accept original `.xlsx/.xls` and reject processed `.csv/.mat` as v2 raw facts**
- [ ] **Step 2: Verify RED with `python -X utf8 -m unittest tests.test_v2_data_guards -v`**
- [ ] **Step 3: Implement workbook/sheet/column/unit/timestamp/interval/missing-rate inventory records and split guards**

```python
def require_train_only(split: str) -> None:
    if split.casefold() != "train":
        raise HeldOutDataAccessError("selection and calibration are Train-only")
```

- [ ] **Step 4: Write the current audit truth: no original Excel/specification is present in the workspace, so data rebuild is NO-GO**
- [ ] **Step 5: Run the data-guard tests and commit**

### Task 3: Implement calibrated FC efficiency, battery energy, and hydrogen accounting

**Files:**
- Create: `src/v2/models/fuel_cell_efficiency.py`
- Create: `src/v2/models/battery_energy.py`
- Create: `tests/test_v2_energy_models.py`
- Create: `docs/v2_fc_efficiency_model.md`

- [ ] **Step 1: Write failing tests for shape-preserving efficiency lookup, valid load fractions, LHV conversion, H2 units, charge efficiency, and discharge efficiency**
- [ ] **Step 2: Verify RED**
- [ ] **Step 3: Implement a provenance-bearing PCHIP efficiency map and formulas**

```python
LHV_H2_KWH_PER_KG = 120.0 / 3.6

def hydrogen_kg(p_fc_kw, dt_seconds, eta):
    return p_fc_kw * (dt_seconds / 3600.0) / (LHV_H2_KWH_PER_KG * eta)

def next_soc(soc, p_batt_kw, dt_seconds, capacity_kwh, eta_chg, eta_dis):
    battery_side_kw = p_batt_kw / eta_dis if p_batt_kw >= 0 else eta_chg * p_batt_kw
    return soc - battery_side_kw * (dt_seconds / 3600.0) / capacity_kwh
```

- [ ] **Step 4: Ensure no v2 module imports `fc_dp0_curve` and keep the formal efficiency map uncalibrated until sourced points are supplied**
- [ ] **Step 5: Run tests and commit**

### Task 4: Implement FC and battery degradation with normalization guards

**Files:**
- Create: `src/v2/models/fuel_cell_degradation.py`
- Create: `src/v2/models/battery_degradation.py`
- Create: `tests/test_v2_degradation_models.py`
- Create: `docs/v2_fc_degradation_model.md`
- Create: `docs/v2_battery_degradation_model.md`

- [ ] **Step 1: Write failing tests for low/high/transient FC loss, OFF exclusion, hysteresis/dwell starts, `F(SOC)`, `G(I)`, and `Q_nominal != Q_lifetime`**
- [ ] **Step 2: Verify RED**
- [ ] **Step 3: Implement raw microvolt and weighted-Ah accounting without inventing relative-life denominators**
- [ ] **Step 4: Add guards that reject degradation-to-CNY conversion until relative lifetime fractions are calibrated**
- [ ] **Step 5: Run tests and commit**

### Task 5: Implement the three-objective nonlinear receding-horizon MPC

**Files:**
- Create: `src/v2/control/causal_base_load.py`
- Create: `src/v2/control/nonlinear_mpc.py`
- Create: `tests/test_v2_nonlinear_mpc.py`
- Create: `docs/method_v2_multiscale_dqn_wmpc.md`

- [ ] **Step 1: Write failing tests for `alpha=exp(-Ts/tau)`, no future measurement access, exact three-objective value, SOC deadband, hard bounds, ramp, power balance, N=5 plan, and first-step-only execution**
- [ ] **Step 2: Verify RED**
- [ ] **Step 3: Implement the causal persistence forecast and nonlinear SLSQP controller**

```python
J = q_base * sum(((p_fc - p_base_ref) / p_fc_scale) ** 2)
J += q_smooth * sum((delta_p_fc / delta_p_fc_scale) ** 2)
J += q_soc * sum(soc_deadband_penalty(soc_path, soc_l, soc_h, soc_scale))
```

- [ ] **Step 4: Enforce `q_base+q_smooth+q_soc=1`, positive weights, physical constraints, deterministic cold/warm starts, and structured numerical-vs-physical failures**
- [ ] **Step 5: Run tests and commit**

### Task 6: Build and screen the three-weight candidate action bank

**Files:**
- Create: `src/v2/dqn/action_space.py`
- Create: `src/v2/analysis/action_screening.py`
- Create: `tests/test_v2_action_space.py`
- Create: `docs/v2_action_space_design.md`

- [ ] **Step 1: Write failing tests for all 36 positive tenth-grid simplex candidates, deterministic IDs, Train-only screening, and no held-out selection**
- [ ] **Step 2: Verify RED**
- [ ] **Step 3: Implement candidate generation and behavior-fingerprint/Pareto/near-duplicate/medoid interfaces**
- [ ] **Step 4: Keep the final DQN catalog unset until Train data and solver audit exist**
- [ ] **Step 5: Run tests and commit**

### Task 7: Implement v2 operating state and macro-interval economics

**Files:**
- Create: `src/v2/dqn/state.py`
- Create: `src/v2/economics.py`
- Create: `tests/test_v2_state_and_economics.py`
- Create: `docs/v2_economic_parameters.md`

- [ ] **Step 1: Write failing tests for seconds-based causal windows, economic component units, terminal recharge to `episode_initial_soc`, and missing shore-price gate**
- [ ] **Step 2: Verify RED**
- [ ] **Step 3: Implement the nine candidate causal state features and raw CNY interval ledger**
- [ ] **Step 4: Ensure economic costs have no extra `0.3/0.4/0.3` weights and keep reward scaling Train-calibrated only**
- [ ] **Step 5: Run tests and commit**

### Task 8: Implement the M=5 macro-step environment and artifact incompatibility

**Files:**
- Create: `src/v2/envs/multirate_weight_env.py`
- Create: `src/v2/training/artifacts.py`
- Create: `tests/test_v2_multirate_env.py`
- Create: `tests/test_v2_artifacts.py`

- [ ] **Step 1: Write failing tests showing one action is held for five actual MPC executions, reward accumulates across exactly that interval, and next state/replay transition appears only at its boundary**
- [ ] **Step 2: Verify RED**
- [ ] **Step 3: Implement macro execution with separate `n_mpc` and `dqn_switch_steps` fields even though both equal five**
- [ ] **Step 4: Reject every v1 checkpoint/replay and prove deterministic v2 resume metadata round-trip**
- [ ] **Step 5: Run tests and commit**

### Task 9: Add Train-only time-scale and solver-reliability audits

**Files:**
- Create: `src/v2/analysis/timescale_audit.py`
- Create: `src/v2/analysis/solver_audit.py`
- Create: `tests/test_v2_train_only_audits.py`
- Create: `docs/v2_timescale_selection.md`

- [ ] **Step 1: Write failing split-guard tests and synthetic autocorrelation/rolling-variance/change-point/regime-duration tests**
- [ ] **Step 2: Verify RED**
- [ ] **Step 3: Implement Train-only `M in {5,10}` sensitivity and paired warm-start/cold-start audit; keep `N=5` fixed for this provisional baseline**
- [ ] **Step 4: Explicitly forbid Validation/Test from selecting `N`, `M`, `tau_LPF`, deadband, state, action catalog, or reward scale**
- [ ] **Step 5: Run tests and commit**

### Task 10: Replace formal entrypoints and documentation

**Files:**
- Create: `src/v2/main/run_preflight.py`
- Create: `src/v2/main/run_train_only_timescale_audit.py`
- Modify: `README.md`
- Create: `docs/v2_preflight_report.md`

- [ ] **Step 1: Write failing tests that formal training stops before any payload read when a required calibration is missing**
- [ ] **Step 2: Verify RED**
- [ ] **Step 3: Make README describe only v2 as current; link v1 history/archive without presenting it as runnable formal method**
- [ ] **Step 4: Generate the 18-section preflight report and state `FORMAL_TRAINING = NO-GO` for every unresolved evidence item**
- [ ] **Step 5: Run all v2 tests, then the full suite, compileall, import boundary scan, and `git diff --check`**

### Task 11: Final verification and branch handoff

**Files:**
- Modify: `docs/v2_preflight_report.md`

- [ ] **Step 1: Record exact test counts, solver smoke results, deleted archive targets, and remaining unsupported assumptions**
- [ ] **Step 2: Verify the working tree contains no unmanifested user-file deletion and no v2-to-v1 formal imports**
- [ ] **Step 3: Run `python -X utf8 -m unittest discover -s tests -v`, `python -X utf8 -m compileall -q src tests`, and `git diff --check`**
- [ ] **Step 4: Use the finishing-development workflow and report the branch without merging or pushing**
