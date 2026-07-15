# N=6 MPC Near-Reference SOC Clamping Diagnostic Implementation Plan

> Scope update (2026-07-15): the user explicitly requested a simpler, core-only implementation. The completed path keeps the fixed eight cases, required metrics/tests, five plot groups, CSV, and report; the broader artifact/provenance/decision API steps below are historical planning notes, not current acceptance requirements.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run an isolated eight-case synthetic diagnostic that determines whether `q_soc=20` over-corrects SOC deviations of only ±0.02, without changing formal MPC configuration or retained voyage results.

**Architecture:** A new runner constructs synthetic arrays and reuses the verified `run_voyage()` and explicit `qsoc_candidate_config()` APIs. It writes numerical/provenance artifacts and plots to a new root, while a separately reviewed decision JSON supplies the qualitative label used by report-only rendering. Existing N=6 voyage runners, QP code, configs, N=60 results, LSTM, and DQN remain untouched.

**Tech Stack:** Python 3.11, NumPy, pandas, matplotlib, OSQP through the existing runner, PyArrow parquet, `unittest`, Git.

---

## File map

- Create `src/main/run_mpc_1s_n6_soc_clamping_diagnostic.py`: synthetic cases, metrics, provenance, plots, CLI, artifact validation, and report rendering.
- Create `tests/test_mpc_1s_n6_soc_clamping_diagnostic.py`: contract, metric, failure, isolation, and report tests.
- Create `outputs/mpc_1s_n6_soc_clamping_diagnostic/`: formal synthetic artifacts only after implementation is committed.
- Create `reports/mpc_1s_n6_soc_clamping_diagnostic.md`: reviewed diagnostic narrative.
- Create `reports/mpc_1s_n6_soc_clamping_metrics.csv`: formal case-level metrics.
- Modify `README.md`, `STATUS.md`, `docs/DATA_PROVENANCE.md`, `docs/PROJECT_MAP.md`, and `docs/UNFINISHED_TASKS.md`: record the evidence and its limits after formal execution.

### Task 1: Lock the synthetic profile and case contracts

**Files:**
- Create: `tests/test_mpc_1s_n6_soc_clamping_diagnostic.py`
- Create: `src/main/run_mpc_1s_n6_soc_clamping_diagnostic.py`

- [ ] **Step 1: Write failing contract tests**

Add tests that import the new module directly from `src/main` and require this public API:

```python
from run_mpc_1s_n6_soc_clamping_diagnostic import (
    build_case_matrix,
    build_constant_profile,
    build_pulse_profile,
    clamping_candidate_config,
)

def test_exact_case_matrix(self):
    cases = build_case_matrix()
    actual = {(c.profile_kind, c.q_soc, c.initial_soc) for c in cases}
    expected = {
        *(('constant', q, soc) for q in (10.0, 20.0) for soc in (0.53, 0.55, 0.57)),
        ('pulse', 10.0, 0.55),
        ('pulse', 20.0, 0.55),
    }
    self.assertEqual(actual, expected)
    self.assertEqual(len(cases), 8)

def test_profiles_have_exact_state_time_boundaries(self):
    times, constant = build_constant_profile()
    _, pulse = build_pulse_profile()
    self.assertEqual(len(times), 3601)
    self.assertTrue(np.array_equal(times, np.arange(3601, dtype=float)))
    self.assertTrue(np.all(constant == 300.0))
    self.assertEqual(pulse[599], 300.0)
    self.assertEqual(pulse[600], 450.0)
    self.assertEqual(pulse[719], 450.0)
    self.assertEqual(pulse[720], 300.0)
    self.assertEqual(int(np.sum(pulse == 450.0)), 120)

def test_candidate_configs_change_only_identity_and_q_soc(self):
    q10 = asdict(clamping_candidate_config(10.0))
    q20 = asdict(clamping_candidate_config(20.0))
    self.assertEqual(q10.pop('q_soc'), 10.0)
    self.assertEqual(q20.pop('q_soc'), 20.0)
    self.assertEqual(q10, q20)
    self.assertEqual(q10['q_h2'], 0.5)
    self.assertEqual(q10['q_batt'], 0.05)
    self.assertEqual(q10['soc_band'], 0.05)
    self.assertEqual(q10['q_terminal_soc'], 0.0)
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```powershell
python -m unittest tests.test_mpc_1s_n6_soc_clamping_diagnostic -v
```

Expected: import failure because `run_mpc_1s_n6_soc_clamping_diagnostic.py` does not exist.

- [ ] **Step 3: Implement the minimal contracts**

Create the module with frozen constants and this structure:

```python
@dataclass(frozen=True)
class SyntheticCase:
    case_id: str
    profile_kind: str
    candidate_id: str
    q_soc: float
    initial_soc: float

def build_constant_profile() -> tuple[np.ndarray, np.ndarray]:
    times = np.arange(3601, dtype=float)
    return times, np.full(times.shape, 300.0, dtype=float)

def build_pulse_profile() -> tuple[np.ndarray, np.ndarray]:
    times, loads = build_constant_profile()
    loads[(times >= 600.0) & (times < 720.0)] = 450.0
    return times, loads

def clamping_candidate_config(q_soc: float) -> QpMpcConfig:
    candidate_id = {10.0: 'QSOC_10', 20.0: 'QSOC_20'}[float(q_soc)]
    return qsoc_candidate_config(candidate_id)
```

`build_case_matrix()` returns six constant cases followed by two pulse cases with stable IDs such as `constant_soc053_qsoc10` and `pulse_soc055_qsoc20`.

- [ ] **Step 4: Run focused and existing N=6 tests**

Run:

```powershell
python -m unittest tests.test_mpc_1s_n6_soc_clamping_diagnostic tests.test_mpc_1s_n6_qsoc_feasibility tests.test_mpc_1s_n6_weight_selection -v
```

Expected: all tests pass; existing q_soc/A-D contracts remain unchanged.

- [ ] **Step 5: Commit the contract slice**

```powershell
git add -- src/main/run_mpc_1s_n6_soc_clamping_diagnostic.py tests/test_mpc_1s_n6_soc_clamping_diagnostic.py
git commit -m "test(mpc): specify SOC clamping diagnostic"
```

### Task 2: Implement timing windows, correction metrics, energy, and recovery

**Files:**
- Modify: `src/main/run_mpc_1s_n6_soc_clamping_diagnostic.py`
- Modify: `tests/test_mpc_1s_n6_soc_clamping_diagnostic.py`

- [ ] **Step 1: Add failing metric tests**

Require these functions:

```python
steady_state_mask(
    frame: pd.DataFrame,
    *,
    trailing_window_steps: int = 60,
    max_load_range_kw: float = 5.0,
    jump_threshold_kw: float = 10.0,
    post_jump_exclusion_s: float = 60.0,
    fc_saturation_kw: float = 560.0,
    start_exclusion_s: float = 60.0,
    end_exclusion_s: float = 60.0,
) -> pd.Series
annotate_correction_power(frame, soc_reference=0.55)
summarize_window(frame, mask, dt_seconds=1.0)
recovery_milestone(
    state_times_s: np.ndarray,
    soc_values: np.ndarray,
    *,
    initial_soc: float,
    reduction_fraction: float,
    soc_reference: float = 0.55,
) -> tuple[float | None, bool]
longest_true_run(mask)
```

Test exact sign behavior:

```python
frame = pd.DataFrame({
    'SOC_before': [0.53, 0.57, 0.55, 0.53],
    'P_batt_actual_kw': [-10.0, 12.0, 50.0, 8.0],
})
annotated = annotate_correction_power(frame)
np.testing.assert_allclose(
    annotated['P_correction_kw'],
    [10.0, 12.0, 0.0, -8.0],
)
self.assertEqual(
    annotated['active_near_reference_correction'].tolist(),
    [True, True, False, False],
)
```

Test one-hour unit conversion with simple 1 kW/1 s samples and verify `E_fc_surplus_kwh`, charge, discharge, and throughput divide by 3600 exactly. Test recovery on a trajectory that reaches 25%, later reaches 50%, and then leaves the threshold so first-reach and sustained flags differ. Test `None` for a milestone that is never reached.

Test the optional real-voyage steady mask with a constant region, a >10 kW jump, a 60 s post-jump exclusion, FC saturation, and configurable start/end exclusions. The test must prove that a slow ramp is rejected by the rolling 60 s range rather than accepted by a one-step slope rule.

- [ ] **Step 2: Run and verify RED**

Run the focused module and confirm failures name the missing metric APIs rather than syntax/import errors.

- [ ] **Step 3: Implement minimal metric helpers**

Use `SOC_before` for the correction sign and a `1e-12` comparison tolerance at the ±0.02 boundary:

```python
error = frame['SOC_before'].astype(float) - SOC_REFERENCE
correction = np.sign(error) * frame['P_batt_actual_kw'].astype(float)
positive = correction.clip(lower=0.0)
active = (correction > 5.0) & (error.abs() <= 0.02 + 1.0e-12)
```

`summarize_window()` reports full requested metrics plus active seconds, longest active run, corrective energy, and wrong-direction energy. It must return finite zero values for empty positive subsets while rejecting an empty analysis window.

Recovery uses a state vector containing `(t=0, initial_soc)` followed by every `(time_s, SOC_actual)` and finds the first absolute error at or below `0.015` and `0.010` for 25%/50%. The sustained flag checks every later state.

- [ ] **Step 4: Run focused tests and refactor names/units**

Expected: metric tests pass and every energy column name ends in `_kwh`, every hydrogen column ends in `_kg`, every power column ends in `_kw`, and every duration ends in `_s`.

- [ ] **Step 5: Commit the metric slice**

```powershell
git add -- src/main/run_mpc_1s_n6_soc_clamping_diagnostic.py tests/test_mpc_1s_n6_soc_clamping_diagnostic.py
git commit -m "feat(mpc): add SOC clamping metrics"
```

### Task 3: Execute cases safely and build auditable artifacts

**Files:**
- Modify: `src/main/run_mpc_1s_n6_soc_clamping_diagnostic.py`
- Modify: `tests/test_mpc_1s_n6_soc_clamping_diagnostic.py`

- [ ] **Step 1: Add failing orchestration and failure tests**

Patch only `run_voyage()` in unit tests and assert `run_synthetic_case()` passes the profile arrays, `initial_soc`, explicit config, and the synthetic case ID. Require it to reject:

- fewer or more than 3600 control rows;
- any failed control step;
- nonzero physical-infeasible count;
- a first row whose `prev_fc_actual_kw` is not 300;
- any q10/q20 pair that differs outside identity/q_soc.

Add path tests asserting all defaults are under `outputs/mpc_1s_n6_soc_clamping_diagnostic/` and the two requested `reports/mpc_1s_n6_soc_clamping_*` files. Assert none equals or is a child of the retained q_soc or N=60 roots.

- [ ] **Step 2: Run and verify RED**

Expected: failures show missing orchestration/provenance/artifact functions.

- [ ] **Step 3: Implement execution and provenance**

`run_synthetic_case()` calls:

```python
controls, solver = run_voyage(
    voyage_id=case.case_id,
    loads_kw=loads,
    times_s=times,
    candidate_id=case.candidate_id,
    initial_soc=case.initial_soc,
    config=clamping_candidate_config(case.q_soc),
)
```

It then calls `build_candidate_metrics()` and refuses to summarize incomplete/invalid results. Add `interval_start_s=decision_time_s`, `state_time_s=time_s`, profile/case fields, correction columns, and explicit scope masks.

`run_all_cases()` builds case, window, and paired-comparison tables; attaches same-q constant-0.55 excess columns; computes matched q20-minus-q10 deltas, pulse-minus-constant increments, and difference-in-differences; and returns selected trajectories for a single parquet file.

Run metadata records a UUID generation, Git HEAD, no-external-input declaration, frozen case matrix, both configs, runtime versions, and SHA256 over:

- the new diagnostic runner;
- the shared N=6 runner;
- q_soc config runner;
- OSQP benchmark helper;
- QP formulation;
- Dp0 curve implementation and CSV.

Safe reset may remove only the exact new output root after resolving it and checking its final directory name. It must refuse symlinks, files, or any resolved target outside the requested parent.

- [ ] **Step 4: Implement numeric artifact writing**

Write `run_metadata.json`, `case_metrics.csv`, `window_metrics.csv`, `comparison_metrics.csv`, and `synthetic_trajectories.parquet`. Invalidate only the new decision/report files at the beginning of a new formal run. Do not write or delete anything under old q_soc/N=60 roots.

- [ ] **Step 5: Run focused and all existing MPC tests**

```powershell
python -m unittest tests.test_mpc_1s_n6_soc_clamping_diagnostic tests.test_mpc_1s_n6_qsoc_feasibility tests.test_mpc_1s_n6_weight_selection tests.test_mpc_solver_benchmark_1s -v
```

Expected: all pass.

- [ ] **Step 6: Commit the execution slice before formal generation**

```powershell
git add -- src/main/run_mpc_1s_n6_soc_clamping_diagnostic.py tests/test_mpc_1s_n6_soc_clamping_diagnostic.py
git commit -m "feat(mpc): add synthetic SOC clamping runner"
```

This commit must precede the formal run so `run_metadata.json` records a committed implementation revision.

### Task 4: Add required plots and reviewed report rendering

**Files:**
- Modify: `src/main/run_mpc_1s_n6_soc_clamping_diagnostic.py`
- Modify: `tests/test_mpc_1s_n6_soc_clamping_diagnostic.py`

- [ ] **Step 1: Add failing plot/report tests**

Require exact plot filenames:

```text
constant_soc053_comparison.png
constant_soc055_comparison.png
constant_soc057_comparison.png
pulse_comparison.png
metric_comparison.png
```

Test that report rendering rejects an unknown label, `classification_status` other than `reviewed`, missing qualitative predicate reasons, mismatched provenance, incomplete case IDs, or missing formal metrics. Test that the rendered report includes `diagnostic synthetic profile`, `offline ideal foresight`, the pulse 60 s exclusion, the allowed label, `formal configuration modified: false`, and `DQN/LSTM started: false`.

Require `load_existing_voyage_context()` to read, without modifying, `outputs/mpc_1s_n6_qsoc_feasibility/candidate_QSOC_20/voyage_metrics.csv`; validate the `voyage_063` and `voyage_065` rows and expose their initial/minimum/final/net-change values for the report. Hash that retained CSV in report provenance so stale background statements are rejected.

- [ ] **Step 2: Run and verify RED**

Expected: failures are due to missing plot/report functions.

- [ ] **Step 3: Implement comparison figures**

The three constant figures compare q10/q20 FC, battery, and SOC for the same initial SOC. The pulse figure includes load, FC, battery, and SOC, with vertical markers at 600, 720, and 780 s. The metric figure compares 50% recovery time, positive correction power, FC surplus, throughput, and hydrogen excess; unreached recovery is annotated `not reached` rather than silently coerced to zero.

Every figure title or visible figure text includes `diagnostic synthetic profile`. Figures must not use “voyage”, “real ship”, or “measured”.

- [ ] **Step 4: Implement decision validation and report-only mode**

`diagnostic_decision.json` must contain:

```python
ALLOWED_LABELS = {
    "no_evidence_of_SOC_clamping",
    "moderate_SOC_clamping",
    "excessive_SOC_clamping",
}
REQUIRED_PREDICATES = {
    "sustained_near_reference_correction_both_sides",
    "long_term_fc_load_deviation_and_continuous_battery_transfer",
    "q20_cost_increase_clearly_material",
    "benefit_only_tighter_soc_not_feasibility",
    "q20_recovery_faster_on_both_sides",
    "pulse_recovery_physically_reasonable",
}
REQUIRED_BEHAVIOR_FLAGS = {
    "wrong_direction_drift",
    "asymmetric_response",
    "overshoot_then_drift",
}
```

The file sets `classification_status` to `reviewed`, stores one exact `ALLOWED_LABELS` value in `diagnostic_label`, stores exactly the required predicate keys with a boolean and nonempty numerical reason, stores exactly the required behavior flags as booleans, and sets `formal_configuration_modified` and `dqn_or_lstm_started` to false. `--report-only` validates the current implementation/runtime fingerprint and copies the case metrics to `reports/mpc_1s_n6_soc_clamping_metrics.csv` while writing the Markdown report.

- [ ] **Step 5: Run focused tests and visually inspect test figures**

Expected: focused tests pass; plot labels, axes, units, boundary markers, and q10/q20 legends are readable.

- [ ] **Step 6: Commit the visualization/report code**

```powershell
git add -- src/main/run_mpc_1s_n6_soc_clamping_diagnostic.py tests/test_mpc_1s_n6_soc_clamping_diagnostic.py
git commit -m "feat(mpc): report SOC clamping evidence"
```

### Task 5: Run the formal synthetic experiment and review the label

**Files:**
- Create: `outputs/mpc_1s_n6_soc_clamping_diagnostic/**`
- Create: `reports/mpc_1s_n6_soc_clamping_diagnostic.md`
- Create: `reports/mpc_1s_n6_soc_clamping_metrics.csv`

- [ ] **Step 1: Run exactly the eight synthetic cases**

```powershell
python src/main/run_mpc_1s_n6_soc_clamping_diagnostic.py --run
```

Expected: eight complete cases, 28,800 applied steps total, and only the new output root is created/reset. No full-voyage input parquet is read.

- [ ] **Step 2: Audit numerical evidence before choosing a label**

Verify all requested full/long-term metrics, q10/q20 matched deltas, same-q baseline excess, pulse 5 kWh load increment, recovery first-reach/sustained flags, solver/physical checks, and plots. Check correlated FC surplus/battery charge evidence is not double-counted.

Evaluate all excessive predicates conjunctively and all moderate predicates as specified in the design. If evidence is incomplete, fix the implementation and rerun rather than forcing a label.

- [ ] **Step 3: Add the reviewed decision JSON**

Use `apply_patch` to add `outputs/mpc_1s_n6_soc_clamping_diagnostic/diagnostic_decision.json` with the observed label, numerical reasons for every predicate, truthful behavior flags, and both formal-configuration/DQN-LSTM fields set to false. The label must be selected from evidence only after Step 2; it is intentionally not preregistered in this plan.

- [ ] **Step 4: Generate and validate the requested reports**

```powershell
python src/main/run_mpc_1s_n6_soc_clamping_diagnostic.py --report-only
```

Expected: the two requested report files are written and the report states all ten requested final items without promoting a formal weight.

- [ ] **Step 5: Visually inspect all five formal figures**

Use the image viewer to check trace alignment, legibility, synthetic annotation, pulse boundaries, and metric labels. Regenerate if any panel is clipped, misleading, or unreadable.

### Task 6: Update project status and perform independent review

**Files:**
- Modify: `README.md`
- Modify: `STATUS.md`
- Modify: `docs/DATA_PROVENANCE.md`
- Modify: `docs/PROJECT_MAP.md`
- Modify: `docs/UNFINISHED_TASKS.md`

- [ ] **Step 1: Update documentation from generated evidence**

Record the exact diagnosis label and key paired values, while stating:

- synthetic/offline ideal foresight, not a real voyage or LSTM result;
- no q17.5/full-voyage rerun;
- no formal configuration change;
- no DQN/LSTM execution;
- N60 and retained q_soc artifacts unchanged;
- no-evidence does not mean the controller is optimal if wrong-direction drift or another behavior flag exists.

- [ ] **Step 2: Request code and numerical review**

Dispatch an independent reviewer with the design, plan, base SHA, current SHA/diff, generated metrics, decision, and plot paths. Require checks for timing off-by-one, power sign, units, baseline/difference-in-differences math, label evidence, artifact isolation, and documentation overclaiming.

- [ ] **Step 3: Resolve all Critical and Important findings**

Add a failing regression test before each code fix. Rerun the formal experiment if a finding affects any numerical artifact; otherwise regenerate report-only artifacts after the fix.

### Task 7: Verify, commit, synchronize, and push

**Files:**
- Stage only files listed in this plan.

- [ ] **Step 1: Run fresh verification**

```powershell
python -m compileall src
python -m unittest tests.test_mpc_1s_n6_soc_clamping_diagnostic -v
python -m unittest discover -s tests -q
python src/main/run_mpc_1s_n6_soc_clamping_diagnostic.py --report-only
git diff --check
git status --short --branch
```

All commands must exit zero. Read the complete test count before editing `STATUS.md`.

- [ ] **Step 2: Audit protected paths and staged contents**

Confirm `git status --short` contains no retained q_soc, A-D, N60, LSTM, DQN, or 10 ms changes. Inspect staged file names, artifact size/count, and `git diff --cached --check`.

- [ ] **Step 3: Create the required experiment commit**

```powershell
git commit -m "experiment: diagnose near-reference SOC clamping"
```

- [ ] **Step 4: Rebase on the current remote and reverify if needed**

```powershell
git pull --rebase origin main
```

If rebase changes any source, dependencies, tests, or artifacts, rerun the full verification and regenerate affected formal outputs before push.

- [ ] **Step 5: Push and verify remote main**

```powershell
git push origin main
git ls-remote origin refs/heads/main
```

The remote hash must equal local `HEAD`; the final response reports the inner repository path, branch/origin, exact label/metrics, no-config/no-DQN status, verification count, commit hash, push result, remote update, and remaining scientific limitations.
