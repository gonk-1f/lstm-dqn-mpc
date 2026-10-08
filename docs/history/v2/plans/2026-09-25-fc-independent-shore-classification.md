# FC-Independent Shore Classification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Classify stationary battery charging as shore charging without using recorded FC power, while preserving the existing shore execution and accounting contracts.

**Architecture:** Change only the frozen supervisory classifier so the shore candidate depends on valid AIS/BMS evidence, near-zero speed, and battery charging. Keep the three-sample confirmation and the existing `FormalEpisodeBackend` shore interlock. Rebuild and authenticate only the mode sidecar; retain all raw telemetry and power/AIS dataset payloads unchanged.

**Tech Stack:** Python 3, pandas, NumPy, SciPy, `unittest`, immutable CSV/JSON manifests with SHA-256 authentication.

---

### Task 1: Freeze the FC-independent classification contract

**Files:**
- Modify: `tests/test_v2_supervisory_rules.py`
- Modify: `src/v2/data/supervisory_rules.py`
- Modify: `src/main/run_v2_objective_scale_audit.py`

- [ ] **Step 1: Write the failing classifier test**

Add a test with three quality-valid samples at zero speed, battery-bus power
`-150 kW`, frozen load `-50 kW`, and recorded FC powers `100`, `106`, and
`107 kW`. Require `(shore_pending, shore_pending, shore_charging)`.

```python
def test_nonzero_recorded_fc_does_not_block_confirmed_shore_charging(self):
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    samples = tuple(
        ModeSample(
            start + timedelta(seconds=30 * index),
            0.0,
            fc_kw,
            -150.0,
            p_load_kw=-50.0,
        )
        for index, fc_kw in enumerate((100.0, 106.0, 107.0))
    )
    self.assertEqual(
        classify_operating_modes(samples),
        (
            OperatingMode.SHORE_PENDING,
            OperatingMode.SHORE_PENDING,
            OperatingMode.SHORE_CHARGING,
        ),
    )
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```powershell
python -X utf8 -B -m unittest tests.test_v2_supervisory_rules.SupervisoryRuleTests.test_nonzero_recorded_fc_does_not_block_confirmed_shore_charging
```

Expected: failure because all three samples are currently `unresolved`.

- [ ] **Step 3: Implement the minimal classifier change**

Remove `abs(sample.p_fc_total_kw) <= FC_ZERO_TOLERANCE_KW` from
`_shore_candidate`. Remove the obsolete `FC_ZERO_TOLERANCE_KW` public
constant and its assertion. In the objective-audit provenance rules replace
the numeric FC tolerance with an explicit string stating that recorded FC
telemetry is ignored for shore classification.

- [ ] **Step 4: Run classifier and formal episode tests**

Run:

```powershell
python -X utf8 -B -m unittest tests.test_v2_supervisory_rules tests.test_v2_formal_episode
```

Expected: all tests pass. The existing formal-episode test must continue to
prove zero executed FC power, no MPC calls during shore, positive battery
degradation cost, and positive shore cost.

### Task 2: Rebuild and authenticate the mode sidecar

**Files:**
- Regenerate: `data/processed/operating_dataset_zero_boundary_v2_modes/**`
- Modify: `tests/test_v2_formal_training_dataset.py`
- Modify: `tests/test_v2_formal_preflight.py`
- Modify: `tests/test_v2_formal_training_cli.py`

- [ ] **Step 1: Add the real-dataset regression expectation**

Change the frozen expectations from 110 Train unresolved rows to zero, while
leaving the 30/8/5 split counts, power/AIS identities, and Test payload policy
unchanged.

- [ ] **Step 2: Run the dataset-focused tests and verify RED**

Run:

```powershell
python -X utf8 -B -m unittest tests.test_v2_formal_training_dataset tests.test_v2_formal_preflight tests.test_v2_formal_training_cli
```

Expected: failure because the existing mode sidecar still contains 110 Train
unresolved rows.

- [ ] **Step 3: Rebuild the sidecar into a temporary destination**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
python -X utf8 -B -m v2.main.build_shore_mode_sidecar `
  --dataset-root data/processed/operating_dataset_zero_boundary_v2 `
  --ais-root data/processed/operating_dataset_zero_boundary_v2_ais `
  --raw-root 'C:\Users\20883\OneDrive\Desktop\氢舟一号' `
  --output-root data/processed/operating_dataset_zero_boundary_v2_modes.building-fc-independent
```

Verify that all 110 previously unresolved `zero_boundary_029` rows join the
already confirmed contiguous event as `shore_charging`, and that the complete
segment contains two `shore_pending`, 121 `shore_charging`, and zero
`unresolved` rows. Verify exact manifest identity against power and AIS, then
replace only the existing mode-sidecar directory.

- [ ] **Step 4: Run the dataset-focused tests and verify GREEN**

Run the Step 2 command again. Expected: all tests pass and formal preflight no
longer reports unresolved operating-mode evidence.

### Task 3: Full regression and formal smoke verification

**Files:**
- No additional production files.

- [ ] **Step 1: Run focused v2 tests**

```powershell
python -X utf8 -B -m unittest tests.test_v2_supervisory_rules tests.test_v2_shore_mode_sidecar tests.test_v2_formal_training_dataset tests.test_v2_formal_episode tests.test_v2_formal_preflight tests.test_v2_formal_training_cli
```

- [ ] **Step 2: Run all v2 and complete tests**

```powershell
python -X utf8 -B -m unittest discover -s tests -p 'test_v2_*.py'
python -X utf8 -B -m unittest discover -s tests
```

- [ ] **Step 3: Run solver smoke**

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path
python -X utf8 -B -m v2.main.train_formal_dqn --smoke-only --output-dir outputs/v2_formal_training_smoke_fc_independent
```

Expected: `SMOKE=PASS`, zero Train/Validation unresolved steps, and no Test
payloads opened.

- [ ] **Step 4: Run compile/import and diff checks**

```powershell
python -X utf8 -B -m compileall -q src tests
$env:PYTHONPATH=(Resolve-Path 'src').Path
python -X utf8 -B -c "import v2.data.supervisory_rules; import v2.data.shore_mode_sidecar; import v2.envs.formal_episode"
git diff --check
```

Expected: every command exits zero. Do not start formal training.
