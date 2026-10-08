# V2 DQN State Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a reproducible Train-only numerical and Markov audit that recommends and documents the formal v2 DQN state without changing training, the action catalog, or the candidate state implementation.

**Architecture:** A pure analysis module validates the active Train manifest, constructs causal 30-second measured state rows, and derives statistics, correlations, regimes, and state comparisons. A thin runner reads only whitelisted Train segment paths and matching raw parents, writes deterministic CSV/PNG/JSON outputs, and renders the final Markdown report. The implementation reuses existing v2 raw-channel parsing and causal supervisory alignment rather than simulating an MPC policy.

**Tech Stack:** Python 3.11, dataclasses, pathlib, hashlib/json/csv, NumPy, pandas, SciPy statistics, Matplotlib, pytest/unittest-compatible tests.

---

### Task 1: Freeze Train-only input boundary

**Files:**
- Create: `tests/test_v2_train_state_audit.py`
- Create: `src/v2/analysis/train_state_audit.py`

- [ ] **Step 1: Write failing tests for active-dataset and held-out guards**

Add tests that construct a manifest with Train, Validation, and Test rows and assert that `load_train_manifest()` returns only Train rows, resolves every source below `train/`, records the active dataset version, and never invokes the injected segment reader for held-out rows. Add rejection tests for a Train row whose relative path escapes the dataset root or points below `validation/`.

```python
selected = load_train_manifest(dataset_root)
assert selected["sample_id"].tolist() == ["train_001"]
assert selected.iloc[0]["relative_path"] == "train/train_001.csv"
with pytest.raises(ValueError, match="Train path"):
    load_train_manifest(root_with_validation_path_labeled_train)
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
python -m pytest tests/test_v2_train_state_audit.py -q
```

Expected: collection/import failure because `v2.analysis.train_state_audit` does not exist.

- [ ] **Step 3: Implement immutable audit constants and Train loader**

Define:

```python
AUDIT_SAMPLE_SECONDS = 30.0
AUDIT_HISTORY_SECONDS = 150.0
AUDIT_TAU_LPF_SECONDS = 90.0
AUDIT_POWER_SCALE_KW = 600.0

@dataclass(frozen=True)
class TrainSegment:
    parent: str
    sample_id: str
    relative_path: str
    start_timestamp: datetime
    end_timestamp: datetime
    sha256: str
```

Implement `load_train_segments(dataset_root) -> tuple[TrainSegment, ...]`. It reads only `metadata/sample_manifest.csv`, filters `split == "train"`, validates the active dataset version from `metadata/qa_summary.json`, requires paths under `train/`, rejects path traversal, verifies file hashes, and returns chronological immutable records. It must not open Validation/Test segment CSVs.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the Task 1 tests and expect all to pass.

- [ ] **Step 5: Commit Task 1**

```powershell
git add tests/test_v2_train_state_audit.py src/v2/analysis/train_state_audit.py
git commit -m "feat(v2): guard Train state audit inputs"
```

### Task 2: Construct causal measured audit rows

**Files:**
- Modify: `tests/test_v2_train_state_audit.py`
- Modify: `src/v2/analysis/train_state_audit.py`

- [ ] **Step 1: Write failing tests for causal feature construction**

Use synthetic `ParentSupervisoryState` rows and one-second Train load fixtures. Verify:

- only eligible states inside the segment boundary are retained;
- no history crosses a segment boundary;
- at least six samples spanning `[t-150, t]` are required;
- LPF uses `alpha = exp(-30/90)` and current/history values only;
- `delta_load_kw = load_kw - base_load_kw`;
- `delta_fc_kw = fc_kw - previous_fc_kw`;
- measured battery bus power uses discharge-positive sign;
- `recent_delta_soc = soc(k) - soc(k-150 s)`;
- future load changes do not alter an earlier audit row.

```python
row = build_causal_feature_rows(segment, states)[-1]
assert row.delta_load_kw == pytest.approx(row.load_kw - row.base_load_kw)
assert row.delta_fc_kw == pytest.approx(row.fc_kw - row.previous_fc_kw)
assert row.recent_delta_soc == pytest.approx(row.soc - states[-6].soc_system)
```

- [ ] **Step 2: Run the new tests and verify RED**

Expected: failure because `build_causal_feature_rows` and `AuditFeatureRow` are missing.

- [ ] **Step 3: Implement causal row construction**

Add `AuditFeatureRow` with physical-unit fields for all S10 and S7 inputs plus measured balance residual. Implement a segment-local 30-second history, first-order LPF, population moments, least-squares trend, and explicit ineligibility counters. Use real timestamps for the trend and require monotonically increasing states. Do not impute FC, battery, or SOC.

Add `build_parent_train_rows(train_segments, raw_parent_loader)` that calls the existing raw parser and `build_parent_supervisory_states()` only for parents present in the Train whitelist, then intersects eligible states with each formal segment boundary and looks up the formal one-second load at the same timestamp.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the entire focused test file and expect all Task 1 and Task 2 tests to pass.

- [ ] **Step 5: Commit Task 2**

```powershell
git add tests/test_v2_train_state_audit.py src/v2/analysis/train_state_audit.py
git commit -m "feat(v2): build causal state audit rows"
```

### Task 3: Compute statistics, redundancy, regimes, and state comparisons

**Files:**
- Modify: `tests/test_v2_train_state_audit.py`
- Modify: `src/v2/analysis/train_state_audit.py`

- [ ] **Step 1: Write failing numerical-analysis tests**

Create a deterministic synthetic audit frame and verify:

- descriptive columns are exactly `count`, `missing_count`, `min`, `max`,
  `mean`, `std`, `p01`, `p05`, `p50`, `p95`, `p99`;
- population standard deviation uses `ddof=0`;
- Pearson and Spearman matrices are symmetric with unit diagonals;
- a deterministic battery identity is flagged separately from nonzero measured
  residuals;
- normalized trend equals `trend_kw_per_s * 150 / 600`;
- S10, S7, S6-A, S6-B, and the minimum-state row have exact feature lists;
- regime summaries never use held-out data and preserve sample counts.

- [ ] **Step 2: Run numerical tests and verify RED**

Expected: failures for missing analysis functions.

- [ ] **Step 3: Implement deterministic analysis functions**

Implement:

```python
def feature_frame(rows: Sequence[AuditFeatureRow]) -> pd.DataFrame: ...
def normalize_feature_frame(frame: pd.DataFrame) -> pd.DataFrame: ...
def descriptive_statistics(frame: pd.DataFrame) -> pd.DataFrame: ...
def correlation_matrices(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]: ...
def redundancy_summary(frame: pd.DataFrame) -> pd.DataFrame: ...
def regime_summary(frame: pd.DataFrame) -> pd.DataFrame: ...
def state_comparison() -> pd.DataFrame: ...
```

Regime labels must use fixed physical controller thresholds where v2 defines
them. Trend and volatility descriptions may use recorded Train-only thresholds
for descriptive evidence, but the output must mark them `DESCRIPTIVE_ONLY` so
they cannot be mistaken for production policy thresholds.

- [ ] **Step 4: Run numerical tests and verify GREEN**

Run the focused file and expect all tests to pass.

- [ ] **Step 5: Commit Task 3**

```powershell
git add tests/test_v2_train_state_audit.py src/v2/analysis/train_state_audit.py
git commit -m "feat(v2): analyze candidate DQN states"
```

### Task 4: Audit hidden state and write reproducible artifacts

**Files:**
- Modify: `tests/test_v2_train_state_audit.py`
- Modify: `src/v2/analysis/train_state_audit.py`
- Create: `src/v2/main/run_train_state_audit.py`

- [ ] **Step 1: Write failing artifact and Markov-inventory tests**

Verify the inventory contains LPF state, previous executed FC power, cumulative
FC voltage loss, cumulative battery weighted Ah, previous DQN action, warm
start, terminal recharge accounting, and macro-step position. Verify every item
has a classification and evidence path. Test the runner against fixture data and
assert it writes the required CSV/JSON/PNG/Markdown files without opening held-out
segment paths.

- [ ] **Step 2: Run artifact tests and verify RED**

Expected: failures because the inventory and runner are absent.

- [ ] **Step 3: Implement the Markov inventory and runner**

Implement a static, code-referenced inventory whose conclusions are rendered
into the report. The runner accepts explicit `--dataset-root`, `--raw-root`, and
`--output-root`, with defaults pointing to the active dataset, desktop raw root,
and `outputs/v2_dqn_state_audit`. It writes CSV with stable column ordering and
UTF-8, writes a manifest with SHA-256 hashes and sample accounting, and creates
compact heatmaps/distribution/regime plots using a noninteractive Matplotlib
backend.

The report writer must include the ten-feature KEEP/REMOVE/REPLACE/CONDITIONAL
table, exact recommended schema and normalization, redundancy findings, Markov
audit, Train-only numeric evidence, full-versus-minimum state, limitations, and
an explicit S7 conclusion. It must not change `src/v2/dqn/state.py`.

- [ ] **Step 4: Run artifact tests and verify GREEN**

Run the focused file and inspect fixture artifact names and manifest hashes.

- [ ] **Step 5: Commit Task 4**

```powershell
git add tests/test_v2_train_state_audit.py src/v2/analysis/train_state_audit.py src/v2/main/run_train_state_audit.py
git commit -m "feat(v2): render Train state audit"
```

### Task 5: Execute the real Train audit and verify the branch

**Files:**
- Create through runner: `outputs/v2_dqn_state_audit/`
- Create through runner: `docs/v2_dqn_state_audit.md`

- [ ] **Step 1: Run the active Train audit**

```powershell
python -m v2.main.run_train_state_audit `
  --dataset-root data/processed/operating_dataset_zero_boundary_v2 `
  --raw-root 'C:\Users\20883\OneDrive\Desktop\氢舟一号' `
  --output-root outputs/v2_dqn_state_audit `
  --report-path docs/v2_dqn_state_audit.md
```

Expected: exactly 38 Train parents selected, zero Validation/Test segment files
opened, nonzero eligible audit rows, and all requested artifacts written.

- [ ] **Step 2: Inspect generated evidence**

Independently verify manifest hashes, row counts, feature order, correlation
symmetry, finite statistics, plot existence, and consistency between CSV values
and the Markdown conclusions. Open each PNG and check labels, units, and clipping.

- [ ] **Step 3: Run focused and full verification**

```powershell
python -m pytest tests/test_v2_train_state_audit.py -q
python -m pytest tests/test_v2_*.py -q
python -m compileall -q src/v2
python -c "import v2; import v2.analysis.train_state_audit; import v2.main.run_train_state_audit"
git diff --check
```

Also run the existing v2 solver smoke command identified by the current test
suite. Expected: all commands exit zero.

- [ ] **Step 4: Confirm forbidden-scope preservation**

Verify no diff touches `src/v2/dqn/state.py`, `src/v2/dqn/action_space.py`, formal
dataset files, split manifests, MPC objectives, economics, or training code.

- [ ] **Step 5: Commit generated evidence**

```powershell
git add docs/v2_dqn_state_audit.md outputs/v2_dqn_state_audit
git commit -m "docs(v2): record Train state audit"
```
