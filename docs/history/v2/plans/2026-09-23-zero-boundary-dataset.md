# Zero-Boundary Operating Dataset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Audit 66 raw parents, build and freeze the 53 parents with bracketed zero boundaries as a one-second operating dataset with deterministic 38/10/5 Train/Validation/Test allocation, and make it the formal loader default only after all artifact checks pass.

**Architecture:** Put reusable power-series assembly, zero-boundary trimming, and deterministic parent stratification in focused `src/v2/data` modules. A new one-shot builder audits all 66 parents, admits exactly 53 bracketed parents, records the 13 approved incomplete-boundary exclusions, writes the versioned dataset and complete provenance metadata without overwriting, then the formal loader changes its default root and accepts the explicitly retained internal negative-power samples. Tests establish each contract before implementation, and the actual raw build happens only after synthetic focused tests are green.

**Tech Stack:** Python 3, `unittest`, NumPy, pandas, SciPy (`CubicSpline`, `PchipInterpolator`), existing audited FC/BMS/AIS readers, SHA-256, JSON/CSV metadata, PowerShell, Git.

---

## Scope and file map

- Create `src/v2/data/segment_power_source.py`: tracked production implementation of the accepted 8-FC/12-BMS near-synchronous power assembly used by the review plots.
- Create `src/v2/data/zero_boundary_dataset.py`: pure boundary detection, component-preserving zero construction, one-second reconstruction, feature extraction, and deterministic parent stratification.
- Create `src/main/build_zero_boundary_operating_dataset.py`: one-shot non-overwriting builder and artifact/provenance writer.
- Create `tests/test_v2_segment_power_source.py`: focused channel-alignment and power-sign tests.
- Create `tests/test_v2_zero_boundary_dataset.py`: focused boundary, PCHIP, split, and builder tests.
- Modify `src/utils/formal_operating_dataset.py`: switch the default root after artifact validation, read frozen expected totals from the new metadata, and permit finite internal negative load.
- Modify `tests/test_formal_operating_dataset.py`: assert the new 53/38/10/5 contract without opening held-out values in ordinary manifest tests.
- Create `data/processed/operating_dataset_zero_boundary_v2/{train,validation,test,metadata}` through the builder only.
- Preserve `data/processed/operating_dataset_final`, raw telemetry, and `outputs/v2_segment_power_review/gap_diagnostics.csv` byte-for-byte.

### Frozen constants

```python
ZERO_DEADBAND_KW = 1.0
ACTIVE_THRESHOLD_KW = 1.0
SUSTAINED_POINTS = 3
ALIGNMENT_TOLERANCE_SECONDS = 10.0
NOMINAL_STEP_SECONDS = 30.0
TRAIN_COUNT = 38
VALIDATION_COUNT = 10
TEST_COUNT = 5
FIXED_TEST_PARENTS = (
    "3月26日14_00_3月26日16_00",
    "3月29日08_00_3月29日15_00",
    "4月18日12_00_4月18日18_00",
    "5月8日08_00_5月8日17_00",
    "6月11日08_00_6月11日11_00",
)
```

### Required output schemas

`metadata/sample_manifest.csv` columns:

```text
parent,sample_id,relative_path,split,point_count_1s,start_timestamp,end_timestamp,duration_s,sha256
```

`metadata/parent_split_manifest.csv` columns:

```text
parent,split,chronological_rank,month,duration_s,mean_load_kw,p95_load_kw,duration_quartile,mean_load_quartile,p95_load_quartile
```

`metadata/trim_boundary_audit.csv` columns:

```text
parent,split,side,boundary_kind,left_timestamp,left_load_kw,right_timestamp,right_load_kw,original_boundary_load_kw,canonical_boundary_load_kw,crossing_fraction,unrounded_boundary_timestamp,rounded_boundary_timestamp,rounding_adjustment_seconds,removed_point_count,removed_duration_s,sustained_block_start_timestamp,sustained_block_end_timestamp
```

`metadata/interpolation_audit.csv` columns:

```text
parent,split,aligned_measured_points,cubic_gap_count,cubic_imputed_points,cubic_max_gap_seconds,cubic_warning_count,cubic_warning_messages,trimmed_anchor_points,pchip_output_points,pchip_floating_negative_zeroed
```

## Task 1: Establish zero-boundary trimming and one-second reconstruction

**Files:**
- Create: `tests/test_v2_zero_boundary_dataset.py`
- Create: `src/v2/data/zero_boundary_dataset.py`

- [ ] **Step 1: Write the focused red tests**

Create a `unittest.TestCase` with a helper that returns timezone-aware anchors containing `timestamp`, `fc_total_kw`, `battery_raw_total_kw`, `source_total_kw`, and `is_cubic_imputed`. Add these exact behavioral tests:

```python
def test_trims_leading_and_trailing_dwell_to_observed_zero_boundaries(self):
    frame = self.frame([0, 0, 0, 20, 30, 40, 35, 25, 10, 0, 0, -15])
    result = trim_to_zero_boundaries(frame)
    self.assertEqual(result.frame.source_total_kw.tolist(), [0, 20, 30, 40, 35, 25, 10, 0])
    self.assertEqual(result.start.kind, "OBSERVED_DEADBAND")
    self.assertEqual(result.end.kind, "OBSERVED_DEADBAND")
    self.assertEqual(result.frame.iloc[0].source_total_kw, 0.0)
    self.assertEqual(result.frame.iloc[-1].source_total_kw, 0.0)

def test_constructs_positive_to_negative_terminal_crossing_with_component_identity(self):
    frame = self.frame([0, 10, 20, 30, 25, 20, 10, -10, -20])
    result = trim_to_zero_boundaries(frame)
    self.assertEqual(result.end.kind, "CONSTRUCTED_CROSSING")
    self.assertAlmostEqual(result.end.crossing_fraction, 0.5)
    last = result.frame.iloc[-1]
    self.assertAlmostEqual(last.source_total_kw, 0.0, places=12)
    self.assertAlmostEqual(last.source_total_kw, last.fc_total_kw - last.battery_raw_total_kw, places=12)

def test_preserves_internal_stop_and_negative_interval(self):
    frame = self.frame([0, 15, 20, 25, 0, -5, 0, 18, 22, 26, 0, -20])
    result = trim_to_zero_boundaries(frame)
    self.assertIn(-5.0, result.frame.source_total_kw.tolist())
    self.assertEqual(result.frame.source_total_kw.tolist().count(0.0), 4)

def test_rejects_missing_start_or_end_zero_bracket(self):
    with self.assertRaisesRegex(ValueError, "start boundary"):
        trim_to_zero_boundaries(self.frame([5, 10, 20, 30, 25, 10, 0]))
    with self.assertRaisesRegex(ValueError, "end boundary"):
        trim_to_zero_boundaries(self.frame([0, 10, 20, 30, 25, 10, 5]))

def test_reconstructs_exact_one_second_axis_and_zero_endpoints(self):
    trimmed = trim_to_zero_boundaries(self.frame([0, 10, 20, 30, 20, 10, 0]))
    output, qa = reconstruct_one_second(trimmed.frame)
    np.testing.assert_array_equal(output.time_s, np.arange(len(output), dtype=float))
    self.assertEqual(output.iloc[0].load_total_kw, 0.0)
    self.assertEqual(output.iloc[-1].load_total_kw, 0.0)
    self.assertEqual(qa["output_points"], len(output))
```

The helper must set `fc_total_kw = source + 100` and `battery_raw_total_kw = 100`, so component identity is exercised rather than mocked away.

- [ ] **Step 2: Run the focused file and confirm the red state**

Run:

```powershell
python -X utf8 -B tests/test_v2_zero_boundary_dataset.py
```

Expected: import failure for `v2.data.zero_boundary_dataset`; no production code has been written.

- [ ] **Step 3: Implement the minimal boundary module**

Implement immutable `BoundaryAudit` and `TrimmedParent` dataclasses plus these functions:

```python
def _sustained_starts(values: np.ndarray, threshold_kw: float, count: int) -> np.ndarray:
    positive = values > threshold_kw
    return np.flatnonzero(np.convolve(positive.astype(int), np.ones(count, dtype=int), mode="valid") == count)

def _crossing_row(left: pd.Series, right: pd.Series, *, side: str) -> tuple[pd.Series, BoundaryAudit]:
    y0 = float(left.source_total_kw)
    y1 = float(right.source_total_kw)
    if y0 == y1 or y0 * y1 > 0.0:
        raise ValueError(f"{side} boundary is not bracketed")
    fraction = -y0 / (y1 - y0)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"{side} boundary requires extrapolation")
    raw_time = pd.Timestamp(left.timestamp) + fraction * (pd.Timestamp(right.timestamp) - pd.Timestamp(left.timestamp))
    row = left.copy()
    row["timestamp"] = raw_time.round("s")
    row["fc_total_kw"] = float(left.fc_total_kw) + fraction * (float(right.fc_total_kw) - float(left.fc_total_kw))
    row["battery_raw_total_kw"] = float(left.battery_raw_total_kw) + fraction * (float(right.battery_raw_total_kw) - float(left.battery_raw_total_kw))
    row["source_total_kw"] = float(row.fc_total_kw - row.battery_raw_total_kw)
    row["is_cubic_imputed"] = False
    if abs(float(row.source_total_kw)) > 1.0e-9:
        raise AssertionError("constructed boundary violates power identity")
    row["source_total_kw"] = 0.0
    return row, BoundaryAudit.constructed(side, left, right, fraction, raw_time, row.timestamp)
```

`trim_to_zero_boundaries` must validate sorted unique timestamps and finite components, locate the first and last three-point sustained-positive windows, choose the last preceding deadband point and first following deadband point, fall back only to bracketed interpolation, canonicalize observed deadband loads to zero, retain every row between the two boundaries, and report removed prefix/suffix counts and durations. It must never extrapolate.

Implement `reconstruct_one_second` by calling the existing `utils.rebuilt_operating_dataset.pchip_to_one_second` on `timestamp` and renamed `load_total_kw`, then force only the already-constrained first and last values to exact `0.0`. Reject a nonzero PCHIP endpoint before forcing, and reject non-finite output.

- [ ] **Step 4: Run focused tests and commit**

Run:

```powershell
python -X utf8 -B tests/test_v2_zero_boundary_dataset.py
git diff --check
```

Expected: all Task 1 tests pass and `git diff --check` prints nothing.

Commit:

```powershell
git add tests/test_v2_zero_boundary_dataset.py src/v2/data/zero_boundary_dataset.py
git commit -m "feat(v2): add zero-boundary trimming"
```

## Task 2: Freeze deterministic parent-level stratification

**Files:**
- Modify: `tests/test_v2_zero_boundary_dataset.py`
- Modify: `src/v2/data/zero_boundary_dataset.py`

- [ ] **Step 1: Add red split tests**

Add tests that build 53 included synthetic feature rows, substitute the five exact fixed Test names, and assert:

```python
assignment = assign_parent_splits(features)
self.assertEqual(assignment.split.value_counts().to_dict(), {"train": 38, "validation": 10, "test": 5})
self.assertEqual(set(assignment.loc[assignment.split.eq("test"), "parent"]), set(FIXED_TEST_PARENTS))
self.assertEqual(assignment.groupby("parent").split.nunique().max(), 1)
pd.testing.assert_frame_equal(assignment, assign_parent_splits(features.sample(frac=1.0, random_state=7)))
```

Also assert that `duration_quartile`, `mean_load_quartile`, and `p95_load_quartile` for non-Test rows are integers 0 through 3 and that changing Test-set feature values does not change any Train/Validation assignment.

- [ ] **Step 2: Verify those tests fail for missing split functions**

Run:

```powershell
python -X utf8 -B tests/test_v2_zero_boundary_dataset.py
```

Expected: failure because `assign_parent_splits`, `segment_features`, and `FIXED_TEST_PARENTS` are absent.

- [ ] **Step 3: Implement feature extraction and split assignment**

Implement:

```python
def segment_features(parent: str, frame: pd.DataFrame) -> dict[str, object]:
    load = frame["load_total_kw"].to_numpy(dtype=float)
    timestamps = pd.to_datetime(frame["timestamp"])
    return {
        "parent": parent,
        "chronological_timestamp": timestamps.iloc[0],
        "month": int(timestamps.iloc[0].month),
        "duration_s": float(frame["time_s"].iloc[-1]),
        "mean_load_kw": float(np.mean(load)),
        "p95_load_kw": float(np.percentile(load, 95)),
    }
```

For each continuous feature column, compute non-Test quartiles with `pd.qcut(non_test[column], q=4, labels=False, duplicates="drop")`; fail unless the resulting labels are exactly `{0, 1, 2, 3}`. Represent each candidate with four labels: `month=<m>`, `duration_quartile=<q>`, `mean_load_quartile=<q>`, and `p95_load_quartile=<q>`. Starting with an empty Validation set, try every remaining candidate and choose the one minimizing the sum over labels of `abs(selected_count - 0.2 * available_count)`. Recompute the score after each addition, break ties by chronological timestamp then parent identifier, select exactly 10, assign the other 38 to Train, and append the fixed five Test parents. Reject duplicate/missing parents and any fixed Test mismatch.

- [ ] **Step 4: Run, inspect, and commit**

Run:

```powershell
python -X utf8 -B tests/test_v2_zero_boundary_dataset.py
git diff --check
```

Expected: all boundary and split tests pass.

Commit:

```powershell
git add tests/test_v2_zero_boundary_dataset.py src/v2/data/zero_boundary_dataset.py
git commit -m "feat(v2): freeze parent stratification"
```

## Task 3: Move the reviewed power assembly into tracked production code

**Files:**
- Create: `tests/test_v2_segment_power_source.py`
- Create: `src/v2/data/segment_power_source.py`
- Modify: `.codex_tmp/segment_power_review.py`

- [ ] **Step 1: Write red tests for alignment and sign semantics**

Copy the synthetic alignment cases from the review script into tracked tests and add:

```python
def test_power_conventions_use_raw_battery_sign(self):
    battery_raw, source = power_conventions([100.0, 120.0], [20.0, -10.0])
    np.testing.assert_allclose(battery_raw, [-20.0, 10.0])
    np.testing.assert_allclose(source, [120.0, 110.0])
    np.testing.assert_allclose(source, np.asarray([100.0, 120.0]) - battery_raw)

def test_alignment_uses_each_channel_sample_at_most_once(self):
    result = align_near_synchronous_cycles(
        self.times(0, 30, 60),
        (self.times(1, 31, 61), self.times(2, 32, 62)),
        tolerance_seconds=10.0,
        max_channel_span_seconds=10.0,
    )
    self.assertEqual(result.valid_reference_positions, (0, 1, 2))
    self.assertEqual(result.channel_indices, ((0, 1, 2), (0, 1, 2)))
```

Add a synthetic `ParentRawChannels` case with 8 FC and 12 battery channels and assert `assemble_parent_power_series` sums every channel, produces strictly increasing snapshot timestamps, exposes AIS availability separately, and never requires AIS to invent load.

- [ ] **Step 2: Confirm the new module is absent**

Run:

```powershell
python -X utf8 -B tests/test_v2_segment_power_source.py
```

Expected: import failure for `v2.data.segment_power_source`.

- [ ] **Step 3: Extract the reviewed implementation without semantic changes**

Move `LoadedParent`, `CycleAlignment`, `align_near_synchronous_cycles`, `_load_parent`, and `power_conventions` from `.codex_tmp/segment_power_review.py` into the tracked module. Add:

```python
@dataclass(frozen=True)
class ParentPowerSeries:
    parent: str
    timestamps: tuple[datetime, ...]
    fc_total_kw: np.ndarray
    battery_raw_total_kw: np.ndarray
    source_total_kw: np.ndarray
    ais_present: np.ndarray
    duplicate_count: int
    duplicate_conflict_count: int
    channel_span_violation_count: int
```

`assemble_parent_power_series` must resolve duplicates for all 21 channels, use FC channel 1 as the reference clock, apply one-to-one ±10-second matching and a 10-second maximum cross-channel span to the 20 power channels, sum 8 FC and 12 battery bus-sign channels with `math.fsum`, convert battery to raw sign, derive source as `FC - battery_raw`, and align AIS only as an availability diagnostic. Keep all power values unchanged.

Add `load_parent_power_series(raw_root, parent)` as the production wrapper that calls `_load_parent`, then `assemble_parent_power_series`, and returns the immutable `ParentPowerSeries`. It must raise rather than return an empty or partial series when no complete 20-power-channel timestamps exist.

Change the ignored review script to import these tracked definitions so the plot and dataset paths cannot drift.

- [ ] **Step 4: Run both source and interpolation tests, then commit**

Run:

```powershell
python -X utf8 -B tests/test_v2_segment_power_source.py
python -X utf8 -B tests/test_v2_power_gap_interpolation.py
git diff --check
```

Expected: both focused files pass; the ignored script remains behaviorally equivalent.

Commit tracked files only:

```powershell
git add tests/test_v2_segment_power_source.py src/v2/data/segment_power_source.py
git commit -m "refactor(v2): track parent power assembly"
```

## Task 4: Build a non-overwriting artifact writer with complete metadata

**Files:**
- Modify: `tests/test_v2_zero_boundary_dataset.py`
- Create: `src/main/build_zero_boundary_operating_dataset.py`

- [ ] **Step 1: Add builder red tests using two small synthetic providers**

Patch the builder's discovery and parent-series provider so no real telemetry is opened. Assert that an existing destination raises `FileExistsError`; a complete synthetic run writes one CSV per parent, starts and ends each CSV at zero, uses exact one-second `time_s`, and creates every required metadata file. Assert `sample_manifest.csv` paths exist, hashes match, and the five fixed Test parents are the only Test rows.

Use a 66-parent raw fixture because source accounting is contractual. Give 53 parents a leading zero, three sustained positive points, an internal stop, a second sustained block, and a trailing zero/negative tail. Give the exact 13 approved exclusions their recorded missing-start or missing-end pattern and assert `excluded_parent_manifest.csv` contains only those parents.

- [ ] **Step 2: Confirm builder tests fail before the script exists**

Run:

```powershell
python -X utf8 -B tests/test_v2_zero_boundary_dataset.py
```

Expected: import failure for `main.build_zero_boundary_operating_dataset`.

- [ ] **Step 3: Implement the builder orchestration**

Expose a testable function and thin CLI:

```python
def build_dataset(
    raw_root: Path,
    output_root: Path,
    *,
    discover_parents: Callable[[Path], list[str]] = discover_parent_ids,
    load_parent: Callable[[Path, str], ParentPowerSeries] = load_parent_power_series,
) -> dict[str, object]:
    if output_root.exists():
        raise FileExistsError(f"destination already exists: {output_root}")
    parents = sorted(discover_parents(raw_root), key=parent_sort_key)
    if len(parents) != 66 or len(set(parents)) != 66:
        raise ValueError("expected exactly 66 recognizable parents")
```

For every parent: assemble measured aligned power; call `interpolate_power_gaps`; construct an anchor frame with measured/imputed provenance; attempt zero-boundary trimming. Permit failure only when the parent and missing-boundary side exactly match the approved 13-parent exclusion contract, and record the raw boundary evidence. Reconstruct and compute features for the other 53 parents. Assign splits, write exactly 38/10/5 CSVs, then write all manifests and audits. Use stable sample IDs `zero_boundary_001` through `zero_boundary_053` in included chronological order.

Write `policy.json` with all constants, the exact five Test parents, the greedy objective, chronological tie-break, natural cubic and PCHIP methods, battery sign convention, and the statement that source load is derived rather than independently measured. Write `source_files.csv` with relative raw path, size, and SHA-256. Write `qa_summary.json` with artifact counts, per-split point totals, negative/internal-stop counts, every acceptance result, raw/code/artifact hashes, and `formal_training_status: "NO-GO"` plus the remaining dataset-external blockers.

Create the destination first as a sibling temporary directory named `<output>.building-<pid>`, remove that temporary directory on failure, and atomically rename it to the requested destination only after all validation passes. Never remove or overwrite the requested destination.

- [ ] **Step 4: Run builder-focused tests and commit**

Run:

```powershell
python -X utf8 -B tests/test_v2_zero_boundary_dataset.py
git diff --check
```

Expected: all focused tests pass, including non-overwrite and metadata/hash checks.

Commit:

```powershell
git add tests/test_v2_zero_boundary_dataset.py src/main/build_zero_boundary_operating_dataset.py
git commit -m "feat(v2): build zero-boundary dataset"
```

## Task 5: Make explicit-root formal loading metadata-driven and compatible with retained internal negative load

**Files:**
- Modify: `tests/test_formal_operating_dataset.py`
- Modify: `src/utils/formal_operating_dataset.py`

- [ ] **Step 1: Add red loader tests**

Extend the synthetic fixture with `metadata/qa_summary.json` containing expected parent, segment, total-point, and split-point counts. Add an internal `-5.0` point to the Validation CSV and assert `load_operating_segment_loads("validation", split.validation_segments[0], split=split)` returns it. Add a test that changes the expected total from 9 to 10 and asserts the explicit-root audit raises `ValueError("formal dataset point_count mismatch")`. Keep the default root unchanged in this task because the real replacement artifact has not yet passed validation.

- [ ] **Step 2: Confirm the current loader fails the new contracts**

Run:

```powershell
python -X utf8 -B tests/test_formal_operating_dataset.py
```

Expected: failures for internal-negative rejection and missing metadata-driven count checks.

- [ ] **Step 3: Implement loader changes without fallback behavior**

Add `_read_expected_counts(root)` that loads `metadata/qa_summary.json`, validates integer `parent_count`, `segment_count`, `point_count`, and all three `split_point_counts`, and returns them. In `load_operating_segment_loads`, keep finite, timestamp, and one-second-grid checks but remove the blanket `loads < 0` rejection. In `audit_formal_operating_dataset`, compare the completed audit against metadata and raise a precise `ValueError` on any mismatch. Do not change `DEFAULT_OPERATING_DATASET_ROOT` yet and do not restore a legacy-manifest fallback.

- [ ] **Step 4: Run loader tests and commit**

Run:

```powershell
python -X utf8 -B tests/test_formal_operating_dataset.py
git diff --check
```

Expected: all formal-loader tests pass against both the synthetic fixture and the built default metadata.

Commit:

```powershell
git add tests/test_formal_operating_dataset.py src/utils/formal_operating_dataset.py
git commit -m "feat(v2): validate formal dataset metadata"
```

## Task 6: Audit all 66 parents, build the eligible 53-parent dataset, and independently validate artifacts

**Files:**
- Create through builder: `data/processed/operating_dataset_zero_boundary_v2/`
- Preserve: `data/processed/operating_dataset_final/`
- Preserve: `outputs/v2_segment_power_review/gap_diagnostics.csv`

- [ ] **Step 1: Record pre-build hashes and spreadsheet-creation marker**

Read the spreadsheet skill's required workflow and scientific-research guidance before authoring CSV artifacts. Then record hashes outside the destination and invoke the required marker exactly once:

```powershell
Get-ChildItem 'C:\Users\20883\OneDrive\Desktop\氢舟一号' -Recurse -File | Get-FileHash -Algorithm SHA256 | Export-Csv .codex_tmp\zero_boundary_raw_hashes_before.csv -NoTypeInformation -Encoding utf8
Get-ChildItem data\processed\operating_dataset_final -Recurse -File | Get-FileHash -Algorithm SHA256 | Export-Csv .codex_tmp\old_dataset_hashes_before.csv -NoTypeInformation -Encoding utf8
Get-FileHash outputs\v2_segment_power_review\gap_diagnostics.csv -Algorithm SHA256 | Export-Csv .codex_tmp\gap_diagnostics_hash_before.csv -NoTypeInformation -Encoding utf8
node 'C:\Users\20883\.codex\plugins\cache\openai-primary-runtime\spreadsheets\26.909.12148\skills\spreadsheets\container_tools\mark_artifact_operation_started.mjs' --operation-kind create --expected-output-count 59 --output-format csv
```

The expected CSV count is 53 segment files plus 6 CSV metadata files (`sample_manifest`, `parent_split_manifest`, `excluded_parent_manifest`, `trim_boundary_audit`, `interpolation_audit`, and `source_files`). JSON metadata is not counted by this CSV marker. The first execution marked 71 outputs before the user approved the 13-parent exclusion; do not invoke the marker a second time merely to revise the count.

- [ ] **Step 2: Run the one-shot real build**

Run:

```powershell
python -X utf8 -B src/main/build_zero_boundary_operating_dataset.py --raw-root 'C:\Users\20883\OneDrive\Desktop\氢舟一号' --output-root data\processed\operating_dataset_zero_boundary_v2
```

Expected: exit code 0; exactly 38 Train, 10 Validation, and 5 Test CSV files; exactly 13 approved exclusions; no `.building-*` directory remains.

- [ ] **Step 3: Run independent acceptance checks**

Run the explicit-root artifact tests and auditor; these load every manifest path and assert exact 53/38/10/5 counts, the 13 approved exclusions, fixed Test membership, unique parents, finite loads, exact `time_s = 0..n-1`, zero endpoints, sustained positive operation, SHA-256 agreement, and no missing/orphan paths or leakage:

```powershell
python -X utf8 -B tests/test_v2_zero_boundary_dataset.py
python -X utf8 -B -c "from pathlib import Path; from utils.formal_operating_dataset import audit_formal_operating_dataset; a=audit_formal_operating_dataset(Path('data/processed/operating_dataset_zero_boundary_v2')); assert (a.parent_voyage_count,a.segment_count)==(53,53); assert a.missing_segment_paths==a.orphan_segment_paths==a.parent_split_leakage==()"
```

Then compare pre/post hashes:

```powershell
Get-ChildItem 'C:\Users\20883\OneDrive\Desktop\氢舟一号' -Recurse -File | Get-FileHash -Algorithm SHA256 | Export-Csv .codex_tmp\zero_boundary_raw_hashes_after.csv -NoTypeInformation -Encoding utf8
Get-ChildItem data\processed\operating_dataset_final -Recurse -File | Get-FileHash -Algorithm SHA256 | Export-Csv .codex_tmp\old_dataset_hashes_after.csv -NoTypeInformation -Encoding utf8
Get-FileHash outputs\v2_segment_power_review\gap_diagnostics.csv -Algorithm SHA256 | Export-Csv .codex_tmp\gap_diagnostics_hash_after.csv -NoTypeInformation -Encoding utf8
Compare-Object (Import-Csv .codex_tmp\zero_boundary_raw_hashes_before.csv) (Import-Csv .codex_tmp\zero_boundary_raw_hashes_after.csv) -Property Path,Hash
Compare-Object (Import-Csv .codex_tmp\old_dataset_hashes_before.csv) (Import-Csv .codex_tmp\old_dataset_hashes_after.csv) -Property Path,Hash
Compare-Object (Import-Csv .codex_tmp\gap_diagnostics_hash_before.csv) (Import-Csv .codex_tmp\gap_diagnostics_hash_after.csv) -Property Path,Hash
```

Expected: every `Compare-Object` command prints nothing.

- [ ] **Step 4: Inspect the split and boundary audits before switching the default**

Verify every included parent in `trim_boundary_audit.csv` has exactly one start and one end row, constructed fractions lie in `[0,1]`, no extrapolation flag exists, all rounded timestamps lie within their brackets, and removed prefix/suffix counts are nonnegative. Verify `excluded_parent_manifest.csv` contains exactly the approved 3 missing-start and 10 missing-end parents. Verify the split manifest has 53 unique parents, exactly 10 validation parents, and the Test names exactly match the frozen list. If any check fails, stop without changing the formal default and leave `FORMAL_TRAINING` at `NO-GO`.

- [ ] **Step 5: Enforce per-file remote size safety and commit the complete versioned artifact**

The existing formal dataset is versioned, so the replacement dataset must also be versioned; otherwise the later default-path switch would create a broken clone. Run:

```powershell
$oversize = Get-ChildItem data\processed\operating_dataset_zero_boundary_v2 -Recurse -File | Where-Object Length -ge 100MB
if ($oversize) { $oversize | Select-Object FullName,Length; throw 'generated file reaches the remote 100 MB limit' }
git add data/processed/operating_dataset_zero_boundary_v2
git commit -m "data(v2): add zero-boundary operating set"
```

Expected: no file reaches 100 MB and the complete versioned root, including all 53 segment CSVs and metadata, is committed. Do not add Git LFS or alter ignore policy in this task.

## Task 7: Switch the validated formal default, run full verification, and hand off

**Files:**
- Verify all modified code, tests, design, plan, and generated artifacts.

- [ ] **Step 1: Write the final red loader-default test**

Replace the old hard-coded 65-parent/145-segment manifest assertion in `tests/test_formal_operating_dataset.py` with:

```python
frame = load_formal_operating_split().manifest
self.assertEqual(frame.parent_voyage.nunique(), 53)
self.assertEqual(len(frame), 53)
self.assertEqual(frame.groupby("split").size().to_dict(), {"test": 5, "train": 38, "validation": 10})
self.assertEqual(int((frame.groupby("parent_voyage").split.nunique() > 1).sum()), 0)
```

Run `python -X utf8 -B tests/test_formal_operating_dataset.py` and expect this test to fail against the old default root.

- [ ] **Step 2: Switch only the formal default and make the test green**

In `src/utils/formal_operating_dataset.py`, change:

```python
DEFAULT_OPERATING_DATASET_ROOT = REPO_ROOT / "data" / "processed" / "operating_dataset_zero_boundary_v2"
```

Run `python -X utf8 -B tests/test_formal_operating_dataset.py` and expect all tests to pass.

- [ ] **Step 3: Run focused tests**

```powershell
python -X utf8 -B tests/test_v2_segment_power_source.py
python -X utf8 -B tests/test_v2_power_gap_interpolation.py
python -X utf8 -B tests/test_v2_zero_boundary_dataset.py
python -X utf8 -B tests/test_formal_operating_dataset.py
```

Expected: all focused tests pass.

- [ ] **Step 4: Run all v2 tests and the complete suite**

```powershell
Get-ChildItem tests\test_v2_*.py | ForEach-Object { python -X utf8 -B $_.FullName; if ($LASTEXITCODE -ne 0) { throw "failed: $($_.Name)" } }
python -X utf8 -B -m unittest discover -s tests -p 'test_*.py'
```

Expected: every v2 file and the complete suite pass.

- [ ] **Step 5: Run solver smoke and compile/import checks**

```powershell
python -X utf8 -B -m unittest tests.test_v2_nonlinear_mpc.NonlinearMPCTests.test_n5_plan_constraints_balance_formal_soc_and_first_step_only -v
python -X utf8 -B -m compileall -q src tests
python -X utf8 -B -c "from v2.data.segment_power_source import load_parent_power_series; from v2.data.zero_boundary_dataset import trim_to_zero_boundaries, assign_parent_splits; from utils.formal_operating_dataset import load_formal_operating_split; s=load_formal_operating_split(); assert (len(s.train_parents),len(s.validation_parents),len(s.test_parents))==(38,10,5)"
git diff --check
```

Expected: solver smoke succeeds, compilation/import succeeds, split tuple is 38/10/5, and `git diff --check` prints nothing.

- [ ] **Step 6: Review scope and commit**

Confirm no DQN episode payload, state audit, action screening, sensitivity analysis, training, MPC objective, objective normalization, or degradation-economics file changed. Confirm `qa_summary.json` still states `FORMAL_TRAINING: NO-GO` with remaining external blockers.

Commit any remaining loader switch, docs, and validated metadata with:

```powershell
git add docs/superpowers/specs/2026-09-23-zero-boundary-dataset-design.md docs/superpowers/plans/2026-09-23-zero-boundary-dataset.md src/v2/data/segment_power_source.py src/v2/data/zero_boundary_dataset.py src/main/build_zero_boundary_operating_dataset.py src/utils/formal_operating_dataset.py tests/test_v2_segment_power_source.py tests/test_v2_zero_boundary_dataset.py tests/test_formal_operating_dataset.py
git add data/processed/operating_dataset_zero_boundary_v2/metadata
git commit -m "feat(v2): freeze zero-boundary operating data"
```

Do not push unless the user explicitly requests it. Final reporting must state the 66 audited / 13 excluded / 53 included counts, 38/10/5 split, fixed Test parents, boundary-kind counts, retained internal-negative count, total one-second points, validation commands, artifact root, commit SHA, and unchanged `FORMAL_TRAINING: NO-GO` blockers.
