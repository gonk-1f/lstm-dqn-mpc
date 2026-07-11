# Spline 1 s Predictability Diagnosis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine from held-out evidence whether the unusually low short-horizon errors on the offline cubic-spline 1 s data are dominated by interpolation regularity/future-endpoint construction, classical LSTM overfitting, or both.

**Architecture:** A read-only diagnostic runner consumes the retained natural-clipped spline voyages, the immutable voyage split, and the retained Task C 30-to-6 checkpoint. It annotates every forecast target as an original 30 s knot or reconstructed interior point, compares LSTM and local baselines by position and horizon, then runs a small repeated-seed/capacity study to measure train-validation-test gaps without changing any retained spline artifact.

**Tech Stack:** Python 3.11, pandas, NumPy, PyTorch 2.8, scikit-learn, Matplotlib, existing `run_lstm_spline_1s_hparam_search.py` model/training utilities, `unittest`.

---

## Constraints And File Map

This plan implements Section 11 of `docs/superpowers/specs/2026-07-10-millisecond-10ms-lstm-design.md` as an independently testable analysis. It must not rewrite the spline dataset, retained Task C checkpoint, existing metrics, or 30 s mainline.

The workspace does not currently expose a functional Git repository, so each task ends with focused tests and the final run uses a new timestamped output directory plus an artifact hash manifest.

**Create:**

- `src/main/diagnose_spline_1s_predictability.py`: data-contract audit, annotated windows, retained-model inference, baselines, repeated-seed/capacity study, decision logic, figures, report, CLI.
- `tests/test_spline_1s_predictability_diagnosis.py`: knot metadata, metric grouping, split/scaler audit, decision-rule, and test-isolation tests.

**Read only:**

- `outputs/spline_1s_diagnostics/data/natural_clipped_by_voyage/`
- `outputs/mpc_solver_benchmark_1s/data/voyage_split_spline_1s_total_load_721.json`
- `outputs/lstm_spline_1s_hparam_search/fixed_taskC_30_to_6_20260709_145010/`
- `src/main/run_lstm_spline_1s_hparam_search.py`

**Generate:**

- `outputs/spline_1s_predictability_diagnosis/<run_id>/`

### Task 1: Data Contract, Split Membership, And Scaler Audit

**Files:**

- Create: `tests/test_spline_1s_predictability_diagnosis.py`
- Create: `src/main/diagnose_spline_1s_predictability.py`

- [ ] **Step 1: Write failing tests for required spline columns and voyage-disjoint split**

```python
class TestSplineDataContract(unittest.TestCase):
    def test_validate_voyage_frame_requires_construction_metadata(self) -> None:
        frame = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=3, freq="s"),
                "time_s": [0.0, 1.0, 2.0],
                "load_total_kw": [1.0, 1.1, 1.2],
                "is_original_30s_point": [True, False, False],
                "online_feasible": [False, False, False],
                "uses_future_endpoint": [True, True, True],
            }
        )
        validated = validate_voyage_frame(frame, voyage_id="voyage_001", expected_split="train")
        self.assertTrue(validated["uses_future_endpoint"].all())
        self.assertFalse(validated["online_feasible"].any())

    def test_split_audit_rejects_overlapping_voyages(self) -> None:
        split = {
            "train_voyages": ["voyage_001"],
            "validation_voyages": ["voyage_002"],
            "test_voyages": ["voyage_001"],
        }
        with self.assertRaisesRegex(ValueError, "voyage overlap"):
            audit_split_membership(split)
```

- [ ] **Step 2: Run the contract tests and confirm import failure**

Run:

```powershell
D:\py\Python3\python.exe -m unittest tests.test_spline_1s_predictability_diagnosis.TestSplineDataContract -v
```

Expected: import failure because `diagnose_spline_1s_predictability` does not exist.

- [ ] **Step 3: Implement strict read-only validation**

Use:

```python
REQUIRED_SPLINE_COLUMNS = (
    "timestamp",
    "time_s",
    "load_total_kw",
    "is_original_30s_point",
    "online_feasible",
    "uses_future_endpoint",
)

def validate_voyage_frame(frame: pd.DataFrame, *, voyage_id: str, expected_split: str) -> pd.DataFrame:
    missing = [name for name in REQUIRED_SPLINE_COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError(f"{voyage_id} missing spline metadata columns: {missing}")
    out = frame.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="raise")
    out["time_s"] = pd.to_numeric(out["time_s"], errors="raise")
    out["load_total_kw"] = pd.to_numeric(out["load_total_kw"], errors="raise")
    if not out["timestamp"].is_monotonic_increasing or out["timestamp"].duplicated().any():
        raise ValueError(f"{voyage_id} timestamps are not strictly increasing")
    if not np.allclose(np.diff(out["time_s"]), 1.0, rtol=0.0, atol=1e-9):
        raise ValueError(f"{voyage_id} is not contiguous 1 s data")
    if out["online_feasible"].astype(bool).any():
        raise ValueError(f"{voyage_id} unexpectedly claims online feasibility")
    if not out["uses_future_endpoint"].astype(bool).all():
        raise ValueError(f"{voyage_id} does not record future-endpoint use")
    return out

def audit_split_membership(split: Mapping[str, Sequence[str]]) -> dict[str, object]:
    """Require pairwise-disjoint train, validation, and test voyage IDs."""
```

Load the split path recorded in the retained Task C `run_summary.json`; do not silently substitute another split. Verify it contains 46 train, 13 validation, and 7 test voyages and that the per-voyage CSV `split` column agrees.

- [ ] **Step 4: Write and satisfy a train-only scaler comparison test**

```python
def test_scaler_audit_compares_checkpoint_with_train_rows_only(self) -> None:
    frames = {
        "voyage_001": pd.DataFrame({"load_total_kw": [0.0, 2.0]}),
        "voyage_002": pd.DataFrame({"load_total_kw": [100.0, 200.0]}),
    }
    audit = audit_checkpoint_scaler(
        checkpoint_scaler={"mean": 1.0, "std": 1.0},
        frames=frames,
        train_voyages=["voyage_001"],
    )
    self.assertTrue(audit["matches_train_only"])
    self.assertEqual(audit["rows_used"], 2)
```

Recompute population mean/std from all train voyage rows only and require both checkpoint values to match within `rtol=1e-12, atol=1e-12`. Also compute the all-split scaler solely as a counterfactual and show that it was not used.

- [ ] **Step 5: Run the data-contract tests**

Run:

```powershell
D:\py\Python3\python.exe -m unittest tests.test_spline_1s_predictability_diagnosis.TestSplineDataContract -v
```

Expected: all contract, membership, and scaler tests pass.

### Task 2: Annotated Forecast Windows And Knot-Distance Labels

**Files:**

- Modify: `tests/test_spline_1s_predictability_diagnosis.py`
- Modify: `src/main/diagnose_spline_1s_predictability.py`

- [ ] **Step 1: Write failing target-label tests**

Create a 70-row synthetic voyage with original-knot flags at rows 0, 30, and 60:

```python
def test_annotated_windows_preserve_target_knot_and_distance(self) -> None:
    frame = _spline_frame(rows=70, knot_indices={0, 30, 60})
    windows = build_annotated_windows(
        {"voyage_x": frame}, history_steps=30, prediction_steps=6, stride=1
    )
    self.assertEqual(windows.x.shape, (35, 30, 1))
    self.assertEqual(windows.y.shape, (35, 6))
    self.assertTrue(windows.target_is_original[0, 0])
    self.assertEqual(windows.target_distance_to_knot_s[0, 0], 0)
    self.assertEqual(windows.target_distance_to_knot_s[0, 1], 1)
    self.assertEqual(windows.target_distance_to_knot_s[0, 5], 5)
    self.assertEqual(windows.voyage_ids[0], "voyage_x")
```

- [ ] **Step 2: Run and confirm the missing-function failure**

Run:

```powershell
D:\py\Python3\python.exe -m unittest tests.test_spline_1s_predictability_diagnosis.TestAnnotatedWindows -v
```

Expected: failure naming `build_annotated_windows`.

- [ ] **Step 3: Implement stable window metadata**

```python
@dataclass(frozen=True)
class AnnotatedWindowSet:
    x: np.ndarray
    y: np.ndarray
    voyage_ids: np.ndarray
    target_timestamps: np.ndarray
    target_is_original: np.ndarray
    target_distance_to_knot_s: np.ndarray

def nearest_knot_distance_s(timestamps: pd.Series, is_original: np.ndarray) -> np.ndarray:
    values_ns = pd.to_datetime(timestamps).astype("int64").to_numpy()
    knot_ns = values_ns[np.asarray(is_original, dtype=bool)]
    if len(knot_ns) == 0:
        raise ValueError("Voyage has no original 30 s knot rows")
    positions = np.searchsorted(knot_ns, values_ns)
    left = knot_ns[np.clip(positions - 1, 0, len(knot_ns) - 1)]
    right = knot_ns[np.clip(positions, 0, len(knot_ns) - 1)]
    return np.minimum(np.abs(values_ns - left), np.abs(right - values_ns)) // 1_000_000_000
```

Build windows separately per voyage and retain target metadata for every horizon. The output order must match voyage order from the split manifest and chronological order within voyage.

- [ ] **Step 4: Add distance-bin tests and implementation**

Use bins `knot_0s`, `interior_1_5s`, `interior_6_10s`, and `interior_11_15s`. Distances above 15 s indicate malformed 30 s knot spacing and must fail validation. `distance_bin` accepts an integer array and returns fixed categorical labels in that order.

- [ ] **Step 5: Run annotated-window tests**

Run:

```powershell
D:\py\Python3\python.exe -m unittest tests.test_spline_1s_predictability_diagnosis.TestAnnotatedWindows -v
```

Expected: window count, voyage IDs, knot flags, distances, and bins pass.

### Task 3: Retained Task C Inference And Simple Baselines

**Files:**

- Modify: `tests/test_spline_1s_predictability_diagnosis.py`
- Modify: `src/main/diagnose_spline_1s_predictability.py`

- [ ] **Step 1: Write a checkpoint-contract test**

```python
def test_checkpoint_contract_requires_fixed_task_c_fields(self) -> None:
    payload = {
        "model_state": {"weight": torch.tensor([1.0])},
        "config": {"history_len": 30, "pred_horizon": 6, "seed": 123},
        "scaler": {"mean": 10.0, "std": 2.0},
        "data_label": "cubic spline",
        "data_caveat": "offline reconstruction",
        "task": {"name": "taskC_30_to_6", "pred_horizon": 6, "fixed_history_len": 30},
        "mode": "fixed_hyperparameters",
    }
    validated = validate_checkpoint_contract(payload)
    self.assertEqual(validated["config"]["history_len"], 30)
    self.assertEqual(validated["config"]["pred_horizon"], 6)
```

Reject any checkpoint that is not 30-to-6, lacks scaler metadata, or lacks the offline data caveat.

- [ ] **Step 2: Implement retained-model loading without rewriting the checkpoint**

Import `MultiHorizonLSTM`, `TrialConfig`, `normalize_xy`, and `inverse_y` from `src/main/run_lstm_spline_1s_hparam_search.py` after inserting `src/main` into `sys.path`, matching the focused-test convention already used in this repository. Load the local checkpoint on CPU with `weights_only=False`, validate it, instantiate from its exact config, and use batched inference on the requested device.

Expose:

```python
def predict_retained_task_c(
    checkpoint_path: Path,
    windows: AnnotatedWindowSet,
    *,
    device: str,
    batch_size: int = 512,
) -> tuple[np.ndarray, dict[str, object]]:
    """Return raw-kW predictions and validated checkpoint metadata."""
```

- [ ] **Step 3: Add current-hold, last-slope, and local-linear tests**

For a linear 30-point history all three formulas must produce the expected arrays: hold repeats the last point; last slope adds the final first difference repeatedly; local linear fits all 30 history positions by least squares and extrapolates positions 30 onward. Use the same raw-kW windows as LSTM and no target information.

- [ ] **Step 4: Implement streaming prediction tables**

For train, validation, and test, write one Parquet file per split with columns:

```text
voyage_id, target_timestamp, horizon, actual_kw, lstm_kw,
current_hold_kw, last_slope_kw, local_linear_trend_kw,
is_original_30s_point, distance_to_knot_s, distance_bin
```

Do not combine h1 and h6 timestamps incorrectly: flatten row-major and repeat each window's voyage ID once per horizon while flattening target metadata in the identical order. Add a round-trip test asserting that h1 and h6 timestamps match the source frame positions.

- [ ] **Step 5: Run retained-inference and baseline tests**

Run:

```powershell
D:\py\Python3\python.exe -m unittest tests.test_spline_1s_predictability_diagnosis.TestCheckpointAndBaselines -v
```

Expected: checkpoint contract, model output shape, baseline formulas, and flattened target alignment pass.

### Task 4: Knot/Interior, Distance, Voyage, And Horizon Metrics

**Files:**

- Modify: `tests/test_spline_1s_predictability_diagnosis.py`
- Modify: `src/main/diagnose_spline_1s_predictability.py`

- [ ] **Step 1: Write failing grouped-metric tests**

```python
def test_group_metrics_keep_knot_and_interior_separate(self) -> None:
    rows = pd.DataFrame(
        {
            "model": ["LSTM"] * 4,
            "horizon": [1, 1, 6, 6],
            "actual_kw": [10.0, 10.0, 10.0, 10.0],
            "predicted_kw": [12.0, 10.5, 13.0, 11.0],
            "is_original_30s_point": [True, False, True, False],
            "distance_bin": ["knot_0s", "interior_1_5s", "knot_0s", "interior_1_5s"],
        }
    )
    result = grouped_metrics(rows, group_columns=["model", "horizon", "distance_bin"])
    knot_h1 = result.query("horizon == 1 and distance_bin == 'knot_0s'").iloc[0]
    interior_h1 = result.query("horizon == 1 and distance_bin == 'interior_1_5s'").iloc[0]
    self.assertEqual(knot_h1.mae_kw, 2.0)
    self.assertEqual(interior_h1.mae_kw, 0.5)
```

- [ ] **Step 2: Implement grouped metrics with explicit denominators**

Return MAE, RMSE, WAPE percent, bias, R-squared, and row count. WAPE is `100 * sum(abs(error)) / sum(abs(actual))`; write `NaN` when the target denominator is zero. Produce:

```text
metrics_by_split_model_horizon.csv
metrics_by_knot_interior.csv
metrics_by_distance_to_knot.csv
metrics_by_test_voyage.csv
train_validation_test_gap.csv
```

- [ ] **Step 3: Add long-horizon baseline audit at h1, h6, h30, and h60**

Build separate 30-history/60-target windows on held-out test voyages and compute only the three non-learned baselines. Write `baseline_short_vs_long_horizon.csv`. Clearly mark LSTM h30/h60 as unavailable because the retained model outputs six steps; never extend or recursively feed the model without a separate approved experiment.

- [ ] **Step 4: Reconcile with retained metrics**

Recomputed Task C test h1/h6 metrics must match `metrics_by_horizon_taskC_30_to_6.csv` within `rtol=1e-6, atol=1e-6`. Recomputed current-hold and last-slope metrics must match retained baseline rows within the same tolerance. Any mismatch aborts the diagnostic and records the conflicting values.

- [ ] **Step 5: Run grouped-metric tests**

Run:

```powershell
D:\py\Python3\python.exe -m unittest tests.test_spline_1s_predictability_diagnosis.TestGroupedMetrics -v
```

Expected: metric formulas, grouping, long-horizon baseline shapes, and retained-metric reconciliation fixtures pass.

### Task 5: Repeated-Seed And Capacity Overfitting Study

**Files:**

- Modify: `tests/test_spline_1s_predictability_diagnosis.py`
- Modify: `src/main/diagnose_spline_1s_predictability.py`

- [ ] **Step 1: Write a test proving model selection never sees test metrics**

```python
def test_capacity_summary_uses_fixed_configs_and_validation_epochs(self) -> None:
    configs = build_capacity_configs(seeds=[42, 123, 20260710])
    self.assertEqual({item.capacity_label for item in configs}, {"small", "retained_capacity"})
    self.assertEqual({item.seed for item in configs}, {42, 123, 20260710})
    self.assertTrue(all(item.history_len == 30 and item.pred_horizon == 6 for item in configs))
```

The study has no hyperparameter selector. It evaluates two predeclared capacities under three fixed seeds, with epoch selection based on validation WAPE only.

- [ ] **Step 2: Define the two capacity configurations**

Use the retained Task C optimization settings for both except architecture:

```python
CAPACITY_CONFIGS = {
    "small": {"hidden_size": 32, "num_layers": 1, "dropout": 0.0, "mlp_head": ()},
    "retained_capacity": {"hidden_size": 128, "num_layers": 3, "dropout": 0.0, "mlp_head": (128,)},
}
```

Both use Huber loss, learning rate `1e-4`, batch size 32, gradient clip 1.0, weight decay `1e-5`, at most 12 epochs, patience 2, and seeds 42/123/20260710. Record parameter count. Do not use test errors to pick architecture, seed, or epoch.

- [ ] **Step 3: Implement training and split evaluation**

Reuse `run_training` from the existing spline search script for validation-selected states. Evaluate the selected state on train, validation, and test with identical sequence-local windows and train-only scaler. Save six checkpoints under the new diagnosis run, not under the retained Task C tree. Store learning curves if the reused function is extended locally through a wrapper; do not modify the original script merely to collect them.

- [ ] **Step 4: Quantify gap, capacity, and seed diagnostics**

For each capacity/seed compute aggregate h1-h6 WAPE and MAE by split. Summarize mean/std and these predeclared flags:

```python
def overfit_flags(summary: pd.DataFrame) -> dict[str, bool]:
    train = float(summary.loc[summary["split"] == "train", "wape_pct_mean"].iloc[0])
    validation = float(summary.loc[summary["split"] == "validation", "wape_pct_mean"].iloc[0])
    test = float(summary.loc[summary["split"] == "test", "wape_pct_mean"].iloc[0])
    return {
        "generalization_gap": max(validation, test) >= train * 1.25 and max(validation, test) - train >= 0.5,
        "extreme_gap": max(validation, test) >= train * 2.0 and max(validation, test) - train >= 1.0,
    }
```

Set `capacity_degradation=true` when retained capacity improves mean train WAPE by at least 10% versus small capacity but worsens mean test WAPE by at least 10%. Set `high_seed_variance=true` when test-WAPE coefficient of variation is at least 0.15. These thresholds are diagnostics, not universal statistical laws, and the report must include raw values.

- [ ] **Step 5: Add a bounded smoke mode**

`--smoke` uses two voyages per split, at most 20,000 windows per split, one seed, two epochs, and both capacity labels. It writes only under `.codex_tmp/spline_1s_predictability_smoke/` and cannot write into retained outputs.

- [ ] **Step 6: Run overfitting-study unit tests**

Run:

```powershell
D:\py\Python3\python.exe -m unittest tests.test_spline_1s_predictability_diagnosis.TestOverfitStudy -v
```

Expected: fixed configs, validation-only epoch selection, thresholds, and smoke limits pass.

### Task 6: Evidence Classification, Figures, And Report

**Files:**

- Modify: `tests/test_spline_1s_predictability_diagnosis.py`
- Modify: `src/main/diagnose_spline_1s_predictability.py`

- [ ] **Step 1: Write table-driven decision-rule tests**

```python
def test_decision_reports_regularity_without_overfit_when_only_regularity_flags_hold(self) -> None:
    decision = classify_evidence(
        simple_baseline_beats_lstm=True,
        interior_easier_than_knots=True,
        long_horizon_degrades=True,
        generalization_gap=False,
        capacity_degradation=False,
        high_seed_variance=False,
    )
    self.assertEqual(decision, "REGULARITY_DOMINANT")

def test_decision_reports_both_when_both_evidence_families_hold(self) -> None:
    decision = classify_evidence(
        simple_baseline_beats_lstm=True,
        interior_easier_than_knots=True,
        long_horizon_degrades=True,
        generalization_gap=True,
        capacity_degradation=False,
        high_seed_variance=False,
    )
    self.assertEqual(decision, "REGULARITY_AND_OVERFITTING")
```

- [ ] **Step 2: Implement transparent evidence flags**

Set:

- `simple_baseline_beats_lstm`: last-slope or local-linear aggregate test MAE and WAPE are both no greater than retained LSTM.
- `interior_easier_than_knots`: best simple baseline interior aggregate MAE is at least 20% lower than its original-knot aggregate MAE.
- `long_horizon_degrades`: best simple baseline h60 WAPE is at least twice its h6 WAPE and exceeds it by at least 0.5 percentage points.
- `overfit_supported`: any of generalization gap, capacity degradation, or high seed variance is true.
- `regularity_supported`: future-endpoint metadata is true and at least two of the three regularity performance flags are true.

Return `REGULARITY_DOMINANT`, `OVERFITTING_MATERIAL`, `REGULARITY_AND_OVERFITTING`, or `INCONCLUSIVE`. Include all raw metrics and booleans so readers can disagree with thresholds without rerunning training.

- [ ] **Step 3: Generate diagnostic figures**

Create:

```text
figures/lstm_vs_baselines_h1_h6.png
figures/error_knot_vs_interior.png
figures/error_by_distance_to_knot.png
figures/baseline_h1_h6_h30_h60.png
figures/train_validation_test_gap.png
figures/capacity_seed_variability.png
figures/test_voyages/<voyage_id>_lstm_baselines_h1_h6.png
```

Titles state `offline natural cubic-spline reconstruction`, `uses future 30 s endpoint`, and `not measured 1 s`. Seven test-voyage figures are required.

- [ ] **Step 4: Write report and artifact manifest**

Produce:

```text
run_summary.json
evidence_decision.json
REPORT_SPLINE_1S_PREDICTABILITY_DIAGNOSIS.md
artifact_manifest.json
```

The report begins with the classification and evidence table, then separates interpolation construction evidence from overfitting evidence. It must state that a baseline beating LSTM is evidence of mathematical predictability, not evidence of online forecasting value. It must state that future 30 s endpoint use makes interior samples unavailable to a causal online forecaster at the reconstruction time.

- [ ] **Step 5: Add artifact and figure completeness tests**

On synthetic inputs, require every table/JSON/report file, all six summary figures, and one trace image per test voyage. Verify PNG dimensions are at least 1000 by 600 and files are nonblank by checking pixel standard deviation greater than zero.

- [ ] **Step 6: Run all diagnostic unit tests**

Run:

```powershell
D:\py\Python3\python.exe -m py_compile src\main\diagnose_spline_1s_predictability.py
D:\py\Python3\python.exe -m unittest tests.test_spline_1s_predictability_diagnosis -v
```

Expected: all contract, annotation, inference, metric, overfit, classification, and artifact tests pass.

### Task 7: Smoke Run, Formal Diagnosis, And Independent Verification

**Files:**

- Generate smoke: `.codex_tmp/spline_1s_predictability_smoke/`
- Generate formal run: `outputs/spline_1s_predictability_diagnosis/<run_id>/`

- [ ] **Step 1: Run bounded smoke diagnosis**

Run:

```powershell
D:\py\Python3\python.exe src\main\diagnose_spline_1s_predictability.py --smoke --output-root .codex_tmp\spline_1s_predictability_smoke --device cuda
```

Expected: both capacity labels run for one seed/two epochs on capped windows, retained-checkpoint inference succeeds, and all smoke tables/figures are generated. Remove the smoke directory after verification.

- [ ] **Step 2: Run the formal read-only diagnosis**

Run:

```powershell
D:\py\Python3\python.exe src\main\diagnose_spline_1s_predictability.py --source-dir outputs\spline_1s_diagnostics\data\natural_clipped_by_voyage --split-json outputs\mpc_solver_benchmark_1s\data\voyage_split_spline_1s_total_load_721.json --retained-run outputs\lstm_spline_1s_hparam_search\fixed_taskC_30_to_6_20260709_145010 --output-root outputs\spline_1s_predictability_diagnosis --seeds 42 123 20260710 --device cuda
```

Expected: retained metrics reconcile, six capacity/seed runs finish, seven test-voyage figures exist, and `evidence_decision.json` contains one of the four declared labels.

- [ ] **Step 3: Independently verify key numerical claims**

With a separate read-only command, recompute from saved prediction Parquet files: h1/h6 LSTM and baseline metrics, knot/interior MAE, h6/h60 baseline ratio, train/test WAPE gaps, parameter counts, and seed coefficient of variation. Compare each value to `evidence_decision.json` within `1e-9` for stored aggregates.

- [ ] **Step 4: Inspect all figures**

Open every summary figure and all seven voyage plots. Reject blank images, mismatched horizon labels, hidden legends, or any chart that describes the rows as measured 1 s data.

- [ ] **Step 5: Verify retained inputs were not modified**

Compare pre/post SHA-256 hashes for the retained Task C checkpoint, fixed config, horizon metrics, baseline metrics, split JSON, and spline per-voyage manifest. Any change fails the read-only requirement.

### Task 8: Integrate Verified Diagnosis Into Project Handoff

**Files:**

- Modify: `project_status.md`
- Modify: `next_steps.md`
- Modify: `thread.md`

- [ ] **Step 1: Update status only after the formal diagnosis is complete**

Record the exact output directory, classification, decisive raw metrics, threshold flags, seven-figure path, test result count, and retained-input hash verification. Keep the result separate from the 10 ms experiment and from any EMS claim.

- [ ] **Step 2: State limitations and next evidence needed**

Record that the diagnosis is about an offline natural cubic-spline reconstruction with future endpoint use. If classification is inconclusive, name the missing evidence directly. If overfitting is supported, distinguish it from interpolation regularity rather than presenting it as the sole explanation unless regularity flags fail.

- [ ] **Step 3: Final acceptance check**

Confirm the report includes voyage-disjoint membership, train-only scaler check, LSTM and three baselines, knot/interior and distance metrics, h1/h6/h30/h60 comparison with LSTM availability labeled correctly, repeated seeds, capacity comparison, train/validation/test gaps, seven test-voyage plots, explicit decision rules, and unchanged retained inputs.
