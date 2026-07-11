# Millisecond 10 ms LSTM Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Copy and hash the two supplied 1 ms workbooks, directly retain every tenth row, construct a leakage-safe exact 7:2:1 split, and run a bounded Optuna search for a univariate 30-step-to-6-step LSTM forecast experiment.

**Architecture:** A standard-library XLSX reader and dataset builder produce immutable, provenance-rich 10 ms atomic sequences and a deterministic split manifest. A separate forecasting module owns windows, train-only scaling, baselines, metrics, and the sequence-to-vector LSTM; a CLI runner owns bounded Optuna search, robust seed selection, held-out evaluation, and reports. This path is isolated from every 30 s LSTM/MPC entrypoint and from the spline 1 s artifacts.

**Tech Stack:** Python 3.11, standard-library `zipfile`/`xml.etree`, pandas, NumPy, PyTorch 2.8, Optuna 4.9, scikit-learn, Matplotlib, `unittest`.

---

## Constraints And File Map

The approved design is `docs/superpowers/specs/2026-07-10-millisecond-10ms-lstm-design.md`. The formal runtime is `D:\py\Python3\python.exe`; it has pandas, NumPy, SciPy, scikit-learn, Matplotlib, Optuna, PyTorch, and PyArrow, but no `openpyxl`. XLSX reading must therefore use the standard library, following the established pattern in `src/main/build_total_load_dataset_721.py`.

The workspace has an empty `.git` directory and `git rev-parse --show-toplevel` fails. Git commit checkpoints are unavailable. Each task instead ends with focused tests, and retained inputs/runs are checkpointed by SHA-256 manifests plus new timestamped output directories.

**Create:**

- `src/main/build_millisecond_10ms_dataset.py`: XLSX parsing, source copy/hash, direct decimation, overlap union, exact grouped split, CSV/JSON artifacts, CLI.
- `src/forecasting/millisecond_multistep_lstm.py`: window construction, train-only scaler, baselines, metrics, model, deterministic prediction helpers.
- `src/main/run_lstm_millisecond_10ms_search.py`: training loop, bounded Optuna study, robust configuration selection, held-out evaluation, plots, report, CLI.
- `tests/test_millisecond_10ms_dataset.py`: workbook, decimation, overlap, split, and artifact tests.
- `tests/test_millisecond_multistep_lstm.py`: windows, scaler, baseline, metric, and tensor-shape tests.
- `tests/test_lstm_millisecond_10ms_search.py`: search limits, trial persistence, model-selection isolation, and checkpoint tests.

**Modify only after formal results exist:**

- `project_status.md`
- `next_steps.md`
- `thread.md`

**Never modify in this plan:**

- `src/main/run_train_lstm_721.py`
- `src/main/run_lstm_mpc_test.py`
- existing `outputs/lstm_721/`, `outputs/lstm_total_load_721/`, `outputs/lstm_spline_1s_hparam_search/`, or `outputs/spline_1s_diagnostics/` files
- MPC, DQN, or KAN-DQN modules

### Task 1: Multi-Sheet XLSX Reader And Source Provenance

**Files:**

- Create: `tests/test_millisecond_10ms_dataset.py`
- Create: `src/main/build_millisecond_10ms_dataset.py`

- [x] **Step 1: Write a minimal multi-sheet XLSX fixture and failing reader tests**

Create a ZIP-based fixture in the test module so the test does not depend on `openpyxl`. It must contain one condition sheet and one overview sheet, with string headers and numeric cells. Add these assertions:

```python
class TestMillisecondWorkbookReader(unittest.TestCase):
    def test_reads_condition_sheets_and_excludes_overview(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.xlsx"
            _write_minimal_multisheet_xlsx(
                path,
                {
                    "segment_1": [
                        ["time_s", "load_kw", "fuel_cell_kw", "battery_kw", "bus_voltage_v"],
                        [0.000, 1.0, 0.7, 0.3, 540.0],
                        [0.001, 1.1, 0.8, 0.3, 540.1],
                    ],
                    "overview": [
                        ["time_s", "load_kw", "fuel_cell_kw", "battery_kw", "bus_voltage_v"],
                        [0.000, 1.0, 0.7, 0.3, 540.0],
                    ],
                },
            )
            sheets = read_condition_sheets(path, overview_names={"overview"})
            self.assertEqual(list(sheets), ["segment_1"])
            self.assertEqual(list(sheets["segment_1"].columns), list(REQUIRED_COLUMNS))
            self.assertEqual(sheets["segment_1"].shape, (2, 5))

    def test_rejects_missing_required_column(self) -> None:
        frame = pd.DataFrame({"time_s": [0.0], "load_kw": [1.0]})
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            normalize_sheet_frame(frame, source="source.xlsx", sheet="segment_1")
```

The fixture helper must write `[Content_Types].xml`, `_rels/.rels`, `xl/workbook.xml`, `xl/_rels/workbook.xml.rels`, and one worksheet XML per supplied sheet. Use relationship IDs to resolve worksheet paths; do not assume `sheet1.xml` is always the first condition sheet.

- [x] **Step 2: Run the reader tests and confirm the intended failure**

Run:

```powershell
D:\py\Python3\python.exe -m unittest tests.test_millisecond_10ms_dataset.TestMillisecondWorkbookReader -v
```

Expected: import failure because `build_millisecond_10ms_dataset` does not yet exist.

- [x] **Step 3: Implement relationship-aware XLSX parsing and header normalization**

Use these public constants and interfaces in `build_millisecond_10ms_dataset.py`:

```python
REQUIRED_COLUMNS = (
    "time_s",
    "load_kw",
    "fuel_cell_kw",
    "battery_kw",
    "bus_voltage_v",
)

HEADER_ALIASES = {
    "\u65f6\u95f4_s": "time_s",
    "\u8d1f\u8f7d\u529f\u7387_kW": "load_kw",
    "\u71c3\u6599\u7535\u6c60\u529f\u7387_kW": "fuel_cell_kw",
    "\u9502\u7535\u6c60\u529f\u7387_kW": "battery_kw",
    "\u6bcd\u7ebf\u7535\u538b_V": "bus_voltage_v",
    "time_s": "time_s",
    "load_kw": "load_kw",
    "fuel_cell_kw": "fuel_cell_kw",
    "battery_kw": "battery_kw",
    "bus_voltage_v": "bus_voltage_v",
}

def normalize_sheet_frame(frame: pd.DataFrame, *, source: str, sheet: str) -> pd.DataFrame:
    renamed = frame.rename(columns=lambda value: HEADER_ALIASES.get(str(value).strip(), str(value).strip()))
    missing = [name for name in REQUIRED_COLUMNS if name not in renamed.columns]
    if missing:
        raise ValueError(f"{source}/{sheet} missing required columns: {missing}")
    out = renamed.loc[:, list(REQUIRED_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    if out.isna().any().any() or not np.isfinite(out.to_numpy(dtype=float)).all():
        raise ValueError(f"{source}/{sheet} contains missing or non-finite required values")
    return out.astype(float)

def read_condition_sheets(path: Path, *, overview_names: set[str]) -> dict[str, pd.DataFrame]:
    """Return workbook-ordered condition sheets after relationship resolution."""
```

Support shared strings, inline strings, `t="str"`, and numeric cells. Preserve workbook sheet order. Exclude only names passed through `overview_names`; production passes `{FULL_OVERVIEW_NAME}` where `FULL_OVERVIEW_NAME = "\u5168\u7a0b\u603b\u89c8"`.

- [x] **Step 4: Add source-copy and SHA-256 behavior with collision protection**

Add:

```python
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def copy_source_with_hash(source: Path, raw_dir: Path) -> dict[str, object]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    destination = raw_dir / source.name
    source_hash = sha256_file(source)
    if destination.exists() and sha256_file(destination) != source_hash:
        raise FileExistsError(f"Existing raw copy differs from source: {destination}")
    if not destination.exists():
        shutil.copy2(source, destination)
    if sha256_file(destination) != source_hash:
        raise IOError(f"Copied source hash mismatch: {destination}")
    return {
        "source_path": str(source.resolve()),
        "copied_path": str(destination.resolve()),
        "sha256": source_hash,
        "bytes": source.stat().st_size,
    }
```

Extend the tests to assert identical hashes and rejection when an existing destination has different bytes.

- [x] **Step 5: Run the focused tests**

Run:

```powershell
D:\py\Python3\python.exe -m unittest tests.test_millisecond_10ms_dataset.TestMillisecondWorkbookReader -v
```

Expected: all reader and source-provenance tests pass.

### Task 2: Direct Decimation And Atomic Overlap Union

**Files:**

- Modify: `tests/test_millisecond_10ms_dataset.py`
- Modify: `src/main/build_millisecond_10ms_dataset.py`

- [x] **Step 1: Write failing tests for exact row selection and time validation**

Add tests using 21 synthetic 1 ms rows. The retained source indices and values must be exact:

```python
def test_direct_decimation_keeps_source_rows_zero_ten_twenty(self) -> None:
    frame = pd.DataFrame(
        {
            "time_s": np.arange(21, dtype=float) / 1000.0,
            "load_kw": np.arange(21, dtype=float) + 100.0,
            "fuel_cell_kw": np.arange(21, dtype=float),
            "battery_kw": np.ones(21),
            "bus_voltage_v": np.full(21, 540.0),
        }
    )
    result = direct_decimate(frame, factor=10, source_workbook="book", source_sheet="segment")
    self.assertEqual(result["source_row_index"].tolist(), [0, 10, 20])
    self.assertEqual(result["load_kw"].tolist(), [100.0, 110.0, 120.0])
    np.testing.assert_allclose(np.diff(result["time_s"]), [0.01, 0.01], atol=1e-12)

def test_direct_decimation_rejects_non_1ms_input(self) -> None:
    frame = _numeric_frame(times=[0.000, 0.001, 0.003])
    with self.assertRaisesRegex(ValueError, "1 ms"):
        direct_decimate(frame, factor=10, source_workbook="book", source_sheet="segment")
```

- [x] **Step 2: Run and confirm the tests fail on the missing function**

Run:

```powershell
D:\py\Python3\python.exe -m unittest tests.test_millisecond_10ms_dataset.TestDirectDecimation -v
```

Expected: failure naming `direct_decimate`.

- [x] **Step 3: Implement direct row decimation with source-value audit columns**

Implement:

```python
def direct_decimate(
    frame: pd.DataFrame,
    *,
    factor: int,
    source_workbook: str,
    source_sheet: str,
) -> pd.DataFrame:
    if factor != 10:
        raise ValueError("This build requires factor=10 for 1 ms to 10 ms direct decimation")
    times = frame["time_s"].to_numpy(dtype=float)
    if len(times) > 1 and not np.all(np.diff(times) > 0):
        raise ValueError(f"{source_workbook}/{source_sheet} time must be strictly increasing")
    if len(times) > 1 and not np.allclose(np.diff(times), 0.001, rtol=0.0, atol=1e-9):
        raise ValueError(f"{source_workbook}/{source_sheet} is not a contiguous 1 ms sequence")
    source_indices = np.arange(0, len(frame), factor, dtype=np.int64)
    out = frame.iloc[source_indices].copy().reset_index(drop=True)
    out.insert(0, "source_row_index", source_indices)
    out.insert(0, "source_sheet", source_sheet)
    out.insert(0, "source_workbook", source_workbook)
    if len(out) > 1 and not np.allclose(np.diff(out["time_s"]), 0.010, rtol=0.0, atol=1e-9):
        raise ValueError(f"{source_workbook}/{source_sheet} direct decimation is not 10 ms")
    return out
```

The source-value equality test compares every retained numeric cell with `frame.iloc[source_row_index]`; no resampling API is allowed.

- [x] **Step 4: Write failing overlap-union tests**

Add one test where two decimated sheets overlap at integer milliseconds with equal values and one where the duplicate load differs:

```python
def test_union_overlap_deduplicates_equal_rows(self) -> None:
    left = _decimated_sequence("book_a", "left", [0.00, 0.01, 0.02], [1.0, 2.0, 3.0])
    right = _decimated_sequence("book_a", "right", [0.02, 0.03], [3.0, 4.0])
    merged = union_sequence_pair(left, right, sequence_id="book_a__left__right")
    self.assertEqual(merged["time_ms"].tolist(), [0, 10, 20, 30])
    self.assertEqual(len(merged), 4)

def test_union_overlap_rejects_disagreeing_values(self) -> None:
    left = _decimated_sequence("book_a", "left", [0.00, 0.01], [1.0, 2.0])
    right = _decimated_sequence("book_a", "right", [0.01, 0.02], [9.0, 3.0])
    with self.assertRaisesRegex(ValueError, "overlap disagreement"):
        union_sequence_pair(left, right, sequence_id="bad")
```

- [x] **Step 5: Implement deterministic integer-millisecond union**

Add `time_ms = np.rint(time_s * 1000).astype(np.int64)`, concatenate the pair, group by `time_ms`, and require every required numeric column to have a within-group peak-to-peak value no greater than `1e-9`. Retain one row per millisecond and store a semicolon-separated `source_members` field so both original sheets remain traceable. Require a 10 ms step after union.

Expose:

```python
def union_sequence_pair(left: pd.DataFrame, right: pd.DataFrame, *, sequence_id: str) -> pd.DataFrame:
    """Union an approved overlap pair and reject conflicting duplicate rows."""

def build_atomic_sequences(
    decimated: dict[tuple[str, str], pd.DataFrame],
    *,
    overlap_pairs: tuple[tuple[tuple[str, str], tuple[str, str]], ...],
) -> dict[str, pd.DataFrame]:
    """Return merged overlap pairs plus untouched independent sheets."""
```

Ensure no source sheet is consumed by more than one pair and no duplicate `(source_workbook, time_ms)` remains across atomic sequences.

- [x] **Step 6: Run all dataset unit tests so far**

Run:

```powershell
D:\py\Python3\python.exe -m unittest tests.test_millisecond_10ms_dataset -v
```

Expected: reader, provenance, decimation, and overlap tests pass.

### Task 3: Exact Grouped 7:2:1 Allocation And Dataset Artifacts

**Files:**

- Modify: `tests/test_millisecond_10ms_dataset.py`
- Modify: `src/main/build_millisecond_10ms_dataset.py`

- [ ] **Step 1: Write failing split tests**

Use synthetic sequence metadata with exact target sums and both workbooks in every split:

```python
def test_allocate_exact_split_is_disjoint_and_deterministic(self) -> None:
    groups = [
        _group("a1", "book_a", 40, 1.0),
        _group("a2", "book_a", 20, 2.0),
        _group("a3", "book_a", 10, 3.0),
        _group("b1", "book_b", 40, 1.5),
        _group("b2", "book_b", 20, 2.5),
        _group("b3", "book_b", 10, 3.5),
    ]
    loads = {group.sequence_id: np.full(group.rows, group.load_mean) for group in groups}
    first = allocate_exact_split(groups, sequence_loads=loads, targets={"train": 80, "validation": 40, "test": 20}, seed=20260710)
    second = allocate_exact_split(groups, sequence_loads=loads, targets={"train": 80, "validation": 40, "test": 20}, seed=20260710)
    self.assertEqual(first, second)
    self.assertEqual({name: sum(g.rows for g in first[name]) for name in first}, {"train": 80, "validation": 40, "test": 20})
    ids = {name: {g.sequence_id for g in first[name]} for name in first}
    self.assertFalse(ids["train"] & ids["validation"])
    self.assertFalse(ids["train"] & ids["test"])
    self.assertFalse(ids["validation"] & ids["test"])
    for name in ("train", "validation", "test"):
        self.assertEqual({g.source_workbook for g in first[name]}, {"book_a", "book_b"})

def test_allocate_exact_split_rejects_impossible_targets(self) -> None:
    with self.assertRaisesRegex(ValueError, "no exact valid assignment"):
        allocate_exact_split(
            [_group("a", "book_a", 7, 1.0)],
            sequence_loads={"a": np.ones(7)},
            targets={"train": 4, "validation": 2, "test": 1},
            seed=1,
        )
```

- [ ] **Step 2: Run the split tests and confirm failure**

Run:

```powershell
D:\py\Python3\python.exe -m unittest tests.test_millisecond_10ms_dataset.TestExactSplit -v
```

Expected: failure naming `allocate_exact_split`.

- [ ] **Step 3: Implement exact enumeration and a data-distribution score**

Create:

```python
@dataclass(frozen=True)
class SequenceGroup:
    sequence_id: str
    source_workbook: str
    rows: int
    load_mean: float
    load_std: float
    load_q10: float
    load_q50: float
    load_q90: float

def allocate_exact_split(
    groups: Sequence[SequenceGroup],
    *,
    sequence_loads: Mapping[str, np.ndarray],
    targets: Mapping[str, int],
    seed: int,
) -> dict[str, list[SequenceGroup]]:
    """Enumerate exact test and validation subsets, then score without model results."""
```

Enumerate `itertools.combinations` by subset size for the exact test sum, then exact validation subsets from remaining groups. Reject candidates unless both workbook IDs occur in all three splits. For each split, concatenate the actual load arrays supplied through a parallel `sequence_loads` mapping in the scoring helper, compute mean/std/q10/q50/q90, and minimize the sum of absolute standardized differences from global statistics. Sort candidate IDs and hash `f"{seed}:{ids}"` only as the final tie-break. Never inspect a model metric.

- [ ] **Step 4: Add window-count, sequence-hash, and artifact-manifest tests**

Build a tiny temporary dataset and assert:

```python
self.assertEqual(manifest["sample_interval_ms"], 10)
self.assertEqual(manifest["history_steps"], 30)
self.assertEqual(manifest["prediction_steps"], 6)
self.assertEqual(manifest["scaler_fit_scope"], "train_rows_only")
self.assertEqual(manifest["split_rows"], {"train": 80, "validation": 40, "test": 20})
self.assertEqual(manifest["window_formula"], "max(rows - 30 - 6 + 1, 0) per atomic sequence")
self.assertTrue(all(len(item["sha256"]) == 64 for item in manifest["atomic_sequences"]))
```

Also assert that the combined CSV has a unique `(sequence_id, time_ms)` key and that its `split` column agrees with the split JSON.

- [ ] **Step 5: Implement artifact writing and manifest self-audit**

Add:

```python
def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)

def build_dataset(
    *,
    source_paths: Sequence[Path],
    raw_root: Path,
    processed_root: Path,
    split_path: Path,
    split_seed: int = 20260710,
) -> dict[str, object]:
    """Copy, parse, decimate, merge, split, write, and re-read audit artifacts."""
```

Write one CSV per atomic sequence under `data/millisecond_10ms/segments/`, the combined `millisecond_load_10ms.csv`, `source_manifest.json`, `dataset_manifest.json`, and `outputs/config/millisecond_10ms_split_721.json`. CSVs use UTF-8 with BOM only if existing project conventions require Excel opening; JSON remains UTF-8. After writing, re-read every artifact and verify hashes, row counts, unique keys, exact 22,400/6,400/3,200 split, 19 sequences, and 32,000 rows before returning success.

- [ ] **Step 6: Add and test CLI defaults**

The parser defaults must point to the two user-supplied workbooks and dedicated project paths:

```python
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", type=Path)
    parser.add_argument("--raw-root", type=Path, default=Path("data/millisecond_1ms"))
    parser.add_argument("--processed-root", type=Path, default=Path("data/millisecond_10ms"))
    parser.add_argument("--split-path", type=Path, default=Path("outputs/config/millisecond_10ms_split_721.json"))
    parser.add_argument("--split-seed", type=int, default=20260710)
    return parser
```

When `--source` is omitted, use the two absolute source paths from the approved design. Print a compact JSON summary containing source hashes, input/decimated/unique row counts, sequence count, and split counts.

- [ ] **Step 7: Run dataset tests and syntax checks**

Run:

```powershell
D:\py\Python3\python.exe -m py_compile src\main\build_millisecond_10ms_dataset.py
D:\py\Python3\python.exe -m unittest tests.test_millisecond_10ms_dataset -v
```

Expected: compilation succeeds and all dataset tests pass.

### Task 4: Forecast Windows, Train-Only Scaling, Baselines, Metrics, And Model

**Files:**

- Create: `tests/test_millisecond_multistep_lstm.py`
- Create: `src/forecasting/millisecond_multistep_lstm.py`

- [ ] **Step 1: Write failing tests for sequence-local windows and scaler scope**

```python
class TestWindowConstruction(unittest.TestCase):
    def test_windows_do_not_cross_sequence_boundaries(self) -> None:
        sequences = {
            "a": np.arange(40, dtype=np.float32),
            "b": np.arange(100, 140, dtype=np.float32),
        }
        windows = build_windows(sequences, history_steps=30, prediction_steps=6)
        self.assertEqual(windows.x.shape, (10, 30, 1))
        self.assertEqual(windows.y.shape, (10, 6))
        self.assertEqual(set(windows.sequence_ids.tolist()), {"a", "b"})
        self.assertFalse(np.any((windows.x[:, -1, 0] < 50) & (windows.y[:, 0] > 50)))

    def test_scaler_uses_only_training_values(self) -> None:
        scaler = fit_standard_scaler({"train_a": np.array([0.0, 2.0])})
        self.assertEqual(scaler.mean, 1.0)
        self.assertEqual(scaler.std, 1.0)
        transformed = scaler.transform(np.array([101.0]))
        np.testing.assert_allclose(transformed, [100.0])
```

- [ ] **Step 2: Run and confirm import failure**

Run:

```powershell
D:\py\Python3\python.exe -m unittest tests.test_millisecond_multistep_lstm.TestWindowConstruction -v
```

Expected: import failure because `millisecond_multistep_lstm` does not exist.

- [ ] **Step 3: Implement immutable scaler and sequence-local windows**

```python
@dataclass(frozen=True)
class StandardScaler1D:
    mean: float
    std: float

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=np.float32) - self.mean) / self.std

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float32) * self.std + self.mean

@dataclass(frozen=True)
class WindowSet:
    x: np.ndarray
    y: np.ndarray
    sequence_ids: np.ndarray
    target_start_indices: np.ndarray

def fit_standard_scaler(train_sequences: Mapping[str, np.ndarray]) -> StandardScaler1D:
    values = np.concatenate([np.asarray(value, dtype=np.float64) for value in train_sequences.values()])
    std = float(values.std(ddof=0))
    if not np.isfinite(std) or std <= 0.0:
        raise ValueError("Training load standard deviation must be positive")
    return StandardScaler1D(mean=float(values.mean()), std=std)

def build_windows(
    sequences: Mapping[str, np.ndarray], *, history_steps: int, prediction_steps: int
) -> WindowSet:
    """Build stride-one windows independently for each ordered sequence."""
```

Each sequence contributes `max(n - history_steps - prediction_steps + 1, 0)` windows. Raise on non-finite values. Return empty arrays with stable shapes when no sequence is long enough.

- [ ] **Step 4: Write failing baseline and metric tests**

```python
def test_baselines_follow_declared_formulas(self) -> None:
    history = np.arange(30, dtype=np.float64)[None, :, None]
    forecasts = baseline_forecasts(history, prediction_steps=6)
    np.testing.assert_allclose(forecasts["current_hold"], np.full((1, 6), 29.0))
    np.testing.assert_allclose(forecasts["last_slope"], [[30, 31, 32, 33, 34, 35]])
    np.testing.assert_allclose(forecasts["local_linear_trend"], [[30, 31, 32, 33, 34, 35]], atol=1e-10)

def test_horizon_metrics_use_wape_not_row_mape(self) -> None:
    truth = np.array([[0.0, 2.0], [2.0, 2.0]])
    pred = np.array([[1.0, 1.0], [1.0, 3.0]])
    table = metrics_by_horizon(truth, pred)
    self.assertAlmostEqual(float(table.loc[table.horizon == 1, "wape_pct"].iloc[0]), 100.0)
    self.assertAlmostEqual(float(table.loc[table.horizon == 2, "wape_pct"].iloc[0]), 50.0)
```

- [ ] **Step 5: Implement baselines and metrics**

`baseline_forecasts` must accept raw-kW histories shaped `(samples, 30, 1)`. Local trend uses `np.linalg.lstsq` over all 30 positions. `metrics_by_horizon` returns horizon, MAE, RMSE, WAPE percent, bias, R-squared, and sample count. Define zero-denominator WAPE as `NaN`, not infinity. Add an aggregate row that computes metrics over all six outputs without averaging per-row percentages.

- [ ] **Step 6: Write and satisfy the model-shape test**

```python
def test_model_maps_30_by_1_to_six_outputs(self) -> None:
    config = ModelConfig(hidden_size=32, num_layers=2, dropout=0.1, mlp_head=(64,))
    model = SequenceToVectorLSTM(config=config, prediction_steps=6)
    output = model(torch.zeros(4, 30, 1))
    self.assertEqual(tuple(output.shape), (4, 6))
```

Implement:

```python
@dataclass(frozen=True)
class ModelConfig:
    hidden_size: int
    num_layers: int
    dropout: float
    mlp_head: tuple[int, ...]

class SequenceToVectorLSTM(nn.Module):
    def __init__(self, *, config: ModelConfig, prediction_steps: int = 6) -> None:
        super().__init__()
        recurrent_dropout = config.dropout if config.num_layers > 1 else 0.0
        self.lstm = nn.LSTM(1, config.hidden_size, config.num_layers, batch_first=True, dropout=recurrent_dropout)
        widths = (config.hidden_size,) + config.mlp_head + (prediction_steps,)
        layers: list[nn.Module] = []
        for index in range(len(widths) - 1):
            layers.append(nn.Linear(widths[index], widths[index + 1]))
            if index < len(widths) - 2:
                layers.append(nn.ReLU())
        self.head = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(inputs)
        return self.head(output[:, -1, :])
```

- [ ] **Step 7: Run core tests and compile the module**

Run:

```powershell
D:\py\Python3\python.exe -m py_compile src\forecasting\millisecond_multistep_lstm.py
D:\py\Python3\python.exe -m unittest tests.test_millisecond_multistep_lstm -v
```

Expected: all window, scaler, baseline, metric, and model-shape tests pass.

### Task 5: Time-Bounded Training And Resumable Optuna Study

**Files:**

- Create: `tests/test_lstm_millisecond_10ms_search.py`
- Create: `src/main/run_lstm_millisecond_10ms_search.py`

- [ ] **Step 1: Write failing parser and search-space tests**

```python
class TestSearchContract(unittest.TestCase):
    def test_default_limits_match_approved_design(self) -> None:
        args = build_parser().parse_args([])
        self.assertEqual(args.n_trials, 24)
        self.assertEqual(args.study_timeout_s, 3600)
        self.assertEqual(args.trial_timeout_s, 180)
        self.assertEqual(args.max_epochs, 25)
        self.assertEqual(args.patience, 4)
        self.assertEqual(args.history_steps, 30)
        self.assertEqual(args.prediction_steps, 6)

    def test_seed_is_not_an_optuna_parameter(self) -> None:
        trial = optuna.trial.FixedTrial(
            {
                "hidden_size": 32,
                "num_layers": 1,
                "mlp_head": "none",
                "loss": "Huber",
                "learning_rate": 1e-3,
                "batch_size": 64,
                "gradient_clip": 1.0,
                "weight_decay": 0.0,
            }
        )
        config = sample_trial_config(trial, fixed_seed=42)
        self.assertEqual(config.seed, 42)
        self.assertNotIn("seed", trial.params)
```

- [ ] **Step 2: Run and confirm the import failure**

Run:

```powershell
D:\py\Python3\python.exe -m unittest tests.test_lstm_millisecond_10ms_search.TestSearchContract -v
```

Expected: import failure because the runner does not exist.

- [ ] **Step 3: Implement parser, trial configuration, and deterministic setup**

```python
@dataclass(frozen=True)
class TrialConfig:
    hidden_size: int
    num_layers: int
    dropout: float
    mlp_head: tuple[int, ...]
    loss: str
    learning_rate: float
    batch_size: int
    gradient_clip: float
    weight_decay: float
    seed: int
    max_epochs: int
    patience: int

def sample_trial_config(trial: optuna.Trial, *, fixed_seed: int, max_epochs: int = 25, patience: int = 4) -> TrialConfig:
    num_layers = int(trial.suggest_categorical("num_layers", [1, 2, 3]))
    dropout = 0.0 if num_layers == 1 else float(trial.suggest_categorical("dropout", [0.0, 0.1, 0.2, 0.3]))
    head_map = {"none": (), "64": (64,), "128": (128,), "128-64": (128, 64)}
    return TrialConfig(
        hidden_size=int(trial.suggest_categorical("hidden_size", [32, 64, 128, 256])),
        num_layers=num_layers,
        dropout=dropout,
        mlp_head=head_map[str(trial.suggest_categorical("mlp_head", list(head_map)))],
        loss=str(trial.suggest_categorical("loss", ["MSE", "Huber"])),
        learning_rate=float(trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True)),
        batch_size=int(trial.suggest_categorical("batch_size", [64, 128, 256])),
        gradient_clip=float(trial.suggest_categorical("gradient_clip", [0.5, 1.0, 5.0])),
        weight_decay=float(trial.suggest_categorical("weight_decay", [0.0, 1e-6, 1e-5, 1e-4])),
        seed=fixed_seed,
        max_epochs=max_epochs,
        patience=patience,
    )
```

Set Python, NumPy, CUDA, and PyTorch seeds. Record whether deterministic algorithms were enabled and the device name; do not claim bitwise CUDA reproducibility when it is not guaranteed.

- [ ] **Step 4: Write failing training-loop tests for early stopping and per-trial timeout**

Inject a clock callable and a one-epoch function into the loop so tests do not train a real network:

```python
def test_training_stops_when_trial_deadline_is_reached(self) -> None:
    clock = iter([0.0, 1.0, 181.0]).__next__
    result = run_training_loop(
        state=_fake_training_state(),
        max_epochs=25,
        patience=4,
        min_delta=1e-6,
        trial_timeout_s=180,
        clock=clock,
        run_epoch=lambda state, epoch: _epoch_result(2.0),
    )
    self.assertTrue(result.stopped_by_timeout)
    self.assertLess(result.epochs_completed, 25)
```

- [ ] **Step 5: Implement the real training loop**

Train on normalized windows, evaluate raw-kW validation predictions after each epoch, and monitor arithmetic mean WAPE across h1-h6. Improvement requires `score < best_score - 1e-6`. Store the best CPU state dict, best epoch, learning-curve rows, elapsed seconds, timeout flag, and validation MAE tie-break. Check elapsed time after every training batch and validation batch, not only after an epoch. A timed-out trial may return its best completed validation score if at least one full validation pass exists; otherwise raise `optuna.TrialPruned("trial timeout before validation")`.

- [ ] **Step 6: Write failing study-limit and immediate-persistence tests**

Use a temporary SQLite path and a synthetic objective:

```python
def test_study_honors_trial_count_and_uses_one_worker(self) -> None:
    calls: list[int] = []
    study = run_study(
        objective=lambda trial: calls.append(trial.number) or float(trial.number),
        storage_path=self.root / "study.sqlite3",
        study_name="test",
        n_trials=3,
        timeout_s=60,
        sampler_seed=20260710,
        trial_csv=self.root / "trials.csv",
    )
    self.assertLessEqual(len(study.trials), 3)
    self.assertTrue((self.root / "trials.csv").exists())
    self.assertEqual(len(calls), len(study.trials))
```

- [ ] **Step 7: Implement resumable study and partial CSV callback**

Use:

```python
storage = f"sqlite:///{storage_path.resolve().as_posix()}"
study = optuna.create_study(
    study_name=study_name,
    storage=storage,
    load_if_exists=True,
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=sampler_seed),
)
study.optimize(objective, n_trials=n_trials, timeout=timeout_s, n_jobs=1, callbacks=[persist_trials])
```

`persist_trials` writes a temporary CSV and atomically replaces `trials_partial.csv` after every COMPLETE, PRUNED, or FAIL state. Before resuming, verify the stored study user attributes match dataset-manifest SHA-256, split-manifest SHA-256, history 30, horizon 6, and search-space version. Refuse resume on mismatch.

- [ ] **Step 8: Save self-describing checkpoints**

Checkpoint dictionaries contain:

```python
{
    "model_state_dict": best_state,
    "trial_config": dataclasses.asdict(config),
    "scaler": dataclasses.asdict(scaler),
    "dataset_manifest_sha256": dataset_hash,
    "split_manifest_sha256": split_hash,
    "source_sha256": source_hashes,
    "best_epoch": best_epoch,
    "validation_metrics": validation_metrics,
    "seed": config.seed,
    "torch_version": torch.__version__,
    "optuna_version": optuna.__version__,
}
```

Load with `weights_only=False` only for these locally generated trusted checkpoints; validate all metadata before inference.

- [ ] **Step 9: Run search-runner unit tests and compile**

Run:

```powershell
D:\py\Python3\python.exe -m py_compile src\main\run_lstm_millisecond_10ms_search.py
D:\py\Python3\python.exe -m unittest tests.test_lstm_millisecond_10ms_search -v
```

Expected: parser, search-space, timeout, early-stopping, persistence, and checkpoint tests pass without invoking a long GPU run.

### Task 6: Robust Selection, Held-Out Evaluation, Figures, And Report

**Files:**

- Modify: `tests/test_lstm_millisecond_10ms_search.py`
- Modify: `src/main/run_lstm_millisecond_10ms_search.py`

- [ ] **Step 1: Write a failing test proving test data cannot affect selection**

```python
def test_configuration_selection_uses_validation_only(self) -> None:
    candidates = [
        _candidate("a", seed_val_wape=[2.0, 2.2, 1.8], seed_test_wape=[99.0, 99.0, 99.0]),
        _candidate("b", seed_val_wape=[3.0, 3.1, 2.9], seed_test_wape=[0.1, 0.1, 0.1]),
    ]
    selected = select_configuration(candidates)
    self.assertEqual(selected.config_id, "a")
```

The selection function must not accept a test metric argument.

- [ ] **Step 2: Implement top-three by Optuna rank and three-seed validation**

Take the three best unique completed trial configurations, retrain each with seeds `42`, `123`, and `20260710`, and rank configurations by mean validation WAPE; use mean validation MAE and canonical JSON config ID as deterministic tie-breaks. Select the epoch separately within each seed from validation. Only after this selection write `selection_complete.json`, then construct test windows and evaluate all three retained selected-config checkpoints. Seed 42 is the designated primary checkpoint, not the best test seed.

- [ ] **Step 3: Write per-seed, aggregate, baseline, and per-sequence metrics**

Produce:

```text
selection/top3_validation_by_seed.csv
selection/configuration_ranking.csv
checkpoints/selected_seed_42.pt
checkpoints/selected_seed_123.pt
checkpoints/selected_seed_20260710.pt
metrics/test_metrics_by_seed_horizon.csv
metrics/test_metrics_seed_mean_std.csv
metrics/test_baseline_metrics_by_horizon.csv
metrics/test_metrics_by_sequence.csv
metrics/train_validation_test_gap.csv
predictions/test_predictions_seed_42.parquet
```

Report MAE, RMSE, WAPE percent, bias, R-squared, sample count, and negative-prediction count for h1-h6 and aggregate. Raw predictions are primary. If clipped predictions are emitted, place them in separate columns and tables suffixed `_nonnegative_clipped`.

- [ ] **Step 4: Implement all declared figures with stable filenames**

Use Matplotlib `Agg` and create:

```text
figures/optuna_optimization_history.png
figures/optuna_parameter_importance.png
figures/learning_curves_selected_seeds.png
figures/error_vs_horizon.png
figures/lstm_vs_baselines.png
figures/prediction_vs_actual_scatter.png
figures/residual_distribution_by_horizon.png
figures/test_sequences/<sequence_id>_prediction_h1_h6.png
```

Every title or subtitle includes `dt=10 ms | history=30 (300 ms) | horizon=6 (60 ms)`. Parameter importance must emit an explanatory figure when too few completed trials make importance unavailable; it must not abort the run.

- [ ] **Step 5: Write the machine-readable summary and engineering report**

Create `run_summary.json`, `REPORT_MILLISECOND_10MS_LSTM.md`, and `artifact_manifest.json`. The report includes source hashes, direct-decimation audit, the exact split and sequence assignments, search limits and actual duration, selected hyperparameters, three-seed validation and test statistics, baseline comparison, aliasing limitation, negative predictions, and explicit statements that the experiment is forecasting-only and the uploaded files' sensor provenance is unverified.

Define the model comparison honestly:

- `LSTM_BEATS_ALL_BASELINES` only if selected-model mean test WAPE and mean test MAE are both lower than every declared baseline.
- `LSTM_MIXED_RESULT` when it wins one primary metric or some horizons but not both aggregate metrics.
- `BASELINE_MATCHES_OR_BEATS_LSTM` when any baseline is no worse on both aggregate MAE and WAPE.

- [ ] **Step 6: Add artifact-completeness tests**

Run a synthetic report build and assert all expected files exist, every PNG is larger than 10 KiB, every test sequence has one trace figure, summary paths are relative to the run root, and the artifact manifest hashes every retained file except itself.

- [ ] **Step 7: Run all focused unit tests**

Run:

```powershell
D:\py\Python3\python.exe -m unittest tests.test_millisecond_10ms_dataset tests.test_millisecond_multistep_lstm tests.test_lstm_millisecond_10ms_search -v
```

Expected: all focused tests pass.

### Task 7: Smoke Run, Formal Data Build, And Bounded GPU Experiment

**Files:**

- Generate temporary smoke files under `.codex_tmp/millisecond_10ms_lstm_smoke/`
- Generate retained data under `data/millisecond_1ms/` and `data/millisecond_10ms/`
- Generate split: `outputs/config/millisecond_10ms_split_721.json`
- Generate run: `outputs/lstm_millisecond_10ms_30_to_6/<run_id>/`

- [ ] **Step 1: Run a two-epoch one-trial smoke experiment**

First build data into a temporary root using the real workbooks, then run:

```powershell
D:\py\Python3\python.exe src\main\run_lstm_millisecond_10ms_search.py --dataset-root .codex_tmp\millisecond_10ms_lstm_smoke\data --split-path .codex_tmp\millisecond_10ms_lstm_smoke\split.json --output-root .codex_tmp\millisecond_10ms_lstm_smoke\runs --n-trials 1 --study-timeout-s 120 --trial-timeout-s 60 --max-epochs 2 --patience 1 --robust-top-k 1 --seeds 42
```

Expected: CUDA is used, one study trial completes or is cleanly pruned, one selected checkpoint is evaluated, and the smoke report/figures are complete. Remove the smoke directory only after recording the passing command in the execution log.

- [ ] **Step 2: Build the retained 10 ms dataset from project-copied sources**

Run:

```powershell
D:\py\Python3\python.exe src\main\build_millisecond_10ms_dataset.py --source "C:\Users\20883\OneDrive\Desktop\26.5.24test各工况段数据+总览.xlsx" --source "C:\Users\20883\OneDrive\Desktop\1036各工况段数据+总览1.xlsx"
```

Expected summary:

```text
condition_sheets=21
rows_1ms=339000
rows_10ms_before_overlap_removal=33900
atomic_sequences=19
unique_rows_10ms=32000
train_rows=22400
validation_rows=6400
test_rows=3200
```

Abort and investigate instead of editing expected counts if any value differs.

- [ ] **Step 3: Independently audit retained artifacts**

Run a read-only verification command that compares both source/copy hashes, re-counts unique `(sequence_id,time_ms)` keys, checks every 10 ms delta, checks split intersections, recomputes per-sequence window counts, and verifies both source workbooks occur in every split. Save it as `data/millisecond_10ms/independent_audit.json` and include its hash in the dataset manifest.

- [ ] **Step 4: Start the bounded formal Optuna run**

Run:

```powershell
D:\py\Python3\python.exe src\main\run_lstm_millisecond_10ms_search.py --dataset-root data\millisecond_10ms --split-path outputs\config\millisecond_10ms_split_721.json --output-root outputs\lstm_millisecond_10ms_30_to_6 --n-trials 24 --study-timeout-s 3600 --trial-timeout-s 180 --max-epochs 25 --patience 4 --robust-top-k 3 --seeds 42 123 20260710 --device cuda
```

Do not increase limits during the run. If interrupted, rerun the exact command with `--resume-run <run_id>`; study metadata validation must prevent accidental resume against different data.

- [ ] **Step 5: Verify the formal run without using test results to retune**

Confirm the study has at most 24 trials and wall-clock search duration at most the declared timeout plus report-writing overhead. Confirm `selection_complete.json` predates test metrics, all three checkpoints share one hyperparameter config but different seeds, and no post-test trial exists. Compare LSTM with current hold, last slope, and local linear trend using the predeclared decision rule.

### Task 8: Final Verification And Project Handoff

**Files:**

- Modify: `project_status.md`
- Modify: `next_steps.md`
- Modify: `thread.md`

- [ ] **Step 1: Run the full focused verification suite**

Run:

```powershell
D:\py\Python3\python.exe -m py_compile src\main\build_millisecond_10ms_dataset.py src\forecasting\millisecond_multistep_lstm.py src\main\run_lstm_millisecond_10ms_search.py
D:\py\Python3\python.exe -m unittest tests.test_millisecond_10ms_dataset tests.test_millisecond_multistep_lstm tests.test_lstm_millisecond_10ms_search -v
```

Expected: compilation succeeds and all focused tests pass.

- [ ] **Step 2: Inspect every generated figure**

Open the optimization history, learning curves, horizon comparison, baseline comparison, scatter, residual distribution, and every test-sequence h1/h6 image. Reject blank axes, clipped labels, unreadable Chinese glyphs, mismatched horizons, or traces that cross sequence boundaries.

- [ ] **Step 3: Verify isolation from existing experiment paths**

Compare hashes and modification times for the active 30 s checkpoint, 30 s split, retained spline Task C checkpoint, and spline source manifest against pre-run values. Any unexplained change fails acceptance.

- [ ] **Step 4: Overwrite-update the three handoff files with verified facts**

`project_status.md` records the new experiment as a separate forecasting-only path, exact data/split facts, run ID, selected config, test metrics, baseline decision, and caveats. `next_steps.md` records only evidence-based follow-ups and explicitly prevents EMS use. `thread.md` records commands, output paths, test count, actual search duration, interruption/resume state, and the spline-diagnosis plan/result link.

- [ ] **Step 5: Final acceptance check**

Confirm every acceptance criterion in the design spec has a corresponding artifact or test result. Report any unmet item directly; do not label the task complete while a required formal run, figure, or diagnostic is absent.
