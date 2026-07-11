# Millisecond 10 ms LSTM Forecast Experiment Design

## 1. Objective

Build an experiment that copies the two supplied 1 ms Excel workbooks into the project, performs direct row decimation to 10 ms, creates a leakage-safe 7:2:1 train/validation/test split, searches LSTM hyperparameters under hard runtime limits, and reports load-forecast performance for a 30-step input and 6-step output.

At 10 ms per point, the model consumes the previous 300 ms and predicts the next 60 ms. This experiment evaluates forecasting only. It does not provide data or forecasts to MPC, DQN, KAN-DQN, or any other energy-management component.

The experiment also diagnoses why the existing offline cubic-spline 1 s dataset is unusually predictable. It must distinguish interpolation regularity and future-endpoint information from classical model overfitting.

## 2. Verified Input Facts

The source workbooks are:

- `C:/Users/20883/OneDrive/Desktop/26.5.24test各工况段数据+总览.xlsx`
- `C:/Users/20883/OneDrive/Desktop/1036各工况段数据+总览1.xlsx`

The audit found:

- 21 condition sheets in total: 10 in the first workbook and 11 in the second.
- Five numeric columns: time, load power, fuel-cell power, lithium-battery power, and bus voltage.
- The forecast target is the load-power column. The other columns are retained for provenance and future analysis but are not LSTM inputs in this experiment.
- All audited condition-sheet values are finite.
- The time increment is 1 ms within floating-point tolerance.
- Directly selecting every tenth source row produces 33,900 rows at 10 ms before overlap removal.
- The load range across condition sheets is approximately 0.101 to 37.464 kW. This low-power experiment is not an energy-management operating scenario.
- The first workbook contains two exact duplicate overlap pairs after 10 ms decimation:
  - `段7_277-304s` and `段8_290-310s`: 1,400 duplicate 10 ms rows.
  - `段9_310-327s` and `段10_322-333s`: 500 duplicate 10 ms rows.
- Removing those duplicate overlaps leaves 32,000 unique 10 ms rows organized as 19 atomic sequences.
- Some sheet labels in the second workbook differ from the numeric time column by 60 s. Processing therefore uses the numeric time column as the time source of truth while retaining the original sheet label in metadata.

The `全程总览` sheets are excluded from model-data construction because their rows duplicate the condition sheets and include periods outside the selected conditions. They remain untouched in the copied raw workbooks.

## 3. File and Artifact Boundaries

The two original workbooks remain untouched. Their project copies and SHA-256 hashes are stored under:

```text
data/millisecond_1ms/raw/
data/millisecond_1ms/source_manifest.json
```

Processed data are stored separately:

```text
data/millisecond_10ms/segments/
data/millisecond_10ms/millisecond_load_10ms.csv
data/millisecond_10ms/dataset_manifest.json
```

The split manifest is isolated from the existing 30 s and spline 1 s splits:

```text
outputs/config/millisecond_10ms_split_721.json
```

Experiment outputs use a dedicated timestamped directory:

```text
outputs/lstm_millisecond_10ms_30_to_6/<run_id>/
```

No existing 30 s checkpoint, 30 s split, spline 1 s dataset, MPC output, or training entrypoint is overwritten or deleted.

## 4. Direct 10 ms Decimation

For each condition sheet, the first data row has source-row index zero. The retained indices are:

```text
0, 10, 20, 30, ...
```

The operation performs no interpolation, filtering, averaging, or smoothing. Each output row records the source workbook, source sheet, zero-based source-row index, original numeric time, and all five numeric values.

Validation requires:

- source rows are strictly increasing in time;
- source median step is 0.001 s within a small floating-point tolerance;
- retained source-row indices differ by exactly 10;
- retained median time step is 0.010 s within tolerance;
- no required value is missing or non-finite;
- every retained value equals its source workbook value.

Direct decimation can alias frequencies above the new 50 Hz Nyquist frequency. This is an explicit limitation of the user-requested direct-point method and must be stated in the report. No anti-alias filter is introduced because that would change the requested preprocessing method.

## 5. Atomic Sequences and Overlap Removal

Sliding windows never cross a condition boundary. Exact duplicate overlaps in the first workbook are handled before splitting:

- `段7_277-304s` and `段8_290-310s` become one time-ordered union sequence.
- `段9_310-327s` and `段10_322-333s` become one time-ordered union sequence.

The union operation keys rows by rounded integer milliseconds and requires duplicate load values to agree within numerical tolerance. A disagreement aborts the build instead of silently choosing one value.

All other condition sheets remain independent sequences, including sheets that meet at a boundary but do not overlap. The resulting 19 atomic sequences are the indivisible units used for data splitting.

## 6. Leakage-Safe 7:2:1 Split

The split target is based on the 32,000 unique 10 ms rows:

- training: 22,400 rows;
- validation: 6,400 rows;
- test: 3,200 rows.

Assignment is performed at atomic-sequence level. A sequence cannot appear in more than one split. The deterministic allocator first enumerates test-group combinations totaling exactly 3,200 rows, then enumerates disjoint validation-group combinations totaling exactly 6,400 rows. It rejects candidates that do not represent both source workbooks in every split. Among the remaining exact candidates, it minimizes differences in load mean, standard deviation, and selected quantiles; seed `20260710` is used only for a stable final tie-break. It does not use any model result or test error. The build aborts if no exact valid assignment exists.

The chosen assignments, row counts, window counts, load statistics, source workbook, source sheets, and sequence hashes are written to the split manifest before training starts. The manifest is immutable for the experiment run.

Windows are generated independently inside each sequence:

- input shape: `30 x 1`;
- output shape: `6`;
- window stride: one 10 ms point;
- no window crossing sequence or split boundaries;
- standardization mean and standard deviation fitted on training rows only.

## 7. Model

The model is a univariate sequence-to-vector predictor:

```text
30 load points -> LSTM -> optional MLP head -> 6 future load points
```

The implementation is isolated from the 30 s control-forecast model. Checkpoints include model state, scaler, hyperparameters, source hashes, split-manifest hash, epoch, validation metrics, PyTorch version, Optuna version, and random seed.

Raw model predictions are the primary reported result. A count of negative predictions is reported. Nonnegative clipping, if shown, is secondary and clearly labeled so post-processing cannot silently improve the main metrics.

## 8. Bounded Optuna Search

Optuna uses a seeded TPE sampler. Random seed is not a hyperparameter.

Hard limits are:

- 24 trials maximum;
- 3,600 s global study timeout;
- 180 s maximum training time per trial;
- 25 epochs maximum per trial;
- early-stopping patience of 4 epochs;
- one trial at a time on the available CUDA GPU.

The search space is:

- hidden size: `32`, `64`, `128`, `256`;
- LSTM layers: `1`, `2`, `3`;
- dropout: `0.0` for one layer; otherwise `0.0`, `0.1`, `0.2`, `0.3`;
- MLP head: none, `(64,)`, `(128,)`, `(128, 64)`;
- loss: MSE or Huber;
- Adam learning rate: log scale from `1e-4` to `3e-3`;
- batch size: `64`, `128`, `256`;
- gradient clipping: `0.5`, `1.0`, `5.0`;
- weight decay: `0`, `1e-6`, `1e-5`, `1e-4`.

The primary Optuna objective is the arithmetic mean of validation WAPE at horizons h1 through h6. Validation mean MAE is the tie-break metric. WAPE is computed as total absolute error divided by total absolute target magnitude for each horizon; per-row MAPE is not used as a primary metric because loads can approach zero.

Early stopping monitors the same mean validation WAPE objective. An improvement must exceed `1e-6`; otherwise the patience counter advances.

Each completed or failed trial is persisted immediately to an SQLite Optuna study and a partial CSV. An interrupted run can resume without repeating completed trials.

## 9. Robust Configuration Selection and Final Evaluation

The three best Optuna configurations are retrained using seeds `42`, `123`, and `20260710`. Hyperparameters are selected by mean validation WAPE across those seeds. A seed is never selected using test performance.

For the selected configuration:

- all three seed checkpoints are retained;
- seed 42 is the designated primary single-model checkpoint;
- test metrics are reported separately for each seed and as mean plus standard deviation;
- the test split is evaluated only after configuration selection is complete;
- no test result is fed back into hyperparameter or epoch selection.

Baselines use the identical test windows:

- current-value hold;
- last-slope extrapolation;
- least-squares local linear trend over the 30-point history.

The LSTM is not called superior unless it beats the relevant simple baselines on the predeclared primary metrics.

## 10. Metrics and Figures

Metrics are reported for train, validation, and test splits, and for each test atomic sequence:

- MAE in kW;
- RMSE in kW;
- WAPE in percent;
- bias in kW;
- R-squared as a secondary diagnostic;
- negative-prediction count;
- metrics at every horizon h1 through h6.

Required figures are:

- Optuna optimization history and parameter importance;
- training and validation learning curves;
- MAE, RMSE, and WAPE versus horizon;
- LSTM versus baseline comparison;
- actual and predicted h1/h6 traces for every test sequence;
- prediction-versus-actual scatter plot;
- residual distribution by horizon.

All figures state `dt=10 ms`, `history=30 points (300 ms)`, and `horizon=6 points (60 ms)` so step counts cannot be confused with seconds.

## 11. Spline 1 s Regularity Versus Overfitting Diagnosis

Two hypotheses are evaluated:

- interpolation-regularity hypothesis: short-horizon errors are low because natural cubic spline reconstruction creates smooth deterministic trajectories and uses future 30 s endpoints;
- overfitting hypothesis: the LSTM memorizes training trajectories and fails to generalize to held-out voyages.

The diagnostic uses the retained spline 1 s Task C artifacts and adds the following comparisons without changing the spline dataset:

1. Verify voyage-disjoint train/validation/test membership, train-only scaler fitting, and the recorded `uses_future_endpoint=true` construction property.
2. Compare LSTM, current hold, last slope, and local linear trend on the held-out test voyages.
3. Report errors separately for original 30 s knot rows and reconstructed interior rows.
4. Report errors by distance to the nearest original 30 s knot.
5. Compare h1/h6 short-horizon errors with h30/h60 behavior where available.
6. Add train, validation, and test gaps plus repeated-seed variability to identify classical overfitting.
7. Compare model capacity and error: a small model or simple extrapolator matching a larger LSTM supports the regularity hypothesis.

Existing retained evidence already shows that on spline 1 s Task C the last-slope baseline has h1/h6 MAE of about `0.0385/0.8026 kW`, while the LSTM has about `1.7851/3.8494 kW`. This is preliminary evidence that smooth interpolation structure, not exceptional LSTM learning, dominates the very low short-horizon baseline error. A final conclusion still requires the train/validation/test gap and knot-position diagnostics above.

Decision rules are:

- regularity is dominant if simple local extrapolation matches or beats LSTM on held-out voyages and reconstructed interior points are materially easier than original knots;
- overfitting is material if training error is much lower than validation/test error, degradation grows with capacity, or seed variance is large;
- both causes are reported if both patterns occur.

The report must not describe the spline 1 s rows as measured 1 s data or as online-feasible forecasts.

## 12. Error Handling and Test Strategy

Implementation follows test-driven development. Tests are written and observed failing before production code is added.

Required tests cover:

- workbook header normalization and missing-column rejection;
- exact every-tenth-row selection;
- 1 ms and 10 ms time-step validation;
- source-value equality after decimation;
- overlap union and disagreement rejection;
- atomic-sequence split disjointness;
- exact 7:2:1 row targets and both-workbook representation;
- no sliding window crossing a sequence boundary;
- train-only scaler fitting;
- fixed 30-step input and 6-step output shapes;
- Optuna timeout and trial-count limits;
- test-set isolation from model selection;
- baseline and horizon-metric calculations;
- spline knot/interior classification.

Before the full search, a temporary smoke run uses one trial and two epochs. Smoke artifacts live under `.codex_tmp` and are removed after verification; they do not enter retained `outputs/`.

## 13. Acceptance Criteria

The experiment is complete only when:

- raw project copies match source SHA-256 hashes;
- the processed dataset contains 32,000 unique 10 ms rows with a successful direct-decimation audit;
- the split manifest proves disjoint atomic sequences and the 22,400/6,400/3,200 row split;
- all focused tests pass;
- the bounded Optuna study completes or reaches its declared timeout without exceeding it;
- the selected configuration is retrained with three fixed seeds;
- test metrics and all required plots are generated;
- LSTM results are compared with all declared baselines;
- the spline report states whether regularity, overfitting, or both are supported by the diagnostics;
- `project_status.md`, `next_steps.md`, and `thread.md` are overwritten with verified final paths and results;
- no MPC, DQN, KAN-DQN, 30 s LSTM, or existing spline artifact is modified.

## 14. Non-Goals

- No energy-management or control claim.
- No MPC integration.
- No DQN or KAN-DQN training.
- No claim that the uploaded 1 ms values are raw sensor measurements without external provenance evidence.
- No anti-alias filtering, interpolation, or averaging during the requested 1 ms to 10 ms conversion.
- No random sliding-window split.
