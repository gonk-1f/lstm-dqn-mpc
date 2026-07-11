# Cubic-Spline 1 s Reconstruction Diagnostic Report

Date: 2026-07-07

## 1. Linear 1 s Cleanup

The old `linear_interp_1s` branch has been removed from the active workflow.

- Manifest: `outputs/cleanup_linear_interp_1s/linear_interp_delete_manifest.csv`
- Cleanup log: `outputs/cleanup_linear_interp_1s/cleanup_log.txt`
- Archive: `outputs/_archived_invalid_linear_interp_1s/`

Archived invalid linear outputs include the old 1 s split files, 1 s LSTM outputs, 1 s MPC trials, 1 s dataset build outputs, and `total_load_excels_1s`.

Deprecated source/test entrypoints retained only as references:

- `src/main/_deprecated_build_total_load_1s_linear_interp_DO_NOT_USE.py`
- `src/main/_deprecated_build_total_load_dataset_1s_linear_interp_721_DO_NOT_USE.py`
- `src/main/_deprecated_run_train_lstm_total_load_1s_linear_interp_721_DO_NOT_USE.py`
- `src/main/_deprecated_run_lstm_mpc_total_load_1s_linear_interp_DO_NOT_USE.py`
- `tests/_deprecated_test_build_total_load_1s_linear_interp_DO_NOT_USE.py`

Protected paths were not deleted: `total_load_excels`, `outputs/lstm_total_load_721`, and `outputs/lstm_mpc_total_load_test_fixed_baseline_v1`. `outputs/lstm_mpc_total_load_test` was protected but already absent.

Conclusion: the `np.interp` branch must not be used as valid high-frequency prediction evidence because its internal 1 s labels depend on future 30 s endpoints.

## 2. Cubic-Spline Method

Input data:

- Original 30 s voyage workbooks: `total_load_excels`
- Split file: `outputs/config/voyage_split_total_load_721.json`

Built outputs:

- `outputs/spline_1s_diagnostics/data/cubic_spline_1s_natural.csv`
- `outputs/spline_1s_diagnostics/data/cubic_spline_1s_not_a_knot.csv`
- `outputs/spline_1s_diagnostics/data/cubic_spline_1s_natural_clipped.csv`
- `outputs/spline_1s_diagnostics/data/cubic_spline_1s_not_a_knot_clipped.csv`

Each voyage was reconstructed separately. No spline was fit across voyage boundaries.

The generated datasets contain `load_total_kw` only; no `speed_knots` column was present in the generated spline CSVs. Therefore the diagnostic feature boundary remains load-only.

## 3. Natural vs Not-a-Knot

- `natural`: second derivative is constrained at the endpoints, which reduces endpoint curvature freedom.
- `not-a-knot`: the first two and last two spline intervals are forced to share the same cubic polynomial continuity condition; this is SciPy's default cubic-spline boundary style.

Both are global piecewise-cubic fits. Neither is a causal online reconstruction method.

## 4. Future-Node Dependence

Every reconstructed 1 s row is marked:

- `online_feasible=false`
- `uses_future_endpoint=true`

Interior samples between two 30 s nodes depend on the future node at the end of the interval, and cubic splines can also be affected by wider neighboring nodes through the spline solve. These rows are offline reconstructed labels, not real measured 1 s ground truth.

## 5. Reference Review Summary

Detailed review file: `outputs/spline_1s_diagnostics/reference_review_spline_prediction.md`.

- [Spline interpolation and ARIMA in stock forecasting](https://arxiv.org/abs/2311.10759): supports interpolation as a preprocessing/forecasting aid in stock data, not ship load 1 s measurement evidence.
- [Gaussian interpolation for sparse electrical load forecasting](https://arxiv.org/abs/2508.14069): supports interpolation/imputation as a sparse-load treatment, but its main method is Gaussian interpolation, not cubic spline.
- [N-HiTS](https://arxiv.org/abs/2201.12886): supports neural hierarchical interpolation inside a forecasting architecture, not 30 s measurement resampling into real 1 s labels.
- [Revisiting forecasting with missing values](https://arxiv.org/abs/2509.23494): supports caution around imputation-then-prediction; reconstructed values are not true observed ground truth.

These references justify an offline sensitivity diagnostic only. They do not justify using cubic-spline 1 s reconstructions as main paper evidence.

## 6. Physical Reasonableness Check

Output: `outputs/spline_1s_diagnostics/spline_physical_check.csv`

Aggregate counts across 66 voyages:

| dataset_version | original_30s_negative | reconstructed_1s_negative | local_endpoint_overshoot | global_overshoot |
|---|---:|---:|---:|---:|
| cubic_spline_1s_natural | 5931 | 187960 | 468531 | 10314 |
| cubic_spline_1s_not_a_knot | 5931 | 187993 | 468999 | 10381 |
| cubic_spline_1s_natural_clipped | 0 | 0 | 417957 | 4516 |
| cubic_spline_1s_not_a_knot_clipped | 0 | 0 | 418271 | 4542 |

Important distinction: some negative values are already present at original 30 s nodes, but most negative values in the raw spline datasets occur at reconstructed 1 s interior samples.

Clipping removes negative values but does not remove the local endpoint overshoot problem.

## 7. Overshoot and Oscillation Evidence

Diagnostic figures:

- `outputs/spline_1s_diagnostics/figures/spline_physical_failure_counts.png`
- `outputs/spline_1s_diagnostics/figures/spline_natural_worst_local_overshoot_window.png`

Largest local endpoint overshoot found in the natural spline diagnostic was approximately `56.17 kW` in `voyage_005`.

This behavior is a known risk of cubic splines near sharp changes. The reconstructed sequence is smooth, but the smoothness is mathematical interpolation behavior, not measured high-frequency dynamics.

## 8. Smoothness and Predictability Audit

Output: `outputs/spline_1s_diagnostics/spline_predictability_audit.csv`

Test-voyage baseline MAE:

| dataset_version | current h1 | last-slope h1 | current h6 | last-slope h6 | current h30 | last-slope h30 | current h60 | last-slope h60 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cubic_spline_1s_natural | 0.599927 | 0.038462 | 3.579486 | 0.802507 | 15.658110 | 15.247880 | 23.352425 | 39.797703 |
| cubic_spline_1s_not_a_knot | 0.599972 | 0.038468 | 3.579688 | 0.802592 | 15.658542 | 15.247814 | 23.352427 | 39.798040 |

The h1 and h6 last-slope errors are extremely small compared with current-hold. This indicates that the short-horizon task is dominated by the smooth mathematical shape of the spline reconstruction.

## 9. Naive Baseline Result

Baseline plot:

- `outputs/spline_1s_diagnostics/figures/spline_naive_baseline_mae.png`

The last-slope extrapolation baseline is very strong at h1/h6 because a cubic-spline signal has locally smooth derivatives. This is not evidence of a strong learned forecasting model.

## 10. LSTM h1/h6/h30/h60 Result

The requested `spline_1s_short_h180_p60` LSTM was not trained.

Reason: pre-training audit already triggered stop conditions:

- raw spline versions produced reconstructed 1 s negative-load samples;
- raw and clipped versions produced large local endpoint overshoot;
- labels are offline and future-node-dependent;
- short-horizon last-slope baseline is already extremely strong.

Recorded skip marker:

- `outputs/spline_1s_diagnostics/models/spline_1s_short_h180_p60/SKIPPED_LSTM_DUE_TO_SPLINE_INVALID.md`

Summary table:

- `outputs/spline_1s_diagnostics/spline_lstm_vs_baseline_summary.csv`

The LSTM columns are intentionally `NaN` because no model was trained after the stop condition was met.

## 11. LSTM vs Current-Hold

No valid comparison is available because LSTM training was skipped by the audit gate. The baseline values are retained in the summary file for traceability.

## 12. LSTM vs Last-Slope

No valid comparison is available because LSTM training was skipped by the audit gate. The last-slope h1/h6 baseline is already strong enough to show that the spline reconstruction creates a mathematical extrapolation task.

## 13. Mathematical Regularity

The spline labels are smooth by construction. The natural and not-a-knot versions have almost identical baseline behavior:

- natural mean absolute second difference: `0.038462`
- not-a-knot mean absolute second difference: `0.038468`

This does not represent verified physical 1 s load dynamics. It represents an offline interpolation surface fitted to 30 s observations.

## 14. Stop Decision

Stop the 1 s cubic-spline reconstruction route.

The route fails the user's stop conditions:

- reconstructed 1 s negative values exist;
- large local endpoint overshoot exists;
- clipped variants still overshoot;
- labels are future-dependent and non-causal;
- naive last-slope is extremely strong at h1/h6.

## 15. MPC and Paper Boundary

No 1 s LSTM-MPC control run was generated.

Do not connect these spline outputs to MPC without a new user decision and a new evidence boundary.

Do not write spline results into Chapter 4 main experiments. At most, they can be described as a failed offline sensitivity diagnostic showing why reconstructed 1 s labels are not acceptable.

## 16. DQN-MPC Recommendation

Return to the original 30 s LSTM-MPC mainline and prepare the DQN-MPC dynamic-weighting stage.

Do not train DQN yet. The next stage should start from the accepted 30 s fixed baseline and keep the controller interface explicit: KAN is the Q-network type, and DQN should not directly output left/right powers unless the controller design changes.

## 17. Explicit Data Label

The cubic-spline datasets are offline reconstructed 1 s sequences.

They must not be called real 1 s measured data.
