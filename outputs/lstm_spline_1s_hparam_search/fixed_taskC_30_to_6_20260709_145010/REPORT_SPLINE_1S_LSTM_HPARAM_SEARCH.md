# Spline 1s LSTM Hyperparameter Search Report

Mode: fixed hyperparameter train/validation/test run

## 1. Data

The experiment uses `C:\Users\20883\OneDrive\Desktop\microgrid-mpc-master\outputs\spline_1s_diagnostics\data\natural_clipped_by_voyage`: natural clipped cubic-spline reconstructed 1s load profile.

## 2. Why This Is Not Measured 1s Data

The data are offline natural-boundary cubic-spline reconstructions from original 30 s vessel load voyages with nonnegative clipping. They are not native measured 1 s load data, not online prediction data, and intermediate 1 s points use future 30 s endpoint information.

## 3. Why Run The Search

The purpose is to find best hyperparameters on spline-reconstructed 1s data for diagnostics. It is not a claim of valid online 1s forecasting capability.

## 4. Reference Clues

The reference table records only sampling, preprocessing, LSTM, horizon, and metric clues. Missing values are marked `not_reported` or `not_applicable`.

## 5. Task Settings

- taskC_30_to_6: dt=1s, fixed_history_len=30, pred_horizon=6, forecast_time=6s.

## 6. Search Space

Search includes task-specific fixed history lengths when configured; otherwise history_len [60, 180, 300, 540]. It searches hidden_size [64, 128, 256], LSTM layers [1, 2, 3], dropout [0.0, 0.1, 0.2, 0.3], MLP heads [128], [256], [256, 128], loss [MSE, Huber], Adam learning rates [1e-3, 5e-4, 2e-4, 1e-4], batch size [32, 64, 128], gradient clip [0.5, 1.0, 5.0], weight_decay [0.0, 1e-6, 1e-5], and seed [42, 123].

## 7. Search Strategy

Optuna minimizes mean validation WAPE across the full task horizon, not h1 only.

## Best taskC_30_to_6 Configuration

{"history_len": 30, "pred_horizon": 6, "hidden_size": 128, "num_layers": 3, "dropout": 0.0, "mlp_head": [128], "loss": "Huber", "learning_rate": 0.0001, "batch_size": 32, "gradient_clip": 1.0, "weight_decay": 1e-05, "seed": 123, "epochs_max": 12, "early_stopping_patience": 2}


## 10-11. LSTM Versus Baselines

See `baseline_compare_taskC_30_to_6.csv` for current-hold, last-slope, moving-average hold, EMA hold, and LSTM metrics.

## 12. Spline Regularity

Very small errors or strong naive baselines are expected diagnostics because the labels are smooth spline reconstructions using future endpoints. This is reported as a limitation, not a stop condition.

## 13. h1 Versus Long Horizons

See `metrics_by_horizon_taskC_30_to_6.csv` and the error-vs-horizon figures.

## 14. Use Of Current Best Hyperparameters

They can be used for later diagnostics on the same spline-reconstructed data source. They should not be treated as validated real 1s predictors.

## 15. Main Paper Recommendation

Do not use this as main paper validity evidence.

## 16. Appendix Recommendation

Use only as appendix or sensitivity evidence, with the offline spline caveat stated explicitly.

## 17. Data Needed For Real Online Forecasting

A real online forecasting claim requires native measured 1s load, causal acquisition timestamps, no future endpoint dependence, and a split built from those measured data.

## Conclusion Boundary

- A. best hyperparameters on spline-reconstructed 1s data: yes, within the run mode above.
- B. valid online 1s load forecasting capability: no.
