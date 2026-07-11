# Millisecond 10 ms LSTM Forecast Report

## Scope

This is a forecasting-only experiment. It is not approved for online energy-management use.

- dt=10 ms | history=30 (300 ms) | horizon=6 (60 ms)
- Direct row decimation: every 10th 1 ms row; no averaging or interpolation.
- Dataset rows: 32000 unique 10 ms rows across 19 atomic sequences.
- Split rows: {"train": 22400, "validation": 6400, "test": 3200}.
- Source SHA-256: d959cc8ea0c8f6f13ff450705ab4500be1c32aaec34a14fa80f0d1916d5b0905, f1877326970fa362df1484a47b9df2ee20016603a30f1ba88abbc9dd9c3d0200.
- Sensor provenance in the supplied workbooks is unverified.
- Direct decimation does not provide anti-alias filtering; sub-100 Hz content may alias.

## Search and selection

- Device: NVIDIA GeForce RTX 5060 Laptop GPU.
- Optuna limits: at most 1 trials, 120 s study time, 60 s per trial.
- Completed trials: 1; measured study call duration: 9.797 s.
- Selection used validation metrics only and completed at `2026-07-10T16:24:46.066729+00:00` before test windows were constructed.
- Selected hyperparameters: `{"batch_size": 64, "dropout": 0.0, "gradient_clip": 5.0, "hidden_size": 32, "learning_rate": 0.00017009062368270733, "loss": "MSE", "max_epochs": 2, "mlp_head": [128, 64], "num_layers": 1, "patience": 1, "seed": 42, "weight_decay": 1e-06}`.
- Seed 42 is the designated primary checkpoint; no best-test-seed selection was performed.

## Held-out test result

- Three-seed aggregate MAE mean: 0.144984 kW.
- Three-seed aggregate WAPE mean: 0.944024%.
- Raw negative prediction count summed across seed aggregate rows: 0.
- Predeclared baseline decision: **BASELINE_MATCHES_OR_BEATS_LSTM**.

Detailed per-horizon, per-seed, per-sequence, baseline, and train/validation/test-gap tables are under `metrics/`. Raw primary-seed predictions are under `predictions/`.
