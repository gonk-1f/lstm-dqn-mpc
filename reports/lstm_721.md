# LSTM18 P6 Historical Archive

## Scope

- These historical LSTM results use an offline 1 s cubic-spline reconstruction of measured 30 s data.
- The spline reconstruction is not measured online 1 s data and does not establish real online 1 s forecasting performance.
- These results are no longer the project's formal load-forecasting result.
- LSTM is not connected to the current MPC and is not part of the current DQN state space.
- Only model files, minimal reproducibility configuration, model definitions, selection manifests, and final metric summaries are retained as a historical archive.
- The LSTM must be retrained and reevaluated when measured 1 s data becomes available.
- Hidden size, layer count, and epoch scale were not increased.
- MAPE is not used as a training loss or as the main selection metric because near-zero load points make it unstable.
- The asymmetric candidates use `w_under=3.0`, high-load P80 bonus `0.5`, and positive-ramp P80 bonus `0.2`.
- Huber delta is specified in kW and converted to normalized target space during training.
- Historical forecast metrics use the nonnegative projection `pred_kw = clip(pred_raw_kw, 0, None)`.

## Candidate Selection

- Historically selected archive candidate: `candidate_asym_weighted_huber_delta10`.
- Selection note: delta10 was the retained candidate in the completed historical comparison.
- This archived selection is not a current formal forecasting recommendation.

## Requested Metrics

| Candidate | Loss | delta kW | h1-h6 MAE | h1-h6 RMSE | h1-h6 WAPE | h1-h6 Bias | 0-3h MAE | 0-3h RMSE | 0-3h WAPE | 0-3h Bias | 0-3h under ratio | Peak under % | Low-load bias | Low-load over % | Low-load over? | Recommendation |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| candidate_asym_weighted_huber_delta10 | asym_weighted_huber | 10.0 | 4.2933 | 5.8077 | 6.7238 | 1.7479 | 2.7546 | 3.5220 | 4.2636 | 0.4886 | 0.4490 | 67.5862 | -1.0952 | 67.5000 | no | recommended |

## Nonnegative Projection Audit

| Candidate | raw min kW | projected min kW | clipped h1-h6 points | clipped h1 points | max adjustment kW |
| --- | ---: | ---: | ---: | ---: | ---: |
| candidate_asym_weighted_huber_delta10 | -3.9462 | 0.0000 | 70 | 23 | 3.9462 |

## Training Sufficiency Diagnostics

| Candidate | epochs | best epoch | early stop | last-best val loss | post-best min delta | train loss reduction % |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| candidate_asym_weighted_huber_delta10 | 13 | 6 | yes | 0.0007 | 0.0007 | 78.2627 |

## Training Condition Coverage

| Condition | points | point share | windows | window share |
| --- | ---: | ---: | ---: | ---: |
| low_load | 21524 | 0.1798 | 3918 | 0.1964 |
| normal_load | 73714 | 0.6158 | 14086 | 0.7060 |
| high_load_p80 | 24474 | 0.2044 | 5840 | 0.2927 |
| high_load_p90 | 12222 | 0.1021 | 3057 | 0.1532 |
| high_positive_ramp | 8653 | 0.0723 | 6124 | 0.3069 |

## Early 0-3h Similarity

- Early 0-3h h1 points: `343`.
- Training windows: `19952`.
- Near train matches within 5 kW on current/h1/delta: `343` (`1.0000`).
- Median nearest scaled distance: `0.0000`; P90: `0.0000`.

## Artifacts

- Model checkpoint and metadata: `outputs/lstm_721/candidate_asym_weighted_huber_delta10/checkpoints/candidate_asym_weighted_huber_delta10/`.
- Metric table: `outputs/lstm_721/candidate_metrics.csv`.
- Condition metrics: `outputs/lstm_721/condition_metrics.csv`.
- Training diagnostics: `outputs/lstm_721/training_curve_diagnostics.csv`.
- Training condition coverage: `outputs/lstm_721/training_condition_coverage.csv`.
