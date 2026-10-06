# Earlier experimental LSTM–LPF–economic MPC–Double DQN chain

The active v3 path now uses current-load persistence. See `v3_persistence_dqn_mpc.md`. This file preserves the earlier LSTM experiment and its diagnostics.

This is a new experimental `v3` path. The existing `v2` method and its checkpoints remain frozen. The `v3` path currently handles a contiguous ONBOARD interval; formal shore-mode training/evaluation is not wired to it.

## Causal clock

At decision time `k`, real load `L_k`, real SOC, executed FC power, committed LPF state, and the cost from `k-1 → k` are known. The previous cost belongs to action `a_(k-1)` and closes its replay transition. The newly revealed one-step error is `e_k = L_k - Lhat_(k|k-1)`.

The LSTM then predicts `Lhat_(k+1:k+5|k)` using only real history through `k`. The 15-value DQN state is, in order:

`[SOC_k, L_k/600, F_k/600, P_FC,k/600, ΔP_FC,k/600, Lhat_(k+1:k+5|k)/600, e_(k:k-2)/600, lambda_ref,k-1, lambda_SOC,k-1]`.

Startup errors and the previous action are zero-padded. The DQN selects independent nonnegative `(lambda_ref,k, lambda_SOC,k)`. The LPF previews the five forecast loads from the committed `F_k`; that preview never changes the committed filter. The economic MPC solves five future FC powers and executes only the first. At `k+1`, real load is revealed, the battery absorbs `L_(k+1) - Lhat_(k+1|k)`, and the exact physical ledger gives `r_k = -C_actual,k+1`. The replay tuple `(s_k, a_k, r_k, s_(k+1))` is available before the next action is selected.

`PredictiveController.start_decision(policy)` and `finish_decision(actual_load_kw)` expose the two clock phases separately. `DoubleDQNAgent.remember_transition()` and `learn()` are called after `finish_decision()` and before the next `start_decision()`. The action catalog and the CNY reward scale must be supplied from Train work; old v2 36-action weights and checkpoints do not match this state or objective.

## Predicted and actual ledgers

Both ledgers call the same formal v2 physical cost functions. For the first planned step, predicted and executed FC power are identical under the perfect FC command-tracking assumption. Therefore first-step hydrogen cost and full FC degradation cost, including actual start/stop, are exactly equal in both ledgers. The predicted battery power uses `Lhat_(k+1|k) - P_FC`; the actual battery power uses `L_(k+1) - P_FC`, so its SOC and degradation can differ. Only the executed ledger becomes the DQN reward.

The SLSQP objective uses a continuous FC degradation proxy because the exact startup cost jumps at zero FC power. The plan ledger still reports the exact formal cost. The objective also uses hydrogen cost, battery degradation cost, LPF tracking, and continuous SOC-center penalties. `nominal_cost_cny` is an explicit positive input; no formal Train calibration or action-catalog scan has been completed for v3.

## LSTM input length and observed result

At 30 s sampling, the candidates `H=6,12,24` represent 3, 6, and 12 minutes of real history. Every model directly outputs five future samples (150 s). The selected input is the 24 real load observations through `k`, shaped `[batch,24,1]`; the output is the five loads at `k+1` through `k+5`, shaped `[batch,5]`. Neither AIS nor speed is an LSTM input. Train windows stay inside one ONBOARD run and one episode; Validation comparisons use the same 2,683 forecast origins for each `H`.

The model has one LSTM layer with 32 hidden units and a linear five-output head. It predicts a residual added to the last observed load. Train mean and standard deviation are 228.883 and 226.954 kW, fitted only on Train ONBOARD loads. Training uses seed 42, Adam learning rate `1e-3`, batch size 256, normalized Smooth L1 loss, at most 20 epochs, and early stopping patience 4. Negative denormalized predictions are clipped to zero at inference and Validation scoring. The best epoch was 1 for each candidate.

The persistence baseline is `Lhat_(k+h|k)=L_k` for each `h=1,...,5`; it is scored on the same Validation forecast origins. The equal-origin five-step Validation results are:

| Predictor | WAPE | MAE (kW) | RMSE (kW) |
| --- | ---: | ---: | ---: |
| Persistence | 10.470% | 35.681 | 86.155 |
| LSTM H=6 | 10.765% | 36.683 | 85.026 |
| LSTM H=12 | 10.865% | 37.024 | 84.939 |
| LSTM H=24 | 10.720% | 36.531 | 85.116 |

`H=24` minimizes five-step Validation WAPE among the LSTM candidates, but its WAPE skill against persistence is **-2.384%**. It does not beat persistence on WAPE or MAE. The LSTM's RMSE is slightly lower. On identical nonnegative-load cases, WAPE is MAE divided by the mean actual load, times 100, so WAPE and MAE rank these candidates identically. The report also records five separate horizon scores and signed bias. Overall MAPE is undefined because one of the 13,415 Validation target values is zero; it is reported as `null` rather than silently discarding that case or adding an arbitrary denominator epsilon. The checkpoint is an experimental artifact, not a validated formal forecasting component. See `outputs/v3_lstm/selection.json` for the raw report.

The LSTM and formal v2 DQN use the same three frozen dataset roots and episode split. The LSTM uses only contiguous ONBOARD load windows (16,690 Train windows for selected `H=24`); v2 DQN uses event-driven episodes and a different state. The v3 Double DQN implementation is not yet trained, so no claim about an identically trained v3 policy is made.

The user-requested Test diagnostic was generated after freezing this checkpoint and Validation selection. It is separate from the sealed formal v2 final evaluation and must not be used to tune this v3 model. `outputs/v3_lstm_test_diagnostic/diagnostic.json` contains manifest and checkpoint hashes, per-episode horizon metrics and plot filenames. Each plot places the forecast made at origin `k` for horizon `h` at target time `k+h`. One-step change cross-correlation is maximal at a positive one-sample shift (30 s) in all five Test episodes; five-step predictions show about five samples (150 s) of shift. This is predictive phase lag, not plot indexing or a measured compute latency. The selected residual model changes persistence predictions by only about 1.2–1.6 kW on average across these episodes, so abrupt load changes are mostly reflected only after observation.

The original [LSTM paper](https://doi.org/10.1162/neco.1997.9.8.1735) motivates recurrent memory; [multiple-output multi-step forecasting](https://doi.org/10.1016/j.neucom.2009.11.030) motivates predicting a vector rather than recursively feeding predictions back; a [short-term load forecasting study](https://link.springer.com/article/10.1007/s44196-022-00128-y) formulates historical sliding windows and future multi-step targets. Their window lengths do not transfer directly to 30 s vessel data, so the project selected `H` on its own Train/Validation split.

## Reproduce

```powershell
$env:PYTHONPATH='src'
python -m v3.training --epochs 20 --patience 4 --output-dir outputs/v3_lstm
python -m unittest tests.test_v3_predictive_chain tests.test_v3_double_dqn tests.test_v3_forecast_metrics tests.test_v3_test_diagnostic -v
```

The active persistence path now has per-step shore settlement and a terminal-transition API, documented in `v3_persistence_dqn_mpc.md`. Train-calibrated `C_nom`, action-catalog screening, an end-to-end formal episode trainer and closed-loop Validation still remain before formal claims.
