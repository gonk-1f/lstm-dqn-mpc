# Reference Hyperparameter Evidence

This table is for hyperparameter clues only. It is not a literature review and not evidence that spline-reconstructed 1 s ship load is valid online ground truth.

| paper_id | usable_hparam_clue | limitations |
|---|---|---|
| ref1 | use physical-time history windows and evaluate the whole horizon | smart-home appliance data, not shipboard spline-reconstructed 1s load |
| ref2 | compare LSTM against naive baselines after interpolation | hourly sparse load, not measured 1s ship load |
| ref3 | LSTM is applicable to electricity time series, but task-specific windows are required | NILM task, not future load forecasting |
| ref4 | separate receptive/history field from target/forecast field | disaggregation/detection task, not load forecasting |
| ref5 | spline interpolation can be used as offline smoothing/filling before forecasting | not electricity load and not LSTM |
| ref6 | if verified as 1s and Np=5, it only supports short-horizon settings | does not replace h60/h180 evaluation and requires direct paper verification before strong claims |
