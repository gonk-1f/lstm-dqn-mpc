# LSTM diagnostic skipped

The requested `spline_1s_short_h180_p60` LSTM diagnostic was not trained because the pre-training audit failed.

Stop conditions triggered:
- Both raw cubic-spline versions produced negative `load_total_kw` samples.
- Both raw versions and both clipped versions produced local endpoint overshoot.
- The reconstructed 1 s labels are offline and future-node-dependent, not measured 1 s ground truth.

No 1 s LSTM-MPC control results were generated.
