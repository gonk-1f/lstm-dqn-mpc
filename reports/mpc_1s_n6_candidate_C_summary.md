# Candidate C final test summary

## Reproducibility record

- Source content SHA-256: `b31b0bdf786c4a9be119098108bdc3a34412bfedf2f9babfc0a419ba878a3623`
- Active split: `outputs/config/voyage_split_total_load_721.json`
- Active split SHA-256: `7ee01bd0e032c7624c051051105b6296cc8c88bee9b7ff2f49a535f71a80ea36`
- MPC input: `outputs/mpc_solver_benchmark_1s/data/test_voyages_spline_1s.parquet`
- MPC input SHA-256: `196ddc059ebea141408bf2e46fc87a12bd97cf747ccdd423627f75c19fee7256`
- Actual test voyages: `voyage_061`, `voyage_063`, `voyage_064`, `voyage_065`, `voyage_066`
- Fixed weights: `q_h2=0.25`, `q_batt=0.4`, `q_soc=12.0`, `q_fc_var=20.0`

The split is the existing chronological 7:2:1 voyage-level method applied to the 50 eligible voyages after the 16 audited abnormal voyages were excluded. It contains 35 train, 10 validation, and 5 test voyages. No random seed is applicable to this chronological allocation.

## Results

| Voyage | Complete / failures | H2 (kg) | final SOC | minimum SOC | max discharge (kW) | max charge magnitude (kW) | FC squared-step-change sum (kW²) |
|---|---:|---:|---:|---:|---:|---:|---:|
| voyage_061 | yes / 0 | 27.311043 | 0.543055 | 0.494881 | 225.941 | 120.727 | 136.366 |
| voyage_063 | yes / 0 | 129.501712 | 0.551926 | 0.450367 | 316.991 | 325.127 | 1032.940 |
| voyage_064 | yes / 0 | 14.768787 | 0.550836 | 0.497811 | 338.751 | 278.066 | 274.338 |
| voyage_065 | yes / 0 | 26.664694 | 0.538851 | 0.466900 | 305.757 | 346.500 | 570.108 |
| voyage_066 | yes / 0 | 29.626492 | 0.545680 | 0.476710 | 273.021 | 339.856 | 521.034 |

All five voyages completed. Solver failures, primal-infeasible statuses, and maximum-iteration statuses are all zero. Total hydrogen use is `227.872727 kg`; the global SOC range is `0.450367–0.556894`.

The maximum power-balance residual is `5.684e-14 kW`, the maximum FC step change is `2.7515 kW` against the retained `48 kW/step` bound, and the battery extrema remain inside the existing tolerance around `±346.5 kW`. The configuration-level mean and p95 solve times are `0.0611 ms` and `0.0725 ms` respectively.

The complete machine-readable provenance, model parameters, tolerances, hashes, and voyage list are also stored in the final Candidate C `config.json`. This remains an offline ideal-foresight N=6 result using future 6 s natural-clipped load values; no LSTM or DQN is used.
