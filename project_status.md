# Project Status

Updated: 2026-07-11 after diagnosing the unchanged mirror-like power split

## Current Mainline

- The active measured-data path remains the original 30 s real-vessel total-load voyages in `total_load_excels`.
- The active 30 s LSTM and CasADi/IPOPT LSTM-MPC mainline was not modified by this work.
- DQN/KAN-DQN has not been trained.

## Formal 1 s Offline OSQP-QP Setup

- Input: `outputs/mpc_solver_benchmark_1s/data/test_voyages_spline_1s.parquet`.
- Output root: `outputs/mpc_solver_benchmark_1s/osqp_n60_Ebatt693_simplified_spec_norm/`.
- Test set: 7 voyages (`voyage_060` through `voyage_066`), 93,037 rows.
- Horizon/sample time: `N=60`, `dt=1 s`.
- Battery capacity: `E_batt=693 kWh`.
- Continuous battery power rate: `0.5C`.
- Battery charge/discharge bound and normalization reference: `P_batt_max=P_batt_ref=346.5 kW`.
- Fuel-cell bound: `P_fc in [0, 560] kW`.
- Total available power: `906.5 kW`; maximum reconstructed test load is `820.134823 kW`, so no test row exceeds the combined power limit.
- Fuel-cell ramp: hard constraint only, `48 kW/s = 48 kW/step`.
- SOC: `SOC_min=0.2`, `SOC_max=0.8`, `SOC_ref=0.55`, `SOC_band=0.05`.
- Objective: `simplified_normalized_literature_v1`, containing normalized H2, SOC-maintenance, and battery-power terms.
- Solver: OSQP using its ADMM-QP algorithm with fixed sparsity reuse.

The input is an offline natural-clipped cubic-spline reconstruction. It is not measured 1 s data, not an online LSTM forecast, and not online-feasible forecasting evidence.

## Fixed-Weight Results

The original 8 candidates completed under the 693 kWh / 346.5 kW setup. No candidate was automatically accepted by every strict gate. The engineering-preferred provisional candidate remains:

`case_spec_norm_h2_low_fc_main`: `q_h2=0.5`, `q_soc=2.0`, `q_batt=0.05`, `q_ramp=0`, `q_terminal=0`.

An additional isolated candidate was run without changing the default 8-case list:

`case_spec_norm_h2_low_soc_1p5`: `q_h2=0.5`, `q_soc=1.5`, `q_batt=0.05`, `q_ramp=0`, `q_terminal=0`.

| metric | `q_soc=2.0` | `q_soc=1.5` |
|---|---:|---:|
| solver success rate | 0.999968 | 0.999925 |
| infeasible count | 0 | 0 |
| H2 total (kg) | 293.873401 | 293.029013 |
| worst voyage final SOC change | -0.016661 | -0.024871 |
| global SOC min / max | 0.445939 / 0.580935 | 0.442662 / 0.578760 |
| battery throughput (kWh) | 1817.623008 | 1992.451576 |
| battery near-limit fraction | 0.003698 | 0.008890 |
| mean solve time (ms) | 0.767 | 0.636 |
| p99 solve time (ms) | 10.259 | 9.315 |

Reducing `q_soc` to `1.5` is not an improvement: it lowers H2 use by only `0.844388 kg`, while worsening the worst-voyage SOC change past the current `-0.02` sustain gate and increasing battery throughput by `174.828567 kWh`. The new candidate is labeled invalid because of `SOC_sustain_failed` and two small ramp residual violations.

The mirror-like power split also did not improve. On valid solved rows, mean absolute battery power increased from `70.334 kW` to `77.102 kW`, and the correlation between consecutive FC-power changes and battery-power changes became more negative, from `-0.469` to `-0.559`. The equality constraint enforces `P_fc + P_batt = P_load`, so the battery is algebraically the residual `P_load - P_fc`. Changing `q_soc` only changes SOC-restoration pressure and cannot remove that residual relationship. Within the existing objective, `q_batt` is the direct weight for reducing battery-power amplitude; no new `q_batt` value was selected or run.

## Current Decision

- Keep `q_soc=2.0` as the provisional offline fixed-weight candidate.
- Do not promote `q_soc=1.5`.
- Do not continue searching `q_soc` as a remedy for the mirror-like power split.
- Do not change the objective structure or select another weight without explicit authorization.
- No weight set has been written into the 30 s mainline or global MPC configuration.
- Do not proceed to DQN/KAN-DQN from this offline reconstructed-load benchmark alone.

## Current Artifacts

- Original 8-case summary: `outputs/mpc_solver_benchmark_1s/osqp_n60_Ebatt693_simplified_spec_norm/simplified_spec_norm_candidate_summary.csv`.
- Original 8-case decision: `outputs/mpc_solver_benchmark_1s/osqp_n60_Ebatt693_simplified_spec_norm/simplified_spec_norm_candidate_decision.md`.
- Additional `q_soc=1.5` result: `outputs/mpc_solver_benchmark_1s/osqp_n60_Ebatt693_simplified_spec_norm/case_spec_norm_h2_low_soc_1p5/`.
- Additional candidate report: `outputs/mpc_solver_benchmark_1s/osqp_n60_Ebatt693_simplified_spec_norm/case_spec_norm_h2_low_soc_1p5/REPORT_case_spec_norm_h2_low_soc_1p5.md`.
- Additional candidate 7-voyage power plot: `outputs/mpc_solver_benchmark_1s/osqp_n60_Ebatt693_simplified_spec_norm/case_spec_norm_h2_low_soc_1p5/figures/power_split_by_voyage.png`.

## Verification

- `D:\py\Python3\python.exe -m py_compile src\main\benchmark_mpc_qp_osqp_1s.py src\main\mpc_solvers\mpc_qp_formulation.py` passed.
- `D:\py\Python3\python.exe -m unittest tests.test_mpc_solver_benchmark_1s tests.test_mpc_ramp_constraint_toggle` passed: 19 tests.
- The additional run completed all 93,037 rows across all 7 voyages.
