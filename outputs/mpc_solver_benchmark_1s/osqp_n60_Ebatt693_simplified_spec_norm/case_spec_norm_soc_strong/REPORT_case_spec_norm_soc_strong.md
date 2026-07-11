# 1 s OSQP-QP MPC Benchmark Report

Scope: this is only the 1 s OSQP-QP MPC benchmark on the offline reconstructed load profile. It does not modify the 30 s LSTM-MPC mainline, train LSTM, or train DQN.

## Configuration

- Input parquet: `C:\Users\20883\OneDrive\Desktop\microgrid-mpc-master\outputs\mpc_solver_benchmark_1s\data\test_voyages_spline_1s.parquet`
- Output directory: `outputs\mpc_solver_benchmark_1s\osqp_n60_Ebatt693_simplified_spec_norm\case_spec_norm_soc_strong`
- Data source: natural-clipped cubic-spline reconstructed 1 s load profile from original 30 s real-vessel voyages.
- Data caveat: this is not measured 1 s data and not online prediction evidence.
- Horizon: `60` steps
- Sample time: `1.0 s`
- SOC_ref: `0.55`
- Objective variant: `simplified_normalized_literature_v1`
- Battery capacity: `693.0 kWh`
- Battery power bound: `[-346.5, 346.5] kW`
- Battery normalization denominator P_batt_ref: `346.5 kW`
- Previous benchmark capacity for comparison: `1806.0 kWh`
- SOC response scale versus 1806 kWh: `2.606061x` faster for the same battery power and step time.
- Fuel-cell ramp: `48.0 kW/s = 48.0 kW/step`
- Fuel-cell ramp is a hard constraint in the formal simplified normalized variant.
- SOC band denominator: `0.05`
- Weights: q_h2=`1.0`, q_soc=`5.0`, q_batt=`0.05`, q_ramp=`0.0`, q_terminal=`0.0`
- Max steps per voyage: `None`

## Solver Timing

- success_rate: `0.999989`
- infeasible_count: `0`
- fallback_count: `0`
- mean / median / p90 / p95 / p99 / max solve time: `2.425` / `1.170` / `2.261` / `7.090` / `34.648` / `4022.230` ms
- real_time_factor_mean: `0.002425`
- real_time_factor_p99: `0.034648`
- 1 s real-time gate passed: `False`

## Control Metrics

- total_h2_cost / H2_total_kg: `297.522933`
- SOC min/max/initial/final/final_minus_initial_mean: `0.460886` / `0.589200` / `0.550000` / `0.568482` / `0.018482`
- SOC final_minus_initial_min by voyage: `0.005534`
- SOC mean_abs_deviation: `0.032248`
- battery throughput/discharge/charge: `1957.198127` / `933.770291` / `1023.427836` kWh
- P_batt mean_abs/max/min: `75.733192` / `260.559329` / `-346.518331` kW
- P_batt abs<=1 kW fraction: `0.160562`
- P_fc mean/max/min: `211.760294` / `560.067454` / `-0.000040` kW
- fc_ramp_max / ramp_violation_count: `48.047137` kW/step / `3`
- power_balance_violation_max: `0.003640` kW
- SOC/battery/FC power violation counts: `0` / `0` / `0`
- battery/FC power violation max residuals: `0.018331` / `0.067454` kW; count tolerance `0.1 kW`

## Validity Decision

- Current weights are `invalid`.
- Failure categories: `solver_too_slow;constraint_violation_failed`
- Need q_soc/q_batt/q_h2/terminal changes: Weight changes are required before accepting the fixed baseline; this report does not auto-select new final weights.
- Continue N=180: Do not promote directly to N=180 as the next formal result until the current invalid reasons are handled; N=180 may still be run only as a solver-scaling diagnostic.
- Enter DQN-MPC preparation: Do not enter DQN-MPC preparation from this run; fixed QP-MPC behavior is not accepted yet.
- Sensitivity output: not run in this base report.

Invalid reasons:
- timing gate failed: mean=2.425 ms, p95=7.090 ms, p99=34.648 ms, max=4022.230 ms
- constraint check failed: balance_max=0.003640 kW, SOC_count=0, battery_count=0, FC_count=0, ramp_count=3

## Output Files

- `solver_benchmark_summary.csv`
- `solver_benchmark_by_voyage.csv`
- `solver_timing_distribution.csv`
- `solver_failure_cases.csv`
- `constraint_violation_summary.csv`
- `control_performance_summary.csv`
- `objective_term_summary.csv`
- `solver_config.json`
- `figures/soc_trajectory_by_voyage.png`
- `figures/power_split_by_voyage.png`
- `figures/load_fc_batt_overlay_by_voyage.png`
- `figures/fc_power_by_voyage.png`
- `figures/batt_power_by_voyage.png`
- `figures/fc_ramp_by_voyage.png`
- `figures/objective_terms_by_voyage.png`
- `figures/solve_time_cdf.png`
- `figures/solve_time_boxplot.png`
