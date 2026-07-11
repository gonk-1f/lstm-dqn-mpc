# 1 s OSQP-QP MPC Benchmark Report

Scope: this is only the 1 s OSQP-QP MPC benchmark on the offline reconstructed load profile. It does not modify the 30 s LSTM-MPC mainline, train LSTM, or train DQN.

## Configuration

- Input parquet: `C:\Users\20883\OneDrive\Desktop\microgrid-mpc-master\outputs\mpc_solver_benchmark_1s\data\test_voyages_spline_1s.parquet`
- Output directory: `outputs\mpc_solver_benchmark_1s\osqp_n60_Ebatt277p2_simplified_spec_norm\case_spec_norm_more_batt`
- Data source: natural-clipped cubic-spline reconstructed 1 s load profile from original 30 s real-vessel voyages.
- Data caveat: this is not measured 1 s data and not online prediction evidence.
- Horizon: `60` steps
- Sample time: `1.0 s`
- SOC_ref: `0.55`
- Objective variant: `simplified_normalized_literature_v1`
- Battery capacity: `277.2 kWh`
- Battery power bound: `[-138.6, 138.6] kW`
- Battery normalization denominator P_batt_ref: `138.6 kW`
- Previous benchmark capacity for comparison: `1806.0 kWh`
- SOC response scale versus 1806 kWh: `6.515152x` faster for the same battery power and step time.
- Fuel-cell ramp: `48.0 kW/s = 48.0 kW/step`
- Fuel-cell ramp is a hard constraint in the formal simplified normalized variant.
- SOC band denominator: `0.05`
- Weights: q_h2=`1.0`, q_soc=`1.0`, q_batt=`0.01`, q_ramp=`0.0`, q_terminal=`0.0`
- Max steps per voyage: `None`

## Solver Timing

- success_rate: `0.978847`
- infeasible_count: `1966`
- fallback_count: `0`
- mean / median / p90 / p95 / p99 / max solve time: `0.242` / `0.186` / `0.323` / `0.448` / `0.913` / `20.351` ms
- real_time_factor_mean: `0.000242`
- real_time_factor_p99: `0.000913`
- 1 s real-time gate passed: `False`

## Control Metrics

- total_h2_cost / H2_total_kg: `319.379747`
- SOC min/max/initial/final/final_minus_initial_mean: `0.459295` / `0.549955` / `0.550000` / `0.493334` / `-0.056666`
- SOC final_minus_initial_min by voyage: `-0.075772`
- SOC mean_abs_deviation: `0.044950`
- battery throughput/discharge/charge: `2202.141511` / `637.581793` / `1564.559718` kWh
- P_batt mean_abs/max/min: `87.051680` / `138.657576` / `-138.647080` kW
- P_batt abs<=1 kW fraction: `0.105612`
- P_fc mean/max/min: `235.500320` / `560.057647` / `-0.000137` kW
- fc_ramp_max / ramp_violation_count: `48.000000` kW/step / `0`
- power_balance_violation_max: `0.002876` kW
- SOC/battery/FC power violation counts: `0` / `0` / `0`
- battery/FC power violation max residuals: `0.057576` / `0.057647` kW; count tolerance `0.1 kW`

## Validity Decision

- Current weights are `invalid`.
- Failure categories: `power_limit_insufficient;solver_feasibility_failed;solver_too_slow;SOC_sustain_failed;battery_power_limit_sticking`
- Need q_soc/q_batt/q_h2/terminal changes: Weight changes are required before accepting the fixed baseline; this report does not auto-select new final weights.
- Continue N=180: Do not promote directly to N=180 as the next formal result until the current invalid reasons are handled; N=180 may still be run only as a solver-scaling diagnostic.
- Enter DQN-MPC preparation: Do not enter DQN-MPC preparation from this run; fixed QP-MPC behavior is not accepted yet.
- Sensitivity output: not run in this base report.

Invalid reasons:
- load_exceeds_power_limit_count=683 above P_available_max=698.600000 kW
- solver_success_rate=0.978847, infeasible_count=1966
- timing gate failed: mean=0.242 ms, p95=0.448 ms, p99=0.913 ms, max=20.351 ms
- SOC_final_minus_initial_min=-0.075772 < -0.02
- fraction(abs(P_batt)>=95% limit)=0.437416 > 0.05

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
