# Simplified Spec-Normalized Candidate Decision

Scope: fixed-weight 1 s OSQP-QP MPC benchmark on offline natural-clipped spline load reconstruction. No DQN, LSTM, 30 s mainline, or CasADi/IPOPT baseline changes.

- Formal objective variant: `simplified_normalized_literature_v1`
- Battery capacity: `277.2 kWh`
- Battery power bound and denominator: `138.6 kW`
- Fuel-cell ramp hard constraint: `48 kW/s = 48.0 kW/step`
- Objective check: `outputs\mpc_solver_benchmark_1s\osqp_n60_Ebatt277p2_simplified_spec_norm\simplified_normalized_objective_check.md`
- Load feasibility check: `outputs\mpc_solver_benchmark_1s\osqp_n60_Ebatt277p2_simplified_spec_norm\load_feasibility_check.md`
- Code cleanup report: `outputs\mpc_solver_benchmark_1s\osqp_n60_Ebatt277p2_simplified_spec_norm\code_cleanup_report.md`

## Candidate Table

| case | label | success | mean ms | p99 ms | H2 kg | batt throughput kWh | active >5kW | SOC drop max | FC share |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| case_spec_norm_base | FAIL_POWER_LIMIT_INSUFFICIENT | 0.983716 | 0.436 | 2.516 | 393.873189 | 2446.845672 | 0.896517 | 0.078936 | 1.344849 |
| case_spec_norm_more_batt | FAIL_POWER_LIMIT_INSUFFICIENT | 0.978847 | 0.242 | 0.913 | 319.379747 | 2202.141511 | 0.859074 | 0.075772 | 1.106715 |
| case_spec_norm_batt_conservative | FAIL_POWER_LIMIT_INSUFFICIENT | 0.982985 | 0.264 | 1.051 | 323.383116 | 1214.469366 | 0.859820 | 0.068228 | 1.115059 |
| case_spec_norm_soc_safe | FAIL_POWER_LIMIT_INSUFFICIENT | 0.983695 | 0.293 | 1.095 | 420.282425 | 2827.363669 | 0.902153 | 0.076513 | 1.426095 |
| case_spec_norm_h2_low_fc_main | FAIL_POWER_LIMIT_INSUFFICIENT | 0.983791 | 0.292 | 1.162 | 421.851750 | 2829.424302 | 0.902151 | 0.064797 | 1.431227 |
| case_spec_norm_h2_high_economy | FAIL_POWER_LIMIT_INSUFFICIENT | 0.980685 | 0.236 | 0.904 | 339.714979 | 1984.771524 | 0.891166 | 0.078700 | 1.171291 |
| case_spec_norm_more_batt_soc_safe | FAIL_POWER_LIMIT_INSUFFICIENT | 0.983727 | 0.304 | 1.261 | 419.814980 | 2832.809121 | 0.902221 | 0.072212 | 1.424553 |
| case_spec_norm_soc_strong | FAIL_POWER_LIMIT_INSUFFICIENT | 0.983845 | 0.434 | 2.179 | 421.850820 | 2836.694517 | 0.902233 | 0.071350 | 1.431112 |

## Decision

- recommended_fixed_mpc_baseline_before_dqn: `NONE_ACCEPTED`
- least_bad_diagnostic_case: `case_spec_norm_h2_low_fc_main`
- accepted: `False`
- selected_label: `FAIL_POWER_LIMIT_INSUFFICIENT`
- reason: No candidate passed all physical baseline gates; least-bad diagnostic label is FAIL_POWER_LIMIT_INSUFFICIENT.

No global config is modified by this decision file.
