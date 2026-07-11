# Simplified Spec-Normalized Candidate Decision

Scope: fixed-weight 1 s OSQP-QP MPC benchmark on offline natural-clipped spline load reconstruction. No DQN, LSTM, 30 s mainline, or CasADi/IPOPT baseline changes.

- Formal objective variant: `simplified_normalized_literature_v1`
- Battery capacity: `693.0 kWh`
- Battery power bound and denominator: `346.5 kW`
- Fuel-cell ramp hard constraint: `48 kW/s = 48.0 kW/step`
- Objective check: `outputs\mpc_solver_benchmark_1s\osqp_n60_Ebatt693_simplified_spec_norm\simplified_normalized_objective_check.md`
- Load feasibility check: `outputs\mpc_solver_benchmark_1s\osqp_n60_Ebatt693_simplified_spec_norm\load_feasibility_check.md`
- Code cleanup report: `outputs\mpc_solver_benchmark_1s\osqp_n60_Ebatt693_simplified_spec_norm\code_cleanup_report.md`

## Candidate Table

| case | label | success | mean ms | p99 ms | H2 kg | batt throughput kWh | active >5kW | SOC drop max | FC share |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| case_spec_norm_base | FAIL_SOC_DROP | 1.000000 | 3.434 | 56.413 | 272.900170 | 1254.324621 | 0.752152 | 0.084364 | 0.940858 |
| case_spec_norm_more_batt | FAIL_SOC_DROP | 1.000000 | 4.471 | 58.856 | 274.210938 | 1372.517707 | 0.751701 | 0.079435 | 0.943742 |
| case_spec_norm_batt_conservative | FAIL_CONSTRAINT | 1.000000 | 1.923 | 19.809 | 271.574541 | 1220.545864 | 0.767028 | 0.083303 | 0.937252 |
| case_spec_norm_soc_safe | FAIL_CONSTRAINT | 0.999936 | 2.240 | 42.477 | 284.362628 | 1803.636475 | 0.794069 | 0.089677 | 0.973292 |
| case_spec_norm_h2_low_fc_main | FAIL_CONSTRAINT | 0.999968 | 0.767 | 10.259 | 293.873401 | 1817.623008 | 0.796709 | 0.016661 | 1.003862 |
| case_spec_norm_h2_high_economy | FAIL_SOC_DROP | 1.000000 | 2.946 | 45.942 | 274.770115 | 1290.672854 | 0.747208 | 0.081720 | 0.947516 |
| case_spec_norm_more_batt_soc_safe | FAIL_SOLVER | 0.999936 | 2.910 | 39.928 | 283.509960 | 1885.219214 | 0.781772 | 0.090767 | 0.969807 |
| case_spec_norm_soc_strong | FAIL_SOLVER | 0.999989 | 2.425 | 34.648 | 297.522933 | 1957.198127 | 0.792360 | 0.000000 | 1.016645 |

## Decision

- recommended_fixed_mpc_baseline_before_dqn: `NONE_ACCEPTED`
- least_bad_diagnostic_case: `case_spec_norm_more_batt`
- accepted: `False`
- selected_label: `FAIL_SOC_DROP`
- reason: No candidate passed all physical baseline gates; least-bad diagnostic label is FAIL_SOC_DROP.

No global config is modified by this decision file.
