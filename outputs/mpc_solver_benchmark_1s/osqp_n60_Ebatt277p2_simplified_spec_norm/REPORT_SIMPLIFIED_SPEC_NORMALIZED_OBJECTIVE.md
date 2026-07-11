# Simplified Spec-Normalized Objective 1 s OSQP-QP MPC Report

Scope: offline 1 s natural-clipped cubic-spline benchmark only. This is not measured 1 s data and not online LSTM forecasting evidence.

## Files

- Input parquet: `C:\Users\20883\OneDrive\Desktop\microgrid-mpc-master\outputs\mpc_solver_benchmark_1s\data\test_voyages_spline_1s.parquet`
- Output directory: `outputs\mpc_solver_benchmark_1s\osqp_n60_Ebatt277p2_simplified_spec_norm`
- Objective check: `outputs\mpc_solver_benchmark_1s\osqp_n60_Ebatt277p2_simplified_spec_norm\simplified_normalized_objective_check.md`
- Load feasibility check: `outputs\mpc_solver_benchmark_1s\osqp_n60_Ebatt277p2_simplified_spec_norm\load_feasibility_check.md`
- Cleanup report: `outputs\mpc_solver_benchmark_1s\osqp_n60_Ebatt277p2_simplified_spec_norm\code_cleanup_report.md`
- Candidate summary: `outputs\mpc_solver_benchmark_1s\osqp_n60_Ebatt277p2_simplified_spec_norm\simplified_spec_norm_candidate_summary.csv`
- Candidate by-voyage summary: `outputs\mpc_solver_benchmark_1s\osqp_n60_Ebatt277p2_simplified_spec_norm\simplified_spec_norm_candidate_by_voyage.csv`
- Candidate decision: `outputs\mpc_solver_benchmark_1s\osqp_n60_Ebatt277p2_simplified_spec_norm\simplified_spec_norm_candidate_decision.md`

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

## Required Answers

1. `P_batt_ref` uses `138.6 kW` because the specification-book ratio is about `900 kW / 1806 kWh = 0.5C`; the scaled `277.2 kWh` framework capacity therefore maps to `138.6 kW`.
2. `350 kW` is not used as the formal power denominator because it is inconsistent with the 0.5C scaling. It remains only as a historical legacy assumption in earlier outputs.
3. `P_batt_max` is also `138.6 kW` so that the physical bound and the normalized battery denominator use the same scaled battery system basis.
4. The fuel-cell ramp soft penalty is deleted because ramp is already enforced by hard constraints and the requested formal objective keeps only H2, SOC, and battery-use terms.
5. The terminal SOC penalty is deleted because this formal baseline uses stage SOC maintenance only; no terminal SOC soft penalty or terminal SOC constraint is introduced.
6. Ramp is retained through hard constraints: `|P_fc[0] - P_fc_prev| <= 48` and `|P_fc[k] - P_fc[k-1]| <= 48`.
7. The objective matches a common normalized three-term literature form: H2 term, SOC maintenance term, and battery power penalty.
8. Denominators are fixed physical scales: `P_fc_max=560 kW`, `P_batt_ref=138.6 kW`, `SOC_band=0.05`, and `m_H2_ref=alpha*560^2+beta*560`.
9. The normalized problem remains a convex QP because all quadratic weights and denominators are nonnegative fixed constants and all constraints are linear.
10. Load feasibility under `P_fc_max + P_batt_max = 698.6 kW`: max load `820.134823 kW`, p99 `689.697188 kW`, exceedance count `683`.
11. Most suitable case by the current gates: `NONE_ACCEPTED`; diagnostic case if none accepted: `case_spec_norm_h2_low_fc_main`.
12. Selected/diagnostic battery active fraction `abs(P_batt)>5 kW`: `0.902151`.
13. Selected/diagnostic SOC min/max: `0.467405` / `0.549951`.
14. Selected/diagnostic FC energy share: `1.431227`.
15. Selected/diagnostic OSQP success and p99 solve time: `0.983791` / `1.162 ms`.
16. Recommended fixed MPC baseline: `NONE_ACCEPTED` with label `FAIL_POWER_LIMIT_INSUFFICIENT`.
17. Usable before DQN dynamic weighting: `False`. If false, do not proceed to DQN from this benchmark.
18. The result is still based on offline spline 1 s reconstruction, not true measured 1 s data.

## Decision

- recommended_fixed_mpc_baseline_before_dqn: `NONE_ACCEPTED`
- least_bad_diagnostic_case: `case_spec_norm_h2_low_fc_main`
- accepted: `False`
- selected_label: `FAIL_POWER_LIMIT_INSUFFICIENT`
- reason: No candidate passed all physical baseline gates; least-bad diagnostic label is FAIL_POWER_LIMIT_INSUFFICIENT.

Voyage rows included in aggregate by-voyage table: `56`.
