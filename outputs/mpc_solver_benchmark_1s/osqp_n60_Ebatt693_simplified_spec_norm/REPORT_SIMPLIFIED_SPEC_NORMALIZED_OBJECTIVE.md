# Simplified Spec-Normalized Objective 1 s OSQP-QP MPC Report

Scope: offline 1 s natural-clipped cubic-spline benchmark only. This is not measured 1 s data and not online LSTM forecasting evidence.

## Files

- Input parquet: `C:\Users\20883\OneDrive\Desktop\microgrid-mpc-master\outputs\mpc_solver_benchmark_1s\data\test_voyages_spline_1s.parquet`
- Output directory: `outputs\mpc_solver_benchmark_1s\osqp_n60_Ebatt693_simplified_spec_norm`
- Objective check: `outputs\mpc_solver_benchmark_1s\osqp_n60_Ebatt693_simplified_spec_norm\simplified_normalized_objective_check.md`
- Load feasibility check: `outputs\mpc_solver_benchmark_1s\osqp_n60_Ebatt693_simplified_spec_norm\load_feasibility_check.md`
- Cleanup report: `outputs\mpc_solver_benchmark_1s\osqp_n60_Ebatt693_simplified_spec_norm\code_cleanup_report.md`
- Candidate summary: `outputs\mpc_solver_benchmark_1s\osqp_n60_Ebatt693_simplified_spec_norm\simplified_spec_norm_candidate_summary.csv`
- Candidate by-voyage summary: `outputs\mpc_solver_benchmark_1s\osqp_n60_Ebatt693_simplified_spec_norm\simplified_spec_norm_candidate_by_voyage.csv`
- Candidate decision: `outputs\mpc_solver_benchmark_1s\osqp_n60_Ebatt693_simplified_spec_norm\simplified_spec_norm_candidate_decision.md`

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

## Required Answers

1. `P_batt_ref` uses `346.5 kW` because the current formal pack basis is `10 x 69.3 kWh = 693 kWh` and `0.5C` gives `346.5 kW`.
2. The old `138.6 kW` denominator came from the previous `277.2 kWh` scaled pack and is legacy only for this formal run.
3. `P_batt_max` is also `346.5 kW` so that the physical bound and the normalized battery denominator use the same scaled battery system basis.
4. The fuel-cell ramp soft penalty is deleted because ramp is already enforced by hard constraints and the requested formal objective keeps only H2, SOC, and battery-use terms.
5. The terminal SOC penalty is deleted because this formal baseline uses stage SOC maintenance only; no terminal SOC soft penalty or terminal SOC constraint is introduced.
6. Ramp is retained through hard constraints: `|P_fc[0] - P_fc_prev| <= 48` and `|P_fc[k] - P_fc[k-1]| <= 48`.
7. The objective matches a common normalized three-term literature form: H2 term, SOC maintenance term, and battery power penalty.
8. Denominators are fixed physical scales: `P_fc_max=560 kW`, `P_batt_ref=346.5 kW`, `SOC_band=0.05`, and `m_H2_ref=alpha*560^2+beta*560`.
9. The normalized problem remains a convex QP because all quadratic weights and denominators are nonnegative fixed constants and all constraints are linear.
10. Load feasibility under `P_fc_max + P_batt_max = 906.5 kW`: max load `820.134823 kW`, p99 `689.697188 kW`, exceedance count `0`.
11. Most suitable case by the current gates: `NONE_ACCEPTED`; diagnostic case if none accepted: `case_spec_norm_more_batt`.
12. Selected/diagnostic battery active fraction `abs(P_batt)>5 kW`: `0.751701`.
13. Selected/diagnostic SOC min/max: `0.390202` / `0.550000`.
14. Selected/diagnostic FC energy share: `0.943742`.
15. Selected/diagnostic OSQP success and p99 solve time: `1.000000` / `58.856 ms`.
16. Recommended fixed MPC baseline: `NONE_ACCEPTED` with label `FAIL_SOC_DROP`.
17. Usable before DQN dynamic weighting: `False`. If false, do not proceed to DQN from this benchmark.
18. The result is still based on offline spline 1 s reconstruction, not true measured 1 s data.

## Decision

- recommended_fixed_mpc_baseline_before_dqn: `NONE_ACCEPTED`
- least_bad_diagnostic_case: `case_spec_norm_more_batt`
- accepted: `False`
- selected_label: `FAIL_SOC_DROP`
- reason: No candidate passed all physical baseline gates; least-bad diagnostic label is FAIL_SOC_DROP.

Voyage rows included in aggregate by-voyage table: `56`.
