# Candidate D constraint audit

The audit uses actual closed-loop quantities after applying only the first QP action.

## Numerical tolerance

Small solver residuals are classified against explicit tolerances; a small value such as 0.0154 kW is a numerical tolerance issue, not automatically a physical strategy failure.

| tolerance | value |
|---|---:|
| `actual_balance_kw` | 0.01 |
| `qp_balance_kw` | 0.1 |
| `power_bound_kw` | 0.1 |
| `ramp_kw` | 0.1 |
| `soc` | 1e-06 |
| `soc_prediction` | 1e-05 |
| `fc_above_load_kw` | 1 |
| `near_limit_kw` | 1 |

## Overall raw maxima

| metric | value |
|---|---:|
| `max_actual_power_balance_residual_kw` | 5.684341886080802e-14 |
| `max_plan_power_balance_residual_kw` | 0.00041820727017238823 |
| `max_fc_bound_residual_kw` | 0.0035812046933187958 |
| `max_battery_bound_residual_kw` | 2.2737367544323206e-13 |
| `max_ramp_residual_kw` | 0.0 |
| `max_soc_bound_residual` | 0.0 |
| `max_soc_prediction_residual` | 9.570686823057173e-07 |
| `physical_infeasible_point_count` | 0 |
| `closed_loop_complete` | False |
| `closed_loop_coverage_fraction` | 0.7352036977319144 |

`physical_infeasible_point_count` counts only residuals beyond the configured physical tolerances. Solver failures remain separately visible in solver_statistics.csv.
