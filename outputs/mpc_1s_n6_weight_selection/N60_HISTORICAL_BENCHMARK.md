# Historical N=60 OSQP benchmark pointer

Status: historical offline solver/performance benchmark only. It is not the default MPC configuration, is not accepted as a paper weight selection, and was not rerun or searched in the N=6 task.

## Retained source artifacts

- N=60 anchor configuration (`q_h2=0.5`, `q_soc=2.0`, `q_batt=0.05`, `SOC_band=0.05`): `outputs/mpc_solver_benchmark_1s/osqp_n60_Ebatt693_simplified_spec_norm/case_spec_norm_h2_low_fc_main/solver_config.json`
- QP formulation check: `outputs/mpc_solver_benchmark_1s/osqp_n60_Ebatt693_simplified_spec_norm/case_spec_norm_h2_low_fc_main/qp_formulation_check.md`
- OSQP timing summary: `outputs/mpc_solver_benchmark_1s/osqp_n60_Ebatt693_simplified_spec_norm/case_spec_norm_h2_low_fc_main/solver_benchmark_summary.csv`
- N=60 candidate summary: `outputs/mpc_solver_benchmark_1s/osqp_n60_Ebatt693_simplified_spec_norm/simplified_spec_norm_candidate_summary.csv`
- Historical decision report: `outputs/mpc_solver_benchmark_1s/osqp_n60_Ebatt693_simplified_spec_norm/simplified_spec_norm_candidate_decision.md`

The detailed N=60 trees remain untouched for a later repository-cleanup task.

## Anchor snapshot and N=6 migration result

| metric | historical N=60 anchor | N=6 candidate A |
|---|---:|---:|
| solver success / attempted | 0.999968 | 0.999985 |
| p99 solve time | 10.259 ms | 0.154 ms in the final recorded run |
| worst-voyage SOC net decrease | -0.016661 | -0.349873 |
| closed-loop coverage | historical runner reported the full data pass | 0.723573 |
| battery charge energy | 919.327 kWh | 303.076 kWh on the applied prefix only |
| battery discharge energy | 898.296 kWh | 1502.014 kWh on the applied prefix only |
| battery throughput | 1817.623 kWh | 1805.090 kWh on the applied prefix only |

The N=6 anchor is much faster, but it loses long-horizon SOC sustainability and reaches a hard infeasible window on `voyage_063`. Its energy figures cover only the successfully applied prefix and must not be used as a full-voyage economy comparison.

This comparison is diagnostic rather than an isolated horizon-only ablation: the new N=6 runner also enforces the corrected `t+1..t+6` ideal-foresight timing, executes only the first action, updates SOC from actual battery power, uses explicit numerical scaling, and terminates at the first final solver failure. The historical N=60 runner is preserved as recorded.
