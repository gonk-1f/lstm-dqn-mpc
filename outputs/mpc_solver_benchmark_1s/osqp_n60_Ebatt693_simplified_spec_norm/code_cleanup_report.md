# 1 s OSQP-QP MPC Code Cleanup Report

Current formal objective variant: `simplified_normalized_literature_v1`.

## Deprecated Experiment Entrypoints

- `raw_weight_retune`: removed from the active CLI workflow for the formal benchmark; historical output files are preserved.
- `weight_sensitivity`: removed from the active CLI workflow for the formal benchmark; historical output files are preserved.
- `physical_baseline_v2`, `soc_reserve_slack`, `fc_lowfreq_reference`, `fc_reference_tracking`, `normalized_objective_v1`, `terminal_soc_penalty_experiment`, and `ramp_soft_penalty_experiment`: not introduced into the current formal 1 s OSQP-QP entrypoint.

## Retained Core Code

- `src/main/mpc_solvers/mpc_qp_formulation.py`: retained as the OSQP-QP formulation module.
- `src/main/benchmark_mpc_qp_osqp_1s.py`: retained as the 1 s offline benchmark runner.
- Fuel-cell ramp is retained only as a hard constraint.
- Objective terms retained: normalized H2, normalized SOC maintenance, normalized battery power penalty.

## Scope Checks

- 30 s mainline: not modified.
- CasADi/IPOPT baseline: not modified.
- outputs: preserved.
- DQN: not modified and not trained.
- LSTM: not trained.
- Existing historical reports: preserved.
