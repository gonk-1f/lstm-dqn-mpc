# Next Steps

Updated: 2026-07-11 after diagnosing the unchanged mirror-like power split

## Immediate Decision

Retain `case_spec_norm_h2_low_fc_main` with `q_h2=0.5`, `q_soc=2.0`, and `q_batt=0.05` as the provisional 1 s offline fixed-weight candidate.

Do not promote the new `q_soc=1.5` case. Its worst voyage final SOC change is `-0.024871`, compared with `-0.016661` for `q_soc=2.0`, and its battery throughput increases from `1817.623008 kWh` to `1992.451576 kWh`. The small H2 reduction does not compensate for the weaker SOC maintenance and greater battery use.

No additional weight search is authorized or required by the current task.

Do not use further `q_soc` tuning to address the mirror-like allocation. The QP enforces `P_batt=P_load-P_fc`, so that exact residual relationship cannot be removed by weight tuning. If the scope remains weight-only and the target is to reduce its amplitude, `q_batt` is the directly relevant existing weight. No new value or run is authorized, and no objective-structure change has been made.

## Current Formal Setup

- `E_batt=693 kWh`.
- Battery continuous rate `0.5C`.
- `P_batt_max=P_batt_ref=346.5 kW`.
- `P_fc_max=560 kW`.
- `N=60`, `dt=1 s`.
- Hard FC ramp `48 kW/s`.
- OSQP / ADMM-QP.
- Seven offline reconstructed test voyages, 93,037 rows.

## Guardrails

- Do not revert the formal 1 s benchmark to the legacy scaled battery setup.
- Do not modify the active 30 s CasADi/IPOPT LSTM-MPC mainline from this result.
- Do not describe spline-reconstructed rows as measured 1 s data.
- Do not treat the offline future-load horizon as LSTM forecast evidence.
- Do not train DQN/KAN-DQN until a fixed MPC baseline is accepted on the intended formal evaluation path.

## Relevant Artifacts

- Provisional `q_soc=2.0` candidate: `outputs/mpc_solver_benchmark_1s/osqp_n60_Ebatt693_simplified_spec_norm/case_spec_norm_h2_low_fc_main/`.
- Rejected `q_soc=1.5` candidate: `outputs/mpc_solver_benchmark_1s/osqp_n60_Ebatt693_simplified_spec_norm/case_spec_norm_h2_low_soc_1p5/`.
- New candidate report: `outputs/mpc_solver_benchmark_1s/osqp_n60_Ebatt693_simplified_spec_norm/case_spec_norm_h2_low_soc_1p5/REPORT_case_spec_norm_h2_low_soc_1p5.md`.
