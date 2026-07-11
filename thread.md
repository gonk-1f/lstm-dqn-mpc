# Thread Handoff

Updated: 2026-07-11 after diagnosing the unchanged mirror-like power split

## Completed Work

The three project status files were corrected to the current formal 1 s offline OSQP-QP device setup:

- battery capacity `693 kWh`;
- continuous battery rate `0.5C`;
- maximum charge/discharge power `346.5 kW`;
- total FC plus battery power `906.5 kW`.

The active 30 s LSTM/CasADi-IPOPT mainline was not modified.

One additional candidate was run over all 7 test voyages and all 93,037 reconstructed 1 s rows. It was based on the current engineering-preferred candidate and changed only `q_soc`:

- output: `outputs/mpc_solver_benchmark_1s/osqp_n60_Ebatt693_simplified_spec_norm/case_spec_norm_h2_low_soc_1p5/`;
- weights: `q_h2=0.5`, `q_soc=1.5`, `q_batt=0.05`, `q_ramp=0`, `q_terminal=0`;
- solver success rate `0.999924761`;
- infeasible count `0`;
- mean / p99 / max solve time `0.635655 / 9.315024 / 44.185800 ms`;
- H2 total `293.029013 kg`;
- worst voyage final SOC change `-0.024871446`;
- SOC range `0.442661982` to `0.578759547`;
- battery throughput `1992.451576 kWh`;
- battery near-limit fraction `0.008889606`;
- two small ramp residual violations, with maximum ramp `48.0000083 kW/step`.

## Comparison and Decision

The existing `q_soc=2.0` candidate remains better for the requested fixed baseline. Compared with it, `q_soc=1.5`:

- reduces H2 by only `0.844388 kg`;
- worsens the worst-voyage final SOC change from `-0.016661` to `-0.024871`;
- increases battery throughput by `174.828567 kWh`;
- increases the near-power-limit fraction from `0.003698` to `0.008890`.

Therefore `q_soc=1.5` is not promoted. The default 8-case list and global configurations were not changed.

## Power-Split Diagnosis

The user's observation that the mirror-like power allocation did not improve is confirmed by the stored time series:

- mean absolute battery power increased from `70.334 kW` at `q_soc=2.0` to `77.102 kW` at `q_soc=1.5`;
- FC-change versus battery-change correlation became more negative, from `-0.469` to `-0.559`;
- battery throughput increased rather than decreased.

This is not a plotting error. The QP enforces `P_fc + P_batt = P_load` at every step, making battery power the exact residual of FC power. Changing `q_soc` cannot remove that relationship. Within the current normalized H2/SOC/battery-power objective, `q_batt` is the direct existing lever for reducing battery-power amplitude, but no new value was selected or run. No objective or controller change was made during this diagnosis.

## Verification

Fresh verification before the run:

```text
D:\py\Python3\python.exe -m py_compile src\main\benchmark_mpc_qp_osqp_1s.py src\main\mpc_solvers\mpc_qp_formulation.py
D:\py\Python3\python.exe -m unittest tests.test_mpc_solver_benchmark_1s tests.test_mpc_ramp_constraint_toggle
```

Result: 19 tests passed. The additional benchmark command exited successfully and generated the report, CSV outputs, and all 7-voyage figures.

## Caveat

This remains an offline controller/objective benchmark using natural-clipped cubic-spline reconstruction and future reconstructed rows as the horizon. It is not measured 1 s data and is not an online LSTM validation result.
