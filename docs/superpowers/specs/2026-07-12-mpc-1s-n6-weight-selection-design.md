# 1 s Offline Perfect-Foresight N=6 MPC Weight Selection Design

## Status

Approved by the user's 2026-07-12 execution brief and executed on 2026-07-13. This document records the implemented boundary; the bounded A-D experiment ended with `no_candidate_selected` and did not broaden the search.

## Scope

The experiment uses the seven test voyages in `outputs/mpc_solver_benchmark_1s/data/test_voyages_spline_1s.parquet`. The signal is a 1 s natural-clipped cubic-spline reconstruction of 30 s vessel load, so it is offline and future-dependent. No LSTM or DQN is trained or invoked.

The formal experiment horizon is fixed at `N=6`, `dt=1 s`. At decision index `t`, the controller receives the ideal future horizon `load[t+1:t+7]`, solves a convex QP, applies only the first FC command at `t+1`, computes actual battery power from `load_actual - P_fc_actual`, and updates actual SOC from that battery power. A new voyage starts at `SOC=0.55`; its previous applied FC power is initialized from the first load sample clipped to the FC bounds.

## Architecture

Create `src/main/run_mpc_1s_n6_weight_selection.py` as a separate experiment entrypoint. It reuses `QpMpcConfig`, `build_qp_problem`, and the persistent OSQP helpers already exercised by the historical benchmark. The old `benchmark_mpc_qp_osqp_1s.py` remains the N=60 solver/performance benchmark and is not used as the active N=6 CLI.

The new runner contains four bounded units:

1. **Contract helpers:** construct an exactly six-point, within-voyage ideal horizon and extract the first planned control into distinct plan/actual state fields.
2. **Rolling controller:** create one persistent OSQP workspace per voyage, solve a strictly equivalent affine-scaled QP, update only bounds, execute one command, and terminate the voyage at the first final solver failure without inventing slack, load shedding or a control fallback.
3. **Physical metrics:** calculate unweighted Dp0 hydrogen, load energy, SOC, battery energy, FC surplus, constraint residuals, solver status and timing per voyage and overall.
4. **Artifact/report writer:** save only lightweight configuration, JSON/CSV summaries, constraint audit and one compact power/SOC figure per voyage. Full step trajectories remain in memory and are not written.

## Timing contract

For a voyage with samples `0..T-1`, decisions are `t=0..T-2` and executions are `t+1=1..T-1`. Near the voyage end, missing look-ahead entries are padded with the final sample from that same voyage; windows never cross voyages. The following fields remain separate internally:

- `load_horizon_kw` and `load_actual_kw`;
- `P_batt_plan_kw` and `P_batt_actual_kw`;
- `SOC_predicted` and `SOC_actual`.

The state transition is not clipped:

```text
P_batt_actual(t+1) = P_load_actual(t+1) - P_fc_actual(t+1)
SOC_actual(t+1) = SOC_actual(t) - P_batt_actual(t+1) / (3600 * 693)
```

This exposes numerical or physical violations instead of hiding them.

## Fixed physics and solver

- `P_fc=[0,560] kW`;
- `E_batt=693 kWh`, `P_batt=[-346.5,346.5] kW`, `P_batt_ref=346.5 kW`;
- `SOC_ref=0.55`, `SOC=[0.2,0.8]`;
- FC hard ramp `48 kW/step`;
- no charge/discharge efficiency, terminal SOC term, reserve slack, low-frequency FC reference, soft ramp term, balance slack or load shedding;
- OSQP warm start and polishing enabled, `eps_abs=eps_rel=1e-5`, `max_iter=10000`, fixed `adaptive_rho_interval=25`;
- exact affine variable/row scaling improves SOC-dynamics numerical conditioning without changing objective, constraints, weights or the physical solution;
- a maximum-iteration result receives one cold restart of the same QP; a final failure terminates that voyage.

Numerical audit tolerances are fixed in the experiment config: actual balance `0.01 kW`, QP balance/power bounds/ramp `0.1 kW`, SOC bounds `1e-6`, predicted-versus-actual first-step SOC `1e-5`, FC-above-load comparison `1 kW`, and near-limit reporting `1 kW`. Raw maxima are always reported, so a small residual is visible even when it is classified as numerical rather than physical.

## Candidates

Exactly four candidates are permitted, in order:

| ID | q_h2 | q_soc | q_batt | SOC_band |
| --- | ---: | ---: | ---: | ---: |
| A | 0.5 | 2.0 | 0.05 | 0.05 |
| B | 0.5 | 1.5 | 0.05 | 0.05 |
| C | 0.5 | 2.0 | 0.05 | 0.075 |
| D | 0.5 | 2.0 | 0.075 | 0.05 |

`q_ramp=q_terminal_soc=0` for every candidate. Candidate A ran and was inspected before B/C/D. No automatic least-bad selector is used. A provisional config is permitted only if a candidate passes the user's five-layer engineering review. None passed, so the explicit outcome is stored in `outputs/mpc_1s_n6_weight_selection/manual_decision.json` and no provisional config exists.

## Metrics and artifacts

Each candidate writes `config.json`, `summary_metrics.json`, `voyage_metrics.csv`, `solver_statistics.csv`, `constraint_audit.md`, and compact per-voyage plots under `outputs/mpc_1s_n6_weight_selection/candidate_<ID>/`. The overall table and report are written to `reports/mpc_1s_n6_weight_selection_table.csv` and `reports/mpc_1s_n6_weight_selection_summary.md`.

Metrics cover the 24 items in the user brief plus explicit plan/actual residuals, worst-voyage SOC delta and numerical tolerance counts. Hydrogen metrics use physical unweighted Dp0 mass. The report states that the input is offline spline reconstruction, future six points are ideal foresight, only the first action is executed, N=60 is historical, and the selected status is provisional unless every acceptance layer is demonstrably satisfied.

## Testing

Tests must prove N=6 variable/state dimensions, six-point timing, within-voyage end padding, first-step-only execution, actual power-balance battery calculation, actual SOC update, first/subsequent ramp structure, candidate identity, unweighted hydrogen metrics, status counting, energy/surplus formulas and lightweight artifact boundaries. The complete repository test suite must remain green before push.
