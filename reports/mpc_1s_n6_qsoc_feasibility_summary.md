# N=6 q_soc-only feasibility diagnosis

## Experiment boundary

This is a bounded structural diagnosis on the natural-clipped cubic-spline 1 s reconstruction. It is offline ideal foresight, not measured online 1 s data and not an LSTM forecast.
At decision time `t`, the QP uses `t+1..t+6`, applies only the first action, computes actual battery power as actual load minus applied FC power, and updates actual SOC from that battery power.
No DQN is trained or invoked. No terminal SOC term, slack, load shedding, soft ramp, model change, or N=60 rerun is introduced.

## Preregistered candidates

Only `q_soc` changes. Every candidate fixes `q_h2=0.5`, `q_batt=0.05`, `SOC_band=0.05`, `q_ramp=0`, and `q_terminal_soc=0`.

| candidate_id | q_soc |
|---|---:|
| QSOC_5 | 5 |
| QSOC_10 | 10 |
| QSOC_20 | 20 |

## Candidate results

| candidate_id | q_soc | closed_loop_complete | closed_loop_coverage_fraction | solver_failure_count | physical_infeasible_point_count | final_soc | worst_voyage_soc_net_change | soc_min | soc_max | hydrogen_total_kg | battery_throughput_kwh | fc_above_load_fraction | fc_surplus_energy_kwh | solve_time_ms_p99 | solve_time_ms_max |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| QSOC_5 | 5 | true | 1 | 0 | 0 | 0.463758 | -0.106636 | 0.338524 | 0.55 | 265.411 | 1326.3 | 0.417295 | 453.972 | 0.1231 | 12.5636 |
| QSOC_10 | 10 | true | 1 | 0 | 0 | 0.508589 | -0.0470825 | 0.415201 | 0.55 | 278.344 | 872.96 | 0.378953 | 336.037 | 0.1305 | 0.6435 |
| QSOC_20 | 20 | true | 1 | 0 | 0 | 0.529671 | -0.02164 | 0.45366 | 0.55 | 284.452 | 594.583 | 0.357336 | 247.984 | 0.1402 | 45.85 |

The companion CSV and per-candidate files retain all requested aggregate, per-voyage, and solver metrics. If a voyage terminates early, its hydrogen, energy, and dispatch-fraction values cover only the successfully applied prefix and are not valid full-voyage comparisons.

## Fixed feasibility gate

A feasibility witness must complete all seven voyages, have zero final solver failures and zero physical infeasible points, keep actual SOC within `[0.2,0.8]`, have worst-voyage SOC net change at least `-0.03`, retain complete aggregate metrics, and solve every QP in under 1 s.
No aggregate score or least-bad ranking is used.

## Diagnostic decision

- Status: `weight_only_sufficient`
- Selected candidate: none (this diagnosis identifies feasibility witnesses; it does not accept a paper weight)
- Feasibility witnesses: `QSOC_20`
- Provisional config created: false
- Accepted config created: false

At least one preregistered q_soc value is a feasibility witness under the exact terminal-free N=6 ideal-foresight setup. This does not constitute an accepted paper weight.

Candidate gate decisions:
- **QSOC_5**: worst-voyage SOC net change is below -0.03.
- **QSOC_10**: worst-voyage SOC net change is below -0.03.
- **QSOC_20**: passed every fixed feasibility gate.

The fixed gates establish only physical, SOC, and solver feasibility. Hydrogen use, battery throughput, and fuel-cell-above-load behavior are reported diagnostics without hard acceptance thresholds, so a passing witness is not an economic or power-allocation acceptance.
`QSOC_20` remains only a feasibility witness: hydrogen total is 284.452 kg and FC-above-load fraction is 0.357336; neither metric passed a hard economic or dispatch-quality gate in this diagnosis.

## Historical boundaries

The retained q_soc=2 A anchor had closed-loop coverage 0.723573 and worst-voyage SOC net change -0.349873; its source remains `outputs/mpc_1s_n6_weight_selection/candidate_A/summary_metrics.json`.
N=60 remains a historical solver/performance benchmark only; its retained pointer is `outputs/mpc_1s_n6_weight_selection/N60_HISTORICAL_BENCHMARK.md` and its result tree was not modified or rerun.

The structural conclusion is limited to `q_soc in {5,10,20}` under this exact terminal-free N=6 setup. Failure of all three does not prove that every larger q_soc is impossible; passing does not automatically authorize DQN training or promote a final paper configuration.
