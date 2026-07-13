# 1 s N=6 OSQP-QP MPC fixed-weight selection

## Experiment boundary

This is an offline ideal-foresight experiment on the natural-clipped cubic-spline 1 s reconstruction; it is not measured 1 s data and it does not use LSTM predictions.
At each second, the future six true samples (`t+1..t+6`) form the N=6 prediction window. The QP has a six-step control horizon, but the closed loop applies the first action only, then rolls forward by one second.
N=60 remains a historical solver/performance benchmark and was not searched in this task.

## Candidate summary

| candidate_id | closed_loop_complete | closed_loop_coverage_fraction | solver_success_rate | physical_infeasible_point_count | final_soc | worst_voyage_soc_net_change | hydrogen_total_kg | battery_throughput_kwh | fc_above_load_fraction | fc_surplus_energy_kwh | solve_time_ms_p99 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | false | 0.723573 | 0.999985 | 0 | 0.302847 | -0.349873 | 130.6 | 1805.09 | 0.280001 | 303.076 | 0.1542 |
| B | false | 0.714511 | 0.999985 | 0 | 0.246734 | -0.349808 | 109.333 | 1908.35 | 0.264897 | 218.605 | 0.1643 |
| C | false | 0.648608 | 0.999967 | 0 | 0.20008 | -0.35 | 89.8562 | 1704.85 | 0.041465 | 3.69513 | 0.264059 |
| D | false | 0.735204 | 0.999985 | 0 | 0.307221 | -0.349695 | 140.959 | 1621.28 | 0.270659 | 221.778 | 0.1693 |

All 24 requested aggregate metrics are preserved in the companion CSV and in each candidate's summary/voyage files.
For a candidate that terminates on solver infeasibility, energy/economy values describe only the successfully applied prefix and are not comparable for selection.

## Engineering selection

- Status: `no_candidate_selected`
- Selected candidate: none
- Method: manual engineering review in the required order: physical feasibility, long-term SOC, power allocation, economy/device use, solver performance
- Priority order: physical feasibility, long-term SOC, power allocation, economy/device use, solver performance.
- The legacy automated "least-bad" conclusion is not used in place of engineering judgment.

Selection reasons:
- No candidate completed all seven voyage closed loops.
- Every candidate violated the worst-voyage SOC net-change gate: the observed worst value was approximately -0.35 versus the required lower bound of -0.03.
- A, B, and D reached a hard primal-infeasible window on voyage_063 after depleting usable SOC; C was incomplete on voyage_063 and voyage_065.
- The historical N=60 anchor with A's weights had worst-voyage SOC net decrease -0.016661, whereas the corrected N=6 anchor reached -0.349873 and only 72.3573% closed-loop coverage; shorter horizon did not preserve SOC sustainability.
- D had the highest closed-loop coverage, but selecting it would be an impermissible least-bad decision because it still failed the first physical gate.
- The four-candidate search is stopped without adding a fifth candidate, as required.

Candidate decisions:
- **A**: Rejected: 72.3573% closed-loop coverage, one final solver failure on voyage_063, and worst-voyage SOC net change -0.349873.
- **B**: Rejected: lowering q_soc reduced coverage to 71.4511%, with one final solver failure on voyage_063 and worst-voyage SOC net change -0.349808.
- **C**: Rejected: widening the SOC normalization band produced the lowest coverage at 64.8608%, two incomplete voyages, and worst-voyage SOC net change approximately -0.35.
- **D**: Rejected: despite the highest coverage at 73.5204%, voyage_063 remained incomplete and worst-voyage SOC net change was -0.349695. It cannot be retained as a least-bad candidate.

## Numerical interpretation

Residuals are interpreted using the tolerances recorded in each config.json. A small residual such as 0.0154 kW is a numerical tolerance observation, not by itself a physical strategy failure.
