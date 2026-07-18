# N=6 Four-Objective MPC Sensitivity

- Coverage: baseline-only; 1 configuration row(s).
- Voyage completion: 7/7 configuration-voyage runs.
- Step completion: 93,030 expected / 93,030 attempted / 93,030 applied.
- Boundary: offline oracle using t+1..t+6 actual natural-clipped spline load.
- Model usage: no LSTM; no DQN; first optimized move only.
- Decision boundary: no automatic best, score, rank, winner, or final weight selection.

## All-ones baseline facts

- Weights: `q_h2=q_batt=q_soc=q_fc_var=1`; `q_ramp=q_terminal_soc=0`.
- Solver outcomes: 0 failures, 0 primal-infeasible statuses, and 0 maximum-iteration events.
- SOC: mean initial `0.550000000`, mean final `0.287935264`, mean delta `-0.262064736`, observed range `0.199997215` to `0.550000000`.
- Power: maximum balance residual `2.8422e-14 kW`, maximum FC ramp `15.990419 kW/step`, FC range `-1.4416e-7` to `560.026469 kW`, maximum battery discharge/charge `260.134823/35.040915 kW`.
- Hydrogen and timing: total H2 `218.448931 kg`; mean/p95/maximum solve time `0.183878/0.464255/16.989700 ms`.
- Raw objective totals `[H2 kg, P_batt^2 kW^2, SOC error^2, FC delta^2 kW^2]`: `[218.448931, 437498713.357305, 4672.761396, 130389.632686]`.
- Normalized totals `[H2, battery, SOC, FC variation]`: `[24712.946847, 3643.932321, 1869104.558203, 56.592723]`. The weighted contributions are identical at the all-ones baseline and sum to `1897518.030093`.

| voyage | completed | steps expected/attempted/applied | SOC initial -> final (delta) | SOC min/max | H2 (kg) | solver failure / primal infeasible / max iter |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| voyage_060 | true | 10650/10650/10650 | 0.550000 -> 0.286047 (-0.263953) | 0.286047/0.550000 | 23.008883 | 0/0/0 |
| voyage_061 | true | 10740/10740/10740 | 0.550000 -> 0.277118 (-0.272882) | 0.277118/0.550000 | 17.160355 | 0/0/0 |
| voyage_062 | true | 10650/10650/10650 | 0.550000 -> 0.230196 (-0.319804) | 0.230196/0.550000 | 16.369603 | 0/0/0 |
| voyage_063 | true | 32370/32370/32370 | 0.550000 -> 0.199999 (-0.350001) | 0.199997/0.550000 | 114.502605 | 0/0/0 |
| voyage_064 | true | 7140/7140/7140 | 0.550000 -> 0.400242 (-0.149758) | 0.400242/0.550000 | 9.121052 | 0/0/0 |
| voyage_065 | true | 10740/10740/10740 | 0.550000 -> 0.320276 (-0.229724) | 0.320276/0.550000 | 18.261102 | 0/0/0 |
| voyage_066 | true | 10740/10740/10740 | 0.550000 -> 0.301669 (-0.248331) | 0.301669/0.550000 | 20.025330 | 0/0/0 |

## Physical and numerical review

- All seven plots show SOC falling from `0.55` without recovery. The fuel cell supplies the main share of load, while the battery mainly discharges to cover the difference. Charging is absent or negligible except for small transients in voyages 060 and 062 and a maximum `35.040915 kW` transient in voyage 063.
- Voyage 063 reaches the lower SOC boundary at about 9,400 s; the battery then remains near zero and the fuel cell follows nearly all subsequent load. No solver-failure markers appear in any of the seven plots.
- The voyage 063 minimum SOC is `0.199997215`, a lower-bound residual of `2.7846e-6`. This exceeds the runner's declared SOC audit tolerance of `1e-6`, so it is recorded as a small numerical constraint-tolerance exceedance, not hidden by the 7/7 solver completion. The FC upper residual `0.026469 kW` remains within the declared `0.1 kW` power-bound tolerance; balance, battery-power, and ramp checks also remain within their declared tolerances.
- `formal_complete=true` records complete formal coverage of the seven voyages; it is not an acceptance decision for the physical behavior or final weights.

The complete baseline-first 17-case one-factor matrix has not been run; only its baseline row exists. Therefore no sensitivity trend, optimum, recommended interval, accepted weight set, or final weight selection is claimed.
