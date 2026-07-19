# Candidate C reproducibility check

## Scope

- Fixed weights: `q_h2=0.25`, `q_batt=0.4`, `q_soc=12.0`, `q_fc_var=20.0`
- Input: `outputs/mpc_solver_benchmark_1s/data/test_voyages_spline_1s.parquet`
- Input SHA-256: `bddec86243a9f2d2fdc6d5ee9e5c2dcde04fdfca9f407744ba0fe429fb349d23`
- Reference output: `outputs/mpc_1s_n6_candidate_C/`
- Recheck output: `outputs/mpc_1s_n6_candidate_C_recheck/`
- Voyages: `voyage_060` through `voyage_066`

## Result

The streamlined source reproduces Candidate C. All seven voyages completed with zero solver failures, zero primal-infeasible statuses, and zero maximum-iteration statuses. The requested physical and objective metrics are exactly equal after round-trip CSV parsing; therefore they are also within the existing project tolerances. All seven plot files have matching SHA-256 hashes, confirming identical plotted trajectories and major trends.

| Voyage | Complete / failures | total_h2_kg | final_soc | min_soc | max discharge kW | max charge magnitude kW | cumulative FC change (sum of squared step changes, kW²) | max balance residual kW | max FC step change kW |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| voyage_060 | yes / 0 | 33.0738170343205 | 0.544356611267314 | 0.488053164825396 | 308.872758780942 | 346.500000003288 | 535.203760672884 | 5.684e-14 | 3.518071149148 |
| voyage_061 | yes / 0 | 27.3110431025353 | 0.543055323834033 | 0.494881408720033 | 225.940634388391 | 120.726967230267 | 136.366056489690 | 2.842e-14 | 0.509961206675 |
| voyage_062 | yes / 0 | 27.7317140065822 | 0.539313790685014 | 0.489483859894060 | 346.500000001136 | 343.331410395412 | 580.057031371477 | 5.684e-14 | 3.245461714699 |
| voyage_063 | yes / 0 | 129.501711688494 | 0.551926305110261 | 0.450367017960383 | 316.990644351174 | 325.127392905402 | 1032.940045679481 | 5.684e-14 | 0.748744178199 |
| voyage_064 | yes / 0 | 14.7687869623833 | 0.550835949805586 | 0.497811480876471 | 338.750877850281 | 278.066421516826 | 274.337942666119 | 5.684e-14 | 0.775956772782 |
| voyage_065 | yes / 0 | 26.6646936622702 | 0.538851308515434 | 0.466899886003192 | 305.757280771942 | 346.500000007302 | 570.107573752080 | 5.684e-14 | 2.751501059294 |
| voyage_066 | yes / 0 | 29.6264915141653 | 0.545680320170567 | 0.476709745560825 | 273.020570244153 | 339.855866447438 | 521.034181958330 | 5.684e-14 | 0.791874713263 |

The only differences are runtime timing statistics, which are expected to vary between executions, and the implementation SHA-256 (`d4008519...` in the retained reference versus `04fe8cb4...` in the recheck) because the runner was streamlined. These differences do not change control trajectories or result metrics.
