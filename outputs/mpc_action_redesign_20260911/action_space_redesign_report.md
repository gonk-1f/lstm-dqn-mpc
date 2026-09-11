# Four-action MPC weight-space redesign

## Scope and selection protocol

- Gamma was restored to `0.99`; replay capacity remains `300000`, batch size `64`, and TD loss remains MSE.
- Candidate construction, pruning, and final selection used Train data only.
- Stage 1 scanned 25 candidates on seven deterministic 600 s Train windows covering low/stable, ordinary/stable, fluctuating, high, sustained-high, rapid-rise, and rapid-fall conditions. All 175 rollouts completed.
- Stage 2 evaluated 11 finalists on five complete Train segments. Four normal-to-severe segments completed for every candidate. `train_parent_021_01` failed for all 11 candidates at the hard lower SOC boundary and was retained as a pressure diagnostic rather than used as a single-case tuning target.
- The final four actions were frozen before running Validation. Validation and the two extreme cases `016_01` and `053_01` were confirmation only.
- No DQN training or Test rollout was run.

## Existing action-space findings

- The old A1 was a real hydrogen-economy endpoint, but depleted SOC more deeply and caused more failures.
- The old A2/A3 distinction was weak on short high-load windows because the fuel cell saturated at 600 kW; changing objective weights cannot create control authority beyond that physical limit.
- On complete Train segments, the candidate families formed useful tradeoffs in hydrogen, FC variation, battery throughput, and SOC. This supports redesigning the four actions around physical roles instead of retaining the former nominal/SOC/fast labels.

## Stage-2 Train finalist scan

The table below aggregates the four complete feasible Train representative segments. Rates are weighted by executed hours; mean minimum/final SOC are per-segment means.

| Candidate | Weights `(qH2,qBatt,qSOC,qFCvar)` | Reward/step | Mean min SOC | Mean final SOC | H2 kg/h | Batt throughput kWh/h | FC-TV kW/h |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E0 | `(0.40,0.25,8,8)` | -74.363 | 0.3320 | 0.4097 | 21.266 | 116.201 | 370.778 |
| E2 | `(0.40,0.35,20,12)` | -37.767 | 0.3701 | 0.4476 | 21.933 | 111.997 | 430.742 |
| B2 | `(0.25,0.50,50,20)` | -23.787 | 0.3980 | 0.4760 | 22.410 | 103.927 | 417.208 |
| B4 | `(0.20,0.50,40,16)` | -22.587 | 0.3998 | 0.4761 | 22.401 | 99.556 | 406.138 |
| S1 | `(0.25,0.50,30,40)` | -41.931 | 0.3712 | 0.4722 | 22.377 | 115.318 | 301.278 |
| S3 | `(0.30,0.40,30,40)` | -46.070 | 0.3657 | 0.4660 | 22.283 | 119.077 | 312.285 |
| S5 | `(0.20,0.65,60,30)` | -24.811 | 0.3964 | 0.4870 | 22.605 | 102.547 | 336.389 |
| R2 | `(0.15,0.65,80,12)` | -17.048 | 0.4125 | 0.4814 | 22.480 | 88.758 | 460.351 |
| R4 | `(0.15,0.80,120,8)` | -15.387 | 0.4165 | 0.4828 | 22.500 | 82.219 | 566.166 |
| R6 | `(0.15,0.65,120,12)` | -16.369 | 0.4146 | 0.4830 | 22.508 | 87.856 | 481.584 |
| R7 | `(0.15,0.70,160,12)` | -15.788 | 0.4162 | 0.4847 | 22.536 | 86.230 | 485.250 |

At the first executed step of each complete Train segment, the maximum cross-action FC/Battery difference was only `0.43-0.47 kW`. This is expected at episode reset because previous FC is initialized from the first load and all actions obey the same ramp and power-balance constraints. The meaningful differences emerge over the closed-loop trajectory, as shown by the SOC, throughput, and FC-TV ranges above.

## Pruning and final action set

- B2 was removed in favor of B4: B4 improved reward, mean minimum/final SOC, hydrogen, battery throughput, and FC-TV in the aggregate.
- E2 was removed because it occupied an intermediate economy/balanced point without defining an endpoint; E0 provides the clear hydrogen-economy role.
- S3 was removed in favor of S1: S1 had lower FC-TV, higher minimum/final SOC, and lower battery throughput, with only a small hydrogen penalty.
- S5 did not provide the lowest FC-TV or strongest SOC floor, so it overlapped the balanced and protection roles.
- R2/R6/R7 were removed in favor of R4 for the protection endpoint: R4 produced the highest mean minimum SOC and lowest battery throughput among the protection family, while its higher FC-TV makes the tradeoff explicit.

| Action | Final weights `(qH2,qBatt,qSOC,qFCvar)` | Physical role |
| --- | --- | --- |
| A0 | `(0.20,0.50,40,16)` | Balanced: moderate SOC support, throughput, hydrogen, and FC movement |
| A1 | `(0.40,0.25,8,8)` | Hydrogen economy: lower FC use and hydrogen, accepting deeper battery discharge |
| A2 | `(0.25,0.50,30,40)` | FC smoothing: lowest FC total variation, accepting higher battery throughput |
| A3 | `(0.15,0.80,120,8)` | SOC protection: strongest SOC floor and lowest battery throughput, accepting higher FC variation |

## Frozen-set Validation confirmation

| Action | Completed | Failures | Reward/step | Mean min SOC | Mean final SOC | H2 kg/h | Batt throughput kWh/h | FC-TV kW/h | Mean FC kW | FC >=590 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A0 balanced | 26 | 1 | -10.728 | 0.4432 | 0.5035 | 19.855 | 97.198 | 575.910 | 361.390 | 22.32% |
| A1 economy | 24 | 3 | -41.530 | 0.3802 | 0.4367 | 18.131 | 111.174 | 529.250 | 330.535 | 19.94% |
| A2 smoothing | 25 | 2 | -22.235 | 0.4123 | 0.5112 | 19.828 | 119.044 | 408.522 | 360.379 | 24.13% |
| A3 protection | 26 | 1 | -8.904 | 0.4618 | 0.4932 | 19.644 | 66.220 | 763.638 | 358.536 | 19.70% |

The Validation result preserves the intended tradeoffs. It does not prove that a newly trained DQN will use all four actions evenly; action-selection proportions can only be assessed after a separate training experiment.

## Extreme Validation cases

| Segment | Action | Completed | Failure step | Min/final SOC | H2 kg/h | Batt throughput kWh/h | FC-TV kW/h |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: |
| 016_01 | A0 | no | 5968 | 0.2005 / 0.2005 | 27.853 | 138.016 | 373.242 |
| 016_01 | A1 | no | 4848 | 0.2005 / 0.2005 | 23.733 | 162.678 | 472.738 |
| 016_01 | A2 | no | 5140 | 0.2004 / 0.2004 | 24.918 | 154.062 | 419.652 |
| 016_01 | A3 | no | 6298 | 0.2005 / 0.2005 | 28.735 | 132.884 | 427.629 |
| 053_01 | A0 | yes | - | 0.2035 / 0.2546 | 29.485 | 151.508 | 275.511 |
| 053_01 | A1 | no | 5562 | 0.2005 / 0.2005 | 25.655 | 151.942 | 454.418 |
| 053_01 | A2 | no | 6245 | 0.2006 / 0.2006 | 27.272 | 143.921 | 300.965 |
| 053_01 | A3 | yes | - | 0.2066 / 0.2577 | 29.458 | 137.670 | 318.624 |

`016_01` remains infeasible for every fixed action and therefore was not used to force the action design. `053_01` confirms that balanced and SOC-protection policies can complete it, while the economy and smoothing endpoints cannot.

## Common-reward one-step winner audit

The frozen action set was evaluated on 840 Train-only representative states. Each state was solved with all four actions, and every first MPC move was evaluated by the same DQN common reward. All 3360 MPC probes solved successfully.

| Action | Winner count | Winner share |
| --- | ---: | ---: |
| A0 balanced | 67 | 7.98% |
| A1 economy | 47 | 5.60% |
| A2 smoothing | 579 | 68.93% |
| A3 protection | 147 | 17.50% |

A3 does not dominate the one-step common reward. A2 wins most low/medium-load and high-SOC states, while A3 wins 50.0% of high-load states and 54.9% of states whose previous FC power is at least 590 kW. A0 and A1 have smaller winner regions.

The ranking margins are extremely small: 39.6% of states have a first-to-second relative common-cost gap no greater than 0.01%, 74.6% no greater than 0.1%, and 100% no greater than 0.5%. The maximum gap is 0.437%. Thus the nominal 68.9% A2 winner share is a near-tie result rather than evidence of a strong immediate-reward preference.

This audit passes the narrow “no A3 one-step dominance” check but does not show four strongly separated one-step reward regions. It is a warning rather than a reason to redesign the actions again: DQN learns discounted return, whereas this audit evaluates only the immediate reward. The complete fixed-action rollouts still show large long-run differences in SOC, hydrogen, battery throughput, and FC-TV.
