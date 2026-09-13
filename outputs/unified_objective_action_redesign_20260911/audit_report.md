# Unified DQN-MPC objective Train-only audit

## Scope

This audit used Train data only. It did not create or update a DQN, replay buffer, checkpoint, Validation result, or Test result.

## Final common MPC objective

For `N=6`:

- `H = sum(k=0..5) m_H2(P_fc[k]) / m_H2(600 kW)`
- `B = sum(k=0..5) (P_batt[k] / 624 kW)^2`
- `S = sum(k=1..6) ((SOC[k] - 0.55) / 0.05)^2`
- `F = ((P_fc[0]-P_fc_prev)/48)^2 + sum(k=1..5) ((P_fc[k]-P_fc[k-1])/48)^2`
- `J* = min(qH*H + qB*B + qS*S + qF*F)`

The four engineering references are common to every action. A sustained unit engineering deviation gives 6 over the six-step horizon for each term. No extra empirical scaling was added.

## Base-term scale audit

The frozen 840 Train states were solved with four predeclared semantic endpoint actions, producing 3360/3360 successful probes.

| Term | Min | P5 | Median | Mean | P95 | P99 | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H | 1.3180 | 1.7636 | 3.1105 | 3.6206 | 5.9158 | 5.9256 | 5.9377 |
| B | 0.0014 | 0.0112 | 0.1275 | 0.2403 | 0.6287 | 2.1468 | 2.7926 |
| S | 0.0000 | 0.0000 | 8.5990 | 11.0391 | 24.0783 | 24.1287 | 24.3110 |
| F | 0.0000 | 0.0053 | 0.1105 | 0.6483 | 2.8568 | 3.0470 | 3.1300 |

The SOC distribution is deliberately multimodal because the frozen probe includes SOC stress values. Lower B/F incidence is a property of the observed controller states; it does not invalidate their engineering normalizers.

## Candidate scan and final actions

Thirty-seven predeclared sum-one candidates were scanned on seven 600 s Train windows. All 259 rollouts completed. Finalists were then checked on five calibration and five parent-disjoint full Train segments. Low-SOC failures from weak-SOC candidates reproduced across the parent-disjoint set; therefore they were rejected rather than repaired by winner-share tuning.

| Action | Weights `(H,B,S,F)` | Physical role |
| --- | --- | --- |
| A0 | `(0.05,0.15,0.70,0.10)` | balanced compromise between FC smoothing and SOC/throughput protection |
| A1 | `(0.10,0.25,0.55,0.10)` | hydrogen economy endpoint among the full-segment-feasible candidates |
| A2 | `(0.05,0.10,0.60,0.25)` | FC smoothing endpoint |
| A3 | `(0.05,0.20,0.70,0.05)` | SOC and battery-throughput protection endpoint |

All four sums equal 1.0.

## Parent-disjoint Train closed-loop confirmation

| Action | Completed | Failures | H2 kg/h | Mean min SOC | Mean final SOC | Batt throughput kWh/h | FC-TV kW/h | Mean FC kW | FC >=590 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A0 | 5/5 | 0 | 15.4722 | 0.4138 | 0.4298 | 67.4721 | 1582.00 | 281.36 | 25.45% |
| A1 | 5/5 | 0 | 14.0024 | 0.3497 | 0.3518 | 80.6092 | 1858.53 | 254.48 | 19.72% |
| A2 | 5/5 | 0 | 15.1996 | 0.3931 | 0.4179 | 84.2992 | 1134.97 | 276.32 | 23.30% |
| A3 | 5/5 | 0 | 15.5585 | 0.4215 | 0.4332 | 60.6789 | 1923.44 | 282.96 | 25.97% |

The roles are physically distinct: A1 saves 1.20-1.56 kg/h versus the other actions but accepts lower SOC; A2 gives the lowest FC-TV; A3 gives the highest SOC and lowest throughput; A0 lies between the smoothing and protection endpoints. Pairwise median first-step FC differences are 2.91-7.74 kW and pairwise median six-step FC-trajectory RMS differences are 7.49-19.29 kW, so no pair is control-equivalent.

All four actions fail only on calibration stress segment `train_parent_021_01`, where every scanned candidate reaches the hard lower SOC bound under the sustained energy deficit. This common failure was not used to favor an action.

## Reward and winner audit

Successful reward is `r=1/(1+J*)`; failure remains `-620`. The reward uses the reconstructed complete physical objective from the same MPC solve. It has no public common reward weights and no division by `sum(weights)`.

All 3360 final state-action probes solved and contained finite objective/reward values.

| Action | J median | J mean | Reward median | Reward mean | Winner count | Winner share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A0 | 6.2030 | 7.9557 | 0.13883 | 0.22445 | 0 | 0.00% |
| A1 | 5.0784 | 6.5106 | 0.16452 | 0.22633 | 639 | 76.07% |
| A2 | 5.3377 | 6.8466 | 0.15779 | 0.23933 | 151 | 17.98% |
| A3 | 6.2060 | 7.9566 | 0.13877 | 0.22436 | 50 | 5.95% |

The median first/second relative reward gap is 3.53% (P95 6.78%). The concentration is therefore not only a numerical tie artifact.

The most serious result is the SOC-conditioned winner direction:

- `SOC < 0.50`: A1 wins 240/240 states; A3 wins 0.
- `0.50 <= SOC < 0.55`: A1 wins 65%, A2 wins 35%.
- `0.55 <= SOC < 0.60`: A2 wins 58.3%, A3 wins 41.7%.
- `SOC >= 0.60`: A1 wins 89.2%.

This is a structural cross-objective comparability problem. In low-SOC states, an action with a larger SOC weight receives a larger value of its own objective even when its control response protects SOC better. Sum-one weights remove global scalar multiplication, but they do not turn different weighted objectives into one common utility. The optimized value `V(q)=min_x q*f(x)` is concave in the weight vector, which also makes an interior balanced action liable to have no minimum-cost winner region.

## Readiness decision

The QP objective, action table, and reward implementation are internally consistent and fully testable. The fixed-action control roles are distinct. The direct selected-objective reward nevertheless has confirmed mathematical structural bias: it chooses the economy action in every low-SOC probe and gives A0 no immediate winner region.

Therefore this configuration is **not ready for formal DQN Round-1 training**. Retuning action weights to manufacture winner shares would hide the issue rather than resolve it.
