# RMS/L2 Train-only audit protocol

Stage 1 verdict: CONDITIONAL GO for cached Train-only rescoring, not formal
training. Mean hydrogen consumption and RMS battery/SOC/FC-change amplitudes
are dimensionless engineering quantities; their common norm is meaningful but
does not establish equal physical importance. MPC quadratic objectives and the
SOC reference denominator 0.05 stay unchanged.

Stage 2: score saved full physical H/B/S/F arrays with
`r=1/(1+sqrt((H/6)**2+B/6+S/6+F/6))`. Reuse 1440x84 original grid outcomes,
the 120 matched SOC groups, 6927 old-policy trace steps, 40x84 supplementary
window states, and all four saved solver-perturbation experiments. No new MPC
solves, no anchors, no training and no Validation/Test data reads.

Compute both RMS coordinates and their squared contributions, reward quantiles,
winners, top-two gaps, SOC response, exact-SOC reconstruction sensitivity and
float32 sensitivity. Interpret old trajectories as old trajectories, never a
rollout of the newly scored policy. Preserve all 84 actions.

Stage 3 gate requires BOTH persistent extreme S contribution AND demonstrable
suppression of H/B/F control influence. To distinguish value-level dominance
from control sensitivity, perform offline leave-one-term-out ranking diagnostics
on the SAME saved trajectories, one term at a time. These are diagnostic
counterfactuals, not proposed production rewards, and do not remove any MPC term.
Do not compare 0.075/0.10/0.15 unless both gate conditions are supported.

No formal source changes, hyperparameter changes, failure penalty selection,
commit or push. Verify original source hashes and independent score calculations.
