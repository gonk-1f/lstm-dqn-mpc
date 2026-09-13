> Historical design, superseded by the 2026-09-13 executed-reward plan and current README. Retained for reproducibility.

# DQN-MPC Unified Objective Design

## Scope

Rebuild only the four-term MPC objective, the four discrete weight actions, and the DQN reward. Keep the physical model, constraints, persistence prediction, state, network, and DQN hyperparameters unchanged. Use Train data only. Do not train DQN or commit/push.

## Objective contract

The common physical terms are

- `H = sum m_H2(P_fc[k]) / m_H2(600 kW)`
- `B = sum (P_batt[k] / 624 kW)^2`
- `S = sum ((SOC[k] - 0.55) / 0.05)^2`, for predicted states `k=1..N`
- `F = ((P_fc[0]-P_fc_prev)/48)^2 + sum ((P_fc[k]-P_fc[k-1])/48)^2`

These references define a common engineering scale and remain identical for every action. Empirical quantiles are evidence about operating incidence, not automatic reasons to rescale terms. Additional scaling is allowed only if the Train-only audit shows that the engineering references fail to create comparable meaningful deviations.

The SOC term is implemented with a nonnegative auxiliary variable constrained to be at least both signed distances from `SOC_ref=0.55`. Squaring it gives the exact continuous symmetric objective while retaining the existing convex QP structure.

## Action contract

Each action has four finite nonnegative weights whose sum is one. Candidate selection uses two separate checks:

1. Same-state four-action probes test objective/reward comparability and local action winners.
2. Fixed-action Train-only closed-loop rollouts test physical behavior, feasibility, dominance, and trajectory redundancy.

Candidate tuning uses a Train calibration subset; a parent-disjoint Train audit subset checks whether selected roles generalize. Winner balance is not a tuning target.

## Reward contract

The solver bank reconstructs the complete nonnegative physical objective from the selected action's solution. This avoids using the raw OSQP quadratic value, which may omit constants. The environment reuses that same solve and computes

`reward = 1 / (1 + J*)`.

There is no common reward, action bonus, second solve, four-action comparison during training, or division by the action-weight sum. Solver failure remains terminal with reward `-620`.

## Readiness gates

- All targeted and full tests pass.
- Every final action sums to one and objective reconstruction agrees with its four contributions.
- Train-only audits contain no NaN or infinity.
- Representative same-state probes and fixed-action rollouts have no solver failures.
- The four final actions show distinct, interpretable closed-loop roles and are not pairwise redundant.
- Any winner concentration can be explained by physical objective contributions rather than a weight-scale multiplier.
