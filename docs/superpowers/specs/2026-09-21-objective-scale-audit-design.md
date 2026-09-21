# v2 Objective Normalization and Train-only Scale Audit Design

## Scope

This increment changes only the three v2 MPC objective definitions, their fixed
normalization constants, the SOC soft working band, and an offline Train-only
objective-scale audit. It does not change the DQN simplex, data pipeline,
economic reward, degradation models, hard constraints, or formal-training
NO-GO policy.

## MPC objective contract

All three components are horizon means. The method constants are
`P_FC_SCALE_KW=600`, `DELTA_P_FC_SCALE_KW=600`, hard SOC bounds `[0.20,0.80]`,
soft zero-penalty band `[0.40,0.60]`, and `SOC_SCALE=0.60`. `MPCConfig` rejects
other objective-scale or SOC-band values so callers cannot silently run a
different v2 objective under the same version. The existing positive,
sum-to-one three-weight simplex is unchanged.

The hard FC ramp constraint remains a separate explicit positive
`fuel_cell_ramp_kw_per_step`. No value is inferred from 48 kW/s or from the new
600 kW smoothness normalization. The repository still has no source-backed,
formally frozen per-step ramp value, so this remains a preflight limitation.

## Train-only audit

`src/v2/analysis/objective_scale_audit.py` receives an exact current-v2 Train
provenance, lazy representative-state loader, canonical candidate actions, and
a solver runner. The split guard runs before state loading or solver execution.
For every state/action pair the runner must return a canonical successful
`MPCPlan`; the audit snapshots the three unweighted components, first FC and
battery commands, and predicted SOC trajectory.

The audit reports population count/mean/std/P50/P90/P95/P99/max for all three
raw components. SOC also reports positive incidence and conditional positive
percentiles. The scale ratio uses full P95 for base/smooth and positive-only P95
for SOC. Missing positive SOC evidence is NO-GO. Ratios `<=5`, `(5,10)`, and
`>=10` map to GO, WARNING, and NO-GO; these are declared project engineering
criteria, not theoretical constants.

For each ordered objective pair the audit records how often `0.1*J_i >
0.7*J_j`, together with affected state/action IDs. Actual `q_i*J_i`
distributions are also recorded. Behavioral redundancy is reported per state
when different actions produce numerically indistinguishable components,
first commands, and SOC trajectories under explicit audit tolerances. It does
not alter the action catalog.

## Evidence and preflight

The repository lacks an accepted raw Train operating cycle and a final action
catalog, so no formal numeric audit is run in this increment. The report must
show unavailable statistics rather than synthetic values and the new
objective-scale preflight item remains NO-GO. Synthetic fixtures validate only
the audit mathematics and access guards.

