# v2 remaining work before formal training

The economic interval contracts are closed for the approved model assumptions,
but formal training remains **NO-GO**. The remaining work is deliberately not
executed by the economic-model closure change.

## Required evidence still missing

- Build and freeze the authorized Train dataset and episode payload contract;
  do not read Validation or Test during selection.
- Complete Train-only candidate-state distribution, redundancy, correlation,
  and sensitivity audits, then freeze the final state.
- Complete Train-only action behavior screening and freeze the final catalog/K.
- Calibrate the formal reward scale from immutable Train interval costs.
- Complete final-catalog paired cold/warm solver-robustness evidence on real
  Train cases.
- Resolve formal-training plant-bound applicability, including battery power
  limits, FC ramp limits, and the raw FC aggregate/reference-unit mapping.

## Explicitly deferred

This change does not run dataset construction or splitting, action screening,
DQN state audits, lifetime-factor sensitivity, formal training, or held-out
evaluation.

`N=5`, `M=5`, `tau_LPF=90 s`, and the battery lifetime factor `15000` are
frozen project-baseline configuration and no longer training blockers. Their
non-measured evidence classifications remain explicit; future sensitivity is
optional and non-blocking.
