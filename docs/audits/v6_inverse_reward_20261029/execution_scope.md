# Execution scope and pre-change gate

Date: 2026-10-29. Remote PR #1 and PR #2 were open/unmerged when inspected. Both the local and remote PR #2 heads were `814994e`; branch `feat/v6-sb3-inverse-reward-audit` starts from that commit and targets `feat/v6-sb3-economic-episode500`.

The attached specification requires a standalone pre-change audit before changing economics or reward code, and explicitly requires stopping the corresponding change if a mathematical/physical conflict cannot be resolved within its rules.

## Execution order

1. Read the frozen physical, accounting, SOC valuation, direct environment and training/Optuna paths. Inspect available Train trajectory evidence without using Test for selection.
2. Produce `pre_audit.md`, numerical component distributions, physical boundary counterexamples and a source manifest. Keep cumulative Train summaries distinct from sparse samples and newly generated bounded audit traces.
3. Compare mathematically explicit real-time storage-value alternatives, including charge conservation, FC-off discharge, initial stored energy, cycles, endpoint accounting and observation requirements.
4. Audit the exact requested `1/C` and terminal `-1`, without offsets, clipping, extra fees, action exclusions or altered failure classification. Record ordering and long-horizon counterexamples.
5. Decide whether the preconditions permit production changes. If not, retain all production code and deliver the reproducible audit, tests, specific design decisions and a review PR. Do not invent unavailable training commands.

## Confirmed initial gate failure

An independent one-step check using the actual `DirectVesselEnv` and frozen account produced a physically legal hold action with load=0, FC=0, battery=0 and SOC=.6. H2, FC degradation, battery degradation and SOC-equivalent costs were all exactly zero. The episode ended normally, not as a physical failure. Thus `1/C` is undefined at a legitimate model point.

The specification forbids changing this into a physical failure, masking out the action, imposing a positive offset, clipping the cost or using an epsilon denominator. A flow-based storage value with unchanged accounting state has no energy change to value in this idle interval; a separate price-revaluation process would require an explicitly justified new convention. Consequently no production inverse mode or corresponding formal training entrypoint will be enabled in this audit branch unless the user changes that design decision in a later instruction.

Separately, a current-efficiency charge allocation is definable, but FC-off discharge valuation needs an explicit economic convention. Historical energy-cost inventory is self-consistent with additional value state; it is not automatically a complete-observation model with the unchanged history5 input. Alternatives and the assumptions needed to select one are documented rather than silently chosen.

Only audit tools, audit tests and reports may be added before the pre-change report is committed. Existing reward modes, failure penalty, physics, action masks, 500-episode schedule, 32:1 cadence, batch size, Target units, checkpoints, data and archived results remain unchanged.
