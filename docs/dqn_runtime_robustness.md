# Formal training runtime: engineering changes

Current formal method: causal seven-state DQN selects 84 integer-composition
MPC weight tuples; gamma=0.99, Ts=T_sw=1 s, N=6. Reward scores the actual
executed first step; see README for the complete formula. The previous
four-action/deadband checkpoints are historical only. Physical constants
remain in `src/utils/physical_config.py`. Numerical failures abort without replay;
physical execution violations use an explicitly calibrated terminal penalty.

## Resume

Each terminal training segment (including an explicitly failed episode) writes
`training_state_latest.pt` in the run directory. Round boundaries additionally
write `round_X/model_roundX.pt` and `round_X/training_state_roundX.pt` before
greedy validation. Full states use `.tmp`, flush/fsync and atomic replace.

Use the existing `--resume-training-state` option with either full-state file.
The checkpoint restores online/target networks, Adam, replay circular position,
epsilon, global/update/sync counters, RNGs, configuration, ordered segment
prefix and small round summaries. Resume starts at the next unprocessed segment;
a finished round starts the following round. It never reruns completed training
segments. An interrupted segment itself must be rerun from its preceding boundary.
If the last segment is saved but round artifacts/validation are interrupted,
`round_finalization_pending` makes resume finish that round's artifacts and
greedy validation before training the next round. Completed training segments
are still skipped. A completed validation clears the flag in both full-state
files. An interrupted validation pass is rerun; its partial artifacts are replaced.

Train order and formal configuration mismatches are rejected. Model-only weights
are not resumable training state. No checkpoint can recover the previous run's
already-lost unsaved progress.

## Memory evidence and limits

The previous runtime retained loss and update-index lists for all global steps,
and sync-index history without a cap. These are now bounded to 1000 entries;
loss mean/count/extrema use cumulative statistics. The reported loss median is
exact over all updates up to 1000, then explicitly scoped to the latest 1000
updates (`median_scope`). Detached loss tensors are transferred in bounded batches
at log/segment boundaries, not as per-update logging scalars.

Training did not retain full per-step physical traces or entire segment arrays
in round summaries. The replay buffer was already capacity-limited. Validation
previously retained all complete DataFrames for concatenation; it now keeps only
the five columns needed for exact Q-gap/load quantiles and regime counts, scoped
to one validation pass. Per-segment plots were already closed.

The loader already selected three columns. It now fixes numerical columns to
float64 and returns an owned load array, releasing the DataFrame/time block.
No chunking, float32 conversion, dataset rewrite or algorithm change is used.

These are proven retention/allocation improvements, not proof of the unique cause
of the reported pandas C-parser OOM. The exited process has no memory profile;
the identified Python histories alone are not sufficient evidence of multi-GB
exhaustion. CSV parsing can be the allocation that fails after process/system
memory is exhausted. No claim is made that a long-run OOM has been reproduced
or conclusively eliminated by a short synthetic test.

## Execution and efficiency

`validate_executed_step` checks actual FC/battery/SOC, power balance and existing
FC ramp bounds in the environment before committing state. Training and both
greedy evaluation paths use the same explicitly calibrated physical-failure penalty;
there is no fallback or new forecast-error constraint.

Warmup/random action selection has zero action-Q forwards. A greedy decision
has one. Each ordinary Bellman update retains one online and one target forward.
Finite Q/target/loss and gradient checks remain; mandatory scalar safety/action
synchronization is not misrepresented as removable logging overhead. Q diagnostic
means/std are transferred only at reporting/checkpoint boundaries. Validation
retains the required 84-Q trace using its single evaluation forward.

The state hot path passes only the last 60 samples to the unchanged state builder.
This has constant bounded work and matches the full-history state exactly, without
running-sum drift or future access. Existing 1000-step logs now include round,
segment index and steps/s.

Unused simplified/legacy objective branches and stale future-preview/reward
comments were removed. Version-1 full-state reading, historical MLP weight-key
remapping and the solver-settings alias used by tests are retained. Independent
ideal energy feasibility, not DQN failure, is the documented reason for excluding
0137/0160 from ordinary learning/selection; 0158 is still normal validation.

## Verification scope

Only unit tests and synthetic checks are run. Held-out load reads in dataset
unit tests use temporary fixtures; the frozen manifest is checked as metadata.
No formal training, validation/test rollout or data construction is launched.
