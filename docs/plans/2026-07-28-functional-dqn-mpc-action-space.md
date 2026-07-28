# Functional DQN-MPC Action-Space Plan

## Scope

This work converts the current numerically distinct DQN action table into a
small library of physically distinct MPC control preferences. It keeps the
existing 11-dimensional state, four-term reward, N=6 QP, physical constraints,
offline-oracle load preview, Candidate C baseline, and 46/13/7 voyage split.
Test voyages `voyage_060` through `voyage_066` remain inaccessible to action
design, correction, training, validation, and checkpoint selection.

## Repository cleanup

1. Preserve formal data, provenance, split files, Candidate C artifacts, current
   MPC/DQN implementation, and tests that freeze current behavior.
2. Remove tracked IDE/personal launch files and obsolete source paths only when
   reference checks prove that they are unused or depend on modules that no
   longer exist.
3. Consolidate v2/v3 and hard-voyage evidence into a minimal action-space
   history. Delete redundant row-level diagnostic artifacts only after their
   conclusions and provenance are retained.
4. Add narrow ignore rules for scratch/debug outputs, temporary rollouts, and
   non-final checkpoints without ignoring formal summaries or final models.

## Evaluation workflow

Create one small reusable action-space evaluator that reuses the existing
environment, QP formulation, scaled OSQP solver bank, reward, and split loader.
It must:

- reject test voyages before reading trajectory data;
- derive load and load-change regimes from train and validation statistics;
- use documented SOC points tied to the physical bounds and reference;
- cover low, medium, and high previous fuel-cell states;
- solve every candidate at every representative state;
- record first moves, horizon trajectories, predicted SOC, hydrogen use,
  reward decomposition, solver status, and solve time;
- run fixed-action coverage over every train and validation voyage;
- separate `primal infeasible` from `maximum iterations reached`.

The evaluator accepts an explicit candidate list for the bounded design stage
but defaults to the frozen production action table. It must not duplicate MPC
equations or alter physical constraints.

## Action design

The first candidate library contains five roles:

1. Nominal / Candidate C.
2. Hydrogen Economy / Battery Assist.
3. Bidirectional SOC Regulation.
4. Fast Fuel-Cell Response.
5. Fuel-Cell Smoothing / Battery Buffer.

Candidate C remains exactly `(0.25, 0.40, 12.0, 20.0)`. The other numerical
weights are selected only after probing the current v3 actions. No grid,
random, Bayesian, or automated search is allowed. At most one correction is
permitted, and only for demonstrated redundancy, physical misbehavior, or
solver scaling failure.

An optional sixth action is allowed only if the five-role probe demonstrates a
missing and non-redundant control function.

## Acceptance

The final library passes only when:

- each retained action has an observed physical role;
- no pair remains behaviorally near-identical across representative states;
- local probes solve successfully;
- no action has systematic `maximum iterations reached`;
- fixed-action train/validation coverage has no action-specific solver
  instability hidden by aggregate success counts;
- state-conditioned rewards and physical responses show that the preferred
  control behavior depends on state rather than one action dominating
  essentially everywhere.

All-action physical infeasibility is reported separately and is not disguised
as numerical failure.

## Training gate

If action-space acceptance fails, work stops before formal DQN training.

If it passes:

- freeze the action table and update action-count-dependent tests;
- handle solver failures transparently without executing another action or
  storing a mislabeled transition;
- accumulate all four reward components during training and validation;
- run only the existing MLP-DQN path with one reproducible configuration;
- compare against Candidate C and meaningful fixed actions;
- analyze within-voyage action changes, conditional action distributions,
  Q-values, and physical responses on validation data only.

## Verification

Use test-first changes for the evaluator, any action-table update, failure
handling, and reward-component aggregation. Finish with focused DQN/MPC tests,
the full existing test suite when practical, `compileall`, `git diff --check`,
and a test-access audit.
