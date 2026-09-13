> Historical design, superseded by the 2026-09-13 executed-reward plan and current README. Retained for reproducibility.

# DQN-MPC Unified Objective Implementation Plan

> Explicit task boundary: no DQN training, no Validation/Test reads, and no commit/push.

## Task 1: Lock the new contracts with failing tests

- Replace deadband assertions with continuous SOC-reference assertions.
- Require four nonnegative actions with sums equal to one.
- Require direct `1/(1+J*)` reward and removal of the weight-sum argument.
- Require full objective reconstruction from `H/B/S/F` contributions.

## Task 2: Implement the common objective and reward

- Update the QP SOC auxiliary constraints and metadata to track `SOC_ref=0.55` continuously.
- Update the physical objective evaluator and solver-bank result metadata.
- Update the environment to use the same-solve objective directly in reward.
- Preserve all physical constraints, N=6, terminal-SOC setting, and failure handling.

## Task 3: Audit base-term scales on Train only

- Freeze representative Train states and report full quantiles for `H/B/S/F`.
- Group by SOC, load, transition, and previous FC power.
- Compare engineering unit deviations and decide whether common additional scaling is justified.

## Task 4: Scan and select four sum-one actions

- Generate a bounded semantic candidate catalog.
- Run same-state probes and representative closed-loop Train windows.
- Reject infeasible, dominated, and trajectory-redundant candidates.
- Select A0 balanced, A1 hydrogen economy, A2 FC smoothing, and A3 SOC protection.
- Confirm roles on a parent-disjoint Train audit subset.

## Task 5: Freeze actions, document, and verify

- Update action mapping, README, training metadata, and focused diagnostics.
- Run targeted tests, full test suite, compile checks, and Train-only final audit.
- Report formulas, scaling decision, actions, physical roles, reward/winner results, bias assessment, readiness, and git status.
