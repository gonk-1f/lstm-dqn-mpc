# V6 baseline audit and Optuna integration plan

Base: `544908d`; branch: `feat/v6-sb3-optuna-audit`.

## Boundaries

- Preserve baseline models, Replay, all historical results and data.
- Do not run Test, full 35-voyage × 8-pass training or a large search.
- Do not change physical parameters or economic coefficients.
- Audit reward modes before proposing or exposing an independent alternative experiment.
- Other worktrees and uncommitted DDPG/TD3 work remain untouched.
- Run only current regressions, offline reward comparisons and tiny smoke workloads.

## Stage 1 — dependency inventory and reversible cleanup

- Record tracked Python files, import edges, runtime entrypoints and current V6 closure.
- Inventory every local worktree and untracked path without changing other worktrees.
- Archive only confirmed historical DQN training tests/scripts; retain shared physics,
  economic, data-integrity, mask, TD and checkpoint coverage.
- Record before/after inventory, limitations of static reachability and regression results.
- Commit this stage independently.

## Stage 2 — economic/reward audit

- Read all four reward implementations and trace physical/economic ledgers.
- Verify start, transient, runtime and EOL accounting with small fixed trajectories.
- Check the gamma=1 full-voyage economic-return identity and isolate failure penalties.
- Compare alternative reward expressions offline on existing Train/Validation records.
- Separate confirmed behavior, errors, testable hypotheses, engineering assumptions and proposals.
- Keep the frozen baseline reward unchanged; commit the audit and regressions separately.

## Stage 3 — isolated tuning implementation

- Lock a compatible RL Zoo/Optuna toolchain in `.venv-optuna`; do not upgrade `.venv`.
- Use direct Optuna, existing MaskedDQN and the original physical executor.
- Freeze split/action/termination identities. Keep Test out of trial loading and selection.
- Require explicit formal budget/trial arguments. A smoke command is strictly bounded.
- Rank by Validation completion, coverage and actual full-set economic costs.
- Persist trial parameters, seeds, sampler state, checkpoints and resumable study storage.
- Support intermediate evaluation/pruning and distinguish a paused trial from a failed one.
- Profile environment execution/step, policy prediction, Replay sampling/storage,
  Adam work and report output; avoid double-counting nested timers.
- Add meaningful no-training integration tests and one tiny synthetic smoke run.
- Commit the integration independently.

## Stage 4 — handoff

- Record actual dependency versions, measured smoke timings and budget estimation method.
- Document formal commands without executing them and compare all changes to the baseline.
- Push the staged commits and create a review PR targeting `feat/v6-sb3-baseline`.
