# V6 economic episode training implementation plan

> For agentic workers: execute with subagent-driven-development, verify each component and review integration before publishing.

**Goal:** Deliver one shared, resumable 500-episode economic DQN runner and a compatible finite-budget Optuna adapter, without launching a formal experiment.

**Architecture:** Keep the frozen Queue adapter and baseline training entrypoint available. Add a direct single-thread environment using existing physical/accounting primitives. The new episode runner owns sampling, collection/update cadence, online metrics, Validation and checkpoint retention; `v6_sb3.tuning` dispatches episode-budget studies into that runner.

**Tech stack:** Existing isolated `.venv-optuna`, SB3 2.9.0, Gymnasium 1.2.3, PyTorch, Optuna 4.9.0, CPU/one Torch thread.

## Verified starting state

- PR #1 is open/unmerged; its head is `b212483`. New branch: `feat/v6-sb3-economic-episode500`; PR target: `feat/v6-sb3-optuna-audit` while PR #1 remains unmerged.
- Existing linked worktree and isolated interpreter are reused. Other worktrees, old models/Replay, data and `docs/results/` remain untouched.
- The user supplied the design and fixed parameters in the attached specification. Routine choices below make that design concrete; no additional design approval is needed for the already requested implementation.

## 1. Direct environment and equivalence

Files: `src/v6_sb3/direct_environment.py`, any minimal shared primitive in `src/v4/`, `tests/test_v6_direct_environment.py`.

- [ ] Write failing step/equivalence tests for both audited rewards, regular completion, genuine successor-mask failure, infeasible reset and compact restore.
- [ ] Implement `DirectVesselEnv.reset(options={'episode': episode})`, `step`, `action_masks`, `snapshot`, `restore` with existing `EconomicMPC.interval`, state/history builders, feasible-action screening, reward and failure primitives.
- [ ] Keep no background thread/queue and no second set of physical equations. Consume transitions online instead of retaining voyage tensors or Q histories.
- [ ] Compare FC/battery/SOC/H2/degradation/reward/mask/terminal reason against the original executor for identical actions.

## 2. Target units and fixed algorithm

Files: `src/v6_sb3/masked_dqn.py`, `tests/test_v6_target_env_steps.py`.

- [ ] Write failing tests that separate warmup, formal environment steps and Adam updates, including 10000 not being divisible by 32.
- [ ] Add an explicit Target synchronization unit, retaining the old Adam mode for historical runs and using formal environment steps for the new runner.
- [ ] Record each hard copy with formal step and Adam count. Disable the Adam-triggered copy in the new mode.
- [ ] Keep ordinary masked one-step DQN, Huber beta 1, uniform Replay and all existing physical action definitions.

## 3. Shared episode runner

Files: `src/v6_sb3/episode_training.py` and focused schedule/checkpoint/metrics helpers if needed; `tests/test_v6_episode_training.py` and schedule tests.

- [ ] Generate 500 independent uniform-with-replacement Train choices with a dedicated RNG; persist actual order, counts and RNG.
- [ ] Separate 50000 random warmup steps from complete formal episodes. Collect 32 formal transitions per Adam update, carrying remainder across episodes without truncating them.
- [ ] Use epsilon 1 to .05 over the first 80% of the configured episode budget, then hold .05; count progress by formal episode index.
- [ ] Aggregate loss/TDMAE/return/physics online, emit one line per finished episode and one per Validation, use null metrics when no Adam occurred.
- [ ] Greedy Validation every 50 episodes through the same direct environment, using completion/coverage/full-set economics. Never load Test.
- [ ] Persist pending update/boundary work, physical state, model, Replay, RNG, schedule, counters and online accumulators for exact pause/resume.
- [ ] Save best model separately from latest recoverable state; bound retained checkpoints and avoid writing full Replay every episode.
- [ ] Provide real CLI help, smoke, resume, selected-parameter import, result inspection/comparison. Require explicit acknowledgement of the audited unchanged failure penalty for formal training.

## 4. Train-only audit and real short benchmark

Files: `scripts/audit_episode_failure_penalty.py`, `scripts/benchmark_v6_environment.py`, `docs/audits/v6_episode500_20261029/`.

- [ ] Inspect Train-only full-cost/reference/prefix evidence, marking missing complete trajectories rather than imputing them.
- [ ] Document failure placement and structural/uncontrollable failure, demonstrate the early-termination risk with correctly labeled evidence, and propose calibration/sensitivity choices without changing the value.
- [ ] Benchmark identical bounded real Train prefixes on Queue/direct environments (hard maximum 256 steps), report measured ratio and equality.
- [ ] Estimate full-run/search budget transparently from actual episode lengths and measured collection/Adam/evaluation work; label unmeasured portions.

## 5. Optuna adaptation and fresh retraining

Files: `src/v6_sb3/tuning.py`, `src/v6_sb3/episode_tuning.py` if needed, `src/v6_sb3/search.py`, `tests/test_v6_episode_optuna.py`.

- [ ] Preserve the previous step-budget API; select the new path with explicit episode-budget arguments.
- [ ] Reuse the shared episode runner, identical seeds/order, 32:1 cadence, pure economics and environment-step Target units.
- [ ] Search LR, batch size, Target interval in environment steps, Replay capacity and network width; keep gamma/reward/failure/physics fixed.
- [ ] Persist SQLite, sampler, trial parameters and resumable active state; prune only after Greedy Validation.
- [ ] Export best sampled parameters for a fresh 500-episode run, never load the best Trial's trained checkpoint as a fresh result.
- [ ] Test real small synthetic trials plus no-training lifecycle/identity/isolation contracts.

## 6. Verification and delivery

- [ ] Run affected old SB3/economic regressions, all new contracts, double-reward smoke and exact pause/resume checks. Log actual commands and outcomes.
- [ ] Verify every delivered CLI with `--help` or bounded execution, including inspection, comparison and best-parameter export.
- [ ] Deliver the eleven requested PowerShell command groups, actual timing, remaining penalty decision and honest budget bounds.
- [ ] Commit logical stages, push the new branch and open a PR against the verified SB3 integration base. Do not merge or start formal 500-episode training/search.
