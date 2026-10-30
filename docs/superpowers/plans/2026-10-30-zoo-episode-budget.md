# Zoo Episode Budget Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Independent file owners implement the algorithm, Zoo manager/CLI, and callbacks; the root coordinates real-data verification, docs, review and delivery.

**Goal:** Add exact formal-episode training and episode-based epsilon to PR #5, preserving the native Zoo/SB3 timestep route and every physical/economic contract.

**Architecture:** The existing masked SB3 subclass adds post-Replay episode accounting and a collection boundary before another action. The inherited SB3 `learn()` still dispatches optimization. A post-collection/update observer drives episode Validation. Official Zoo setup, objective and Optuna trial management remain responsible for experiments. User's attached specification is the approved design and authorization; no additional design approval is required.

**Tech Stack:** RL Zoo 2.9.0, SB3/contrib 2.9.2, Gymnasium 1.2.3, Optuna 4.9.0, PyTorch CPU, native Logger/CSV/TensorBoard.

## Decisions and invariants

- Work only in `.worktrees/v6-zoo-vessel`, branch `feat/v6-zoo-vessel`, starting at verified PR head `7df880d`. Do not edit `src`, physical/data files, dependencies, old experiments or checkpoints.
- Require an explicit mutually exclusive `--n-episodes [N]` or `--n-timesteps N`; selecting `--n-episodes` without N means 500. Preserve explicit exploration choices. The defaults are episode scheduling for an episode budget, native timestep scheduling for a timestep budget.
- Warmup counts actual transitions only. The voyage containing the warmup boundary retains epsilon 1 and is excluded from formal episode counts. The first later natural reset starts formal episode 1. Post-warmup transitions in the bridge remain in Replay and retain the original formal Target/Adam clocks.
- Count successful and physical-failure terminals once, after Replay insertion. Never invent truncation/done/action. Stop before the next action after N formal terminals. A partial final 32-step block has no extra update; a full one keeps its normal update. Every 10000 formal stored steps still synchronizes Target before any coincident Adam.
- Reuse the native collector and inherited `learn`; avoid a copied physics/training loop. A private collector boundary may interrupt before the next action, after native terminal bookkeeping; report the actual rollout size and drain post-update observers.
- Epsilon in episode mode is fixed within each voyage; for target 500 use `max(.05, 1-.95*completed/400)`. Save the selected budget/schedule and all clocks.
- Episode Validation runs after the relevant stored transition and any due Adam at formal terminals 50,100,...,500, plus the final terminal for small budgets. Use the independent eight-voyage evaluator and its own masks, preserve RNG and training counters. Keep timestep Validation unchanged.
- The native total-step ceiling in episode mode is explicitly a mathematical safety bound (warmup plus enough maximum voyage horizons), never a timesteps approximation for N episodes. Reaching it without N terminals raises a fatal error. Explicit short safety ceilings support failure tests.
- Official Optuna objective/scheduler remains active. Record actual trial budget, steps, Adam, parameters and validation result. COMPLETE requires the exact requested formal-episode count. Pruning occurs after storage/update at a validation boundary.
- Existing outputs are immutable. New bounded real runs go to fresh `outputs/zoo_vessel_dev` directories. Only small text evidence is committed.

## Tasks

- [x] Algorithm RED tests: crossing warmup, three formal episodes, final terminal Replay/mask/no-bootstrap, partial/full32, formal Target, epsilon invariance and 400/500 schedule. Run with `.venv-zoo\Scripts\python.exe -m pytest tests/test_zoo_episode_algorithm.py -q` and retain the expected failures before implementation.
- [x] Implement the minimal collector/accounting/schedule extension in `algorithm.py` (and a focused budget helper if needed), then make algorithm and retained algorithm tests pass.
- [x] Manager/CLI RED tests: mutually exclusive explicit budgets, schedule defaults/explicit overrides, protective bound failure, official objective and exact trial completion. Implement only `manager.py`/`launch.py`, then pass new and retained manager tests.
- [x] Callback RED tests: post-update episode Validation, every50/final500 event simulation, deterministic masks/state isolation and native log fields. Implement only `callbacks.py`, then pass new and retained callback tests.
- [x] Preserve timestep behavior with a baseline-vs-current synthetic run comparing weights, Replay, epsilon, Adam and Target history. Do not change YAML numerical hyperparameters.
- [x] Run the integrated unit/regression suite under the existing isolated `.venv-zoo`, CPU and one numerical thread. Run `pip check`; install or alter no dependencies.
- [x] Run fresh bounded real Train tests: small exact episode budget, timestep regression and one official episode-budget Optuna Trial. Independently load/evaluate the saved model on all eight Validation voyages. Verify CSV, TensorBoard and actual native tables, Replay terminal/action masks and counters. No Test payloads or formal 500 training/search.
- [x] Update README, YAML comments and an independent dated audit report with actual commands/results/limits. Include the five requested Windows command routes and distinguish episode/step budget and epsilon schedule.
- [x] Review the diff and source/output isolation, run required final checks, commit only source/config/docs/tests/small evidence, push the same branch and update draft PR #5. Verify remote head and keep PR open/unmerged. Stop after delivery.

## Completion evidence

Record exact test commands, exit codes and outputs; factual bounds include physical failures as legitimate episode ends. Never describe N ended episodes as N successful voyages. Any incomplete safety-budget run is FAIL, not COMPLETE. Best Validation actor and final trained actor are saved and identified separately.
