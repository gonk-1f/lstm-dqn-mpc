# V2 Formal Training Readiness Implementation Plan

> Execute this plan in the current branch with red-green TDD. Do not start the
> long formal training run. Do not import the legacy 84-action trainer.

**Goal:** Deliver a validated, resumable v2-only DQN training entrypoint using
the frozen S9 state, complete 36-action catalog, signed shore accounting, and
the current parent-safe dataset plus AIS sidecar.

**Architecture:** Keep the immutable power dataset and split unchanged. Build a
separate authenticated 30 s AIS sidecar from raw parent AIS, then join it only
inside the v2 formal loader. A v2 episode backend owns mode classification,
plant state, MPC calls, and interval ledgers. A small PyTorch DQN layer owns
epsilon scheduling, replay, updates, checkpointing, deterministic shuffle, and
visible CLI progress.

**Baseline:** Python, NumPy, pandas, SciPy PCHIP, PyTorch, existing v2 nonlinear
MPC and economic models.

---

## Task 1: AIS supervisory sidecar

**Files:**

- Create: `src/v2/data/ais_speed_sidecar.py`
- Create: `src/v2/main/build_ais_speed_sidecar.py`
- Create: `tests/test_v2_ais_speed_sidecar.py`
- Generate: `data/processed/operating_dataset_zero_boundary_v2_ais/`

1. Write red tests for latest-causal alignment, no source reuse, bounded PCHIP
   gap fill, no extrapolation, provenance labels, manifest hashes, all 53
   segments, and split preservation.
2. Implement the pure sidecar builder and CLI with atomic output creation.
3. Run the focused tests and build the checked-in sidecar from the raw
   `氢舟一号` folder.
4. Verify 38/10/5 identities, 30,909 Train interval rows, finite nonnegative
   speed, Train raw max 15.9 kn, and expected provenance counts.

## Task 2: Freeze S9 state and 36 actions

**Files:**

- Modify: `src/v2/dqn/state.py`
- Modify: `src/v2/dqn/action_space.py`
- Modify: `src/v2/dqn/__init__.py`
- Modify: `src/v2/contracts.py`
- Modify: `tests/test_v2_state_and_economics.py`
- Modify: `tests/test_v2_action_space.py`
- Create: `tests/test_v2_formal_state.py`

1. Write red tests for exact S9 order, dimension, speed/20 normalization,
   binary shore flag, causal cold start, no clipping, and stable schema digest.
2. Write red tests freezing all 36 lexicographic actions, IDs, weights, digest,
   and rejecting legacy or reordered catalogs.
3. Implement the minimum formal schema/catalog while retaining candidate audit
   helpers for historical evidence.
4. Run focused state/action tests.

## Task 3: Formal dataset loader and deterministic episode schedule

**Files:**

- Create: `src/v2/data/formal_training_dataset.py`
- Create: `src/v2/training/schedule.py`
- Create: `tests/test_v2_formal_training_dataset.py`
- Create: `tests/test_v2_training_schedule.py`

1. Write red tests for power/AIS manifest authentication, exact 30 s left
   boundaries, signed load retention, Test payload non-access, fixed-seed
   per-round Train shuffle, fixed Validation order, and exact 6,197 macro
   transitions per round.
2. Implement fail-closed loaders and serializable RNG/permutation state.
3. Run focused tests against synthetic fixtures and the checked-in manifests.

## Task 4: Mode-aware formal episode backend

**Files:**

- Create: `src/v2/envs/formal_episode.py`
- Modify: `src/v2/envs/multirate_weight_env.py`
- Create: `tests/test_v2_formal_episode.py`
- Modify: `tests/test_v2_multirate_env.py`

1. Write red tests proving negative signed power sets `shore_connected=1`, FC
   is forced to zero, MPC is never called, positive stationary hotel load is
   not shore, idle is action-invariant, AIS alone cannot enable/disable shore,
   and five interval ledgers sum once.
2. Test SOC charging, accepted grid energy, shore cost, terminal remaining
   recharge, mode-transition LPF reset, episode reset, and partial terminal
   macros.
3. Implement the minimum backend and authenticated production factory.
4. Run focused environment/economic tests.

## Task 5: Close formal model and preflight contracts

**Files:**

- Modify: `src/v2/models/fuel_cell_degradation.py`
- Modify: `src/v2/preflight.py`
- Modify: `src/v2/main/run_preflight.py`
- Modify: `tests/test_v2_formal_preflight.py`
- Modify: `tests/test_v2_degradation_models.py`

1. Write red tests for the authorized aggregate-equivalent FC degradation
   mapping, bounded EOL distance, frozen S9/action/AIS/data identities, and
   remaining integrated solver/smoke blockers.
2. Implement the explicit project-model 600/100 aggregate mapping without
   describing it as vessel-measured.
3. Make formal preflight GO only after dataset, solver, checkpoint, and smoke
   checks pass; preserve evidence classification separately from configuration
   freeze.
4. Run focused preflight/degradation tests.

## Task 6: V2 DQN core and resumable artifacts

**Files:**

- Create: `src/v2/training/dqn.py`
- Create: `src/v2/training/replay.py`
- Create: `src/v2/training/checkpoint.py`
- Modify: `src/v2/training/artifacts.py`
- Create: `tests/test_v2_dqn_training.py`
- Create: `tests/test_v2_checkpoint_resume.py`

1. Write red tests for the 9-to-128-to-128-to-36 network, Double-DQN target,
   Huber loss, gradient clipping, replay capacity 200,000, batch 256, warmup
   5,000, target sync 1,000, gamma 0.99, learning rate 1e-4, and deterministic
   seeds.
2. Test epsilon as a global macro-step linear schedule 1.0 to 0.05 over
   150,000 steps and log `greedy_rate=1-epsilon`.
3. Test exact resume of networks, optimizer, replay, Python/NumPy/Torch RNG,
   shuffle permutation, episode position, and global counters; reject legacy
   84-action/S7/S10 artifacts.
4. Implement and run focused tests.

## Task 7: Independent v2 CLI and visible progress

**Files:**

- Create: `src/v2/main/train_formal_dqn.py`
- Create: `tests/test_v2_formal_training_cli.py`
- Modify: `README.md` or the existing v2 runbook if present

1. Write red tests for mutually exclusive `--preflight-only`, `--smoke-only`,
   normal, and `--resume` paths; assert the CLI never imports the legacy
   trainer and never opens Test payloads.
2. Implement default 30 rounds, seed 42, one-line flushed progress containing
   round/episode/macro/global step, action ID/weights, epsilon/greedy rate,
   reward, loss, replay, SOC/load/base/FC/delta, mode counts, elapsed time, and
   checkpoint path.
3. Add bounded no-gradient smoke and atomic latest/periodic checkpoints.
4. Run CLI focused tests, preflight-only, and smoke-only. Do not start normal
   formal training.

## Task 8: Full verification, cleanup, commit, and push

1. Run focused tests from Tasks 1-7.
2. Run all v2 tests.
3. Run the existing nonlinear-MPC solver smoke.
4. Run compile/import checks for all new modules.
5. Run `git diff --check` and inspect the complete diff.
6. Remove only reproducible caches, temporary builder directories, and
   disposable smoke artifacts; preserve historical evidence and v1 code.
7. Commit using concise Conventional Commit messages and push
   `refactor/multiscale-dqn-wmpc-v2`.
8. Report the final SHA and exact PyCharm PowerShell commands for preflight,
   smoke, new training, and resume.
