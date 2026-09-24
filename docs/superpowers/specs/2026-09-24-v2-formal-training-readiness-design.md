# V2 Formal Training Readiness Design

## Scope

Build a new, independent v2 DQN training path that is ready to be started by
the user from a PyCharm terminal. The implementation freezes the approved S7
state and the complete canonical 36-action bank, validates the current
`operating_dataset_zero_boundary_v2` index and runtime contracts, and exposes
visible training progress. This task stops before a long-running formal
training job is launched.

The implementation must not call or adapt the legacy 84-action training
entrypoint. It must not change the MPC objective, economic formulas, dataset
split, interpolation policy, or held-out Test data.

## Selected architecture

Create a v2-only training stack under `src/v2/`. Reuse generic numerical or
PyTorch primitives only when they carry no legacy action, state, reward, or
time-scale semantics. The command-line entrypoint must import v2 contracts
directly and must fail closed when any state, action, data, solver, reward, or
artifact identity differs from the frozen baseline.

The legacy `src/main/train_dqn_mpc_mlp.py` remains historical code. It is not a
dependency of the new entrypoint.

## Frozen DQN state

The formal state is the following ordered seven-scalar tuple:

1. `soc`: current battery SOC fraction;
2. `causal_base_load_fraction`: current causal LPF state divided by 600 kW;
3. `load_residual_fraction`: `(P_load - P_base) / 600 kW`;
4. `recent_load_population_std_fraction`: population standard deviation over
   the inclusive 150 s causal load window divided by 600 kW;
5. `recent_load_window_trend_fraction`: least-squares load slope over the
   inclusive causal window, multiplied by 150 s and divided by 600 kW;
6. `fuel_cell_power_fraction`: current executed FC power divided by 600 kW;
7. `fuel_cell_delta_fraction`: current minus previous executed FC power,
   divided by 600 kW.

The state schema receives a stable version identifier and a frozen status.
No Train min-max normalization, clipping, future observation, battery power,
raw load duplication, recent load mean, or recent SOC delta is permitted.

At an episode start, the causal history is cold-started without future data:
the first load value initializes the 150 s history, the LPF initializes from
the first observation, and both current and previous FC power initialize to
the explicitly configured episode FC state. This retains the zero-boundary
start instead of discarding the first 150 s.

## Frozen action catalog

The formal catalog is the complete canonical positive tenth-grid simplex:

```text
(n_base, n_smooth, n_soc), each >= 1, sum = 10
(q_base, q_smooth, q_soc) = numerators / 10
```

The 36 actions retain lexicographic order and canonical IDs
`w_<n_base>_<n_smooth>_<n_soc>`. The status is
`FROZEN_PROJECT_BASELINE`: it is approved for the first formal baseline but is
not claimed to be a screened optimum. The catalog version and digest are
stored in every checkpoint and replay artifact. Integer DQN indices map to
these ordered IDs exactly; no legacy 84-action mapping is accepted.

## Dataset and episode contract

The sole formal dataset is
`data/processed/operating_dataset_zero_boundary_v2`. Startup validation must:

- validate the manifest schema and every relative path;
- verify every segment SHA-256 against the manifest;
- require exactly 53 unique segments split 38 Train, 10 Validation, 5 Test;
- require parent-level split disjointness;
- validate each CSV timestamp and `time_s` index, one-second source cadence,
  finite signed source power, declared point count, start/end timestamps,
  and duration;
- derive the 30 s supervisory index without interpolation or nearest-neighbor
  replacement;
- prevent Test access during training and model selection.

One Train segment is one episode. Every round visits all 38 Train episodes.
The episode order is reshuffled every round by one fixed-seed RNG. Checkpoints
store the RNG state, current permutation, round, episode position, and global
macro step so resume reproduces the uninterrupted order exactly. Validation
uses a fixed manifest order and never writes replay or performs optimizer
updates. Test is not opened by the training command.

Each DQN macro action is held for five real 30 s supervisory steps. A sailing
step performs one rolling MPC solve with `N=5` and executes only its first
command. Shore and idle steps use the explicit mode-aware branches below and
do not invoke the MPC optimizer. A terminal macro may contain one to four
supervisory steps when a segment ends. Episode reset must reset SOC, LPF,
previous FC power, and cumulative FC/battery degradation.

## Signed source-power and shore-mode contract

The frozen dataset intentionally retains negative `load_total_kw` values. They
must not be passed to `NonlinearMPC`, clipped, made absolute, interpolated, or
deleted. The 30 s environment uses the dataset's frozen `1 kW` zero deadband:

- `load_total_kw > 1`: sailing mode; execute MPC;
- `load_total_kw < -1`: modeled shore-charging mode;
- `abs(load_total_kw) <= 1`: idle/deadband mode.

In shore mode, the selected DQN weight action remains held for macro timing
but has no physical control effect. FC power is forced to zero. The magnitude
of the negative signed source power is treated as the available battery-bus
charging power, constrained by the frozen battery charge limit and the room
between current SOC and the fixed episode initial SOC of `0.60`. Accepted
charge updates SOC through the approved aggregate `eta_chg=0.95`; shore cost
uses the corresponding accepted grid-side energy and `1.10 CNY/kWh` exactly
once. The energy channel is classified as modeled, not a measured grid meter.

Idle mode forces zero FC and battery power and has a zero interval ledger.
The causal FC base-load filter is reset when sailing resumes after a shore or
idle block, preventing a negative shore trace from becoming an FC reference.
The signed load history remains causal and available to the S7 load residual,
standard-deviation, and trend features, so shore states are distinguishable.

Shore and idle supervisory steps remain inside the original parent episode
and inside DQN replay; no source CSV or split assignment is changed. Such
steps are action-invariant by design and must be identified in progress and
summary logs. At the end of the parent episode, modeled terminal recharge
charges only any remaining deficit below SOC `0.60`; already accepted interval
shore energy is not charged again.

## Training-volume baseline

The 38 Train segments contain 927,560 s of trace duration. Using complete
30 s intervals gives 30,909 supervisory steps and 6,197 macro transitions per
round, including partial terminal macros. With the frozen 1 kW mode deadband,
one round contains 25,175 sailing/MPC steps, 2,868 modeled shore steps, and
2,866 idle steps.

| Rounds | Macro transitions | Supervisory steps | MPC solves |
| ---: | ---: | ---: | ---: |
| 20 | 123,940 | 618,180 | 503,500 |
| 30 | 185,910 | 927,270 | 755,250 |
| 40 | 247,880 | 1,236,360 | 1,007,000 |
| 50 | 309,850 | 1,545,450 | 1,258,750 |

The default is 30 rounds. It provides about 185,910 macro transitions while
avoiding the unproven compute cost of 40 or 50 rounds. A completed 30-round
checkpoint can be resumed to a higher round budget without changing training
semantics.

## Exploration and optimization baseline

Use epsilon-greedy exploration with a global-macro-step linear schedule:

```text
epsilon_start = 1.0
epsilon_end = 0.05
epsilon_decay_steps = 150000
```

The decay spans about 24.2 Train rounds. The remaining approximately 35,910
macro steps of the default run stay at epsilon 0.05. Logs must name `epsilon`
as the random exploration probability and `greedy_rate = 1 - epsilon` as the
greedy-action probability.

All remaining optimizer and network hyperparameters must be explicit,
validated, serializable configuration values. Defaults may be selected as
project baseline settings, but must not be described as literature-optimal or
copied implicitly from the legacy trainer. The random seed controls Python,
NumPy, PyTorch, episode shuffle, exploration, and replay sampling. Determinism
limits of the selected solver and hardware must be reported rather than
hidden.

## Reward and environment semantics

The environment reward remains the negative sum of the executed macro
interval's raw-CNY ledgers:

```text
-(hydrogen + FC degradation increment + battery degradation increment
  + terminal shore charge if incurred)
```

No MPC objective term is used as DQN reward. No economic formula or interval
degradation semantics is changed. Any numerical reward conditioning used by
the optimizer must be an explicit frozen transform with raw CNY retained in
logs and artifacts; it cannot alter action ranking or economic reports.

## Training runtime and visible output

Add a module entrypoint under `v2.main` with mutually exclusive modes:

- `--preflight-only`: validate contracts and exit without an episode;
- `--smoke-only`: execute a bounded, no-gradient integrated Train smoke and
  write only disposable smoke artifacts;
- normal mode: run or resume formal training.

The normal progress line must expose at least:

- round and episode/segment;
- episode macro step and global macro step;
- action index, canonical action ID, and three MPC weights;
- `epsilon` and `greedy_rate`;
- current raw-CNY macro reward and cumulative episode reward;
- optimizer loss when an update occurs and replay size;
- SOC, load, causal base load, FC power, and FC delta;
- number of executed MPC steps, terminal reason, elapsed time, and checkpoint
  path at checkpoint boundaries.

`--log-every` controls progress frequency and defaults to one macro transition
so PyCharm displays continuous progress. Output flushing is mandatory.

## Preflight and smoke gate

Formal training is enabled only when all existing calibrations plus the
following checks are verified:

- frozen S7 identity, order, dimension, history, and normalization;
- frozen 36-action identity, order, and digest;
- full dataset index/integrity audit;
- finite causal state construction over every Train episode boundary;
- action-to-MPC mapping and macro-step timing;
- signed-load mode classification, interval shore accounting, and no-double-
  charge terminal accounting;
- episode reset, EOL-distance, and fixed initial-SOC contracts;
- representative Train cold/warm solver reproducibility;
- checkpoint/replay round-trip and incompatible legacy rejection;
- one bounded integrated no-gradient smoke.

Unit-test success is not evidence of learned-policy quality. The preflight may
establish implementation readiness only; it must not claim controller
superiority or convergence before training and held-out evaluation.

## Testing strategy

Implementation follows red-green TDD. Focused tests cover state construction,
action freezing, data-index validation, deterministic shuffle/resume,
epsilon scheduling, replay updates, terminal accounting, logging fields,
artifact compatibility, CLI modes, and legacy isolation. Final verification
includes all v2 tests, solver smoke, compile/import, preflight, integrated
no-gradient smoke, and `git diff --check`.

No long-running formal training is launched by this task.

## Cleanup and delivery

Cleanup is limited to reproducible caches, temporary files, disposable smoke
outputs, and outputs directly superseded by this implementation after a
reference check. Tracked v1 code, `archive_v1`, historical audit evidence, and
the user's external `microgrid-mpc-master` directory are not deleted.

After verification, commit and push the branch. The handoff includes exact
PyCharm PowerShell commands for preflight, optional smoke, new training, and
resume, plus the commit SHA and remaining scientific limitations.
