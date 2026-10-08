# V2 Terminal Failure Learning Design

## Scope

Formal v2 DQN training may encounter an MPC state whose frozen five-step
prediction problem is physically infeasible. Such an outcome is a valid failed
episode during exploration; it must not terminate the complete training job and
must not be silently discarded. This design keeps the frozen episode initial
SOC `0.60`, SOC soft band `[0.40, 0.60]`, hard bounds `[0.20, 0.80]`, 36-action
catalog, MPC objective, plant model, dataset and raw economic formulas unchanged.

The observed reproducer is Train episode `zero_boundary_015`: the failing macro
executes two valid 30 s intervals, then `NonlinearMPC` raises
`PhysicalInfeasibilityError` because no five-step SOC trajectory stays above the
hard lower bound. The saved checkpoint is at global step 3017 and contains zero
gradient updates. It is forensic evidence only and is not resumable after this
transition-contract change.

## Alternatives considered

1. **Terminal failure transition with a separate calibrated penalty — selected.**
   Preserve every economic ledger already executed, emit one terminal replay
   transition, penalize the failed decision, checkpoint the episode boundary and
   continue with the next episode. This makes failure learnable without calling
   the penalty a real operating cost.
2. **Skip the failed macro or episode — rejected.** The DQN would receive no
   evidence about the action sequence that depleted feasibility, and early
   termination could appear cheaper than completing the route.
3. **Override the action, relax SOC bounds, delete high-load voyages, or change
   the MPC objective — rejected for this change.** These options alter the
   approved controller or dataset and conceal the failure instead of teaching
   the DQN about it.

## Failure classification

Only `PhysicalInfeasibilityError` is converted into a learnable terminal
failure. It represents a deterministic physical infeasibility under the frozen
plant, state and horizon contracts. `NumericalSolverError`, malformed outputs,
nonfinite values, programming errors and unknown exceptions remain fail-closed
process errors and must not generate fabricated replay samples.

The canonical failure kind is:

```text
physical_mpc_infeasibility
```

The transition records the original exception message, failing supervisory
index, number of already executed MPC intervals, action ID and episode ID in
logs. Exception text is diagnostic metadata and is not part of the network
state.

## Reward and accounting contract

Raw economic accounting remains exactly:

```text
raw_economic_cost_cny
  = H2 cost
  + FC interval degradation cost
  + battery interval degradation cost
  + shore cost
```

For every successful macro:

```text
failure_penalty_score = 0
learning_reward = -raw_economic_cost_cny
```

For a physical-infeasibility terminal macro:

```text
failure_penalty_score = 50_000
learning_reward = -raw_economic_cost_cny - failure_penalty_score
done = True
episode_completed = False
```

The `50,000` score is a frozen project-design penalty calibrated only from
Train: the completed `w_8_1_1` reference sweep has 29 successful episodes and a
maximum full-episode raw economic cost of `20,779.575664249034 CNY`;
`50,000` is more than twice that observed maximum. Classification is
`DERIVED_TRAIN_ONLY / PROJECT_DESIGN`, not measured cost, tariff, equipment
price or literature parameter. The exact calibration inputs, manifests, action
catalog digest and result are stored in a versioned audit artifact and
authenticated by preflight.

`RawCnyIntervalLedger` never receives the penalty. Logs and summaries expose
three separate values:

```text
raw_economic_cost_cny
failure_penalty_score
learning_reward
```

Economic reports aggregate only `raw_economic_cost_cny`. Policy selection may
use `learning_reward` and completion rate. No terminal recharge is fabricated
for an episode that failed before its actual terminal interval.

## Partial macro semantics

The outer M=5 environment retains ledgers from every interval that returned
successfully before the failure. If failure occurs after two executions, the
terminal transition contains those two ledgers and reports
`executed_mpc_steps=2`. If failure occurs before any execution, it contains a
zero ledger and `executed_mpc_steps=0`. Zero executed steps are permitted only
for an explicitly failed terminal transition.

The transition's next state is the finite physical state at the point of
failure. The backend is not rolled back. The environment becomes terminal and
cannot execute another step. The training loop writes the transition to replay,
performs the ordinary update when warmup permits, records the episode as failed,
saves the next-episode checkpoint and continues.

## Training, validation and test behavior

- Train writes failed terminal transitions to replay and updates the network.
- Validation and Test use the identical failure score for comparable returns,
  but never write replay or update the optimizer.
- All three paths separately report completed episodes, failed episodes,
  completion rate, raw economic cost, penalty score and learning return.
- Training success is not required to be 100 percent. The greedy Validation and
  final Test completion rates are explicit controller-quality results and may
  not be hidden, skipped or reported as completed.
- A failed episode counts as one terminal macro transition and one global DQN
  step. The two successfully executed physical intervals in the reproducer
  remain part of that single macro transition.

## Checkpoint and resume contract

The checkpoint version is incremented. The frozen penalty value, provenance
classification and failure-transition schema are included in checkpoint
identity. Older v2 checkpoints are rejected instead of silently mixing replay
semantics. A failed episode is checkpointed at the following episode boundary,
so resume never reruns an already committed failed transition.

The existing `outputs/v2_formal_dqn/latest.pt` checkpoint must be archived or
removed before the new run. It has no gradient updates, so restarting loses no
learned network update.

## Preflight and tests

Preflight authenticates the Train-only penalty calibration artifact and the
new checkpoint/reward schema. It no longer requires every exploratory episode
to complete. It does require that the known `zero_boundary_015` reproducer
terminates deterministically, emits one finite failed transition, preserves the
two executed interval ledgers and permits the following episode to start.

Focused tests cover:

- physical infeasibility after zero and after two successful intervals;
- partial-ledger preservation and no rollback;
- exact separation of raw cost, penalty score and learning reward;
- numerical/programming failures still abort without replay;
- Train replay insertion and continuation to the next episode;
- Validation has no replay/optimizer mutation and reports failure;
- checkpoint version rejection and failed-episode resume boundary;
- deterministic reproduction of `zero_boundary_015` from the frozen dataset;
- progress and summary logs expose completion rate and all three reward fields.

Formal training remains blocked until focused tests, all v2 tests, the full
suite, solver smoke, compile/import and `git diff --check` pass under this
contract.
