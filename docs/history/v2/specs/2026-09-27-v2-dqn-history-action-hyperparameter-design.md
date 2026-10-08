# V2 DQN History, Action Audit, and Hyperparameter Study Design

## Goal

Diagnose and correct the single-action collapse observed in the completed v2
formal DQN run without changing the operating dataset, the MPC objective, the
economic model, or the frozen multi-rate timing. The work proceeds in three
ordered stages:

1. expose the Train-only behavioral distinguishability of all 36 MPC weight
   actions;
2. replace the instantaneous S8 DQN input with a causal ten-sample S8 history
   encoded for an MLP; and
3. run a small controlled hyperparameter study that isolates reward scaling
   and learning rate before any broader search.

The existing completed S8 model and its already-opened Test result remain a
historical baseline. They are not overwritten or used to tune the new model.

## Motivation and Current Evidence

The current Double-DQN uses one instantaneous S8 vector, a `128-128` MLP, 36
actions, raw-CNY rewards, `gamma=1.0`, learning rate `1e-4`, batch size 256,
hard target synchronization every 1,000 optimizer updates, and epsilon decay
from 1.0 to 0.05 over 150,000 global macro steps. The selected round-8 model
chose `w_1_7_2` for every Train and Test state inspected even though exploratory
behavior had visited all 36 actions. Its action-centered Q-value spread was
small relative to the common Q-value magnitude and TD-error scale. This is
evidence of value-function collapse or weak action identifiability, not proof
that `w_1_7_2` is globally optimal.

The current Train-only objective audit already evaluates all 36 actions on six
deterministic representative cases, yielding 216 accepted MPC solves. Its core
result object computes behavioral redundancy using explicit tolerances, but
the production report omits that result. This design reuses and extends that
authenticated evidence rather than creating a second incompatible screening
path.

## Literature Basis and Transfer Limits

The design uses the three supplied papers as structural precedent, not as a
source of directly transferable numeric hyperparameters.

### Zarrouki et al. (2024)

The safe PPO-WMPC study first derives candidate MPC weights through
multi-objective optimization and reduces them with clustering before RL. Its
policy is a feed-forward `128-256-128` network, with PPO learning rate decayed
from 0.005 to 0.0001, `gamma=0.8`, and 1.5 million training steps. Its
observation also contains future reference information. The transferable
principles are action distinguishability, safe action preselection, normalized
inputs, and matching the RL switching interval to controlled dynamics. PPO
learning rates and discounting are not copied into the present Double-DQN.
Future measured load from the complete episode is not permitted as an input;
that would leak information unavailable online.

### Yuan et al. (2025)

The fuel-cell hybrid vehicle study uses tabular dual Q-learning, not a neural
DQN. It uses `Np=Nc=5`, a learning rate of 0.99, discount factor 0.01, and up to
500 episodes. It supports the use of a five-step MPC horizon and finite
discrete weight choices as defensible project designs. Its tabular learning
rate and near-myopic discount factor are algorithm-specific and must not be
copied into the current neural-network optimizer.

### Sun et al. (2024)

The adaptive parameterized MPC study is the closest algorithmic reference. Its
DQN example uses a `64-256-64` MLP, learning rate `1e-3`, batch size 512,
replay capacity `1e5`, `gamma=0.99`, soft target update rate 0.01, epsilon from
1.0 to 0.01, and 15-step TD. It supports explicit state normalization, reward
scaling, and testing a higher neural-network learning rate. Its traffic-control
domain, 46-dimensional state, 11 actions, and decision interval differ from
this ship controller, so those values define comparison candidates rather
than defaults.

## Non-Negotiable Boundaries

- Keep `Ts=30 s`, `N_MPC=5`, `DQN_SWITCH_STEPS=5`, and
  `TAU_LPF_SECONDS=90.0` unchanged.
- Keep the current `operating_dataset_zero_boundary_v2` content and
  Train/Validation/Test assignments unchanged.
- Keep the MPC objective, objective normalization, constraints, plant models,
  degradation increment semantics, shore behavior, and raw-CNY ledger
  unchanged.
- Keep the canonical 36-action order, IDs, and weight values during this task.
  The action audit may report redundancy but must not silently publish a new
  catalog or reduce `K`.
- Keep the policy as an MLP. Do not add LSTM, GRU, attention, a load forecast,
  or future measured samples.
- Use Train for action auditing, reward-scale calibration, replay, and
  optimizer updates. Use Validation only for model/checkpoint comparison. Do
  not open Test during development or selection.
- Preserve raw CNY and the 50,000 failure score in all economic reports. A
  scaled training reward is an optimizer input only.

## Stage 1: Train-Only Action Behavior Audit

### Evidence source

Extend the existing objective-scale audit runner and result serialization. It
must retain the current authenticated Train manifest, raw-source inventory,
six representative case IDs, 36 canonical actions, 216 accepted solves,
solver reproducibility checks, and current behavior tolerances:

- objective absolute tolerance: `1e-12`;
- power absolute tolerance: `1e-6 kW`;
- SOC absolute tolerance: `1e-10`.

Validation and Test cannot participate in representative-case selection,
threshold selection, or result generation.

### Required outputs

Persist, for every representative case and action:

- canonical action ID and exact weight triple;
- objective components `J_base`, `J_smooth`, and `J_soc`;
- first FC command and first battery command;
- complete five-step predicted SOC path; and
- solver feasibility and repeated-solve consistency.

Persist the behavioral-redundancy pairs already computed by the audit core.
Also report, per representative case:

- number of distinct behavioral equivalence classes;
- class membership and deterministic representative ID;
- maximum and percentile pairwise differences in first FC power, first battery
  power, and predicted SOC; and
- actions whose objective weights differ but whose resulting control behavior
  is indistinguishable at the frozen tolerances.

The output is diagnostic. It answers whether the 36 actions are locally
distinguishable on the approved Train cases. It does not claim global
closed-loop equivalence and cannot mutate `FINAL_DQN_ACTION_CATALOG`.

### Interpretation gate

- If most cases contain several distinct behavior classes, proceed with all 36
  actions and treat policy collapse primarily as a learning/state problem.
- If most actions belong to the same class on most cases, record action
  identifiability as the primary limitation. The present task still retains 36
  actions; any later catalog reduction requires a separately approved,
  versioned Train-only screening decision.
- A single learned greedy action is not itself an error if it has a stable,
  materially positive Q margin and superior Validation economics. Action
  diversity is never optimized for its own sake.

## Stage 2: Ten-Sample Causal History State

### Base observation

Keep the current ordered S8 frame unchanged:

1. `soc`;
2. `causal_base_load_fraction`;
3. `load_residual_fraction`;
4. `recent_load_population_std_fraction`;
5. `recent_load_window_trend_fraction`;
6. `fuel_cell_power_fraction`;
7. `fuel_cell_delta_fraction`; and
8. `speed_fraction`.

The existing fixed physical normalizations remain authoritative. No Train
min-max normalization or clipping is introduced.

### Stack layout

At every DQN decision boundary, encode the latest ten valid 30-second
supervisory S8 frames in chronological order, oldest to newest:

```text
[S8(k-9), S8(k-8), ..., S8(k), mask(k-9), ..., mask(k)]
```

The feature portion contains `10 * 8 = 80` floats and the mask contains ten
floats, giving a 90-dimensional MLP input. Ten sample instants span 270 seconds
from the oldest timestamp to the current timestamp. This history length is
independent of both MPC horizon `N=5` and action hold count `M=5`.

Missing pre-episode history is left-padded with zero S8 frames and zero mask
values. Every real frame has mask value 1. The mask is required because a
physical normalized observation can legitimately contain zeros.

### Update and reset semantics

- Record one S8 frame for every successfully executed ONBOARD supervisory
  interval, including the four internal intervals that do not emit a replay
  transition.
- Constructing or inspecting a state must be side-effect free; repeated calls
  at the same decision boundary cannot duplicate a frame.
- A DQN macro transition continues to contain only one state and one next
  state. The replay cadence remains one transition per action decision, not
  one transition per S8 frame.
- Reset the stack at episode reset.
- Reset the stack whenever the formal backend enters shore or idle mode, at
  the same boundary that resets the causal onboard history and LPF. The first
  later ONBOARD decision therefore contains only its current real frame and
  nine padded frames.
- A terminal state must still have the exact 90-dimensional schema, although
  Double-DQN masks its bootstrap value when `done=True`.
- A physical infeasibility may terminate with a valid 90-dimensional next
  state, but failed plant execution cannot fabricate an additional successful
  history frame.

The stack encoder is a small stateful component owned by the formal episode
backend. The S8 frame builder remains responsible only for one causal physical
frame. This separation prevents MLP history bookkeeping from contaminating
physical state construction.

### Schema and compatibility

Define a new state-schema version and digest that bind:

- the eight base feature names and order;
- history length 10;
- chronological flattening order;
- mask order and padding value;
- reset semantics; and
- total dimension 90.

The old S8 checkpoints, replay buffers, selection artifacts, and Test result
are incompatible with the new schema and must be rejected rather than
silently loaded. New runs use a new output namespace. Historical artifacts are
retained read-only for comparison.

## Stage 3: Controlled Hyperparameter Study

### Reward-scale calibration

Generate one sealed Train-only reward-scale artifact before experimental
training. Reuse the authenticated fixed-action `w_8_1_1` Train sweep and
collect the raw total cost of every emitted 30-second interval ledger,
including zero-cost intervals and excluding the separate failure penalty. The
existing `positive_arithmetic_mean_v1` rule produces `C_ref` from that exact
immutable tuple. Bind dataset version, Train provenance, source hashes,
reference action, sample count, derivation rule, and artifact digest.

For scaled experiments only, replay stores:

```text
r_train = transition.learning_reward / C_ref
```

This divides both the raw economic reward and any 50,000 failure score by the
same positive constant. It therefore preserves their ordering and relative
meaning. `MacroTransition.learning_reward`, ledgers, Validation ranking,
checkpoint summaries, and all published economic results remain in raw CNY or
the separately labelled raw failure score.

### First experiment matrix

All new candidates use the 90-dimensional ten-frame state, the complete 36
actions, seed 42, the `128-128` MLP, Double-DQN, one-step TD, Huber loss,
gradient clipping at 10, `gamma=1.0`, batch size 256, replay capacity 200,000,
warmup 5,000, hard target synchronization every 1,000 optimizer updates, and
the current global-step epsilon schedule.

| ID | Replay reward | Learning rate |
| --- | --- | ---: |
| H1 | raw `learning_reward` | `1e-4` |
| H2 | `learning_reward / C_ref` | `1e-4` |
| H3 | `learning_reward / C_ref` | `3e-4` |
| H4 | `learning_reward / C_ref` | `1e-3` |

This matrix first isolates history, reward scale, and optimizer step size. It
does not simultaneously change network width, discount semantics, batch size,
target-update rule, or TD horizon.

### Pilot and continuation protocol

Run each candidate for ten complete Train rounds. Each round visits every
Train episode exactly once in a fixed-seed reshuffled order. Validation remains
ordered, greedy, read-only, and optimizer-free after every round. Select each
candidate's best pilot checkpoint by the existing lexicographic rule:

1. higher completed-episode count;
2. lower failure-penalty score; and
3. lower raw economic cost.

Rank the four pilot candidates by the same rule and continue the best two
candidate runs from their exact round-10 `latest` checkpoints to round 40.
Resume must preserve replay, optimizer, target network, epsilon global step,
episode-shuffle RNG, and all schema/config digests. Select the final checkpoint
from the two completed candidates using Validation only.

No pilot or continuation run may open Test. Because the original five Test
episodes have already been opened for the historical S8 model, a later reuse
of them for the new model is a labelled reused benchmark, not a fresh unbiased
final test. A genuinely new final claim requires a separately frozen unseen
holdout; creating one is outside this task.

### Required diagnostics

For every Train round and greedy Validation evaluation, record:

- raw economic cost, failure score, completion counts, and scaled training
  reward when applicable;
- epsilon and `1-epsilon` with their correct meanings;
- behavior action counts and greedy action counts separately;
- greedy unique-action count, maximum action share, Shannon entropy, and
  action distribution by causal operating regime;
- Q-value common-mode mean, centered action-advantage spread, top-1/top-2
  margin quantiles, and finite-value checks;
- TD-error absolute P50, P95, and maximum, Huber loss, optimizer update count,
  and gradient norm before clipping; and
- state schema, action catalog, reward-scale, dataset, seed, and training
  configuration identities.

Action entropy is diagnostic, not a model-selection objective. A policy is not
rewarded merely for choosing more actions.

### Deferred secondary ablations

Only if the first matrix remains numerically unstable or action-collapsed may a
later separately approved study test one change at a time:

- batch size 512;
- soft target update rate 0.01;
- `gamma=0.99`;
- three-step or five-step TD; or
- a `128-256-128` MLP.

The literature values do not authorize combining these changes or replacing
the economic objective without local evidence. Yuan's `gamma=0.01` and
learning rate 0.99 are explicitly excluded from neural-DQN candidates.

## Failure Handling

- Deterministic physical infeasibility retains the current terminal-transition
  contract and 50,000 failure score.
- Numerical errors, non-finite state/reward/Q values, schema mismatches,
  artifact-digest mismatches, and programming errors remain fail-closed
  exceptions; they are not converted into RL experience.
- A failed action-audit solve prevents a clean 36-action diagnostic result and
  must be reported by action and representative case.
- Hyperparameter runs write to distinct directories and cannot resume from a
  checkpoint with a different state schema, reward mode, learning rate, seed,
  action digest, or dataset digest.

## Testing Strategy

Use test-driven implementation. Focused tests must prove:

1. action audit serialization includes all 216 observations and behavioral
   redundancy without reading Validation or Test;
2. ten ordered S8 frames plus ten masks produce exactly 90 finite values;
3. left padding and masks distinguish missing history from real zero values;
4. repeated state reads do not duplicate history;
5. all internal ONBOARD supervisory steps update history even though replay is
   emitted only at macro boundaries;
6. episode, shore, and idle boundaries reset history and prevent cross-mode
   leakage;
7. `N=5`, `M=5`, and the history length remain independent;
8. old S8 checkpoints/replay are rejected by the new schema identity;
9. reward scaling uses authenticated Train-only 30-second ledger costs and
   applies the same divisor to the failure score;
10. raw ledgers, raw CNY validation ranking, and failure-score reports are
    unchanged by scaling;
11. the four experiment configurations differ only in their declared reward
    mode and learning rate;
12. pilot ranking and continuation never open Test and resume deterministically;
13. Q, TD, gradient, action-count, and entropy diagnostics are deterministic on
    controlled fixtures; and
14. existing economic, shore, dataset, solver, and interval-degradation tests
    remain green.

Final verification includes focused tests, all v2 tests, solver smoke,
compile/import checks, `git diff --check`, and a bounded training smoke for raw
and scaled reward modes. Long pilot or formal training is started by the user,
not by the implementation task.

## Expected Deliverables

- an authenticated expanded Train-only action-behavior audit and report;
- a versioned 90-dimensional history-state contract and encoder;
- checkpoint and replay compatibility guards for the new schema;
- a sealed Train-only reward-scale artifact and raw/scaled replay modes;
- an independent hyperparameter-study entrypoint/configuration with visible
  progress and diagnostics;
- a runbook containing PyCharm PowerShell commands for the action audit,
  four pilots, top-two continuation, and Validation-only selection; and
- no dataset rebuild, no action-catalog mutation, no Test execution, and no
  automatic long-running training.
