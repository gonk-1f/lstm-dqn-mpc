# v2 Validation Checkpoint Selection and Final Test Design

## Objective

Add two strictly separated formal v2 evaluation entrypoints:

1. select one frozen DQN checkpoint using Validation only; and
2. evaluate that frozen checkpoint once on Test against fixed action
   `w_8_1_1`.

Test must never influence checkpoint selection, training, action design, state
design, or any other model-development decision.

## Existing Boundary

Formal training produced `round_001.pt` through `round_040.pt` under checkpoint
schema `v2_formal_dqn_checkpoint_v3`. Every checkpoint uses S8, the frozen
36-action catalog, the terminal physical-infeasibility score, and the current
power/AIS/mode manifests. The training entrypoint evaluated Validation after
each round but printed the metrics only; it did not persist a selection table
or create a best-model artifact. The existing training dataset API rejects
Test access.

## Architecture

### Shared evaluator

A focused evaluation module executes an immutable sequence of
`FormalEpisode` values under either:

- a greedy `DqnAgent`; or
- one fixed canonical action ID.

It returns immutable aggregate and per-episode results. It never appends to
replay, calls `optimize`, consumes epsilon/exploration RNG, shuffles episodes,
or mutates a checkpoint on disk.

Each result records:

- episode completion and failure kind;
- transition and executed MPC-solve counts;
- H2, FC degradation, battery degradation, shore, and total raw CNY costs;
- separate failure penalty score and learning reward;
- minimum and maximum executed SOC diagnostics;
- DQN action counts and fractions, or the fixed action identity.

Raw economic costs and the 50,000 terminal failure score remain separate.

### Validation selector

The Validation selector accepts the directory containing exactly the candidate
round checkpoints. It authenticates every checkpoint against current v3
state/action/reward/failure semantics and verifies that the filename round
matches stored checkpoint metadata.

It evaluates each candidate greedily on the same eight Validation episodes in
manifest order. Candidate ordering is deterministic:

1. more completed episodes;
2. lower total failure penalty score;
3. lower total raw economic cost;
4. earlier round when all preceding values are exactly tied.

The selector writes a new output directory atomically and refuses overwrite.
It produces:

- `validation_checkpoint_metrics.csv`;
- `selection_manifest.json`;
- `best_validation.pt`, an exact byte copy of the selected checkpoint.

The manifest binds the current power/AIS/mode manifest hashes, state schema,
action catalog, reward and failure semantics, ordered candidate checkpoint
hashes, all selection metrics, the ranking rule, selected round, source hash,
best-model hash, and a canonical result digest.

No Test payload is opened during selection.

### Final Test evaluator

The final Test entrypoint is separate from selection. It requires an
authenticated `selection_manifest.json` and a `best_validation.pt` whose
SHA-256 equals the selected source hash. It refuses direct arbitrary-checkpoint
input.

The Test loader is exposed only through an explicit final-evaluation authority,
not through the training/Validation `load_split` path. The Test payload is
loaded once into an immutable episode tuple and reused to simulate:

- the selected greedy DQN policy; and
- fixed action `w_8_1_1`.

The entrypoint creates a new output directory and refuses any existing
destination. It writes:

- `test_summary.json` with side-by-side DQN and fixed-policy aggregates;
- `test_episode_metrics.csv` with all five Test episodes for both policies;
- `test_action_distribution.csv`;
- an authenticated run manifest binding the selection digest, best-model hash,
  Test payload hashes, evaluation code semantics, and result digest.

The Test entrypoint performs no checkpoint selection and makes no claim that
the lower Test cost may be used to revise the model.

## Failure and Integrity Rules

- Incompatible, unreadable, missing, duplicate, or round-mismatched
  checkpoints fail before evaluation.
- A changed current manifest, state/action schema, reward contract, failure
  policy, selection manifest, or best-model hash fails closed.
- Numerical solver errors, nonfinite results, malformed backend values, and
  programming errors abort evaluation; they are not converted to physical
  failure outcomes.
- Deterministic `PhysicalInfeasibilityError` retains the approved terminal
  failure-transition semantics.
- Evaluation outputs are never overwritten.
- Validation and Test episode order is fixed by authenticated manifests and is
  never shuffled.

## Test Strategy

TDD covers:

- deterministic ranking and exact tie-breaking;
- checkpoint filename/metadata/hash authentication;
- selected checkpoint byte identity;
- denial of Test through the training/Validation API;
- denial of final Test without valid selection authority;
- refusal to overwrite prior Test results;
- identical ordered Test episodes for DQN and fixed policy;
- no replay, optimizer, network, or exploration-RNG mutation;
- exact separation of ledger costs, failure penalty, and learning reward;
- action distribution accounting;
- synthetic integration fixtures and current Validation selection.

Development verification must not open the real Test payload. After current
Validation selection completes, report the selected round, metrics, and
SHA-256. The user will explicitly invoke the one-time real Test command in the
PyCharm terminal.

## Cleanup

After implementation and verification succeed, delete only these confirmed
obsolete, ignored output directories:

- `outputs/dqn_mpc_literature_executed_reward_84_v1_review_20260914/`;
- `outputs/v2_formal_dqn/`;
- `outputs/v2_segment_power_review/`.

Preserve all current v2 authenticated audits, data-review evidence,
`outputs/v2_formal_dqn_v3/round_001.pt` through `round_040.pt`, the Validation
selection artifacts, and future final Test results. Unselected current v3
checkpoints are not deleted in this work.

## Acceptance Criteria

- Validation selection uses no Test payload and produces one authenticated
  `best_validation.pt`.
- The selection rule is deterministic and exactly matches the approved metric
  precedence.
- Final Test cannot run without the authenticated selected model and cannot
  overwrite a prior result.
- DQN and `w_8_1_1` are evaluated on the same five ordered Test episodes with
  complete separated metrics.
- Existing formal-training, preflight, economic-ledger, and failure semantics
  remain unchanged.
- Focused tests, all v2 tests, the full repository suite, live preflight,
  solver smoke, compile/import, and `git diff --check` pass before handoff.
