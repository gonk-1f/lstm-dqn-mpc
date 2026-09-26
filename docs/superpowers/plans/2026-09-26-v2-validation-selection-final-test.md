# v2 Validation Selection and Final Test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Select one authenticated v3 DQN checkpoint using Validation only, then provide a separate one-time Test evaluator comparing the frozen DQN with fixed action `w_8_1_1`.

**Architecture:** A shared immutable policy evaluator owns episode execution and metrics. A Validation-only selector authenticates and ranks all round checkpoints, copies the winner byte-for-byte, and seals a selection manifest. A separate Test CLI requires that sealed selection, acquires explicit final-Test access, locks a fresh output directory before reading Test, and evaluates DQN and the fixed baseline without learning.

**Tech Stack:** Python 3.11, NumPy, pandas, PyTorch checkpoints, CSV/JSON, `unittest`, existing v2 nonlinear MPC and formal dataset contracts.

---

### Task 1: Shared immutable formal policy evaluator

**Files:**
- Create: `src/v2/evaluation/__init__.py`
- Create: `src/v2/evaluation/formal_policy.py`
- Modify: `src/v2/envs/formal_episode.py`
- Modify: `src/v2/main/train_formal_dqn.py`
- Test: `tests/test_v2_formal_policy_evaluation.py`

- [ ] **Step 1: Write failing evaluator tests**

Add synthetic tests that request the public API:

```python
result = evaluate_formal_policy(
    episodes=(episode_a, episode_b),
    policy=FixedActionPolicy("w_8_1_1"),
)
assert result.episode_ids == ("a", "b")
assert result.completed_episodes == 1
assert result.failed_episodes == 1
assert result.raw_economic_cost_cny == 27.0
assert result.failure_penalty_score == 50_000.0
assert result.learning_reward == -50_027.0
assert result.action_counts == (("w_8_1_1", 2),)
```

Add a greedy-policy fixture whose agent records calls. Assert that
`select_action`, `optimize`, and replay append are never called, while
`greedy_action` is called once per macro transition. Snapshot agent network,
optimizer, replay, and RNG state before/after and require exact equality.

Add an SOC diagnostic test requiring the formal backend to expose an immutable
tuple beginning at `0.60` and containing every successfully executed interval's
post-step SOC. Verify min/max are finite and within `[0.20, 0.80]` for a
successful synthetic episode.

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path "src").Path
.\venv\Scripts\python.exe -X utf8 -B -m unittest -v tests.test_v2_formal_policy_evaluation
```

Expected: import failure for `v2.evaluation.formal_policy` and missing executed
SOC diagnostics.

- [ ] **Step 3: Implement the minimal evaluator**

Create exact frozen result records:

```python
@dataclass(frozen=True)
class EpisodeEvaluation:
    sample_id: str
    completed: bool
    failure_kind: str | None
    transition_count: int
    executed_mpc_steps: int
    h2_cost_cny: float
    fc_degradation_cost_cny: float
    battery_degradation_cost_cny: float
    shore_cost_cny: float
    raw_economic_cost_cny: float
    failure_penalty_score: float
    learning_reward: float
    soc_min: float
    soc_max: float
    action_counts: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class PolicyEvaluation:
    policy_id: str
    episodes: tuple[EpisodeEvaluation, ...]
    # validated aggregate properties derived only from episodes
```

Define `FixedActionPolicy` and `GreedyDqnPolicy` with one exact
`action_index(state)` boundary. Reject unknown actions and non-finite S8.
Implement `build_formal_environment(episode)` in this module and have the
trainer import it as its existing `_environment` name so existing tests and
behavior remain stable.

In `FormalEpisodeBackend.reset`, initialize:

```python
self.executed_soc = [self.INITIAL_SOC]
```

Append each validated `next_state` immediately after committing the interval.
The evaluator snapshots `tuple(backend.executed_soc)` and derives SOC bounds.

- [ ] **Step 4: Run focused and existing trainer tests GREEN**

Run:

```powershell
.\venv\Scripts\python.exe -X utf8 -B -m unittest -v `
  tests.test_v2_formal_policy_evaluation `
  tests.test_v2_formal_episode `
  tests.test_v2_formal_training_cli
```

Expected: all tests pass; real Test payloads remain unopened.

- [ ] **Step 5: Commit**

```powershell
git add src/v2/evaluation src/v2/envs/formal_episode.py src/v2/main/train_formal_dqn.py tests/test_v2_formal_policy_evaluation.py
git commit -m "feat(v2): add immutable policy evaluator"
```

### Task 2: Validation ranking and sealed selection identity

**Files:**
- Create: `src/v2/evaluation/checkpoint_selection.py`
- Test: `tests/test_v2_checkpoint_selection.py`

- [ ] **Step 1: Write failing deterministic ranking tests**

Construct exact candidate metrics and assert this precedence:

```python
winner = select_best_candidate((
    candidate(round_index=8, completed=8, penalty=0.0, raw=100.0),
    candidate(round_index=9, completed=7, penalty=0.0, raw=1.0),
))
assert winner.round_index == 8

winner = select_best_candidate((
    candidate(round_index=8, completed=8, penalty=50_000.0, raw=10.0),
    candidate(round_index=9, completed=8, penalty=0.0, raw=100.0),
))
assert winner.round_index == 9

winner = select_best_candidate((
    candidate(round_index=8, completed=8, penalty=0.0, raw=100.0),
    candidate(round_index=9, completed=8, penalty=0.0, raw=90.0),
))
assert winner.round_index == 9

winner = select_best_candidate((
    candidate(round_index=8, completed=8, penalty=0.0, raw=90.0),
    candidate(round_index=9, completed=8, penalty=0.0, raw=90.0),
))
assert winner.round_index == 8
```

Add tests rejecting duplicate rounds, non-contiguous required rounds, wrong
checkpoint filename/metadata round, altered v3 semantics, and candidate hashes
that are not lowercase SHA-256. Add a manifest round-trip test that recomputes
the canonical digest after deleting only `result_digest`.

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
.\venv\Scripts\python.exe -X utf8 -B -m unittest -v tests.test_v2_checkpoint_selection
```

Expected: import failure for the new selection module.

- [ ] **Step 3: Implement ranking and sealing**

Implement:

```python
def selection_key(value: ValidationCandidate) -> tuple[int, float, float, int]:
    return (
        -value.completed_episodes,
        value.failure_penalty_score,
        value.raw_economic_cost_cny,
        value.round_index,
    )
```

Bind these identities in `SelectionManifest`:

- schema `v2_validation_checkpoint_selection_v1`;
- current power/AIS/mode manifest SHA-256 values;
- `FORMAL_STATE_SCHEMA_DIGEST` and `ACTION_CATALOG_DIGEST`;
- `control_semantics()` and `asdict(FORMAL_FAILURE_POLICY)`;
- ordered candidate metrics and checkpoint SHA-256 values;
- exact ranking rule;
- selected round, selected source hash, copied best hash;
- canonical `result_digest`.

Use strict built-in scalar types and reject non-finite metrics. Provide
`write_selection_outputs` that creates a temporary sibling directory, writes
CSV/JSON and a byte-for-byte checkpoint copy, verifies the copied hash, then
renames into a destination that must not exist.

- [ ] **Step 4: Run tests GREEN**

Run the Task 2 test command and require all tests to pass.

- [ ] **Step 5: Commit**

```powershell
git add src/v2/evaluation/checkpoint_selection.py tests/test_v2_checkpoint_selection.py
git commit -m "feat(v2): seal Validation checkpoint selection"
```

### Task 3: Validation-only selection CLI

**Files:**
- Create: `src/v2/main/select_formal_dqn_checkpoint.py`
- Modify: `src/v2/data/formal_training_dataset.py`
- Modify: `docs/v2_formal_training_runbook.md`
- Test: `tests/test_v2_formal_training_dataset.py`
- Test: `tests/test_v2_validation_selection_cli.py`

- [ ] **Step 1: Write failing CLI boundary tests**

Test parser defaults:

```python
args = _parser().parse_args([])
assert args.checkpoint_dir.name == "v2_formal_dqn_v3"
assert args.output_dir.name == "v2_formal_dqn_selection"
assert args.first_round == 1
assert args.last_round == 40
```

With injected dataset/checkpoint/evaluator fixtures, assert that all 40 files
are visited in numeric order, only `load_validation()` is called, the Test load
method is never called, and the winning checkpoint bytes are copied exactly.
Assert a pre-existing output directory fails before any dataset payload is
opened.

Add a dataset boundary test for `split_episode_ids("train")`: it returns the
authenticated Train sample IDs in ascending manifest order without calling
`_load_episode`, populating `_cache`, or incrementing `opened_test_payloads`.
Reject `"test"` and unknown split names through this public metadata-only API.

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
.\venv\Scripts\python.exe -X utf8 -B -m unittest -v tests.test_v2_validation_selection_cli
```

Expected: import failure for the new CLI.

- [ ] **Step 3: Implement the CLI**

For each checkpoint:

1. hash bytes;
2. create a fresh `DqnAgent(DqnTrainingConfig.formal_baseline(), seed=42,
   device=args.device)`;
3. obtain authenticated ordered Train IDs through the new metadata-only
   `FormalTrainingDataset.split_episode_ids("train")` API and create
   `EpisodeShuffleSchedule` from those IDs without opening Train payloads;
4. call existing `load_checkpoint`;
5. require stored `round_index` to equal the filename round and
   `episode_position == 0`;
6. evaluate the same cached Validation tuple through `GreedyDqnPolicy`;
7. print one progress line with round, completed/failed episodes, penalty,
   raw cost, learning reward, and elapsed time.

After all candidates, rank and atomically write selection outputs. Print the
selected round, Validation metrics, best path, and SHA-256.

Document the command:

```powershell
python -X utf8 -u -m v2.main.select_formal_dqn_checkpoint `
  --checkpoint-dir outputs/v2_formal_dqn_v3 `
  --output-dir outputs/v2_formal_dqn_selection `
  --device cpu
```

- [ ] **Step 4: Run CLI-focused tests GREEN**

Run Task 3 tests plus `tests.test_v2_formal_training_dataset` and require all
to pass.

- [ ] **Step 5: Commit**

```powershell
git add src/v2/main/select_formal_dqn_checkpoint.py src/v2/data/formal_training_dataset.py docs/v2_formal_training_runbook.md tests/test_v2_formal_training_dataset.py tests/test_v2_validation_selection_cli.py
git commit -m "feat(v2): add Validation model selector"
```

### Task 4: Explicit final-Test authority and one-time evaluator

**Files:**
- Modify: `src/v2/data/formal_training_dataset.py`
- Create: `src/v2/evaluation/final_test.py`
- Create: `src/v2/main/evaluate_formal_dqn_test.py`
- Modify: `docs/v2_formal_training_runbook.md`
- Test: `tests/test_v2_final_test_evaluation.py`

- [ ] **Step 1: Write failing authorization and no-overwrite tests**

Keep this existing behavior:

```python
with self.assertRaises(PermissionError):
    dataset.load_split("test")
```

Add synthetic Test-root fixtures and require:

```python
authorization = authenticate_final_test_selection(selection_dir)
episodes = dataset.load_final_test(authorization)
assert tuple(item.split for item in episodes) == ("test",) * 5
assert dataset.opened_test_payloads == 5
```

Reject forged/subclassed authorization, modified selection digest, mismatched
best checkpoint, changed current manifest, and a second call on the same
dataset instance. Test the CLI rejects an existing output destination before
calling `load_final_test`.

Add a synthetic integration test that supplies two Test episodes and asserts
both DQN and fixed policy results use the same ordered IDs. Assert no agent
learning/RNG mutation and exact action counts. Verify JSON/CSV score identity:

```python
learning_reward == -raw_economic_cost_cny - failure_penalty_score
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
.\venv\Scripts\python.exe -X utf8 -B -m unittest -v tests.test_v2_final_test_evaluation
```

Expected: missing authorization type, loader, and CLI.

- [ ] **Step 3: Implement final-Test authorization**

Create an exact frozen `FinalTestAuthorization` bound to:

- selection schema and canonical result digest;
- selected round;
- selected/best checkpoint SHA-256;
- current power/AIS/mode manifest hashes;
- current state/action/control/failure semantics.

`authenticate_final_test_selection` reads and fully revalidates the selection
directory, returning the authorization only after all checks pass.

Add `FormalTrainingDataset.load_final_test(authorization)`. It rejects any
non-exact authorization, permits only one call per dataset instance, loads the
five authenticated Test episodes in sample-ID order, increments
`opened_test_payloads` by five, and does not populate the Train/Validation
cache. `load_split("test")` remains forbidden.

- [ ] **Step 4: Implement the one-time Test CLI and output seal**

Before opening the dataset, the CLI must atomically create the requested output
directory and write `TEST_ACCESS_STARTED.json` containing selection digest,
best hash, and a status of `STARTED`. Existing output refuses execution.

Load the best checkpoint, evaluate the cached Test tuple first with
`GreedyDqnPolicy`, then with `FixedActionPolicy("w_8_1_1")`. Write:

- `test_summary.json`;
- `test_episode_metrics.csv`;
- `test_action_distribution.csv`;
- `run_manifest.json` with canonical result digest and `status=COMPLETE`.

Replace the start marker contents with `status=COMPLETE` only after every file
has been written and hashed. If evaluation fails, leave `status=STARTED` so a
second accidental run remains blocked.

Expose only this command:

```powershell
python -X utf8 -u -m v2.main.evaluate_formal_dqn_test `
  --selection-dir outputs/v2_formal_dqn_selection `
  --output-dir outputs/v2_formal_dqn_test `
  --device cpu `
  --confirm-final-test FINAL_TEST_ONCE
```

Any other confirmation text fails before Test access.

- [ ] **Step 5: Run focused tests GREEN without real Test access**

Run Task 4 tests plus:

```powershell
.\venv\Scripts\python.exe -X utf8 -B -m unittest -v `
  tests.test_v2_formal_training_dataset `
  tests.test_v2_formal_training_cli
```

Expected: all pass using synthetic fixtures; repository Test payload access
count remains zero in training/preflight tests.

- [ ] **Step 6: Commit**

```powershell
git add src/v2/data/formal_training_dataset.py src/v2/evaluation/final_test.py src/v2/main/evaluate_formal_dqn_test.py docs/v2_formal_training_runbook.md tests/test_v2_final_test_evaluation.py
git commit -m "feat(v2): add guarded final Test evaluator"
```

### Task 5: Verification and current Validation selection

**Files:**
- Modify only if verification exposes defects in Task 1-4 files
- Generate locally: `outputs/v2_formal_dqn_selection/`

- [ ] **Step 1: Run focused evaluation tests**

```powershell
$env:PYTHONPATH=(Resolve-Path "src").Path
.\venv\Scripts\python.exe -X utf8 -B -m unittest -v `
  tests.test_v2_formal_policy_evaluation `
  tests.test_v2_checkpoint_selection `
  tests.test_v2_validation_selection_cli `
  tests.test_v2_final_test_evaluation
```

- [ ] **Step 2: Run all v2 and full repository suites**

```powershell
.\venv\Scripts\python.exe -X utf8 -B -m unittest discover -s tests -p "test_v2*.py" -v
.\venv\Scripts\python.exe -X utf8 -B -m unittest discover -s tests -v
```

- [ ] **Step 3: Run live preflight, solver smoke, compile/import, and diff check**

```powershell
.\venv\Scripts\python.exe -X utf8 -u -m v2.main.train_formal_dqn --preflight-only
.\venv\Scripts\python.exe -X utf8 -u -m v2.main.train_formal_dqn --smoke-only
.\venv\Scripts\python.exe -X utf8 -B -m compileall -q src tests
.\venv\Scripts\python.exe -X utf8 -B -c "import v2.evaluation.formal_policy; import v2.evaluation.checkpoint_selection; import v2.evaluation.final_test"
git diff --check
```

- [ ] **Step 4: Run the real Validation-only selector**

Run the Task 3 documented command. This may open Train manifests and Validation
payloads but must report zero Test payloads opened. Do not run the final-Test
command.

Record selected round, completed/failed Validation episodes, total penalty,
raw economic cost, learning reward, output digest, best checkpoint SHA-256,
and elapsed time in the handoff.

- [ ] **Step 5: Re-run selection manifest authentication tests**

Run the focused selection tests again after generating the real selection
artifact. Require no code or manifest drift.

### Task 6: Delete only approved obsolete local outputs

**Files:**
- Delete local ignored directory: `outputs/dqn_mpc_literature_executed_reward_84_v1_review_20260914/`
- Delete local ignored directory: `outputs/v2_formal_dqn/`
- Delete local ignored directory: `outputs/v2_segment_power_review/`

- [ ] **Step 1: Resolve and verify exact deletion targets**

In PowerShell, resolve each literal path and require every result to be a direct
child of the repository `outputs` directory. Abort if any target resolves
outside that directory or equals the output directory itself.

- [ ] **Step 2: Remove only the three approved directories**

Use `Remove-Item -LiteralPath <verified-absolute-path> -Recurse -Force` for each
verified target. Do not use globs or environment variables as deletion targets.

- [ ] **Step 3: Verify preserved artifacts**

Require these paths still exist:

```text
outputs/v2_formal_dqn_v3/round_001.pt
outputs/v2_formal_dqn_v3/round_040.pt
outputs/v2_formal_dqn_selection/best_validation.pt
outputs/v2_dqn_state_audit/audit_manifest.json
outputs/v2_objective_scale_audit/audit_summary.json
outputs/v2_failure_penalty_audit/audit_summary.json
```

Confirm the three approved obsolete directories no longer exist. Report that
they were local ignored artifacts and are not recoverable from Git.

- [ ] **Step 4: Final commit and push**

If documentation or code remains uncommitted after verification:

```powershell
git add docs src tests .gitignore
git commit -m "docs(v2): document final model evaluation"
```

Push `refactor/multiscale-dqn-wmpc-v2` only after fresh verification succeeds.
Do not add model checkpoints or local selection/Test output directories to Git.
