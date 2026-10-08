# V2 Terminal Failure Learning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert deterministic MPC physical infeasibility into a penalized terminal DQN transition while preserving raw economic accounting and allowing formal training to continue with the next episode.

**Architecture:** Add one immutable failure-policy contract and one Train-only calibration artifact. `MultiRateWeightEnvironment` converts only `PhysicalInfeasibilityError`; all numerical and programming failures remain exceptions. Training and evaluation consume `learning_reward`, report raw cost and penalty separately, and bind the changed semantics into v3 checkpoints and preflight.

**Tech Stack:** Python 3, dataclasses, NumPy, PyTorch, unittest, existing v2 nonlinear MPC and manifest-authentication utilities.

---

## File structure

- Create `src/v2/failure_policy.py`: immutable failure kind, penalty and provenance contract.
- Create `src/v2/main/run_failure_penalty_audit.py`: Train-only reproducible reference sweep and JSON writer.
- Modify `src/v2/envs/multirate_weight_env.py`: failed terminal transition construction and partial-ledger preservation.
- Modify `src/v2/main/train_formal_dqn.py`: replay, logging, continuation, validation summaries and new output fields.
- Modify `src/v2/training/artifacts.py`: serialize the changed transition schema.
- Modify `src/v2/training/checkpoint.py`: v3 identity and legacy rejection.
- Modify `src/v2/contracts.py`: increment reward semantics version.
- Modify `src/v2/preflight.py`: authenticate the failure-penalty audit.
- Create `outputs/v2_failure_penalty_audit/audit_summary.json`: frozen generated audit.
- Create `docs/v2_failure_penalty_calibration.md`: evidence and interpretation.
- Modify `docs/v2_formal_training_runbook.md` and `docs/v2_preflight_report.md`: failure behavior and fresh-run command.
- Modify focused `tests/test_v2_*.py` files listed below.

### Task 1: Freeze failure-policy values and provenance

**Files:**
- Create: `src/v2/failure_policy.py`
- Test: `tests/test_v2_failure_policy.py`

- [ ] **Step 1: Write the failing policy tests**

```python
def test_formal_failure_policy_is_separate_train_only_score():
    from v2.failure_policy import FormalFailurePolicy
    policy = FormalFailurePolicy.formal_baseline()
    assert policy.penalty_score == 50_000.0
    assert policy.failure_kind == "physical_mpc_infeasibility"
    assert policy.evidence_status == "DERIVED_TRAIN_ONLY / PROJECT_DESIGN"

def test_failure_policy_rejects_forged_values():
    from v2.failure_policy import FormalFailurePolicy
    with pytest.raises(ValueError):
        FormalFailurePolicy(1.0, "physical_mpc_infeasibility",
                            "DERIVED_TRAIN_ONLY / PROJECT_DESIGN",
                            "v2_train_w_8_1_1_v1")
```

- [ ] **Step 2: Run the test and verify RED**

Run: `python -X utf8 -B -m unittest -v tests.test_v2_failure_policy`

Expected: import failure because `v2.failure_policy` does not exist.

- [ ] **Step 3: Implement the exact immutable contract**

```python
FORMAL_TERMINAL_FAILURE_PENALTY_SCORE = 50_000.0
FORMAL_FAILURE_KIND = "physical_mpc_infeasibility"
FAILURE_EVIDENCE_STATUS = "DERIVED_TRAIN_ONLY / PROJECT_DESIGN"
FAILURE_CALIBRATION_ID = "v2_train_w_8_1_1_v1"

@dataclass(frozen=True)
class FormalFailurePolicy:
    penalty_score: float
    failure_kind: str
    evidence_status: str
    calibration_id: str

    @classmethod
    def formal_baseline(cls) -> "FormalFailurePolicy":
        return cls(50_000.0, FORMAL_FAILURE_KIND,
                   FAILURE_EVIDENCE_STATUS, FAILURE_CALIBRATION_ID)
```

Validate exact types, finiteness, positivity and equality with the four frozen
values. Export one canonical `FORMAL_FAILURE_POLICY` instance.

- [ ] **Step 4: Run the focused test and verify GREEN**

Run: `python -X utf8 -B -m unittest -v tests.test_v2_failure_policy`

Expected: all tests pass.

### Task 2: Emit a failed terminal macro transition

**Files:**
- Modify: `src/v2/envs/multirate_weight_env.py`
- Modify: `tests/test_v2_multirate_env.py`
- Modify: `tests/test_v2_economic_interval_semantics.py`

- [ ] **Step 1: Replace the old physical-failure expectation with RED tests**

Use a backend that returns two ledgers and then raises the real
`PhysicalInfeasibilityError`. Assert:

```python
transition = environment.step("w_2_3_5")
assert transition.done
assert transition.failed
assert not transition.episode_completed
assert transition.executed_mpc_steps == 2
assert transition.ledger.total_cost_cny == 27.0
assert transition.raw_economic_cost_cny == 27.0
assert transition.failure_penalty_score == 50_000.0
assert transition.learning_reward == -50_027.0
assert transition.failure_kind == "physical_mpc_infeasibility"
assert environment.transitions == (transition,)
assert replayed == [transition]
```

Add a zero-execution case with a zero ledger and `executed_mpc_steps == 0`.
Retain a separate `RuntimeError` backend test that still raises
`MacroStepExecutionError` and emits no transition.

- [ ] **Step 2: Run and verify RED**

Run: `python -X utf8 -B -m unittest -v tests.test_v2_multirate_env tests.test_v2_economic_interval_semantics`

Expected: `MacroTransition` has no failure fields and physical infeasibility is
still wrapped as `MacroStepExecutionError`.

- [ ] **Step 3: Implement the minimal transition schema and exception branch**

Change `MacroTransition` to carry:

```python
learning_reward: float
failure_penalty_score: float
failure_kind: str | None

@property
def raw_economic_cost_cny(self) -> float:
    return self.ledger.total_cost_cny

@property
def failed(self) -> bool:
    return self.failure_kind is not None

@property
def episode_completed(self) -> bool:
    return self.done and not self.failed
```

Successful transitions require zero penalty, no failure kind and
`learning_reward == ledger.reward_cny`. Failed transitions require `done=True`,
the canonical kind and penalty, and
`learning_reward == ledger.reward_cny - penalty`. Permit zero executed steps
only for a failed transition.

In `MultiRateWeightEnvironment.step`, catch only
`PhysicalInfeasibilityError`, aggregate all already-returned ledgers (or a zero
ledger), obtain the current finite state, build and commit the terminal failed
transition, notify the replay sink and return normally. Keep the generic error
path unchanged for every other exception.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the Task 2 command and require all tests to pass.

### Task 3: Propagate the schema through artifacts and checkpoints

**Files:**
- Modify: `src/v2/training/artifacts.py`
- Modify: `src/v2/contracts.py`
- Modify: `src/v2/training/checkpoint.py`
- Modify: `tests/test_v2_artifacts.py`
- Modify: `tests/test_v2_checkpoint_resume.py`

- [ ] **Step 1: Write RED round-trip and legacy-rejection tests**

Require serialized transitions to contain:

```python
{
    "learning_reward": -50_027.0,
    "raw_economic_cost_cny": 27.0,
    "failure_penalty_score": 50_000.0,
    "failure_kind": "physical_mpc_infeasibility",
    "episode_completed": False,
}
```

Assert that a v2 checkpoint with the prior checkpoint/reward version is rejected,
while a new checkpoint round-trip restores replay and counters.

- [ ] **Step 2: Run and verify RED**

Run: `python -X utf8 -B -m unittest -v tests.test_v2_artifacts tests.test_v2_checkpoint_resume`

Expected: missing new transition keys and unchanged checkpoint version.

- [ ] **Step 3: Implement serialization and identity changes**

Set:

```python
REWARD_VERSION = "macro_interval_economic_plus_terminal_failure_v2"
CHECKPOINT_VERSION = "v2_formal_dqn_checkpoint_v3"
```

Serialize/deserialise all new fields without deriving penalty from the ledger.
Include `FORMAL_FAILURE_POLICY` values in checkpoint expected identity. Reject
all older versions before restoring agent, schedule or replay state.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the Task 3 command and require all tests to pass.

### Task 4: Continue Train and evaluate Validation without mutation

**Files:**
- Modify: `src/v2/main/train_formal_dqn.py`
- Modify: `tests/test_v2_formal_training_cli.py`
- Modify: `tests/test_v2_formal_episode.py`

- [ ] **Step 1: Write RED integration tests**

Create a bounded synthetic sequence where the first episode returns a failed
transition and the second succeeds. Assert that Train:

```python
assert len(agent.replay) == 2
assert first_done is True
assert second_episode_started is True
assert failure_count == 1
assert completed_count == 1
```

For Validation, assert no replay, optimizer, exploration RNG or shuffle mutation,
and require completion-rate and separated reward fields in output. Add the real
`zero_boundary_015` deterministic reproducer and assert it returns a failed
transition rather than raising.

- [ ] **Step 2: Run and verify RED**

Run: `python -X utf8 -B -m unittest -v tests.test_v2_formal_episode tests.test_v2_formal_training_cli`

Expected: the real reproducer raises and the CLI has no failure summaries.

- [ ] **Step 3: Implement training and evaluation accounting**

Use only `transition.learning_reward` for replay and Bellman updates. Maintain
per-episode and per-round totals for raw cost, penalty score and learning reward.
Log:

```text
raw_economic_cost_cny=...
failure_penalty_score=...
learning_reward=...
episode_completed=YES|NO
failure_kind=NONE|physical_mpc_infeasibility
completion_rate=...
```

After a failed episode, execute the existing episode-boundary checkpoint path
and proceed to the next entry in the frozen permutation. Validation uses the
same transition semantics with greedy actions but performs no replay append or
optimizer update.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the Task 4 command and require all tests to pass.

### Task 5: Generate and authenticate the Train-only calibration audit

**Files:**
- Create: `src/v2/main/run_failure_penalty_audit.py`
- Create: `outputs/v2_failure_penalty_audit/audit_summary.json`
- Create: `docs/v2_failure_penalty_calibration.md`
- Modify: `src/v2/preflight.py`
- Modify: `tests/test_v2_formal_preflight.py`

- [ ] **Step 1: Write RED preflight tests**

Assert preflight is `NO-GO` when the audit is missing, its manifest hash differs,
Test payload count is nonzero, reference action differs, observed maximum differs,
or penalty is not exactly `50_000.0`. Assert valid evidence is reported as
`DERIVED_TRAIN_ONLY / PROJECT_DESIGN`.

- [ ] **Step 2: Run and verify RED**

Run: `python -X utf8 -B -m unittest -v tests.test_v2_formal_preflight`

Expected: preflight has no failure-policy gate.

- [ ] **Step 3: Implement the Train-only audit generator**

The command loads only Train, runs fixed `w_8_1_1`, sums each transition's raw
ledger, records 29 completions, the failed ID `zero_boundary_015`, the observed
maximum `20_779.575664249034`, current manifest hashes, action digest, zero Test
payload opens, penalty `50_000.0`, and a canonical result digest. It writes JSON
atomically.

- [ ] **Step 4: Generate the artifact and freeze its digest**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path "src").Path
python -X utf8 -u -m v2.main.run_failure_penalty_audit
```

Copy the resulting canonical digest into `src/v2/preflight.py`; do not copy data
from Validation or Test.

- [ ] **Step 5: Implement and verify the preflight gate**

Run the Task 5 focused test and require `FORMAL_TRAINING=GO` only when the audit
authenticates exactly.

### Task 6: Documentation, fresh-run protection and final verification

**Files:**
- Modify: `docs/v2_formal_training_runbook.md`
- Modify: `docs/v2_preflight_report.md`
- Modify: `docs/method_v2_multiscale_dqn_wmpc.md`
- Test: all focused and repository tests

- [ ] **Step 1: Update current documentation**

Document that failure rate is permitted in Train, greedy completion rate is a
reported Validation/Test result, and the penalty is excluded from economic
cost. Explicitly state that the old v2 checkpoint is incompatible and the next
run uses a new output directory.

- [ ] **Step 2: Run focused tests**

```powershell
python -X utf8 -B -m unittest -v tests.test_v2_failure_policy tests.test_v2_multirate_env tests.test_v2_economic_interval_semantics tests.test_v2_artifacts tests.test_v2_checkpoint_resume tests.test_v2_formal_episode tests.test_v2_formal_training_cli tests.test_v2_formal_preflight
```

- [ ] **Step 3: Run all v2 and full tests**

```powershell
python -X utf8 -B -m unittest discover -s tests -p "test_v2_*.py"
python -X utf8 -B -m unittest discover -s tests
```

- [ ] **Step 4: Run preflight and failure-aware smoke**

```powershell
$env:PYTHONPATH=(Resolve-Path "src").Path
python -X utf8 -u -m v2.main.train_formal_dqn --preflight-only
python -X utf8 -u -m v2.main.train_formal_dqn --smoke-only
```

Expected: preflight GO; smoke PASS; no formal checkpoint written.

- [ ] **Step 5: Compile, import and check the diff**

```powershell
python -X utf8 -m compileall -q src tests
$env:PYTHONPATH=(Resolve-Path "src").Path
python -X utf8 -c "import v2.failure_policy; import v2.main.train_formal_dqn; print('IMPORT_OK')"
git diff --check
```

- [ ] **Step 6: Give the fresh PyCharm command**

Use a new directory so the incompatible v2 checkpoint remains untouched:

```powershell
python -X utf8 -u -m v2.main.train_formal_dqn --rounds 40 --seed 42 --device cpu --log-every 1 --output-dir outputs/v2_formal_dqn_v3
```

Do not start this long-running command inside the implementation session.
