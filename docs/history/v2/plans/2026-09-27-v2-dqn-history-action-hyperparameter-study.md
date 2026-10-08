# V2 DQN History, Action Audit, and Hyperparameter Study Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an auditable 36-action behavior report, a causal ten-frame S8 MLP state, and a Train/Validation-only four-profile DQN study without changing data, MPC, economics, or Test payloads.

**Architecture:** Reuse the existing six-case/36-action objective audit and expose its omitted behavior evidence. Keep one-frame S8 construction pure, wrap it in a dedicated ten-frame history encoder owned by `FormalEpisodeBackend`, and bind the resulting 90-dimensional schema into replay/checkpoints. Add a sealed Train-only reward-scale artifact and four immutable experiment profiles; the existing v2 trainer executes profiles while a separate selector ranks pilots using Validation only.

**Tech Stack:** Python 3.11, NumPy, pandas, PyTorch Double-DQN, pytest/unittest, SciPy nonlinear MPC, JSON/CSV artifacts, PowerShell CLI.

---

## File Map

### New files

- `src/v2/dqn/history.py` — pure ten-frame history buffer and 90-value encoder.
- `src/v2/training/reward_scaling.py` — authenticated calibration-document loader and raw/scaled replay reward adapter.
- `src/v2/training/experiments.py` — immutable H1-H4 profile catalog and identity validation.
- `src/v2/training/diagnostics.py` — Q/TD/gradient/action-distribution summaries.
- `src/v2/main/run_reward_scale_calibration.py` — Train-only fixed-policy calibration artifact generator.
- `src/v2/main/train_history_dqn_study.py` — explicit study CLI that delegates to the v2 training loop.
- `src/v2/evaluation/study_selection.py` — pilot/final cross-profile Validation ranking primitives.
- `src/v2/main/select_history_dqn_study.py` — Validation-only study selector and continuation manifest.
- `tests/test_v2_history_state.py`
- `tests/test_v2_reward_scale_calibration.py`
- `tests/test_v2_dqn_experiments.py`
- `tests/test_v2_dqn_diagnostics.py`
- `tests/test_v2_history_study_cli.py`
- `tests/test_v2_history_study_selection.py`

### Modified files

- `src/v2/analysis/train_objective_scale_runner.py` — derive deterministic action behavior groups and pairwise statistics.
- `src/main/run_v2_objective_scale_audit.py` — serialize action observations, tolerances, redundancy, and groups.
- `src/v2/dqn/state.py` — separate S8 frame identity from the new 90-dimensional formal state identity.
- `src/v2/envs/formal_episode.py` — own/reset/commit the history encoder and expose interval ledgers for calibration.
- `src/v2/envs/multirate_weight_env.py` — enforce the new formal state dimension without changing macro timing.
- `src/v2/training/dqn.py` — accept the 90-dimensional schema and return structured optimizer diagnostics.
- `src/v2/training/checkpoint.py` — bump version and bind experiment/reward-scale identity.
- `src/v2/main/train_formal_dqn.py` — reusable profile-aware training core; raw CNY reporting remains unchanged.
- `src/v2/evaluation/checkpoint_selection.py` — authenticate the new checkpoint version and experiment identity.
- `tests/test_v2_objective_scale_audit.py`
- `tests/test_v2_train_objective_scale_runner.py`
- `tests/test_v2_formal_state.py`
- `tests/test_v2_formal_episode.py`
- `tests/test_v2_multirate_env.py`
- `tests/test_v2_dqn_training.py`
- `tests/test_v2_checkpoint_resume.py`
- `tests/test_v2_formal_training_cli.py`
- `tests/test_v2_checkpoint_selection.py`
- `docs/v2_objective_scale_audit.md`
- `docs/v2_formal_training_runbook.md`

Do not modify dataset CSVs/manifests, `src/v2/control/nonlinear_mpc.py`, economic formulas, degradation models, shore classification, or `src/v2/dqn/action_space.py`.

## Task 1: Persist the Existing 36-Action Behavior Evidence

**Files:**
- Modify: `src/v2/analysis/train_objective_scale_runner.py`
- Modify: `src/main/run_v2_objective_scale_audit.py`
- Modify: `tests/test_v2_train_objective_scale_runner.py`
- Modify: `tests/test_v2_objective_scale_audit.py`

- [ ] **Step 1: Write failing behavior-summary tests**

Add fixture observations for three actions where two satisfy the frozen behavior tolerance and one differs. Require deterministic connected groups, canonical action-ID ordering, and pairwise power/SOC statistics:

```python
def test_action_behavior_summary_groups_redundant_actions_deterministically(self):
    from v2.analysis.objective_scale_audit import BehaviorTolerance, run_objective_scale_audit
    from v2.analysis.train_objective_scale_runner import summarize_action_behavior
    from v2.dqn.action_space import CANDIDATE_ACTION_BANK

    result = run_objective_scale_audit(
        provenance=self._provenance(),
        cases_loader=self._cases,
        actions=CANDIDATE_ACTION_BANK,
        solver_runner=lambda case, action: self._plan(
            action,
            (1.0, 1.0, 1.0),
            case_id=case.case_id,
        ),
        behavior_tolerance=BehaviorTolerance(1e-12, 1e-9, 1e-12),
    )
    summary = summarize_action_behavior(result)

    case = summary.cases[0]
    self.assertEqual(case.case_id, "state-a")
    self.assertEqual(case.behavior_groups, (tuple(action.action_id for action in CANDIDATE_ACTION_BANK),))
    self.assertEqual(case.distinct_behavior_count, 1)
    self.assertEqual(case.max_first_fc_difference_kw, 0.0)
    self.assertGreaterEqual(case.p95_first_battery_difference_kw, 0.0)
    self.assertGreaterEqual(case.max_soc_path_difference, 0.0)
```

Add a runner-level test that asserts the JSON-ready summary contains exactly 216 observations, six cases, all 36 action IDs per case, frozen tolerances, behavior groups, and no held-out provenance.

- [ ] **Step 2: Run the focused tests and verify red**

Run:

```powershell
$env:PYTHONPATH=(Resolve-Path "src").Path
python -m pytest tests/test_v2_objective_scale_audit.py tests/test_v2_train_objective_scale_runner.py -q
```

Expected: FAIL because `summarize_action_behavior` and serialized behavior fields do not exist.

- [ ] **Step 3: Add immutable behavior-report values**

Implement exact dataclasses and one deterministic graph-component derivation in `train_objective_scale_runner.py`:

```python
@dataclass(frozen=True)
class CaseBehaviorSummary:
    case_id: str
    behavior_groups: tuple[tuple[str, ...], ...]
    distinct_behavior_count: int
    pair_count: int
    max_first_fc_difference_kw: float
    p95_first_fc_difference_kw: float
    max_first_battery_difference_kw: float
    p95_first_battery_difference_kw: float
    max_soc_path_difference: float
    p95_soc_path_difference: float


@dataclass(frozen=True)
class ActionBehaviorSummary:
    observation_count: int
    case_count: int
    action_count: int
    cases: tuple[CaseBehaviorSummary, ...]


def _components(action_ids: tuple[str, ...], edges: tuple[tuple[str, str], ...]):
    adjacency = {action_id: set() for action_id in action_ids}
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    remaining = set(action_ids)
    groups = []
    while remaining:
        root = min(remaining)
        stack = [root]
        group = set()
        while stack:
            current = stack.pop()
            if current in group:
                continue
            group.add(current)
            stack.extend(sorted(adjacency[current] - group, reverse=True))
        remaining -= group
        groups.append(tuple(sorted(group)))
    return tuple(sorted(groups, key=lambda values: values[0]))
```

Calculate pairwise absolute differences from `ObjectiveAuditObservation` using `np.percentile(pairwise_differences, 95)` and maximum absolute difference across aligned predicted SOC paths. Revalidate the exact result type with `result.validate()` before deriving any output.

- [ ] **Step 4: Serialize the full evidence without changing selection semantics**

In `src/main/run_v2_objective_scale_audit.py`, add these fields to `run()`:

```python
behavior = summarize_action_behavior(result)
return {
    # existing fields remain byte-for-byte equivalent in meaning
    "behavior_tolerance": result.behavior_tolerance,
    "action_observations": result.observations,
    "behavioral_redundancy": result.behavioral_redundancy,
    "action_behavior_summary": behavior,
    "action_catalog_mutated": False,
    "validation_payloads_opened": 0,
    "test_payloads_opened": 0,
    # existing result_digest remains result.digest
}
```

Do not call `finalize_action_catalog` and do not change `FINAL_DQN_ACTION_CATALOG`.

- [ ] **Step 5: Run focused tests and regenerate the audit artifact**

Run the exact existing raw root and current metadata root used by the accepted audit:

```powershell
$env:PYTHONPATH=(Resolve-Path "src").Path
python -m pytest tests/test_v2_objective_scale_audit.py tests/test_v2_train_objective_scale_runner.py -q
python -X utf8 -u src/main/run_v2_objective_scale_audit.py `
  --raw-root "C:/Users/20883/OneDrive/Desktop/氢舟一号" `
  --metadata-root data/processed/operating_dataset_zero_boundary_v2/metadata `
  --output-path outputs/v2_action_behavior_audit/audit_summary.json
```

Expected: tests PASS; artifact contains 216 observations and does not alter the accepted objective-audit file. If the actual raw root differs, resolve it from the existing accepted run metadata; do not guess or scan unrelated directories.

- [ ] **Step 6: Commit the action-audit increment**

```powershell
git add src/v2/analysis/train_objective_scale_runner.py src/main/run_v2_objective_scale_audit.py tests/test_v2_objective_scale_audit.py tests/test_v2_train_objective_scale_runner.py outputs/v2_action_behavior_audit/audit_summary.json
git commit -m "feat(v2): expose action behavior audit"
```

## Task 2: Define the S8 Frame and 90-Dimensional History Contract

**Files:**
- Create: `src/v2/dqn/history.py`
- Create: `tests/test_v2_history_state.py`
- Modify: `src/v2/dqn/state.py`
- Modify: `tests/test_v2_formal_state.py`

- [ ] **Step 1: Write failing history layout and padding tests**

```python
def test_history_encodes_oldest_to_newest_with_tail_mask(self):
    from v2.dqn.history import FormalStateHistory

    history = FormalStateHistory()
    first = tuple(float(index) for index in range(8))
    current = tuple(float(index + 10) for index in range(8))
    history.commit(first)
    encoded = history.encode(current)

    assert len(encoded) == 90
    assert encoded[:64] == (0.0,) * 64
    assert encoded[64:72] == first
    assert encoded[72:80] == current
    assert encoded[80:] == (0.0,) * 8 + (1.0, 1.0)


def test_real_zero_frame_is_distinguished_by_mask(self):
    history = FormalStateHistory()
    encoded = history.encode((0.0,) * 8)
    assert encoded[:80] == (0.0,) * 80
    assert encoded[80:] == (0.0,) * 9 + (1.0,)


def test_encode_is_side_effect_free_and_reset_discards_history(self):
    history = FormalStateHistory()
    frame = (0.0,) * 8
    assert history.encode(frame) == history.encode(frame)
    assert history.committed_count == 0
    history.commit(frame)
    history.reset()
    assert history.committed_count == 0
```

Also test rejection of wrong-length, bool, NaN, infinity, list inputs, and more than ten retained frames.

- [ ] **Step 2: Run focused tests and verify red**

```powershell
$env:PYTHONPATH=(Resolve-Path "src").Path
python -m pytest tests/test_v2_history_state.py tests/test_v2_formal_state.py -q
```

Expected: FAIL because history constants and `FormalStateHistory` do not exist.

- [ ] **Step 3: Split frame identity from formal state identity**

In `state.py`, define:

```python
FORMAL_FRAME_SCHEMA_VERSION = "v2_s8_onboard_ais_frame_v1"
FORMAL_FRAME_FEATURE_NAMES = (
    "soc", "causal_base_load_fraction", "load_residual_fraction",
    "recent_load_population_std_fraction",
    "recent_load_window_trend_fraction", "fuel_cell_power_fraction",
    "fuel_cell_delta_fraction", "speed_fraction",
)
FORMAL_FRAME_DIMENSION = 8
FORMAL_STATE_HISTORY_LENGTH = 10
FORMAL_STATE_MASK_DIMENSION = 10
FORMAL_STATE_DIMENSION = 90
FORMAL_STATE_SCHEMA_VERSION = "v2_s8_stack10_mask_v1"
```

Rename `build_formal_operating_state` to `build_formal_operating_frame` and make its validation/error text explicitly require eight finite floats. Compute `FORMAL_STATE_SCHEMA_DIGEST` from the frame feature order, physical scales, history length, chronological order, left-zero-padding, tail-mask order, reset policy, and total dimension. Do not retain an ambiguous builder whose name says “state” but returns only eight values.

- [ ] **Step 4: Implement the bounded history component**

Create `history.py`:

```python
class FormalStateHistory:
    def __init__(self) -> None:
        self._frames: deque[tuple[float, ...]] = deque(
            maxlen=FORMAL_STATE_HISTORY_LENGTH - 1
        )

    @property
    def committed_count(self) -> int:
        return len(self._frames)

    def reset(self) -> None:
        self._frames.clear()

    def commit(self, frame: tuple[float, ...]) -> None:
        self._frames.append(_validate_frame(frame))

    def encode(self, current_frame: tuple[float, ...]) -> tuple[float, ...]:
        current = _validate_frame(current_frame)
        frames = tuple(self._frames) + (current,)
        frames = frames[-FORMAL_STATE_HISTORY_LENGTH:]
        missing = FORMAL_STATE_HISTORY_LENGTH - len(frames)
        values = (0.0,) * (missing * FORMAL_FRAME_DIMENSION)
        values += tuple(value for frame in frames for value in frame)
        mask = (0.0,) * missing + (1.0,) * len(frames)
        encoded = values + mask
        if len(encoded) != FORMAL_STATE_DIMENSION:
            raise RuntimeError("formal history encoder produced the wrong dimension")
        return encoded
```

Use exact tuple/float and finite checks consistent with the existing v2 contracts.

- [ ] **Step 5: Run focused tests and commit**

```powershell
python -m pytest tests/test_v2_history_state.py tests/test_v2_formal_state.py -q
git add src/v2/dqn/history.py src/v2/dqn/state.py tests/test_v2_history_state.py tests/test_v2_formal_state.py
git commit -m "feat(v2): add causal S8 history state"
```

Expected: PASS.

## Task 3: Integrate History at Every ONBOARD Supervisory Step

**Files:**
- Modify: `src/v2/envs/formal_episode.py`
- Modify: `src/v2/envs/multirate_weight_env.py`
- Modify: `tests/test_v2_formal_episode.py`
- Modify: `tests/test_v2_multirate_env.py`

- [ ] **Step 1: Write failing integration tests**

Add tests that run seven ONBOARD intervals with `M=5` and prove the second DQN boundary includes five committed internal frames plus the current frame, while only one replay transition was emitted. Add a repeated-`state()` assertion and a shore-reentry reset assertion:

```python
initial = environment.reset()
assert len(initial) == 90
assert initial[80:] == (0.0,) * 9 + (1.0,)
assert backend.state() == backend.state()

first = environment.step("w_1_1_8")
assert first.executed_mpc_steps == 5
assert first.next_state[80:] == (0.0,) * 4 + (1.0,) * 6
assert len(environment.transitions) == 1
```

For ONBOARD → shore → ONBOARD, require the reentry mask to equal nine zeros plus one and ensure no pre-shore frame appears in the encoded feature block. For a solver physical failure, require `done=True`, a finite 90-vector next state, and no fabricated commit for the failed interval.

- [ ] **Step 2: Run focused tests and verify red**

```powershell
python -m pytest tests/test_v2_formal_episode.py tests/test_v2_multirate_env.py -q
```

Expected: FAIL on old eight-value state and missing reset/commit behavior.

- [ ] **Step 3: Make frame construction pure and history ownership explicit**

In `FormalEpisodeBackend`:

```python
def _reset_onboard_history(self) -> None:
    self._base_filter = CausalBaseLoadFilter(
        sample_seconds=self.timescale.ts_mpc_seconds,
        tau_seconds=TAU_LPF_SECONDS,
    )
    self._past_samples = []
    self._state_history = FormalStateHistory()

def _current_frame(self) -> tuple[float, ...]:
    sample, speed, mode = self._current_sample()
    if mode is not OperatingMode.ONBOARD:
        raise ValueError("DQN state is available only at an ONBOARD decision boundary")
    return build_formal_operating_frame(
        tuple(self._past_samples + [sample]),
        current_time_seconds=sample.timestamp_seconds,
        speed_kn=speed,
    )

def state(self) -> tuple[float, ...]:
    if self._done:
        return self._terminal_state
    return self._state_history.encode(self._current_frame())
```

During a successful ONBOARD step, compute one `current_frame`, solve, commit the physical filter/sample, then call `self._state_history.commit(current_frame)` exactly once. Do not commit if `mpc.solve` raises before a command is executed. Existing `_reset_onboard_history()` calls in shore/idle processing clear both LPF history and DQN history.

- [ ] **Step 4: Build a 90-value terminal state and enforce it in the macro environment**

Encode the terminal S8 frame through the same `FormalStateHistory`; do not return an eight-value exception. Update formal-mode dimension guards and error messages in `multirate_weight_env.py` to refer to the current formal schema rather than S8.

- [ ] **Step 5: Run focused tests and commit**

```powershell
python -m pytest tests/test_v2_formal_episode.py tests/test_v2_multirate_env.py tests/test_v2_economic_interval_semantics.py -q
git add src/v2/envs/formal_episode.py src/v2/envs/multirate_weight_env.py tests/test_v2_formal_episode.py tests/test_v2_multirate_env.py
git commit -m "feat(v2): integrate supervisory history"
```

Expected: PASS, including unchanged economic interval tests.

## Task 4: Bind the New State to DQN, Replay, and Checkpoints

**Files:**
- Modify: `src/v2/training/dqn.py`
- Modify: `src/v2/training/checkpoint.py`
- Modify: `tests/test_v2_dqn_training.py`
- Modify: `tests/test_v2_checkpoint_resume.py`
- Modify: `tests/test_v2_artifacts.py`

- [ ] **Step 1: Write failing dimension and compatibility tests**

Replace hard-coded eight-value replay fixtures with `np.arange(FORMAL_STATE_DIMENSION)`. Assert the network maps `[batch, 90]` to `[batch, 36]`. Add a forged old-S8 checkpoint with the prior schema version/digest and require rejection before agent, replay, or schedule mutation.

```python
assert DqnTrainingConfig.formal_baseline().state_dim == 90
network = QNetwork(DqnTrainingConfig.formal_baseline())
assert network(torch.zeros(4, 90)).shape == (4, 36)
```

- [ ] **Step 2: Run focused tests and verify red**

```powershell
python -m pytest tests/test_v2_dqn_training.py tests/test_v2_checkpoint_resume.py tests/test_v2_artifacts.py -q
```

Expected: FAIL because the old schema/dimension remains embedded in tests and checkpoint version.

- [ ] **Step 3: Update the formal DQN identity without widening the network**

Keep `hidden_dims=(128, 128)` and all baseline algorithm settings unchanged. Update validation text to “formal history/36 contracts.” Bump:

```python
CHECKPOINT_VERSION = "v2_history_dqn_checkpoint_v1"
```

The checkpoint continues to bind `FORMAL_STATE_SCHEMA_VERSION`,
`FORMAL_STATE_SCHEMA_DIGEST`, config identity, replay arrays, all RNG state,
permutation, and global step. Loading the historical S8 v3 checkpoint must
raise `IncompatibleCheckpointError` before calling `agent.load_state_dict` or
`schedule.load_state_dict`.

- [ ] **Step 4: Run focused tests and commit**

```powershell
python -m pytest tests/test_v2_dqn_training.py tests/test_v2_checkpoint_resume.py tests/test_v2_artifacts.py -q
git add src/v2/training/dqn.py src/v2/training/checkpoint.py tests/test_v2_dqn_training.py tests/test_v2_checkpoint_resume.py tests/test_v2_artifacts.py
git commit -m "feat(v2): bind history checkpoint schema"
```

Expected: PASS.

## Task 5: Generate and Authenticate the Train-Only Reward Scale

**Files:**
- Create: `src/v2/training/reward_scaling.py`
- Create: `src/v2/main/run_reward_scale_calibration.py`
- Create: `tests/test_v2_reward_scale_calibration.py`
- Modify: `src/v2/envs/formal_episode.py`
- Modify: `src/v2/economics.py`
- Modify: `tests/test_v2_formal_episode.py`

- [ ] **Step 1: Write failing calibration and uniform-scaling tests**

```python
def test_scaled_training_reward_divides_raw_cost_and_failure_score_together():
    provenance = DatasetProvenance(
        "operating_dataset_zero_boundary_v2",
        "sha256:reward-scale-fixture",
        DataSplit.TRAIN,
    )
    calibration = calibrate_reward_scale(
        (10.0, 20.0, 30.0),
        provenance=provenance,
        audit_id="fixture",
        reason="verify uniform scaling",
    )
    assert scale_learning_reward(
        -50_030.0,
        calibration=calibration,
    ) == pytest.approx(-2501.5)


def test_document_loader_rejects_validation_provenance_and_digest_mutation():
    provenance = DatasetProvenance(
        "operating_dataset_zero_boundary_v2",
        "sha256:reward-scale-fixture",
        DataSplit.TRAIN,
    )
    calibration = calibrate_reward_scale(
        (10.0, 20.0, 30.0),
        provenance=provenance,
        audit_id="fixture",
        reason="verify document authentication",
    )
    document = build_reward_scale_document(
        calibration=calibration,
        reference_action_id="w_8_1_1",
        manifest_hashes={"power": "a" * 64, "ais": "b" * 64, "modes": "c" * 64},
        train_segment_ids=("train-a",),
        test_payloads_opened=0,
    )
    document["split"] = "validation"
    with pytest.raises(HeldOutSelectionError):
        load_reward_scale_document(document)


def test_backend_retains_one_ledger_per_executed_physical_interval():
    backend = FormalEpisodeBackend(
        load_kw=np.asarray([100.0, 100.0, 100.0]),
        speed_kn=np.asarray([3.0, 3.0, 3.0]),
        fc_power_kw=np.asarray([80.0, 80.0, 80.0]),
        battery_bus_kw=np.asarray([20.0, 20.0, 20.0]),
        operating_mode=("onboard", "onboard", "onboard"),
        mpc=self._solver(),
    )
    weights = FINAL_DQN_ACTION_CATALOG[0].to_mpc_weights()
    for _ in range(3):
        backend.execute_mpc_step(weights)
    assert len(backend.interval_ledgers) == 3
```

Also prove the generator calls only `load_train()`, leaves `opened_test_payloads == 0`, includes zero-cost ledgers in the immutable tuple, excludes the 50,000 score from `C_ref`, and writes atomically.

- [ ] **Step 2: Run focused tests and verify red**

```powershell
python -m pytest tests/test_v2_reward_scale_calibration.py tests/test_v2_state_and_economics.py -q
```

Expected: FAIL because full learning-reward scaling, interval-ledger capture, and document authentication are absent.

- [ ] **Step 3: Add full learning-reward scaling**

In `economics.py` add:

```python
def scale_learning_reward(
    learning_reward: object,
    *,
    calibration: RewardScaleCalibration,
) -> float:
    reward = _finite_scalar(learning_reward, "learning_reward")
    checked = _validate_reward_scale(calibration)
    result = reward / checked.scale_cny
    if not math.isfinite(result):
        raise ValueError("scaled learning reward must remain finite")
    return result
```

This function does not modify `RawCnyIntervalLedger` or `MacroTransition`.

- [ ] **Step 4: Capture immutable interval-ledger snapshots**

Initialize `self._interval_ledgers = []` on backend reset, append a detached
`RawCnyIntervalLedger(*ledger.components_cny)` after every successfully
constructed 30-second ledger, and expose only:

```python
@property
def interval_ledgers(self) -> tuple[RawCnyIntervalLedger, ...]:
    return tuple(
        RawCnyIntervalLedger(*ledger.components_cny)
        for ledger in self._interval_ledgers
    )
```

- [ ] **Step 5: Implement the calibration document and loader**

`run_reward_scale_calibration.py` must run fixed `w_8_1_1` over all Train
episodes, concatenate each backend's interval total cost, call
`calibrate_reward_scale`, and write a canonical document containing:

```python
{
    "schema_version": "v2_train_interval_reward_scale_v1",
    "dataset_version": DATASET_VERSION,
    "split": "train",
    "reference_action_id": "w_8_1_1",
    "input_manifest_sha256": _manifest_hashes(power_root, ais_root, mode_root),
    "train_segment_ids": [episode.sample_id for episode in train],
    "train_raw_interval_costs_cny": list(train_raw_interval_costs_cny),
    "sample_count": calibration.sample_count,
    "scale_cny": calibration.scale_cny,
    "derivation_rule": calibration.derivation_rule,
    "calibration_digest": calibration.digest,
    "test_payloads_opened": 0,
    "result_digest": canonical_result_digest(document_without_result_digest),
}
```

`build_reward_scale_document` produces the canonical mapping used by both the
runner and fixtures. `load_reward_scale_document` rehashes the document, reconstructs exact Train
provenance, calls the sealed factory with the stored tuple, and compares every
derived field and digest. It must reject held-out provenance, missing costs,
non-finite values, changed manifest hashes, reference action changes, and
digest changes.

- [ ] **Step 6: Run tests, generate the artifact, and commit**

```powershell
python -m pytest tests/test_v2_reward_scale_calibration.py tests/test_v2_state_and_economics.py tests/test_v2_formal_episode.py -q
python -X utf8 -u -m v2.main.run_reward_scale_calibration `
  --output outputs/v2_history_dqn_study/reward_scale_calibration.json
git add src/v2/economics.py src/v2/envs/formal_episode.py src/v2/training/reward_scaling.py src/v2/main/run_reward_scale_calibration.py tests/test_v2_reward_scale_calibration.py outputs/v2_history_dqn_study/reward_scale_calibration.json
git commit -m "feat(v2): calibrate Train reward scale"
```

Expected: focused tests PASS; generator reports zero Test payloads.

## Task 6: Freeze the H1-H4 Experiment Matrix

**Files:**
- Create: `src/v2/training/experiments.py`
- Create: `tests/test_v2_dqn_experiments.py`
- Modify: `src/v2/training/dqn.py`
- Modify: `src/v2/training/checkpoint.py`

- [ ] **Step 1: Write failing profile-identity tests**

```python
def test_experiment_matrix_changes_only_reward_mode_and_learning_rate():
    calibration = calibrate_reward_scale(
        (10.0, 20.0, 30.0),
        provenance=DatasetProvenance(
            "operating_dataset_zero_boundary_v2",
            "sha256:experiment-fixture",
            DataSplit.TRAIN,
        ),
        audit_id="experiment-fixture",
        reason="bind scaled experiment identities",
    )
    profiles = history_study_profiles(calibration)
    assert tuple(profiles) == ("H1", "H2", "H3", "H4")
    assert [profiles[key].learning_rate for key in profiles] == [1e-4, 1e-4, 3e-4, 1e-3]
    assert [profiles[key].reward_mode for key in profiles] == ["raw", "scaled", "scaled", "scaled"]
    identities = [profiles[key].dqn_config(rounds=10) for key in profiles]
    for config in identities:
        assert config.hidden_dims == (128, 128)
        assert config.gamma == 1.0
        assert config.batch_size == 256
        assert config.replay_capacity == 200_000
        assert config.warmup_steps == 5_000
        assert config.target_sync_steps == 1_000
```

Add tests that raw H1 rejects a calibration argument, scaled H2-H4 require one,
and checkpoint resume rejects any profile ID, learning rate, reward mode, or
calibration digest mismatch.

- [ ] **Step 2: Run focused tests and verify red**

```powershell
python -m pytest tests/test_v2_dqn_experiments.py tests/test_v2_checkpoint_resume.py -q
```

Expected: FAIL because profiles and reward identity are not defined.

- [ ] **Step 3: Implement immutable profiles**

```python
@dataclass(frozen=True)
class DqnExperimentProfile:
    experiment_id: str
    reward_mode: str
    learning_rate: float
    reward_scaling_identity: str

    def dqn_config(self, *, rounds: int) -> DqnTrainingConfig:
        return replace(
            DqnTrainingConfig.formal_baseline(),
            learning_rate=self.learning_rate,
            rounds=rounds,
            experiment_id=self.experiment_id,
            reward_scaling_identity=self.reward_scaling_identity,
        )

    def replay_reward(self, transition, calibration) -> float:
        if self.reward_mode == "raw":
            return transition.learning_reward
        return scale_learning_reward(
            transition.learning_reward, calibration=calibration
        )
```

Extend `DqnTrainingConfig` with exact non-empty `experiment_id` and
`reward_scaling_identity` fields. `formal_baseline()` becomes the H1-compatible
history baseline and retains all unchanged algorithm constants.

- [ ] **Step 4: Bind profile identity into checkpoints**

The existing `_config_identity(agent)` already serializes all config fields
except `rounds`; retain that rule so extending 10 → 40 rounds is allowed but
profile/reward identity changes are rejected.

- [ ] **Step 5: Run focused tests and commit**

```powershell
python -m pytest tests/test_v2_dqn_experiments.py tests/test_v2_checkpoint_resume.py tests/test_v2_dqn_training.py -q
git add src/v2/training/experiments.py src/v2/training/dqn.py src/v2/training/checkpoint.py tests/test_v2_dqn_experiments.py tests/test_v2_checkpoint_resume.py tests/test_v2_dqn_training.py
git commit -m "feat(v2): freeze history DQN profiles"
```

Expected: PASS.

## Task 7: Add Q, TD, Gradient, and Action Diagnostics

**Files:**
- Create: `src/v2/training/diagnostics.py`
- Create: `tests/test_v2_dqn_diagnostics.py`
- Modify: `src/v2/training/dqn.py`

- [ ] **Step 1: Write failing deterministic diagnostic tests**

```python
def test_q_diagnostics_separate_common_mode_advantage_and_margin():
    q = np.asarray([[100.0, 101.0, 99.0], [5.0, 5.0, 5.0]])
    result = summarize_q_values(q)
    assert result.common_mode_mean == 52.5
    assert result.centered_advantage_std > 0.0
    assert result.top_two_margin_p50 == 0.5


def test_action_summary_does_not_treat_entropy_as_selection_score():
    result = summarize_actions((0, 0, 1, 2), action_dim=36)
    assert result.unique_action_count == 3
    assert result.max_action_share == 0.5
    assert result.shannon_entropy > 0.0
    assert not hasattr(result, "selection_score")
```

Test TD P50/P95/max, finite checks, deterministic action-ID ordering, empty
input rejection, and pre-clip gradient norm.

- [ ] **Step 2: Run focused tests and verify red**

```powershell
python -m pytest tests/test_v2_dqn_diagnostics.py tests/test_v2_dqn_training.py -q
```

Expected: FAIL because structured diagnostics do not exist.

- [ ] **Step 3: Implement pure summaries**

Create frozen `QValueDiagnostics`, `ActionDistributionDiagnostics`, and
`DqnOptimizationDiagnostics`. Use `np.percentile` and natural-log Shannon
entropy. `summarize_q_values` must compute per-row means, center each row before
the advantage standard deviation, and obtain top-two margins from sorted rows.

- [ ] **Step 4: Return optimizer diagnostics from `DqnAgent.optimize()`**

Before clipping, compute:

```python
td_error = expected - predicted
gradient_norm = nn.utils.clip_grad_norm_(
    self.online.parameters(), self.config.gradient_clip_norm
)
diagnostics = DqnOptimizationDiagnostics.from_tensors(
    loss=loss,
    td_error=td_error,
    q_values=self.online(states).detach(),
    gradient_norm=gradient_norm,
)
```

Return the dataclass instead of only the scalar loss. Preserve Double-DQN,
Huber loss, hard target sync, and optimizer update ordering.

- [ ] **Step 5: Run focused tests and commit**

```powershell
python -m pytest tests/test_v2_dqn_diagnostics.py tests/test_v2_dqn_training.py -q
git add src/v2/training/diagnostics.py src/v2/training/dqn.py tests/test_v2_dqn_diagnostics.py tests/test_v2_dqn_training.py
git commit -m "feat(v2): log DQN learning diagnostics"
```

Expected: PASS.

## Task 8: Add the Explicit History-Study Training Entry Point

**Files:**
- Create: `src/v2/main/train_history_dqn_study.py`
- Create: `tests/test_v2_history_study_cli.py`
- Modify: `.gitignore`
- Modify: `src/v2/main/train_formal_dqn.py`
- Modify: `tests/test_v2_formal_training_cli.py`

- [ ] **Step 1: Write failing CLI and reward-path tests**

Require `--experiment H1|H2|H3|H4`, exactly ten pilot rounds by default, a
nonexistent experiment-specific output directory, and a calibration file for
H2-H4 only. Mock one raw and one failed transition to prove replay receives the
profile-transformed reward while raw logs retain original values.

```python
assert parsed.rounds == 10
assert parsed.seed == 42
assert parsed.experiment == "H1"

agent.replay.append.assert_called_once()
assert agent.replay.append.call_args.args[2] == expected_replay_reward
assert "raw_economic_cost_cny=10.000000000" in stdout
assert "failure_penalty_score=50000.000000000" in stdout
assert "replay_reward=" in stdout
```

Assert that each round reshuffles Train with the fixed RNG, Validation remains
ordered, and neither path calls `load_test()`.

- [ ] **Step 2: Run focused tests and verify red**

```powershell
python -m pytest tests/test_v2_history_study_cli.py tests/test_v2_formal_training_cli.py -q
```

Expected: FAIL because the study CLI and profile-aware replay path do not exist.

- [ ] **Step 3: Make the existing loop profile-aware without duplicating it**

Change `_train` to receive an exact `DqnExperimentProfile` and optional exact
`RewardScaleCalibration`. Build its config with
`profile.dqn_config(rounds=args.rounds)` and append
`profile.replay_reward(transition, calibration)` to replay. Continue accumulating
and printing the raw `transition.learning_reward` separately.

Track behavior-selected and greedy Validation actions in separate counters.
Aggregate structured optimizer diagnostics per log window and print stable
field names:

The stable log field names are `replay_reward`, `td_abs_p50`, `td_abs_p95`,
`td_abs_max`, `q_common_mean`, `q_advantage_std`, `q_margin_p50`,
`gradient_norm_preclip`, `greedy_unique_actions`, `greedy_max_share`, and
`greedy_entropy`. Each field prints its finite runtime value; unavailable
optimizer diagnostics before warmup print the exact token `NA`.

- [ ] **Step 4: Implement the guarded study CLI**

`train_history_dqn_study.py` parses the explicit profile, loads/authenticates
the scale document when required, sets the default output to
`outputs/v2_history_dqn_study/<experiment>`, and delegates to the public
profile-aware functions in `train_formal_dqn.py`. It must reject an existing
nonempty output unless `--resume` points to that directory's exact `latest.pt`.
Preflight/smoke remain bounded and do not write checkpoints.

Add only the machine-local run directories to `.gitignore`:

```gitignore
outputs/v2_history_dqn_study/H1/
outputs/v2_history_dqn_study/H2/
outputs/v2_history_dqn_study/H3/
outputs/v2_history_dqn_study/H4/
outputs/v2_history_dqn_study/pilot_selection/
outputs/v2_history_dqn_study/final_selection/
```

Keep `outputs/v2_history_dqn_study/reward_scale_calibration.json` trackable.

- [ ] **Step 5: Update state preflight to validate the stack**

Replace the old direct S8 length check with a `FormalStateHistory` per episode.
Commit each ONBOARD frame, clear on shore/idle, encode every causal current
frame, and require 90 finite values. This is a data/schema check only; it does
not run training or open Test.

- [ ] **Step 6: Run focused tests and commit**

```powershell
python -m pytest tests/test_v2_history_study_cli.py tests/test_v2_formal_training_cli.py tests/test_v2_formal_episode.py -q
git add .gitignore src/v2/main/train_history_dqn_study.py src/v2/main/train_formal_dqn.py tests/test_v2_history_study_cli.py tests/test_v2_formal_training_cli.py
git commit -m "feat(v2): add history DQN study runner"
```

Expected: PASS.

## Task 9: Rank Pilot Profiles with Validation Only

**Files:**
- Create: `src/v2/evaluation/study_selection.py`
- Create: `src/v2/main/select_history_dqn_study.py`
- Create: `tests/test_v2_history_study_selection.py`
- Modify: `src/v2/evaluation/checkpoint_selection.py`
- Modify: `tests/test_v2_checkpoint_selection.py`

- [ ] **Step 1: Write failing cross-profile ranking tests**

Create four fixture runs with rounds 1-10. Require the selector to authenticate
each profile/checkpoint identity, select the best round within each profile,
then rank profiles by completed episodes, failure score, raw cost, profile ID,
and round. Require `top_k=2` to return exactly two resumable `latest.pt` paths.

```python
selected = select_study_candidates(candidates, required_profiles=("H1", "H2", "H3", "H4"), top_k=2)
assert tuple(item.experiment_id for item in selected) == ("H3", "H2")
assert all(item.completed_rounds == 10 for item in selected)
```

Test rejection of missing rounds, mixed state/action/dataset/scale identities,
duplicate profiles, non-Train checkpoints, output overwrite, and any Test loader
call.

- [ ] **Step 2: Run focused tests and verify red**

```powershell
python -m pytest tests/test_v2_history_study_selection.py tests/test_v2_checkpoint_selection.py -q
```

Expected: FAIL because study selection does not exist.

- [ ] **Step 3: Generalize checkpoint authentication only where required**

Keep the existing final selector's 40-round default. Add an explicit
`required_rounds` argument through its internal authentication/evaluation APIs;
do not loosen CLI defaults for historical final selection. Checkpoint metadata
must include exact experiment ID and reward-scale identity.

- [ ] **Step 4: Implement study selection and manifest sealing**

Define:

```python
STUDY_RANKING_RULE = (
    "-completed_episodes",
    "failure_penalty_score",
    "raw_economic_cost_cny",
    "experiment_id",
    "round_index",
)
```

For each profile, greedily evaluate every required checkpoint on the same
ordered Validation episodes, select its best checkpoint, rank profiles, and
write atomically:

- `profile_validation_metrics.csv`;
- `study_selection_manifest.json` with input/checkpoint hashes and result digest;
- `top_profiles.json` with exact resume paths and target round 40.

Do not copy a “final model” at pilot time and do not load Test.

- [ ] **Step 5: Run focused tests and commit**

```powershell
python -m pytest tests/test_v2_history_study_selection.py tests/test_v2_checkpoint_selection.py -q
git add src/v2/evaluation/study_selection.py src/v2/main/select_history_dqn_study.py src/v2/evaluation/checkpoint_selection.py tests/test_v2_history_study_selection.py tests/test_v2_checkpoint_selection.py
git commit -m "feat(v2): rank history DQN pilots"
```

Expected: PASS.

## Task 10: Document Commands and Verify the Complete Change

**Files:**
- Modify: `docs/v2_objective_scale_audit.md`
- Modify: `docs/v2_formal_training_runbook.md`

- [ ] **Step 1: Update the action-audit report**

Document the actual generated behavior-group counts and pairwise statistics.
State explicitly that the evidence is local to six deterministic Train cases,
the catalog remains complete36, and no Validation/Test record selected a group
or threshold.

- [ ] **Step 2: Add exact PowerShell study commands**

The runbook must contain these command families with the generated calibration
path and separate output directories:

```powershell
$env:PYTHONPATH=(Resolve-Path "src").Path
$env:OPENBLAS_NUM_THREADS="1"
$env:OMP_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"

python -X utf8 -u -m v2.main.train_history_dqn_study --preflight-only --experiment H1
python -X utf8 -u -m v2.main.train_history_dqn_study --smoke-only --experiment H1

python -X utf8 -u -m v2.main.train_history_dqn_study --experiment H1 --rounds 10 --seed 42 --device cpu --log-every 1 --output-dir outputs/v2_history_dqn_study/H1
python -X utf8 -u -m v2.main.train_history_dqn_study --experiment H2 --reward-scale outputs/v2_history_dqn_study/reward_scale_calibration.json --rounds 10 --seed 42 --device cpu --log-every 1 --output-dir outputs/v2_history_dqn_study/H2
python -X utf8 -u -m v2.main.train_history_dqn_study --experiment H3 --reward-scale outputs/v2_history_dqn_study/reward_scale_calibration.json --rounds 10 --seed 42 --device cpu --log-every 1 --output-dir outputs/v2_history_dqn_study/H3
python -X utf8 -u -m v2.main.train_history_dqn_study --experiment H4 --reward-scale outputs/v2_history_dqn_study/reward_scale_calibration.json --rounds 10 --seed 42 --device cpu --log-every 1 --output-dir outputs/v2_history_dqn_study/H4

python -X utf8 -u -m v2.main.select_history_dqn_study --study-root outputs/v2_history_dqn_study --rounds 10 --top-k 2 --output-dir outputs/v2_history_dqn_study/pilot_selection --device cpu
```

Add continuation commands only for the two profile IDs written by
`top_profiles.json`, using `--rounds 40 --resume <profile>/latest.pt`. Do not
include or invoke a Test command in this study runbook section.

- [ ] **Step 3: Run focused tests**

```powershell
python -m pytest `
  tests/test_v2_objective_scale_audit.py `
  tests/test_v2_train_objective_scale_runner.py `
  tests/test_v2_history_state.py `
  tests/test_v2_formal_state.py `
  tests/test_v2_formal_episode.py `
  tests/test_v2_multirate_env.py `
  tests/test_v2_reward_scale_calibration.py `
  tests/test_v2_dqn_experiments.py `
  tests/test_v2_dqn_diagnostics.py `
  tests/test_v2_history_study_cli.py `
  tests/test_v2_history_study_selection.py `
  tests/test_v2_checkpoint_resume.py `
  tests/test_v2_formal_training_cli.py `
  tests/test_v2_checkpoint_selection.py -q
```

Expected: all focused tests PASS.

- [ ] **Step 4: Run all v2 tests**

```powershell
python -m pytest tests/test_v2_*.py -q
```

Expected: PASS with no skipped failures and no Test payload access.

- [ ] **Step 5: Run solver and bounded training smoke**

```powershell
python -X utf8 -u -m v2.main.train_history_dqn_study --preflight-only --experiment H1
python -X utf8 -u -m v2.main.train_history_dqn_study --smoke-only --experiment H1
python -X utf8 -u -m v2.main.train_history_dqn_study --smoke-only --experiment H2 --reward-scale outputs/v2_history_dqn_study/reward_scale_calibration.json
```

Expected: preflight and both smoke modes PASS, print the 90-dimensional state
schema, profile/reward identity, diagnostics, and `test_payloads_opened=0`.

- [ ] **Step 6: Run compile/import and whitespace checks**

```powershell
python -m compileall -q src/v2 tests
python -c "import v2; import v2.dqn.history; import v2.training.experiments; import v2.training.reward_scaling; import v2.main.train_history_dqn_study"
git diff --check
git status --short
```

Expected: all commands exit zero; status contains only intended source, test,
documentation, and authenticated audit/calibration artifact changes.

- [ ] **Step 7: Commit documentation and verification updates**

```powershell
git add docs/v2_objective_scale_audit.md docs/v2_formal_training_runbook.md
git commit -m "docs(v2): add history DQN study runbook"
```

- [ ] **Step 8: Stop before long training**

Report action-audit findings, new state/checkpoint identities, reward scale,
focused/all-v2/smoke/compile/diff-check results, commit SHAs, and the four pilot
commands. Do not start any 10-round pilot, 40-round continuation, or Test run.
