# v4 reward feedback and economic update implementation

Base: `4a425b4`. Execute inline under the user's detailed approved specification. Preserve the previous archive byte for byte. No formal training, Test access, KAN or physical-model changes.

## Design decisions

Read B's tariff from `v2.economics.FORMAL_PRICE_CATALOG`, capacity from the accountant's PlantConfig, charging efficiency from its calibrated efficiency, and reference SOC from `v3.control.SHORE_TARGET_SOC`. Each contiguous ONBOARD segment owns its initial SOC and terminal correction. Correction uses pre-SHORE terminal SOC. A failed unfinished segment never receives correction.

Keep legacy entry defaults (original reward, sample-based episode16, round target sync) reproducible. Add an explicit single-configuration feedback entry: redistributed reward, beta500, optimizer target sync, configurable replay32/replay16/episode16 and n=1/8. In the new entry episode16 grants 16 updates per completed ONBOARD segment; the legacy sample counting remains explicitly available. Retain historical replay8 compatibility.

Store n-step economic entries only from completed contiguous voyages. One entry per original decision, rewards summed over at most n decisions, endpoint mask preserved, terminal tails shortened. Failed suffix remains outcome-only and is separately diagnosed. No failure penalty or safety network is introduced.

## Tasks

- [x] Write `tests/test_v4_reward_feedback.py`: fixed-action cases A–H, formal ledger equality, per-segment reward sum identity, signed feedback, terminal SOC before SHORE, failed suffix and parameter sourcing. Run failing tests before implementation.
- [x] Add `src/v4/reward_feedback.py`; extend `src/v4/control.py` with opt-in redistribution and explicit original/immediate/correction/new reward fields. Keep all physical and ledger calls identical.
- [x] Extend scheduling tests for replay32, carried credit, target intervals250/500/1000, bootstrap/round separation and capacity saturation. Add n-step tests for n1/n8, shortened tails, SHORE boundaries, terminal ledger/correction and masks.
- [x] Extend `src/v4/dqn.py` with trajectory-level n-step insertion and TD statistics; extend `experiment_schedule.py` without changing existing schedules.
- [x] Add fixed-state 61-action Q diagnostics and transition/profile telemetry. Extend `train.py` summaries, `monitored_training.py` configuration, greedy trajectory collection and RNG-isolation tests.
- [x] Add `src/v4/feedback_study.py` single-configuration CLI with required output directory and round count. Add shared `experiment_paths.py` guards to the new/historical entry and monitored runner; preserve legacy learning semantics.
- [x] Run targeted unit/contract/synthetic smoke tests (at most2 rounds per synthetic verification, no formal data loaded). Validate original archive SHA256 and that git shows no changes under the previous results directory.
- [x] Write a new implementation report with exact formulas, test results, update counts, failed-suffix limitations, and a minimal future experiment proposal. Git commit/push follow this verification checkpoint; remote HEAD and final clean status provide the completion evidence. Formal experiments require subsequent user confirmation.

Independent requesting-code-review found two reproduced defects: historical resume could overwrite archived summaries, and a new voyage infeasible at its first step incorrectly labeled its completed predecessor as a failed suffix. Both now have failing-first regression tests and fixes, independently rechecked. Final targeted suite: 153 tests and330 subtests passed. Fixed-trajectory maximum cumulative reward difference:1.1368683772161603e-13. No formal training or Test payload access.

## Verification commands

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python -m pytest -q tests/test_v4_reward_feedback.py tests/test_v4_n_step.py tests/test_v4_feedback_monitoring.py
python -m pytest -q tests/test_v4_control.py tests/test_v4_dqn.py tests/test_v4_training.py tests/test_v4_monitored_training.py tests/test_v4_experiment_schedule.py tests/test_v4_review_metrics.py tests/test_v3_shore_settlement.py tests/test_v2_energy_models.py tests/test_v2_degradation_models.py tests/test_v2_state_and_economics.py tests/test_v2_formal_preflight.py
git diff --check
```

The first test command must initially fail on missing feedback/n-step functionality. The final commands must pass before committing. Archive inventory bytes and recorded dataset manifest hashes must match. Synthetic metrics verify implementation only and cannot establish an economic or completion-rate improvement.
