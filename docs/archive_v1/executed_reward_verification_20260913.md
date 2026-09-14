# Executed reward verification — 2026-09-13

## Scope and result

- Fixed positive-integer simplex: 84 lexicographically ordered actions; MLP 7-128-64-84.
- Reward uses executed first FC/battery power, previous FC and actual next SOC only. The original persistence forecast and t -> t+1 execution convention are unchanged.
- Model, complete training state and replay carry action/reward semantics; inference models and replay additionally preserve/validate the calibrated failure policy.
- Both legacy reward functions remain reproducible in `src/dqn/utils/legacy_reward.py`; frozen four-action diagnostics retain their own table.
- `terminal_failure_penalty=None`, calibration=None. No value was calibrated or guessed. Formal train/evaluation gates run before data reads. Numerical and forecast-QP failures raise without adding fake transitions; calibrated physical violations terminate.

## Verification evidence

- Focused contract suite: `python -X utf8 -m unittest discover -s tests -p test_executed_reward_contract.py -v` — 12 tests passed.
- Full unittest discovery: 200 tests passed, with a Python audit hook blocking held-out production payload opens. Final run: 0 attempted/blocked held-out opens.
- The initial guarded run exposed a preexisting real Validation scan in `test_effective_training_and_validation_exclude_only_physical_stress_cases`; the audit hook blocked it before the file opened. The test now checks split aggregation using synthetic payloads. No Validation/Test payload was consumed.
- Unit tests exercise only synthetic DQN interactions/optimizer updates; no formal DQN training, reward scans, new SOC-scale audit or held-out evaluation was run. Real data accesses in final tests were the FC curve, split metadata and one existing Train loader test.
- `python -X utf8 -m compileall -q src tests` passed.
- Independent read-only review found missing inference penalty persistence and replay penalty compatibility; both were fixed with regressions. Follow-up review found no further actionable issues.
- Raw/processed data and manifest files have no Git diff. Cleanup was hash-checked and restricted to seven listed obsolete diagnostic CSV files.

`git diff --check` and `git diff --cached --check` passed. No data/checkpoint/cache file above 1 MB is staged; changed/new files total approximately 0.9 MB. Git commit/push and remote branch hash evidence are reported in the task's final response.

## Limits

Passing unit tests establishes code/serialization/timing consistency, not trained control superiority or certified battery ageing/thermal limits. The 84 points are a fixed grid, not a certified Pareto-safe set. Existing historical cached reports remain historical evidence; protected-source hashes inside them identify their original snapshot and are not rewritten to claim a new experiment.
