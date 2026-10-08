# v4 explicit failure termination and economic TD

Base: remote/local `3c6d59b`. Execute the user's approved specification inline. No formal training, Test, physics/state/network/split changes. Preserve all prior archives.

## Decisions

Use a pure learning-view conversion of genuine `NoFeasibleFCActionError` prefixes. Real replay exceptions retain their original executed transitions. Only the unfinished ONBOARD suffix receives a terminal label, empty mask, exactly one penalty, and (if enabled) battery potential correction. Preserve earlier completed voyages and all real states/ledgers. If that suffix is empty, record an event only. Program/model errors are never converted.

Classify power-only infeasibility separately from SOC-induced infeasibility using the existing grid and battery power bounds. Both genuine no-feasible failures with executed suffixes have explicit labeled failure terminals as requested; structural failures are not interpreted as policy-remediable failures and are reported separately.

Freeze reference penalty at nearest-rank P95 of the30 complete Train comparable economic costs in the already archived beta500 best profiles (source SHA256 recorded). No Validation/Test rows or new formal dataset reads. `C_fail = configurable_dimensionless_scale * frozen_Train_P95`; initial scale1, sensitivity plan0/0.5/1/2. Units are reward-equivalent CNY, not real expenditure. Reference10132.660087898294 comes from measured archive values, not an arbitrary large literal.

New monitored/feedback training enables failure economic insertion; historical staged reproduction explicitly opts out. Preserve replay32/target500/batch64/gamma1/n1 and all optimizer/network settings. Episode credit counts only successful terminations; failed transitions contribute only actual replay insertion credit. Existing outcome remains diagnostic.

## Tasks and verification

- [x] Add failing tests in `tests/test_v4_failure_td.py` for controllable SOC failure vs safe actions, structural classification, failed Bellman target, learning views, no fabricated action/shore/modelled settlement, feedback equality, multi-voyage boundaries, program errors and penalty sourcing.
- [x] Add `failure_replay.py` and frozen Train-only penalty reference JSON. Add failure metadata and successful-terminal predicate in `control.py`; share potential terminal correction in `reward_feedback.py`.
- [x] Extend `dqn.py` to store outcome/reason/penalty tags, count success/failure/failed-terminal insertions and failure-terminal TD statistics. Keep Double-DQN and optimizer unchanged; only terminal bootstrap suppression is made explicit.
- [x] Connect preparation/insertion to bootstrap, exploration and read-only greedy reporting in `monitored_training.py`. Add failure categories/event-only counters, replay ratios and per-round counts. Configure identical penalty through `feedback_study.py`; retain old staged opt-out for reproduction.
- [x] Add deterministic repeated replay-learning test using the actual61-action MLP, batch64, Adam1e-4, clip10, gamma1, n1, replay32/target500. Confirm online Q ordering and physical greedy feasibility change while outcome is untouched. Synthetic only; record actual insertions/update/sync counts.
- [x] Run related contracts/regressions and archive/hash checks; independently review critical conversion and accounting boundaries. Record sensitivity targets and limitations, no claim of formal performance improvement.
- [x] Write report and archive synthetic evidence. Final delivery commands are commit/push followed by remote HEAD/clean-status verification. Subsequent original-vs-redistributed experiments must use identical frozen reference and penalty scale.

Commands: `python -m pytest -q tests/test_v4_failure_td.py --tb=short` initially fails on missing APIs; then run it with the existing153-test selection. All test runs use synthetic data. Final source/previous archive manifest hashes and `git diff --check` are verified before push.
