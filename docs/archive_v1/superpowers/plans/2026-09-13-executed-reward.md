# Executed closed-loop reward implementation plan

Goal: apply the user's fixed 84-action / executed first-step reward specification, without DQN training or Validation/Test data access.

Architecture: keep the causal state, persistence forecast, N=6 convex QP and physical execution unchanged. Version integer-composition actions, inference checkpoints and replay semantics together. Score only executed power and next SOC. Keep historical scoring in an explicit legacy module.

Tech stack: Python, NumPy, PyTorch, OSQP, unittest.

- [x] Check seven primary papers and archive precise applicability limits.
- [x] Inventory output/script dependencies; write KEEP/DELETE dry-run before deletion. Preserve raw/processed data, manifests, historical baselines and latest marginal audit inputs.
- [x] Add failing tests for 84 deterministic actions, first-step reward invariance/symmetry/linearity, incompatible checkpoints and numerical failure exclusion from replay.
- [x] Implement action metadata in `src/dqn/utils/action_mapper.py`, physical reward in `reward.py`, and named old formulas in `legacy_reward.py`; connect environment after execution validation.
- [x] Update DQN config, model/replay/training state serialization, failure classification and training entrypoint gate; use a new formal output namespace.
- [x] Update README with equations, units, evidence limits, timing, references and baseline reproduction instructions.
- [x] Run focused tests, full unittest with a guard against held-out production data reads, compileall and git diff --check. Review final diff independently.
- [x] Prepare only source, tests and compact documentation/archive artifacts for delivery.

Delivery after verification: commit and push the current branch, then verify the remote hash. The final task response records the delivery result.

Acceptance: r=-(h+0.5*b^2+0.5*s^2+0.5*f^2), h from existing Dp0 curve at actual FC power divided by rate at 600 kW; b=P_batt/624; s=(SOC_next-.55)/.05; f=(P_fc-P_fc_prev)/48. T_sw=Ts=1 s, gamma=.99. No new stress thresholds or penalty calibration; uncalibrated terminal penalty blocks formal entry before loading data. Numerical failures abort without a fabricated transition.
