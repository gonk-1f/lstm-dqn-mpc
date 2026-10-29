# V6 one-plus economic reward implementation plan

> **For agentic workers:** Use subagent-driven-development; audit must precede production edits.

**Goal:** Audit the small Train loads, then add an isolated `inverse_one_plus_economic` mode with exact CNY input and true-transition failure reward replacement.

**Architecture:** Keep source loads and frozen Zhou accounting intact unless the source audit establishes a supported positive deadband. Use a shared reward contract for the direct environment, episode runner and episode Optuna adapter. Preserve legacy defaults and strict resume identity checks. No formal training, long search or Test payload access is authorized.

**Tech stack:** Python, Gymnasium, SB3 MaskedDQN, pytest, Optuna SQLite; existing isolated interpreter.

- [ ] Audit source lineage, asymmetric normalization, contracts and tests; scan all 35 Train load arrays, with explicit provenance for associated physical values. Write a standalone report and commit it before production edits.
- [ ] Add failing numeric/domain and environment terminal tests. Implement `1.0 / (1.0 + total_cost_cny)` for finite CNY costs with finite positive denominator. Keep old formulas unchanged. A real physical failure replaces the last real reward with -1 and terminates, without a synthetic action or ledger penalty.
- [ ] Add failing runner/interface tests. Propagate reward mode and contract through configuration, validation, checkpoints, resume and comparison. Reject incompatible exports and ambiguous imports for the new mode. Preserve strict code/data/dependency checks on resume.
- [ ] Extend only the complete-episode Optuna adapter with explicit mode selection, a distinct study name for the new mode, identity binding and mode-correct exports. Reject accidental use through the old step-budget path.
- [ ] Run necessary unit/regression checks and bounded synthetic smoke: three 64-step episodes, exact pause/resume, at most one two-episode Optuna Trial, fresh export/retrain and comparison. No real Validation or Test payload is needed for these checks.
- [ ] Review the diff independently, fix material findings, record actual commands/output and preservation evidence. Provide paths, known limitations and future commands; do not execute formal commands.

New reward contract: numeric costs remain CNY; reward is dimensionless; C=0 gives 1; negative C greater than -1 is allowed and gives reward above 1; C<=-1 and nonfinite inputs fail before Replay insertion. Normal completion has no bonus. Existing failure penalty 10132.66 applies only to legacy modes. This change does not resolve the separately deferred real-time SOC inventory design and does not establish equivalence between reward maximization and minimum cumulative economic cost.

Compatibility: source hashes deliberately prevent old-checkpoint continuation under changed code. Preserve old directories and resume them using their original commit/dependencies. Never rewrite old identities, reload an old-mode Replay into the new mode, or overwrite a previous study.
