# v4 Repository and Numerical Audit Implementation Plan

> **For agentic workers:** Use parallel independent audit/diagnostic tasks, then root integration and verification. User already approved the complete task specification; no formal training or Test access.

**Goal:** Organize current and historical documentation, remove only proven reproducible local temporaries, and diagnose the unchanged economic TD algorithm on the same synthetic replay at reward scales 1/0.01/0.001.

**Architecture:** Keep all production algorithm/model/ledger files byte-identical to d111c99. Isolate numeric instrumentation in a standalone synthetic diagnostic script. Keep results/figures, frozen penalty reference, calibration inputs and formal datasets at their existing paths. Root moves only audited Markdown/plans, updates relative references, and records exact before/after inventories and hashes.

**Tech Stack:** Python standard library for inventories/import/link/hash audits; existing PyTorch Double-DQN/Adam/Huber/clip and pytest for synthetic diagnostics.

## Approved scope and decisions

- [x] Verify local and remote d111c99, clean branch, previous cleanup report; capture inventory and protected hashes before parallel work.
- [x] Independently audit docs links and source/test dependencies, including dynamic legacy calibration paths. Preserve ambiguous candidates and list the reason.
- [x] Archive superseded reports/plans with relative links remapped; retain hard-read/generated/provenance document paths. Add docs index and update root/current method so old training commands are not presented as current defaults.
- [x] Add TDD-tested standalone reward-scale synthetic diagnosis using identical corpus/RNG/MLP/learning-rate/optimizer budget/target schedule. Keep complete CNY ledgers unchanged.
- [x] Record normal/failure reward and target/Q scales, normalized TD MAE, Smooth L1, sample counts, unclipped gradient norm and clip frequency, full action order and actual synthetic greedy completion.
- [x] Execute two reward-mode matrices of three bounded synthetic diagnostic configurations each. Distinguish numerical conditioning from policy improvement; do not change any formal default.
- [x] Delete scoped, verified old synthetic pytest temporaries and identical local duplicates; record approval-rejected cache/new-basetemp deletion as deferred after preserving proof. Keep all unproven old formal output copies, source/tests, MPC/KAN and calibration fixtures.
- [x] Verify every Markdown link/image, preserved artifacts/manifest SHA256, dependency contracts and relevant synthetic regression tests; report baseline historical missing generated references separately.
- [x] Write repository and numerical reports, exact count/size changes and remaining limitations. Commit/push current branch and verify remote HEAD/clean status; no automatic formal training.

## Verification commands and gates

Use `PYTHONPATH=src` and run the new diagnostic contract tests plus the existing v4 failure, reward, schedule, monitoring and required legacy preflight tests. The formal loader is never invoked by the numerical script. The diagnostic budget is exactly shared among scales; sample-sequence digests and source hashes must match.

Existing protected tracked source/tests/results/figures were SHA256-snapshotted in `outputs/v4_repository_td_audit_20261008/before_inventory.json`. Manifest hashes are checked from the existing five metadata files only. Any discrepancy is a failure until its cause is explicitly resolved; do not rewrite historical evidence hashes to pass.

For deletion use one PowerShell shell with resolved `-LiteralPath`, ensure every target stays inside the active worktree and an explicit approved temporary directory, then remove only those targets. Never clean data or formal output directories. Do not use `git clean` across the workspace.
