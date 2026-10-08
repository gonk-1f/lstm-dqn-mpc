# v4 Uniform Reward Scaling Implementation Plan

> Execute the user's approved specification in the existing direct-power worktree. The user will launch the formal run from PyCharm; no formal training or official trajectory loading is authorized in this implementation session.

**Goal:** Support an explicit economic-reward scale of 0.001, retain the default 1, verify the contracts synthetically, and provide one fresh 40-round command.

**Architecture:** Multiply raw, fully composed training rewards only in `DirectPowerDDQN.remember`. Physical control, observed and modeled ledgers, reward redistribution, failure termination and n-step assembly retain their existing original-unit semantics. The monitored entry records the scale, units and source/data provenance, preserves complete rounds on interruption, and emits read-only diagnostics and curves.

**Tech Stack:** Existing PyTorch MLP Double-DQN, Adam, masked targets and clipping; pytest synthetic fixtures; standard JSON/CSV; Matplotlib.

## Implementation and verification

- [x] Verify branch/base `3d75e5a`; snapshot old model/results hashes and the five existing data manifests without loading trajectories.
- [x] Core TDD: add a finite positive `reward_scale=1.0` constructor argument, scale once on replay insertion, and retain original penalty metadata and outcome labels.
  - RED/GREEN: `python -m pytest tests/test_v4_reward_scaling.py -q`.
  - Cover completed, actual SHORE, modeled settlement, failed suffix, n=1/8, prefill and capacity-saturated insertions.
- [x] Add unit-labelled TD/Q/target statistics, original-unit conversions, real preclip norm and clipping/sample fractions. Keep optimizer, random streams and target cadence unchanged.
- [x] Entry TDD: `--reward-scale`, default 1, invalid scale rejected before loading data, scale recorded in report/model, strict Train/Validation selection and per-round persistence.
  - RED/GREEN: `python -m pytest tests/test_v4_scaled_training_entry.py -q` with only fake datasets.
  - Persist logs, completed-round JSON/CSV and explicit abort metadata; no resume or unqualified best checkpoint.
- [x] Diagnostic TDD: label fixed-probe Q values, retain physical/raw reward columns, add SOC maximum, FC power/delta distributions and read-only learning curves.
  - RED/GREEN: `python -m pytest tests/test_v4_scaled_diagnostics.py -q`.
  - Skipped Validation and incomplete split cost stay missing, never zero.
- [x] Integrate and run focused v4 reward/failure/replay/schedule/monitoring regressions sequentially. No official Train/Validation/Test payloads, no concurrent Torch workers.
- [x] Recheck protected hashes/manifests, document formulas, output files and the PyCharm command, commit/push the implementation.

## Fixed formal command contract

`v4.feedback_study --rounds 40 --reward-feedback redistributed --reward-scale 0.001 --beta-soc 500 --failure-penalty-scale 1 --cadence replay32 --target-interval 500 --n-step 1 --output-dir <new directory>`.

The existing entry fixes seed42, batch64, Adam1e-4, gamma1, replay100000, MLP8–128–64–61, epsilon1→0.05 and progress every50 decisions. Only one process/configuration is requested. Do not invoke this command during implementation.

## Reward contract

For raw final transition reward `r` (including applicable terminal adjustment and once-only failure penalty), replay stores `alpha*r`. N-step assembly happens before this multiplication. For each complete segment, `sum(r_shift)=sum(r_original)`; for a failure suffix, both modes equal the same executed-prefix original reward minus the single `C_fail`. Thus the corresponding scaled sums obey the same identities multiplied by alpha. Neither original CNY economic ledger nor outcome labels are scaled.

## Delivery boundary

This change prepares the formal run; it does not contain its 40-round result. Report generation must distinguish an eligible best round from round40, refuse economic comparison of incomplete splits, and retain any abort evidence. Learning-rate comparisons remain a later decision after the user's run.
