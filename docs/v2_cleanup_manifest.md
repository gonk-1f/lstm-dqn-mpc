# v2 Cleanup Manifest

Recorded before destructive cleanup on 2026-09-14 (Asia/Shanghai).

## Repository state

- Source branch: `feat/dqn-four-action-diagnostic`
- v2 branch: `refactor/multiscale-dqn-wmpc-v2`
- v1 HEAD retained in Git: `ba81281da8236f0816d352a51ccfd7d176baabaf`
- HEAD subject: `refactor: score executed steps with 84 MPC actions`
- Tracked files before v2 work: `1499`
- Baseline verification: `200` tests passed in `31.421 s`
- Pre-existing untracked user file: `docs/rl_mpc_seven_papers_review_20260914.md` (retained and archived, not deleted)

The source commit and its ancestors remain intact. Any tracked deletion below can be recovered from commit `ba81281`. The complete removed directory trees, including ignored experiment-only files, were moved to the repository-local recovery cache `.git/v2-cleanup-cache/outputs/`; they are absent from the v2 working tree but remain locally recoverable until that cache is deliberately removed.

## Delete from the current v2 tree

The following paths contain only v1 training, ablation, calibration, or diagnostic outputs:

- `outputs/dqn_mpc_mlp_causal_soc_deadband_formal_rounds/` (172 tracked files)
- `outputs/dqn_mpc_mlp_causal_soc_deadband_formal_rounds_20260907_092051/` (142 tracked files)
- `outputs/dqn_mpc_mlp_qsoc80_huber_round1/` (87 tracked files)
- `outputs/dqn_mpc_mlp_qsoc80_loss_ablation_comparison/` (2 tracked files)
- `outputs/dqn_mpc_mlp_qsoc80_mse_buffer300k_gamma099_round1/` (87 tracked files)
- `outputs/dqn_mpc_mlp_qsoc80_mse_buffer300k_gamma0999_round1/` (87 tracked files)
- `outputs/dqn_mpc_mlp_qsoc80_mse_buffer300k_round1/` (87 tracked files)
- `outputs/dqn_mpc_mlp_qsoc80_mse_round1/` (87 tracked files)
- `outputs/fixed_action_validation_016_053_qsoc80/` (10 tracked files)
- `outputs/fixed_physical_l2_84_train_20260912/` (17 tracked plus ignored diagnostics)
- `outputs/ideal_reference_84_train_20260912/` (17 tracked plus ignored diagnostics)
- `outputs/mpc_action_redesign_20260911/` (35 tracked files)
- `outputs/rms_l2_84_train_20260912/` (14 tracked plus ignored diagnostics)
- `outputs/rms_marginals_84_train_20260912/` (6 tracked plus ignored diagnostics)
- `outputs/unified_objective_action_redesign_20260911/` (2 tracked plus ignored diagnostics)
- `outputs/dqn_mpc_train_executed_reward_84_v1_gate_20260913/` (110 ignored v1 gate artifacts)
- `outputs/dqn_mpc_progress_executed_reward_84_v1_handoff_20260914/` (3 ignored v1 handoff artifacts)
- Empty v1 placeholders: `outputs/action_space/` and `outputs/dqn_mpc_mlp_causal_soc_reference_v2_formal_rounds/`

## Archive, do not delete

The following documents move to `docs/archive_v1/`; all were tracked except
the explicitly preserved pre-existing paper review noted above:

- `docs/CLUSTER_BASED_TOTAL_LOAD_REBUILD.md`
- `docs/diagnostic_cleanup_20260913.md`
- `docs/diagnostic_cleanup_20260913.json`
- `docs/dqn_runtime_robustness.md`
- `docs/executed_reward_references_20260913.md`
- `docs/executed_reward_verification_20260913.md`
- `docs/superpowers/plans/2026-09-08-final-dataset.md`
- `docs/superpowers/plans/2026-09-11-dqn-mpc-unified-objective.md`
- `docs/superpowers/plans/2026-09-13-executed-reward.md`
- `docs/superpowers/specs/2026-09-11-dqn-mpc-unified-objective-design.md`
- `docs/rl_mpc_seven_papers_review_20260914.md`

## Retain

- Complete Git history and the source v1 commit.
- `data/processed/operating_dataset_final/metadata/` as historical data provenance only; it is not an accepted v2 raw-data source.
- `outputs/dqn_mpc_literature_executed_reward_84_v1_review_20260914/` as a local literature-source cache, not an experiment result.
- Raw Excel and technical specifications if later supplied. None are present in this workspace at cleanup time.
- Project-configuration values 600 kW and 624 kWh, explicitly labeled as project configuration rather than attributed to Zhou/Yang 2025.

## Compatibility rule

v1 checkpoints, replay buffers, action IDs, objectives, reward semantics, 1 s state windows, and processed 1 s datasets are not compatible with `multiscale_dqn_wmpc_v2`. v2 loaders must reject them before loading payload arrays or model state.
