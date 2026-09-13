# Diagnostic cleanup — 2026-09-13

Legacy committed reproduction point: `adbd5526d107faeaca62a1d00c03d209b4829fdc`. This inventory was written before deletion. No formal data or held-out payload was read.

Parent KEEP rows preserve their tree except the explicitly listed DELETE children. Reference scan covers README, source, tests and docs; manifests/data are protected wholesale.

| Decision | Path | Reason | References | Paper reproduction |
|---|---|---|---|---|
| KEEP | `outputs/action_space/` | Historical baseline or audit dependency; no evidence supports deleting unique results | No direct source/doc/test reference found | Yes |
| KEEP | `outputs/dqn_mpc_mlp_causal_soc_deadband_formal_rounds/` | Historical baseline or audit dependency; no evidence supports deleting unique results | No direct source/doc/test reference found | Yes |
| KEEP | `outputs/dqn_mpc_mlp_causal_soc_deadband_formal_rounds_20260907_092051/` | Historical baseline or audit dependency; no evidence supports deleting unique results | No direct source/doc/test reference found | Yes |
| KEEP | `outputs/dqn_mpc_mlp_causal_soc_reference_v2_formal_rounds/` | Historical baseline or audit dependency; no evidence supports deleting unique results | README.md | Yes |
| KEEP | `outputs/dqn_mpc_mlp_qsoc80_huber_round1/` | Historical baseline or audit dependency; no evidence supports deleting unique results | No direct source/doc/test reference found | Yes |
| KEEP | `outputs/dqn_mpc_mlp_qsoc80_loss_ablation_comparison/` | Historical baseline or audit dependency; no evidence supports deleting unique results | No direct source/doc/test reference found | Yes |
| KEEP | `outputs/dqn_mpc_mlp_qsoc80_mse_buffer300k_gamma0999_round1/` | Historical baseline or audit dependency; no evidence supports deleting unique results | No direct source/doc/test reference found | Yes |
| KEEP | `outputs/dqn_mpc_mlp_qsoc80_mse_buffer300k_gamma099_round1/` | Historical baseline or audit dependency; no evidence supports deleting unique results | No direct source/doc/test reference found | Yes |
| KEEP | `outputs/dqn_mpc_mlp_qsoc80_mse_buffer300k_round1/` | Historical baseline or audit dependency; no evidence supports deleting unique results | No direct source/doc/test reference found | Yes |
| KEEP | `outputs/dqn_mpc_mlp_qsoc80_mse_round1/` | Historical baseline or audit dependency; no evidence supports deleting unique results | No direct source/doc/test reference found | Yes |
| KEEP | `outputs/fixed_action_validation_016_053_qsoc80/` | Historical baseline or audit dependency; no evidence supports deleting unique results | No direct source/doc/test reference found | Yes |
| KEEP | `outputs/fixed_physical_l2_84_train_20260912/` | Historical baseline or audit dependency; no evidence supports deleting unique results | src\main\audit_fixed_physical_l2_84_train.py, src\main\audit_rms_l2_84_train.py | Yes |
| KEEP | `outputs/ideal_reference_84_train_20260912/` | Historical baseline or audit dependency; no evidence supports deleting unique results | src\main\audit_ideal_reference_84_train.py, src\main\audit_rms_l2_84_train.py, src\main\audit_rms_marginals_84_train.py | Yes |
| KEEP | `outputs/mpc_action_redesign_20260911/` | Historical baseline or audit dependency; no evidence supports deleting unique results | src\main\calibrate_unified_mpc_actions_train.py | Yes |
| KEEP | `outputs/rms_l2_84_train_20260912/` | Historical baseline or audit dependency; no evidence supports deleting unique results | src\main\audit_rms_l2_84_train.py | Yes |
| KEEP | `outputs/rms_marginals_84_train_20260912/` | Latest Train-only SOC marginal conclusion; keep inputs/provenance | src\main\audit_rms_marginals_84_train.py | Yes |
| KEEP | `outputs/unified_objective_action_redesign_20260911/` | Historical baseline or audit dependency; no evidence supports deleting unique results | src\main\audit_ideal_reference_84_train.py, src\main\calibrate_unified_mpc_actions_train.py | Yes |
| DELETE | `outputs/mpc_action_redesign_20260911/normalized_objective_reward_audit/action_distribution.csv` | Superseded by unified objective and latest 84-action audits; conclusion preserved in adjacent summary.json; original state_probe.csv retained | No direct source/doc/test reference found | No unique result; conclusion/source retained |
| DELETE | `outputs/mpc_action_redesign_20260911/normalized_objective_reward_audit/per_state_action.csv` | Superseded by unified objective and latest 84-action audits; conclusion preserved in adjacent summary.json; original state_probe.csv retained | No direct source/doc/test reference found | No unique result; conclusion/source retained |
| DELETE | `outputs/mpc_action_redesign_20260911/normalized_objective_reward_audit/state_winners_and_reward_gap.csv` | Superseded by unified objective and latest 84-action audits; conclusion preserved in adjacent summary.json; original state_probe.csv retained | No direct source/doc/test reference found | No unique result; conclusion/source retained |
| DELETE | `outputs/mpc_action_redesign_20260911/normalized_objective_reward_audit/winner_by_fc_power.csv` | Superseded by unified objective and latest 84-action audits; conclusion preserved in adjacent summary.json; original state_probe.csv retained | No direct source/doc/test reference found | No unique result; conclusion/source retained |
| DELETE | `outputs/mpc_action_redesign_20260911/normalized_objective_reward_audit/winner_by_load.csv` | Superseded by unified objective and latest 84-action audits; conclusion preserved in adjacent summary.json; original state_probe.csv retained | No direct source/doc/test reference found | No unique result; conclusion/source retained |
| DELETE | `outputs/mpc_action_redesign_20260911/normalized_objective_reward_audit/winner_by_load_transition.csv` | Superseded by unified objective and latest 84-action audits; conclusion preserved in adjacent summary.json; original state_probe.csv retained | No direct source/doc/test reference found | No unique result; conclusion/source retained |
| KEEP | `outputs/mpc_action_redesign_20260911/normalized_objective_reward_audit/winner_by_soc.csv` | Superseded by unified objective and latest 84-action audits; conclusion preserved in adjacent summary.json; original state_probe.csv retained | src\main\audit_ideal_reference_84_train.py | No unique result; conclusion/source retained |
| DELETE | `outputs/mpc_action_redesign_20260911/normalized_objective_reward_audit/winner_by_soc_regime.csv` | Superseded by unified objective and latest 84-action audits; conclusion preserved in adjacent summary.json; original state_probe.csv retained | No direct source/doc/test reference found | No unique result; conclusion/source retained |
| KEEP | `src/main/audit_fixed_physical_l2_84_train.py` | Retained audit reproducibility or dataset construction dependency | No direct source/doc/test reference found | Yes |
| KEEP | `src/main/audit_ideal_reference_84_train.py` | Retained audit reproducibility or dataset construction dependency | src\main\audit_fixed_physical_l2_84_train.py, tests\test_ideal_reference_84_audit.py | Yes |
| KEEP | `src/main/audit_rms_l2_84_train.py` | Retained audit reproducibility or dataset construction dependency | src\dqn\utils\legacy_reward.py | Yes |
| KEEP | `src/main/audit_rms_marginals_84_train.py` | Retained audit reproducibility or dataset construction dependency | No direct source/doc/test reference found | Yes |
| KEEP | `src/main/build_spline_1s_diagnostics.py` | Retained audit reproducibility or dataset construction dependency | src\main\build_total_load_dataset_721.py, tests\test_spline_1s_diagnostics.py | Yes |
| KEEP | `src/main/calibrate_unified_mpc_actions_train.py` | Retained audit reproducibility or dataset construction dependency | tests\test_unified_objective_calibration.py | Yes |
| KEEP | `data/ (all raw/processed/fuel_cell/manifests and referenced files)` | Protected formal data; no payload read or deletion | formal data loaders and dataset tests | Yes |

## Existing removals recovered from the interrupted task

`src/main/diagnose_four_action_pretraining_soc_deadband.py`, `tests/test_four_action_pretraining_soc_deadband.py`, and `tests/test_mpc_soc_deadband.py` were already deleted in the recovered worktree. Their historical versions remain at the Git commit above; the continuous SOC reference tests replace the obsolete deadband assertions. This task does not claim to have newly deleted them.

## Baselines and dependency chain

`legacy_yuan_like_self_cost` retains the pre-change project reward. `legacy_predicted_common_reward` freezes the RMS/L2 predicted scoring candidate; the isolated ideal/reference and physical-L2 scripts remain available as historical variants. `calibrate_unified_mpc_actions_train.py` explicitly uses the frozen legacy four-action table.

`unified_objective_action_redesign_20260911/final_state_action_probe.csv` -> `ideal_reference_84_train_20260912/{states.csv,state_physical_results.npz}` -> fixed/RMS/L2 and latest marginal audits. These inputs and scripts are retained. Old trained model/replay files remain legacy artifacts and cannot resume the current method.

No new deletion of diagnostic scripts: the remaining scripts have dependencies or methodological value. Large retained local caches are excluded from this commit; compact reports/provenance are archived. The JSON inventory records hashes and sizes of deletion candidates.

Execution: the seven DELETE files were hash-checked against the dry-run inventory and removed. All targets were verified inside the named diagnostic directory before deletion.
