# Frozen DDPG v5: 500-episode fixed-voyage baseline

This archive preserves every file from the completed initial run, its resumed
continuation, and the final deterministic Test4 evaluation. Original outputs
remain in place. Other historical runs and Optuna studies are outside this archive.

## Experiment identity

- Training code: `200dcfd87f9ed17486eca5cf8bf7062f5d34614c`.
- Reward: `ddpg_economic_negative_no_start_stop_v5`; six negative economic costs,
  failure reward -2000, no independent SOC penalty.
- One aggregate FC and battery; five observations; 256x256 Actor/Critic;
  Huber Critic delta=1; gamma=1; learning_rate=0.001; tau=0.01;
  normal noise_std=0.1; train_freq=32; gradient_steps=1; batch_size=64.
- Fixed voyage `parent_005_seg01`, initial SOC=0.6, seed=42, 500 episodes.
- All 500 noisy training episodes completed. Final model: 4,630,186 environment
  steps and 144,537 Adam updates.
- Validation at episode 500 completed 4/8 voyages; mean coverage 0.855292.
- Final Test4 completed 4/4 voyages; total economic cost 4640.38798673774 CNY.
  Model/optimizer/counter/Replay/normalization/file integrity checks all passed.

## Files

- `initial/`: original run 4, episodes 1-195, its model, Replay and checkpoints.
- `resume/`: continuation through episode 500, final model, Replay, checkpoints,
  fixed-Train diagnostics and periodic Validation8 records.
- `test4/`: final frozen-checkpoint results, trajectories and figures.
- `manifest.json`: original relative paths, byte counts, SHA-256 checksums,
  compression and verification results for all 196 source files.

Large CSV and Replay files are losslessly gzip-compressed. Other files,
including all 32 model ZIPs and 18 figures, are copied byte-for-byte. To restore,
decompress files marked `gzip`, remove their added `.gz` suffix and copy files
to `source_relative_path` from the manifest. Restored file SHA-256 must equal
`source_sha256`. Every archived file was checked against its source this way.

These are historical v5 artifacts. Current v6 uses environment-managed FC
start/stop, SOC weight 10, failure reward -20000 and 128x128 networks. Its identity
checks intentionally reject these models and Replay for v6 continuation.
Run new v6 training from scratch using the project-root `run_ddpg_train.py`.
Historical resume commands in this archive require the recorded v5 code revision.
