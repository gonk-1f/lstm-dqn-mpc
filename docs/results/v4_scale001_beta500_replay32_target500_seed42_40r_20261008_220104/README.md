# v4 redistributed reward scale 0.001: completed 40-round run

Source `c382943f8ddea75624dc08d75445ca4a304b95ba`, clean worktree at launch. Run started 2026-10-08 22:01:04; original output directory retained in `outputs/v4_scale001_beta500_replay32_target500_seed42_40r_20261008_220104/`.

## Verified result

- Completed all 40 rounds; no automatic resume or new experiment.
- Peak greedy Train 22/30, rounds 9 and 10; final round 40 greedy Train 12/30.
- Validation skipped in every round because greedy Train never reached 30/30. Validation is unevaluated, not 0/8.
- No qualified checkpoint and no `best_agent.pt`; no invented final or best model.
- Test payloads opened: 0; five dataset manifests verified unchanged.
- Actual/modelled economic costs retain CNY; incomplete split cost is null and cannot be compared as a complete voyage cost.

## Reports and figures

- [Result report](../../v4_scaled_reward_40r_2026-10-08.md)
- [Original report](raw/report.json), [round history](raw/round_history.json), [round metrics](raw/round_metrics.csv)
- [Run metadata and hyperparameters](raw/run_metadata.json), [full console log](raw/train.log)
- [Completion history](raw/completion_rates.png)
- [Validation economic cost figure](raw/validation_economic_costs.png): no evaluated Validation points
- [TD and gradients](raw/td_and_gradients.png), [SOC and epsilon](raw/soc_and_epsilon.png)
- [FC and update counts](raw/fc_and_update_counts.png)
- Final round representative [completed](raw/final_round_040_greedy_train_completed_trajectory.png) and [failed](raw/final_round_040_greedy_train_failed_trajectory.png) Train profiles
- [Plot metadata and underlying series](raw/learning_curves_metadata.json)

## Complete lossless archive

[Archive manifest](archive_manifest.json) enumerates all 93 original files with original path/size/SHA256, archived path/size/SHA256 and decoded SHA256.

The 80 per-round exploratory/greedy Train trajectory JSON files are independently compressed to `.json.gz` under `raw/`; all 13 other files are verbatim copies. Original payload totals 1,897,649,995 bytes; archived payload totals 97,989,780 bytes. Every decoded file has exactly the original SHA256. Original local files are not deleted or overwritten.

The local `.gitattributes` disables Git line-ending conversion under `raw/`, preserving manifest hashes across platforms.

For example, [round 40 greedy Train trajectory](raw/round_040_train_trajectories.json.gz) can be read with Python `gzip.open(path, 'rt', encoding='utf-8')`, or decompressed into a separate local inspection directory. Compression changes storage only; no rewards, ledgers or physical trajectory values are rewritten.

## Execution counts

Economic replay: 495246 insertions = 431434 success + 63812 failure; failure terminals: 468. Updates: 15476 = 571 prefill + 14905 formal. Target: 31 copies = initial 1 + scheduled 30. Transition credit: 14. All counts satisfy replay32 and target500. Training executed 476973, prefill 18273, and greedy evaluation 358165 ONBOARD transitions.

This archive preserves evidence. It does not assert that reward scaling solved completion, diagnose a definitive cause, or authorize parameter changes/new training.
