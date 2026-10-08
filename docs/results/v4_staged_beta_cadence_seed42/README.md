# Paused v4 training study

Training was stopped at the user's request on 2026-10-08. No training or agent evaluation was started during archival. Test payloads remained unopened.

- `study_summary.json`: authoritative paused status; four complete Stage 1 runs, partial Stage 2 A/B, and C not run.
- `stage1_beta_*.json` / `*_rounds.csv`: complete 40-round reports and tabular histories.
- `stage2_A_paused.json` / `stage2_B_paused.json`: 18/19 complete monitored rounds. Counters exclude their interrupted 19th/20th rounds.
- `raw/`: every existing file from the experiment output directory, including logs, qualified best weights, available profiles/figures, execution helpers, and prior interruption/resource-failure evidence.
- `raw_archive_inventory.json`: byte count and SHA256 for every copied raw file. Copies were verified against their original files.

Six qualified best models are preserved at `raw/stage1/beta_{250,500,1000,2000}/best_agent.pt` and `raw/stage2/{A_episode16,B_replay16}/best_agent.pt`. These are inference checkpoints containing online weights and selection metadata, **not resumable training snapshots**: optimizer, replay, current policy at interruption, and RNG continuation state were not saved. Stage 1 best checkpoints were independently replay-verified before archival; Stage 2 best checkpoints were not independently replayed after interruption.

`stage1_summary.json` and scripts/logs within `raw/` retain their historical contents. They can describe the earlier "Stage 2 not started" state; use the top-level `study_summary.json` for the final paused status. Source commit for all formal runs is `b35c8408ad1258c089dcc72df9a7c0e5e285d4f5`.

Economic costs and modeled terminal settlement remain separate from the SOC shaping statistic. Incomplete-set costs are not used as a full-set economic comparison. Stage 2 has no completed three-way cadence comparison.
