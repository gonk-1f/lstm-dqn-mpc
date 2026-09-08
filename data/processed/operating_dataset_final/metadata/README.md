# Final operating dataset

This is a data-only, raw-telemetry rebuild. Train covers normal operating fragments; validation contains independently assigned high-quality operating fragments; test contains observed departure-to-arrival voyages. A parent folder is a recording window, not a voyage. No historical controller result is a selection input.

`policy.json` freezes every selection threshold and its empirical or engineering basis. `parent_split_manifest.csv` freezes the final 46/13/7 roles before formal extraction. If the original chronological Test pool is inadequate, `parent_quality_inventory.csv` documents the raw-only completeness/feasibility survey and `policy.json` documents deterministic feature-based parent redesign. Historical trained weights that saw reassigned Test parents are not valid clean benchmark models. No parent may move between roles after observing controller results. `test_voyage_manifest.csv` is the frozen test list; descriptors use within-test feature quartiles only and never affect inclusion.

`test_candidate_boundary_audit.csv` records every strict complete natural-voyage candidate from the formal 8-FC plus 12-cluster chain. Stable load below 10 kW means at least three consecutive observed 30 s points and is a preference only. The survey found no candidate with a stable low-load boundary, so all eight existing representative Test voyages remain frozen. Their FC power is zero at the observed endpoints while positive battery discharge supplies vessel-side load; the recorded reason is `onboard auxiliary load under vessel-side independent supply`. No load was edited and no voyage was trimmed or joined.

`total_load_excels/` is permanently prohibited from formal Test candidate search because it contains 8-FC plus BDM totals rather than the required 8-FC plus 12-cluster reconstruction. Formal candidates come only from the original raw telemetry aligned through the documented 8-FC plus 12-cluster chain.

`aligned_30s/` records every reference point, physical state, disposition and sample owner. `source_point_accounting.csv` covers every reference point exactly once. `exclusion_manifest.csv` includes unused records and stress cases; `source_time_gaps.csv` separately describes unobserved time. `channel_quality.csv` records invalid timestamps and duplicate handling; conflicting duplicates are unavailable, never averaged. Original per-channel files and hashes are in `source_files.csv`.

`alignment_repairs.csv` logs every nonzero matched channel offset, with a separate flag for offsets beyond the former 1 s tolerance, including whether it survives physical and semantic checks. PCHIP is only applied after sample acceptance and never across excluded points or a missing sampling cycle. `load_corrections.csv` logs the narrowly defined stationary sub-kW zero drift; PCHIP floating correction counts are in QA.

Battery power is positive for discharge. Total load is measured FC plus measured battery discharge; it is not an independent demand meter. Inverter power is retained as an auxiliary measured channel; unknown circuit topology prevents labeling it propulsion or service load. Stationary FC-on charging with no independent external-supply evidence is ambiguous and excluded. Sustained stationary net charging with FC power at the empirical shutdown noise floor is high-confidence external supply, not direct plug-state proof.

Raw interval durations use last-first+30 s support and must not be confused with 1 s sample elapsed durations (last-first). Energy and 1 s duration QA use values[1:] to match the existing initialization convention. Descriptive load bins 200/400 kW are fractions of the 600 kW FC rating, not extraction thresholds.

`physical_feasibility_test.csv` includes all candidate audits with assigned/included role; ordinary test rows are feasible. Feasible test allocation witnesses are saved in `feasibility_witnesses/` and checked independently. They are existence certificates under the specified lossless model, not DQN/MPC trajectories. SOC starts at 0.55 in the audit only; no terminal target is imposed. Measured SOC is metadata only.

Rebuild to a NEW destination:

```powershell
python -X utf8 -B src/main/build_final_operating_dataset.py --raw-root "C:/Users/20883/OneDrive/Desktop/氢舟一号" --output-root data/processed/operating_dataset_final --legacy-manifest data/processed/operating_segments_1s_rebuilt/split_manifest.csv
```

An existing output is refused. Do not remove CSVs by hand. Source/code hashes permit deterministic reproduction. Old datasets, controller defaults, staged changes and historical outputs are outside this builder's write scope.
