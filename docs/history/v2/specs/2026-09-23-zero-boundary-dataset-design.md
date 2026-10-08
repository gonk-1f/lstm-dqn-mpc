# Zero-Boundary Operating Dataset Design

## Purpose

Audit all 66 recognizable Hydrogen Boat 1 parent recording windows and build a new versioned operating dataset from the 53 windows that contain auditable zero boundaries. Each included parent produces one trimmed operating segment whose load starts and ends at zero. Remove the leading stationary zero-power dwell and the terminal stationary or shore-charging tail while retaining all behavior between the departure and arrival boundaries.

This change builds and splits the operating dataset only. It does not construct a DQN episode payload, audit or freeze the DQN state, screen actions, run sensitivity analysis, train a model, or make `FORMAL_TRAINING` automatically `GO`.

## Source and Power Semantics

- Use the audited original 8-FC, 12-battery-cluster, and AIS telemetry readers.
- Use the accepted near-synchronous alignment with a 10-second tolerance.
- Use the approved review-series gap filling for all diagnosed internal gaps: local natural cubic splines interpolate FC total and raw-sign battery total independently on the nominal 30-second grid; source total is derived as `P_fc_total - P_batt_raw_total`.
- Preserve the raw battery convention: negative battery power is discharge and positive battery power is charge.
- The dataset load is `P_source_total`. It is not described as an independently measured whole-vessel load.
- Every measured, aligned, cubic-imputed, boundary-constructed, and one-second PCHIP value retains distinct provenance.

## One Segment per Eligible Parent

Each of the 53 eligible parent recording windows produces exactly one final segment. Internal zero-power periods, stops, and negative-power intervals remain in the segment when they occur between the selected start and end boundaries. The rule removes only the pre-departure prefix and the post-arrival suffix.

The build audits all 66 parents and excludes only the following 13 windows whose available records do not bracket the required boundary:

- Missing start boundary: `3月25日14_00_3月25日17_00`, `4月23日13_00_4月23日18_00`, `6月13日08_00_6月13日14_00`.
- Missing end boundary: `3月28日08_00_3月28日11_00`, `4月7日08_00_4月7日12_00`, `4月21日08_00_4月21日16_00`, `4月24日07_00_4月24日17_00`, `4月29日08_00_4月29日18_00`, `6月6日10_00_6月6日21_00`, `7月9日08_00_7月9日16_00`, `7月19日07_00_7月19日09_00`, `7月22日08_00_7月22日11_00`, `7月24日14_00_7月24日17_00`.

These are incomplete recording windows for this boundary contract, not failed interpolation cases. The build records them in `excluded_parent_manifest.csv`; it does not force their observed active boundary value to zero, extrapolate outside the recording, or silently exclude any additional parent. Any unexpected missing boundary or different exclusion set fails the build.

## Boundary Detection

Use `P_source_total_kw` as the boundary signal.

- The zero-power deadband is `abs(P_source_total_kw) <= 1 kW`.
- Sustained operation requires three consecutive approximately 30-second points with `P_source_total_kw > 1 kW`.
- Identify the first and last sustained-operation blocks across the whole parent. This retains internal stops between them.

Start boundary:

1. Search backward from the first sustained-operation block.
2. Prefer the last observed or accepted aligned point in the zero-power deadband.
3. If the signal crosses from nonpositive to positive without a deadband point, linearly solve for the zero crossing between the bracketing points.
4. Do not extrapolate before the available record.

End boundary:

1. Search forward from the last sustained-operation block.
2. Prefer the first observed or accepted aligned point in the zero-power deadband.
3. If the signal crosses directly from positive operation to negative shore charging, linearly solve for the zero crossing between the bracketing points.
4. Discard every later stationary, zero-power, or shore-charging suffix point.
5. Do not extrapolate after the available record.

For a deadband boundary, canonicalize the dataset boundary load to exactly zero and record the original value and correction in the boundary audit. For a constructed crossing, linearly interpolate FC total and raw battery total using the same crossing fraction and derive source total; the resulting source boundary must be zero within numerical tolerance.

## One-Second Reconstruction

- Retain the selected 30-second anchors, including the two zero boundaries.
- Reconstruct the trimmed load series at one-second intervals using PCHIP.
- Round a constructed boundary timestamp to the nearest second for the provenance timestamp and record the sub-second adjustment in the audit.
- Set the model time axis to exact float seconds `0, 1, 2, ...`.
- Require the first and last `load_total_kw` values to be exactly zero.
- Preserve original timestamps as provenance only; formal models use `time_s`.
- Do not interpolate beyond either selected boundary.

Each output CSV contains:

- `timestamp`
- `time_s`
- `load_total_kw`

## Fixed Test Set

The following five parent windows form the entire Test set:

- `3月26日14_00_3月26日16_00`
- `3月29日08_00_3月29日15_00`
- `4月18日12_00_4月18日18_00`
- `5月8日08_00_5月8日17_00`
- `6月11日08_00_6月11日11_00`

These names each match exactly one recognizable parent. No fragment from any of these parents may appear in Train or Validation.

## Train and Validation Split

After the 13 approved exclusions and five fixed Test parents, the remaining 48 parents are split into:

- Train: 38 parents
- Validation: 10 parents

Selection is deterministic and uses no controller, model, reward, or performance result.

1. Calculate trimmed-segment month, duration, mean load, and P95 load.
2. Calculate duration, mean-load, and P95 quartile bins using only the 48 non-Test included parents.
3. Represent every parent by its month and three quartile labels.
4. Select 10 Validation parents using deterministic greedy iterative stratification. At each step, select the candidate that minimizes the total absolute deviation from the 20% target counts across month and feature-bin labels.
5. Break equal scores by chronological parent order.
6. Assign the remaining 38 parents to Train.

The manifest freezes the resulting parent-level allocation. No parent can span splits.

## Output

Write a new dataset root:

`data/processed/operating_dataset_zero_boundary_v2/`

Required structure:

- `train/`: 38 CSV files
- `validation/`: 10 CSV files
- `test/`: 5 CSV files
- `metadata/sample_manifest.csv`
- `metadata/parent_split_manifest.csv`
- `metadata/excluded_parent_manifest.csv`
- `metadata/trim_boundary_audit.csv`
- `metadata/interpolation_audit.csv`
- `metadata/qa_summary.json`
- `metadata/source_files.csv`
- `metadata/policy.json`

The existing `data/processed/operating_dataset_final/` remains unchanged. The new builder refuses to overwrite an existing destination. After all checks pass, update the formal dataset loader's default root to the new dataset.

`sample_manifest.csv` is the authoritative segment-level input to the existing formal loader. It contains one row per parent segment and freezes `parent`, `sample_id`, `relative_path`, `split`, and `point_count_1s`. `parent_split_manifest.csv` is the parent-level split and stratification audit; it is not a runtime fallback for the formal loader.

## Audit Requirements

`trim_boundary_audit.csv` records, for both sides of every parent:

- parent identifier and assigned split;
- boundary kind: observed zero/deadband or constructed crossing;
- adjacent source timestamps and loads;
- original deadband value, if canonicalized;
- crossing fraction and unrounded crossing timestamp, if constructed;
- rounded provenance timestamp;
- removed prefix or suffix point and duration counts;
- first/last sustained-operation evidence.

`excluded_parent_manifest.csv` records each approved excluded parent, whether the missing boundary is `start` or `end`, first/minimum/maximum/last available source power, deadband and nonpositive point counts, and the explicit `NO_BRACKET_WITHIN_RECORDING` reason.

`interpolation_audit.csv` records the 30-second cubic gaps, generated-point counts, natural-cubic warnings, and one-second PCHIP point counts. It distinguishes measured and imputed records.

`policy.json` freezes every threshold, fixed Test parent, split count, stratification feature, tie-break rule, and interpolation method.

`qa_summary.json` reports all acceptance checks and hashes of sources, relevant code, and delivered artifacts.

## Acceptance Criteria

- Exactly 53 final segment CSV files exist.
- Train/Validation/Test counts are exactly 38/10/5.
- The exclusion manifest contains exactly the 13 approved incomplete-boundary parents and no others.
- The five specified parents are the complete Test set.
- Parent leakage count is zero.
- Every CSV has finite numeric load values.
- Every `time_s` starts at zero and advances by exactly one second.
- Every segment starts and ends with `load_total_kw == 0`.
- Sustained positive operation exists after the start boundary and before the end boundary.
- No terminal stationary or shore-charging suffix remains.
- Internal stops and internal negative-power intervals are retained.
- Every constructed boundary is fully auditable and does not use extrapolation.
- The 30-second source identity `P_source = P_fc - P_batt_raw` holds at interpolated boundaries.
- Original telemetry, the old dataset, and `outputs/v2_segment_power_review/gap_diagnostics.csv` retain their pre-build hashes.
- The formal loader successfully loads all three new splits.

## Tests and Verification

Test-driven implementation covers:

- leading zero-dwell trimming;
- terminal zero-dwell trimming;
- positive-to-negative shore-tail zero crossing;
- preservation of internal stops and negative intervals;
- explicit failure without bracketed zero boundaries;
- boundary component interpolation and power identity;
- one-second time-axis reset and zero endpoints;
- fixed Test-parent assignment;
- deterministic 38/10 Train/Validation stratification;
- parent leakage prevention;
- refusal to overwrite an existing destination;
- formal loader integration with the new root.

Run focused tests, all v2 tests, the complete test suite, compile/import checks, artifact/hash validation, and `git diff --check` before completion.
