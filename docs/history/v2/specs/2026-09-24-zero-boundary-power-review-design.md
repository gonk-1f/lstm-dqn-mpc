# Zero-Boundary Dataset Power Review Design

## Purpose

Generate a visual review of every segment in the frozen zero-boundary v2
operating dataset. The review is an inspection artifact only. It must not
change segment values, interpolation, split membership, or formal-training
preflight state.

## Source contract

- Read `data/processed/operating_dataset_zero_boundary_v2/metadata/sample_manifest.csv`.
- Require exactly 53 manifest rows with the frozen 38 Train, 10 Validation,
  and 5 Test split counts.
- Resolve every input CSV through its manifest `relative_path`.
- Plot the stored `time_s` and `load_total_kw` columns without resampling,
  clipping, smoothing, or interpolation.
- Require finite values, a strictly one-second time axis beginning at zero,
  and exact zero-power endpoints before plotting.
- Preserve and display all internal negative-power intervals.

## Figure design

Create one PNG per manifest row under
`outputs/v2_zero_boundary_power_review/{train,validation,test}/`.

- Use a separate y-axis range for every segment, as selected by the user.
  Include zero in the range and add 5% padding around that segment's finite
  minimum-to-maximum span; use a fixed 1 kW pad for an all-zero segment.
- Plot total power as one continuous line.
- Use elapsed time from zero on the x-axis in seconds and total power in kW
  on the y-axis.
- Draw a visible 0 kW reference line and keep the first and last zero-power
  samples visible.
- Title each figure with parent identifier, sample ID, and split.
- Show duration, point count, minimum, maximum, and mean power as compact
  factual metadata.
- Use a legible, restrained technical style suitable for manual dataset
  screening. The independent y-scales must be stated in the index so that
  visual heights are not treated as cross-segment magnitude comparisons.

## Index and provenance

Generate `outputs/v2_zero_boundary_power_review/index.html` with separate
Train, Validation, and Test sections in manifest order. Each entry embeds its
PNG and displays the same segment identifiers and summary statistics.

Also write `review_manifest.csv` containing each image's relative path, source
CSV relative path, split, point count, duration, summary statistics, and source
CSV SHA-256. This review manifest records provenance but does not replace the
formal dataset manifest.

## Reproducibility and safety

- Add a tracked command-line builder under `src/main/`.
- Refuse to overwrite an existing output directory unless an explicit
  replacement flag is supplied.
- When replacement is requested, replace only the exact review output
  directory after resolving and validating that path under repository
  `outputs/`.
- Do not modify the dataset, the old `outputs/v2_segment_power_review`, raw
  telemetry, split metadata, or preflight files.
- Use a non-interactive Matplotlib backend and close every figure.

## Verification

Focused tests must cover manifest validation, preserved negative values,
per-segment output naming, independent y-axis behavior, exact image count,
index split grouping, provenance hashes, overwrite refusal, and forbidden
source mutation. The generated review must contain exactly 53 PNG files and
one entry per formal segment. Run focused tests, relevant v2 tests,
compile/import checks, `git diff --check`, and a source-hash comparison before
reporting completion.
