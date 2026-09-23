# Segment Power Review Cubic-Spline Design

## Purpose

Regenerate the human-review plots in `outputs/v2_segment_power_review` with every existing internal gap greater than 45 seconds populated on a nominal 30-second grid. The result is a visualization artifact only. It does not alter raw Hydrogen Boat 1 telemetry, construct a formal dataset, select segments, or change training/preflight state.

## Scope

- Process all 140 diagnosed internal gaps, including gaps up to 630 seconds.
- Use the existing relaxed near-synchronous observed points as interpolation anchors.
- Preserve every observed timestamp and observed value.
- Generate new values only strictly between two observed gap-boundary points.
- Do not extrapolate before the first or after the last observed point in a parent segment.
- Do not modify FC, BMS, AIS, or EMS source CSV files.
- Do not modify dataset splits, formal training inputs, control configuration, economics, or preflight logic.

## Interpolation Model

For each gap independently:

1. Select up to the three nearest observed supervisory points before the gap and up to the three nearest observed points after it.
2. Require at least four distinct anchor timestamps. If this invariant is not met, fail the parent plot explicitly instead of silently extrapolating or substituting another interpolation method.
3. Fit a local natural cubic spline using actual elapsed seconds as the independent variable.
4. Compute the nominal missing count as `round(gap_seconds / 30) - 1`, then generate that many timestamps at `left_boundary + n * 30 seconds`. This respects the accepted 28–31-second clock jitter and avoids creating a synthetic point only 1–10 seconds before a measured right boundary.
5. Evaluate separate splines for:
   - `P_fc_total_kw`
   - `P_batt_raw_total_kw`
6. Derive, rather than independently interpolate:

   `P_source_total_kw = P_fc_total_kw - P_batt_raw_total_kw`

This preserves the raw battery sign convention and the source-power accounting identity at every generated point.

No clipping is applied to the spline result. Negative FC values or values outside the local observed range are recorded as interpolation warnings so that the visualization does not silently disguise spline overshoot.

## Plot Semantics

- Draw observed and interpolated points as a connected time series.
- Keep the existing colors and labels for source, FC, and raw battery power.
- Mark interpolated 30-second points with hollow markers.
- Shade interpolated intervals lightly and label them as cubic-spline imputation, not measured telemetry.
- Remove the previous dashed endpoint-only visual bridge for gaps that have been interpolated.
- Continue to show the zero-power reference line.
- Show observed-point and interpolated-point counts separately in plot metadata.

## Index and Diagnostics

Existing observed-data statistics remain based only on observed aligned samples. They must not be silently recomputed from imputed values.

Add review metadata sufficient to audit the interpolation:

- `interpolated_point_count`
- `interpolation_gap_count`
- `max_interpolated_gap_seconds`
- `interpolation_method`
- `interpolation_warning_count`

Keep `gap_gt_45s_count` as the count of gaps in the observed telemetry. Preserve `gap_diagnostics.csv` unchanged as the raw-evidence record.

## Failure Handling

- A parent with insufficient distinct anchors for any requested gap receives an explicit non-OK plot status and reason.
- Duplicate timestamps are resolved by the existing duplicate policy before interpolation.
- Non-finite spline results fail the affected parent explicitly.
- Overshoot is reported but not clipped or treated as measured data.

## Tests

Test-driven implementation will cover:

- generation of strictly interior 30-second timestamps;
- no start/end extrapolation;
- local natural-cubic interpolation of FC and raw battery totals;
- exact preservation of observed points;
- `P_source = P_fc - P_batt_raw` at every interpolated point;
- preservation of the raw battery sign convention;
- explicit failure when fewer than four distinct anchors exist;
- interpolation metadata and warning counts;
- removal of NaN breaks and endpoint-only bridges from regenerated plots;
- regression of the existing relaxed alignment and raw-power aggregation behavior.

## Acceptance Criteria

- All 66 recognizable parent segments are processed.
- All 140 diagnosed internal gaps receive nominal 30-second interpolated points unless a documented anchor invariant fails.
- Plots visibly distinguish observed from interpolated data.
- Raw source files and `gap_diagnostics.csv` are unchanged.
- No dataset split, segment-selection, preflight, MPC, economics, degradation, or training artifact is modified.
