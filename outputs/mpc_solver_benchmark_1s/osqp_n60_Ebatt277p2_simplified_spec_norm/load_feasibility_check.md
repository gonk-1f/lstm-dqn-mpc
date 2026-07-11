# Load Feasibility Check

Scope: test split of the offline natural-clipped 1 s spline benchmark. This is not measured 1 s data.

- P_fc_max = 560.0 kW
- P_batt_max = 138.6 kW
- P_available_max = 698.6 kW
- P_load_max = 820.134823 kW
- P_load_p99 = 689.697188 kW
- P_load_p95 = 520.521065 kW
- P_load_mean = 208.291112 kW
- num_steps_load_exceeds_698p6 = 683
- fraction_load_exceeds_698p6 = 0.007341165
- voyages_with_exceedance = `voyage_063`

If load exceeds `698.6 kW`, the benchmark must not silently revert to 350 kW. Any resulting infeasibility should be attributed to insufficient physical power under the scaled 0.5C battery limit.
