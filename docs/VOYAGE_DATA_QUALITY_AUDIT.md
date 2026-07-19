# Voyage data quality audit

## Decision

The audit covers all 66 original 30 s Excel voyages. Sixteen voyages are abnormal and are excluded as whole voyages:

`voyage_001`, `voyage_003`, `voyage_004`, `voyage_011`, `voyage_017`, `voyage_022`, `voyage_024`, `voyage_026`, `voyage_032`, `voyage_033`, `voyage_045`, `voyage_052`, `voyage_058`, `voyage_059`, `voyage_060`, and `voyage_062`.

Each excluded voyage contains an explicit raw-data contradiction: `fuel_cell_total_kw=0` and `battery_total_kw=0` while `propulsion_inverter_total_kw` remains between 33.6 and 93.7 kW. In the multirow intervals SOC falls while both recorded battery currents are zero. The remaining 50 voyages have no such raw source-load contradiction and are retained. Abrupt load changes, SOC variation by itself, negative load, and 1 s spline behavior were not used as exclusion grounds.

## Raw 30 s evidence

| Voyage | Abnormal range from voyage start (s) | Raw contradictory rows | Propulsion range (kW) | SOC evidence |
|---|---|---:|---:|---|
| voyage_001 | 7620–7860; 8220; 8310; 8550–8580; 8670–8700; 8790; 9240–9270 | 18 | 33.583–40.500 | SOC falls in every multirow run; main run 78.10%→75.80% |
| voyage_003 | 3660–3720 | 3 | 57.150–59.033 | 90.162%→89.462% |
| voyage_004 | 1860–1920 | 3 | 53.533–54.867 | 93.340%→92.690% |
| voyage_011 | 4020–4230 | 8 | 55.890–66.040 | 94.570%→91.910% |
| voyage_017 | 4740–4860; 4950–5070 | 10 | 46.233–52.967 | drops 1.133 and 1.100 percentage points |
| voyage_022 | 2670–2700 | 2 | 57.680–58.670 | 91.292%→91.030% |
| voyage_024 | 4530–4710 | 7 | 63.707–78.183 | 91.060%→89.310% |
| voyage_026 | 26310–26340 | 2 | 49.680–50.110 | 71.680%→71.357% |
| voyage_032 | 2790–2850 | 3 | 80.987–93.700 | 91.073%→90.327% |
| voyage_033 | 2220–3090 | 30 | 38.490–52.620 | 94.130%→86.120% |
| voyage_045 | 720–960 | 9 | 62.700–77.300 | 95.883%→93.620% |
| voyage_052 | 2670–2700 | 2 | 64.520–67.920 | 87.590%→87.270% |
| voyage_058 | 3420–3600; 3900–4170; 9450–9480 | 19 | 43.800–66.133 | three drops total 5.283 percentage points |
| voyage_059 | 4230; 4980–5610 | 23 | 61.733–89.260 | long run 86.193%→79.293% |
| voyage_060 | 4410–4740 | 12 | 51.073–74.987 | 91.120%→87.820% |
| voyage_062 | 4500–4890 | 14 | 57.290–83.440 | 76.320%→72.615% |

All 66 workbooks have the same 19-column structure. Across the audited timestamp, source-power, total-load, propulsion, battery-current, and SOC fields there are no missing values, duplicate timestamps, or non-30 s gaps. `total_load_fc_plus_batt_kw` equals `fuel_cell_total_kw + battery_total_kw` to a maximum absolute error of approximately `1.1e-6 kW`, well inside the existing `1e-3 kW` dataset identity tolerance. The zero load is therefore present in the source columns and is not introduced by the current load-sum code.

## 1 s natural-clipped audit

The retained 1 s files reproduce the current natural cubic-spline and `clip(lower=0)` implementation to floating-point precision. Natural-spline negative values occur in 65 of 66 voyages; 62 voyages have a negative interior value between two strictly positive 30 s endpoints. Thirty-four voyages with no zero raw node acquire zero 1 s samples after clipping. These interpolation artifacts are not used to exclude a voyage.

For all 16 abnormal voyages the spline/clip chain increases the number of zero-valued samples. It also extends a zero boundary beyond a raw abnormal interval for `voyage_001`, `voyage_003`, `voyage_004`, `voyage_011`, `voyage_024`, `voyage_032`, `voyage_045`, `voyage_059`, `voyage_060`, and `voyage_062`. The other six abnormal voyages gain zero samples without extending the adjacent time boundary.

Within the required `voyage_060` audit window of 4200–4900 s, the raw contradiction is at 4410–4740 s. The unclipped natural spline has 173 negative points with a minimum of `-37.6975 kW`; clipping creates 185 zero points and moves the first zero to 4385 s, 25 s before the first raw zero node. The 1 s transform therefore amplifies the observed zero behavior but does not originate the raw contradiction.

## Interpretation and limitation

The evidence supports a data-channel or acquisition-mode inconsistency, but it cannot distinguish among a collection fault, a field definition that omits another supplying source, or a mode switch. There is no evidence of timestamp field displacement or a current-code sum error. No raw value, spline value, or source file was repaired, filled, replaced, or deleted during this audit.
