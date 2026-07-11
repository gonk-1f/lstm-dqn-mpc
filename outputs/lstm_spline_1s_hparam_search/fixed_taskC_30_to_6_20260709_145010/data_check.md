# Spline 1s LSTM Data Check

- Data source: `C:\Users\20883\OneDrive\Desktop\microgrid-mpc-master\outputs\spline_1s_diagnostics\data\natural_clipped_by_voyage`
- Data label: natural clipped cubic-spline reconstructed 1s load profile
- Caveat: The data are offline natural-boundary cubic-spline reconstructions from original 30 s vessel load voyages with nonnegative clipping. They are not native measured 1 s load data, not online prediction data, and intermediate 1 s points use future 30 s endpoint information.
- Window crossing voyage boundaries: false
- Scaler fit scope: train split only
- online_feasible=false
- uses_future_endpoint=true
- not_measured_1s=true

| split | voyages | rows | load_min | load_max | load_mean | load_std | has_nan | has_negative_load | duplicate_time_s |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
| train | 46 | 1235746 | 0.000000 | 869.363827 | 156.012519 | 193.472035 | False | False | False |
| validation | 13 | 355483 | 0.000000 | 935.165137 | 202.604285 | 220.871043 | False | False | False |
| test | 7 | 93037 | 0.000000 | 820.134823 | 208.291112 | 182.718332 | False | False | False |
