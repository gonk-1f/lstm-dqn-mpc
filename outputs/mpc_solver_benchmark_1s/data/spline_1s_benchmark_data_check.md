# 1 s Spline Benchmark Data Check

Scope: offline solver benchmark input only. These rows are natural cubic-spline reconstructions with negative load clipped to zero; they are not measured 1 s data.

- Source directory: `C:\Users\20883\OneDrive\Desktop\lstm-dqn-mpc\lstm-dqn-mpc\outputs\spline_1s_diagnostics\data\natural_clipped_by_voyage`
- Output parquet: `C:\Users\20883\OneDrive\Desktop\lstm-dqn-mpc\lstm-dqn-mpc\outputs\mpc_solver_benchmark_1s\data\test_voyages_spline_1s.parquet`
- Split JSON: `C:\Users\20883\OneDrive\Desktop\lstm-dqn-mpc\lstm-dqn-mpc\outputs\mpc_solver_benchmark_1s\data\voyage_split_spline_1s_total_load_721.json`
- Inferred sample interval: `1.0 s`
- Source voyage split count summary: `{'train': 46, 'validation': 13, 'test': 7}`
- `online_feasible`: `false`
- `uses_future_endpoint`: `true`
- `not_measured_1s`: `true`

## Test Voyage Checks

| voyage_id | rows | median_dt_s | duplicate_time_s | nan_core_values | negative_load_rows | min_load_total_kw | max_load_total_kw | original_30s_rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| voyage_060 | 10651 | 1.000 | 0 | 0 | 0 | 0.000000 | 767.347110 | 0 |
| voyage_061 | 10771 | 1.000 | 0 | 0 | 0 | 0.000000 | 368.037528 | 0 |
| voyage_062 | 10680 | 1.000 | 0 | 0 | 0 | 0.000000 | 720.117216 | 0 |
| voyage_063 | 32371 | 1.000 | 0 | 0 | 0 | 0.000000 | 909.147355 | 0 |
| voyage_064 | 7170 | 1.000 | 0 | 0 | 0 | 0.000000 | 500.380071 | 0 |
| voyage_065 | 10771 | 1.000 | 0 | 0 | 0 | 0.000000 | 694.858466 | 0 |
| voyage_066 | 10771 | 1.000 | 0 | 0 | 0 | 0.000000 | 612.245221 | 0 |
