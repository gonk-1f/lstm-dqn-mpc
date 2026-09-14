# 基于设备通道 1 s 重构的总负荷审计

## 计算口径

- FC：左/右各 #1～#4 的 `发电功率(kW)`，8 路分别做 natural cubic spline 后求和。
- 电池：左/右各簇1～6，先在原始同一行按 `-(总电压(V)×总电流(A))/1000` 计算12路簇功率，再分别做 natural cubic spline 后求和。
- 每航段1 s时间轴是20路必需功率通道有效范围的共同交集；不外推、不跨航段、不 `fillna(0)`、不使用BDM回退。
- 原始内部断档允许由 natural cubic spline 连接；逐通道最大断档记录在审计CSV。
- `total_load_kw = fuel_cell_total_kw + battery_cluster_total_kw` 保留未裁剪恒等式；`load_total_kw` 与 `total_load_clipped_kw` 是现有 natural-clipped 非负建模列。
- BDM、逆变器和SOC只用于审计，不参与正式总负荷。
- 实际字段映射：`fuel_cell_left/right_1..4` 对应 `左/右氢燃料电池#1..#4_*.csv::发电功率(kW)`；`battery_left/right_cluster_1..6` 对应 `左/右电池簇1..6_*.csv::总电压(V), 总电流(A)`。

## 构建与旧异常结论

- 全部航段：66；成功重构：66；结构性排除：0。
- `resolved_by_cluster_reconstruction`：voyage_001, voyage_003, voyage_004, voyage_011, voyage_017, voyage_022, voyage_024, voyage_026, voyage_032, voyage_033, voyage_045, voyage_052, voyage_058, voyage_059, voyage_060, voyage_062。
- 旧16航段中恢复：voyage_001, voyage_003, voyage_004, voyage_011, voyage_017, voyage_022, voyage_024, voyage_026, voyage_032, voyage_033, voyage_045, voyage_052, voyage_058, voyage_059, voyage_060, voyage_062。
- `persistent_source_power_inconsistency`：voyage_026, voyage_028, voyage_029, voyage_041, voyage_050, voyage_052, voyage_061, voyage_066。该标签仅作审计，不超出任务列出的结构性排除条件。
- 可唯一消解的重复：可用航段去除完全相同的重复时间行 118044 条；排除航段仅为时间断档审计折叠 0 条；忽略SHA256完全相同的重复文件 20 份。未经显式审计勘误的冲突重复值不参与插值。
- 已审计定点时间勘误：voyage_054 共 20 路各1条；修正时刻分布：2024-07-09T09:56:48 × 4, 2024-07-09T09:56:49 × 16。只在构建时修正时间戳，功率值与原始CSV均未修改。
- natural spline 后未裁剪总负荷为负的行数：428854；仅 `load_total_kw`/`total_load_clipped_kw` 按既有规则裁剪为0，源侧恒等式列 `total_load_kw` 不裁剪。

### 排除航段

无。

## 各航段最大原始断档

| 航段 | 最大断档(s) | 通道 |
|---|---:|---|
| voyage_001 | 390.000 | battery_left_cluster_1 |
| voyage_002 | 360.000 | battery_left_cluster_1 |
| voyage_003 | 330.000 | battery_left_cluster_1 |
| voyage_004 | 390.000 | battery_left_cluster_1 |
| voyage_005 | 390.000 | battery_left_cluster_1 |
| voyage_006 | 361.000 | fuel_cell_left_1 |
| voyage_007 | 360.000 | battery_left_cluster_1 |
| voyage_008 | 360.000 | battery_left_cluster_1 |
| voyage_009 | 390.000 | battery_left_cluster_1 |
| voyage_010 | 360.000 | battery_left_cluster_1 |
| voyage_011 | 360.000 | battery_left_cluster_1 |
| voyage_012 | 390.000 | battery_left_cluster_1 |
| voyage_013 | 420.000 | battery_left_cluster_1 |
| voyage_014 | 390.000 | battery_left_cluster_1 |
| voyage_015 | 390.000 | battery_left_cluster_1 |
| voyage_016 | 360.000 | battery_left_cluster_1 |
| voyage_017 | 390.000 | battery_left_cluster_1 |
| voyage_018 | 31.000 | battery_left_cluster_1 |
| voyage_019 | 390.000 | battery_left_cluster_1 |
| voyage_020 | 391.000 | battery_left_cluster_1 |
| voyage_021 | 390.000 | battery_left_cluster_1 |
| voyage_022 | 390.000 | battery_left_cluster_1 |
| voyage_023 | 480.000 | battery_left_cluster_1 |
| voyage_024 | 450.000 | battery_left_cluster_1 |
| voyage_025 | 540.000 | battery_left_cluster_1 |
| voyage_026 | 450.000 | battery_right_cluster_1 |
| voyage_027 | 390.000 | battery_left_cluster_1 |
| voyage_028 | 420.000 | battery_left_cluster_1 |
| voyage_029 | 480.000 | battery_left_cluster_1 |
| voyage_030 | 420.000 | battery_left_cluster_1 |
| voyage_031 | 420.000 | battery_left_cluster_1 |
| voyage_032 | 390.000 | battery_left_cluster_1 |
| voyage_033 | 600.000 | battery_left_cluster_1 |
| voyage_034 | 420.000 | battery_left_cluster_1 |
| voyage_035 | 361.000 | battery_right_cluster_1 |
| voyage_036 | 390.000 | battery_left_cluster_1 |
| voyage_037 | 600.000 | battery_left_cluster_1 |
| voyage_038 | 420.000 | battery_left_cluster_1 |
| voyage_039 | 360.000 | battery_left_cluster_1 |
| voyage_040 | 420.000 | battery_left_cluster_1 |
| voyage_041 | 510.000 | battery_left_cluster_1 |
| voyage_042 | 420.000 | battery_left_cluster_1 |
| voyage_043 | 420.000 | battery_left_cluster_1 |
| voyage_044 | 510.000 | battery_left_cluster_1 |
| voyage_045 | 390.000 | battery_left_cluster_1 |
| voyage_046 | 420.000 | battery_left_cluster_1 |
| voyage_047 | 390.000 | battery_left_cluster_1 |
| voyage_048 | 390.000 | battery_left_cluster_1 |
| voyage_049 | 630.000 | battery_left_cluster_1 |
| voyage_050 | 450.000 | battery_left_cluster_1 |
| voyage_051 | 391.000 | fuel_cell_left_1 |
| voyage_052 | 450.000 | battery_left_cluster_1 |
| voyage_053 | 421.000 | battery_right_cluster_1 |
| voyage_054 | 360.000 | battery_left_cluster_1 |
| voyage_055 | 390.000 | battery_left_cluster_1 |
| voyage_056 | 390.000 | battery_left_cluster_1 |
| voyage_057 | 360.000 | battery_left_cluster_1 |
| voyage_058 | 420.000 | battery_left_cluster_1 |
| voyage_059 | 360.000 | battery_left_cluster_1 |
| voyage_060 | 390.000 | battery_left_cluster_1 |
| voyage_061 | 450.000 | battery_left_cluster_1 |
| voyage_062 | 510.000 | battery_left_cluster_1 |
| voyage_063 | 480.000 | battery_right_cluster_1 |
| voyage_064 | 420.000 | battery_left_cluster_1 |
| voyage_065 | 450.000 | battery_left_cluster_1 |
| voyage_066 | 420.000 | battery_left_cluster_1 |

## Chronological 7:2:1 划分

- train (46)：voyage_001, voyage_002, voyage_003, voyage_004, voyage_005, voyage_006, voyage_007, voyage_008, voyage_009, voyage_010, voyage_011, voyage_012, voyage_013, voyage_014, voyage_015, voyage_016, voyage_017, voyage_018, voyage_019, voyage_020, voyage_021, voyage_022, voyage_023, voyage_024, voyage_025, voyage_026, voyage_027, voyage_028, voyage_029, voyage_030, voyage_031, voyage_032, voyage_033, voyage_034, voyage_035, voyage_036, voyage_037, voyage_038, voyage_039, voyage_040, voyage_041, voyage_042, voyage_043, voyage_044, voyage_045, voyage_046
- validation (13)：voyage_047, voyage_048, voyage_049, voyage_050, voyage_051, voyage_052, voyage_053, voyage_054, voyage_055, voyage_056, voyage_057, voyage_058, voyage_059
- test (7)：voyage_060, voyage_061, voyage_062, voyage_063, voyage_064, voyage_065, voyage_066
- 三集合互斥；同一航段不拆分；所有成功重构航段只出现一次。

## 正式输出与限制

- 正式1 s：`outputs/spline_1s_diagnostics/data/natural_clipped_by_voyage`。
- 派生30 s汇总：`outputs/total_load_dataset_build/total_load_66_segments.csv`；由正式1 s每30点抽样，不是原始30 s实测总负荷。
- natural spline 会跨越记录断档，断档越长，区间内功率的不确定性越高；本任务只记录断档，不新增滤波或修复规则。
- 原始文件没有独立通道倍率、标定和拓扑说明；仅依据字段名称及V、A、kW单位计算。
