# Data Provenance

## Scope and evidence rule

本文件只记录能从仓库数据、manifest 和构建代码核对的事实。路径均为仓库相对路径；已有 JSON/报告中的旧本地绝对路径仅作为待迁移元数据，不在此复制。未知的采集设备、传感器标定、原始采集软件版本和外部拥有者信息标记为待核验。

## A. Original 30 s vessel voyages

### Source and sampling

- 项目内原始目录：`total_load_excels/`。
- 文件数量：66 个 `.xlsx`，每个文件映射为一个 `voyage_id`。
- 构建入口：`src/main/build_total_load_dataset_721.py`。
- 读取方式：解析每个工作簿的首个工作表，按时间戳排序并检查必需列。
- 采样间隔：代码常量 `DT_SECONDS = 30.0`；抽查工作簿时间戳也按 30 s 递增。
- 原始采集设备、校准过程和工作簿导出链：**待核验**。

### Fields

抽查的工作表包含以下 19 列：

```text
timestamp
time_h
fuel_cell_left_kw
fuel_cell_right_kw
fuel_cell_total_kw
battery_power_left_kw
battery_power_right_kw
battery_total_kw
total_load_fc_plus_batt_kw
battery_voltage_left_v
battery_current_left_a
battery_voltage_right_v
battery_current_right_a
soc_left_pct
soc_right_pct
soc_mean_pct
propulsion_inverter_left_kw
propulsion_inverter_right_kw
propulsion_inverter_total_kw
```

正式总负荷字段是 `total_load_fc_plus_batt_kw`，定义为：

```text
fuel_cell_total_kw + battery_total_kw
```

构建脚本将其标记为 `energy_side_equivalent_total_load` 并逐行核对恒等式。该字段不是推进电机单独功率，也不应被描述为包含所有船载辅助负载的已标定母线功率，除非后续获得额外数据说明。

QP 主线的符号约定是电池正功率放电、负功率充电，满足 `P_fc + P_batt = load`；SOC 动力学因此在正电池功率时下降。原始工作簿的电池符号与该约定是否在所有航次完全一致，仍应在正式数据审计中用电流/SOC 联合确认。

### Build artifacts

- 原始数据保持在 `total_load_excels/`；重复合并 CSV/Parquet 和旧航次摘要已在仓库精简中删除，未重新生成。
- 权威划分：`outputs/config/voyage_split_total_load_721.json`。

### 66-voyage list and split

划分依据是按航次开始时间排序，不使用模型或 test 指标。完整的 `voyage_id -> 原始文件名` 映射保存在权威 JSON 中；文件名含中文日期，在不同终端编码下可能显示不同，因此此处列出稳定 ID 和 split，不手工复制易损文件名。

| Split | Voyage IDs | Count |
| --- | --- | ---: |
| train | `voyage_002`, `voyage_005`, `voyage_006`, `voyage_007`, `voyage_008`, `voyage_009`, `voyage_010`, `voyage_012`, `voyage_013`, `voyage_014`, `voyage_015`, `voyage_016`, `voyage_018`, `voyage_019`, `voyage_020`, `voyage_021`, `voyage_023`, `voyage_025`, `voyage_027`, `voyage_028`, `voyage_029`, `voyage_030`, `voyage_031`, `voyage_034`, `voyage_035`, `voyage_036`, `voyage_037`, `voyage_038`, `voyage_039`, `voyage_040`, `voyage_041`, `voyage_042`, `voyage_043`, `voyage_044`, `voyage_046` | 35 |
| validation | `voyage_047`, `voyage_048`, `voyage_049`, `voyage_050`, `voyage_051`, `voyage_053`, `voyage_054`, `voyage_055`, `voyage_056`, `voyage_057` | 10 |
| test | `voyage_061`, `voyage_063`, `voyage_064`, `voyage_065`, `voyage_066` | 5 |

排除航段为 `voyage_001, 003, 004, 011, 017, 022, 024, 026, 032, 033, 045, 052, 058, 059, 060, 062`。原始文件和稳定 ID 均保留；排除证据见 `docs/VOYAGE_DATA_QUALITY_AUDIT.md`。划分仍为按开始时间排序的 7:2:1，50 个正常航段按 floor(70%)/floor(20%)/余数得到 35/10/5，不使用随机种子。

旧 30 s LSTM 训练入口已从 `src/main` 清理；历史输出未在本轮越权删除，也不属于当前固定 MPC 的运行依赖。

## B. Offline 1 s cubic-spline reconstruction

### Construction

- 输入：上述 66 个 30 s 航段，逐航次独立处理。
- 生成入口：`src/main/build_spline_1s_diagnostics.py`。
- 插值：SciPy `CubicSpline`，同时诊断 `bc_type="natural"` 和默认 not-a-knot；后续 retained 输入使用 natural。
- 时间网格：每 1 s 一个点。
- 裁剪：natural 结果做 `max(value, 0)` 非负裁剪，形成 natural-clipped 数据。
- 边界：不跨航次拟合。保存的每航段 1 s CSV 未改写传感器或样条值；当前 benchmark builder 以权威活动 JSON 在内存覆盖其历史内嵌 split 并排除异常航段。
- 关键标记：构建输出明确记录 `online_feasible=false`、`uses_future_endpoint=true`。

### Storage and consumers

- 逐航次 66 个 natural-clipped CSV：`outputs/spline_1s_diagnostics/data/natural_clipped_by_voyage/`。
- 逐航次 manifest：同目录下 `manifest.csv`。
- 重复合并文件、旧图、模型和诊断汇总已清理；逐航次 natural-clipped 文件及 manifest 保留。
- OSQP benchmark 数据构建：`src/main/build_mpc_solver_benchmark_1s_data.py`。
- benchmark 输入：`outputs/mpc_solver_benchmark_1s/data/test_voyages_spline_1s.parquet`，包含 5 个 test 航次、71,735 行。

### N=6 four-objective offline-oracle experiment

- 唯一入口：`src/main/run_mpc_1s_n6_four_objective_sensitivity.py`；唯一 focused test：`tests/test_mpc_1s_n6_four_objective_sensitivity.py`。
- 固定输入：`outputs/mpc_solver_benchmark_1s/data/test_voyages_spline_1s.parquet`。正式运行从权威活动 JSON 读取 `voyage_061, 063, 064, 065, 066`，并强制检查 `split=test`、数据版本和逐航段严格 1 s 间隔。
- 时刻 `t` 使用同航段的真实重构点 `t+1..t+6`；航段尾部只重复本航段末点，不跨航段。该信息在决策时并非因果可得，因此实验标签必须是 offline oracle/ideal foresight。
- MPC 预测/控制时域均为 6，但每次只执行第一步；实际电池功率由 `load_actual(t+1)-P_fc(t+1)` 得到，SOC 用该实际功率更新。
- 四项目标是 `H2_norm`、`Batt_power_sq_norm`、`SOC_tracking_sq_norm` 和 `FC_variation_sq_norm`。归一化参考分别为 `m_H2(560 kW, 1 s)=0.00883945296644347 kg/step`、`346.5 kW`、`SOC_ref=0.55` 与 `SOC_band=0.05`、`48 kW/step`；FC variation 包含第一步相对 `P_fc_prev` 以及后续相邻步差。
- 唯一配置为 candidate_C：`q_h2=0.25`、`q_batt=0.4`、`q_soc=12.0`、`q_fc_var=20.0`。旧 17 组 one-factor 配置和 CLI 分支已删除。
- 兼容命令为 `python src/main/run_mpc_1s_n6_four_objective_sensitivity.py --baseline`；它不使用 LSTM 或 DQN，只执行优化第一步。
- 运行产物路径为 `outputs/mpc_1s_n6_candidate_C/`。正式结果覆盖新测试集五个航段，均完整且无 solver failure；配置记录源码、活动划分、输入哈希和实际航段列表。

### Causality boundary

每个 1 s 插值点由相邻 30 s 节点共同决定，因此在这 30 s 区间内使用了预测时刻尚不可得的未来端点。即使 train/validation/test 航次完全隔离，该信号本身仍有在线因果性问题。它可以用于：

- QP 形式与求解器速度测试；
- 离线控制器闭环机制搭建；
- 对插值伪高频信号的可预测性诊断。

它不能用于：

- 声称存在真实在线 1 s 实测负荷；
- 不加限定地证明 1 s LSTM 在线预测优势；
- 以未来 60 个重构点直接替代正式 LSTM 预测并称为完整预测闭环。
- 把 N=6 ideal-foresight 的未来 6 个真实重构点写成 LSTM 输出，或用该实验宣称在线因果可用性。

## C. Millisecond source and 10 ms auxiliary data

### Source and transformation

- 项目内原始副本：`data/millisecond_1ms/raw/`，两个 `.xlsx` 工作簿。
- source manifest：`data/millisecond_1ms/source_manifest.json`。
- 构建入口：`src/main/build_millisecond_10ms_dataset.py`。
- 变换：保留源序列第 0、10、20... 行，即从连续 1 ms 行直接 decimation 到 10 ms；不插值、不做抗混叠滤波。
- 构建器检查 1 ms 输入连续性和 10 ms 输出间隔；无滤波意味着高频混叠风险必须在解释结果时保留。
- 脚本默认 source 路径仍含个人桌面绝对路径，但 `--source`、`--raw-root`、`--processed-root` 可覆盖。默认值应在后续迁移为仓库相对路径/配置项。

### Dataset and split

- 数据：`data/millisecond_10ms/millisecond_load_10ms.csv`。
- 行数：32,000；`time_ms` 唯一的 10 ms 行来自 19 个原子序列（源表共 21 个，合并/筛选规则由构建器和 manifest 记录）。
- 实测 `load_kw` 范围：约 0.102–37.461 kW，均值约 14.916 kW，显著低于 560 kW FC/346.5 kW 电池的船舶主线尺度。
- 字段：`split`, `source_workbook`, `source_sheet`, `source_row_index`, `time_s`, `load_kw`, `fuel_cell_kw`, `battery_kw`, `bus_voltage_v`, `source_members`, `time_ms`, `sequence_id`。
- split manifest：`outputs/config/millisecond_10ms_split_721.json`，seed `20260710`。
- 行数：train/validation/test = 22,400/6,400/3,200。
- history 30、prediction 6 时的窗口数：22,050/6,225/3,060。
- 两个源工作簿都出现在三个 split 中，但分配的原子序列不同；manifest 保存各序列 SHA-256。
- split 的声明依据是行数匹配、每个 split 覆盖两个工作簿和负载分布相似性，不使用模型或 test 指标。

### 保留范围

保留 1 ms 原始数据、10 ms 抽点数据、构建/审计入口和活动 split JSON。旧 10 ms LSTM 搜索入口与 smoke 产物已删除；该数据不进入当前船舶 MPC 或 DQN 流程。

## Leakage and integrity audit

| Risk | Current control | Remaining issue |
| --- | --- | --- |
| 30 s 跨航次窗口 | split manifest 声明不跨航次，训练代码按 voyage 分组 | 应增加对所有正式入口的统一 invariant 测试 |
| 30 s scaler 泄露 | 只拟合 train 航次 | 保存的旧 checkpoint/run_config 含旧绝对路径，需迁移后重验 |
| 1 s 跨 split 泄露 | 逐航次拟合，审计后 35/10/5 航次隔离 | 信号在每个 30 s 区间使用未来端点，存在在线因果泄露 |
| 1 s benchmark 真实度 | 当前 N=6 四目标契约显式固定 offline oracle、`lstm_used=false`、`dqn_used=false`、第一步执行、可获得的 `t+1..t+6` 及航段尾部同航段末样本 edge-hold | 使用未来重构负荷作为 horizon，不能等同 LSTM 闭环或在线控制；candidate_C 结果只提供离线控制证据 |
| 10 ms 跨序列窗口 | 按原子序列建窗并保存 hash | 同一工作簿的不同工况段跨 split，可能共享采集条件；需在论文中说明 |
| 10 ms scaler 泄露 | manifest 和模型均指定 train-only；audit 严格要求 train/validation/test key 集合 | 仍需在 CI 中持续执行完整数据审计 |
| 10 ms 抽点混叠 | 明确记录 direct decimation | 未做抗混叠滤波，不能把高频谱结论外推 |
| 文件路径可移植性 | 数据副本在仓库内 | 多个 manifest/run config 仍保存旧本地绝对路径 |

## Required provenance work before publication

1. 记录 30 s 数据的采集设备、通道标定、缺失/异常处理和原始文件 hash。
2. 用独立审计确认原始电池功率符号在全部 66 航次一致。
3. 为正式在线实验选择真实 1 s/更高频实测数据，或把 spline 路线严格限定为离线模拟。
4. 在 CI 中持续审计 10 ms manifest、assignment key、行数、hash、边界和 scaler 范围。
5. 将构建脚本和保留元数据中的本地绝对路径迁移为相对路径/显式配置，同时保留原始 hash 和 lineage。
