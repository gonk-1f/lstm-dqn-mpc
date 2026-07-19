# Project Status

## 当前主线

- 唯一 N=6 OSQP 离线滚动入口：`src/main/run_mpc_1s_n6_four_objective_sensitivity.py`。
- `dt=1 s`、`N=6`，使用同航段未来 `t+1..t+6` natural-clipped 样条点，每次只执行第一步。
- 该路径是 offline oracle/ideal foresight，不接入 LSTM 或 DQN。
- 唯一固定配置 candidate_C：`q_h2=0.25`、`q_batt=0.4`、`q_soc=12.0`、`q_fc_var=20.0`。
- 旧 17 组 one-factor、candidate_A/B、N=60 结果和旧权重实验入口均已清理；不得擅自恢复权重搜索。

## 保留数据

- 30 s 原始实船数据：`total_load_excels/`。
- 1 s natural-clipped 数据：`outputs/spline_1s_diagnostics/data/natural_clipped_by_voyage/`。
- 当前 N=6 输入及其来源证据：`outputs/mpc_solver_benchmark_1s/data/`。
- 1 ms 台架数据：`data/millisecond_1ms/`。
- 10 ms 抽点数据：`data/millisecond_10ms/`。
- 活动 30 s/1 s 航段划分：`outputs/config/voyage_split_total_load_721.json`；排除 16 个原始矛盾航段后为 35/10/5。10 ms 独立划分仍为 `outputs/config/millisecond_10ms_split_721.json`。
- MPC 氢耗曲线：`data/fuel_cell/FC_Dp0_curve_for_Python.csv`。

## candidate_C 证据

- 输出目录：`outputs/mpc_1s_n6_candidate_C/`。
- 7 个非空文件：正式配置、指标和 `voyage_061, 063, 064, 065, 066` 五张图；不再保留旧诊断副本。
- 正式指标恰好覆盖 5 个新测试航段，5/5 均完成；求解失败、primal infeasible 和 maximum-iterations 事件均为 0。
- 配置记录的源码、活动划分和输入 SHA-256 均与当前文件一致。
- 本轮按要求精简了 runner，故旧配置保存的 implementation SHA-256 与当前源码不同。旧结果保留其原生成版本，不覆盖、不伪造复用一致性。

## voyage_060 数据边界

`voyage_060` 在原始 30 s Excel 的 `4410–4740 s` 已存在动力源合计零值，但同期推进逆变器功率非零且 SOC 持续下降。natural spline 与非负裁剪又扩大了零值覆盖。该航段已与另外 15 个同类矛盾航段一起整段排除；完整证据见 `docs/VOYAGE_DATA_QUALITY_AUDIT.md`。

## 已知未决问题

- 仓库不包含原始采集设备、字段字典和上游导出链，无法区分采集/同步/导出异常、字段定义不完整或运行模式切换。
- 部分保留 manifest 仍含旧工作区绝对路径；本轮遵守“不修数据”边界，未改写这些元数据。
- 1 s 样条使用未来 30 s 端点，不能作为在线实测或因果预测证据。
