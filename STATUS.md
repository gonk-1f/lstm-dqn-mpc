# Project Status

## 当前主线

- 唯一 N=6 OSQP 离线滚动入口：`src/main/run_mpc_1s_n6_four_objective_sensitivity.py`。
- `dt=1 s`、`N=6`，使用同航段未来 `t+1..t+6` natural-clipped 样条点，每次只执行第一步。
- 该路径是 offline oracle/ideal foresight，不接入 LSTM 或 DQN。
- 唯一固定配置 candidate_C：`q_h2=0.25`、`q_batt=0.4`、`q_soc=12.0`、`q_fc_var=20.0`。
- 电池等效容量 `624 kWh`，充电/放电边界 `-624/+1248 kW`，电池功率归一化参考 `624 kW`；正功率表示放电。
- SOC 初始值/参考值 `0.55`，硬边界 `[0.2, 0.8]`，归一化带宽 `0.05`。
- 旧 17 组 one-factor、candidate_A/B、N=60 结果和旧权重实验入口均已清理；不得擅自恢复权重搜索。
- DQN 尚未开始正式实现或训练；后续动作是选择完整四权重组合，不直接输出功率。

## 保留数据

- 30 s 原始实船数据：`total_load_excels/`。
- 1 s natural-clipped 数据：`outputs/spline_1s_diagnostics/data/natural_clipped_by_voyage/`。
- 当前 N=6 输入及其来源证据：`outputs/mpc_solver_benchmark_1s/data/`。
- 1 ms 台架数据：`data/millisecond_1ms/`。
- 10 ms 抽点数据：`data/millisecond_10ms/`。
- 活动 30 s/1 s 航段划分：`outputs/config/voyage_split_total_load_721.json`，为 46/13/7；测试集是 `voyage_060` 至 `voyage_066`。10 ms 独立划分仍为 `outputs/config/millisecond_10ms_split_721.json`。
- 当前正式数据版本：`device_channel_natural_spline_1s`。
- MPC 氢耗曲线：`data/fuel_cell/FC_Dp0_curve_for_Python.csv`。

## candidate_C 证据

- 输出目录：`outputs/mpc_1s_n6_candidate_C/`。
- 9 个非空文件：正式配置、指标和 `voyage_060` 至 `voyage_066` 七张图；不再保留旧诊断副本。
- 正式指标恰好覆盖 7 个测试航段，7/7 均完成；求解失败、primal infeasible 和 maximum-iterations 事件均为 0。
- 配置保留生成时的源码、活动划分和输入 SHA-256；本次运行层清理不重新生成或改写结果。

## voyage_060 当前口径

旧 BDM 掉零口径不再作为正式排除依据；当前按 12 簇电池和 8 路燃料电池功率重构，`voyage_060` 属于正式测试集，测试集完整范围为 `voyage_060` 至 `voyage_066`。

## 已知未决问题

- 仓库不包含原始采集设备、字段字典和上游导出链，无法区分采集/同步/导出异常、字段定义不完整或运行模式切换。
- 部分保留 manifest 仍含旧工作区绝对路径；本轮遵守“不修数据”边界，未改写这些元数据。
- 1 s 样条使用未来 30 s 端点，不能作为在线实测或因果预测证据。
