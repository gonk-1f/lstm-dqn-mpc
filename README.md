# Multi-rate DQN adaptive nonlinear MPC (v2)

本仓库当前方法仅为 `METHOD_VERSION = multiscale_dqn_wmpc_v2`。下层非线性
MPC 每 `Ts_MPC` 秒滚动求解，上层 DQN 动作在 `M` 个已执行 MPC 周期内保持
不变；`N` 是预测步数，不能与 `M` 混用。DQN 奖励是宏区间真实经济成本，
不是旧版单步 MPC 自身目标值。

当前状态是 **`FORMAL_TRAINING = NO-GO`**。仓库没有正式训练入口，也不会用
占位值绕过缺失证据。主要未冻结项包括燃料电池与电池寿命归一化、真实 Train
工况数据、`Ts_MPC/N/M/tau_LPF/SOC deadband`、最终 DQN state、最终 action
catalog 与完整求解器审计。36 个十分位正单纯形权重只是候选库，不是正式动作表。

## 当前 v2 边界

- 精确版本、时间尺度语义：[方法定义](docs/method_v2_multiscale_dqn_wmpc.md)
- 数据与设备来源：[数据来源](docs/v2_data_provenance.md)、[设备配置](docs/v2_plant_configuration.md)
- 能量与经济参数：[FC 效率](docs/v2_fc_efficiency_model.md)、[经济参数](docs/v2_economic_parameters.md)
- 退化模型：[FC 退化](docs/v2_fc_degradation_model.md)、[电池退化](docs/v2_battery_degradation_model.md)
- 动作与时间尺度：[动作筛选](docs/v2_action_space_design.md)、[Train-only 审计](docs/v2_timescale_selection.md)
- 完整状态表：[v2 preflight 报告](docs/v2_preflight_report.md)

所有方法选择和标定只允许读取 Train。Validation/Test 只能在方法完全冻结后用于
评估，不得反向选择 `N`、`M`、`tau_LPF`、deadband、state、action catalog 或
reward scale。

## 安全入口

从仓库根目录运行当前预检：

```powershell
python -m src.v2.main.run_preflight
```

当前返回码为 `2`，表示预期的 NO-GO；返回 `0` 才表示全部 13 项均已冻结。此命令
不读取训练 payload，也不启动训练。

Train-only 时间尺度诊断的输入必须是已审计 Train 数值序列 JSON，并显式给出
预先登记的变化阈值：

```powershell
python -m src.v2.main.run_train_only_timescale_audit `
  --split Train `
  --train-json <audited-train-series.json> `
  --provenance-id <immutable-train-provenance-id> `
  --change-threshold <train-registered-threshold>
```

该入口只执行 `N=5`、`M in {5,10}` 的诊断，不选择正式参数、不访问
Validation/Test、不启动 DQN 训练。即使诊断完成，当前仍返回 `2`，因为正式时间
尺度状态保持 NO-GO。仓库当前没有满足 v2 原始数据门槛的真实 Train payload，
因此不要用合成数据结果声称完成了正式审计。

## v1 历史归档

v1 文档与输出仅用于历史追溯，入口、84 动作 checkpoint/replay 和 1 秒时序都不
属于当前正式方法。历史说明位于 [`docs/archive_v1/`](docs/archive_v1/)；不得把旧
`src/main/` 脚本描述或调用为 v2 正式训练入口，也不得将 v1 artifact 恢复到 v2。
