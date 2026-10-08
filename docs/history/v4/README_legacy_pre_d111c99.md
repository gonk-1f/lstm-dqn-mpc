# 船舶燃料电池与锂电池能量管理研究

当前开发分支 `feat/direct-power-mlp-ddqn` 是 **30 s 直接功率 MLP Double-DQN**
实验。实现、参数、40 轮训练对照和未解决问题见
[v4 项目状态（2026-10-07）](v4_status_2026-10-07.md)。β=250 已完成
[固定配置 40 轮复核](v4_beta250_40r_review_2026-10-07.md)：最终 greedy Train
10/30、Validation 4/8，未保住 5 轮初筛的完成率提升；最终低 SOC 问题仍严重。
Test 未用于训练或选模，没有可选的正式模型。
清理范围及保留依赖见 [v4 清理清单](v4_cleanup_2026-10-07.md)。
`outputs/` 仅放本地生成结果，正式结论保存在 `docs/`。

## 历史 v2/v3 路线（下文为旧阶段记录）

An experimental [v3 persistence DQN–MPC path](../v3/v3_persistence_dqn_mpc.md)
uses five-step current-load persistence and a 30 s two-weight DQN cadence.
The earlier [LSTM diagnostic](../v3/v3_predictive_control.md) did not improve
Validation WAPE over persistence. The former formal baseline described below was v2.

v2 阶段的正式基线为 `METHOD_VERSION = multiscale_dqn_wmpc_v2`。下层非线性
MPC 每 30 s 滚动求解，预测步数 `N=5`；上层 DQN 的同一权重动作保持
`M=5` 个真实 supervisory steps。`N` 与 `M` 都对应 150 s，但语义独立。

正式 DQN 使用冻结的 ONBOARD-only S8 状态、完整 36 个正十分位三权重动作和
原始 CNY 宏区间奖励。S8 为 SOC、因果 base/residual/std/trend、FC power、
FC delta 和当前 AIS 航速。岸电不是交给 DQN 猜测：独立 mode sidecar 联合
8 个 FC、12 簇 BMS、AIS 与数据质量证据判定。`SHORE_PENDING` 和
`SHORE_CHARGING` 均暂停 DQN/MPC，但 SOC、岸电费用与电池退化继续更新并归入
前一个尚未闭合的 DQN transition。

当前冻结 manifest 为 30/8/5 个 Train/Validation/Test 航段。每轮 Train 有 23,590 个
30 s 物理 steps；按当前 mode sidecar 与 M=5 统计有 3,721 个 ONBOARD macro 候选。
默认训练设计为 30 轮。Train 每轮使用固定
seed 重新打乱，Validation 顺序固定且不写 replay、不更新网络；训练入口不打开
Test payload。

## 正式训练入口

在 PyCharm 的 PowerShell 终端、仓库根目录运行：

```powershell
$env:PYTHONPATH="src"
python -m v2.main.train_formal_dqn --preflight-only
python -m v2.main.train_formal_dqn --smoke-only
```

当前 `--preflight-only` 正确返回 `FORMAL_TRAINING=NO-GO`：Train 仍有 685 个、
Validation 仍有 230 个负总功率点缺少完整复合岸电证据。解决这些 mode evidence
前不得执行正式训练命令。`--smoke-only` 仅用于有界事件链联调。

恢复训练或把已完成的 30 轮检查点延长到更高轮数：

```powershell
$env:PYTHONPATH="src"
python -m v2.main.train_formal_dqn --rounds 40 --seed 42 --device cpu --log-every 1 --resume outputs/v2_formal_dqn/latest.pt
```

入口会先执行实时 dataset/state/solver preflight；`--smoke-only` 只做有界、
无梯度集成检查且不生成 `latest.pt`。正常训练逐 macro 刷新 episode、动作、
`epsilon`、`greedy_rate=1-epsilon`、原始 CNY reward、loss、SOC、功率和模式计数，
并原子保存可精确恢复的检查点。详见
[正式训练说明](../v2/v2_formal_training_runbook.md)。

## 当前 v2 边界

- [方法定义](../v2/method_v2_multiscale_dqn_wmpc.md)
- [正式训练规格](../v2/specs/2026-09-24-v2-formal-training-readiness-design.md)
- [数据来源](../../reference/data/v2_data_provenance.md)与[设备配置](../../reference/models/v2_plant_configuration.md)
- [FC 效率](../../v2_fc_efficiency_model.md)与[经济参数](../../reference/models/v2_economic_parameters.md)
- [FC 退化](../../reference/models/v2_fc_degradation_model.md)与[电池退化](../../reference/models/v2_battery_degradation_model.md)
- [动作合同](../v2/v2_action_space_design.md)与[状态审核](../../v2_dqn_state_audit.md)

配置/实现预检通过只表示具备开始训练的条件，不表示策略已经收敛或优于基线；
当前 mode evidence gate 尚未通过。
Validation/Test 不能反向选择 state、action、时间尺度或经济公式。

## v1 历史归档

v1 的 84 动作入口、checkpoint/replay 和 1 s 数据仅供历史追溯，位于
[`docs/archive_v1/`](../../archive_v1)，不得由正式 v2 入口加载。
