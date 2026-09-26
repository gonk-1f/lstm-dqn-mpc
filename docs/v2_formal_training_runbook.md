# v2 正式 DQN 训练说明

## 冻结合同

- 状态：ONBOARD-only S8，含当前因果 AIS 航速，不含 `shore_connected`；
- 动作：完整、固定顺序的 36 个正十分位三权重；
- 时间尺度：`Ts=30 s`、`N=5`、`M=5`、`tau_LPF=90 s`；
- 数据：30 Train、8 Validation、5 Test；Train 为 23,590 个 supervisory steps、
  18,448 个 ONBOARD steps 和 3,721 个 DQN macro transitions；
- 默认：40 轮、seed 42、epsilon 按 global macro step 从 1.0 线性降至 0.05，
  衰减长度 150,000；40 轮共 148,840 个 macro steps，末端 epsilon 约 0.05735；
- 正常 reward：已执行 interval ledger 的负原始 CNY 总成本，不使用 MPC objective；
- 物理不可行：当前 episode 终止并写入 replay，学习 reward 为
  `-raw_economic_cost_cny-50000`；50000 是独立训练分数，不进入 CNY ledger；
- discount：`gamma=1.0`，对应有限 episode 的未折扣总人民币成本。

岸电由独立 mode sidecar 硬联锁。分类使用质量有效的 AIS 近零航速和 12 BMS
充电证据；8 FC 原始功率保留为诊断但不作为岸电 gate。负电池功率或零航速都
不能单独判岸电。
`SHORE_PENDING` 与 `SHORE_CHARGING` 强制 FC=0，并按仿真 SOC 可接受的电池侧
充电功率更新 SOC、岸电成本与电池退化。岸电物理时间不增加 MPC/DQN/global/
epsilon/replay/gradient 计数，费用归入前一个动作的 transition。

## PyCharm PowerShell 命令

先确认终端当前目录为仓库根目录，然后运行：

```powershell
$env:PYTHONPATH=(Resolve-Path "src").Path
python -X utf8 -u -m v2.main.train_formal_dqn --preflight-only
python -X utf8 -u -m v2.main.train_formal_dqn --smoke-only
```

当前两项均已通过。正式训练命令为：

```powershell
python -X utf8 -u -m v2.main.train_formal_dqn --rounds 40 --seed 42 --device cpu --log-every 1 --output-dir outputs/v2_formal_dqn_v3
```

本次必须显式使用新目录 `outputs/v2_formal_dqn_v3`。`--preflight-only` 和
`--smoke-only` 都不会开始训练；smoke 不写正式 checkpoint。

中断后恢复：

```powershell
$env:PYTHONPATH=(Resolve-Path "src").Path
python -X utf8 -u -m v2.main.train_formal_dqn --rounds 40 --seed 42 --device cpu --log-every 1 --output-dir outputs/v2_formal_dqn_v3 --resume outputs/v2_formal_dqn_v3/latest.pt
```

训练量对比：20/30/40/50 轮分别对应 74,420/111,630/148,840/186,050
个 macro steps；相应末端 epsilon 约为 0.529/0.293/0.057/0.050。40 轮能基本走完
既定 150,000 步探索衰减，同时避免 50 轮中约 36,050 步长期停留在 epsilon 下限。

完成 40 轮后如需延长到 50 轮，只提高 `--rounds`：

```powershell
$env:PYTHONPATH=(Resolve-Path "src").Path
python -X utf8 -u -m v2.main.train_formal_dqn --rounds 50 --seed 42 --device cpu --log-every 1 --output-dir outputs/v2_formal_dqn_v3 --resume outputs/v2_formal_dqn_v3/latest.pt
```

检查点绑定 S8 schema digest、36-action catalog digest、控制语义、网络/优化器、
replay、所有 RNG、当前随机排列、episode 位置和 global macro step。不能降低恢复时
的总轮数，也不能载入旧 S7/S10、84-action artifact 或失败语义变更前的
`v2_formal_dqn_checkpoint_v2`。旧 `outputs/v2_formal_dqn/latest.pt` 仅保留作诊断，
不得用于本次正式训练恢复。

每个 macro 日志分别输出 `raw_economic_cost_cny`、
`failure_penalty_score`、`learning_reward`、`episode_completed` 和
`failure_kind`。确定性的 `PhysicalInfeasibilityError` 是可学习的 terminal outcome；
数值求解错误、非有限值和程序错误仍立即中止，不能伪装成可学习失败。

Validation 每轮按 manifest 固定顺序运行纯贪婪评估，使用同一失败评分并报告
完成率，但不 shuffle、不写 replay、不更新优化器。Test payload 不会由此训练命令
打开；Test 留待训练和模型选择结束后的一次最终评估。
