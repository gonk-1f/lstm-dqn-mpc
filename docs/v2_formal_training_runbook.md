# v2 正式 DQN 训练说明

## 冻结合同

- 状态：ONBOARD-only S8，含当前因果 AIS 航速，不含 `shore_connected`；
- 动作：完整、固定顺序的 36 个正十分位三权重；
- 时间尺度：`Ts=30 s`、`N=5`、`M=5`、`tau_LPF=90 s`；
- 数据：38 Train、10 Validation、5 Test；当前诊断为 5,652 个 ONBOARD macro
  候选，但 unresolved mode 审核后必须重新确认；
- 默认：30 轮、seed 42、epsilon 按 global macro step 从 1.0 线性降至 0.05，
  衰减长度 150,000；
- reward：已执行 interval ledger 的负原始 CNY 总成本，不使用 MPC objective；
- discount：`gamma=1.0`，对应有限 episode 的未折扣总人民币成本。

岸电由独立 mode sidecar 硬联锁。sidecar 联合当前/历史 AIS、8 FC、12 BMS、
freshness 和 conflict 证据；负电池功率、负总功率或零航速都不能单独判岸电。
`SHORE_PENDING` 与 `SHORE_CHARGING` 强制 FC=0，并按仿真 SOC 可接受的电池侧
充电功率更新 SOC、岸电成本与电池退化。岸电物理时间不增加 MPC/DQN/global/
epsilon/replay/gradient 计数，费用归入前一个动作的 transition。

## PyCharm PowerShell 命令

先确认终端当前目录为仓库根目录，然后运行：

```powershell
$env:PYTHONPATH="src"
python -m v2.main.train_formal_dqn --preflight-only
python -m v2.main.train_formal_dqn --smoke-only
```

当前 `--preflight-only` 返回 `NO-GO`（Train unresolved=685，Validation
unresolved=230），因此不得运行正式训练。默认输出目录是
`outputs/v2_formal_dqn`。`--preflight-only` 和
`--smoke-only` 都不会开始训练；smoke 不写正式 checkpoint。

中断后恢复：

```powershell
$env:PYTHONPATH="src"
python -m v2.main.train_formal_dqn --rounds 30 --seed 42 --device cpu --log-every 1 --resume outputs/v2_formal_dqn/latest.pt
```

完成 30 轮后如需延长到 40 轮，只提高 `--rounds`：

```powershell
$env:PYTHONPATH="src"
python -m v2.main.train_formal_dqn --rounds 40 --seed 42 --device cpu --log-every 1 --resume outputs/v2_formal_dqn/latest.pt
```

检查点绑定 S8 schema digest、36-action catalog digest、控制语义、网络/优化器、
replay、所有 RNG、当前随机排列、episode 位置和 global macro step。不能降低恢复时
的总轮数，也不能载入旧 S7/S10 或 84-action artifact。

Validation 每轮按 manifest 固定顺序运行纯贪婪评估，不 shuffle、不写 replay、
不更新优化器。Test payload 不会由此训练命令打开；Test 留待训练和模型选择结束后的
一次最终评估。
