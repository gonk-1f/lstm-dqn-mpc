# v2 正式 DQN 训练说明

## 冻结合同

- 状态：ONBOARD-only 10 帧因果 S8 历史，按最旧到最新排列，左侧零填充并附
  10 位有效掩码，共 90 维；每帧含当前因果 AIS 航速，不含 `shore_connected`；
- 动作：完整、固定顺序的 36 个正十分位三权重；
- 时间尺度：`Ts=30 s`、`N=5`、`M=5`、`tau_LPF=180 s`；
- LPF 更新合同：solver 只做无副作用 preview，每个成功执行的 ONBOARD 30 s
  interval 由 backend 提交一次，版本为
  `causal_single_commit_per_executed_interval_v1`；
- 数据：30 Train、8 Validation、5 Test；Train 为 23,590 个 supervisory steps、
  18,448 个 ONBOARD steps 和 3,721 个 DQN macro transitions；
- 历史状态 pilot 默认：10 轮、seed 42；后续仅续训 Validation 排名前两名至
  40 轮。epsilon 按 global macro step 从 1.0 线性降至 0.05，
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

## 当前正式基线：H4 + 180 s LPF

先在 PyCharm PowerShell 的仓库根目录设置环境并确认 preflight/smoke：

```powershell
$env:PYTHONPATH=(Resolve-Path "src").Path
$env:OPENBLAS_NUM_THREADS="1"
$env:OMP_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"

python -X utf8 -u -m v2.main.train_history_dqn_study --preflight-only --experiment H4 --reward-scale outputs/v2_history_dqn_study/reward_scale_calibration.json
python -X utf8 -u -m v2.main.train_history_dqn_study --smoke-only --experiment H4 --reward-scale outputs/v2_history_dqn_study/reward_scale_calibration.json
```

Train-only reward scale 已在单次提交修复后重新写入
`outputs/v2_history_dqn_study/reward_scale_calibration.json`。它使用真实 180 s LPF、
固定动作 `w_8_1_1` 的 23,408 个实际执行 30 s interval ledger（包含零成本
interval），在氢价 21.9 CNY/kg、FC replacement fraction 0.5 和 OFF→ON
启动计数修正后，算术均值为 `4.42374585960071 CNY`。50,000 失败分数不进入该均值；
scaled profile 会把完整学习 reward（原始成本和失败分数）统一除以该尺度。

正式新基线采用 H4：scaled reward、learning rate `1e-3`、90 维状态和 36 动作。
必须从头训练，禁止恢复重复提交条件下的任何 H4 checkpoint；checkpoint control
identity 同时包含 `tau_lpf_seconds=180.0` 和单次提交版本。旧 `H4_tau180` 的训练、
Validation selection 与 Test 结果均不得作为修复后模型的正式证据。

```powershell
python -X utf8 -u -m v2.main.train_history_dqn_study --experiment H4 --reward-scale outputs/v2_history_dqn_study/reward_scale_calibration.json --rounds 40 --seed 42 --device cpu --log-every 50 --output-dir outputs/v2_m10_dqn_study/M5_control
```

每轮重新使用固定 seed 的 RNG 打乱 Train 航段；Validation 保持 manifest 顺序、
纯贪婪、不写 replay、不更新参数。日志明确区分 `epsilon`（探索率）和
`greedy_rate=1-epsilon`，并输出 `replay_reward`、TD、Q advantage/margin、裁剪前
梯度以及 behavior/Validation greedy 动作分布。

安全暂停后，必须从新目录自己的 `latest.pt` 恢复：

```powershell
python -X utf8 -u -m v2.main.train_history_dqn_study --experiment H4 --reward-scale outputs/v2_history_dqn_study/reward_scale_calibration.json --rounds 40 --seed 42 --device cpu --log-every 50 --output-dir outputs/v2_m10_dqn_study/M5_control --resume outputs/v2_m10_dqn_study/M5_control/latest.pt
```

本节不提供或执行 Test 命令。必须先完成 40 轮训练，再仅用 Validation 选择
round checkpoint；冻结后才允许进行一次 Test evaluation。

旧 H4 checkpoint 绑定旧 reward/economic semantics，不能作为新经济模型的
M=5 对照。

## M=10 动作保持时间消融（仅 Train + Validation）

M=10 保持 `Ts=30 s`、`N=5`、`gamma=1.0`，每个动作最多覆盖十次实际 MPC
solve，即 300 s。每轮 Train 的估计 macro transition 从 M=5 的 3,721 降为
1,881；40 轮分别约为 148,840 与 75,240。为匹配物理训练量，M=10 使用
warmup 2,500、epsilon decay 75,000、replay capacity 100,000。target sync 仍为
1,000 macro steps，属于本次“其余参数不变”的限制，因此对应物理时间是 M=5
的两倍。

M=10 的 Train-only reward scale 已独立重算并绑定 `M=10/300 s`：

```powershell
python -X utf8 -u -m v2.main.run_m10_reward_scale_calibration
python -X utf8 -u -m v2.main.train_m10_dqn_study --preflight-only
python -X utf8 -u -m v2.main.train_m10_dqn_study --smoke-only
python -X utf8 -u -m v2.main.train_m10_dqn_study --rounds 40 --seed 42 --device cpu --log-every 50
```

中断恢复：

```powershell
python -X utf8 -u -m v2.main.train_m10_dqn_study --rounds 40 --seed 42 --device cpu --log-every 50 --resume outputs/v2_m10_dqn_study/M10/latest.pt
```

M=5 control 与 M=10 都训练完后，下面的命令只读取 Train 身份和 Validation
payload，分别按“完成数最多、失败分数最小、经济成本最小、较早 round”选择
checkpoint，再输出两者及固定 `w_8_1_1` 的成本、动作分布、动作熵和最大动作
占比。它不读取 Test：

```powershell
python -X utf8 -u -m v2.main.compare_m5_m10_validation --m5-rounds 40 --m10-rounds 40 --device cpu
```

## 旧 40 轮单配置入口（保留兼容，不用于本次 H1-H4 比较）

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

当前检查点版本为 `v2_history_dqn_checkpoint_v1`，绑定 90 维历史 schema digest、
36-action catalog digest、experiment/reward-scale 身份、控制语义、网络/优化器、
replay、所有 RNG、当前随机排列、episode 位置和 global macro step。不能降低恢复时
的总轮数，也不能载入旧 8 维 S8、S7/S10、84-action artifact 或任何旧
`v2_formal_dqn_checkpoint_v*`。旧 `outputs/v2_formal_dqn/latest.pt` 和
`outputs/v2_formal_dqn_v3/latest.pt` 仅保留作历史诊断，不得用于本次恢复。

每个 macro 日志分别输出 `raw_economic_cost_cny`、
`failure_penalty_score`、`learning_reward`、`episode_completed` 和
`failure_kind`。确定性的 `PhysicalInfeasibilityError` 是可学习的 terminal outcome；
数值求解错误、非有限值和程序错误仍立即中止，不能伪装成可学习失败。

Validation 每轮按 manifest 固定顺序运行纯贪婪评估，使用同一失败评分并报告
完成率，但不 shuffle、不写 replay、不更新优化器。Test payload 不会由此训练命令
打开；Test 留待训练和模型选择结束后的一次最终评估。

## Validation checkpoint 选择

训练完成后，使用固定顺序的 8 个 Validation 航段对 40 个 round checkpoint
逐一进行纯贪婪评估。排序依次为：完成 episode 数最多、失败惩罚最小、原始经济
成本最小；完全相同时选择更早的 round。该过程不写 replay、不更新网络或优化器，
也不读取 Test payload：

```powershell
$env:PYTHONPATH=(Resolve-Path "src").Path
python -X utf8 -u -m v2.main.select_formal_dqn_checkpoint `
  --checkpoint-dir outputs/v2_formal_dqn_v3 `
  --output-dir outputs/v2_formal_dqn_selection `
  --device cpu
```

输出目录必须不存在。完成后生成 `validation_checkpoint_metrics.csv`、
`selection_manifest.json` 和逐字节复制的 `best_validation.pt`。manifest 绑定当前
power/AIS/mode manifest、90 维历史状态、36-action、控制/失败语义和所有候选 checkpoint
SHA-256。此步骤只完成 Validation 选择，不授权或执行最终 Test。

## 一次性最终 Test

只有完整且未篡改的 Validation selection bundle 才能签发最终 Test 授权。最终评估
使用完全相同、固定顺序的 5 个 Test 航段，依次运行所选贪婪 DQN 和固定动作
`w_8_1_1`。不进行探索、replay 写入或优化器更新。

该命令必须由用户显式执行一次：

```powershell
$env:PYTHONPATH=(Resolve-Path "src").Path
python -X utf8 -u -m v2.main.evaluate_formal_dqn_test `
  --selection-dir outputs/v2_formal_dqn_selection `
  --output-dir outputs/v2_formal_dqn_test `
  --device cpu `
  --confirm-final-test FINAL_TEST_ONCE
```

`outputs/v2_formal_dqn_test` 必须不存在。命令在读取 Test payload 前先创建
`TEST_ACCESS_STARTED.json`；成功后才把状态改为 `COMPLETE`。若运行中断或失败，
锁保持 `STARTED`，程序拒绝覆盖原目录，避免无意重复查看 Test。输出包括两策略
汇总、逐 episode 指标、动作分布和带 SHA-256/result digest 的 run manifest。
