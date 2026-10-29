# 独立 Optuna 入口与运行规则

## 接入边界

入口是 `python -m v6_sb3.tuning`，直接使用固定版本 Optuna 的 ask/tell API。
RL Zoo 用于核对官方搜索机制和参数转换；运行时不调用 Zoo 的通用训练器、默认 reward 评价或 YAML 环境创建流程。
保留 `v6_sb3.training` 正式基线，继续复用 `MaskedDQN`、13 个功率增量动作、当前/后继合法动作 Mask、真实物理执行器及失败终止语义。

正式数据仅经 `V6TrainingDataset.load_train()` 与 `load_validation()` 读取，要求 35 条 Train 和 8 条 Validation。
Test 的文件清单可以存在于 split manifest，但搜索没有 Test payload 加载或选模路径。
每个 Trial 固定预算统计为实际 Adam 更新数；本适配固定每采集一步执行一次更新，因此正式 transition 数与更新数一致。
随机热身单独计数，不消耗正式更新预算。训练按原顺序循环 Train，预算结束可位于航段中部；Validation 始终执行固定完整航段，失败保留在结果中。

## 搜索空间

`src/v6_sb3/search.py` 定义项目搜索空间，Optuna 在 `trial.params` 记录采样值：

| 参数 | 搜索范围 |
| --- | --- |
| Adam 学习率 | `1e-5..3e-4`，对数采样 |
| batch size | `32 / 64 / 128` |
| Target 硬复制间隔 | `500 / 1000 / 2000` 次实际 Adam 更新 |
| Replay 容量 | `100000 / 300000 / 1000000` |
| 两层网络宽度 | `[128,128] / [256,256]` |

固定 `gamma=1`、一步 TD、uniform Replay、`train_freq=gradient_steps=1`、CPU 和一个环境。
沿用当前航段顺序和 epsilon 调度，不把 Zoo 的时间步探索调度直接套入。
固定模型 seed 与 sampler seed 均记录在配置中，默认都是 42；这不是多随机种子稳健性实验。
不搜索物理参数、失败惩罚、reward scale 或奖励模式。

## Validation 评价与剪枝

评价使用 Greedy policy，不插入 Replay、不执行 Adam。每次写出逐航段指标、汇总和失败详情。
完整集合的真实经济账本独立于训练奖励，两种奖励模式使用同一个评价口径。

对于固定的 `n` 条 Validation，记完成数 `k`、平均时长覆盖率 `c`、全部完成时的全集经济费用 `C`：

```text
未全部完成：score = k + 0.5*c
全部完成：score = n + 1.5 - atan(C/100000)/pi
```

完成数严格优先；未完成时比较覆盖率，不用失败前缀的低费用获利；全部完成后费用越低越好。
`atan` 只用于将经济偏好映射到 Optuna 的单个目标值，保留 signed SOC 费用，没有改训练奖励。
若所有候选均失败，导出的 best trial 只代表本次已观察候选中得分最高，不能据此声称具备完整航段能力。

Trial 返回已评价 checkpoint 中最高分，且记录对应 `best_checkpoint`。
每到 `--eval-every` 更新数和最终预算都评价并 `trial.report`；默认 MedianPruner 有 5 个 startup trials。
前 5 个 Trial 不会被该 pruner 剪枝，小预算可用 `--pruner none` 显式关闭。
不同奖励、固定预算或 Validation 身份的实验使用不同 study，不混合排名。

## 环境与短 Smoke Test

所有命令均从本 worktree 根目录运行，使用隔离解释器：

```powershell
Set-Location 'C:\Users\20883\OneDrive\Desktop\lstm-dqn-mpc\lstm-dqn-mpc\.worktrees\v6-sb3-baseline'
$env:PYTHONPATH = Join-Path (Get-Location).Path 'src'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:PYTHONUTF8 = '1'
& '.\.venv-optuna\Scripts\python.exe' -m pip check
& '.\.venv-optuna\Scripts\python.exe' -m v6_sb3.tuning --study-dir 'outputs/optuna_smoke_example' --smoke
```

`--smoke` 使用极短合成负载及正式物理执行器，默认 1 个 Trial、4 步热身、16 次正式更新、每 8 次更新评价。
CLI 硬上限为 2 个 Trial、每 Trial 64 次正式更新、32 步热身；合成运行不读取正式负载。
重复运行应换新目录；恢复则在原命令添加 `--resume`。
`--reward-mode zhou_economic_only` 可对独立合成目录验证纯经济通路，不代表已批准正式奖励替换。

## 审查后的真实数据计时命令

下面是单 Trial 的有限计时任务，交付过程未执行。它使用全部固定 Train/Validation 数据，但只做 1000 次正式更新；热身为 50000 步。
该命令可能仍耗时较长，因为完整 Validation 的长度不由正式更新预算缩短。

```powershell
& '.\.venv-optuna\Scripts\python.exe' -u -m v6_sb3.tuning --study-dir 'outputs/optuna_real_pilot_v6_seed42' --trials 1 --train-steps 1000 --learning-starts 50000 --eval-every 1000 --seed 42 --sampler-seed 42 --reward-mode v6_energy_soc_ramp --pruner none
```

正式搜索必须显式给 `--trials` 与 `--train-steps`；没有隐式启动完整训练的默认预算。
审查后根据计时和科学问题决定总 Trial 数、更新数、评价间隔及奖励方案，使用新目录启动。
选择纯经济方案时仅切换显式 reward mode，并重新建立实验目录；冻结 V6 的模型和 Replay 不覆盖、不跨奖励载入。

## 暂停、恢复与持久化

在前台按一次 Ctrl+C 请求安全暂停。程序完成当前 transition/Adam 边界后写模型、Replay、物理动作前缀、训练进度和 RNG 状态，然后退出当前 Trial。
恢复使用原命令和相同参数，加 `--resume`。恢复会先处理暂停时尚未执行的边界 Validation，包括最后一次更新后的 Validation。
不得在写 checkpoint 尚未完成时关机；以终端显示 `paused safely` 为准。
强制杀进程、断电及任意写盘瞬间的崩溃不承诺自动恢复；缺少完成 checkpoint 或 sampler 时明确失败，避免静默重建搜索序列。

SQLite study 保存 Trial 参数、中间值及状态；sampler 状态另行保存，`active_trial.json` 指向暂停中的 Trial。
总 Trial 预算计算所有已分配 Trial，包括失败、剪枝和正在运行的 Trial。孤立 RUNNING 状态会拒绝继续分配。
study 采用单进程文件锁。恢复校验当前源码、依赖、Train/Validation 文件摘要、经济身份、奖励、预算和种子。
源码变化也会拒绝恢复；如需改变实现，应建立新实验，而非继续旧 study。

每个 Trial 的 `configuration.json`、`result.json`、`performance.json` 和 checkpoint 保存在独立子目录。
`trials.csv` 用于同一 study 比较；`selected_trial.json` 记录 Validation 选出的 Trial/参数/checkpoint。
完整试验目录保留在忽略的 `outputs/` 中，交付只收录微型运行的报告证据，不收录新的正式训练产物。

## 性能计时与预算估计

`performance.json` 使用墙钟时间，分别记录 environment step、policy prediction、Replay 写入/采样、网络 forward、Adam step、整个训练更新、Validation、报告和 checkpoint。
`adam_step` 仅是优化器 step；backward、梯度裁剪等包含在 `training_update_total`。
`environment_step` 包括真实物理执行与队列等待。嵌套 inclusive 时间不能直接相加；exclusive 时间会扣除同线程嵌套阶段。

从已获批的真实计时 Trial 估算同机、同数据范围、同网络配置的预算：

```powershell
& '.\.venv-optuna\Scripts\python.exe' -m v6_sb3.budget --trial-dir 'outputs/optuna_real_pilot_v6_seed42/trial_0000' --formal-steps 100000 --eval-every 25000
```

估算由实测热身、每次正式更新的采集/学习总耗时、完整 Validation、报告和 checkpoint 组成。
缺少正式更新或至少一次完整评价时返回 `can_estimate=false`。
工具要求 `result.json` 为完成、正常暂停或已剪枝状态，用其中的实际 `adam_updates` 核对计时调用数；缺失、损坏或数量不一致时不估算。
Validation 文件数也必须与对应报告计时数一致，避免从旧 checkpoint 恢复后使用了较新的残留报告。
合成 Smoke 只能估算相同合成范围，不可换算成 35/8 条真实航段的训练时间。
估计覆盖已计时的采集、学习、评价和写盘部分；数据装载、模型构造、恢复重建与 study 汇总等额外开销需另用进程总墙钟计入。
不同网络/Replay 容量、失败频率、报告量和系统负载会改变耗时；总串行搜索预算还须按各 Trial 的配置和评价次数汇总。
