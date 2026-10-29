# V6：纯经济、单线程、500 episode 与共享 Optuna

起点 `b212483`，分支 `feat/v6-sb3-economic-episode500`。创建分支时 PR #1 尚未合并，因此本次 PR 以 `feat/v6-sb3-optuna-audit` 为目标。

**正式训练前必须先确认失败代价。** 新 CLI 保留旧 `10132.660087898294`，但默认拒绝真实数据训练/搜索，直到调用者显式给出 `--acknowledge-failure-penalty`。这只记录接受风险，不证明旧值已经校准。当前 Train 同 ID 的既有行为离线重评分中，122/125 组失败回报高于完成回报；旧参考与当前冻结 split 也不一致，详见 [Train-only 审计](failure_penalty_audit.md)。本交付没有选择任意新惩罚或增加 SOC 人为门控。

## 实现与语义

- `direct_environment.py` 的 `DirectVesselEnv` 每次 `step` 直接调用原物理/经济 primitives，保留 `history5`、13 个增量动作、真实可行集和末步失败语义。旧 Queue 环境和历史训练 CLI 仍可用。`v4.control` 仅提取共享奖励组合函数，未另造物理公式或改系数。
- 新 `episode_training.py` 是正式固定基线与 Optuna 共用的唯一 episode runner。默认纯经济奖励；历史 `v6_energy_soc_ramp` 为显式对照选项。gamma=1，不重复终端 SOC 结算。
- 500 个正式 episode 用独立 `random.Random(42)` 从固定 35 条 Train 有放回均匀抽样。每个都从 fresh SOC .6、FC off 和初始寿命开始，直到真实完成或失败。实际抽样顺序与次数保留。
- 50000 随机预热 transition 单独计数，使用独立 warmup RNG seed 43。正式每 32 条新 transition 执行一次实际 SB3 Adam，跨 episode 保留余数，不在 minibatch 边界截断航段；结束时不足 32 条不伪造额外更新。
- epsilon 按一基 episode 序号衰减：第 1 个为 1，第 400 个达到 .05，其后保持。Optuna 短预算采用同样的 80% 规则，不沿用八遍旧调度。
- Target 每 10000 个已入 Replay 的正式环境步硬复制，在该步的 Adam 前执行；warmup 不计入。同步记录同时含环境步与当时 Adam 数。新模式关闭旧 Adam-count 复制，历史模式默认行为不变。
- 默认每完成一个 episode 输出一行；Loss/TDMAE 只聚合真实 minibatch，没更新时为 null。每 50 episode 执行 Greedy Validation，以完成数、覆盖率、全完成费用选模，不向 Replay 插入经验。

## 暂停与保留

前台一次 Ctrl+C 请求安全暂停；当前真实 transition、可能到期的 Adam 及 episode 收尾先完成，再写 latest generation。也可用 `--stop-after-steps N` 对本次调用做可复现的安全暂停，这不是把航段标为 truncated。

恢复保存模型/优化器、Replay、物理状态、全局和专用 RNG、episode 抽样与在线指标，不通过重放旧动作恢复物理状态。恢复会检查数据、源码、依赖和配置身份；源码变化应另开实验目录。

默认保留独立 `best_model.zip` 与最近 2 个可恢复 generation（`--checkpoint-keep` 可改）。百万 Replay 只在 Validation、安全暂停、最终结束等必要点保存，不每 episode 重写。部分写入/未登记的 generation 会避让编号；这不等于承诺任意断电瞬间所有输出都可自动恢复。Optuna 支持正常 Ctrl+C，任意硬终止造成孤立 SQLite/marker 状态时会明确拒绝替代 Trial。

## Zoo、SB3、Optuna 的对应关系

SB3 提供 DQN policy/网络、uniform Replay 和 Adam 实现；项目 `MaskedDQN` 扩展合法动作 TD 和明确的 Target 时钟。Optuna 提供 TPE、SQLite study、参数记录、报告和剪枝。RL Baselines3 Zoo 是 SB3 上层的训练/配置/调参工具集，可参考其搜索机制与官方 DQN 参数转换。

本项目实际命令为 `python -m v6_sb3.tuning --episode-search`，直接调用 Optuna 并复用 `run_episode_training`，没有运行通用 `rl_zoo3.train`。因此不是完整 Zoo 默认 DQN 复现。

| Zoo 机制 | 本项目适配 |
| --- | --- |
| YAML 与默认环境构建 | 冻结 V6 dataset、真实 direct env 与显式 config |
| 官方 DQN sampler | 项目限定 LR、batch、Replay、宽度、Target 环境步间隔 |
| 通用步数预算/平均 reward | 统一较小完整 episode 预算、固定抽样 seed、分层 Validation 目标 |
| EvalCallback/TrialEvalCallback | 共享 runner 的 Greedy Validation 与 Optuna report/prune callback |
| best trial 参数 | 导出参数 JSON；新目录重新初始化模型/Replay，完整训练 500 episode |

搜索空间：LR 对数 `1e-5..3e-4`，batch `32/64/128`，Replay `100000/300000/1000000`，两层宽度 `128/256`，Target 间隔 `5000/10000/20000` 个正式环境步。固定 train_freq=32、gradient_steps=1、gamma=1、奖励、失败分数及物理参数。

## 运行命令

以下使用本机实际 worktree 和已存在的隔离解释器。所有 CLI 均已通过 help 或短时执行验证。正式命令的输出目录必须是新的；以下正式训练/真实数据 Trial/长搜索均未由交付过程自动执行。

1. 设置环境并检查依赖：

```powershell
Set-Location 'C:\Users\20883\OneDrive\Desktop\lstm-dqn-mpc\lstm-dqn-mpc\.worktrees\v6-sb3-baseline'
$env:PYTHONPATH = Join-Path (Get-Location).Path 'src'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:PYTHONUTF8 = '1'
$v6Python = Join-Path (Get-Location).Path '.venv-optuna\Scripts\python.exe'
& $v6Python -m pip check
```

2. 查看训练与新 Optuna CLI：

```powershell
& $v6Python -m v6_sb3.episode_training --help
& $v6Python -m v6_sb3.tuning --episode-search --help
```

3. 新环境双奖励 Smoke（合成负载，均走真实物理）：

```powershell
& $v6Python -u -m v6_sb3.episode_training --run-dir 'outputs/episode500_smoke_user' --smoke
& $v6Python -u -m v6_sb3.episode_training --run-dir 'outputs/episode500_smoke_history_user' --smoke --reward-mode v6_energy_soc_ramp
```

4. 确认失败代价风险后，正式固定参数 500 episode：

```powershell
& $v6Python -u -m v6_sb3.episode_training --run-dir 'outputs/v6_economic_fixed_seed42' --episodes 500 --learning-starts 50000 --train-freq 32 --gradient-steps 1 --learning-rate 1e-4 --batch-size 64 --buffer-size 1000000 --network-width 256 --target-update-interval 10000 --validation-every 50 --seed 42 --order-seed 42 --checkpoint-keep 2 --acknowledge-failure-penalty
```

5. 暂停时在正在运行的前台终端按一次 **Ctrl+C**，等 `paused` 摘要及 latest 写完再关机。恢复：

```powershell
& $v6Python -u -m v6_sb3.episode_training --run-dir 'outputs/v6_economic_fixed_seed42' --resume --acknowledge-failure-penalty
```

可复现的短暂停验证为 `--smoke --stop-after-steps 65`，其后在同目录使用 `--smoke --resume`。

6. 查看 Validation 与最佳模型：

```powershell
& $v6Python -m v6_sb3.episode_training --run-dir 'outputs/v6_economic_fixed_seed42' --inspect
Get-Content 'outputs/v6_economic_fixed_seed42/validation.jsonl' -Tail 1
Get-Item 'outputs/v6_economic_fixed_seed42/best_model.zip'
```

7. 确认风险后做一个真实数据 Trial 计时；它完成 1 个自然 episode 并评价固定 8 条 Validation，warmup 另计，可能仍耗时：

```powershell
$v6TrialTime = Measure-Command { & $v6Python -u -m v6_sb3.tuning --episode-search --study-dir 'outputs/v6_episode_pilot_seed42' --trials 1 --episodes-per-trial 1 --learning-starts 50000 --validation-every 1 --seed 42 --order-seed 42 --sampler-seed 42 --pruner none --acknowledge-failure-penalty }
$v6TrialTime.TotalHours
```

8. 按实测结果确认 Trial 数后，再启动有限搜索。以下 8×100 episode 是明确的示例预算，不能直接套用短 smoke 的秒数：

```powershell
& $v6Python -u -m v6_sb3.tuning --episode-search --study-dir 'outputs/v6_episode_optuna_seed42' --trials 8 --episodes-per-trial 100 --learning-starts 50000 --validation-every 50 --seed 42 --order-seed 42 --sampler-seed 42 --pruner median --acknowledge-failure-penalty
```

9. Optuna 在前台按一次 **Ctrl+C** 安全暂停；用相同预算/种子加 `--resume`：

```powershell
& $v6Python -u -m v6_sb3.tuning --episode-search --study-dir 'outputs/v6_episode_optuna_seed42' --trials 8 --episodes-per-trial 100 --learning-starts 50000 --validation-every 50 --seed 42 --order-seed 42 --sampler-seed 42 --pruner median --acknowledge-failure-penalty --resume
```

也可以在首次启动参数上加 `--stop-after-steps N` 请求本次调用的确定性暂停；恢复时去掉该参数。

10. 导出最优参数并从头进行独立 500 episode 重训：

```powershell
& $v6Python -m v6_sb3.tuning --episode-search --study-dir 'outputs/v6_episode_optuna_seed42' --export-best-parameters 'outputs/v6_episode_optuna_seed42/best_parameters.json'
& $v6Python -u -m v6_sb3.episode_training --run-dir 'outputs/v6_economic_retrained_seed42' --episodes 500 --learning-starts 50000 --validation-every 50 --seed 42 --order-seed 42 --parameters-json 'outputs/v6_episode_optuna_seed42/best_parameters.json' --acknowledge-failure-penalty
```

新目录从网络/优化器/Replay 重新初始化；导入的只有学习率、batch、Replay 容量、宽度与环境步 Target 间隔。没有载入 Trial 的已训练 checkpoint。500 episode 的探索计划重新开始。

11. 在同一 Validation 上比较固定基线与独立重训：

```powershell
& $v6Python -m v6_sb3.episode_training --compare-runs 'outputs/v6_economic_fixed_seed42' 'outputs/v6_economic_retrained_seed42'
```

比较命令拒绝 Validation 身份不同或尚无 Validation 的目录；其偏好沿用完成数/覆盖率/完整费用层级。它不运行 Test。

## 证据、性能与限制

[失败代价审计](failure_penalty_audit.md)、[原始短时性能与预算](environment_benchmark.md)、[完整验证](verification.md)、`regression.txt`、`cli_verification/` 保存此次交付的证据。Git 只收录轻量汇总，不收录新模型、Replay 或 SQLite。

真实短前缀 Queue/Direct 的耗时中位比为 1.4167/1.4341/1.3000。seed42 的 500 次抽样若全完成，共 6119941 正式步、191248 Adam；加预热和十次 Validation 的已测物理/Adam工作项约 4.789 小时，100/150 episode Trial 对应约 .990/1.473 小时。这不是可靠总时长：策略预测、成熟 Replay、重置、报告、checkpoint、失败造成的长度与行为变化尚未完整计入。8 个 100 episode Trial 仅同类工作项算术约 7.92 小时；需以上述真实计时入口校准。

纯经济累计恒等证明经济口径一致，不证明当前失败分数合理、500 episode 足够收敛或 Optuna 必然改善。现有聚合 FC/电池及 EOL 模型的建模限制保留，没有实施逐堆/分簇控制。当前训练状态仍不显式观测累积寿命，EOL 附近奖励对历史的依赖也未因本次环境重构而消失。
