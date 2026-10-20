# 独立航段DDQN交错更新对照

## 已确认设计与实施清单

目标：仅改变正式探索阶段Q更新时序。`episode_end`保留默认旧行为；`interleaved`每新增32条已经确定奖励、后继状态和mask的经验调用现有调度器。Batch64、PER、一步masked DDQN、三维状态、7增量、奖励、学习率与tau均不变。

- [x] 在正式入口增加 `--update-timing episode_end|interleaved`，新选项仅用于独立increment7、n=1、replay32、soft Target、有明确experiment-id的实验。
- [x] 环境增加可选的已闭合transition通知。没有合法后继且尚未被判定为真实失败的末步不得通知；失败处理仍由原 `prepare_failure_replay` 决定，奖励只缩放一次。
- [x] 每个通知直接写入原Replay并调用原grant/consume，不重写DDQN或PER。最后真实失败动作在原失败处理后只入池一次。
- [x] 将失败航段已经入池的前缀标记为failure，仅改变来源/诊断分类，不改变原奖励、后继、priority或动作。采样时尚未知未来失败，因此流式TD类别按抽样当时已知结局记录，并在元数据明确；最终Replay前缀分类与旧模式一致。
- [x] 原bootstrap集中预填充及更新、Greedy Train、Validation与选模规则均保留。独立任务没有SHORE/UNKNOWN；旧模式对它们的处理完全不变，新流式开关拒绝此类任务。
- [x] checkpoint绑定时序身份。仅完整轮次保存，经验通知不留跨轮pending队列；复用原Replay/Adam/RNG/credit保存恢复，拒绝跨时序续训。
- [x] 少量合成定向检查：航段内实际更新时刻、失败末步/惩罚/物理账本、旧模式、参数路由、恢复身份与状态。

本轮不启动正式训练、不读取Test、不提交推送、不改变已有replay16准备改动。完成后在本文补充检查结果与正式命令。

## 对照口径

旧100轮是历史基准。新实验从相同seed42的新模型、空Replay、新Adam开始，100轮、独立输出。两组按相同新增经验/优化比例运行，但策略变化会影响交互长度，因此必须报告实际入池、优化和Target更新次数；不能仅凭同为100轮声称实际预算完全相等。

文献依据：Wu 2021 Algorithm 1在航段时间步循环内按Z步回放/梯度更新/软更新Target；§5.2为Z=32。论文：[开放全文](https://discovery.ucl.ac.uk/id/eprint/10135003/1/manuscript_accepted.pdf)。这支持检验时序，不保证消除FC振荡。

## 使用与边界

- 默认 `--update-timing episode_end`，新实验显式指定 `--update-timing interleaved --cadence replay32`；CLI拒绝与replay16组合，避免同时改变更新密度。
- `onboard_callback_optimizer_updates`、`failure_finalize_optimizer_updates`、`episode_end_optimizer_updates`进入逐轮报告及CSV；总Q/Target更新、credit、原经济评价与失败统计沿用原记录。
- 失败末步不提前入池；失败确认后使用原惩罚，只入池一次。不为了立即学习新失败而额外授予更新额度。若credit不足32，失败经验需等后续额度达到后才可能被采样，这是固定更新预算的结果。
- 最终Replay仍将失败航段前缀标为failure。历史TD类别按抽样时已知结局记录：失败尚未发生时，前缀被视作普通非终止经验；不把后来才知道的失败信息伪装成当时已知。跨组比较失败前缀时优先对固定经验重算快照TD，不直接混同比较这两种时序的历史类别均值。
- 仅在完整轮次保存checkpoint；中途终止恢复到最后完整轮次，未完成轮次重新执行。已完成100轮的checkpoint不能再作为“剩余轮次续训”。
- 新旧时序身份写入metadata，跨模式恢复在读数据前拒绝；本次源码变化也受既有源码SHA检查约束，旧正式100轮仅作历史对照，不迁移续训。
- 保留原共享RNG行为；同seed表示同初始化和可复现协议，不承诺时序变化后逐个探索动作/抽样相同。

## PowerShell命令

先在新终端执行公共配置；启动与恢复二选一。启动使用新权重、空Replay、新Adam，路径当前尚不存在。代码在训练开始后须保持不变，否则原有源码身份检查将拒绝恢复。

```powershell
Set-Location 'C:\Users\20883\OneDrive\Desktop\lstm-dqn-mpc\lstm-dqn-mpc\.worktrees\v4-increment7-zhou'
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
$ddqnPython = 'D:\py\Python3\python.exe'
$runDir = Join-Path (Get-Location) 'outputs\v4_independent35_m3_delta7_zhou_td1_per_interleaved_seed42_100r'
$ddqnArgs = @(
    '--dataset-mode', 'independent',
    '--data-root', 'data/processed/operating_dataset_independent_v2',
    '--state-mode', 'minimal3', '--action-mode', 'increment7',
    '--reward-cost-mode', 'zhou_joint', '--n-step', '1',
    '--rounds', '100', '--seed', '42', '--beta-soc', '200',
    '--reward-scale', '0.001', '--reward-feedback', 'original',
    '--replay-sampling', 'prioritized',
    '--per-alpha', '0.6', '--per-beta', '1', '--per-epsilon', '1e-6',
    '--cadence', 'replay32', '--update-timing', 'interleaved',
    '--failure-terminal-quota', '2', '--failure-penalty-scale', '1',
    '--episode-credit-scope', 'voyage', '--unknown-replay-mode', 'observed_nstep',
    '--experiment-id', 'V4-INDEP35-M3-DELTA7-ZHOU-TD1-PER-INTERLEAVED-S42',
    '--diagnostic-rounds', '1,10,20,30,40,50,60,70,80,90,100'
)
```

全新启动：

```powershell
if (Test-Path -LiteralPath $runDir) { throw '输出目录已存在；不要覆盖，续训请使用恢复命令。' }
& $ddqnPython -m v4.feedback_study --output-dir $runDir @ddqnArgs
```

断点恢复（新终端先重新执行上面的公共配置）：

```powershell
& $ddqnPython -m v4.feedback_study --output-dir $runDir @ddqnArgs --resume-from (Join-Path $runDir 'training_state_latest.pt')
```

本轮只准备接口，未执行上述正式命令，不提交推送。

## 本轮检查结果

26项定向检查通过（60.21秒），涵盖闭合通知、真实失败惩罚、Replay与priority一致性、正常/失败样本的中断恢复等价性、CLI/身份隔离、旧入口及UNKNOWN旧路径。6个Python文件AST解析通过，git diff --check通过（仅原已改文件有CRLF提示）。55个历史输出、checkpoint及Train数据/manifest文件SHA未变化。没有启动正式训练、没有读取真实Validation/Test payload，没有Git提交或推送。
