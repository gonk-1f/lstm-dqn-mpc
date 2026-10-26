# V6 轮边界保存与续训

**最终状态（2026-10-26）：正式run02已完成8/8轮。** 下文记录早期第1轮暂停时的操作；
之后曾丢弃第5轮未保存进度，从第4轮完整断点重跑第5轮，最终完成至第8轮。
完整最新断点及全部逐轮模型已纳入[分析归档](results/v6_dqn_1s_20261026/README.md)。
本地生成结果将在确认推送后清理；下述原`outputs`路径届时不再存在。
配置中的总轮数仍为8；对已完成8轮的完整断点调用当前`--resume`会直接返回，不追加训练。

本次停止点为第 1 轮 Train、Greedy Validation、checkpoint 和全部报告完成之后。
运行目录中的 `pause_after_round.json` 使用 `{"round": 1}` 请求在报告写完后等待。
`round_boundary_ready.json` 出现且状态为 `ready_for_normal_stop`，表示报告脚本已进入等待门；
其中记录运行目录、已完成轮次、续训 checkpoint 路径以及父训练进程和报告进程 PID。

等待期间对共享控制台发送 Ctrl+C，正常结束父训练进程与报告子进程。等待门不保存或重写 checkpoint。
停止后把请求文件重命名为已完成记录，避免后续继续运行时再次进入同一等待门。

续训从保存的轮边界状态开始第 2 轮，恢复网络、Target、Replay、Adam、计数与随机状态。
它不是航段执行到一半的环境快照。学习率 `0.0001`、SOC 权重 `0.2`、FC 波动权重 `0.05`
及本次其余训练参数保持不变；不能把该续训当作重新暖身或从旧 V5 模型开始。

运行器提供 `--resume CHECKPOINT --output-dir ORIGINAL_RUN_DIRECTORY`，从完整断点的
`completed_rounds + 1` 继续至总第 8 轮。恢复固定探针、失败案例、最佳候选分数及 Q 诊断状态，
不重复预填充或已完成轮次的 Validation。仅有模型权重的 `round_01.pt` 不能用于完整续训。

在项目目录的 PowerShell 中，下次续训可以使用：

```powershell
& .\scripts\start_v6_dqn.ps1 -ResumeCheckpoint '.\outputs\v6_dqn_1s_seed42_20261025_run02\models\training_state_latest.pt'
```

需要在后续某个完整轮次保存并退出时，可增加 `-StopAfterRound 2`（绝对轮号）。
本次任务结束后不会执行上述续训命令，须等待用户通知。
