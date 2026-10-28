# PyCharm终端启动：SB3 DQN对照

本次交付只完成代码与训练前必要检查，**正式训练尚未启动**。新工作树为
`C:\Users\20883\OneDrive\Desktop\lstm-dqn-mpc\lstm-dqn-mpc\.worktrees\v6-sb3-baseline`，
分支`feat/v6-sb3-baseline`，基准实际commit为`fce060ca12fd842d8496aba8d6f922eae179075e`。

## 直接在PyCharm的Terminal / PowerShell运行

```powershell
Set-Location 'C:\Users\20883\OneDrive\Desktop\lstm-dqn-mpc\lstm-dqn-mpc\.worktrees\v6-sb3-baseline'
$env:PYTHONPATH = Join-Path (Get-Location).Path 'src'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:PYTHONUTF8 = '1'
& '.\.venv\Scripts\python.exe' -u -m v6_sb3.training --output-dir '.\outputs\sb3_dqn_1s_seed42_run01'
```

该输出目录必须是新目录；已有运行需要使用恢复命令。也可使用简短启动器，省略OutputDir会自动使用带时间戳的新目录：

```powershell
& .\scripts\start_v6_sb3.ps1
```

命令显式使用新工作树`.venv\Scripts\python.exe`，不依赖PyCharm当前选择了哪个解释器。若需要在PyCharm的Run配置里启动，也应选择这个解释器，Working directory设为上述工作树，模块为`v6_sb3.training`，环境变量`PYTHONPATH=上述工作树\src`。

## 输出与内存

SB3设为`verbose=0`、`progress_bar=False`、`log_interval=None`；不启用TensorBoard。
终端仅有启动信息和每遍一行Train完成/Greedy完成/FC变载/相邻反向摘要。
没有每步Q值、每5000步打印或逐航段终端输出。详细航段摘要追加到`events.jsonl`；
每5000步仅覆盖`progress.json`，不会在终端持续累积。

频繁输出会增加终端缓存，但现有证据没有证明它是此前OOM唯一原因。这里同时限制了训练对象保留范围：

- Replay是SB3固定容量100万的数值数组，补充next-mask/失败标签；不保留计算图列表。
- 只保留当前航段完整物理记录，结束后处理并释放，不把所有Train逐秒轨迹堆在内存。
- worker动作/事件队列各容量1，一条航段一个worker，reset/close会join并释放。
- Validation的Q批次最多2048行，轨迹逐行写gzip，不构造整条13Q字典列表。
- TD统计的P95 reservoir每类8192，诊断随机源与训练隔离。
- 每遍记录一次进程工作集和私有提交量；这不是整个Windows系统的提交量。

## 暂停与恢复

在终端按一次`Ctrl+C`，表示请求暂停。程序在一条真实转移入Replay并完成对应Adam后保存完整断点再退出；
如果正在写报告或checkpoint，会完成该阶段后处理暂停。等待终端显示`Paused safely`后再关闭终端。

```powershell
& '.\.venv\Scripts\python.exe' -u -m v6_sb3.training --resume '.\outputs\sb3_dqn_1s_seed42_run01\models\latest_checkpoint.json'
```

上例自动继续最近完整断点；也可显式给出`models\round_NN`或`models\pause_step_N`目录。
`models/latest_checkpoint.json`记录最近的完整目录。不要用`best_research_candidate.zip`冒充完整续训断点。

每个完整目录包含`model.zip`（Online/Target/Adam与SB3计数）、`replay.pkl`、`training_state.pt`
（Python/NumPy/Torch随机状态、课程位置、失败诊断及报告），最后写入`checkpoint.json`完整标记。
中途暂停的物理状态通过保存的实际动作前缀重建；重建不调用策略、不添加Replay、不做Adam，也不计作新训练。
恢复显式固定CPU，不使用SB3默认`device=auto`。已完成8遍的断点不追加训练。

## 预算与比较边界

用户已选择固定8遍、每遍35条Train各执行一次，总步数随策略失败时刻变化。
warmup恰好50,000条真实Train转移，无Adam；正式每步一次Adam，Target每1000次Adam硬复制。
前6遍按原pass/episode进度衰减epsilon，后2遍保持0.05。
不能同时声称总正式更新恰好等于旧V6的2,782,159次。

状态、动作、物理模型、奖励和约束与run02相同。原1秒数据新增插值点为0，本次没有重新处理数据。
没有PER、Double-DQN、额外可恢复性约束或Test读取。

## 训练结束后查看

- `training_result.md`、`training_records.csv`、`round_NN_summary.md/.json`：完成、失败、波动、成本与四类TD统计。
- `round_NN_validation/*.csv.gz/.png`：全部13Q/合法排名、FC/负载/电池/SOC、奖励分项。
- `q_probes/round_NN.json/.csv`：与V6同44个固定物理状态的全部13Q与跨checkpoint排序。
- `round_NN_replay_td_fit.csv.gz/.json`：原只读分层Replay诊断；TD目标不是Q*。
- `round_NN_train_failure_values*`、`round_NN_train_failure_prefixes*`：真实Train失败与价值传播。
- `round_NN_failure_details.csv/.json`：失败F时状态与13动作；区别最后执行F−1时刻。
- `round_NN_reference_comparison*`、`round_NN_common_completed_comparison.csv`：V6第2/8遍及V5第32轮，成本仅比较共同完整航段。
- `models/round_NN/`：每遍完整恢复文件；`runtime.json`和`source_snapshot/`记录本次实际环境与新增源码。

源码审查见[v6_sb3_source_audit.md](v6_sb3_source_audit.md)；没有正式训练结果时，不判断SB3是否优于原V6。

## 最终Test（由用户在终端执行）

本次8遍已结束。按原定Validation完成数、覆盖率、全完成经济成本的顺序，冻结选择第5轮。
选择依据保存在运行目录的`analysis/model_selection.json`，逐轮比较见`analysis/final_validation_selection.csv`。
Test入口读取该选择记录，固定加载`models/round_05/model.zip`，不会使用第8轮latest模型。

在项目根目录的PowerShell终端执行：

```powershell
$env:PYTHONPATH = Join-Path (Get-Location).Path 'src'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:PYTHONUTF8 = '1'
& '.\.venv\Scripts\python.exe' -u -m v6_sb3.evaluation --run-dir '.\outputs\sb3_dqn_1s_seed42_run01'
```

只打印开始和结束摘要。仅当用户执行该命令时，从冻结manifest指向的原数据目录读取4条已有canonical 1秒Test航段；
不重新插值或复制数据，不做Adam更新，不写训练Replay，不根据Test重选模型。
结果保存到`outputs/sb3_dqn_1s_seed42_run01/test_round_05/`：

- `test_summary.json`、`test_result.md`、`test_sample_metrics.csv`：完成率、奖励、氢耗、退化成本、经济成本及变载统计。
- `round_05_test/*.csv.gz`、`*.png`：逐秒13Q与合法排序、FC/负载/电池功率、SOC和奖励分项。
- `test_failure_details.json`及失败时的CSV：失败状态、物理合法范围和13档动作情况。

只有4条全部完成才报告全Test集合经济成本；若失败，成本和氢耗仅覆盖真实执行前缀。
该入口尚未由助手执行Test，不能提前宣称Test完成或通过。
