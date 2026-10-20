# 独立航段DDQN：replay16单变量对照准备

## 范围与实现

- 基准为已完成的 `V4-INDEP35-M3-DELTA7-ZHOU-TD1-PER-S42`、`replay32` 100轮实验。
- 新组仅将经济Q更新密度改为 `replay16`：每新增16条经验获得一次经济Q更新，余数继续保留，bootstrap也使用相同调度。沿用现有调度器，不复制DDQN或重写Replay。
- 正式入口默认仍为replay32。显式replay16仅放行 `independent + minimal3 + increment7 + zhou_joint + n_step=1` 且提供非空experiment-id的实验；其他冻结检查保留。
- 数据仍为 `operating_dataset_independent_v2 / splits_v2_exclude046_seg01`，Train35/Validation8/Test4；本轮未读取数据载荷或Test。
- 状态、3–128–128–7网络、七增量、gamma1、Adam0.001、Batch64、Replay300000、PER0.6/1/1e-6、失败配额2、tau0.001、reward_scale0.001、SOC奖励及Zhou账本全部不变。
- cadence和experiment-id原本已进入元数据、诊断checkpoint及完整续训身份；跨cadence恢复在数据载荷读取前拒绝。沿用已有全新目录和归档保护。
- 新组必须使用新网络、空Replay及新Adam。不得从旧replay32的第100轮续训，也不得覆盖其输出。

只修改入口放行、直接相关的定向测试及本文；本轮不启动正式训练、不提交Git。

## 解释限制和验收

固定每次Adam后tau0.001软更新，增加更新密度也会加快Target相对环境经验的跟踪；这项耦合必须报告，不能把任何收益全部解释成动作覆盖改善。交互路径改变后，实际总更新次数也不保证恰好是旧组两倍。

对比固定Train35/Validation8的完整完成率、物理失败、同一完整集合的Zhou成本、FC平均绝对变载/反向次数、SOC范围和各类别未加权TD误差。不得把更短失败前缀的低费用当作改善，不使用Test调参。若TD降低而完成率/经济/振荡未改善，不能宣称解决了控制问题；单seed收益仍需后续复核稳定性。

9项定向检查通过：入口默认replay32与replay16实际传递；无实验ID或旧任务不能借replay16绕过范围；合成元数据的训练超参数仅cadence不同；跨cadence续训被拒绝；旧增量入口、Q诊断、结构身份和更新额度余数继续通过。两个修改的Python文件语法检查及git diff空白检查通过。没有运行正式训练或全量测试。

## PyCharm PowerShell：公共设置

先执行此段。解释器和工作树路径已在本机核对；新输出目录在准备时不存在。

```powershell
Set-Location 'C:\Users\20883\OneDrive\Desktop\lstm-dqn-mpc\lstm-dqn-mpc\.worktrees\v4-increment7-zhou'
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
$ddqnPython = 'D:\py\Python3\python.exe'
$runDir = Join-Path (Get-Location) 'outputs\v4_independent35_m3_delta7_zhou_td1_per_replay16_seed42_100r'
$ddqnArgs = @(
  '--dataset-mode','independent',
  '--data-root','data/processed/operating_dataset_independent_v2',
  '--state-mode','minimal3','--action-mode','increment7',
  '--reward-cost-mode','zhou_joint','--n-step','1',
  '--rounds','100','--seed','42','--beta-soc','200',
  '--reward-scale','0.001','--reward-feedback','original',
  '--replay-sampling','prioritized','--per-alpha','0.6','--per-beta','1','--per-epsilon','1e-6',
  '--cadence','replay16','--failure-terminal-quota','2','--failure-penalty-scale','1',
  '--episode-credit-scope','voyage','--unknown-replay-mode','observed_nstep',
  '--experiment-id','V4-INDEP35-M3-DELTA7-ZHOU-TD1-PER-R16-S42',
  '--diagnostic-rounds','1,10,20,30,40,50,60,70,80,90,100'
)
```

### 首次启动

```powershell
& $ddqnPython -m v4.feedback_study --output-dir $runDir @ddqnArgs
```

目录已非空时入口拒绝重新开始；不要删除旧产物来绕过保护，应恢复该新组或另选全新目录。

### 中断后恢复本组

重开终端先重新执行“公共设置”，然后只运行下面一行，不运行首次启动行：

```powershell
& $ddqnPython -m v4.feedback_study --output-dir $runDir @ddqnArgs --resume-from (Join-Path $runDir 'training_state_latest.pt')
```

仅从该新输出目录内的最新完整轮次恢复。源码内容/HEAD、cadence、实验ID、数据manifest、诊断轮次和其他超参数必须与启动时一致。若准备提交代码，应在启动训练前完成；本轮没有替用户提交或启动。
