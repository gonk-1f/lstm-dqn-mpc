# v4 纯经济 Double DQN 正式配置

正式入口为 `python -m v4.feedback_study`。它固定使用 8–128–128–61 ReLU MLP、61 个 0–600 kW 动作、30 s 控制周期、100 轮、seed 42、300000 条经济 Replay、batch 64、γ=1、8-step 回报、Adam 学习率 0.0001、Smooth L1（δ=1）、梯度范数上限 10、replay32、失败终止批次配额 2、失败惩罚倍率 1、奖励缩放 0.001。目标网络初始化时硬复制一次，之后每次经济 Q 优化后以 τ=0.001 软更新；Outcome 网络更新不触发目标网络更新。

轮次 ε：第 1 轮 1.0，第 70 轮 0.05，第 71–100 轮保持 0.05。Train 每轮按 `seed + round` 独立打乱；Greedy Train 和 Validation 始终 ε=0。只有 Greedy Train 全部完成，才运行 Validation；两者全部完成的轮次才可按 Validation 可比成本选模。Test 始终封存。

ONBOARD 原始奖励仅为氢耗、燃料电池退化、电池退化及岸电四项成本之负值。`beta_soc=0`、电池能量奖励重分配关闭。失败终止惩罚单独计入训练奖励，不是人民币经济支出。8-step 奖励先在原始单位累计，插入经济 Replay 时缩放一次。一个数据样本内部跨 SHORE 连接 Bellman 轨迹；不同样本绝不连接。

每个已有 SHORE 区间是**模型化固定 SOC 目标补能**：不使用原始 `battery_bus_kw` 决定补能量，也不受记录的岸电时长限制。若抵港 SOC 小于 0.6，则用既有 `EconomicMPC.shore_interval` 的充电功率上限、充电效率、电价及电池退化模型，计算补到 0.6 的费用和寿命损耗；SOC 不低于 0.6 时不充放电。FC 在岸电边界关闭，已有退化模型结算停机。计算中所需的充电间隔仅是经济/退化模型的积分步骤，`ShoreBlock.settlement_basis` 标注为 `modeled_fixed_target_soc_0.6`，不生成实测 30 s 充电功率轨迹。末尾有 SHORE 时不追加 MODELED terminal settlement；末尾无 SHORE 且 SOC 低于 0.6 时保留独立的模型化终端结算。

每轮结束后原子写入 `training_state_latest.pt`，其中包含在线/目标/Outcome 网络、两个 Adam、Replay 与抽样池、随机数状态、更新额度、轮次与进度及数据 manifest。`best_agent.pt` 仍仅用于合格轮次选模；诊断 checkpoint 与续训 checkpoint 互不替代。续训要求源码提交、模型结构、完整超参数与数据 manifest 一致，并使用同一输出目录。运行中断产生的旧曲线会移入该目录下的 `resume_prior_curves_*` 文件夹，保留证据。

在 PyCharm PowerShell、仓库当前工作树根目录运行：

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python -m v4.feedback_study --output-dir .\outputs\v4_economic_ddqn_100r_seed42
```

从最近一个完整轮次续训：

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python -m v4.feedback_study --output-dir .\outputs\v4_economic_ddqn_100r_seed42 --resume-from .\outputs\v4_economic_ddqn_100r_seed42\training_state_latest.pt
```
