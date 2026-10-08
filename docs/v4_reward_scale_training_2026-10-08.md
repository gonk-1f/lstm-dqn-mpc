# v4 正式训练准备：统一奖励缩放（2026-10-08）

基于 `3d75e5a`，依据[合成数值诊断](v4_td_numerical_diagnostic_2026-10-08.md)接入可配置缩放。本次只修改代码、测试和监控，正式40轮由用户在PyCharm终端启动；没有正式完成率、最佳轮次或学习率调整结论。

## 奖励与账本

设已有完整训练奖励（原奖励单位）为 `r`，它已经包含当步经济费用、SOC软惩罚、适用的电池价值重分配/终端校正、真实SHORE或MODELED终端补能，以及失败终止时的一次 `C_fail`。

经济replay中只在 `DirectPowerDDQN.remember` 做：

\[
r_{replay}=\alpha r,\qquad\alpha=\texttt{reward_scale}.
\]

默认仍为1，本次命令显式选0.001。`control.py`、物理状态、SOC递推、动作mask及四项真实经济账本不缩放。失败惩罚的人民币等价参考尺度保持原值，不冒充实际支出。outcome经验的原经济reward、标签和训练方式保持不变。

完整航段的原奖励与重分配奖励累计相等；失败后缀按既有机制闭合势函数并扣一次失败惩罚。乘同一个alpha后恒等式仍成立。n-step先在原单位组成回报，再统一缩放一次；重叠的n-step回报不能通过直接相加来核验整段累计奖励。

Replay中的历史字段 `reward_cny` 保存缩放后的训练奖励，不能当作实际人民币费用；报告和诊断明确给出单位。Q、TD target与TD error可以除以alpha换回原奖励单位。Smooth L1使用原来的固定阈值1，其loss不能通过简单除alpha当作未缩放Huber loss。

## 本次固定配置

| 项目 | 本次运行值 |
|---|---|
| reward / reward_scale | redistributed / 0.001 |
| beta_soc / failure_penalty_scale | 500 / 1 |
| 网络 / 动作 | MLP 8–128–64–61 / FC 0:10:600 kW |
| Adam学习率 / gamma | 0.0001 / 1 |
| batch / replay容量 | 64 / 100000 |
| 更新 / Target | replay32 / 每500次经济optimizer更新hard sync |
| n-step | 1 |
| epsilon | 每轮线性1.0→0.05，第40轮0.05 |
| rounds / seed | 40 / 42 |
| 数据 | 冻结Train30、Validation8；Test封存 |
| 初始化 | 新随机网络、新replay、新optimizer；保留原有当前Train预填充流程 |

网络、学习率、gamma、Target机制、失败倍率、物理/退化/经济参数都不调整。本次不复用历史模型或经验池，不启动其他配置。

## PyCharm PowerShell终端命令

在当前分支对应工作树执行一次：

```powershell
Set-Location -LiteralPath "C:\Users\20883\OneDrive\Desktop\lstm-dqn-mpc\lstm-dqn-mpc\.worktrees\direct-power-mlp"
$env:PYTHONPATH = (Resolve-Path .\src).Path
$v4RunOutput = "outputs/v4_scale001_beta500_replay32_target500_seed42_40r_" + (Get-Date -Format "yyyyMMdd_HHmmss")
python -X utf8 -u -m v4.feedback_study --rounds 40 --reward-feedback redistributed --reward-scale 0.001 --beta-soc 500 --failure-penalty-scale 1 --cadence replay32 --target-interval 500 --n-step 1 --episode-credit-scope voyage --output-dir $v4RunOutput
```

目录必须全新且为空；入口拒绝旧归档和已有运行目录，不会自动覆盖或恢复。seed42、batch64、epsilon1→0.05等固定值由入口传递，学习率/gamma/replay/network由现有agent配置确定。

每50个训练决策步输出一次进度，每轮输出探索和greedy完成率、Validation费用及更新数量。该命令在本次实现验证中没有执行。

## 结果保存与解释

训练日志、运行元数据、逐轮JSON/CSV及轨迹保存在所选新目录。Q/TD日志同时包含alpha及原单位换算，记录真实裁剪前梯度范数、裁剪比例、成功/失败采样、更新与Target计数。经济费用一直为人民币，SOC shaping、势函数重分配和失败惩罚独立于经济账本。

主要文件：`train.log`、`run_metadata.json`、`report.json`、`round_history.json`、`round_metrics.csv`、逐轮探索/greedy轨迹JSON。结束或中断后生成 `completion_rates.png`、`validation_economic_costs.png`、`td_and_gradients.png`、`soc_and_epsilon.png`、`fc_and_update_counts.png`，以及最佳合格/最后完整轮次的代表性FC、Battery、SOC图；各图原始序列写入 `learning_curves_metadata.json`。

每轮先完整greedy Train；只有30/30完成才评估Validation8。只有Train30/30且Validation8/8的轮次，才按Validation可比费用保存 `best_agent.pt`。模型与报告记录reward_scale、源码提交和数据manifest，未出现合格轮次时不生成合格best模型。

未完成航段的部分费用单独记账，完整费用为null，不与完成航段直接比较。异常/中断保留已完成轮次和日志，停止运行，不加载不完整模型或擅自修改超参数继续。

最终解读需要分别报告最佳合格轮次和第40轮；缺失/跳过的Validation不是0成本，也不是8/8。完成40轮后再判断是否值得做0.0001与0.0003的单变量学习率对照。

## 实现验证结果

最终12个相关测试文件共 **143 passed，81.82s**。全部使用合成轨迹或fake dataset，不读取正式Train/Validation/Test功率数据，不启动正式实验。

- 原奖励/重分配、实际SHORE/MODELED、失败一次惩罚/无bootstrap、n=1/8及容量满后插入均验证只缩放一次。
- 同一固定动作下物理状态、原账本、充电退化与终端结算一致；outcome的原经济reward和标签一致。
- 默认1与显式1的实际optimizer更新、replay和RNG/权重一致；旧监控兼容性回归通过。
- metadata/report/best模型记录实际scale、源码提交和manifest；首轮前即可保存来源信息。
- 中断、模型异常、最终manifest校验失败均保留完整轮次并停止。最终数据校验失败会把选模有效性标为false，原权重保留为无效证据。
- Windows长图像路径问题已通过实际≥260字符PNG回归；恒功率归一化舍入误差只在诊断中视为零，不改变任何动作/物理模型。

合成30/8 CLI样例（仅1轮、每Train样例4步）实际插入240条经济经验：bootstrap3次、训练4次，共7次经济更新，剩余credit16；Target仅初始化复制1次。另一个既有失败样例实际插入110条（69成功、41失败），更新3次、失败terminal1条，当前失败后缀排除数0。这些数字核验调度执行，**不是正式40轮结果**。

277项旧模型/资料/结果/校准保护哈希及5个数据manifest全部保持一致。未删除或覆盖旧实验结果。

[验证与哈希/更新计数](results/v4_reward_scale_implementation_20261008/verification.json)、[最终回归日志](results/v4_reward_scale_implementation_20261008/regression_final.log)及[证据索引](results/v4_reward_scale_implementation_20261008/README.md)。
