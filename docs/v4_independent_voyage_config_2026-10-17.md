# V4独立航段35/8/4与Zhou三成本配置

## 1. 当前冻结身份

- 实现基于HEAD：`a10b73eb32fc653516ff5c5e59550a83efb1f666`；未启动正式训练。
- 数据目录：`data/processed/operating_dataset_independent_v2`。
- 正式划分：`splits_v2_exclude046_seg01`，35 Train、8 Validation、4 Test。
- 只排除`parent_046_seg01`的正式Train资格（人工审查：25.94小时、尾部长近零待机），保留其全部数据和原始映射；`parent_046_seg02`仍为Train。
- 原`splits_v1_seed42`全部11个文件保持原哈希。新版剩余manifest行未改变原值，Validation/Test名单字节不变，48个1秒候选及48个30秒预览全部保留。
- 当前sample manifest SHA256：`3273f3b25adff6f1cb93ff8d845729bc21d8e0499fccb04f3e7039aaef140ca0`。
- 当前active SHA256：`fe00778e08ce1bf41b4ae34638c3aab3334872cd83bf1771a2ec06488f4b92a0`。
- 当前policy SHA256：`d328544d4bff1ff148a19abd6c2ee204e74bbd96af9d8733a7852db4ea286b40`。
- 运行身份还绑定payload哈希清单与exclusions。加载时只验证请求的Train/Validation载荷，不打开或哈希Test载荷。

所有ID均加前缀`parent_`；下列为完整清单：

| 集合 | ID后缀 |
|---|---|
| Train35 | 003_seg01, 005_seg01, 006_seg01, 007_seg01, 008_seg01, 009_seg01, 009_seg02, 010_seg01, 010_seg02, 012_seg01, 015_seg01, 017_seg01, 018_seg01, 019_seg01, 019_seg02, 021_seg01, 022_seg01, 029_seg01, 029_seg02, 034_seg01, 035_seg01, 036_seg01, 037_seg01, 038_seg01, 041_seg01, 042_seg01, 043_seg01, 046_seg02, 047_seg01, 048_seg01, 049_seg01, 050_seg01, 051_seg01, 052_seg01, 053_seg01 |
| Validation8 | 002_seg01, 011_seg01, 016_seg01, 020_seg01, 033_seg01, 040_seg01, 044_seg01, 045_seg01 |
| Test4 | 001_seg01, 004_seg01, 013_seg01, 032_seg01 |

## 2. 任务与时间边界

`IndependentTrainingDataset`按canonical 1秒文件读取已有的`time_s=30k<duration_s`点，不重新插值、改负载或补零；区间内沿用现有左端负载保持口径。最后一步`dt=min(30,duration-t)`，终点仅作边界记录，不增加一个控制步。该方式没有逐秒重建负载积分，所有基线必须采用同一约定。

- 每个独立段重新初始化SOC=0.6、上一FC=0、FC损耗和电池退化累计状态=0。
- 原始非零首尾保留；raw负载、时间戳和来源行保留。控制侧继续使用既有`[-1,0)`kW数值容差；例如原始-0.568kW仍保存在载荷/轨迹raw字段，物理输入为0。这不是把几十kW边界改成0，也没有新定义负载容差。
- 当前与后继动作掩码、SOC、实际氢耗及原FC/电池退化均使用相同真实区间时长。SOC占用和功率等时间统计按时长加权。
- 不执行SHORE、不调用`modeled_terminal_settlement()`，不加电费或补能退化，不恢复末端SOC。
- 完整末步`done=True`、无bootstrap；物理不可行仍使用最后真实动作及原失败机制，不虚构失败动作。n-step不跨独立样本。
- 末端保留最后实际FC状态，不制造额外停机时间步；既有模型在0→正功率时收取启动循环费用，末端不再重复收取。非零边界可能使该段成为测量记录的独立片段，不宣称真实完整靠泊循环。

## 3. 经济评价独立于训练奖励

代码：`src/v4/independent_economics.py`，`ZhouEvaluationConfig.from_formal_curve()`与`evaluate_zhou_economics()`。

\[
\eta_{ref}=\frac1{60}\sum_{P=10,20,\ldots,600}\eta_{formal}(P)
=0.5711999752720877
\]

实现调用原`formal_fuel_cell_efficiency_map().eta(P)`，以`math.fsum`稳定求和，不包含0kW；60个功率点、效率值、方法、PCHIP曲线来源及源文件哈希写入元数据和checkpoint。

论文表述：本文采用燃料电池效率特性曲线在10～600kW范围内、间隔10kW的60个功率点对应效率的算术平均值，作为储能净能量变化折算等效氢耗的统一参考效率。该参考值与具体航次及控制算法无关。燃料电池实际氢耗仍按照其瞬时输出功率对应的效率计算。该值是透明、可复现的研究假设，未经实船试验标定，不是Zhou论文直接报告的数值。

\[
m_{Bat,equ}=\frac{(0.6-SOC_T)\,624}{\eta_{ref}\,(120/3.6)},\quad
C_{H2,equ}=(m_{H2,actual}+m_{Bat,equ})\,21.9
\]
\[
C_{total}=C_{H2,equ}+C_{FC,deg}+C_{Bat,deg}.
\]

容量kWh、LHV为kWh/kg、氢耗kg、费用CNY。SOC修正有符号，不截负，不另乘或除电池效率，不同时结算逐步电池能源价值或末端购电。实际氢耗与FC/电池退化沿用原模型及价格，包括原有EOL费用封顶；没有借此改成新退化模型。旧非负四项物理账本保留，SOC修正在单独评价结构中计算。

方法依据为Zhou等2025，JMSE13(1):34，DOI `10.3390/jmse13010034`，式(31)/(32)；本研究没有超级电容项。624kWh、21.9元/kg等沿用本项目研究参数，不能称为Zhou原文同一设备标定。

## 4. 冻结学习与选模

3–128–128–61、61档绝对FC动作、Adam0.001、gamma1、n8、batch64、Replay300000、replay32、Target软更新tau0.001、PER alpha0.6/beta1/epsilon1e-6、失败终止配额2、reward_scale0.001、seed42、100轮与原epsilon计划不变。

正常控制奖励仍为`-[0.11025*abs(actual_FC-previous_FC)+200*(SOC_next-0.6)^2]`，进入Replay仅缩放一次。三成本评价不加入此奖励。

失败惩罚沿用固定历史参考`10132.660087898294`、倍率1，未重估。其来源是旧30条Train-P95，其中部分旧Train如今属于新Test，不能再称新任务Train-only统计；元数据明确标注历史固定参考。本次未用新Test或新闭环结果调整该值；如后续决定重标定，须单独批准实验。

每轮独立执行Greedy Train和Validation。仅固定Train35及Validation8全部完成、无失败/截断/执行错误，才按Validation完整三成本选best。失败前缀费用只能诊断。未合格不产生`best_agent.pt`。

metadata/完整续训checkpoint绑定任务契约、曲线参考、全部超参数、源码身份及新数据哈希；缺少新任务身份或持有旧数据/旧权重/旧Replay均拒绝续训。新Train重点Q诊断ID与实际划分交叉筛选，不再使用旧zero_boundary ID。

## 5. 已执行的必要检查及限制

43条Train/Validation共20062个控制区间，积分时间601324秒；全部连续、总dt等于真实时长，无SHORE/UNKNOWN。11个非零起点、7个非零终点保留。未打开新Test载荷。

43条均通过瞬时功率界及连续FC松弛SOC可达区间必要条件；**不能据此宣称61档离散动作下全部可行**。只有最短Train `parent_046_seg02`完整经过正式物理执行：110步/3272秒、尾步2秒，SOC0.6→0.6004798813878119，无学习、无未来负载输入、无岸充和末端补能；三成本1071.9493373710259元，其中SOC等效修正-0.0157272058276277kg。其余42条完整离散可行轨迹尚未确认，不自动删除。

检查详情保存在`outputs/v4_independent_readiness_20261016/physical_checks.json`。没有重跑旧100轮、开展正式训练或Test策略评价。最终定向检查共47项通过（45项集中检查及两项诊断身份/单位检查），AST与目标文件`git diff --check`通过，未运行全量pytest。

实际代码范围：

- 新增`src/v4/independent_dataset.py`、`src/v4/independent_economics.py`。
- 修改`src/v4/feedback_study.py`、`monitored_training.py`、`control.py`、`train.py`、`experiment_schedule.py`，接通新数据、评价、终止和选模。
- 修改`src/v4/diagnostics.py`、`decision_diagnostics.py`、`telemetry.py`，记录真实时长、三成本、当前Train诊断ID和控制奖励单位。
- `src/v3/control.py`只给既有interval增加可选实际时长，默认仍为30秒，物理/退化公式及设备参数不变。
- 新增`scripts/exclude_independent_voyage_sample.py`、三份`tests/test_v4_independent_*.py`；新增v2冻结metadata并更新active和数据README。未改`src/v4/dqn.py`、PER或任何候选功率文件。

## 6. PyCharm PowerShell命令（仅供手动执行）

新启动或重启终端后先执行共同设置：

```powershell
Set-Location -LiteralPath 'C:\Users\20883\OneDrive\Desktop\lstm-dqn-mpc\lstm-dqn-mpc\.worktrees\direct-power-mlp'
$ddqnPython = 'D:\py\Python3\python.exe'
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
$runDir = Join-Path (Get-Location) 'outputs\v4_independent35_m3_control_per_seed42_100r_splitv2'
$ddqnArgs = @(
    '--dataset-mode', 'independent',
    '--data-root', 'data/processed/operating_dataset_independent_v2',
    '--state-mode', 'minimal3', '--reward-cost-mode', 'control_only',
    '--replay-sampling', 'prioritized',
    '--per-alpha', '0.6', '--per-beta', '1', '--per-epsilon', '1e-6',
    '--rounds', '100', '--seed', '42',
    '--beta-soc', '200', '--reward-scale', '0.001',
    '--cadence', 'replay32', '--n-step', '8',
    '--failure-terminal-quota', '2', '--failure-penalty-scale', '1',
    '--episode-credit-scope', 'voyage',
    '--experiment-id', 'V4-INDEP35-M3-CTRL-PER-S42',
    '--diagnostic-rounds', '1,10,20,30,40,50,60,70,80,90,100'
)
```

全新启动（目录必须为空；不加载任何旧checkpoint）：

```powershell
& $ddqnPython -m v4.feedback_study --output-dir $runDir @ddqnArgs
```

恢复：重开终端先重新执行上述共同设置，**随后只执行以下恢复行**。仅恢复本次输出目录完整轮次checkpoint；不足一轮的进度按原机制重新执行，不能把旧100轮checkpoint放入此目录。

```powershell
& $ddqnPython -m v4.feedback_study --output-dir $runDir --resume-from (Join-Path $runDir 'training_state_latest.pt') @ddqnArgs
```

首次尚未生成完整轮次checkpoint时不能恢复。开始后固定源码、配置和新数据身份，避免运行期间修改导致严格恢复校验失败。
