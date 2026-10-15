# 氢耗、岸电购电及SOC奖励：独立PER对照

分支：`feat/v4-energy-soc-per`。基于已保存的PER候选`6a470de`（其父提交为原DDQN基线`9cd6063`）。原`direct-power-mlp`工作树及历史结果不改。本轮仅准备代码和合成检查，不启动正式训练。

## 奖励和账本

通过显式参数`--reward-cost-mode energy_only`选择候选：

\[
r_k=-0.001\left[C_{H_2,k}+C_{\mathrm{shore,purchase},k}
 +200(SOC_{k+1}-0.6)^2+C_{\mathrm{fail}}\mathbf1_{\mathrm{failure\ terminal}}\right].
\]

- SOC使用实际ONBOARD动作后、岸充前的值；岸电阶段没有DQN动作或SOC辅助惩罚。
- H2包含FC船上补能的实际氢耗。没有额外电池放电费、能量势函数或终端势函数校正。
- FC和电池退化的物理状态及四项人民币账本完整保留。仅在此候选的训练奖励中去掉退化费用；`excluded_degradation_cost_cny`单列被排除的金额，包括岸充和末端补能的电池退化。
- 全成本选项`--reward-cost-mode full_economic`保留原奖励；默认仍为全成本，以免无参数调用时静默更改目标。
- 合格模型仍按固定完整Validation集合的**四项可比经济总费用**选择，资格和UNKNOWN口径不变。
- 失败惩罚仍使用既有参考额和倍率1，单独计一次，不冒充设备经济费用。目标简化后它相对正常回报的权重会改变，本轮不随之调参。

## 岸电费用如何传回DQN

中途SHORE按既有模型补至SOC0.6；购电费用只附着前一个真实ONBOARD动作。后续还有ONBOARD时，`done=False`，后继使用岸充后的真实SOC、停机后的FC历史及下个已观测负载。没有虚构岸电控制步。

样本尾已有SHORE时不再补一份终端结算；没有尾随SHORE才使用既有MODELED终端结算。UNKNOWN不虚构补能；物理失败不执行尚未发生的岸电。

例如抵港SOC0.5、目标0.6，按624kWh、效率0.95和1.10元/kWh，购电量约65.684kWh、费用72.253元，进入奖励后为−0.072253。若FC已在船上补回，则其能耗由氢耗体现，不再重复收电池能源费。

这是延迟反馈：末端费用可直接进入最后8个起点的n-step窗口，更早的动作依赖bootstrap。gamma1不折扣该费用；PER增加已有经验重用，但不保证已学准。重叠回放窗口重复学习同一个结果，不是账本再次收费。

## 固定配置

7–128–128–61 ReLU MLP；61个0–600kW绝对动作；Adam0.001；gamma1；n8；batch64；Replay300000；replay32；每次经济Adam后soft target tau0.001；Huber1；梯度裁剪10；reward_scale0.001；betaSOC200；失败终止配额2；失败倍率1；seed42；100轮；epsilon第1～70轮从1降到0.05，此后保持。沿用v3标签、UNKNOWN observed_nstep、每轮独立Greedy Train/Validation和Test封存。

PER使用保留实现：`alpha=0.6, beta_IS=1, epsilon=1e-6`，普通池比例抽样及逐样本重要性权重，失败池保留配额2。此处beta_IS不是betaSOC。没有新动作门控、强制FC出力或功率滤波。

## PyCharm PowerShell

新实验必须从空Replay、新网络和新Adam开始。第一次使用以下独立目录；已有目录会被入口拒绝覆盖。训练前不需从原分支搬运或加载旧模型。

```powershell
Set-Location 'C:\Users\20883\OneDrive\Desktop\lstm-dqn-mpc\lstm-dqn-mpc\.worktrees\v4-energy-soc-per'
$env:PYTHONPATH = Join-Path (Get-Location).Path 'src'
$env:OMP_NUM_THREADS = '16'
$env:MKL_NUM_THREADS = '16'
& 'D:\py\Python3\python.exe' -u -m v4.feedback_study --output-dir 'outputs\v4_energy_soc_per_beta200_seed42_100r_20261015' --experiment-id 'V4-ENERGY-SOC-PER-S42' --reward-cost-mode energy_only --replay-sampling prioritized --per-alpha 0.6 --per-beta 1 --per-epsilon 0.000001 --beta-soc 200 --reward-feedback original --reward-scale 0.001 --failure-terminal-quota 2 --failure-penalty-scale 1 --cadence replay32 --n-step 8 --unknown-replay-mode observed_nstep --episode-credit-scope voyage --seed 42 --rounds 100 --diagnostic-rounds '1,10,20,30,40,50,60,70,80,90,100'
```

16个CPU线程与B100归档配置一致；本轮定向合成检查使用1线程，不是正式训练配置变化。

续训恢复最后一个完整轮次，须保留同一源码、参数、输出路径和diagnostic-rounds：

```powershell
Set-Location 'C:\Users\20883\OneDrive\Desktop\lstm-dqn-mpc\lstm-dqn-mpc\.worktrees\v4-energy-soc-per'
$env:PYTHONPATH = Join-Path (Get-Location).Path 'src'
$env:OMP_NUM_THREADS = '16'
$env:MKL_NUM_THREADS = '16'
& 'D:\py\Python3\python.exe' -u -m v4.feedback_study --output-dir 'outputs\v4_energy_soc_per_beta200_seed42_100r_20261015' --resume-from 'outputs\v4_energy_soc_per_beta200_seed42_100r_20261015\training_state_latest.pt' --experiment-id 'V4-ENERGY-SOC-PER-S42' --reward-cost-mode energy_only --replay-sampling prioritized --per-alpha 0.6 --per-beta 1 --per-epsilon 0.000001 --beta-soc 200 --reward-feedback original --reward-scale 0.001 --failure-terminal-quota 2 --failure-penalty-scale 1 --cadence replay32 --n-step 8 --unknown-replay-mode observed_nstep --episode-credit-scope voyage --seed 42 --rounds 100 --diagnostic-rounds '1,10,20,30,40,50,60,70,80,90,100'
```

运行元数据和完整checkpoint记录奖励模式/权重以及PER参数；不同奖励模式不能续训。网络、优化器、回放及优先级、RNG和更新调度沿用完整轮次恢复机制。未完成轮次从上一完整checkpoint重新执行。

## 判定边界

此候选优化能源费用和SOC偏好，不直接优化全部设备寿命费用。以完整供电成功率、固定完整集合的四项总费用、FC变化及启停、SOC和各类TD拟合共同判定；奖励曲线或TD误差下降不能单独证明改进。若与旧Uniform B100比较，同时包含采样和奖励两项变化，不能称单变量结果。

## 本轮验证

- 新增8项奖励/岸电/失败/UNKNOWN检查和6项入口检查，均先观察到接口缺失导致的失败，再实现通过。
- 最后一次核心定向检查：20 passed，13.68秒，另包含原PER采样/恢复、完整轮次恢复、软更新、DDQN目标和实际网络优化器设置。
- 入口及原PER/缩放入口回归：63 passed，22.46秒。该集合与上述检查有重叠，不能直接相加当不同用例数。
- 5个改动源码AST检查、CLI帮助加载及git diff --check通过；14个冻结数据metadata文件哈希与原工作树一致。
- 独立只读代码复核未发现新增的确定错误。原工作树已跟踪src/tests未改，未启动正式训练，未读取正式Validation/Test轨迹。
