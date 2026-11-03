# 第四、第五章待补实验与图表

本文件只规定后续证据需求。本次未执行训练、验证、Test、Optuna或策略推理；不把既有旧奖励结果转成当前奖励的性能结论。

## 1. 实验身份与前提

- 固定奖励身份`ddpg_economic_tanh_v1`及SOCλ=17，保存完整配置/源码/模型hash。
- 正式YAML现为tau=.03、Gaussianσ=.12；新reward Optuna正在运行，最终取值尚未批准，不引用旧奖励Trial18作为替代。
- 记录实际Python/SB3/Zoo/PyTorch版本、硬件、CUDA信息、sampler种子与每trial/每正式run模型种子。不能把命令seed与实际model_seed混同。
- 保留原始处理元数据：源负载为FC/LB遥测重构、30s标称与插值到1s；Train35/Validation8/Test4按parent隔离。
- 既有随机Train35模式已经实现；固定`parent_005_seg01`选择器尚未实现。先由另行授权的开发任务实现/确认，再称为固定航段500Episode实验。
- 500指终止Episode计数，其中可能含物理失败。正常全部完成的固定9258s航段对应4629000物理秒；20M只是安全上限。

## 2. 图表清单

| 图表 | 所需数据/绘制内容 | 验收要点 |
|---|---|---|
| Fig1 系统结构 | FC/LB/DCbus/聚合load/EMS | 无SC、分堆控制、独立BoP动态等不存在分支 |
| Fig2 η与H2特性 | 现有效率工作簿/实际PCHIP、120MJ/kg | 标明参考100→600映射和源测量边界未定项 |
| Fig3 FC退化示意 | 四项公式/参考功率/寿命归一化 | low/high互斥，trans是ΔkW，启动单次计费 |
| Fig4 LB退化示意 | busP→I432V→F/G→weightedAh→Qlife | 15000为假设而非15000完整循环 |
| Fig5 Actor–Critic | 3→256→256→1 Actor、单Critic、target/replay | 原生DDPG，不画TD3双critic |
| Fig6 流程 | Trainrollout/update及100ep独立Val8 | Test在冻结选择后独立使用 |
| Fig7 训练收敛 | 当前reward的逐ep R和R/N、episode长度、success、env/Adam计数 | 标明平滑窗；失败单独标记；不冒充Guo原代码完全复现 |
| Fig8 功率分配 | load/requestFC/actualFC/signedLB/feasiblebounds/projected | 固定时间基，正常与失败末步区分 |
| Fig9 SOC | 真实SOC、.2/.8hardlimits、.5rewardreference | 同时给initial/final/min/max |
| Fig10 FC/LB退化 | FC四项累计μV、raw/cappedlife；LBraw/weightedAh | 聚合结果不能代表六堆健康均衡 |
| Fig11 成本 | actualH2/FCdeg/LBdeg/signedEq分项 | SOC罚项单列，不计为实际支出，不重复终端补电 |
| Table5 实验协议 | hash/seed/finalparams/训练航段/选择方法 | 每run身份可追溯 |
| Table6 经济比较 | 同航段同约束、完成标记、各项成本、SOC | 失败前缀成本不冒充全航段成本 |

首六幅图目前也保留为明确图位；没有以概念图伪装实验结果。后续可先制图1/3/4/5/6，图2需补工作簿来源说明。

## 3. 第4章各节证据

### 4.1 环境与设置

补实际设备/软件版本、固定与随机训练组、对照控制器、独立seed数、模型与数据hash。参考策略尚未开展统一协议的当前奖励比较，不预填结果行。若增加DP/MPC/ECMS对照，必须另行确认其实现、信息权限、成本口径与约束一致性；本次不开发对照器。

### 4.2 收敛

至少保存每episode累计return、平均步reward、计划/已执行秒、失败原因、Adam次数和criticloss。Guo式平均奖励在完整1s航段为R/持续秒数；SB3`rollout/ep_rew_mean`是累计return的均值，不能直接替代。失败有未执行terminalcall，应明确envcall与物理duration，并避免短失败轨迹形成假性高均值。

报告Gaussianσ是否固定；当前没有epsilon-greedy或σ衰减。需要多独立run才支持稳定性/鲁棒性主张，不凭单条平滑曲线宣称普遍收敛。

### 4.3–4.4 功率与SOC

使用现有轨迹导出字段，不增加环境步。SOC硬边界与.5参考值分开。低负载大FC是否合理，应结合充电SOC收益、实际η、退化成本和投影边界分析，不仅看瞬时FC曲线。正常未执行失败步不能伪造batteryactual或intervalend。

### 4.5 退化及不确定性

保留FC runtime/transient/start分解、原始μV、70mV归一化；保留LBraw/weightedAh及Qlife。15000分母仍需补原始数值依据，建议日后在10000/15000/20000比较成本结论稳健性。效率工作簿与50%replacement假设也需适用性说明。不得把代理寿命模型结果写成台架实测寿命延长。

### 4.6 经济性

同一完整航段、SOC0=.6、FC0=0、同数据/物理边界进行配对。分别报告Cphys与C4；实际氢耗kg与货币成本分开；SOCpenalty不是实际付款。若引入真实补电方案，应重新定义其与signedEq边界，不能两者重复计价。tanh总回报改善不能直接推出CNY总成本下降。

### 4.7 泛化与周期验证

收集100/200/300/400/500episode独立checkpoint的全部8航段结果。评价attempted8与success8不同，只有全成功时才有现有callback定义的完整经济均值。保存失败coverage而不是补0。

新Study当前按平均累计return评分；正式模型选择前需要检查completion、coverage、完整经济cost及SOC，说明最终选择与原始Optuna目标之间的关系。Test4不参与训练/调参/选checkpoint；选择冻结后才独立测试。

## 4. 第5章填写条件

结论仅填写当前reward、新实验身份、匹配航段证据支持的改善。每个百分比注明基线与分母；完成率不同应先披露，不能以早失败省成本。若结果不显著或不优，应据实记录而不预设DDPG胜出。

局限应包含：聚合模型未分堆/分簇、模型寿命归一化的不确定性、未建模温度/容量反馈、部分观测、signedEq非真实cashflow以及奖励饱和。未来个体健康分配不能写成无人研究或现已实现。

## 5. 现有入口的角色（本轮均未运行）

目录根为：

`C:\Users\20883\OneDrive\Desktop\lstm-dqn-mpc\lstm-dqn-mpc\.worktrees\v6-sb3-baseline`

- `run_ddpg_train.py`：现有正式随机Train35入口。没有固定航段选择器，不应现在运行后声称固定航段训练。
- `run_ddpg_validate.py`：现有独立Validation入口；周期验证已由trainingcallback自动调用内部验证模块，手动入口仍用于指定其他model/复核。
- `run_ddpg_test.py`：现有独立Test入口，冻结选择后才用于最终评估。
- `run_ddpg_optuna.py`：现有搜索入口；本次没有启动，也不建议为写作重跑正在进行的Study。

以上是现有代码用途说明，不是本次执行记录；实际文件路径与将来运行参数应在开始实验时再次确认。
