# V4：经验采样、拟合干扰与状态歧义的文献及开源依据

核查时间：2026-10-12～13。分支：`feat/direct-power-mlp-ddqn`；HEAD：`a40afa2c8fd0a967d40d8aa30472b228232fbc0e`，工作树有此前接口与 SOC 边界修复的未提交改动。本轮仅新增本报告；未修改源码、参数、运行中的训练、历史实验或 checkpoint，未训练、未运行测试、未读取 Test。

## 1. 结论与证据边界

有文献支持的候选办法，但三类问题需要分别处理：

| 问题 | 当前项目证据 | 最直接候选 | 不能据此承诺什么 |
| --- | --- | --- | --- |
| 1. 关键经验很少抽到 | 历史最终 Replay 固定抽取 600 批，`_015/97` 三条指定记录抽中 0 次；正常终止 5336 条中 4721 条未在该次抽样出现 | 按 TD 残差随机优先回放，配合正确的重要性采样权重；保留可靠的已观测窗口 | PER 不会生成未执行动作的经验，也不能修复缺失状态信息 |
| 2. 一类拟合改善、其他类别变差 | 冻结目标专训正常终止后，留出 MAE 2.4303→0.05636，但其他类别 guard MAE 0.8939→2.7761 | 代表性混合回放、逐类别留出验证；先测实际更新干扰，再单变量研究步长或表达能力 | 尚未测出本项目的负梯度内积；局部能拟合不代表能共同拟合 |
| 3. 相近状态对应不同后果 | `_036/46` 的 Train 近邻实际目标约 −3.44，Validation 实际后续评价目标约 −11.20；终止／中途近邻亦存在差异 | 确认任务终止语义；补充部署时可观测的阶段信息或过去历史 | 近邻不同目标不等于已证实相同输入的不可解冲突；历史不能恢复未观测未来 |

上述数值来自此前已完成的 [Replay 核查](v4_replay_coverage_td_audit_2026-10-12.md) 和 [冻结目标诊断](v4_ddqn_root_cause_diagnostic_2026-10-12.md)，不是正在运行 B 组的新实验成绩。“600 批未抽到”不能解释为原 100 轮从未学过。

## 2. 此前能源管理文献逐项核对

**证据标记：**正文＝本轮核对本地全文相关章节；摘录＝仅出版社摘要、章节摘录或索引；历史＝核对此前记录，本轮未重获正文。未报告不能等同于确认没有使用。

| 文献 | 可核对方法与位置 | 对问题 1／2／3 的含义 |
| --- | --- | --- |
| [Wu 2020](https://doi.org/10.1016/j.apenergy.2020.115258)，正文 | p5 式(10)：岸电可用性、负荷、FC、SOC；p6 式(16)、附录 Algorithm 2：表格型 Double Q-learning | 无 Replay；没有共享 MLP 干扰，不能把两个 Q 表说成 DQN 抗干扰方法 |
| [Wu 2021](https://discovery.ucl.ac.uk/id/eprint/10135003/1/manuscript_accepted.pdf)，正文 | PDF p4：随机回放、Huber、软目标更新；p6 式(11)：DDQN；p9–11：连续四维输入、每 32 transition 更新 | 稳定训练机制有依据，但未报告 PER、稀少类别配额或梯度干扰修复；连续输入主要减少网格量化损失 |
| [Shang 2022](https://doi.org/10.1016/j.epsr.2022.107885)，正文 | p6 §4.1 随机批次、普通 DQN；p7 §5.1 输入 `[t,P_load,SOC]` | 时间输入是可借鉴的任务上下文实例；其日调度与本项目变长航次不同，不能直接照搬时间口径 |
| [Zhu 2024](https://doi.org/10.1016/j.ijhydene.2024.10.192)，正文 | 期刊 p1262–1263 式(12)–(16)：DQL、回放、`[P_dem,SOC]`、动作限制和多目标奖励 | 未报告优先采样或类别抗干扰；动作限制和奖励权重改变控制任务，不能当作价值拟合修复 |
| [Xiao 2025](https://doi.org/10.1016/j.energy.2025.138782)，正文 | p8 §4.3、Algorithm 1：**DDPG 使用 SumTree PER**，给出采样概率与 IS 权重 | 此前能源管理文献中直接对应问题 1 的实例。算法列出 IS，但展示 critic loss 未显式使用，实际权重实现未核实；其安全屏蔽不属于类别梯度干扰处理 |
| [刘俊峰等 2020](https://doi.org/10.13335/j.1000-3673.pst.2020.0144)，正文 | PDF p5–6：回放、冻结网络、`[t,SOC]`；p8：网络规模及迁移初始化比较 | 时间代表固定外生场景；不是未知航次历史歧义已解决的证明，也不是梯度冲突算法 |
| [Yuan 2025](https://doi.org/10.1016/j.ijhydene.2025.151770)，正文 | p8–9 式(45)–(52)：两个 Q 表调双层 MPC；状态含负荷变化率等 | 负荷变化方向是状态设计参考；不是直接功率 DDQN，也无 Replay 修复依据 |
| [Liu 2025，120081](https://doi.org/10.1016/j.enconman.2025.120081)，正文 | p7–8：MPC＋TD3、普通 mini-batch；p2 优先回放属引用他人工作 | 不应把引言中的 hindsight prioritized replay 算成作者自身算法 |
| [Shuai 2024，1586](https://www.mdpi.com/1996-1073/17/7/1586)，摘录 | DDQL、经验池、SOC／退化奖励；完整算法访问未成功 | 未核实 PER、历史编码或抗类别干扰，不能补写参数 |
| [Wang 2024，132771](https://www.sciencedirect.com/science/article/pii/S0360544224025453)，摘录 | 五类 DRL、奖励权重及超参数比较 | 支持受控参数比较，不足以确认采样或抗干扰实现 |
| [Deng 2024，117921](https://www.sciencedirect.com/science/article/pii/S0196890423012670)，摘录 | 表格 Q-learning；负荷、SOC；实车数据估计转移概率矩阵 | 不存在同等的共享 DQN 网络问题；其统计环境假设不能自动迁移 |
| [Wu 2018](https://doi.org/10.1016/j.apenergy.2018.03.104)，摘录／历史 | DQL 连续状态与 SOC 管理；本轮未取得全文 | 不确认 PER、历史窗口或抗干扰机制 |
| [李卫等 2021](https://jcjs.siat.ac.cn/article/doi/10.12146/j.issn.2095-3135.20210118001)，摘录 | 寿命奖励、动作限制；正文未取得 | 不把功率限制当成 Replay／共享拟合修复 |
| [Mutation protection 2024](https://doi.org/10.1016/j.ijhydene.2024.03.025)，摘录 | 突变量化、信息熵、动态阈值及保护模块；全文未取得 | 未核实其对 Replay 或状态的作用；不能以标题支持加入动作覆盖规则 |
| [Zhu 2026](https://doi.org/10.3390/jmse14050460)，历史 | 此前清单标为 TD3；本轮未重新核实正文 | 不充当 DDQN 三类问题的直接实证 |
| [Zhou 2025 多时间尺度](https://doi.org/10.3390/jmse13010034)，历史 | 此前讨论的规划、寿命协同 | 本轮未重新核实学习流程，不推断 PER 或抗干扰 |
| [Liu／Yang／Zhou 2025 双层](https://doi.org/10.3390/jmse13010094)，历史 | 此前讨论的先验／双层功率管理 | 不将滤波或控制优化等同于 DQN 学习修复 |
| [Jiang 2024](https://doi.org/10.1088/1742-6596/2876/1/012052)，历史 | 负荷预测相关研究 | 不作为当前无预测 DDQN 的采样／干扰证据 |
| [Jiang 2025 三层](https://doi.org/10.1016/j.epsr.2025.111494)，历史 | 预测与三层 EMS | 同上；本轮未重新核实其学习流程 |
| [IEDM-DeePC 2026](https://www.sciencedirect.com/science/article/pii/S2773153726000629)，历史 | 概率区间预测及预测控制 | 不能代替 DDQN 学习机制证据 |
| [Yang／Sun／Zhang 2026](https://doi.org/10.1177/01445987251394041)，历史 | MPC 研究 | 不能直接用于本轮三类拟合问题 |
| [KAN 2024](https://arxiv.org/abs/2404.19756)，历史 | 函数表示架构研究 | 不提供本项目 Replay 覆盖、终止歧义或类别共同拟合的实证；不据此改用 KAN |

覆盖了此前四篇核心船舶论文、用户给出的十五篇研究清单，以及 Xiao、Yuan、刘俊峰、KAN 等已有资料。没有取得全文的条目明确保留证据缺口，不宣称全部全文重新读完。另有 RL 调 MPC 的七篇历史综述，但其未来参考输入、PPO、滤波／优化层不能直接作为本轮 DDQN 修复方案。

## 3. 五个指定 GitHub 仓库的实际实现

下列链接固定 commit；仓库存在某文件不代表正式入口调用它。

| 仓库与固定源码 | 核查结果 | 借鉴边界 |
| --- | --- | --- |
| [FCHEV-EMS / dqn_model.py](https://github.com/sicilyala/FCHEV-EMS/blob/e2c8c9c5d591a5b5b8f7f29954ef206023995a2c/common/dqn_model.py) | `Memory.uniform_sample` 有放回均匀抽样；transition 无 done；目标用 target max，普通 DQN；Q 输出也用 ReLU | 原仓库主要对应 SAC，DQN 为对照。非负 Q 输出不能表示其负成本回报；本项目已用线性输出，不应倒退复制 |
| [FCHEV-ACC-EMS / dqn_model.py](https://github.com/sicilyala/FCHEV-ACC-EMS/blob/5d33752bcbb6227a7bed45b1018c8bb6e443f33f/common/dqn_model.py) | 61 个绝对功率动作；DQN 均匀、单步、无 done；存在 Priority_Replay 文件但 DQN 入口不调用 | 原仓库主要对应 DDPG；与本船同动作数不代表同学习机制；同样存在输出 ReLU 问题 |
| [DRL-Energy-Management / DeepQNetwork_Prius.py](https://github.com/lryz0612/DRL-Energy-Management/blob/307c2ba1cfae8048061793c6d4e37ac00b2a4ac1/DeepQNetwork_Prius.py) | 确实调用 SumTree PER；目标仍普通 DQN；loss 先整批均值再乘 ISWeights，priority 又乘权重 | 支持能源管理已有 PER 使用；其 IS 不是标准逐样本校正，不宜照搬。对象是 Prius 发动机／电池 |
| [Fc_Hybrid... / DQN_TrainningBase.m](https://github.com/XinhaoTsl/Fc_Hybrid_Vehicle_Energy_Management_Strategy/blob/f2ba2e5c95dd9943d9195278bcff83a3452ddd40/HESS_control/DQN_TrainningBase.m) | 状态 v/a/P_h，动作是超级电容 SOC 参考值，`UseDoubleDQN=false` | 不是直接 FC 功率 DDQN；其规划使用剩余行程信息，在线可得性需另证 |
| [HEMS-RL / double_dqn.py](https://github.com/lys-hh/HEMS-RL/blob/69ff4a5fa63c77e238b897dee9a906509a55a85e/model/double_dqn.py) | 有正确 DDQN 在线选择／目标估值、done 和 Huber；均匀单步回放；选动作归一化与回放 raw 状态存在口径不一致 | 家庭储能、多设备动作分支；没有本船类别干扰或历史序列修复证明，不直接迁移输入处理 |

补充实现参考：[Tianshou 优先回放](https://github.com/thu-ml/tianshou/blob/f2402056b03acafeedf2518a5088db55add004ab/tianshou/data/buffer/prio.py) 与 [DQN 更新](https://github.com/thu-ml/tianshou/blob/f2402056b03acafeedf2518a5088db55add004ab/tianshou/algorithm/modelfree/dqn.py)。已核版本的 MSE 分支使用逐样本权重，但 Huber 分支未乘 weight。本项目若接入 PER＋Huber，须明确使用 `reduction='none' → 逐样本乘权重 → mean`，不能完整复制该分支。

## 4. 问题 1：高残差经验更新不足

[Schaul 等 PER，§3.3–3.4](https://arxiv.org/pdf/1511.05952)给出随机优先化与 IS 校正：

\[
p_i=|\delta_i|+\epsilon_p,\quad P(i)=\frac{p_i^\alpha}{\sum_jp_j^\alpha},\qquad w_i=(N P(i))^{-\beta_{IS}}.
\]

其中 `beta_IS` 与 SOC 奖励系数无关。结合本项目 Huber 的候选损失为：

\[
L(\theta)=\frac1B\sum_{i\in batch}\bar w_i\,\operatorname{Huber}_1(Q_\theta(s_i,a_i)-y_i).
\]

本项目的意义：优先级可以覆盖高残差失败**前缀**和正常终止，而不仅是已较好拟合的失败终止。它影响抽样与损失权重，动作仍由合法集合中的 Q argmax 决定。

**兼容性必须先设计：**当前固定失败终止配额 2 已改变基线分布。若保留这两条并只对其余普通池优先化，IS 应对应明确的基线条件分布；不能直接套全池均匀公式。无放回、分层或混合抽样均须核对真实概率。重复 n8 窗口是相关经验，应统计不同来源轨迹覆盖，不把多抽同一最终结果当作新独立证据。

尚未决定 PER 参数、是否保留配额或正式接口；本轮不实施。若一个高残差源持续被重抽但留出误差没有下降，需要转查问题 2／3，不能只继续增其优先级。

## 5. 问题 2：共享网络的拟合副作用

[Fu 等 2019，§5、§7](https://proceedings.mlr.press/v97/fu19a/fu19a.pdf)提供了分离采样、逼近和目标变化的诊断方法，并支持保持较广 Replay 覆盖。它没有证明任意类别等额抽样最优。[Fedus 等 2020](https://proceedings.mlr.press/v119/fedus20a/fedus20a.pdf)区分容量、经验年龄与更新比，说明不能把扩大容量或增加更新视为普适收益。

[Bengio／Pineau／Precup 2020](https://proceedings.mlr.press/v119/bengio20a/bengio20a.pdf)给出梯度关系的诊断依据。本项目可在冻结目标副本中测量：

\[
g_c=\nabla_\theta L_c,\quad \cos(c,d)=\frac{g_c^Tg_d}{\|g_c\|\|g_d\|},
\]

并记录一次实际 Adam 更新后其他类别留出损失的变化。SGD 的一阶关系 `ΔL_d≈−ηg_dᵀg_c` 不能当作 Adam 的精确等式。

最小处理顺序：

1. 用代表性混合经验共同训练，保持普通／失败前缀／正常终止／失败终止覆盖，并分别检查留出误差；不只专训问题类别。
2. 在同一固定池、目标、初始模型和更新预算下，先检验一个采样候选。以前配额 8 已使其他类别恶化，不应直接定稿。
3. 若充分曝光仍明显互相损害，单独比较较小学习率的实际跨类别扰动。较小步长也可能只让学习更慢，不预先选值。
4. 若已排除状态混合且较小步长仍无法共同拟合，再单独研究表示能力；当前局部拟合结果不支持立即扩大网络。

[Riemer 等 MER](https://arxiv.org/abs/1810.11910)有专门降低干扰的方法，但引入元学习优化，不是本项目第一步；PCGrad 面向多任务，本项目 Replay 标签不是自然独立任务。已有文献不意味着应把这些算法同时加进 DDQN。

正常终止目标没有 bootstrap，不能仅靠调整 tau 解释其较大残差。放慢 Target、增大 batch、加更新次数也都有预算和目标推进副作用，必须分别实验。

## 6. 问题 3：状态无法区分后续任务

当前源码 `control.build_state`／`dqn.STATE_FEATURES` 为：

`[SOC, 前FC, 两个FC差分, 当前负荷, 两个负荷差分]`，除 SOC 外除以 600。它包含最近三个负荷点的短期变化，没有任务阶段或较长历史；不能据此断言维度越多越好。

[Pardo 等 2018，§2–3](https://proceedings.mlr.press/v80/pardo18a/pardo18a.pdf)区分真正有限任务与训练采集截断。对本项目：先确认样本终点的业务含义；若部署时存在可信任务计划，可检验计划阶段／计划剩余时间。不能使用归档真实剩余行数、样本 ID 或未来真实负载。仅加已航行时间也不保证识别不同长度任务。

若无可信计划信息，先研究过去负荷／实际动作的历史堆叠，仍可保持 MLP；更复杂候选是 [Hausknecht／Stone 2015 DRQN](https://arxiv.org/pdf/1507.06527)，通过循环网络利用历史观测。循环方案需序列 Replay、隐藏状态初始化及 UNKNOWN／任务边界处理，不是只替换一层网络。历史窗口或 RNN 只能利用已经出现的信息。

**n8 延长的是训练回报窗口，不增加策略的观测记忆；γ=1 保留未来权重，也不补足缺失的信息。**如果不同真实任务在所有可观测历史上都无法区分，策略学到条件平均价值是信息限制；PER 不能让一个输入同时输出两个任务特有的正确值。

## 7. 建议验证顺序，均尚未执行

| 顺序 | 最小动作 | 固定条件／验收 |
| --- | --- | --- |
| 1 | 保存当前 B 的完整结果，确认实际曝光与分类误差 | 不在运行途中替换采样、奖励或状态；不将中期指标当最终结论 |
| 2 | 独立冻结目标副本：当前抽样 vs 一个正确校正的优先采样候选 | 同经验、网络、目标、优化器初始条件、随机种子及 batch×更新预算；按来源轨迹分离留出；检查所有类别，不只总体均值 |
| 3 | 必要时测梯度关系／Adam 一步跨类别损失，再单独比较步长 | 相同抽样次序；不同时调整 tau、batch、奖励和容量；不得将低干扰直接当作好泛化 |
| 4 | 若残差伴随任务混合，检验一个可观测上下文或历史候选 | 在线可得性先确认；固定经济目标与采样；需要新模型，不恢复不兼容 checkpoint |
| 5 | 获批后，仅将通过联合拟合筛查的候选做闭环对照 | 至少报告多 seed；固定完整 Train／Validation 集合的供电率与费用，UNKNOWN 前缀独立统计；检查 SOC、启停、跳变和后期退化；Test 封存 |

所有方法是候选，不保证本船改善。优先级为**覆盖与联合拟合 → 必要的可观测信息 → 经证据支持的单变量训练设置**。不建议先同时提高 SOC 权重、扩大网络、增加更新和引入平滑项。当前经济目标、物理 mask 和自主 Q 排序保持不变。
