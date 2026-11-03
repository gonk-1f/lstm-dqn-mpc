# 基于深度确定性策略梯度的船舶燃料电池–锂电池混合动力系统退化感知能量管理策略研究

*作者姓名及单位：待补充。*

## 摘要

船舶燃料电池–锂电池混合动力系统的经济性能既取决于氢气消耗，也取决于能量源所承受的运行应力。本文面向质子交换膜燃料电池（PEMFC）与锂离子电池耦合系统，建立基于深度确定性策略梯度（DDPG）的能量管理框架。采用系统效率映射、基于能量平衡的电池模型及具有文献依据的退化核算方法，描述600 kW聚合燃料电池和624 kWh锂电池。燃料电池电压损失与运行负荷、功率变化及聚合系统启动相关；电池损耗则采用荷电状态和电流加权的安时吞吐量表示。连续动作Actor输出聚合燃料电池功率请求，通过确定性的物理投影满足燃料电池爬坡、电池功率及荷电状态约束。奖励函数综合实际氢耗支出、燃料电池与电池的增量更换成本、有符号电池等效氢耗项以及二次荷电状态惩罚。实现采用Stable-Baselines3原生DDPG，每100个已终止训练Episode独立进行一次确定性验证。重构运行记录划分为35条训练航段、8条验证航段和4条保留测试航段。在已实现的随机航段训练模式之外，另将固定航段500 Episode实验作为拟开展方案。当前奖励定义下的数值收敛性、退化及经济性比较结果仍有待实验确定。

**关键词：** 船舶混合动力推进；质子交换膜燃料电池；锂离子电池；退化成本；深度确定性策略梯度；能量管理。

## 符号说明

| 符号 | 定义 | 单位 |
|---|---|---|
| $t$, $\Delta t$ | 决策步索引与物理采样间隔 | –, s |
| $P_{L,t}$ | 重构的船载直流母线负载 | kW |
| $P_{F,t}$, $P_{B,t}$ | 实际执行的聚合燃料电池与电池母线功率 | kW |
| $P_{F,t}^{req}$ | Actor请求的燃料电池绝对功率设定值 | kW |
| $E_N$, $V_N$, $Q_N$ | 电池标称能量、标称电压与标称电荷容量 | kWh, V, Ah |
| $z_t$, $z_{ref}$ | 电池SOC及其奖励参考值 | – |
| $\eta_F$, $\eta_c$, $\eta_d$ | 燃料电池系统效率、电池充电效率与放电效率 | – |
| $H_L$ | 氢气低位热值 | kJ $\mathrm{kg}^{-1}$ |
| $\Delta m_{H,t}$ | 单个区间内燃料电池实际氢耗 | kg |
| $U_t$, $W_t$ | 燃料电池累计电压损失与电池累计加权吞吐量 | μV, Ah |
| $D_{F,t}$, $D_{B,t}$ | 燃料电池与电池未经截断的归一化寿命消耗 | – |
| $K_F$, $K_B$ | 聚合燃料电池与电池更换成本 | CNY |
| $C_{H,t}$, $C_{F,t}$, $C_{B,t}$ | 氢耗成本、燃料电池损耗成本与电池损耗成本 | CNY |
| $C_{eq,t}$ | 有符号电池等效氢耗成本 | CNY |
| $C_{SOC,t}$, $q_t$ | SOC惩罚及奖励计算总成本 | CNY |
| $o_t$, $a_t$, $r_t$ | 智能体观测、归一化动作与奖励 | – |
| $\mu_\theta$, $Q_\phi$ | 确定性Actor与动作价值Critic | – |
| $\gamma$, $\tau$, $\sigma$ | 折扣因子、目标网络混合系数与噪声标准差 | – |
| DPG, DDPG | 确定性策略梯度；深度确定性策略梯度 | – |
| DP, ECMS, MPC | 动态规划；等效消耗最小化策略；模型预测控制 | – |
| EOL, FC, LB | 寿命终点；燃料电池；锂离子电池 | – |

## 1. 引言

### 1.1 研究背景与动机

减少航运温室气体排放，需要同时改变能源供给方式和船上能源利用方式。国际海事组织2023年战略提出，国际航运应于2050年或其前后实现温室气体净零排放，并以2008年为基准，设定了2030年年度温室气体排放总量至少减少20%、力争减少30%，以及2040年至少减少70%、力争减少80%的指示性阶段目标[1]。这些目标属于行业层面的愿景，而非对特定船舶性能的保证。对于具体船舶，替代动力方案是否适用，取决于其航线、装机容量、能源基础设施和运行工况。因此，应结合明确的运行任务和物理系统评估能量管理策略，而不应将其表述为普遍适用的脱碳解决方案。

混合电力架构允许具有互补特性的能源共同满足电力需求。Guo等[2]综述了船舶混合动力系统配置、控制目标与智能能量管理之间的关系。对于以氢为燃料的质子交换膜燃料电池–锂电池（PEMFC–LB）系统，燃料电池将储存的氢能转化为电能，而电池能够吸收短时功率失配并满足瞬态需求。电池还可能使燃料电池运行于系统效率曲线上更有利的工作点。然而，这些潜在优势受到电池有限储能、能量转换损失、燃料电池功率限值以及两类能源运行应力的共同约束。因此，系统容量配置和上层控制不能被视为彼此独立的运行成本决定因素；Wu和Bucknall[3]关于渡船建模与多目标优化的研究说明了二者之间的相互作用。

仅凭氢耗无法表征功率分配的全部经济后果。反复改变燃料电池功率的策略，可能降低瞬时燃料消耗项，却增加表征电压衰减的代理指标。相反，通过大量电池循环来抑制燃料电池波动，也可能将寿命损耗转移至电池。此外，若不计入由放电造成的SOC亏损，较为激进的放电策略还可能通过消耗初始储存的电能，形成表面上的节省。因此，应分别报告实际燃料支出、能源设备退化和储能变化，并明确各成本项的核算边界。对于被称为退化感知的控制器，应说明其考虑了哪些退化效应、如何将这些效应货币化，以及哪些物理老化过程尚未纳入研究范围。

本文研究一个等效燃料电池系统与一个等效锂电池系统之间的聚合功率分配，不对单个燃料电池堆或电池簇的调度进行优化。这一区别十分重要：聚合电压衰减代理指标可以为上层功率请求提供依据，却不能据此确定各堆的启机次数、非均匀老化情况，或特定下层调度算法的经济收益。此类扩展需要额外的状态信息，以及经过独立验证的功率分配模型。

### 1.2 文献综述

基于规则的策略具有决策透明、计算需求较低的优点，因此仍然具有吸引力。功率阈值、SOC区间以及能源优先级规则，无需在线求解大型优化问题即可实现。不过，其有效性取决于所选阈值以及设计这些阈值时所针对的运行条件。在一条航段中能够维持SOC的规则，面对另一种负载曲线时，可能导致持续充电或电池能量耗尽。因此，相关综述和船用燃料电池研究通常将基于规则的方法与基于优化、基于学习的方法并列讨论，而不将某一类方法视为在所有情况下均更优[2,3,4,5]。

等效消耗最小化方法将储能装置的电能贡献转换为燃料等效量，并最小化瞬时目标。该方法在燃料消耗与电池能量之间建立了具有物理可解释性的联系，但需要适当的等效因子，以及对效率和终端SOC的一致处理。当完整需求序列和系统模型已知时，动态规划能够提供有价值的有限时域参考；然而，其计算与信息需求限制了它直接作为不受这些条件约束的在线控制器使用。模型预测控制在滚动时域内重复求解优化问题，并能够显式施加运行约束，但其性能取决于需求预测、模型精度、时域选择以及计算可行性。Zhou等[5]将时间尺度分析与能源设备寿命核算相结合，说明了同时考虑燃料电池与电池损耗的必要性。这些方法为评价学习得到的策略提供了有价值的参考原则，但公平的数值比较必须采用相同的需求、初始条件、约束和成本边界。

具体的船舶研究还表明，不同类别的方法可以相互结合。Bassam等[6]为燃料电池客船开发了多方案控制器，在基于状态、等效消耗、电量消耗/电量保持以及比例–积分等方案之间进行选择。Kalikatzarakis等[7]采用ECMS和混合整数优化构建了考虑岸电充电的船舶能量管理问题，并以动态规划作为基准。针对配置电池/飞轮储能的全电船舶，Hou等[8]利用动态规划研究性能权衡，并开发了引入SOC参考值的MPC，以应对较短预测时域的局限。这些研究提供了有价值的控制先例，但其能源配置和优化边界与本文的燃料电池–锂电池系统不同，不能将其已发表的性能变化直接移用于本研究。

与上述方法不同，强化学习通过与环境交互来调整控制策略。Wu等[4]利用双Q学习研究了具有成本效益的船舶燃料电池–电池能量管理，其后续研究[9]则采用双深度Q网络，实现插电式混合动力推进系统的近优能量管理。两者存在实质性区别：表格型双Q学习并非深度神经网络实现，双DQN也不等同于DDPG。以Mnih等[10]的奠基性研究为代表，深度Q学习针对预先给定的离散决策集合估计动作价值。离散化提供了便于计算的动作选择过程，但功率等级的数量及间距决定了可用的控制分辨率。扩大该集合可能增加动作价值近似和探索的负担。

Zhu等[11]针对燃料电池混合动力船舶提出了深度Q学习策略，同时考虑航段成本、燃料电池寿命和电池SOC等多个目标。该研究的相关性在于，其将燃料电池寿命指标转化为经济后果，而非仅将其保留为功率平滑指标。Jung和Chang[12]研究了面向液氢燃料混合电力推进系统的DQN策略，并考虑了燃料供给和辅助系统。该文随后发表了勘误[13]，因此对其数值表格的解释应以修正后的记录为准。其离散动作DQN及显式辅助系统模型，不应与本文采用的聚合DDPG系统混淆。随后，Wu等[14]在智能船舶混合电力能量管理框架中采用双延迟DDPG（TD3），并实现燃料电池簇控制。这表明，连续动作Actor–Critic方法已应用于船舶领域；因此，本文的创新性不能合理地建立在“首次在该领域使用连续动作Actor”这一主张之上。

DDPG将确定性策略梯度理论[15]与神经网络Actor–Critic近似、经验回放及缓慢更新的目标网络相结合[16]。Actor直接生成连续指令，从而无需预先设定有限的燃料电池功率网格；Critic对该指令进行评价，并提供用于改进Actor的局部动作梯度。在相关交通运输应用中，Wu等[17]针对串并联插电式混合动力电动客车，研究了引入交通信息的连续控制能量管理。这支持了连续控制在方法上的相关性，但其车辆拓扑和可用信息仍与船舶燃料电池–锂电池系统不同。

Guo等[18]将DDPG应用于一项多能源邮轮案例，其动作包括四台柴油发电机的功率指令。该研究的Actor–Critic表述、经验回放机制和目标网络处理，为应用层面的方法设计提供了有价值的参考。然而，其基于燃料成本和SOC的奖励对应另一种动力系统配置，并不构成本文氢耗与能源设备退化核算的定义。尤其需要指出，借用奖励变换的数学形式，并不足以支持将柴油燃料成本模型、SOC系数或发电机动作映射直接移植至燃料电池系统。同样，当航段持续时间不同时，其报告的平均奖励曲线也必须与原始Episode回报区分开来。

表1总结了与本文最相关的算法差异，旨在明确方法之间的联系，而非利用不同船舶、数据集和核算假设下获得的结果对文献进行排名。

**表 1. 船舶及连续控制领域的部分强化学习研究。**

| 研究 | 算法及应用 | 相关性与边界 |
|---|---|---|
| Wu等[4] | 双Q学习；船用插电式燃料电池–电池系统 | 经济性能导向的能量管理；并非双DQN |
| Wu等[9] | 双DQN；船用插电式推进系统 | 离散动作深度强化学习；并非DDPG |
| Zhu等[11] | 深度Q学习；燃料电池混合动力船舶 | 航段成本、燃料电池寿命和SOC目标 |
| Jung和Chang[12,13] | DQN；液氢混合动力推进 | 采用修正后的文章；辅助系统模型不同 |
| Wu等[14] | TD3；船舶混合电力推进 | 连续控制及簇级功率分配 |
| Wu等[17] | 连续控制深度强化学习；混合动力电动客车 | 相关控制理论；车辆及拓扑不同 |
| Guo等[18] | DDPG；多能源邮轮 | 四个柴油机功率动作；燃料模型不同 |
| 本文框架 | 原生DDPG；船舶聚合燃料电池–锂电池系统 | 联合成本核算与显式物理投影；性能待验证 |

### 1.3 研究空缺与贡献

现有文献已经表明，强化学习能够用于构建船舶能量管理问题，但这并未消除对透明建模和可复现比较的需求。三个实际问题构成了本文的研究动机。首先，聚合连续功率分配必须在能源设备、爬坡和SOC约束同时作用时保持可行。其次，退化惩罚与货币成本之间需要建立可追溯的联系：电压衰减比例、更换成本比例与完整设备资本支出（CAPEX）是不同的量。最后，必须区分电池能量估值与能源设备退化，避免将未报告的电池能量消耗误认为燃料节省。已有研究已涉及其中部分问题，因此，本文将研究空缺界定为具体的实现与评估问题，而非笼统声称以往从未考虑联合退化。

本文已实现的贡献之一，是构建聚合燃料电池–锂电池环境，将连续DDPG功率请求通过具有明确物理定义的可行功率投影进行处理。在统一的人民币（CNY）核算约定下，明确区分氢气消耗、燃料电池电压衰减成本、锂电池吞吐量成本以及带符号的电池等效氢耗。第二项已实现的贡献，是基于Episode计数的检查点保存和独立Validation8流程，从而将模型推理与训练更新分离。计划开展的实证研究，将在当前奖励定义下，分别采用固定训练航段和现有随机航段模式，对经济与退化结果进行受控比较。有关成本、耐久性、收敛性或泛化性能改善的证据，将由第4章的数值研究给出。

## 2. 船舶混合动力系统建模

### 2.1 混合动力系统配置

本仿真采用聚合式质子交换膜燃料电池（PEMFC）系统和聚合式锂电池（LB）系统，并通过公共直流母线上的功率平衡将二者连接。选用的燃料电池额定功率为600 kW，电池标称能量为624 kWh，与Yang等[19]报告的容量配置一致；该文所述物理系统包含六台100 kW燃料电池单元。其系统架构还包含超级电容器，而本研究未对超级电容器进行建模。六单元配置为下文采用的参考功率缩放提供了依据，但并不意味着本仿真包含六个可独立控制的电堆。

图1预留用于展示所建模系统能量边界的示意图。燃料电池功率和电池功率均在母线侧定义。变换器功能可用接口模块表示，但当前被控对象模型并未包含独立的变换器开关动态、单独解析的燃料电池辅助功率支路，也未包含分别测量的推进负载和船舶生活负载通道。输入需求为集总船载电负载。因此，示意图不应暗示这些未建模子系统已得到独立仿真。

**图 1. 聚合式燃料电池–锂电池船舶混合动力系统架构。**

*预留图位：聚合式燃料电池系统→直流接口→公共母线；锂电池↔双向接口↔公共母线；母线→集总船载负载。虚线能量管理系统信号表示燃料电池设定值和测得的观测量。接口损耗仅在方程明确规定的位置计入。*

表2列出了当前采用的被控对象模型参数。尽管若干容量和功率限值与Yang等[19]一致，但项目采用的初始SOC为0.6，允许范围为0.2–0.8；这些设置不同于该文中的SOC设置，本文将其明确列为项目运行假设。

**表 2. 当前实现采用的聚合式被控对象模型参数。**

| 参数 | 数值 | 依据 |
|---|---|---|
| 燃料电池聚合额定功率 | 600 kW | Yang等[19]；当前项目被控对象模型 |
| 缩放所用参考燃料电池额定功率 | 100 kW | 聚合功率至参考功率的模型假设 |
| 燃料电池爬坡限值 | 48 kW $\mathrm{s}^{-1}$ | 当前项目约束 |
| 电池标称能量 | 624 kWh | Yang等[19] |
| 电池标称电压 | 432 V | Yang等[19]；固定电流换算基准 |
| 电池母线功率限值 | −624至+1248 kW | 充电为负，放电为正 |
| 电池充电/放电效率 | 0.95 / 0.95 | 当前能量模型；文献[19]亦有报告 |
| 初始/最小/最大SOC | 0.6 / 0.2 / 0.8 | 项目运行假设 |
| 初始燃料电池功率 | 0 kW | 每条航段复位 |
| 决策间隔 | 1 s | 处理后数据与环境的接口约定 |

负载记录由已记录的聚合燃料电池功率和电池功率重构得到。在上游源数据边界，电池放电的符号与当前被控对象模型所采用的符号相反，因此源数据重构过程减去原始电池通道。经插值及后续1 s重采样，得到决策时间网格上的需求。因此，这些输入应表述为重构运行负载记录，而非独立负载仪表直接测得的1 Hz数据。采用文献[19]中的被控对象模型容量，并不能证明这些运行记录采集自同一艘船舶。

### 2.2 燃料电池模型

令$P_{F,t}$表示实际执行的燃料电池母线净功率。其允许范围和单步爬坡约束为

$$0\leq P_{F,t}\leq P_F^N=600,\qquad |P_{F,t}-P_{F,t-1}|\leq 48\Delta t.\tag{1}$$

式（1）中，功率单位为kW，时间单位为s。系统效率函数根据项目提供的燃料电池功率–效率数据，采用保形分段三次Hermite插值多项式构建。若$p=P_{F,t}/6$表示对应的100 kW参考单元功率，则当前映射为

$$\eta_F(P_{F,t})=\mathcal{I}_{PCHIP}(p),\qquad p=P_{F,t}/6.\tag{2}$$

原始数据表的功率范围超过参考额定功率。实现时，首先通过源数据插值函数确定100 kW端点，保留低于该额定功率的支撑点及该端点，再将其映射至600 kW。因此，式（2）是一项基于归一化负荷的缩放假设，而非新测得的600 kW效率图。在将该曲线视为经过验证的船用系统效率图之前，仍须查明所提供工作簿的测量来源及试验条件。

恒功率区间内的氢气消耗量为

$$\Delta m_{H,t}=\begin{cases}\dfrac{P_{F,t}\Delta t}{\eta_F(P_{F,t})H_L},&P_{F,t}>0,\\0,&P_{F,t}=0,\end{cases}\qquad H_L=120000\ {\rm kJ\,kg^{-1}}.\tag{3}$$

由于1 kW等于1 kJ $\mathrm{s}^{-1}$，式（3）直接得到以kg为单位的氢耗量。这一表达式中不应再引入3600因子；换算为能量单位时，等价低位热值为33.3333 kWh $\mathrm{kg}^{-1}$。当前实现在选定的系统边界上处理燃料电池输出和效率，不再扣除单独建模的压缩机、泵或变换器寄生功率。若不调整该边界而直接增加此类损耗项，可能导致损耗重复计入。图2将展示实际插值得到的效率函数及其对应的氢耗特性，并标明源数据点。

**图 2. 燃料电池系统效率图与氢耗特性。**

*预留图位，用于展示根据当前项目效率数据和式（3）生成的曲线；未编造任何效率或氢耗曲线。*

### 2.3 锂离子电池模型

电池采用集总能量平衡表示。正的$P_{B,t}$表示电池向母线供电；负功率表示电池从母线充电。将电池内部能量提取速率定义为

$$P_{cell,t}=\begin{cases}P_{B,t}/\eta_d,&P_{B,t}\geq0,\\\eta_cP_{B,t},&P_{B,t}<0.\end{cases}\tag{4}$$

相应的SOC状态转移为

$$z_{t+1}=z_t-\dfrac{P_{cell,t}\Delta t}{3600E_N},\qquad E_N=624\ {\rm kWh}.\tag{5}$$

式（4）–（5）在各自的能量转换边界上分别计入一次充电和放电效率。当$\eta_c=\eta_d=0.95$时，放电所消耗的储存能量大于母线接收的能量，而充电能量仅有一部分转化为储存能量。运行限值为

$$-624\leq P_{B,t}\leq1248,\qquad 0.2\leq z_t\leq0.8,\qquad z_0=0.6.\tag{6}$$

式（6）规定了硬性运行区间，其与后文引入的SOC奖励参考值相互独立。本模型并非Rint模型或电化学电池模型：未动态解析开路电压、极化、内阻、热演化及容量衰减反馈。第2.6节引入的标称电压仅用于通量核算换算，并非电压状态方程。每个1 s区间内，实际执行的燃料电池功率和电池功率保持恒定。因此，该模型面向上层能量分配，而非亚秒级母线调节。

### 2.4 功率平衡与系统约束

直流母线功率平衡为

$$P_{L,t}=P_{F,t}+P_{B,t}.\tag{7}$$

电池提供式（7）中的剩余功率。航行过程中不存在岸电供能，也不包含超级电容器项。电池能量转换损耗已在式（4）中计入，式（7）不再额外应用具有独立参数的变换器效率。

燃料电池请求功率既须满足电池可用能量约束，也须满足额定参数限值。对于状态$z_t$，将电池允许功率区间定义为

$$\begin{aligned}B_t^- &=\max\!\left[-624,-\dfrac{(0.8-z_t)3600E_N}{\eta_c\Delta t}\right],\\B_t^+&=\min\!\left[1248,\dfrac{(z_t-0.2)3600E_N\eta_d}{\Delta t}\right].\end{aligned}\tag{8}$$

对应的燃料电池功率区间为

$$\begin{aligned}F_t^- &=\max(0,P_{F,t-1}-48\Delta t,P_{L,t}-B_t^+),\\F_t^+&=\min(600,P_{F,t-1}+48\Delta t,P_{L,t}-B_t^-).\end{aligned}\tag{9}$$

式（8）–（9）给出电池功率、SOC、燃料电池额定功率及燃料电池爬坡限制的交集。当该区间非空时，将请求设定值投影至该区间。若超出实现所采用的数值比较容差后交集仍为空，则在已建模约束下不存在物理可行动作。这会导致真实任务失败，而非针对原本可行控制动作施加奖励惩罚。对于此类未成功执行的尝试步，物理账本不向前推进。

### 2.5 燃料电池退化模型

PEMFC退化取决于运行轨迹及仅凭氢耗无法表征的退化机制。开路或低负荷运行可能使电极处于高电位，从而加速催化剂或膜的退化；高负荷运行可能加剧反应物传输和水管理应力。动态负荷变化可能引发电位循环、反应物饥饿以及温度或湿度条件变化。启停事件可能形成局部氢气–空气界面，并造成严重的阴极腐蚀。Ren等[20]以及Schmittinger和Vahidi[21]的综述阐述了这些机制，但不应将其视为本文所用全部数值系数的直接试验标定依据。

本研究的核算采用Zhou等[5]讨论的四工况电压损失结构，项目同时将数值来源记录为Saponaro等[22]的沿海船舶分析。当前实现并未采用Yang等[19]使用的电化学活性表面积（ECSA）模型。将参考单元功率和聚合系统启动指示量定义为

$$p_t=P_{F,t}/6,\qquad \delta_t^{start}=\mathbf{1}\{P_{F,t-1}\leq0<P_{F,t}\}.\tag{10}$$

采用式（10）中的参考功率和启动指示量，实际区间电压损失计算为

$$\Delta U_t=\left[d_1\mathbf{1}_{0<p_t<80}+d_2\mathbf{1}_{p_t\geq80}\right]\dfrac{\Delta t}{3600}+d_3|p_t-p_{t-1}|+d_4\delta_t^{start}.\tag{11}$$

式（11）中，$d_1=10.17$ μV $\mathrm{h}^{-1}$，$d_2=11.74$ μV $\mathrm{h}^{-1}$，$d_3=0.0441$ μV $\mathrm{kW}^{-1}$，$d_4=23.91$ μV/计数循环。两个运行时间分支相互排斥：代码对所有低于额定功率80%的正功率运行点采用较低退化率，对达到或高于80%的运行点采用较高退化率。高负荷时不会将两项退化率叠加。这一覆盖范围较宽的低负荷分支是实现中的简化，不应解释为经过独立标定的近怠速阈值。

变载量是参考功率绝对变化量的累积值，单位为kW，而非仅为动作变化次数。将未经加权的事件次数与单位为μV $\mathrm{kW}^{-1}$的系数相乘，会造成量纲不一致。功率转入零值或由零值转出的变化均计入该瞬态项。启停贡献在聚合功率由零转为正值的启动时刻计入一次；之后的停机不会再次触发相同的循环费用。这一约定是聚合核算的代理表征，并非对六个独立调度电堆启动次数的重构。

累积电压损失和未经截断的燃料电池归一化寿命消耗分别为

$$U_{t+1}=U_t+\Delta U_t,\qquad D_{F,t}=\dfrac{U_t}{70000}.\tag{12}$$

式（12）的分母为70 mV，数值上等于0.7 V单电池参考电压的10%。该值是固定的文献/模型基础归一化参数，而非根据船舶实测数据得到的剩余寿命估计。进行货币核算时，实现将累计计费寿命限制为一次替换等效值：

$$\Delta\bar D_{F,t}=\min(D_{F,t+1},1)-\min(D_{F,t},1).\tag{13}$$

式（13）防止在累积账本中超过一次替换后仍重复计费。其并未对替换事件建模，也不会在寿命终止（EOL）后恢复被控对象状态。因此，应在保留货币化寿命比例的同时保留原始电压损失诊断量。图3将展示四项贡献、参考功率映射以及独立的寿命归一化步骤。

**图 3. 四工况聚合燃料电池电压损失与增量替换成本模型。**

*预留图位：相互排斥的运行时间分支、参考功率绝对变化量及聚合启动次数共同形成累积电压损失，随后进行70 mV归一化与替换成本映射。*

### 2.6 锂电池退化模型

电池退化采用应力加权电荷通量代理模型描述，其形式遵循Zhou等[5]所用模型。可直接追溯的较早表达见Kwon等[23]。Ghaderi等[24]提供了相关的健康感知多电堆能量管理研究背景，Xu等[25]和Han等[26]则讨论了更广泛的电池退化建模及其机制。这些文献并不能证明，当前系数和寿命归一化分母已针对所建模的船用电池完成试验验证。

为与当前实现保持一致，采用固定标称电压，根据电池母线功率计算带符号的等效电流：

$$\begin{aligned}I_t&=\dfrac{1000P_{B,t}}{V_N},\qquad V_N=432\ {\rm V},\\Q_N&=\dfrac{624000}{432}\ {\rm Ah},\qquad I_N=Q_N/(1\ {\rm h}).\end{aligned}\tag{14}$$

因此，正电流表示放电，负电流表示充电。式（14）采用代码中基于母线功率的通量约定，并非对电池内部电路进行解析后得到的电流。尤其需要指出，该电流并非通过对$P_{cell,t}$再次应用电池效率得到。

SOC应力函数和电流应力函数为

$$F(z_t)=1+3.25(1-z_t)^2,\tag{15}$$

$$G(I_t)=\begin{cases}1+0.45I_t/I_N,&I_t\geq0,\\1+0.55|I_t|/I_N,&I_t<0.\end{cases}\tag{16}$$

对于1 s恒功率区间，加权通量增量为

$$\Delta W_t=F(z_t)G(I_t)|I_t|\dfrac{\Delta t}{3600},\qquad W_{t+1}=W_t+\Delta W_t.\tag{17}$$

式（15）采用状态转移前的SOC。在这一经验函数中，允许区间内的SOC越低，加权系数越大；但不应将该数学性质推广为“高SOC无害”的普遍结论。同样，式（16）使通量惩罚随电流幅值增大，并对充电和放电采用不同系数。根据式（17），即使SOC净变化很小，反复循环仍会增加绝对通量。本模型未包含温度、日历老化、通过雨流计数解析的循环深度、特定化学体系机制以及容量/内阻反馈。

项目采用的寿命归一化与增量计费比例为

$$\begin{aligned}Q_{life}&=15000Q_N,\qquad D_{B,t}=W_t/Q_{life},\\\Delta\bar D_{B,t}&=\min(D_{B,t+1},1)-\min(D_{B,t},1).\end{aligned}\tag{18}$$

式（18）中的15000因子是具有二手文献来源属性的固定项目寿命通量假设。它既不是船用电池实测寿命，也不表示已经证实15000次完整充放电循环。其独立定量标定依据仍未解决；在对绝对寿命或替换成本作出较强结论之前，需要检验结果对该分母的敏感性。图4将区分已发表的加权函数与该项目特定归一化假设。

**图 4. SOC与电流加权的锂电池通量模型及寿命归一化。**

*预留图位：母线功率至电流的换算、动作执行前SOC与带符号电流加权、累积加权Ah通量，以及假定的寿命通量分母。*

### 2.7 经济成本模型

所有区间成本均以人民币元（CNY）计算，不转换为分。物理运行与退化成本为

$$C_{phys,t}=C_{H,t}+C_{F,t}+C_{B,t},\qquad C_{H,t}=c_H\Delta m_{H,t}.\tag{19}$$

式（19）中的氢价为$c_H=21.9$ CNY $\mathrm{kg}^{-1}$，与Yang等[19]报告的参数一致，而非Zhou等[5]报告的35 CNY $\mathrm{kg}^{-1}$。这些价格属于情景参数，不代表当前市场报价。动力源退化成本仅针对本区间新增消耗、并经上限截断的经济寿命比例计费：

$$C_{F,t}=K_F\Delta\bar D_{F,t},\qquad C_{B,t}=K_B\Delta\bar D_{B,t}.\tag{20}$$

式（20）采用的替换价值为

$$\begin{aligned}K_F&=\rho_Fc_FP_F^N=0.5\times3500\times600=1050000\ {\rm CNY},\\K_B&=c_BE_N=2000\times624=1248000\ {\rm CNY}.\end{aligned}\tag{21}$$

式（21）中，燃料电池比例$\rho_F=0.5$表示替换支出占完整设备资本支出（CAPEX）的比例。Mylonopoulos等[27]采用该比例，是因为替换发生退化的燃料电池单电池不一定需要同时替换全部配套设备；其文中引用了Thomas的Hydrogenics演示文稿[28]。后者讨论的是电解槽技术，因此属于间接背景来源，而不能独立证明所有船用燃料电池装置均具有50%的替换比例。项目将该比例作为明确的经济假设采用。该比例仅在$K_F$中应用一次，并不表示物理退化率降低。Mylonopoulos等采用的15%电压损失替换假设，与本文采用的10%/70 mV归一化相互独立；两种阈值并未合并。

为计入电池储存能量的变化，奖励函数采用带符号的等效氢耗项。将参考燃料电池效率定义为

$$\eta_{eq,t}=\begin{cases}\eta_F(P_{F,t}),&P_{F,t}>0,\\\max_{0<P\leq600}\eta_F(P),&P_{F,t}=0.\end{cases}\tag{22}$$

式（22）中的停机状态参考效率由现有正功率效率曲线计算得到，当前曲线最大值为0.638554。这是已获认可的项目等效换算约定，既不是零功率效率，也不是单独指定的平均效率。电池等效成本为

$$C_{eq,t}=c_H\dfrac{(z_t-z_{t+1})E_N}{\eta_{eq,t}(H_L/3600)}.\tag{23}$$

式（23）在放电时为正，在充电时为负。SOC变化已通过式（4）–（5）包含电池能量转换损耗；式（23）不再额外乘入充放电效率或DC/DC损耗。等效能量价值并非额外测得的购氢支出。因此，在经济报告中应将其与$C_{phys,t}$分别列示。奖励函数采用的四项成本之和为

$$C_{4,t}=C_{H,t}+C_{F,t}+C_{B,t}+C_{eq,t}.\tag{24}$$

由于$\eta_{eq,t}$随运行功率变化，式（23）的累加结果通常不能通过逐项抵消化简为单一的终端SOC修正。不应将其表述为精确的全寿命现金流模型，也不应将其再次叠加到单独的终端补电费用中。比较时必须分别报告实际氢耗、各项退化成本、带符号的等效项以及初始/最终SOC。表3汇总了经济假设及其作用。

**表 3. 货币参数与核算边界。**

| 参数或成本项 | 当前数值或定义 | 含义 |
|---|---|---|
| 氢气价格 | 21.9 CNY $\mathrm{kg}^{-1}$ | 项目情景；不同于Zhou采用的35 |
| 完整燃料电池设备CAPEX系数 | 3500 CNY $\mathrm{kW}^{-1}$ | 设备成本基准 |
| 燃料电池替换比例 | 0.5 | 仅应用一次，并非退化率因子 |
| 电池替换成本系数 | 2000 CNY $\mathrm{kWh}^{-1}$ | 聚合电池替换成本基准 |
| 燃料电池计费寿命分母 | 70000 μV | 项目采用的10%单电池参考电压约定 |
| 锂电池寿命分母 | $15000Q_N$ | 需要进行敏感性分析的项目假设 |
| 电池等效项 | 带符号的式（23） | 储存能量估值，并非实际购氢支出 |
| 航行期间的岸电 | 无 | 不存在航行岸电成本 |

## 3. 所提能量管理策略

### 3.1 马尔可夫决策过程建模

有限时域能量管理问题可表示为

$$\mathcal{M}=(\mathcal{S},\mathcal{A},\mathcal{P},\mathcal{R},\gamma),\qquad J(\pi)=\mathbb{E}_{\pi}\!\left[\sum_{t=0}^{T-1}\gamma^t r_t\right].\tag{25}$$

在式（25）中，$\mathcal{S}$ 表示包含充分信息的动力系统与航段状态，$\mathcal{A}$ 为允许的指令域，$\mathcal{P}$ 为包括需求演化和物理投影在内的状态转移规律，$\mathcal{R}$ 为奖励函数。当前实现向 Actor 提供三分量观测，而非完整的底层状态。该观测不包含航段标识、已运行时间、未来功率需求及累计老化状态。重构的需求轨迹为外生输入，相同的当前负载观测不一定对应相同的未来负载。因此，当前实现是基于观测对该马尔可夫决策过程的近似，可能具有部分可观测性；不能仅因将观测称为状态向量，就认定其满足完整的马尔可夫充分性。

每个 Episode 从一条完整航段的首个可执行区间开始，SOC 重置为 0.6，FC 功率重置为零，各项核算累加器清零。执行完最后一个可执行区间后，航段正常完成。可行功率交集为空时，Episode 因物理失败而终止。上述情况属于真实的环境终止，而因外部训练预算耗尽而中断，并不表示航段已经完成。该区别同时用于自举目标的终止掩码和性能报告。

### 3.2 DDPG 算法

DDPG 采用确定性 Actor $\mu_\theta$ 和标量 Critic $Q_\phi$，并设置对应的目标网络 $\mu_{\theta'}$ 和 $Q_{\phi'}$ [15,16]。Actor 将可用观测映射为动作。在学习过程中，探索通过在归一化动作空间中加入噪声实现：

$$a_t=\operatorname{clip}\!\left(\mu_\theta(o_t)+\epsilon_t,-1,1\right),\qquad \epsilon_t\sim\mathcal{N}(0,\sigma^2).\tag{26}$$

式（26）描述预热阶段之后的训练策略。在达到配置的学习启动阈值之前，SB3 采样随机动作；确定性验证时则禁用噪声项。当前固定配置中的动作噪声标准差保持不变，不采用 DQN 式的 epsilon-greedy 概率，也没有自动衰减的贪婪度参数。

经验回放缓冲区存储转移元组 $(o_i,a_i,r_i,o_{i+1},d_i)$。对于大小为 $N$ 的小批量样本，单步目标值为

$$y_i=r_i+\gamma(1-d_i)Q_{\phi'}\!\left(o_{i+1},\mu_{\theta'}(o_{i+1})\right).\tag{27}$$

其中，任务真实终止时，包括发生物理失败时，$d_i=1$，因此这些转移不进行自举。Critic 通过最小化以下损失进行更新：

$$L_Q(\phi)=\dfrac{1}{N}\sum_{i=1}^{N}\left[Q_\phi(o_i,a_i)-y_i\right]^2.\tag{28}$$

对于 Actor，确定性策略梯度理论给出以下基于采样的梯度估计 [15]：

$$\nabla_\theta J\approx\dfrac{1}{N}\sum_{i=1}^{N}\left.\nabla_aQ_\phi(o_i,a)\right|_{a=\mu_\theta(o_i)}\nabla_\theta\mu_\theta(o_i).\tag{29}$$

在实际实现中，与上述梯度等价的 Actor 损失为：Critic 对 Actor 生成动作所输出价值的负均值。两个网络均采用 Adam 优化器。经验回放允许以离策略方式重复利用历史转移；目标网络则降低自举目标的变化速度，但在采用非线性函数近似时，两种机制均不能保证算法收敛。每次更新后，目标网络参数按以下规则更新：

$$\theta'\leftarrow\tau\theta+(1-\tau)\theta',\qquad \phi'\leftarrow\tau\phi+(1-\tau)\phi'.\tag{30}$$

本研究使用 SB3 原生 DDPG [29]。尽管其与 TD3 共用部分软件组件，本实验仍不是 TD3 控制器：当前实现仅使用一个 Critic，每次梯度更新均更新 Actor，且不采用 TD3 的目标策略平滑或双 Critic 最小值机制。Fujimoto 等 [30] 讨论了 TD3 的这些方法特征。Actor 和 Critic 均包含两个具有 256 个单元的 ReLU 隐藏层；Actor 末端通过 tanh 输出一个归一化指令，Critic 输出一个动作价值。图 5 将展示上述四个网络及式（27）—（30）的更新路径。

**图 5. DDPG 的 Actor–Critic、经验回放及目标网络结构。**

*图位预留：三个观测输入、具有 256–256 隐藏层的 Actor、单一动作输出、具有 256–256 隐藏层的观测–动作 Critic，以及采用软更新的目标网络。不包含双 Critic 或 TD3 特有路径。*

### 3.3 观测空间与连续动作空间

实际使用的三维观测为

$$o_t=\left[z_t,\ P_{L,t}/600,\ P_{F,t-1}/600\right]^{\mathsf{T}}.\tag{31}$$

由于负载可以由 FC 和电池共同供给，式（31）的第二个分量可以大于 1，并未按 FC 额定功率截断。第三个分量为上一时刻实际执行的 FC 功率，用于计算下一步爬坡约束。Actor 不接收未来负载预测、剩余航行时间特征、独立电堆健康状态向量或累计退化状态。

标量动作域及其物理映射为

$$a_t\in[-1,1],\qquad P_{F,t}^{req}=300(a_t+1),\qquad P_{F,t}=\operatorname{clip}(P_{F,t}^{req},F_t^-,F_t^+).\tag{32}$$

式（32）给出绝对功率指令，而非功率增量。电池提供的剩余功率由式（7）确定。此处采用连续、确定性的执行器投影，不同于另一条 DQN 路线使用的离散动作掩码。SB3 经验回放存储归一化的请求动作，而观测转移和经济账本对应投影后的实际物理动作。因此，Critic 学习的是在环境执行器映射作用下，请求动作所具有的价值。评估时应保留请求功率、实际执行设定值、可行区间及投影标记，以便识别策略的饱和行为。

### 3.4 经济奖励函数

当前奖励版本标识为 `ddpg_economic_tanh_v1`。SOC 惩罚基于下一物理状态计算：

$$C_{SOC,t}=\lambda_z(z_{t+1}-0.5)^2,\qquad \lambda_z=17,\qquad q_t=C_{4,t}+C_{SOC,t}.\tag{33}$$

式（33）既不对 SOC 偏差进行区间归一化，也不额外乘以一秒时间间隔。四项经济成本采用式（24）的定义，不以未定价的 FC 爬坡惩罚替代。FC 功率变化通过式（11）的退化核算影响奖励。原有独立爬坡惩罚已从当前目标中移除。

对于正常执行区间和物理不可行的尝试转移，奖励分别定义为

$$r_t=\begin{cases}\tanh(1/q_t),&q_t>0\ \text{and physical step executed},\\1,&q_t=0\ \text{and physical step executed},\\-1,&\text{true physical failure}.\end{cases}\tag{34}$$

式（34）中的成本采用固定货币约定，以 CNY（元）的数值输入。零成本分支取正成本表达式在零点的右极限。四项成本之和或奖励总成本出现负值或非有限值时，程序明确报错，不取绝对值、不静默截断，也不将其解释为新增的物理失败条件。在物理不可行的尝试转移中，现有环境返回失败奖励和终止标志，不执行实际区间，也不推进物理成本账本。该失败信号属于训练奖励，不是负的货币支出。正常完成时，不增加额外终止奖励或终端 SOC 恢复费用。

对于正成本，式（34）在数学上严格介于 0 和 1 之间，并随成本增加而减小。成本为很小的正数时，浮点饱和可能使计算结果在数值上等于 1。该变换对单个区间成本保持单调性，但最大化变换后折扣奖励之和，并不在数学上等价于最小化未折扣的累计 CNY 成本。它还可能压缩低成本决策之间的差异。因此，仅凭奖励收敛不能证明经济优势；必须直接评估经济成本分项和任务完成情况。

SOC 系数具有明确的、仅基于训练集的标定记录。采用已有确定性基线策略运行 Train35，共获得 448434 个实际执行的物理秒，其中两条航段在完整时长之前终止。在这些已执行区间上，四项经济成本之和的中位数为

$$\begin{aligned}C_{typ}&=0.17429534153424198\ {\rm CNY},\\\lambda_z^{cal}&=\dfrac{C_{typ}}{0.1^2}=17.429534153424198.\end{aligned}\tag{35}$$

随后确认采用的实际系数为 17，而非式（35）的精确标定值。当 SOC 偏差为 0.1 时，其惩罚为 0.17 CNY。此次标定不能证明当前策略的性能，未使用 Validation8 或 Test4 样本，且排除了未执行的航段后缀和失败尝试区间。基线模型在原奖励下的表现，不作为式（34）下策略收敛的证据。

### 3.5 训练、验证与测试流程

本实现采用 RL Baselines3 Zoo 训练框架和 SB3 原生 DDPG 算法，仅运行一个环境，当前配置不使用 VecNormalize 包装器。因此，式（31）中的项目观测缩放不同于通过运行统计量学习得到的归一化。表 4 列出正式 YAML 基线参数。其中目标网络混合系数和噪声标准差为当前固定值，均不作为新奖励下已经确定的最优值。

**表 4. 当前正式 DDPG 基线参数与训练预算。**

| 参数设置 | 当前数值 | 状态 |
|---|---|---|
| 折扣因子 $\gamma$ | 0.999 | 固定基线 |
| Actor/Critic 学习率 | 0.001 | Adam |
| 小批量大小 | 64 | 固定基线 |
| 学习启动阈值 | 5000 个环境步 | 随机动作预热 |
| 数据采集/梯度更新节奏 | 32 个环境步/1 个梯度步 | 单环境 |
| 经验回放容量 | 1000000 条转移 | 固定容量 |
| 隐藏层 | 256、256 | Actor 和 Critic 均采用 ReLU |
| 目标网络混合系数 $\tau$ | 0.03 | YAML 基线；最终选择待定 |
| 高斯动作噪声 $\sigma$ | 0.12 | 归一化动作单位；最终选择待定 |
| 训练设备 | CUDA | 实际硬件/软件信息待报告 |
| Episode 上限 | 500 个已终止 Episode | 原生 Episode 计数回调 |
| 环境步安全上限 | 20000000 | 独立上界 |
| 独立验证 | 每 100 个已终止 Episode | 全部八条验证航段 |
| 正式启动器随机种子 | 42 | 各 Trial 可复现性需另行核实 |

数据集包含 35 条训练航段、8 条验证航段和 4 条测试航段。划分以源父记录为组，同一父记录中的不同片段不会跨越上述三个子集。一秒 V6 训练/验证数据包保留已有处理后的航段边界，不对上游数据重新插值。每个可执行区间采用左端点负载，最后的边界节点不产生额外物理步。由此避免将完整航段替换为任意训练窗口。策略拟合、SOC 权重标定和超参数选择均不需要读取测试集负载。

当前训练模式在每次 reset 时，从 35 条训练航段中等概率、有放回地抽取一条航段。拟开展的受控实验则将在一条指定训练航段上重复运行 500 个 Episode，其内部标识为 `parent_005_seg01`。该航段包含 9258 个可执行的一秒区间。固定航段选择器属于计划扩展，尚不是当前环境的有效功能。如果 500 个 Episode 全部完成，该设计将包含 4629000 个物理训练区间；提前失败会改变实际执行数量。现有 20000000 步上限仅为安全预算，不应作为 500 个 Episode 的等价定义。随机航段模式将予以保留，用于单独开展面向泛化能力的比较。

每当环境 Episode 终止，计数器均递增，包括发生物理失败的情况；因此达到计数上限并不意味着成功完成了 500 条完整航段。在计数达到 100、200、300、400 和 500 时，回调保存模型检查点，并按固定顺序对全部八条验证航段启动独立确定性评估。按照当前 SB3 的回调位置，模型快照保存在触发转移写入经验回放之前，也在后续采集批次的梯度更新之前；该时序记录于检查点元数据。独立评估进程在 CPU 上加载快照，禁用探索噪声，调用确定性预测，不进行学习或经验回放更新。周期验证机制不包含 Test4。图 6 将明确展示上述隔离关系。

**图 6. 训练–环境交互及按 Episode 触发的独立 Validation8 工作流程。**

*图位预留：仅使用训练集的轨迹采集及经验回放/更新循环；按 Episode 计数保存检查点；独立的确定性 Validation8 进程；保留的 Test4 仅在模型选择完成后使用。*

当前新奖励 Optuna Study 与历史 Study 相互独立。在本次检查的快照中，Study 同时包含已完成和进行中的 Trial，但尚未形成最终模型选择决策。其当前目标为 Zoo 的验证 Episode 累计回报均值，而非按完成率和经济成本进行字典序排序的目标。必须报告该限制：较高的目标值本身不能证明完整航段成本最低。最终选择应结合完成率、覆盖率、独立经济成本分项及 SOC 进行审查；在该选择冻结之前，应保留 Test4 不用于选择。搜索空间边界、采样器设置、各 Trial 模型随机种子、软件版本及最终所选参数，仍属于有待完善的实验报告内容；不能用旧奖励版本的 Trial 18 替代这些证据。

为使学习曲线具有可比性，需要区分 Episode 累计回报与按时长归一化的平均奖励：

$$R_k=\sum_{t=0}^{N_k-1}r_{k,t},\qquad \bar r_k=R_k/N_k.\tag{36}$$

其中，$N_k$ 为实际返回的训练转移数量。对于完整执行的一秒航段，该数量在数值上等于以秒计的执行时长，因此式（36）与 Guo 等 [18] 所述的“将累计奖励按航行工况时长求平均”一致。失败 Episode 可能包含一个未执行物理动作的终止转移，必须同时标记其物理覆盖率；不能将其较短的累计回报解释为可直接比较的完整航段性能。应明确说明所有平滑窗口，以及平均值是逐 Episode 计算，还是按时间步汇总后计算。固定航段实验能够消除 Episode 长度变化的一个来源，但不能消除探索和策略更新引起的波动。

独立 Validation8 及最终 Test4 报告将包含完成状态、物理时长覆盖率、累计奖励与平均奖励、实际氢耗、FC 与 LB 退化成本、有符号等效氢耗、四项总成本以及初始/最终 SOC。时长覆盖率及评估集完成率定义为

$$\kappa_j=\dfrac{N_j^{phys}}{T_j^{planned}},\qquad CR=\dfrac{1}{M}\sum_{j=1}^{M}\mathbf{1}\{j\ \text{completed}\}.\tag{37}$$

在式（37）中，覆盖率的分子和分母均为一秒区间数量，$M$ 为评估集中的航段数：验证集为 8，测试集为 4。失败航段前缀的成本并非完整航段成本。因此，经济比较应在完成航段之间进行配对，披露失败情况，并避免将提前终止导致的低成本作为能效提升的证据。在完成下述实验之前，不作数值比较结论。

## 4. 仿真结果与分析

### 4.1 仿真环境与参数设置

*待补充实际硬件、Python/SB3/Zoo/PyTorch准确版本、源码修订、奖励及配置哈希、最终选定超参数、数据集来源与独立重复运行信息。表2–4中的系统与基线参数属于模型定义，并非实验结果。*

**表 5. 最终实验方案与运行身份。**

| 项目 | 固定航段DDPG | 随机航段DDPG | 对照策略 |
|---|---|---|---|
| 源码修订／模型哈希 | 待补充 | 待补充 | 待补充 |
| 训练航段／随机种子 | 待补充 | 待补充 | 待补充 |
| 最终$\tau$／噪声／更新次数 | 待补充 | 待补充 | 不适用或待补充 |
| 验证检查点选择 | 待补充 | 待补充 | 待补充 |
| 保留集评估状态 | 待补充 | 待补充 | 待补充 |

### 4.2 DDPG训练收敛性分析

*待补充。按式(36)同时展示Episode累计回报与平均奖励，标识失败Episode并说明平滑方式。补充Critic损失及更新次数作为学习诊断，不单独将Critic损失视为策略质量证据。关于鲁棒性的主张需要独立重复运行间的变异性支持。*

**图 7. 当前经济奖励下的训练平均奖励变化。**

*结果图待补充；尚未提供曲线或收敛轮次。*

### 4.3 燃料电池与锂电池功率分配

*待依据确定性评估轨迹补充。在统一时间基准下绘制负载、请求及实际执行的FC功率、有符号LB功率，并披露投影频率和FC爬坡统计。结合实际可行功率区间与SOC轨迹检查低负载充电行为，不能仅凭FC功率判断运行是否不合理。*

**图 8. 代表性完整航段的确定性FC–LB功率分配。**

*结果图待补充。LB功率为正表示放电，为负表示充电。*

### 4.4 电池SOC表现

*待补充。报告初始、最终、最低与最高SOC、相对于0.5奖励参考值的偏离及物理任务完成情况。硬运行边界与奖励参考值须作为不同物理含义的量分别标注。*

**图 9. 电池SOC轨迹与运行边界。**

*结果图待补充；尚不作SOC维持性能结论。*

### 4.5 燃料电池与锂电池退化分析

*待补充。将FC电压损失分解为四种已核算工况，并同时报告未经截断的寿命消耗及经济计费寿命比例。报告LB未加权／加权吞吐量及其对假设寿命分母的敏感性。聚合结果不能证明单堆寿命均衡或通过实测确认的服役寿命延长。*

**图 10. FC退化分项与聚合能量源损耗比较。**

*结果图待补充。相应LB吞吐量及归一化参数敏感性结果将列入配套表格或子图。*

### 4.6 经济成本比较

*待补充。对各控制器采用相同的完整航段及重置条件。分别列示实际氢耗、FC损耗、LB损耗及有符号等效能量估价，将SOC惩罚作为奖励分项，而非实际采购支出。任何百分比变化均须明确其基准、分母及完整航段纳入条件。*

**表 6. 匹配航段的经济性与退化比较。**

| 策略／数据划分 | 完成航段数 | 实际氢耗 | FC损耗成本 | LB损耗成本 | 有符号等效成本 | 四项总成本 | 最终SOC |
|---|---|---|---|---|---|---|---|
| 对照策略 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 |
| 固定航段DDPG | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 |
| 随机航段DDPG | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 |

**图 11. 匹配运行条件下的经济成本分项比较。**

*结果图待补充；尚不作成本节约或最优性结论。*

### 4.7 验证与泛化性能

*待补充五个Episode节点触发的Validation8报告，以及选择冻结后的一次独立Test4评估。展示全部航段结果、完成率与覆盖率、配对完整航段成本及SOC。比较固定与随机航段方案时，不使用Test4调参或选择检查点。解释Optuna平均回报目标与最终经济性解释之间可能存在的差异。*

## 5. 结论

*本章将在当前奖励定义下的实验与独立评估完成后补充，现阶段不作有效性或优越性的定量结论。*

### 5.1 能量管理目标与建模贡献

*待补充：总结聚合连续功率控制器、满足物理可行性的执行器投影，以及明确区分的氢耗、FC损耗、LB损耗、储能等效估价和SOC奖励项。*

### 5.2 数值证据支持的结论

*待补充：仅陈述当前奖励版本、匹配完整航段比较、独立重复运行、Validation8及保留Test4所支持的结论。定量变化只有在核对分母、不确定性与核算边界后才能写入。*

### 5.3 局限性与后续研究

*待补充：讨论效率映射来源、经验退化模型可迁移性、LB寿命归一化假设、部分可观测性、非线性奖励饱和，以及聚合损耗核算与分堆／分簇调度的差别。未来多堆控制需要个体健康状态与经验证的分配层，不属于当前控制器已实现的贡献。*

## 参考文献

[1] International Maritime Organization. 2023 IMO Strategy on Reduction of GHG Emissions from Ships. Resolution MEPC.377(80), adopted 7 July 2023. https://www.imo.org/en/OurWork/Environment/Pages/2023-IMO-Strategy-on-Reduction-of-GHG-Emissions-from-Ships.aspx.

[2] Guo X, Lang X, Yuan Y, Tong L, Shen B, Long T, Mao W. Energy management system for hybrid ship: Status and perspectives. Ocean Engineering 2024;310:118638. https://doi.org/10.1016/j.oceaneng.2024.118638.

[3] Wu P, Bucknall R. Hybrid fuel cell and battery propulsion system modelling and multi-objective optimisation for a coastal ferry. International Journal of Hydrogen Energy 2020;45(4):3193–3208. https://doi.org/10.1016/j.ijhydene.2019.11.152.

[4] Wu P, Partridge J, Bucknall R. Cost-effective reinforcement learning energy management for plug-in hybrid fuel cell and battery ships. Applied Energy 2020;275:115258. https://doi.org/10.1016/j.apenergy.2020.115258.

[5] Zhou X, Yang X, Zhou M, Liu L, Niu S, Zhou C, Wang Y. Multi-Temporal Energy Management Strategy for Fuel Cell Ships Considering Power Source Lifespan Decay Synergy. Journal of Marine Science and Engineering 2025;13(1):34. https://doi.org/10.3390/jmse13010034.

[6] Bassam AM, Phillips AB, Turnock SR, Wilson PA. Development of a multi-scheme energy management strategy for a hybrid fuel cell driven passenger ship. International Journal of Hydrogen Energy 2017;42(1):623–635. https://doi.org/10.1016/j.ijhydene.2016.08.209.

[7] Kalikatzarakis M, Geertsma RD, Boonen EJ, Visser K, Negenborn RR. Ship energy management for hybrid propulsion and power supply with shore charging. Control Engineering Practice 2018;76:133–154. https://doi.org/10.1016/j.conengprac.2018.04.009.

[8] Hou J, Sun J, Hofmann H. Control development and performance evaluation for battery/flywheel hybrid energy storage solutions to mitigate load fluctuations in all-electric ship propulsion systems. Applied Energy 2018;212:919–930. https://doi.org/10.1016/j.apenergy.2017.12.098.

[9] Wu P, Partridge J, Anderlini E, Liu Y, Bucknall R. Near-optimal energy management for plug-in hybrid fuel cell and battery propulsion using deep reinforcement learning. International Journal of Hydrogen Energy 2021;46(80):40022–40040. https://doi.org/10.1016/j.ijhydene.2021.09.196.

[10] Mnih V, Kavukcuoglu K, Silver D, et al. Human-level control through deep reinforcement learning. Nature 2015;518(7540):529–533. https://doi.org/10.1038/nature14236.

[11] Zhu L, Liu Y, Zeng Y, Guo H, Ma K, Liu S, Zhang Q. Energy management strategy for fuel cell hybrid ships based on deep reinforcement learning with multi-optimization objectives. International Journal of Hydrogen Energy 2024;93:1258–1267. https://doi.org/10.1016/j.ijhydene.2024.10.192.

[12] Jung W, Chang D. Deep Reinforcement Learning-Based Energy Management for Liquid Hydrogen-Fueled Hybrid Electric Ship Propulsion System. Journal of Marine Science and Engineering 2023;11(10):2007. https://doi.org/10.3390/jmse11102007.

[13] Jung W, Chang D. Correction: Jung, W.; Chang, D. Deep Reinforcement Learning-Based Energy Management for Liquid Hydrogen-Fueled Hybrid Electric Ship Propulsion System. J. Mar. Sci. Eng. 2023, 11, 2007. Journal of Marine Science and Engineering 2024;12(3):484. https://doi.org/10.3390/jmse12030484.

[14] Wu P, Partridge J, Anderlini E, Liu Y, Bucknall R. An intelligent energy management framework for hybrid-electric propulsion systems using deep reinforcement learning. International Journal of Hydrogen Energy 2025;106:282–294. https://doi.org/10.1016/j.ijhydene.2025.01.380.

[15] Silver D, Lever G, Heess N, Degris T, Wierstra D, Riedmiller M. Deterministic Policy Gradient Algorithms. Proceedings of the 31st International Conference on Machine Learning, PMLR 2014;32(1):387–395. https://proceedings.mlr.press/v32/silver14.html.

[16] Lillicrap TP, Hunt JJ, Pritzel A, Heess N, Erez T, Tassa Y, Silver D, Wierstra D. Continuous control with deep reinforcement learning. arXiv preprint arXiv:1509.02971, 2015; presented at ICLR 2016. https://arxiv.org/abs/1509.02971.

[17] Wu Y, Tan H, Peng J, Zhang H, He H. Deep reinforcement learning of energy management with continuous control strategy and traffic information for a series-parallel plug-in hybrid electric bus. Applied Energy 2019;247:454–466. https://doi.org/10.1016/j.apenergy.2019.04.021.

[18] Guo X, Tang D, Yuan Y, Yuan C, Shen B, Guerrero JM. A New Energy Management Strategy Supported by Reinforcement Learning: A Case Study of a Multi-Energy Cruise Ship. Journal of Marine Science and Engineering 2025;13(4):720. https://doi.org/10.3390/jmse13040720.

[19] Yang X, Zhou X, Li X, Wang Y, Long F, Tang T, Liu L, Liu Y. Multi-time scale energy management and optimization for hybrid power ships considering lifetime balance and extension of multi-stack fuel cell systems. Ocean Engineering 2026;358:125687. https://doi.org/10.1016/j.oceaneng.2026.125687.

[20] Ren P, Pei P, Li Y, Wu Z, Chen D, Huang S. Degradation mechanisms of proton exchange membrane fuel cell under typical automotive operating conditions. Progress in Energy and Combustion Science 2020;80:100859. https://doi.org/10.1016/j.pecs.2020.100859.

[21] Schmittinger W, Vahidi A. A review of the main parameters influencing long-term performance and durability of PEM fuel cells. Journal of Power Sources 2008;180(1):1–14. https://doi.org/10.1016/j.jpowsour.2008.01.070.

[22] Saponaro G, Stefanizzi M, Torresi M, Camporeale SM. Analysis of the degradation of a Proton Exchange Membrane Fuel Cell for propulsion of a coastal vessel. International Journal of Hydrogen Energy 2024;61:803–819. https://doi.org/10.1016/j.ijhydene.2024.02.349.

[23] Kwon L, Cho DS, Ahn C. Degradation-Conscious Equivalent Consumption Minimization Strategy for a Fuel Cell Hybrid System. Energies 2021;14(13):3810. https://doi.org/10.3390/en14133810.

[24] Ghaderi R, Kandidayeni M, Soleymani M, Boulon L, Trovão JPF. Online Health-Conscious Energy Management Strategy for a Hybrid Multi-Stack Fuel Cell Vehicle Based on Game Theory. IEEE Transactions on Vehicular Technology 2022;71(6):5704–5714. https://doi.org/10.1109/TVT.2022.3167319.

[25] Xu B, Oudalov A, Ulbig A, Andersson G, Kirschen DS. Modeling of Lithium-Ion Battery Degradation for Cell Life Assessment. IEEE Transactions on Smart Grid 2018;9(2):1131–1140. https://doi.org/10.1109/TSG.2016.2578950.

[26] Han X, Lu L, Zheng Y, Feng X, Li Z, Li J, Ouyang M. A review on the key issues of the lithium ion battery degradation among the whole life cycle. eTransportation 2019;1:100005. https://doi.org/10.1016/j.etran.2019.100005.

[27] Mylonopoulos F, Durgaprasad S, Coraddu A, Polinder H. Lifetime design, operation, and cost analysis for the energy system of a retrofitted cargo vessel with fuel cells and batteries. International Journal of Hydrogen Energy 2024;91:1262–1273. https://doi.org/10.1016/j.ijhydene.2024.10.235.

[28] Thomas D. Cost Reduction Potential for Electrolyser Technology. Hydrogenics; 2018. Presentation cited in [27], reference 54. https://www.humsterlandenergie.nl/resources/LInks-duurzaam/Linkpagina/20180619_Hydrogenics_EU-P2G-Platform_for-distribution.pdf. No DOI assigned; direct slide-level verification remains pending.

[29] Raffin A, Hill A, Gleave A, Kanervisto A, Ernestus M, Dormann N. Stable-Baselines3: Reliable Reinforcement Learning Implementations. Journal of Machine Learning Research 2021;22(268):1–8. https://jmlr.org/papers/v22/20-1364.html.

[30] Fujimoto S, van Hoof H, Meger D. Addressing Function Approximation Error in Actor-Critic Methods. Proceedings of the 35th International Conference on Machine Learning, PMLR 2018;80:1587–1596. https://proceedings.mlr.press/v80/fujimoto18a.html.
