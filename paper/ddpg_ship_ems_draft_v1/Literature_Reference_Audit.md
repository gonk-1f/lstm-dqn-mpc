# DDPG 船舶能量管理论文：代码与文献核查

核查日期：2026-11-03。本文档服务于初稿审阅，不构成新训练或性能实验。PDF 页码均为从 1 开始的文件页码；需要时另列印刷页码。

## 1. 核查边界与当前代码身份

实际 DDPG 工作树：

`C:\Users\20883\OneDrive\Desktop\lstm-dqn-mpc\lstm-dqn-mpc\.worktrees\v6-sb3-baseline`

分支 `feat/sb3-ddpg-aggregate`；核查 HEAD `f28e26e97077e9ac2dd4be38d9a76ed7d7f3e2d2`。以下源码路径均相对此完整工作树根目录。

开始时已有用户未提交修改：`experiments/sb3_manual/ddpg_aggregate/zoo_optuna.py`。其 tau 搜索上界由 0.15 调为 0.06，noise 搜索区间由 [0.03,0.12] 调为 [0.07,0.15]。本次不改动、不暂存、不提交该文件。论文描述区分当前工作区与 HEAD，不能声称它们完全相同。

本次只新增 `paper/ddpg_ship_ems_draft_v1/` 的文稿。没有执行训练、Validation8、Test4、模型推理或 Optuna 搜索；没有修改数据、模型、Replay、奖励或 YAML；没有介入 PyCharm 进程。对正在进行的新奖励 Study 仅执行 SQLite 只读查询，不调用 Optuna API。

## 2. 重要证据结论

| 内容 | 已核实依据 | 写入初稿的处理 |
|---|---|---|
| FC 600 kW / 6×100 kW、LB 624 kWh | Yang 2026 PDF p3，p14 §5.1/Table 6 | 容量采用其研究重配置；不是把原船装机数据改写为600/624 |
| 原船与研究重配置 | Yang p3：原船8×70 kW、约1800 kWh；研究配置6×100/624，另含SC | 本项目仅聚合 FC–LB，无SC、无单堆/单簇控制 |
| 432 V、LB −624/+1248 kW、η=.95/.95 | Yang p14 Table 6；项目源代码 | 写为实际能量/电流核算基准 |
| SOC 初值 .6、边界 .2–.8 | `env.py`/`v3/control.py` | 属项目设置；Yang 原文为初值 .65、边界 .2–.9，明确不混同 |
| FC 四工况结构及四系数 | Zhou p5 Eq(2)/Table 2；代码另列 Saponaro 2024 为系数来源 | 来源是既有模型；不是本文原创、不是 Yang 的 ECSA 模型 |
| FC 低/高运行项 | 当前函数两个互斥分支 | 0<P<480只计10.17；P≥480只计11.74 μV/h，不叠加 |
| FC 变载变量 | 代码按 `abs(ΔP_aggregate/6)` | 单位为kW的累计绝对变化，不是无量纲次数 |
| FC 启停 | 0→正功率计23.91 μV一次；停机不再加一次 | 聚合启动记账约定，不冒充六堆独立启停 |
| FC 70 mV EOL | 当前常量70000 μV | 0.7 V参考电压×10%的模型归一化，不是本船寿命实测 |
| LB F/G 函数 | Kwon2021 Eq(14)–(16)；Zhou p7–8 Eq(12)–(14)；Ghaderi Eq(31)–(32) | 正电流放电、负电流充电；使用步前SOC |
| LB 15000 倍吞吐分母 | 当前代码及历史说明明确为 secondary-literature assumption | 未找到独立数值原始依据，不写成15000完整循环寿命 |
| FC 更换费50% | Mylonopoulos p8/印刷1269 §4.4/Table 5 | 一次乘入FC更换费；不是退化率折半 |
| Thomas/Hydrogenics 上游 | Mylonopoulos p12 ref54；原报告受访问验证拦截 | 只确认书目/间接引用，不伪称读到支持船用FC50%的原始幻灯片 |
| DPG/DDPG | Silver2014 Theorem1/Eq9；Lillicrap Algorithm1 | 直接引用原始理论，不只转引Guo |
| 当前奖励 | `ddpg_economic_tanh_v1`、λ=17 | 四项经济成本+SOC平方项，tanh倒数，失败−1 |
| 固定航段500次训练 | 当前 reset 仍随机有放回Train35 | 只能写计划，未实现固定parent005选择器 |
| 当前新奖励Optuna | 只读数据库确认搜索进行中 | 不使用旧Trial18，不把中途best当最终参数/经济最优 |

## 3. 代码—公式对应表

公式编号对应 `DDPG_Ship_EMS_Draft.md` 与 Word。行号对应上列 HEAD/当前工作区快照，不保证未来版本不变。

| 公式/内容 | 当前代码位置 | 核对要点 |
|---|---|---|
| (1)、(6)，容量/功率/SOC | `src/v2/config.py:84`；`experiments/sb3_manual/ddpg_aggregate/env.py:21`、`:99`、`:128`；`src/v3/control.py` 的 SOC_MIN/MAX | FC600、LB624、48kW/s、.6初值、.2–.8边界 |
| (2) FC效率 | `src/v2/models/fuel_cell_efficiency.py:16`、`:179` | 本地工作簿A2:B12，PCHIP，先求100kW端点，再按6映射600 |
| (3) 氢耗 | `src/v2/models/fuel_cell_efficiency.py:289` | 120MJ/kg=33.3333kWh/kg；kJ形式不再乘3600 |
| (4)–(5) SOC | `src/v2/models/battery_energy.py:71` | 正放电P/ηdis，负充电ηchgP；并非Rint |
| (7)–(9) 物理约束 | `experiments/sb3_manual/ddpg_aggregate/env.py:128`、`:178`、`:198` | 交集投影；空交集失败，不人为放宽 |
| (10)–(13) FC退化 | `src/v2/models/fuel_cell_degradation.py:13`、`:65`、`:128`、`:171`、`:280`、`:445` | 100kW参考功率；运行项互斥；启动计次；累计裁剪寿命增量 |
| (14)–(18) LB退化 | `src/v2/models/battery_degradation.py:10`、`:63`、`:72`、`:98`、`:275`；`src/v3/control.py:151` | I=1000Pbus/432；Qnom=1444.444Ah；Qlife=15000Qnom；累计差计费 |
| (19)–(21) 经济价格/更换费 | `src/v2/economics.py:31`；`fuel_cell_degradation.py:24`、`:560`；`battery_degradation.py:377` | H2=21.9；FC=3500×600×.5；LB=2000×624；不能重复乘.5或6 |
| (22)–(24) 电池等效氢耗 | `experiments/sb3_manual/ddpg_aggregate/economic_reward.py:47`、`:59`、`:78` | 开机实际功率点η，关机曲线峰值η；SOC差已含ηchg/dis，不重复损耗 |
| (25)、(31) MDP/观测 | `env.py:66`、`:88` | 3维观测不含time/future/累计寿命；不宣称严格全可观测Markov |
| (26)–(30) 原生DDPG | 当前解释器安装包 `stable_baselines3/ddpg/ddpg.py`、`td3/td3.py`、`common/off_policy_algorithm.py` 的静态源码 | 单critic、policy_delay1、targetnoiseclip0；Adam和soft-target更新；不是TD3双critic |
| (32) 连续动作 | `env.py:167`、`:198` | 绝对功率300(a+1)，再投影；不是ΔP，也不是DQN action mask |
| (33)–(35) 奖励及标定 | `economic_reward.py:123`；`economic_reward_config.json:2`；`env.py:178` | SOCnext、λ17、q0→1、负/非有限报错、真实失败−1 |
| 每100Episode验证 | `periodic_validation.py:64`、`:117`；`zoo_ddpg.yml:19` | dones计数含失败；独立CPU子进程Validation8；无Test |
| 正式配置及入口 | `zoo_ddpg.yml:1`；`run_ddpg_train.py:16`；`zoo_register.py` | RL Zoo原生入口；eval-freq−1；20M安全步上限不替代500Episode |
| 独立评估 | `validate.py:122`、`:173`、`:227`；`evaluate.py`、`test4.py`、`test4_env.py`；根目录直接运行入口 | DDPG.load，action_noise=None，deterministic=True；不learn，不更改Replay |

特别说明：复用对象虽然名为 `EconomicMPC`，DDPG 环境只调用 `interval()` 推进物理及账本，没有调用 `solve()`。论文不能把它描述为“DDPG + 下层MPC控制”。

FC/LB累计寿命用来计算成本；当前没有把磨损反馈为FC效率/额定功率下降或LB容量/内阻变化。每个episode重置物理与账本，不是连续多航次多年寿命仿真。

## 4. 效率曲线和数据证据

### 4.1 FC效率

源工作簿为 `C:\Users\20883\OneDrive\Desktop\氢耗\FC_Data.xlsx`。项目曲线保留其功率/效率列，而不是把同表旧氢耗列直接用于现有120MJ/kg氢耗计算。原功率支持点最高116.668kW；100kW参考区间的端点先由原始PCHIP求得。百分比效率转小数后映射到600kW系统。当前峰值效率0.6385540000000001，对应原参考功率11.3938kW/聚合68.3628kW。

峰值是由现有曲线计算，不是人为添加的η参数。原工作簿的测试工况、净/毛功率测量边界及设备实测校准说明尚未建立完整书目级来源。初稿将其称为project-supplied efficiency data，不声称是Yang解析曲线或已完成本船台架验证。

### 4.2 负载来源与划分

| 证据文件 | 内容 |
|---|---|
| `data/processed/operating_dataset_zero_boundary_v2/metadata/policy.json:19`、`:28` | 原始FC为8堆总功率、LB为12簇功率；原LB负放电正充电；负载 `P_fc_total−P_battery_raw_total` 是重构量，不是独立测量 |
| 同上 | 源名义30s；缺口采用局部natural cubic，航段内1s曲线PCHIP；不能称1Hz原始实测 |
| `data/processed/operating_dataset_independent_v2/metadata/splits_v2_exclude046_seg01/policy.json` | 冻结35/8/4；Train35来自31parent，Val8/Test4各来自8/4parent，跨split父组交集为0；月份分层seed42，Test固定 |
| `data/processed/operating_dataset_v6_1s/metadata/data_processing_summary.json:9` | 43个Train/Val载荷gzip原样复用，无新增插值；Train457562、Val143762执行区间；Test载荷未复制/未打开 |
| `src/v4/v6_dataset.py:64` | 最后1s边界节点不产生额外物理动作 |
| `src/v2/data/supervisory_rules.py:35` | 负载<−1报错；−1至0之间负值归零；任意正值保留。这不是±1kW对称死区 |

本次只读数据来源与分组元数据，没有读取Test负载。现有负载重构/插值及小正值规则均未修改。

## 5. 原文、上游来源和证据等级

### 5.1 Yang容量参数

本地原文：`C:\Users\20883\OneDrive\Desktop\Yang 等 - 2026 - Multi-time scale energy management and optimization for hybrid power ships considering lifetime bala.pdf`。

DOI [10.1016/j.oceaneng.2026.125687](https://doi.org/10.1016/j.oceaneng.2026.125687)，Ocean Engineering358(2026):125687。已读正文/参考文献，p14参数页已渲染视觉核对。

p3明确区分原船8×70kW/约1800kWh与论文重配置6×100kW/624kWh；p14给432V及.95效率。Yang提出ECSA/分堆均衡等内容，不属于本项目已实现方法。本文只采用适用容量、相关参数及聚合参考映射，不搬用其退化模型或SOC设置。

### 5.2 FC四工况与原始数值来源

本地 Zhou 原文：`C:\Users\20883\OneDrive\Desktop\Zhou 等 - 2025 - Multi-temporal energy management strategy for fuel cell ships considering power source lifespan deca.pdf`。

DOI [10.3390/jmse13010034](https://doi.org/10.3390/jmse13010034)，JMSE13(1)(2025):34。2024-12-29在线发表与2025卷年不矛盾。已读p5 Eq(2)/Table2、p7–8电池公式、p16成本式及末尾参考文献；p5/7/8已渲染核对。

Zhou p5列四项结构及四系数，段落引用Ren[26]与Schmittinger[27]支持退化机理，但数值表未逐项说明原始实验标定。因而不能把10.17/11.74/.0441/23.91全部断言为Ren/Schmittinger的原始实测结果。

代码明确另列 Saponaro2024 DOI [10.1016/j.ijhydene.2024.02.349](https://doi.org/10.1016/j.ijhydene.2024.02.349)。其出版社/机构存储库搜索索引Table3可见相同系数、0–80%和80–100%区间。直接机构PDF返回403，未得到完整本地原件，故证据等级低于已完整读取的Zhou。其上游指向Wu/Bucknall2020 DOI10.1016/j.ijhydene.2019.11.152及Saponaro2023 SAE DOI10.4271/2023-24-0145；终极实验来源尚未完全直接核实，初稿不虚构该层证明。

Ren2020 DOI [10.1016/j.pecs.2020.100859](https://doi.org/10.1016/j.pecs.2020.100859)，PECS80:100859，支持启停、动态负载、开路/低载等机理。该文另有勘误10.1016/j.pecs.2020.100871，涉及图9纵轴单位及两条引文；初稿未采用这些图/数值。Schmittinger/Vahidi2008 DOI [10.1016/j.jpowsour.2008.01.070](https://doi.org/10.1016/j.jpowsour.2008.01.070)，JPS180(1):1–14，为PEMFC耐久性综述。两篇当前核查到可靠出版元数据与相关机理记录，未把未逐页读取的内容冒称已全文核验。

### 5.3 LB半经验形式与寿命分母

Kwon L、Cho DS、Ahn C，**Degradation-Conscious Equivalent Consumption Minimization Strategy for a Fuel Cell Hybrid System**，Energies14(13)(2021):3810，DOI [10.3390/en14133810](https://doi.org/10.3390/en14133810)。这是项目源码明确记录的模型DOI；出版社索引公式(14)–(16)可核对F/G，非“船用电池432V”出处。其自身示例battery80V40Ah，不能移用为本船参数。

Zhou p7 Eq(12)、p8 Eq(13)–(14)使用相同加权Ah形式，其所引Ghaderi2022 DOI [10.1109/TVT.2022.3167319](https://doi.org/10.1109/TVT.2022.3167319)，IEEE TVT71(6):5704–5714，机构作者版索引Eq(31)–(32)可核对应力函数。该研究是多堆车辆而非本项目船舶。

Xu2018 DOI [10.1109/TSG.2016.2578950](https://doi.org/10.1109/TSG.2016.2578950)，IEEE TSG9(2):1131–1140；2016是online/DOI年，2018是正式卷年。Han2019 DOI [10.1016/j.etran.2019.100005](https://doi.org/10.1016/j.etran.2019.100005)，eTransportation1:100005。二者用于更广泛退化机理/建模适用性说明，不冒称直接给出15000这一数值。

历史提交 `435f630` 的 `docs/v2_battery_degradation_model.md` 通过只读git show核查：15000Qnom明确为secondary-literature lifetime-throughput modeling assumption，非manufacturer spec、非Yang参数、非vessel measured。未恢复历史文件。未找到能独立确认15000的原始文献页码；该项保留待决，不通过改写论文掩盖。建议以后比较10000/15000/20000分母，但本轮不运行任何敏感性实验。

### 5.4 50%更换成本

本地原文：`C:\Users\20883\OneDrive\Desktop\Mylonopoulos 等 - 2024 - Lifetime design, operation, and cost analysis for the energy system of a retrofitted cargo vessel wi.pdf`。

DOI [10.1016/j.ijhydene.2024.10.235](https://doi.org/10.1016/j.ijhydene.2024.10.235)，IJHE91(2024):1262–1273。PDF p8、印刷1269、§4.4及Table5明确replacement=.5CAPEX：周期替换可退化cells，而非全部BoP；p12 ref54列Thomas2018。该页已视觉核对。

Mylonopoulos采用15%电压退化更换假设，且其低高工况阈值/变载系数与本项目不同。本文只从其提取更换成本比例的经济假设，不移入其退化参数或EOL阈值。

Thomas D，Cost Reduction Potential for Electrolyser Technology，Hydrogenics，2018。指定地址 [报告镜像](https://www.humsterlandenergie.nl/resources/LInks-duurzaam/Linkpagina/20180619_Hydrogenics_EU-P2G-Platform_for-distribution.pdf) 返回验证页；备用原地址也未取得完整PDF。索引可确认演示文稿题目与电解槽主题，但不能确认其中确有“marine FC replacement=.5CAPEX”原始句。未赋予不存在的DOI、期刊卷页。初稿明确这是Mylonopoulos的直接假设以及尚未直接证实的上游链。

### 5.5 DDPG、船舶RL与综述

| 文献 | 实际阅读范围及定位 | 可支持的描述与限制 |
|---|---|---|
| Wu2020 Applied Energy275:115258，10.1016/j.apenergy.2020.115258 | 本地18页；p1摘要、p5–6§4/Eq14–16 | tabular Double Q，不是DDQN；其shore成本/终端补电不能移植 |
| Wu2021 IJHE46(80):40022–40040，10.1016/j.ijhydene.2021.09.196 | 本地19页；p2–3综述与组织，p4–9算法/模型；风格页已渲染 | Double DQN；五作者含Anderlini/Liu；只借鉴论证组织不复制文字 |
| Wu2025 IJHE106:282–294，10.1016/j.ijhydene.2025.01.380 | 本地`wu-TD3.pdf`；p3§2.4/Eq4–8，p4算法 | TD3、多FC簇增量动作；不是当前1维绝对功率DDPG |
| Zhu2024 IJHE93:1258–1267，10.1016/j.ijhydene.2024.10.192 | 出版社摘要/highlights、Crossref元数据；未取得全文 | DQL，voyagecost/FC寿命/SOC；不编写其未核实详细奖励公式；Ma为Kuangqi |
| Jung/Chang2023 JMSE11(10):2007，10.3390/jmse11102007 | 已读出版社v2修正版20页，p10成本，p11–12 DQN/100离散动作 | 其FGSS/BoP是显式模型，本项目不具备；必须结合2024勘误 |
| Jung/Chang2024 correction，10.3390/jmse12030484 | 出版社修正记录 | 修正FC退化/成本/单位/公式等，不能引用旧版错误结果 |
| Guo2025 JMSE13(4):720，10.3390/jmse13040720 | 本地24页；p12 Eq22–23，p13 Eq24–25已视觉核对，p15 Figure10 | DDPG四柴油机、SC；平均奖励为每轮累计后按航行持续时间平均；未给足失败步口径 |
| Guo2024 OceanEng310:118638，10.1016/j.oceaneng.2024.118638 | Crossref及作者机构Chalmers摘要 | 广义综述背景，不以未读全文的细节支持公式 |
| Silver2014 PMLR32(1):387–395 | 原始proceedings PDF p3 Theorem1/Eq9及p4–5 | DPG梯度；不证明本项目部分观测/任意神经critic必然收敛 |
| Lillicrap2015 arXiv1509.02971 / ICLR2016 | 原始PDF p3–5，Algorithm1 | Replay、criticMSE、actor梯度、softtargets；原文OU噪声不等于本项目Gaussian |
| Wu2019 hybrid bus，10.1016/j.apenergy.2019.04.021 | Crossref/可靠出版记录，未全文 | 连续控制相关交通应用；作者Yuankai Wu不是Peng Wu |
| Bassam2017，10.1016/j.ijhydene.2016.08.209 | 作者机构NERC摘要/accepted记录 | 状态规则/ECMS/CDCS/PI多方案；不转移其节能百分比 |
| Kalikatzarakis2018，10.1016/j.conengprac.2018.04.009 | 作者机构TU Delft摘要 | marine ECMS/MINLP/DP基准；Boonen作者缩写EJ |
| Hou2018，10.1016/j.apenergy.2017.12.098 | 作者UMich PDF p1摘要/p2 | battery/flywheel，DP权衡和MPC；非本项目FC–LB试验 |
| Mnih2015，10.1038/nature14236 | 出版元数据及DQN基本方法 | DQN相关背景，不作为DDPG原始理论 |
| Raffin2021 JMLR22(268):1–8 | JMLR原文记录 | SB3软件来源；实际配置还需查项目源码 |
| Fujimoto2018 PMLR80:1587–1596 | 原始proceedings/算法摘要 | TD3与DDPG的区别，不把继承类名当实验算法 |

IMO采用 [2023官方战略](https://www.imo.org/en/OurWork/Environment/Pages/2023-IMO-Strategy-on-Reduction-of-GHG-Emissions-from-Ships.aspx)：2050附近净零，2030/2040较2008减排检查点20%/70%、努力30%/80%。不把后续草案写成已生效法规，也不把船端零排放写成全生命周期零排放。

另查到Zhu2026 DOI10.3390/jmse14050460已经研究multi-stack health-aware TD3。初稿不主张船舶连续控制、退化感知或未来分堆健康分配属于首次提出。正文参考文献为30项，按首次出现重排；核查材料中的作者—年份/DOI为稳定识别依据。

## 6. 标定、正在进行的Study与不可当作结果的记录

奖励配置记录Ctyp=0.17429534153424198，理论100Ctyp=17.429534153424198；用户后明确将当前λ改为17。标定baseline是旧控制型奖励模型：

`outputs/ddpg_train_500ep/ddpg/AggregateShipDDPG-v0_1/AggregateShipDDPG-v0.zip`

SHA256 `caf67c28eea863c01c8d6f9a8f7e38ea27b2f3ce924fe1d12dc4a843551f100f`。标定只包括Train35已执行448434秒，33条完整，另两条parent015/036失败前缀；计划总秒457562并未全部执行。该数据是权重标定历史，**不是当前经济奖励训练结果**。标定配置记录无Validation/Test使用。

对 `outputs/ddpg_optuna_economic_v1/study.db` 的只读快照：study `ddpg_tau_noise_economic_tanh_v1`，8 COMPLETE、1 RUNNING、0 PRUNED；方向MAXIMIZE。快照best Trial1：value5706.497630624999、tau0.03082649433960295、noise0.11789267873576292。此快照会随用户搜索继续变化，不能写成最终最优结果。数据库时间字段未说明时区，不擅自换算。

其目标是Zoo平均Validation累计reward；不是完成率优先/完整经济成本其次的排序。trial模型有效seed字段为null，不能仅凭命令--seed42保证各trial模型种子相同。Study命令解释器路径与`.venv-optuna`不同；静态包metadata显示该venv SB3 2.9.0、Zoo2.9.1，不能据此冒称已核实正在运行Study的实际历史包版本。论文不固定这组未经完整留档的运行版本。

正式YAML仍tau=.03、noise=.12，只作为当前baseline。新正式目录`outputs/ddpg_train_500ep_economic_v1`在核查时不存在。旧500Episode及旧Validation/Test结果保留原位，不纳入第四章。新reward Study也不与旧Trial18结果混用。

## 7. 当前方法需要诚实保留的限制

1. FC效率工作簿缺少完整测试/设备/功率边界的原始说明，不能称完成本船效率标定。
2. FC四系数数值可在Zhou和Saponaro引用链找到，但最终原始实验适用性未完全核实。
3. LB15000Qnom缺独立数值源，应补资料与敏感性，不能写成实测寿命。
4. 50%替换成本可直接引Mylonopoulos；Thomas上游报告支持尚未直接确认。
5. 3维观测可能部分可观测；未进行堆/簇健康反馈和容量退化动态。
6. `tanh(1/q)`小成本区可能饱和，累计奖励最大化不等于累计CNY成本最小化。
7. signedequivalentH2是能量估价而非实际购买，实时η使其通常不是仅终SOC差的望远镜求和。
8. 固定parent005的500Episode选择器尚未实现；不能启动现有随机训练后声称固定航段实验已完成。
9. 原生maxepisodes及周期验证计数包含物理失败；8条均已评估不等于8条均成功。
10. 周期checkpoint位于当前transition入Replay/Adam前；第500个done触发stop后该末终止transition不再入Replay。结果应记录实际env/physical/Adam计数。
11. 运行中的Optuna以meanreturn为目标且模型seed需单独核实；论文暂不宣称完成了经济优先调参。

以上限制均未通过修改训练代码来“修正”；它们是本文稿的适用范围和后续审阅项。

## 8. 文稿检查与Word排版证据

Word与Markdown使用相同正文，恰有5个编号一级章节；Abstract/Keywords/Nomenclature/References为非编号前后置内容。包含37个连续编号显示公式、30条文献、6张编号表及符号表、Fig1–11图位。所有结果图位和第4/5章仍明确pending；未生成随机或伪造的性能曲线。

公式通过LaTeX→MathML→Office MML2OMML转换为原生Word数学对象，可编辑而非截图。长公式分行以避免越过页边。运行环境为Codex bundled Python；公式转换依赖仅下载至任务TEMP目录，不安装到项目虚拟环境。

技能自带`render_docx.py`在本Windows上因没有LibreOffice soffice.exe而失败（FileNotFoundError），不声称该检查通过。改用独立隐藏Word COM实例只读打开文稿导出PDF，再用bundled Poppler逐页栅格化检查。该过程不调用训练Python、不修改项目依赖或已打开的PyCharm进程。最终22页均已逐页检查，未发现缺字、越界、重叠或公式裁切；长公式、标题横线及单位上标排版已修复。结构检查确认119个原生数学对象、37个连续公式全部有正文引用、30条文献首次出现顺序一致；Markdown空白检查与git diff --check通过。最终交付目录只包含5份文稿/说明，不包含QA临时PDF、PNG或项目运行产物。
