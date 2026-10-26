# V6：1秒DQN八轮训练与完整分析资料

本目录对应分支 `feat/v6-dqn-1s`。正式运行 **run02 已完成8/8轮**，最终 Greedy Validation 仅完成 **3/8**，尚未形成稳定、可完整执行全部航段的控制策略。第2轮曾达到7/8且功率较平稳，但后续没有保持。不要把最后一轮、更新次数增加或部分TD误差下降等同于最优控制器。

先读 [控制与逐航段分析](ANALYSIS_CONTROL.md) 和 [Q价值、TD拟合与动作排序分析](ANALYSIS_Q.md)。[给网页版GPT的审查说明](WEB_GPT_PROMPT.md)列出关键入口与证据限制。

## 配置与运行边界

|项目|正式run02|
|---|---|
|算法与网络|普通DQN，单步TD，MLP 5→256→256→13，每个网络70,669个可训练参数|
|时间尺度与动作|1秒；13个实际增量 −60:10:+60 kW；FC绝对功率0～600 kW|
|学习与Target|Adam学习率0.0001；SmoothL1 beta=1；梯度裁剪10；每步更新；纯硬Target每1000次Adam复制；γ=1|
|奖励|缩放1；SOC权重0.2；FC平方增量权重0.05；一次失败惩罚10132.660087898294；退化只在评价账本|
|数据与预算|Train35 / Validation8；随机预填充50,000条且不更新；seed42；8遍Train；Test payload读取0|
|实际累计|最终断点状态分支Adam更新2,782,159次；Target复制2,783次（含初始化）；Replay保留最近1,000,000条|

1秒输入复用原有完整1秒航段，43条Train/Validation文件共601,367个节点，新增插值点为0。这是正确的数据处理结果；未重新处理原始数据。±60 kW/s是本项目仿真假设，不能写成实船已标定爬坡能力。

run01是旧配置（学习率0.001，SOC与波动权重均0.1），仅第1轮完整保存，随后暂停；与正式run02分开归档。run02曾按要求丢弃未保存的第5轮部分进度，从第4轮完整断点重跑第5轮。原日志因此包含重复的第5轮尝试，最终统计和模型以最终保存状态分支为准，不能把两个尝试重复相加。

## 网页可直接阅读的证据

|资料|内容|
|---|---|
|[逐轮指标](browser/round_metrics.csv)|Train/Greedy Validation完成、失败分类、FC变载/反向、成本口径、累计更新|
|[全部Validation逐航段结果](browser/validation_all_voyages.csv)|8轮×8条的完成、时长、SOC、成本与变载|
|[共同完整航段成本](browser/common_completed_validation_costs.csv)|仅020与040跨8轮均完整；固定子集公平比较，不代表8条全套|
|[四类经验TD统计](browser/td_fit_all_rounds.csv)|在线实际draw与checkpoint分层样本分开；Q/target/带符号残差/MAE/P95|
|[全部13动作Q与排名](browser/q_all_13_actions_all_checkpoints.csv)|round00～08，44个固定物理状态，5,148条动作行；合法标志与排名变化|
|[探针排序摘要](browser/q_probe_rank_summary.csv)|SOC/负载/上一FC、合法与原始排序、保持与升降动作Q差、第一二名差|
|[保存的终止TD样本](browser/all_saved_terminal_td_samples.csv)|正常和失败终止的状态、Q13、Target下一状态Q13、奖励、done与残差|
|[全部失败详情](browser/all_failure_details.csv)|真实失败时刻、SOC、负载、上一FC、物理网格合法区间、13动作候选及mask|
|[失败前最后一次真实决策](browser/validation_last_executed_before_failure.csv)|22次Greedy失败的最后一秒全部13Q和排序，并列下一秒失败条件|
|[失败前300秒](browser/validation_failure_windows/)|直接摘录实际1秒Validation轨迹，保留全部13Q与奖励分项|
|[30秒抽取的完整航段概览](browser/validation_30s/)|每轮8条，直接抽取原行及末行；不是新插值，不用于精确计数变载|
|[真实Train失败价值](browser/train_failure_values/)|固定高/低SOC失败案例的实际行为累计回报、当前Q、TD目标和13动作排序|
|[真实Train失败前缀](browser/train_failure_prefixes/)|原有失败前300秒及更早锚点，完整转为CSV供网页读取|
|[参数变化](browser/parameter_changes.csv)|只读比较已有8个checkpoint参数；没有构造网络或执行训练|
|[图与全量轨迹索引](PLOTS.md)|全部64条正式Validation功率/SOC图及原始1秒轨迹|

## 原始记录与checkpoint

- [raw/run02](raw/run02/)：正式运行所有原有文件，含65幅PNG、64份完整1秒Validation CSV.gz、round00～08的Replay诊断样本、完整Q探针、Train失败记录、配置与报告。
- [raw/run02/models](raw/run02/models/)：`round_01.pt`～`round_08.pt`分别保存Online、Target和Adam；`latest.pt`为第8轮；`best_research_candidate.pt`仍是第1轮，按原有完成数→覆盖率规则选出，**并非已经满足8/8的合格策略**。
- [完整最新断点](raw/run02/models/training_state_latest.pt)：第8轮完整Replay、Online、Target、Adam、随机状态和诊断状态。只有最新完整Replay被保留，不能声称每轮都保存了完整Replay。
- [checkpoint可读元数据](metadata/checkpoint_metadata.json)：参数形状/数量、各轮Adam状态与参数变化、最新Replay数组与计数。
- [raw/run01](raw/run01/)：旧配置的首轮结果、checkpoint、8条失败曲线与[不可达专项分析](raw/run01/pause_reachability_analysis_20261024/analysis_report.md)。
- [运行日志](raw/logs/)：包含正常完成日志及用户暂停产生的KeyboardInterrupt；中断日志不是算法崩溃的证据。
- [原始文件清单](metadata/source_file_inventory.csv)与[运行背景](metadata/archive_context.json)；未额外生成文件哈希。
- [运行环境快照](metadata/runtime_snapshot.json)：使用原训练解释器采集的归档时版本，不冒充每轮保存的独立环境锁定文件。

压缩CSV保留全量，`browser/`提供可直接浏览的表格。原始JSON/Markdown中的Windows绝对路径是历史来源记录，访问资料请用本目录相对链接。`pause_state.json`与PID是历史进程操作记录；最终完成状态以 `raw/run02/report.json`、`progress.json` 为准。

## 源码与数据入口

- [实际训练闭环](../../../src/v4/v6_training.py)：冻结配置、真实Train交互回调、Validation、逐轮保存与恢复。
- [DQN与Replay](../../../src/v4/dqn.py)：`DirectPowerDQN`、`_UniformDQNReplay`、单步Target最大化、Huber、反向传播、Adam与纯硬Target。
- [环境推进与奖励](../../../src/v4/control.py)：`replay_episode`，实际负载、FC动作、电池功率、SOC、奖励和终止。
- [13动作及可达集合](../../../src/v4/actions.py)、[物理/经济执行](../../../src/v3/control.py)、[电池能量模型](../../../src/v2/models/battery_energy.py)。
- [TD记录](../../../src/v4/v6_td_diagnostics.py)、[固定Q排序](../../../src/v4/v6_q_probes.py)、[Train失败价值](../../../src/v4/v6_train_values.py)。
- [1秒输入与处理报告](../../../data/processed/operating_dataset_v6_1s/)：仅35条Train与8条Validation负载payload；Test仅有划分元数据。
- [依赖声明](../../../requirements.txt)；[当前配置说明](../../v6_lr1e4_q_reporting.md)。

实际运行器还从相邻V5工作树的 `outputs/v5_qfit_interference_20261023/fixed_probe_states.json` 读取四个固定探针的原始来源。该文件作为[引用输入](raw/reference_inputs/v5_fixed_probe_states.json)一起保留。原运行器的这个路径没有为归档改写，因此不能声称新克隆仓库无需准备外部目录就能直接重跑；已有结果和checkpoint本身可独立读取。V5比较所需的既有逐航段指标也在 `raw/reference_inputs/`，无Test输入。

## 证据能回答与不能回答的事

代码、已有参数变化、Adam状态、实际轨迹和Replay共同支持：这是实际运行过的自定义船舶能量管理环境及DQN神经网络训练，控制质量仍未达标。未使用Gymnasium不是错误。

失败终止经验的拟合改善，没有转化为稳定的Greedy控制改善。第8轮5次失败均是低SOC附近所需FC升幅超过+60 kW，保持±60只增加动作档位无法解决；末秒负载变化均很小，不能统称最终突发负载阶跃。单步物理合法不保证下一秒仍能在动作范围内恢复。

TD目标是随网络变化的学习目标，不是Q*；实际探索行为的累计回报也不是Q*或反事实最优收益。Train存在随机探索，不能和Greedy Validation混为一谈。Train没有保存每一步完整历史，只保存了原有失败前缀/锚点、样本统计与最新Replay；不得补造未记录轨迹。

此次只分析并整理已有资料，没有新增训练、环境仿真、测试或Test读取。下一版建议优先审视下一步可恢复性和终止价值拟合；本次没有修改动作、奖励、TD或训练预算。

## 推送后的本地存放

已确认推送后完成本地清理，见[清理记录](metadata/cleanup_receipt.json)。

确认远端收到提交后，清理当前V6工作目录中的两个V6运行目录、对应日志和PID。`outputs`中的17个旧版正式校准/预检输入仍被V2源码读取，保留原位，`raw/legacy_outputs/`仅为它们的归档副本。未删除其他工作树的结果或训练输入。

远端归档和Git对象保留；本地大体积归档通过当前工作树的稀疏检出排除，避免留下第二份训练结果。需要取回时可使用 `git sparse-checkout disable`，或直接从本分支下载指定文件；这会重新占用本地空间。Git对象数据库仍保留已提交数据，清理并非清除Git历史。清理不会自动恢复训练。
