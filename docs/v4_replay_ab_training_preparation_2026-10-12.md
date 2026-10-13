# V4 Replay A/B 正式训练接口与方案

## 1. 代码与证据状态

本地工作树：`C:\Users\20883\OneDrive\Desktop\lstm-dqn-mpc\lstm-dqn-mpc\.worktrees\direct-power-mlp`。

分支 `feat/direct-power-mlp-ddqn`；核对远端后，基准HEAD为 `a40afa2c8fd0a967d40d8aa30472b228232fbc0e`，与origin一致。本轮修改尚未提交或推送，命令使用本轮本地文件。

| 已阅读报告 | 本轮开始时的状态 | 证据适用范围 |
|---|---|---|
| [Replay覆盖/TD审计](v4_replay_coverage_td_audit_2026-10-12.md) | 本地存在、未跟踪，GitHub未包含 | 原100轮/30万池，不是新前缀修复后的表现 |
| [根因诊断](v4_ddqn_root_cause_diagnostic_2026-10-12.md) | 本地存在、未跟踪，GitHub未包含 | 原结果、独立冻结目标拟合与有限物理诊断 |
| [已观测前缀修复](v4_observed_prefix_replay_fix_2026-10-12.md) | 已随a40afa2提交 | 新窗口插入及定向检查，尚无正式闭环效果 |

本轮没有重新核算正式训练轨迹。历史数值引用上述报告；没有读取Test、修改历史模型/Replay/日志或启动正式训练。

## 2. 当前问题与推荐顺序

| 判断 | 证据 | 最小处理/判别 |
|---|---|---|
| 确定：原UNKNOWN整段弃前缀导致训练覆盖损失 | 原累计舍弃520554个已执行决策；新接口保留完整已观测n8窗口 | 先做同源码A/B，不更换网络/奖励/采样 |
| 确定：部分已存高风险目标与Q有较大残差 | `_015`/97当前目标约−11.40、Q约−2.88；正常终止5336窗MAE约2.389 | 看新实际采样、关键Replay ID目标变化及分类误差 |
| 已观察到拟合副作用，但尚未直接测量梯度方向冲突 | 独立局部拟合降低局部误差，却提高其他留出误差；normal配额8也使其他类别恶化 | 不立即加采样配额、增更新或扩网络；必要时用独立副本固定y测分类梯度余弦和留出误差 |
| 较强假设：7维状态的任务阶段/负载条件信息不充分 | 近邻中末端与中途目标不同；Validation附近Train目标未包含同样风险 | 先验证覆盖修复；仍异常时确认可在线获得的信息，不加真实未来负载或样本真实剩余长度 |
| 确定：本模型下`_036`可行，但归档策略过早失去余程可行性 | 最高合法FC策略完成；原策略row20已离线不可恢复，row102才出现空mask | 关注早期价值，不仅失败最后8步；可达门槛不能成为控制规则 |
| 确定：`_005`部分大跳变来自固定Q排序交叉，非mask强迫 | 40次>200 kW两动作均合法；row208保持590即时费用少43.42元 | 保留已有FC变载经济费，不重复加平滑罚项；观察Q/控制结果是否同步改善 |
| 确定：β200已计入，但它不能保证SOC轨迹或平稳FC | `_006`高SOC下0仍合法而Q最低；有正常末端拟合偏差 | 保持β不变；不以SOC阈值覆盖Q排名 |

四项经济模型含FC低载、高载、变载和启停损耗，因而有平稳运行的经济激励，但不保证训练出的函数已学到其最优权衡。当前SOC辅助项为 `200*(SOC_next_actual-0.6)^2`，与经济账本分开。没有证据支持本轮再改奖励或新增FC切换费用。

当前 `control.py` 是直接执行FC动作、共享 `EconomicMPC.interval` 核算，没有MPC优化器二次改写输出，没有人工停机门控或新增ramp。策略候选为物理可行动作；Online合法argmax、Target估值、n8累计/一次缩放、真终止禁bootstrap、失败一次惩罚及UNKNOWN未闭合尾窗均保持。

## 3. 已有功能与本轮最小改动

已经有 `--unknown-replay-mode discard/observed_nstep`、独立输出目录保护、完整轮checkpoint、每轮Greedy Train/Validation、固定评价资格、四项费用/SOC/FC分布、分类TD与真实采样及关键目标诊断。本轮没有重新设计训练系统。

| 文件/函数 | 本轮改动 |
|---|---|
| `feedback_study.main` | 新增 `--seed`（默认42）与 `--experiment-id`，真实传入Agent训练链；其他冻结设置不松绑 |
| `monitored_training.run_monitored_training` | 记录实验编号、源码树SHA256；续训严格核对编号、seed/模式/超参数、源码commit与实际源码、manifest、线程和诊断配置 |
| `monitored_training._preserve_uncommitted_diagnostics` | 修复诊断文件写出而round checkpoint未提交的恢复窗口：将超出已提交轮次的诊断文件保留归档，完成轮文件不动 |
| `monitored_training`三处执行统计 | 传入每条独立样本的真实transition长度；失败/UNKNOWN前缀也统计，不拼接不同样本 |
| `diagnostics.executed_transition_statistics` | 补严格`>100/>200/>300` kW次数；分`sample_initial/shore_departure/continuous_onboard`，不加物理控制规则 |
| `dqn`诊断侧 | 本轮实际优化batch抽到的唯一Replay ID数及动态池覆盖率；不改变`random.sample/shuffle`、TD公式或更新额度 |
| `decision_diagnostics.save_diagnostic_snapshot` | 诊断checkpoint/决策表携带实验编号、源码树摘要，与best文件继续分离 |
| 相关五个测试文件 | CLI、完整轮恢复、旧合同拒绝、唯一覆盖及跳变边界合成检查 |

唯一覆盖率定义：

`本轮实际成功optimizer batches抽到的不同Replay IDs / (本轮reset时池大小 + 本轮新增插入数)`。

重复抽取只算一个ID，分母包括淘汰前未抽到的记录；不是“当前最终池覆盖率”，不是有效学习成功率。集合只保留本轮，不把大集合写进报告或checkpoint；恢复后下一轮reset重新统计。来源分箱、跳变阈值仅描述数据，不进入策略输入、mask或reward。

## 4. A/B固定条件与公平性

| 设置 | 两组相同值 |
|---|---|
| 轮次/主种子 | 100 / 42 |
| 网络/动作 | MLP7–128–128–61；0～600 kW每10 kW |
| Adam/TD | lr0.001，γ1，n8，Huber beta1，gradclip10 |
| Replay/更新 | 容量300000、batch64、replay32、失败终止配额2、失败倍率1 |
| Target/奖励 | tau0.001每次经济Adam后软更新；scale0.001；β200、SOC_ref0.6；重分配关闭 |
| 探索 | 第1轮1.0，至第70轮0.05，71～100保持0.05 |
| 物理/模式 |600 kW FC、624 kWh电池、30s；原SOC硬界/电池功率约束、v3标签、固定目标岸电 |
| 数据/选择 |30Train/8Validation原划分；自动完整集合现21/7；UNKNOWN9/1前缀无物理失败也为资格条件；Validation每轮执行；Test封存 |
| 诊断 | 轮1、10、20、30、40、50、60、70、80、90、100；不参与选模 |

两组从空Replay、新Adam和新种子初始化开始。A=`discard`，B=`observed_nstep`。A也必须重跑，历史100轮不是本次同源码A。

同seed保证相同初始化和 `seed+round` 样本排列，不保证分岔后完全相同探索随机流：现有动作探索和Replay抽样共用Agent RNG，B额外更新会消耗更多随机数。B还自然增加软Target推进次数。因此本轮检验“恢复前缀＋现有credit机制”的整体效果，不能直接归因为纯数据覆盖收益；不为凑公平性擅自拆RNG或更改Target/更新预算。

若有改善，再补seed43/44各一对重复，避免直接铺开超参数网格。若必须区分数据与更新收益，先在同初始化、冻结目标、相同有效更新预算上做独立经验对照；闭环等预算实验需明确新预算调度与checkpoint计数，另行批准后实施。本轮不新增采样/PER/预算算法，也不提供当前入口会拒绝的消融命令。解析器历史cadence/n-step选项仍受正式冻结校验约束。

## 5. PyCharm PowerShell命令

以下目录和解释器已核实存在。两个输出路径目前不存在，是将由入口新建的目录；若后来已运行，不能以同路径重新从头训练。按顺序执行A、再B，不并行开两组以增加内存负担。

### 公共环境（每个新终端先执行）

```powershell
Set-Location -LiteralPath 'C:\Users\20883\OneDrive\Desktop\lstm-dqn-mpc\lstm-dqn-mpc\.worktrees\direct-power-mlp'
$env:PYTHONPATH = (Resolve-Path -LiteralPath '.\src').Path
$env:OMP_NUM_THREADS = '16'
$env:MKL_NUM_THREADS = '16'
```

两组线程均16，与原正式报告记录一致；实际值会写入metadata，续训须保持一致。`--seed`控制权重/探索/抽样及派生样本顺序；`--experiment-id`仅标识，不影响学习；`--diagnostic-rounds`保存独立诊断Online/Target快照，不改变best资格。

### A组

```powershell
& 'D:\py\Python3\python.exe' -m v4.feedback_study --output-dir 'outputs\v4_replay_ab_A_discard_seed42_100r' --experiment-id 'V4-RPL-A-s42' --seed 42 --unknown-replay-mode discard --rounds 100 --diagnostic-rounds '1,10,20,30,40,50,60,70,80,90,100'
```

### B组

```powershell
& 'D:\py\Python3\python.exe' -m v4.feedback_study --output-dir 'outputs\v4_replay_ab_B_observed_seed42_100r' --experiment-id 'V4-RPL-B-s42' --seed 42 --unknown-replay-mode observed_nstep --rounds 100 --diagnostic-rounds '1,10,20,30,40,50,60,70,80,90,100'
```

未在命令中列出的学习参数均为上述冻结值，由metadata记录实际生效配置。不得给A/B不同诊断轮或不同源码。需要seed43/44时，两组同时改seed、编号和各自新输出目录；不能复用seed42输出。

### A组完整轮checkpoint恢复

```powershell
& 'D:\py\Python3\python.exe' -m v4.feedback_study --output-dir 'outputs\v4_replay_ab_A_discard_seed42_100r' --resume-from 'outputs\v4_replay_ab_A_discard_seed42_100r\training_state_latest.pt' --experiment-id 'V4-RPL-A-s42' --seed 42 --unknown-replay-mode discard --rounds 100 --diagnostic-rounds '1,10,20,30,40,50,60,70,80,90,100'
```

### B组完整轮checkpoint恢复

```powershell
& 'D:\py\Python3\python.exe' -m v4.feedback_study --output-dir 'outputs\v4_replay_ab_B_observed_seed42_100r' --resume-from 'outputs\v4_replay_ab_B_observed_seed42_100r\training_state_latest.pt' --experiment-id 'V4-RPL-B-s42' --seed 42 --unknown-replay-mode observed_nstep --rounds 100 --diagnostic-rounds '1,10,20,30,40,50,60,70,80,90,100'
```

仅恢复最后完整已保存轮次；未完成轮会从头重做。预填充/首轮尚未保存checkpoint时不能续训。训练期间不要修改源代码、切换commit或更改参数/线程；源码树摘要也核对未提交代码。旧100轮、best和diagnostic checkpoint均不是本次续训源。重启终端后先重做公共环境。

## 6. 报告字段与验收

现有 `round_metrics.csv/report.json/round_history.json/train.log` 继续保存逐轮统计与资格；正式轨迹/独立诊断checkpoint继续保存，完整轮续训仍包含Online、Target、Adam、Replay/FIFO池顺序、RNG、credit和ε进度。

- 首先确认B真实增加 `economic_unresolved_replay_insertions`，A的该字段始终为0；真正独立闭合段仍按success插入并单独统计retained。计数与保留/舍弃、unique draws、credit闭合；不出现UNKNOWN terminal、罚款或虚构补能。
- 比较每轮/首次/最佳/最后10轮Greedy完成数、SOC受限失败及资格轮数；固定完整Train与Validation均完成、UNKNOWN前缀无物理失败才是合格checkpoint。不能把完整集合缩水。
- 同时报告env transitions、Replay插入量、实际梯度/Target次数、不同ID覆盖、各类别采样数、normal terminal/failed prefix/ordinary/unresolved误差及原单位换算。正常终止/失败终止数代表n-step窗口，不是独立事件。
- 看关键相同Replay ID的当前Q/y及目标漂移；它们不是历史固定监督真值。类间副作用须一起报告；只降低末端误差而普通/失败前缀恶化，不支持全面学习改善。
- 看Train/Validation泛化差距及真实功率/SOC轨迹。高SOC、SOC低于0.4只是诊断区间，不是新增动作规则。FC原有均值/最大变化之外，分别比较连续ONBOARD、模型岸电后的出航以及样本首步；首次样本是否此前靠港不能由缺少的历史推断。
- 跳变统计覆盖全部已执行决策，包括失败/UNKNOWN前缀；因此原始次数也受执行长度影响。报告原始次数、每1000个相应控制步的比例，完整同一批样本再作性能比较，不能用提前失败导致的少跳变称更平稳。
- 费用只比较固定完整集合、同模型口径的CNY账本。氢耗、FC退化、电池退化、模型岸电/终端补能分别报告，SOC辅助项/失败惩罚不计实际费用。不完整策略的总可比成本保持null；只完成部分样本的低成本不是优势。

若B在后期仍更少物理失败、更多资格轮、关键Q拟合与FC/SOC控制同步改善，再用43/44重复支持有效性；单seed或单类TD下降不足以定论。若仅TD降低而完成率不改善，优先检查共同拟合、目标漂移与可观测状态。若Q拟合明显改善但完整同集合的经济/跳变仍差，再研究经济目标权衡，当前不加平滑奖励。

本轮不设置未经验证的“控制必须达到某FC功率”指标，也不承诺所有7维状态能辨认未知未来高负载。

## 7. 实际检查

- CLI缺seed/id的失败检查、诊断孤立产物的恢复失败已先复现，再做最小修复。
- 66项定向合成/单元检查通过：51项Replay/统计；8项CLI参数路由与非法值；6项前缀模式、Greedy/资格、诊断隔离、同种子恢复（含孤立产物保留及错误实验ID拒绝）；1项旧Replay续训合同拒绝检查。没有全量测试或正式训练。
- `python -m v4.feedback_study --help`显示新参数；10个相关Python文件编译、`git diff --check`通过。
- 控制、奖励、失败惩罚、更新调度、物理及网络定义无diff；未提交推送。本轮交付为接口准备和手动命令，不是已取得A/B结果。
