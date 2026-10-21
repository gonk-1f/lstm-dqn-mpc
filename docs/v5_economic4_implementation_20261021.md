# V5四维因果状态与纯经济奖励

## 版本与隔离

- 新分支：`feat/v5-economic4-ddqn`。
- 新工作树：`C:\Users\20883\OneDrive\Desktop\lstm-dqn-mpc\lstm-dqn-mpc\.worktrees\v5-economic4-ddqn`。
- 基准HEAD：`a5713bde00c1202175f2b1b2a428a6408eb77eb1`。
- 从原增量工作树复制5份已修改文件、2份未跟踪的交错更新实现/检查文件；复制后的全src规范化指纹与原工作区一致：`f8be52530a6fc8575f7ece4feeda2c6c62335879e628fdbe2a7119dae65547dd`。
- `v5_source_baseline_20261021.json`记录原状态及逐文件SHA。原工作树和HEAD未变；已有权重、Replay、训练输出均未写入。本次没有提交、推送或启动正式训练。
- 相对Git HEAD的diff含此前未提交的交错更新；V5没有重新发明该机制。`interleaved_replay.py`与原工作树逐字节一致。

## 状态和动作

`load_delta4 = [SOC_t, measured_load_t/600, previous_executed_FC_t/600, (measured_load_t-measured_load_{t-1})/600]`。

初始/新航段差分0；下一状态以真实执行后的FC/SOC与下一次观测构造，恰等于下一步当前状态。当前动作只看当前和过去负载。没有未来信息、剩余航时、AIS或新门控。

新状态保留实测负载及差分，包括既有死区中的微小负数；物理执行仍沿用原`normalize_onboard_load_kw`规则，没有修改载荷或物理模型。旧minimal3/history7仍保持原归一化行为。

网络`4–128–128–7`、ReLU；动作`[-60,-30,-10,0,10,30,60] kW`，执行`previous_FC+delta`。候选逐个与原物理mask相交，不裁剪、不回退绝对动作。Agent、共享统计和诊断通过特征名读取上一FC，不将4维误认成7维。

## 奖励与成本来源

`zhou_economic_only`正常未缩放奖励：

```
r_raw = -(C_H2_actual + C_H2_Bat_equ_signed + C_FC_deg + C_Bat_deg)
C_H2_Bat_equ_signed = (SOC_before-SOC_after)*624/(eta_ref*LHV_H2)*hydrogen_price
r_replay = 0.001*r_raw
```

|分项|复用来源|
|---|---|
|实际氢耗|正式FC瞬时效率曲线/氢耗模型；`EconomicMPC.interval()`返回原`RawCnyIntervalLedger.h2_cost_cny`|
|电池有符号等效氢耗|`independent_economics.soc_equivalent_hydrogen_cost_cny()`|
|FC退化|原四工况退化模型及经济寿命增量；账本`fuel_cell_degradation_cost_cny`，含真实变载、启动损耗|
|电池退化|原SOC/电流加权Ah模型及寿命换算；账本`battery_degradation_cost_cny`|

固定参考效率`0.5711999752720877`、LHV`33.333333333333336 kWh/kg`、氢价`21.9元/kg`和电池`624 kWh`不变。SOC已经包含原充放电效率；等效氢耗不再次除充放电效率。SOC修正有正有负，不截断，逐步望远镜求和只结算一次。

新奖励强制beta_soc=0并跳过SOC辅助项计算；额外线性/平方变载项均0。独立航段无SHORE、无模拟终端充电、无额外终端SOC收费。四分项对应Zhou的三大类成本，未增加第五项经济费用。

真实无可行动作继续走原失败处理，最后一个已执行动作附加一次原始失败惩罚`10132.660087898294`，随后统一缩放；失败done且不bootstrap。失败惩罚不入实际经济账本。

**有限固定惩罚不能从数学上保证压倒每一条可能路径的全部未来成本。** 本轮按要求保留原数值；失败样本的完整可比较成本仍为null，且不能成为合格最佳模型。是否仍有策略偏好提前失败，必须用正式训练及完整供电率检验，不能仅凭一次失败惩罚就宣称已消除该风险。

## 冻结训练机制与兼容性

保留gamma=1、单步TD、Adam0.001、SmoothL1 beta1、Batch64、Replay300000、PER(.6/1/1e-6)、失败终止配额2、replay32、Target tau.001、梯度裁剪10、seed42和100轮。

探索阶段每新增32条已闭合经验按原调度授予更新；每次实际Adam后软更新Target，后续动作使用最新Online。原bootstrap预填充及其更新时序保留，与正式交错更新分别计数。epsilon第1轮1、第70轮.05、之后.05；Greedy Train/Validation均0，且每轮Validation独立执行。

V5合法配置为独立数据集+load_delta4+zhou_economic_only+increment7+n_step1+beta0+interleaved/replay32+原PER参数。入口保留旧配置的独立校验，没有全局放宽约束。状态mode/dim/features、4–128–128–7、奖励身份、manifest、源码指纹、实验ID均进入checkpoint身份；新3/4维不能互载，不加载旧Replay进行新实验。

原3维/7维、历史绝对动作与旧奖励模式保留。新增加权TD指标只是记录`sum(w*|TD|)/N`及`sum(w*|TD|)/sum(w)`，没有修改原IS归一化、Loss、priority或更新。

## 必要检查结果

- 22项V5定向检查通过，另7项旧3/7维、入口及诊断回归检查通过；没有执行全套pytest。
- 合成检查覆盖因果历史/跨样本重置、7档索引/mask、一次64条PER反向更新、Target初始复制及tau软更新、正常/失败终止不bootstrap、失败惩罚一次、4维航段内交错更新、评价无学习以及旧checkpoint拒绝。
- 9个相关模块静态解析、`git diff --check`通过。
- 一条已知完整Train航段`parent_006_seg01`按其归档固定动作执行352步；不是重新训练或新Greedy策略回放。

|经济分项|累计元|
|---|---:|
|实际氢耗|509.6291535428835|
|电池SOC等效氢耗|-101.50407387401766|
|FC退化|2163.712500000003|
|电池退化|45.89422478380546|
|逐步经济总额|2617.7318044526737|
|独立Zhou评价|2617.731804452674|

差约`-4.55e-13元`，SOC望远镜误差0；账本/动作/掩码/SOC与归档一致，辅助项/岸电/终端补能0。只打开1个Train载荷，Validation/Test载荷0，未更新网络。数据身份和详细数值见`v5_accounting_check_20261021.json`。

仅用Train35/Validation8元数据时长与正式物理/动作界得到保守寿命上界：FC≤15267.4643 μV<70000 μV；电池≤172157.1704 Ah<21666666.6667 Ah。当前独立新寿命初态下不触及经济封顶，累计寿命不会通过封顶改变边际退化成本，无需因此继续扩维。

## 预设结果指标

- 每轮探索Train、Greedy Train/Validation完整完成率与固定集合成本；分别记录SOC、结构功率及increment_action_limited失败，失败部分成本不冒充完整成本。
- 四分项经济累计及SOC签名修正，失败惩罚单列。
- FC平均绝对变化、非零变化比例、实际启动、反向次数；反向比较不跨样本/SHORE/终止边界。
- 加权/未加权TD、Huber、梯度、Replay覆盖/抽样和分类误差。
- 真正执行的环境步、入池数、优化器数、Target数和UTD，区分bootstrap、正式训练与仅评价。
- 固定诊断轮次1、10、…、100：每条Train/Validation最多16个确定性状态，真实物理重建变载/保持各自后继，记录Q优势、即时奖励差、masked DDQN目标差与排序一致性。此为有限采样诊断，不是全状态正确率或最优Q真值。

## 新实验启动与恢复

复用`v4.feedback_study`，没有复制另一套DDQN。`scripts/start_v5_economic4.ps1`含完整冻结参数，默认创建带时间戳的全新目录：

```powershell
Set-Location 'C:\Users\20883\OneDrive\Desktop\lstm-dqn-mpc\lstm-dqn-mpc\.worktrees\v5-economic4-ddqn'
& .\scripts\start_v5_economic4.ps1
```

需要明确保存目录时：

```powershell
& .\scripts\start_v5_economic4.ps1 -OutputDir '.\outputs\v5_economic4_delta7_td1_per_interleaved_seed42_100r_run01'
```

仅恢复该V5运行自身的完整轮次checkpoint：

```powershell
& .\scripts\start_v5_economic4.ps1 -ResumeFrom '.\outputs\v5_economic4_delta7_td1_per_interleaved_seed42_100r_run01\training_state_latest.pt'
```

不跨V4权重恢复。运行期间不要修改源码或提交改变HEAD，否则严格身份校验将拒绝续训；中途未完整保存的轮次按原机制重新执行。

## 已实现与仍待验证

已实现：训练/评价经济目标对齐、因果负载差分、4维映射及恢复隔离、原交错更新接入与必要诊断字段。

未宣称解决：FC振荡、Q动作优势偏差、供电成功率或经济性。7档动作可达性缺口保持且单列；四维仍无法知道未知未来负载和终止位置。新奖励与新状态是本轮授权的两项变化，不能把未来效果拆归给其中一项而不做进一步受控比较。本次不启动任何正式训练。
