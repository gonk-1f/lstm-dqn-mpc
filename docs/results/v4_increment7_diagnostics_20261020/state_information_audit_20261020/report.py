from pathlib import Path
import csv,json
from collections import defaultdict,Counter
import numpy as np
OUT=Path(__file__).resolve().parent;RUN=OUT.parent
def read(n):return list(csv.DictReader((OUT/n).open(encoding='utf-8-sig')))
def md(h,rs):return '\n'.join(['|'+'|'.join(h)+'|','|'+'|'.join(['---']*len(h))+'|']+['|'+'|'.join(map(str,r))+'|' for r in rs])
rows=read('actual_states_q_returns.csv');pairs=read('near_state_pairs.csv');pred=read('causal_history_prediction.csv');cats=read('category_q_return_diagnostics.csv')
groups=defaultdict(list)
for r in rows:groups[(float(r['soc']),float(r['load_kw']),float(r['previous_fc_kw']),int(r['action_index']))].append(r)
dup=[]
for state,rr in groups.items():
    if len(rr)<2:continue
    record=dict(soc=state[0],load_kw=state[1],previous_fc_kw=state[2],action_index=state[3],observations=len(rr),parents=len(set(r['parent'] for r in rr)),categories=json.dumps(dict(Counter(r['category'] for r in rr))))
    for field in ('future_mean_load_1','future_mean_load_8','future_mean_load_32','realized_greedy_return','dload_kw'):
        vals=np.array([float(r[field]) for r in rr if r[field]])
        record.update({field+'_min':float(vals.min()),field+'_max':float(vals.max()),field+'_std':float(vals.std())})
    dup.append(record)
with (OUT/'identical_state_action_groups.csv').open('x',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(dup[0]));w.writeheader();w.writerows(dup)
same=[r for r in pairs if r['pair_kind']=='nearest_same_action'];pstats=[]
distance=np.array([float(r['normalized_euclidean_distance']) for r in same]);quartiles=np.quantile(distance,[0,.25,.5,.75,1])
for q in range(4):
    rr=[r for r in same if quartiles[q]<=float(r['normalized_euclidean_distance']) and (float(r['normalized_euclidean_distance'])<=quartiles[q+1] if q==3 else float(r['normalized_euclidean_distance'])<quartiles[q+1])]
    for field in ('delta_soc','delta_load_kw','delta_previous_fc_kw','delta_future_mean_load_1','delta_future_mean_load_8','delta_future_mean_load_32','delta_realized_return','delta_phase_diagnostic_only','delta_observed_suffix_steps'):
        vals=np.array([abs(float(r[field])) for r in rr if r[field]])
        pstats.append(dict(distance_quartile=q+1,distance_lower=quartiles[q],distance_upper=quartiles[q+1],field=field,n=len(vals),median_absolute_difference=float(np.median(vals)),p90_absolute_difference=float(np.quantile(vals,.9))))
with (OUT/'near_state_distance_quartiles.csv').open('x',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(pstats[0]));w.writeheader();w.writerows(pstats)
predtable=md(['数据','标签','方法','输入','MAE/kW','RMSE/kW'],[
    ['Train parent留出' if r['split']=='heldout_train_parent' else 'Validation',r['horizon_steps']+'步','Ridge' if r['method']=='ridge_alpha1' else 'kNN32',r['features'],f"{float(r['mae']):.4f}",f"{float(r['rmse']):.4f}"]
    for r in pred if r['horizon_steps']=='1'])
qttable=md(['上下文','类别','步数','单步冻结TD MAE','Q减实际剩余回报均值'],[
    [c,k,next(r['n'] for r in cats if r['context']==c and r['category']==k and r['field']=='td'),
     f"{float(next(r['abs_mae'] for r in cats if r['context']==c and r['category']==k and r['field']=='td')):.5f}",
     f"{float(next(r['mean'] for r in cats if r['context']==c and r['category']==k and r['field']=='q_minus_realized')):.5f}"]
    for c in ('train','validation') for k in ('ordinary_nonterminal','normal_terminal','failure_prefix','failure_terminal')])
text=f'''# 三维状态信息与长期价值拟合诊断

日期：2026-10-20。只读正式代码、数据及历史模型；新建本诊断目录。未修改单步TD、gamma=1、奖励、动作、网络或正式数据，未训练DDQN/回放策略/调用优化器/访问Test/提交Git。

## 1. 结论

**当前三维足以描述当前功率分配的一步物理约束，但没有证据证明它是完整长期Markov状态；也没有足够证据证明增加一个或两个负载差分能明显改善长期控制。现在不建议直接启动状态扩维训练。**

相同/相近观测确实对应不同未来负载和回报。但外部负载随机性、不同独立任务的终止长度，以及当前策略的失败路径，都可产生这种现象；不能只凭不同回报断言少了某个状态量。新旧诊断一致显示，简单因果负载差分预测收益小、不稳健。

## 2. 实际配置、状态构建与泄漏审查

- 本地分支 `feat/v4-increment7-zhou-td1`，HEAD `dc3c1947bc4dc400710896dfe3acca2e9ae3f54d`；保留既有交错更新准备等5个tracked及4个untracked文件，本次不动它们。
- 正式100轮实验源码提交 `82e21cc46fef163b8680550e35252e2f7c8fa0db`，minimal3、7档增量、单步TD、gamma=1；源码指纹及全部metadata见 `verification.json`。原100轮是航段末集中更新，本地交错接口尚未正式训练。
- `src/v4/control.py:258–287` 明确构造 `[SOC_before, current_load/600, previous_actual_FC/600]`。
- `control.py:443–460` 当前负载进入当前状态和物理mask后才选择动作；SOC来自执行前AccountState，上一FC来自本次仿真实际动作，不取历史实船FC。
- `control.py:463–488` 动作执行后更新SOC和FC历史，`:559–582` 才构造next_state与下一决策mask。next_state中下一时刻负载是Bellman后继观测，不进入当前动作选择；执行后SOC只进入next_state及既定奖励，未放进当前state。
- `src/v4/independent_dataset.py:230–243` 按当前左端点取原canonical负载，不用未来插值。新航段历史缺失差分设0，初始SOC=.6、FC=0。
- 无未来SOC、未来负载、剩余时长、航段ID、未来岸电标志作为策略输入。**没有发现当前三维构建的未来数据泄漏。**
- 本诊断的未来均值、实际剩余回报、实际航段相位和剩余步数仅作离线标签/分层。它们都不能据此成为在线状态。
- 任务末尾可能有不足30s的真实区间，dt影响物理转移/费用而不在三维输入中；终止边界也不是三维的确定函数。这是有限任务同观测不同转移/终止语义的潜在来源，不能据此擅自输入不可在线获知的剩余时长或改变终止规则。

## 3. 数据与方法

- Train35/Validation8，冻结manifest身份核对一致；只读取这43个航段载荷。Test载荷0。
- 第100轮Greedy实际记录：Train13931步，Validation3726步，共17657步；包含完整31/35与6/8及实际失败前缀，不重跑Greedy。
- 读取真实R100 checkpoint，CPU张量只读提取、NumPy float32三层前向，**17657/17657贪心动作复现归档**；155个既有真实Torch状态的7档Q最大误差1.91e-6。未更新Q/Target或恢复优化器执行。
- 复用此前R100探索路径的Train parent留出历史预测与冻结目标拟合结果，不重复其200次副本实验。
- 新增的Ridge(alpha=1)与kNN(k=32)是固定负载诊断器，非DDQN或部署控制器；没有搜索参数。特征标准化仅用Train拟合集。
- 一组评估沿用此前7个Train parent留出，另一组用全部Train拟合、Validation作外部分布描述；Validation不参与拟合或挑选诊断模型参数。这些探索性Validation分析已经使用其信息，应如实记录，不称为全新盲验证。
- 标签：下一实际采样负载，以及未来8/32步负载均值（约4/16min，单位kW）。只用存在完整窗口的行，不跨航段、不补零。标签可来自实际失败后的原始需求序列，但绝不伪造失败后SOC、动作或回报；访问状态仍仅为真实执行前缀，存在策略选择偏差。

## 4. 状态混叠与外部负载不确定性

### 4.1 完全相同观测与相同已执行动作

`[SOC=.6, load=0, previous_FC=0]`且同为保持动作，共34个不同parent真实观测：

|标签|最小|最大|
|---|---:|---:|
|下一负载/kW|1.0142|9.7888|
|后8步均值/kW|5.9092|106.3114|
|后32步均值/kW|25.4350|292.7936|
|实际Greedy剩余训练回报（已缩放）|−18.5151|−1.8249|

这组当前历史差分也全部为0；因此新增一个上一负载/差分**无法区分这组开航状态**。其中30条路径最终完整、4条最终失败，但这不是在线可用标签，也不是34次独立同分布随机试验。它证明观察不唯一确定未来，不足以单独证明违反随机MDP条件或存在能观测到的缺失变量。

另有同一parent内129个完全相同SOC=.7998714239、负载0、FC0、保持动作的状态，后32步均值0–33.5623kW、实际剩余回报−1.6770至−.6544。这体现停留阶段/外部需求出现时刻差异，而非当前电池物理模型不同。

### 4.2 跨Train/Validation近邻（明确距离，不设“充分覆盖”阈值）

在三维原始归一化坐标上用欧氏距离：`sqrt(dSOC²+(dLoad/600)²+(dFC/600)²)`。对每个Validation状态找Train前32近邻，另取其中最近且**实际动作索引相同**的一点；得到3700对。未给任意距离阈值并宣布状态充分/不足；另按距离四分位保存敏感性表。

相同动作配对的中位差：SOC .002345、当前负载1.609kW、上一FC 0kW；下一负载差11.743kW、后8步均值差24.322kW、后32步均值差78.601kW。未来均值可用对数分别3689/3613/3349，不把缺窗当0。实际剩余回报差中位数2.9072，但已观测剩余步数差中位数169步，所以不能把全部回报差归为状态混叠。

代表例（从距离最低四分位中按后32步差选取的说明例，非总体效应）：Validation045第328个0基决策 vs Train003第199个决策，SOC差.0006925、当前负载差−1.3173kW、上一FC一致、实际动作均−30kW；后32步均值相差约524.92kW。两个当前负载变化量差18.58kW，提示历史可能提供上下文，但是否普遍有效必须看留出预测，而非只挑这一对。

`near_state_pairs.csv`保存了每对单独dSOC/dLoad/dFC、未来均值、真实回报、类别和仅诊断相位/长度；`near_state_distance_quartiles.csv`避免一个任意阈值主导结论。

### 4.3 三种来源不能混同

1. **外部负载不确定性**：同当前需求可有不同下一需求，即使随机Markov模型也会有方差；单条记录不足以恢复真实条件分布。
2. **任务阶段差异**：近似当前物理状态可能位于负载平台的开头/结尾，有限历史或可观测任务信息可能有用；目前尚未确认可靠的在线阶段信号。
3. **终止边界差异**：正常终止目标只有r，非终止有bootstrap；失败尾部另含一次罚项。实际剩余回报同时受终止位置、未来策略动作和失败影响，不是仅由当前外部负载决定。

因此本次支持“有观测别名/阶段差异现象”，尚不支持“一个新增差分已被证明可消除它”。

## 5. 加因果历史是否降低条件不确定性

候选仅 `d_t=(load_t−load_(t−1))/600`。给定当前负载，它与加入上一负载在信息上等价；没有未来负载输入。候选三维加一差分为4维诊断器，未改正式状态。

{predtable}

重点：三维加单差分的下一负载Ridge MAE在Train parent留出15.7064→15.6542kW（约.33%），Validation19.6118→19.5139（约.50%）；Validation残差方差2110.024→2105.903，仅降低约.20%。kNN对应MAE反而21.0005→22.9716、22.6866→24.4879。

后8步均值的Validation Ridge MAE43.6777→43.6140kW，**RMSE77.6967→77.7525及残差方差略恶化**；kNN MAE46.6217→47.2890。不能只选择MAE小降宣布不确定性明显降低。

经验近邻标签离散度：Validation仅当前负载加差分时，下一负载邻居标准差均值36.997→34.495kW，但P90−P10宽度42.535→53.932kW；给定完整三维再加差分，标准差37.729→42.235kW、宽度59.646→79.390kW。邻域定义和稀疏性随维度变化，不能将这些量冒充已标定条件熵；综合外部预测误差，不支持稳健且明显的信息增益。

此前独立R100探索状态加**两个**差分的结果也类似：下一负载Ridge MAE15.5434→15.5144（.19%），kNN25.3432→31.2087；后8步需求电能Ridge2.38156→2.37848kWh（.13%）。这是不同路径和标签单位，不能与本次Greedy均值kW表直接混算。旧parent bootstrap下一负载宏均值改变量95%区间跨0；后8步能量仅有很小改善。

**判断：候选历史包含少量相关信息，但目前没有稳定、实质的降低未来负载不确定性的证据。** 条件均值预测器能力有限，本结论既不证明任何历史都无用，也不证明三维充分。

## 6. Q、冻结一步目标和实际路径回报

下表使用同一R100模型、实际贪心状态；Q/Y及回报均为scale=.001的训练单位，非人民币。`G_t=sum_{{j=t}}^{{最后真实执行步}} r_j`，失败包含原一次失败罚，完整航段正常终止；没有失败后虚构费用。G是固定Greedy真实路径的已实现回报，**不是最优Q真值**；换未执行动作后，未来状态及动作都会变化。

{qttable}

### 6.1 失败风险在多早被反映？

Train失败路径中，距最后有效动作仍有>100个已观测步时，Q均值+.1445，而实际剩余回报均值−14.2842；距尾部33–100步Q约−1.0185、回报−12.3699；9–32步Q−7.2244、回报−10.9936；最后8步中的非终止前缀Q−8.5459、回报−10.3512。

Validation对应>100步Q+.1472、回报−14.2566；9–32步Q−4.3987、回报−10.9962；最后8步非终止Q−9.0748、回报−10.3602。

这表明**这几条失败路径的负面后果主要在临近失败时才明显体现在Q里**。不等于未传播的唯一原因是state：bootstrap偏差、更新时序、数据分布、共享拟合和无法从当前观察识别的风险均可能贡献。特别是普通完整路径也有Q与实际剩余回报大差距，不能只归因于失败罚项。

### 6.2 动作排序的复现、敏感性与历史限制

- 17657/17657动作复现说明状态/索引/推理与归档匹配，**不是动作排名正确或长期稳定的证明**。
- 复用020窗口：Online与Target最佳动作仅13/64一致；最佳−次佳差中位数.005795、最小8.93e-6，但最佳−保持中位数.050903。说明两个变载选项可能接近，保持仍被明显低排。不能把全部振荡说成浮点噪声。
- 020全部64步即时变载劣于保持；40/64冻结动作对目标偏保持而Online偏变载。这是相对当前估计目标的排序拟合偏差，不是保持在全航段必然最优的证明。
- 精确R38、R62权重不存在，不能使用R40/R60/R100替代比较它们在相同状态的排名。本次没有新增历史排名假数据。

### 6.3 TD下降是否对应成功率

|轮次|训练期间采样TD MAE|Greedy Train|Greedy Validation|
|---|---:|---:|---:|
|38|.2316785|30/35|6/8|
|62|.0801703|30/35|7/8|
|100|.1884401|31/35|6/8|

R38→R62，TD下降、Validation改善但Train未改善；R62→R100，TD上升、Validation下降而Train略增。**不存在TD单调对应闭环成功的证据**。采样池、策略和参考目标随轮次变化，不能把表当成受控因果实验。

最终Replay冻结四类MAE（此前完整池审计）普通.07696、正常终止.14386、失败前缀.17799、失败终止.62796；它与本次Greedy访问分布不同，不能混用。此前冻结目标副本能改善普通/失败前缀，但失败终止未共同稳健改善，支持继续查拟合与分布，而非先把问题全归为三维表达能力。

## 7. 岸电标志及其他候选风险

- 独立数据加载器 `IndependentEpisode.operating_mode` 对每行返回onboard，并禁止独立任务非onboard；Train35/Validation8无SHORE/UNKNOWN，不存在中途补能或终端充电。
- 所以新增“是否岸电”在当前训练任务恒定，没有有效区分信息，不建议为了模仿论文加入。
- 因果负载差分真实可观测，允许作为后续单变量候选，但目前收益弱。FC差分已由动作历史可获得，是否有额外信息本次未证明；当前一步FC费用已显式依赖上一FC，无需为拟合即时变载成本机械恢复旧7维。
- 原始实船FC/原始SOC不是DDQN反事实物理状态，不能用来补“历史”；实际未来负载、航段结束标签、剩余步数和未来岸电均禁止作为当前输入。
- 累计寿命账本已保留，当前R100轨迹未触及FC寿命封顶；若以后任务跨寿命封顶，其隐藏状态可能改变费用增量，但不能据本次猜测直接加寿命比例。

## 8. 是否值得启动状态消融

**当前结论：暂不启动状态扩维正式实验，保留三维。** 有状态别名现象，但没有证明单差分能稳定解除与控制风险相关的混叠；当前更直接的证据是长期回报估计偏差、Online/Target动作优势差及已确认的增量可达性缺口。

只有后续用同样的parent隔离证据证明一个可在线观测量能稳健降低关键状态条件不确定性/失败风险不可分性，才考虑唯一变量的3维vs4维（加一个因果负载差分）对照。不得同时恢复多步、改gamma、奖励、动作或隐藏层。

届时3–128–128–7与4–128–128–7只能改变必要输入层，各自新初始化、空Replay、新Adam，不能续训旧模型或伪称输入维度不同还共享逐张量相同初值。使用配对seed、固定Train35/Validation8、相同环境步/梯度预算、target更新预算、epsilon暴露、reward及实际更新时序；不同完成长度导致同轮数不等预算，必须预注册共同步数检查点与epsilon协议，不能简单各100轮声称完全等预算。

验收：固定完整样本供电率/失败位置、同一共同完成集合Zhou成本、SOC储备、FC反向次数及平均变载、每类未加权TD、固定风险状态Q排序/实际路径回报校准。只降低负载预测MAE或TD、但闭环不改善，不能认为扩维有效。Validation本次已用于诊断，需披露选择过程；Test继续封存。

## 9. 产物和限制

新增仅本目录：`actual_states_q_returns.csv`、`identical_state_action_groups.csv`、`near_state_pairs.csv`、`near_state_distance_quartiles.csv`、`causal_history_prediction.csv`、`conditional_neighbor_dispersion.csv`、`category_q_return_diagnostics.csv`、`td_vs_completion.csv`、`verification.json`。

未来负载标签用于诊断，不是预测准确未来的在线策略前提；一条历史负载并不能识别真实随机生成机制。本次没有证明三维严格Markov、没有证明扩维无效，也没有把有限轨迹回报当Q最优标签。

受保护源码、checkpoint、原始档案SHA前后一致；正式Q更新0、策略回放0、Test载荷读取0、Git提交0。保留全部已有产物及本地未提交修改。
'''
with (OUT/'state_information_audit.md').open('x',encoding='utf-8') as f:f.write(text)
print('Report written')
