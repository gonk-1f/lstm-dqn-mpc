"""Summarize existing evidence and this read-only audit; never trains a model."""
from pathlib import Path
import csv
import json
from collections import defaultdict
import numpy as np

OUT=Path(__file__).resolve().parent
RUN=OUT.parent

def read(path):
    return list(csv.DictReader(path.open(encoding='utf-8-sig')))

def write(name,rows):
    with (OUT/name).open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def md(headers,rows):
    return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+
                     ['|'+'|'.join(map(str,row))+'|' for row in rows])

report=json.loads((RUN/'report.json').read_text(encoding='utf-8'))
verified=json.loads((OUT/'verification.json').read_text(encoding='utf-8'))
steps=read(OUT/'reward_step_components.csv')
voyages=read(OUT/'reward_voyage_totals.csv')
stats=read(OUT/'reward_component_statistics.csv')
td=read(OUT/'snapshot_td_q_target_statistics.csv')
scale=read(OUT/'analytical_td_rescaling.csv')
names={'h2_actual':'实际氢耗','soc_equivalent':'SOC等效氢耗','fc_degradation':'FC退化',
       'battery_degradation':'电池退化','soc_aux':'SOC辅助惩罚','fc_squared_ramp_aux':'平方变载辅助',
       'failure':'失败惩罚','shore':'岸电','legacy_adjustment_correction':'旧势函数修正'}
contexts={'exploratory':'探索Train','train':'Greedy Train','validation':'Greedy Validation'}
categories={'ordinary_nonterminal':'普通非终止','normal_terminal':'正常终止',
            'failure_prefix':'失败前缀','failure_terminal':'失败终止'}

# Original training logs describe changing networks and prioritized draws;
# the frozen pool below describes a different, single final snapshot.
sampled=[]
for roundrow in report['rounds']:
    s=roundrow['td_statistics']
    groups=dict(s['nonterminal_categories'])
    groups.update(normal_terminal=s['normal_terminal'],failure_terminal=s['failure_terminal'])
    for cat in categories:
        g=groups[cat]
        sampled.append(dict(round=roundrow['round'],category=cat,sample_count=g['sample_count'],
            td_mae=g['mean_absolute_td_error'],max_abs_td=g['max_absolute_td_error'],
            mean_q=g.get('mean_q_scaled','not logged by category'),
            mean_target=g.get('mean_target_scaled','not logged by category'),
            strictly_above_one_count=g.get('absolute_td_above_one_count','not logged'),
            strictly_above_one_fraction=g.get('absolute_td_above_one_fraction','not logged'),
            exact_equal_one_count='not logged; >1 cannot certify >=1 or <1 fractions',
            weighted_loss=s['mean_smooth_l1_loss'],
            unweighted_loss=s['replay_sampling']['mean_unweighted_smooth_l1_loss'],
            mean_is_weight=s['replay_sampling']['mean_importance_weight'],
            mean_gradient_norm=s['gradient_statistics']['mean_preclip_norm'],
            scope='actual sampled pre-update observations across this round; not frozen R100 pool'))
write('logged_sampled_td_by_round.csv',sampled)

# Reuse exact, previously reconstructed candidate interval costs; do not replay a policy.
decomp=read(RUN/'checkpoint_audit_round100/reward_decomposition.csv')
grouped=defaultdict(dict)
for r in decomp:
    for role in r['candidate_role'].split('|'):
        grouped[(r['sample_id'],r['step_1based'])][role]=r
contrasts=[]
costfields=('h2_cost_cny','signed_soc_equivalent_hydrogen_cost_cny','fuel_cell_degradation_cost_cny',
            'battery_degradation_cost_cny','soc_auxiliary_reward_equivalent_penalty',
            'squared_ramp_auxiliary_reward_equivalent_penalty')
for sample in ('parent_020_seg01','parent_015_seg01'):
    pairs=[(g['actual'],g['hold']) for (sid,_),g in grouped.items() if sid==sample and 'actual' in g and 'hold' in g]
    if not pairs:continue
    for field in costfields:
        diff=np.array([float(a[field])-float(h[field]) for a,h in pairs])
        contrasts.append(dict(sample_id=sample,matched_hold_legal_steps=len(pairs),component=field,
            mean_actual_minus_hold=float(diff.mean()),median_actual_minus_hold=float(np.median(diff)),
            p95_abs_difference=float(np.quantile(abs(diff),.95)),
            scope='exact existing one-step counterfactual; not a full alternative trajectory'))
write('existing_actual_vs_hold_cost_differences.csv',contrasts)

qrows=[]
for sample in ('020','015'):
    for r in read(RUN/f'checkpoint_audit_round100/q_action_audit_parent{sample}.csv'):
        for arm,factor in [('A',1),('B',10),('C',1000),('D',1000)]:
            qrows.append(dict(sample_id=r['sample_id'],step=r['step_1based'],arm=arm,
                best_minus_second=float(r['best_minus_second_q'])*factor,
                best_minus_hold=(float(r['best_minus_hold_q'])*factor if r['best_minus_hold_q'] else ''),
                scope='A actual archived R100 Q; B/C/D hypothetical co-scaled Q, not new predictions'))
write('q_margin_scale_conversion.csv',qrows)

fc=[]
aggregate=[]
for context in contexts:
    for status in ('completed','failed_prefix'):
        vv=[v for v in voyages if v['context']==context and v['status']==status]
        ff=[sum(float(v[k]) for v in vv) for k in ('fc_low','fc_high','fc_transient','fc_start')]
        total=sum(ff)
        for key,val in zip(('low_runtime','high_runtime','transient','start_cycle'),ff):
            fc.append(dict(context=context,status=status,component=key,total_cny=val,
                share_of_fc_degradation=val/total,scope='this exact R100 subset only; failed values are observed prefix'))
        aggregate.append([contexts[context], '完整' if status=='completed' else '失败前缀',len(vv),
            sum(int(v['steps']) for v in vv),*[f'{value:.2f}' for value in ff]])
write('fc_degradation_subtotals.csv',fc)

plans=[]
for arm,s,b in [('A',.001,1.),('B',.01,1.),('C',1.,1.),('D',1.,1000.)]:
    plans.append(dict(arm=arm,reward_scale=s,smooth_l1_beta=b,seed=42,gamma=1,n_step=1,
        update_timing='interleaved',cadence='replay32',batch_size=64,learning_rate=.001,
        state_mode='minimal3',action_mode='increment7',reward_cost_mode='zhou_joint',beta_soc=200,
        target_tau=.001,per_alpha=.6,per_beta=1.,per_epsilon=1e-6,failure_terminal_quota=2,
        fresh_model=True,fresh_adam=True,empty_initial_replay=True,
        phase1='same bootstrap and first exploratory round epsilon=1; no execution authorized',
        required_gate='approve CLI loss-beta/scale passthrough and one-round budget; hash common initial tensors',
        initialization='identical unscaled initial network tensors; not the historical R100 model',
        output_directory=f'reward_scale_interleaved_{arm}_seed42_NEW_UNIQUE_DIRECTORY'))
write('proposed_scale_experiment_matrix.csv',plans)

mainstats=[r for r in stats if r['context']=='train' and r['status']=='completed' and r['group']=='all']
table_main=md(['分项','均值','绝对中位数','绝对P90','绝对P95','有符号累计','绝对累计占比','单步最大频率'],[
    [names[r['component']],*[f'{float(r[k]):.5f}' for k in ('mean','abs_median','abs_p90','abs_p95')],
     f"{float(r['signed_sum']):.2f}",f"{float(r['absolute_component_share'])*100:.2f}%",
     f"{float(r['largest_absolute_component_fraction'])*100:.2f}%"] for r in mainstats])
table_td=md(['类别','池内数量','TD MAE','绝对TD P95','二次区间比例','线性区间比例'],[
    [categories[r['category']],r['count'],f"{float(r['abs_mean']):.6f}",f"{float(r['abs_p95']):.6f}",
     f"{float(r['quadratic_fraction'])*100:.4f}%",f"{float(r['linear_fraction'])*100:.4f}%"]
    for r in td if r['quantity']=='td' and r['category']!='all'])
table_scales=md(['组','scale','SmoothL1 beta','假设同尺度转换TD MAE','二次区间比例','线性区间比例','未加权Loss'],[
    [r['arm'],r['scale'],r['beta'],f"{float(r['td_mae']):.6f}",
     f"{float(r['quadratic_fraction'])*100:.3f}%",f"{float(r['linear_fraction'])*100:.3f}%",
     f"{float(r['unweighted_smooth_l1']):.6f}"] for r in scale if r['category']=='all'])
binrows=[r for r in stats if r['context']=='train' and r['status']=='completed' and r['group']=='soc_after' and r['component']=='soc_aux']
table_bins=md(['执行后SOC区间','步数','SOC惩罚均值','该区间绝对分项占比'],[
    [r['bin'],r['count'],f"{float(r['mean']):.4f}",f"{float(r['absolute_component_share'])*100:.2f}%"] for r in binrows])

text=f'''# 第100轮奖励分项与 reward_scale 定量审计

日期：2026-10-19。仅离线核算、档案读取及CPU前向计算；没有训练、反向传播、优化器调用、Test读取、Git提交或正式源码修改。

## 1. 身份、范围和证据强度

- 本地分支 `feat/v4-increment7-zhou-td1`，HEAD `dc3c1947bc4dc400710896dfe3acca2e9ae3f54d`。
- 已完成实验 `V4-INDEP35-M3-DELTA7-ZHOU-TD1-PER-S42`，训练源码提交 `82e21cc46fef163b8680550e35252e2f7c8fa0db`，源码指纹 `3edf333281cbe23024ba2c6f2bfdd18cb3c564e2bf557ef9dca20923ca0dc812`。数据身份详见 `verification.json` 中原始 metadata；不读取Test载荷。
- 实际100轮采用 **航段结束集中更新**；本地未提交的交错更新接口尚无正式实验结果。以下历史统计不能表述为交错更新效果。
- 本地原有5个tracked修改、4个untracked文件均保留；包括交错更新和replay16准备。本次新增仅此诊断目录。
- 现有配置：Train35/Validation8；minimal3，3–128–128–7；增量±60/±30/±10/0；单步TD、gamma=1、Adam=.001、Batch64、Replay300000、replay32、tau=.001；PER alpha=.6、IS beta=1、epsilon=1e-6、失败配额2；SOC beta=200、参考.6、reward_scale=.001。无SHORE、无终端充电。
- 使用第100轮探索Train35、Greedy Train35、Validation8，共78条实际轨迹、**30,174步**，逐条恢复正式SOC/FC寿命/电池寿命核算。完整情况分别25/35、31/35、6/8；不存在合格best checkpoint。
- 这是第100轮的定量分项审计，**不是重算全部100轮所有大轨迹**。历史100轮TD日志复用既有report；最终Replay的300000条经验用于单一冻结快照统计。失败航段只分析已观测前缀，不作完整费用优劣比较。

## 2. 从执行路径还原公式与单位

令 d_t=SOC_t−SOC_(t+1)，ΔP_t=P_FC,t−P_FC,t−1。正常执行步未经缩放奖励为：

`r_raw = −[C_H2,actual + C_SOC,eq + C_FC,deg + C_Bat,deg + 200*(SOC_next−.6)^2 + (.11025/60)*ΔP^2]`。

真实失败在最后一个已执行动作加一次 `−F`，**F=10132.660087898294**，不虚构下一动作或补能。最终进入Replay：`r=.001*r_raw`（包含该次失败惩罚）。

| 分项 | 定义、符号及单位 | 时机/权重 | 执行位置（当前本地行号） |
|---|---|---|---|
|实际氢耗|`21.9*m_H2`；`m=P*dt_h/(eta_FC(P)*LHV)`，P=0取0；实际氢成本元|每个执行区间，权重1|`src/v3/control.py:134–169` 调用正式效率图及账本；`src/v4/control.py:51–59`|
|SOC等效氢耗|`(SOC_before−SOC_next)*624*21.9/(eta_ref*LHV)`；放电为正、充电为负；参考等效元，**并非实际购氢支出**|每个执行区间，权重1|`src/v4/independent_economics.py:127–149`；`src/v4/control.py:498–503`|
|FC退化|`1050000*[min(L_after/70000,1)−min(L_before/70000,1)]` 元|每区间，权重1；保留寿命上限|`src/v3/control.py:146–168`、`src/v2/economics.py:425`|
|电池退化|既有设备价值×截断至1的累计加权Ah寿命比例差；加权Ah=`abs(I)*dt_h*f_SOC(SOC_before)*g_I(I)`；元|充放电都计，权重1；电流按现有母线功率/额定电压|`src/v3/control.py:151–168`、`src/v2/models/battery_degradation.py:99`、`:302`、`:377`|
|SOC辅助|`200*(SOC_next−.6)^2`，非负，训练等效量|每个真实ONBOARD动作一次；使用动作后SOC|`src/v4/control.py:493–503`|
|平方变载辅助|`.0018375*ΔP^2`，非负，训练等效量；不是额外人民币账单|每个真实ONBOARD动作一次；实际前后FC功率|`src/v4/control.py:43–48`、`:498–503`|
|失败|F，非负，训练等效量；不进入经济账本|真实无可行动作时附着最后有效transition一次|`src/v4/failure_replay.py:59–91`|
|终端/势函数/岸电|当前独立航段均为0；正常终止不bootstrap；失败也不bootstrap|无终端二次SOC折算，无充电|`src/v4/control.py:535–555` 独立分支及redistribution开关|
|统一缩放|`.001`，所有训练分项及失败项共同缩放|Agent.remember入池一次；learn读取已缩放reward不再缩放|`src/v4/dqn.py:241`、`:482–515`|

eta_ref=0.5711999752720877，LHV=33.333333333333336 kWh/kg；SOC等效系数为 **717.7311235 元/SOC**。eta_ref是固定曲线算术平均的研究假设；实际氢耗仍用瞬时效率。SOC能量模型已经含电池效率，此处不再除一次电池效率。

FC原始退化四项在本轮没有触及寿命截断，故可精确展开：运行时P在(0,480)kW为15*10.17*dt_h元，P≥480为15*11.74*dt_h元；变载 `.11025*abs(ΔP)` 元；0→正功率记一次启动循环，15*23.91=358.65元，下降到0本身不再另收一次完整循环。600kW聚合到100kW参考单元的1/6映射仅用于变载项，不能把常数重复乘6。

### 2.1 缩放及账本逐步对账结果

- 30,174步SOC、电池功率、原四项物理费用、SOC等效项和辅助项均按正式核算重建。
- 原始总奖励最大误差 **5.69e-14**；缩放奖励最大误差 **5.56e-17**。
- 本轮探索经验 **12,517/12,517** 与真实Replay的状态、动作索引、已缩放reward一致；最大reward误差5.56e-17。
- 16条失败轨迹（探索10/Greedy Train4/Validation2）各有且仅有一次失败罚项；正常轨迹0次。正常终止不附加失败罚项。
- 每航段逐步SOC等效项与首尾SOC折算误差≤4.55e-13；完整Zhou评价对账误差≤8.37e-11。
- 档案把未发生的岸电/终端补能表示为**全0账本字典**，不是全部None；已逐项确认数值为0。这不是补能收费。
- 未发现此批记录重复缩放、失败重复收费或终端等效氢耗重复计入。这里的结论限于实际路径与本次对账，不能替代未来新模式的验证。

## 3. 分项统计：规模不等于动作影响

### 3.1 第100轮Greedy Train完整的31条航段、13200步

以下数值均为**缩放前分项值**；真实成本与训练辅助等效量混列仅用于解释奖励，不能把总和称为人民币费用。绝对份额分母为逐步所有分项绝对值之和，SOC等效项的充放电抵消另见有符号累计。

{table_main}

这31条的Zhou完整经济费用115101.96331元，SOC辅助62224.74037等效量、平方变载辅助33865.49250等效量，训练累计回报−211192.19618，缩放后−211.192196。仅描述**这一固定完成集合**，不能与另一轮不同完成集合总额比较。

SOC等效项净额−1882.48元，但绝对份额6.94%；净额负不能解释为免费电池，也不能用绝对份额代替净能量评价。

### 3.2 完整与失败分开

所有78条航段分别保存于 `reward_voyage_totals.csv`，包含每个分项有符号累计、绝对累计、回报及明确的失败前缀标志。

- 探索完整25条10748步：SOC辅助绝对份额42.68%、FC退化27.29%、平方变载8.43%。
- Greedy Train失败4条731步：失败罚项绝对份额68.33%，SOC辅助11.35%、FC退化8.43%；其前缀经济费用9508.85元**不是完成这些航段的经济费用**。
- Validation完整6条3276步：FC退化34.84%、SOC辅助23.55%、平方变载15.18%；失败2条450步的失败罚项份额65.83%。这两个集合不能直接相互比较经济优劣。
- 稀疏失败罚项中位数和P95可为0，因为它只发生于尾部一次；不能据此称失败风险不重要。单次值为10132.66（缩放后10.13266）。

### 3.3 负载/SOC条件分布

完整条件表见 `reward_component_statistics.csv`：每个context、完成状态分别划分负载[0,200)、[200,600)、[600,1000)、≥1000 kW，以及动作后SOC区间，不跨类别混算。

{table_bins}

Greedy Train完整集合：负载<200kW时FC退化均值6.6930、SOC辅助4.9598；600–1000kW时SOC辅助均值9.0684、绝对份额38.05%。≥1000kW仅10个**完整集合**观察点，不能据此推断所有高负载的分布，失败样本另列。

SOC辅助在低SOC<.4及>.79时分别占该区间绝对分项的61.64%、57.95%。它确实显著塑造目标，并非数值小到完全无作用。

### 3.4 FC账本已有变载成本

{md(['上下文','结局','航段数','执行步数','低载运行元','高载运行元','线性变载元','启动元'],aggregate)}

完整Greedy Train的FC退化中，线性变载50545.215元，占**64.51%**；启动11118.15元（31次启动循环）。平方变载辅助另为33865.4925等效量，**不再进入Zhou经济费用**。

ΔP=10/30/60kW时，线性账本变载为1.1025/3.3075/6.615元，辅助平方项为.18375/1.65375/6.615等效量。最大档已经受到两层平稳性偏好；这属于已选联合目标，**不是账本重复收费错误**。在当前≤60档内，平方项≤线性变载项，因此它几乎不可能成为单步最大的单独分项；“最大分项频率0”不代表它没有影响。

## 4. SOC两个目标是否重复或冲突

1. **净能量修正**：gamma=1、完整航段、系数固定时，`sum C_SOC,eq = K*(.6−SOC_T)`。消耗净储能付参考代价，净增加储能获得参考价值，不要求中间SOC等于.6，也不模拟真实岸电。
2. **轨迹偏好**：`sum 200*(SOC_t−.6)^2`与中途偏离多久相关，即使最终SOC相同，不同轨迹仍不同。它不是同一笔购氢费用，不能直接以“双重计费”删除。
3. **潜在冲突**：峰值负载需要提前充至>.6时，上侧惩罚会反对储备；允许经济使用电池时，下侧惩罚会反对深放电。高SOC储能信用与对高SOC的路径惩罚可能拉向不同方向。是否不合理要用完整供电和同集合经济结果判断。
4. 原始辅助量：SOC=.4/.8各8，SOC=.2为32，SOC=.79为7.22；分别缩放为.008/.032/.00722。不能把beta=200误当成每步总会罚200。
5. **有限失败罚与长期负回报的尺度**：完整Greedy Train31条中8条的整航段原始负回报绝对值超过F（范围最大18515.14）。这证明F不是“任何情况下完成都优先”的数学保证；并不证明提前失败就一定更优，因为还需相同分岔状态的可行成功路径、失败前缀费用及剩余代价对照。F来源仍为旧30Train参考成本P95，不是本独立航段联合目标的标定结果。它是需进一步核查的适用性问题，不能在此自行增大。

### 4.1 哪些项真正解释020反向变载

复用已存在的正式模型反事实分解：020的64个状态保持全部合法，**64/64实际变载即时奖励差于保持**。实际−保持的平均分项差：FC退化+4.392773、平方变载+3.281660；氢耗−.004430、SOC等效+.003662、电池−.003626、SOC辅助−.000223。合计实际即时奖励平均低 **.00766982（缩放后）**。

这些配对差比“全局SOC累计很大”更接近动作选择机制：此窗口的反复变载并非即时奖励缺少变载代价所鼓励。已有Q证据为64/64 Online仍选变载、40/64冻结Bellman目标更偏保持；bootstrap含估计误差，不能当真值，但足以反对“只因平方项太小，直接加大就能解决”的结论。

独立Zhou评价仅含`实际H2+首尾SOC等效H2+FC退化+电池退化`；SOC辅助、平方辅助及失败罚都不计经济费用。合格模型仍需完成资格达标，并按固定Validation集合Zhou费用选模，因此训练目标与选模目标**有意不同**。

## 5. reward_scale与Huber：冻结快照证据

### 5.1 数值方法及限制

PyTorch导入因系统提交内存不足报WinError1455，未能用本轮新Torch进程读取模型。因此使用只读zip/pickle白名单读取真实CPU张量和Replay，以NumPy float32执行同一三层ReLU、Masked Double-DQN前向计算；没有创建优化器或调用学习。

结果与之前真实PyTorch记录交叉核对：155个Train015状态的7档Q最大差1.91e-6；四类别TD MAE最大差1.99e-7。Replay没有|δ|距离1小于2e-5的点，故以下Huber分段比例不受这一级浮点差异影响。**数值近似复核，不宣称与Torch逐bit一致**。

### 5.2 最终Replay300000条、同一冻结网络

{table_td}

总体TD绝对值中位数.0621774、P90=.1261375、P95=.1692416、P99=.5893969，MAE=.0900325；**99.4873%二次区间、.5127%线性区间**。

选定动作Q范围[−10.675098,.449900]，目标范围[−10.543148,.365853]。完整Q/Y类别范围及分位数见 `snapshot_td_q_target_statistics.csv`，不是用奖励范围推测Q范围。

正常终止Q均值+.093731，对应目标均值−.0121774，残差均值−.105908；终止不bootstrap，但价值仍偏高，是拟合偏差证据，不是终止奖励重复收费。

### 5.3 实际训练日志与冻结池不能混同

第100轮391次更新、25024次抽样，训练期间网络和目标在变化，且PER有偏抽样：TD MAE=.188440；失败前缀=.517942，普通=.111643，正常终止=.188765，失败终止=.543617。

日志普通20273次抽样中370次严格|δ|>1（1.8251%），正常终止87中1次（1.1494%），失败前缀3882中384次（9.8918%）。原记录用**>1而非>=1**，未保存等于1的计数及失败终止分段计数，因此不能把这些日志比率直接称为精确>=1比例。最终池的>=1比例已在本次按定义计算。`logged_sampled_td_by_round.csv`保留全部100轮已有分组数值及缺失标记，不编造训练期间TD分位数。

## 6. 是否存在“梯度太小所以Adam没学习”

- R100未加权SmoothL1=.109848917，PER加权=.0000273139，平均IS=.00100260；平均裁剪前梯度L2=.000632143，最大.00785901，0次达到clip10。
- IS权重与逐样本损失相关，所以不能用“平均IS×平均Loss”精确还原加权Loss。
- 从最终真实Adam保存的m、v、step，按零weight_decay、无AMSGrad的公式重建最后一次更新：`Δθ=−lr*(m/(1−b1^t))/(sqrt(v/(1−b2^t))+eps)`。17927个参数总更新L2 **.00990952**，相对最终参数L2 **.0237038%**。这是实际最终moments对应的更新向量推算，存在float32应用舍入误差；不是测得整轮平均，也不能由最终moments反推出历史每次位移。
- 复用之前独立冻结目标副本的实测200次更新：PER平均梯度约.000750，平均实际Adam位移L2约.013257，最终参数相对位移1.0913%；Uniform约.039346、3.4102%。这是**已有独立监督拟合实验**，不是本次正式在线训练的更新均值，也不是scale对照。
- 因而“加权loss很小→参数不更新”被反驳；小权重是否降低各类别共同拟合效率仍是实验问题。Adam对固定整体梯度比例有一定抵消作用，但Huber分段、eps、裁剪、moments历史和样本权重变化使其不严格尺度不变。

## 7. A/B/C/D的量级分析——不是已训练结果

PyTorch正式实现是SmoothL1（beta默认1）：`L=.5*δ²/beta`（|δ|<beta），否则`abs(δ)−.5*beta`。这与未除beta的某些Huber定义不能混用。[官方SmoothL1定义](https://docs.pytorch.org/docs/2.14/generated/torch.nn.SmoothL1Loss.html)。Adam更新公式见[官方Adam文档](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Adam.html)。

全30,174条已有奖励：原始绝对中位数12.57484、P95=36.33494；A后为.01257484/.03633494；B后为.1257484/.3633494；C/D回到原始量级。失败尾部最大绝对单步约10182.0024，A后10.1820024；正常步各分项的缩放后中位数/P95已单列CSV。

下表**假设Q和Y都同比换单位**，仅隔离尺度和阈值效应；不是从相同初始权重训练出的B/C/D成绩，也不是实际新网络目标分布：

{table_scales}

- A在原始回报等效单位对应Huber转折1000；B为100；C为1；D为1000。直接去掉缩放并保留beta1会使同一假设残差的**99.717%进入线性段**，不是单纯把原有训练所有梯度放大1000倍。
- D相对A的δ、beta都×1000，同尺度假设下分段比例相同、SmoothL1数值×1000；不是×1000000。对Q的导数分段形态对应，但fresh网络、Adam状态和训练路径不会因此自动等价。
- 理论最优Q整体正比例变化不改变argmax；但新实验要求相同初始网络，初始Q不随reward同比放大，因此相对初始乐观性、输出层拟合行程及目标漂移不同。Q优势应同时报告原始数值和除以scale的换算值；纯数值变大不等于排名更可靠。
- PER优先级`(abs(δ)+1e-6)^.6`在忽略epsilon时对整体同倍率近似保留相对概率；固定epsilon、历史优先级滞后和分段学习改变δ分布后不能保证相同。IS仍按当前全局最小概率归一化，不能未经批准改变归一化补偿尺度。[PER原论文](https://arxiv.org/html/1511.05952v4)。

## 8. 权重/项目建议（仅建议，不实施）

|项目|本次建议|证据与限制|
|---|---|---|
|实际H2、SOC等效、FC/电池退化|保留当前定义与权重1|与Zhou账本对账；没有重复终端折算证据；删除会改变经济任务|
|SOC beta=200|保持作基线，不再凭不收敛直接提高到300/400；完成scale对照后才考虑单独200对100的候选|完整Train中占27.18%，低/高SOC区间占比显著；可与预充储备冲突；降低也可能增加耗尽风险，尚无闭环因果证据|
|平方变载系数.0018375|当前保留，不直接增大；若以后做奖励目标消融，可单独与0或半值.00091875比较，不能与scale同时改|020即时目标已经偏好保持；FC线性变载占自身退化64.51%；不能据振荡认定惩罚不足|
|失败罚F|保留现值；单列旧参考集与新联合目标的适用性风险，未批准前不增减|稀疏尾部值主导失败前缀绝对总量，仍不保证完成优先；需要同状态成功/失败剩余成本对照|
|岸电、终端充电、旧势函数|维持独立航段中的关闭状态；无须新增删除补丁|实际全部0，已确认不重复结算|
|reward_scale|**暂保留.001作为可复现对照；没有证据支持直接删除**|不存在重复缩放；实际Adam更新非零；存在Huber相对阈值效应，值得受控检验|

上述权重候选仅回答后续可检验方向，**不是授权实施，也不与本轮A–D混改**。首要仍是价值共同拟合、目标漂移和训练时序；本审计没有把任何一项确认为唯一根因。

## 9. 最小可执行消融设计（待确认，未运行）

### 9.1 保留全部四组

{md(['组','reward_scale','SmoothL1 beta','目的'],[['A','.001','1','当前尺度参考'],['B','.01','1','提高10倍及相对阈值变化'],['C','1','1','直接去掉缩放的真实候选'],['D','1','1000','保留相对阈值的诊断对照']])}

统一minimal3/increment7/Zhou联合目标、beta_soc200、gamma1、n1、Adam.001、Batch64、PER .6/1/1e-6、quota2、tau.001、replay32；只允许表中scale和SmoothL1 beta变化。Train35/Val8固定，Test封存；数据manifest/源码指纹、依赖版本、初始Online及Target张量SHA、空Replay、新Adam/RNG身份全部保存。所有组使用同一套**航段内交错更新**，不要用旧集中更新A历史结果当唯一对照。

当前 `feedback_study.py:89–93` 明确拒绝scale≠.001，`dqn.py:505–508` 的SmoothL1 beta目前固定1。**因此现有正式命令无法执行B/C/D**。批准后需要最小地显式传递scale及loss beta，写入resume身份、元数据和统计；本次不改源码，不提供虚假的可运行训练命令。

### 9.2 首批预算：四组各相同bootstrap + 第一轮探索（epsilon=1）

建议先做四组各一个完整首轮，而非四组100轮。相同初始张量、环境/动作/采样随机种子，bootstrap按原规则。首轮ε=1，贪心评价ε=0且恢复评价前RNG，不把评价写入Replay。该阶段通常能保持相同随机行为路径，**必须逐项校验实际动作/SOC/load序列哈希、环境步数、Replay插入数及梯度次数一致**；不能只假定相同seed就一定相同。若任何一项不一致，停止配对解释、查明随机调用或执行路径原因，不用人工覆盖动作强凑公平。

固定验证快照：初始化（可选，只前向）、bootstrap结束、首轮结束；需要更细分时按共同环境步/梯度计数保存同一组切点。每组预算记为bootstrap的U0次加首轮U1次，四组4*(U0+U1)，用实际交错运行计数登记。旧集中bootstrap记录468次可作规模参考，不能把旧首轮步数当成新交错预算的已知值。新目录、新权重、新Replay和新Adam，禁止从R100续训。

### 9.3 为什么不能先承诺“同100轮就是同预算”

进入ε<1后，各组策略、失败位置、航段长度会分化；原ε按轮衰减时，**同轮数通常不同环境步及优化次数**。固定环境步预算又会使轮次/ε进度不同。这是控制条件冲突，不能静默忽略。

首批ε=1的短诊断先满足严格匹配。后续长程对照若获批准，需预先共同决定预算协议：使用固定环境步的共同ε日历并把该变化对所有组一致应用，或保留原轮次日历、在共同环境/更新切点作有限配对并披露非等ε暴露。不能在本任务里自行改ε、扩大训练或声称完全匹配。首轮不足以判断策略最终收敛。

### 9.4 固定指标与验收/证伪

- 先看固定35/8完整供电成功率、物理失败类型和位置，不忽略失败样本；成本只在四组共同完成的同一ID集合比较，未全部完成不冒充完整总体经济最优。
- 同集合Zhou总成本及三分项；SOC min/max、低/高SOC停留、末端SOC；平均绝对FC变载、最大变载、±60次数、相邻反向次数。当前动作上限60，>100/200kW计数必为0，不能拿它证明平稳，预先约定≥60或>30为本实验的大跳变口径。
- 每类**未加权**TD MAE/P90/P95及Huber分段比例；PER加权/未加权loss分开；梯度裁剪前后、Adam每次真实参数差L2/相对L2；动作Q优势同时报告原生尺度和除scale后的等效尺度。
- 对020固定状态和015风险前缀比较Q排名、保持动作优势及Online/Target差；固定状态诊断不代替各策略真实闭环。
- 支持改变尺度：在相同预算下，类别误差共同改善且归一化Q排名更可靠，闭环成功率不下降、同集合成本及反向变载改善，并经后续更长预算/随机种子复核。
- 证伪“缩放是主因”：只改变loss数值/梯度量级，归一化误差、排名、成功率或经济性无改善/恶化；C恶化而D改善提示相对Huber阈值重要；B/C/D均不能共同拟合则应继续查共享网络冲突、状态别名和bootstrap，不继续单纯放大奖励。
- 所有四组报告完整配置和源码指纹；短程结果只能决定是否值得追加预算，不能宣告正式收敛或默认删掉表现不好的对照组。

## 10. 文件与保护

- `reward_step_components.csv`：30174步原始/缩放奖励及各分项、FC四子项。
- `reward_component_statistics.csv`：完成/失败、训练/验证、负载及SOC分组均值/分位/份额/最大频率。
- `reward_voyage_totals.csv`：78条航段逐项累计（失败明确为前缀）。
- `fc_degradation_subtotals.csv`、`existing_actual_vs_hold_cost_differences.csv`：FC账本分解和已有精确配对证据。
- `reward_scale_magnitudes.csv`、`snapshot_td_q_target_statistics.csv`、`analytical_td_rescaling.csv`、`q_margin_scale_conversion.csv`：真实量级及明确标注的假设换单位量级。
- `logged_sampled_td_by_round.csv`：100轮既有日志；缺失字段保留未知。
- `adam_final_step_reconstruction.csv`：真实最终moments的逐张量更新重建。
- `proposed_scale_experiment_matrix.csv`：待确认四组实验配置，不是已经执行的训练。
- `verification.json`、`protected_hashes_before.json`：计算误差、数据/实验身份及保护校验。

正式源码、模型/Replay/checkpoint、原始report及轨迹SHA保护校验全部一致。没有启动A/B/C/D，也没有增加正式配置、奖励项或人工动作机制。
'''
with (OUT/'reward_scale_audit.md').open('x',encoding='utf-8') as f:f.write(text)
print('Summary written; no experiments executed.')
