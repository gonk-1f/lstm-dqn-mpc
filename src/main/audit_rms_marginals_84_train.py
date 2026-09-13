"""Cached Train-only marginal action analysis. No solver/training imports."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'outputs/ideal_reference_84_train_20260912'
OUT=ROOT/'outputs/rms_marginals_84_train_20260912'
TERMS=('H','B','S','F')
REQUESTED_SOC=(.55,.45,.35,.25,.22)
LOAD_BINS=('<=300','300-600','600-900','>900')


def write(path,value):
    def convert(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,default=convert)+'\n',encoding='utf-8')


def dist(v):
    v=np.asarray(v,dtype=float).ravel();finite=v[np.isfinite(v)]
    if not len(finite):return dict(count=0,undefined_count=len(v))
    return dict(count=len(finite),undefined_count=len(v)-len(finite),mean=float(finite.mean()),
                **dict(zip(('min','p01','p50','p99','max'),map(float,np.quantile(finite,[0,.01,.5,.99,1])))))


def inputs():
    s=pd.read_csv(SOURCE/'states.csv')
    with np.load(SOURCE/'state_physical_results.npz') as d:
        values=d['f'][:,4:];fc=d['fc_plans'][:,4:];valid=d['valid'][:,4:]
    assert values.shape==(1440,84,4) and fc.shape==(1440,84,6) and valid.all()
    assert np.array_equal(s.audit_id,np.arange(1440)) and s.segment_id.str.startswith('train_').all()
    assert np.isfinite(values).all() and (values>=0).all()
    ci=values/6.;ci[:,:,0]**=2
    return s,ci,fc


def run():
    s,ci,fc=inputs();n=len(s);ix=np.arange(n);total=ci.sum(2)
    order=np.argsort(total,axis=1,kind='stable');a1,a2=order[:,0],order[:,1]
    delta=ci[ix,a2]-ci[ix,a1]
    gap=total[ix,a2]-total[ix,a1]
    sum_delta=np.array([math.fsum(row) for row in delta])
    sum_abs=np.abs(delta).sum(1)
    # All-identical term vectors have undefined A; never hide with epsilon.
    A=np.divide(abs(delta),sum_abs[:,None],out=np.full_like(delta,np.nan),where=sum_abs[:,None]!=0)
    ranges=np.ptp(ci,axis=1);sigma=np.std(ci,axis=1,ddof=0)
    residual=gap-sum_delta
    identity_tol=8*np.finfo(float).eps*(np.abs(ci[ix,a1]).sum(1)+np.abs(ci[ix,a2]).sum(1))
    assert (abs(residual)<=identity_tol).all() and (gap>=0).all()
    s['load_bin']=pd.cut(s.current_load_kw,[-np.inf,300,600,900,np.inf],labels=LOAD_BINS).astype(str)
    state=s.copy();state['a1']=a1;state['a2']=a2;state['C_a1']=total[ix,a1];state['C_a2']=total[ix,a2]
    state['delta_C']=gap;state['sum_signed_delta_Ci']=sum_delta;state['identity_residual']=residual
    state['identity_roundoff_bound']=identity_tol;state['sum_abs_delta_Ci']=sum_abs;state['A_defined']=sum_abs>0
    state['winner_fc0']=fc[ix,a1,0];state['runner_up_fc0']=fc[ix,a2,0]
    state['top2_fc0_difference_kw']=fc[ix,a2,0]-fc[ix,a1,0]
    state['top2_plan_rms_difference_kw']=np.sqrt(np.mean((fc[ix,a2]-fc[ix,a1])**2,axis=1))
    state['largest_R_term']=[TERMS[j] for j in ranges.argmax(1)]
    state['largest_sigma_term']=[TERMS[j] for j in sigma.argmax(1)]
    state['largest_A_term']=[TERMS[j] if np.isfinite(A[i]).all() else 'undefined' for i,j in enumerate(np.nan_to_num(A,nan=-1).argmax(1))]
    for j,t in enumerate(TERMS):
        state['R_'+t]=ranges[:,j];state['sigma_'+t]=sigma[:,j]
        state['signed_delta_'+t]=delta[:,j];state['abs_delta_'+t]=abs(delta[:,j]);state['A_'+t]=A[:,j]
        state['C_'+t+'_a1']=ci[ix,a1,j];state['C_'+t+'_a2']=ci[ix,a2,j]
    state.to_csv(OUT/'state_marginals.csv',index=False)
    omission=[]
    for j,t in enumerate(TERMS):
        # Recompute remaining terms to avoid subtracting a large state offset.
        reduced=np.delete(ci,j,axis=2).sum(2);wo=np.argmin(reduced,axis=1)
        d=s.copy();d['omitted_term']=t;d['full_winner']=a1;d['omitted_winner']=wo;d['winner_changed']=wo!=a1
        d['full_fc0']=fc[ix,a1,0];d['omitted_fc0']=fc[ix,wo,0]
        d['signed_fc0_difference_kw']=fc[ix,wo,0]-fc[ix,a1,0]
        d['abs_fc0_difference_kw']=abs(d.signed_fc0_difference_kw)
        d['plan_rms_difference_kw']=np.sqrt(np.mean((fc[ix,wo]-fc[ix,a1])**2,axis=1))
        d['full_cost_increase']=total[ix,wo]-total[ix,a1]
        omission.append(d)
    omissions=pd.concat(omission,ignore_index=True);omissions.to_csv(OUT/'omission_controls.csv',index=False)
    groups=[]
    for cohort in ('all1440','frozen840','matched_soc'):
        mask=np.ones(n,dtype=bool) if cohort=='all1440' else s.cohort.eq(cohort).to_numpy()
        soc_values=['all']+sorted(s.loc[mask,'soc'].unique().tolist())
        for soc in soc_values:
            sm=mask if soc=='all' else mask & s.soc.eq(soc).to_numpy()
            for load in ('all',*LOAD_BINS):
                ids=ix[sm if load=='all' else sm & s.load_bin.eq(load).to_numpy()]
                groups.append((cohort,soc,load,ids))
    margins,omits,group_notes=[],[],[]
    for cohort,soc,load,ids in groups:
        tag=dict(cohort_group=cohort,soc_group=soc,load_bin=load,states=len(ids))
        if not len(ids):
            group_notes.append(tag|{'note':'no saved states; no extrapolation or new solves'})
            continue
        for j,t in enumerate(TERMS):
            rec=tag|dict(term=t)
            for name,v in [('R',ranges[ids,j]),('sigma',sigma[ids,j]),('signed_delta',delta[ids,j]),
                           ('abs_delta',abs(delta[ids,j])),('A',A[ids,j])]:
                rec.update({name+'_'+k:v for k,v in dist(v).items()})
            rec['delta_favors_a1_count']=int((delta[ids,j]>0).sum())
            rec['delta_favors_a2_count']=int((delta[ids,j]<0).sum())
            rec['delta_exact_zero_count']=int((delta[ids,j]==0).sum())
            rec['largest_R_count']=int((ranges[ids].argmax(1)==j).sum())
            rec['largest_sigma_count']=int((sigma[ids].argmax(1)==j).sum())
            rec['largest_A_count']=int((np.nan_to_num(A[ids],nan=-1).argmax(1)==j).sum())
            margins.append(rec)
            a=omission[j].iloc[ids];changed=a[a.winner_changed]
            rec=tag|dict(omitted_term=t,winner_changed_count=len(changed),winner_change_rate=len(changed)/len(a))
            for kind,frame in [('all',a),('changed_only',changed)]:
                for col in ('signed_fc0_difference_kw','abs_fc0_difference_kw','plan_rms_difference_kw'):
                    rec.update({kind+'_'+col+'_'+k:v for k,v in dist(frame[col]).items()})
            # Diagnostic physical thresholds never affect ranking.
            rec['changed_and_fc_over_01kw']=int((changed.abs_fc0_difference_kw>.1).sum())
            rec['changed_and_fc_over_1kw']=int((changed.abs_fc0_difference_kw>1).sum())
            rec['changed_and_plan_rms_over_1kw']=int((changed.plan_rms_difference_kw>1).sum())
            omits.append(rec)
    pd.DataFrame(margins).to_csv(OUT/'group_marginals.csv',index=False)
    pd.DataFrame(omits).to_csv(OUT/'group_omission_controls.csv',index=False)
    write(OUT/'empty_groups.json',group_notes)
    write(OUT/'summary.json',{'states':n,'actions_per_state':84,'std_ddof':0,'ranking':'ascending C; exact ties by ascending action ID',
           'identity_residual':dist(residual),'max_absolute_identity_residual':float(abs(residual).max()),
           'undefined_A_states':int((sum_abs==0).sum()),'C_gap':dist(gap),
           'formula_unchanged':True,'soc_scale':.05,'new_solves':0,'dqn_trained':False,
           'top2_A_mean':dict(zip(TERMS,np.nanmean(A,axis=0))),
           'largest_R_counts':state.largest_R_term.value_counts().to_dict(),
           'largest_sigma_counts':state.largest_sigma_term.value_counts().to_dict(),
           'largest_A_counts':state.largest_A_term.value_counts().to_dict(),
           'load_range_kw':[float(s.current_load_kw.min()),float(s.current_load_kw.max())],
           'grouping_note':'All 1440 included; five-SOC tables use matched_soc 120 pairs to avoid duplicate weighting at SOC .45/.55.'})
    write(OUT/'provenance.json',{'inputs':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
          for p in [SOURCE/'states.csv',SOURCE/'state_physical_results.npz']},
          'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'new_mpc_solves':0,'validation_test_reads':False,'formula_or_scale_changes':False})
    print('ANALYZED 1440 x 84 cached actions; 5760 omission controls; all grouped outputs saved',flush=True)


def verify():
    s,ci,fc=inputs();state=pd.read_csv(OUT/'state_marginals.csv');om=pd.read_csv(OUT/'omission_controls.csv')
    assert len(state)==1440 and len(om)==5760
    total=np.sum(ci,axis=2);ix=np.arange(len(s));order=np.argsort(total,axis=1,kind='stable')
    np.testing.assert_array_equal(state.a1,order[:,0]);np.testing.assert_array_equal(state.a2,order[:,1])
    delta=ci[ix,order[:,1]]-ci[ix,order[:,0]]
    for j,t in enumerate(TERMS):
        np.testing.assert_allclose(state['signed_delta_'+t],delta[:,j],rtol=1e-12,atol=1e-17)
        np.testing.assert_allclose(state['R_'+t],np.max(ci[:,:,j],1)-np.min(ci[:,:,j],1),rtol=1e-12,atol=1e-17)
        np.testing.assert_allclose(state['sigma_'+t],np.sqrt(np.mean((ci[:,:,j]-np.mean(ci[:,:,j],1,keepdims=True))**2,axis=1)),rtol=1e-12,atol=1e-17)
        A=abs(delta[:,j])/np.sum(abs(delta),axis=1)
        np.testing.assert_allclose(state['A_'+t],A,rtol=1e-12,atol=1e-16)
        rows=om[om.omitted_term==t].sort_values('audit_id')
        ids=np.argmin(sum(ci[:,:,k] for k in range(4) if k!=j),axis=1)
        np.testing.assert_array_equal(rows.omitted_winner,ids)
        expected_fc=fc[ix,ids,0]-fc[ix,order[:,0],0]
        expected_rms=np.sqrt(np.mean((fc[ix,ids]-fc[ix,order[:,0]])**2,axis=1))
        np.testing.assert_allclose(rows.signed_fc0_difference_kw,expected_fc,rtol=1e-12,atol=1e-13)
        np.testing.assert_allclose(rows.plan_rms_difference_kw,expected_rms,rtol=1e-12,atol=1e-13)
    np.testing.assert_allclose(state[['A_'+t for t in TERMS]].sum(axis=1),1,atol=1e-14)
    # Cancellation example: signed terms cancel, while A remains bounded.
    d=np.array([1.,-1.+1e-10,0.,0.]);aa=abs(d)/sum(abs(d))
    assert np.all((aa>=0)&(aa<=1)) and abs(sum(aa)-1)<1e-15
    previous=json.loads((OUT/'protected_before.json').read_text(encoding='utf-8'))
    changed=[p for p,h in previous.items() if not (ROOT/p).exists() or hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h]
    assert not changed,changed
    provenance=json.loads((OUT/'provenance.json').read_text(encoding='utf-8'))
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in provenance['inputs'].items())
    write(OUT/'verification.json',{'states_verified':1440,'omission_control_rows_verified':5760,
          'protected_files_unchanged':len(previous),'signed_delta_identity_passed':True,
          'bounded_A_cancellation_check_passed':True,'new_solves':0,'dqn_trained':False})
    print('VERIFIED all marginal statistics and 5760 control comparisons;',len(previous),'original files unchanged',flush=True)


def report():
    m=pd.read_csv(OUT/'group_marginals.csv');o=pd.read_csv(OUT/'group_omission_controls.csv')
    state=pd.read_csv(OUT/'state_marginals.csv');summary=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
    def fmt(v):return 'NA' if pd.isna(v) else f'{float(v):.6g}'
    def table(headers,rows):
        return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+
                         ['| '+' | '.join(map(str,row))+' |' for row in rows])
    def mg(soc,load='all'):
        return m[(m.cohort_group=='matched_soc')&(m.soc_group==str(soc))&(m.load_bin==load)].set_index('term').loc[list(TERMS)]
    def og(soc,t,load='all'):
        return o[(o.cohort_group=='matched_soc')&(o.soc_group==str(soc))&(o.load_bin==load)&(o.omitted_term==t)].iloc[0]
    text=['# 固定RMS/L2与SOC_scale=0.05：84动作边际差异审核',
          '**结论：0.05主要降低低SOC状态的绝对reward，同时提供较温和的SOC恢复选择偏好；它没有系统性垄断同一低SOC状态下的84动作选择。当前不建议进入0.075/0.10/0.15扫描。**',
          '这个判断依据动作间范围、标准差、前两名带符号差值及移除单项后的控制差异，不使用任何项占绝对C的百分比。S并非对动作选择毫无作用：在SOC=.22时去掉S会改变所有120个winner，但实际控制变化远小于去掉H/F。',
          '## 范围和计算口径',
          '仅复用原1440 Train状态×84动作，包含840个原状态和120×5个匹配SOC状态。没有调用MPC求解器、没有训练DQN、没有修改公式/动作/分母/训练参数、没有读取Validation/Test或commit/push。没有引入40个补充状态或旧闭环轨迹。',
          'C_H=(H/6)^2，C_B=B/6，C_S=S/6，C_F=F/6。按总C升序排序；因为reward对C严格递减，这与最大化reward一致。精确并列按较小action ID优先，不加评分epsilon。总体标准差使用ddof=0，对每个状态的全部84动作计算。',
          '每状态保存R_i、sigma_i、a1/a2、C_i(a1/a2)、signed delta_Ci、abs(delta_Ci)、A_i及前两名控制差异。A_i仅使用sum(abs(delta_Cj))归一化；不使用delta_Ci/delta_C。若分母恰好为0，应标记A未定义；本样本没有这种状态。',
          f'恒等式delta_C=sum(delta_Ci)全部通过双精度舍入误差核验，最大绝对残差{summary["max_absolute_identity_residual"]:.6g}。误差界仅用于验证，没有改动分数。',
          '原1440状态全部进入逐状态文件和总体分组；下列五SOC表使用matched_soc的120组对照，避免SOC=.45/.55在两个cohort重复加权。负荷区间为≤300、(300,600]、(600,900]、>900 kW；本样本实际150.322–687.866 kW，>900没有样本，不作外推。',
          '## 动作间范围R和标准差sigma',
          '下表均为每个SOC组120个状态的中位数，先逐状态计算84动作的范围/标准差，再跨状态汇总。']
    for metric,label in [('R_p50','R_i中位数'),('sigma_p50','sigma_i中位数')]:
        text.append(table(['SOC',*[label+' '+t for t in TERMS]],[[soc,*[fmt(mg(soc).loc[t,metric]) for t in TERMS]] for soc in REQUESTED_SOC]))
    text += ['全部1440状态中，最大R项是F的有1437个、B有3个，S为0；最大sigma项全部是F。此结果描述整个动作集合的变化幅度，本身不等于winner因果归属，下面用前两名和控制移除检查交叉验证。',
             '## 前两名signed delta_Ci和A_i',
             '定义delta_Ci=C_i(a2)-C_i(a1)：正号表示该项支持完整C的winner a1，负号表示该项偏向runner-up a2。必须保留符号以观察相互抵消。分组中位数不满足可加性，恒等式在逐状态原始值上核验。']
    for metric,label in [('signed_delta_p50','signed delta中位数'),('abs_delta_p50','abs(delta)中位数')]:
        text.append(table(['SOC',*[label+' '+t for t in TERMS]],[[soc,*[fmt(mg(soc).loc[t,metric]) for t in TERMS]] for soc in REQUESTED_SOC]))
    text.append(table(['SOC','平均A_H %','平均A_B %','平均A_S %','平均A_F %','S为最大A的状态数'],
             [[soc,*[fmt(100*mg(soc).loc[t,'A_mean']) for t in TERMS],f'{int(mg(soc).loc["S","largest_A_count"])}/120'] for soc in REQUESTED_SOC]))
    text += ['这些A是同一对前两名之间的差值幅度分配，不是项占绝对C的比例，也不是独立因果贡献率或winner概率。全部1440状态的最大A分别为H:881、F:507、B:49、S:3。',
             'S为最大A的3个局部例外如下；其前两名首步FC差都不足0.35 kW，不能据此认定系统性选择垄断。']
    special=state[state.largest_A_term=='S']
    text.append(table(['audit_id','SOC','负荷kW','a1/a2','A_S %','a2-a1首步FC kW','六步FC RMS kW'],
         [[r.audit_id,r.soc,fmt(r.current_load_kw),f'{r.a1}/{r.a2}',fmt(100*r.A_S),fmt(r.top2_fc0_difference_kw),fmt(r.top2_plan_rms_difference_kw)] for r in special.itertuples()]))
    text += ['## 移除评分项：ID变化和实际控制差异',
             '只从离线总C中去掉一项，重新排序同一84条已保存轨迹；没有移除任何MPC内部目标。下面的首步和六步差异分位数均仅统计action ID确实改变的状态，避免大量未变化零值掩盖控制影响。CSV另存包括所有状态的分布以及signed首步差。六步RMS在全部6个FC功率点上计算，单位kW。']
    rows=[]
    for soc in REQUESTED_SOC:
        for t in TERMS:
            a=og(soc,t)
            rows.append([soc,t,f'{int(a.winner_changed_count)}/120',fmt(a.winner_change_rate*100),
               fmt(a.changed_only_abs_fc0_difference_kw_p50),fmt(a.changed_only_abs_fc0_difference_kw_p99),
               fmt(a.changed_only_plan_rms_difference_kw_p50),fmt(a.changed_only_plan_rms_difference_kw_p99)])
    text.append(table(['SOC','移除项','ID变化数','变化率%','|首步差| P50','|首步差| P99','六步RMS P50','六步RMS P99'],rows))
    text += ['SOC=.22时，移除S的首步差为向下约2.10 kW，六步RMS约5.19 kW；移除H为向上约12.03 kW，RMS约28.77 kW；移除F为向下约31.75 kW，RMS约115.36 kW（均为相应中位数）。因此S有恢复导向，但H/F仍强烈影响实际控制。不能只看三者都100%换ID，就认为作用强度相同。',
             '## 按SOC和负荷进一步分组',
             '每个SOC均有≤300 kW 47个、300–600 kW 31个、600–900 kW 42个状态。以下给出分组边际幅度、A和控制差异；所有指标的min/P1/P50/P99/max及均值完整保存在group_marginals.csv和group_omission_controls.csv。']
    for metric,label in [('R_p50','R中位数'),('sigma_p50','sigma中位数'),('A_mean','平均A百分数')]:
        rows=[]
        for soc in REQUESTED_SOC:
            for load in LOAD_BINS[:3]:
                a=mg(soc,load);factor=100 if metric=='A_mean' else 1
                rows.append([soc,load,int(a.iloc[0].states),*[fmt(a.loc[t,metric]*factor) for t in TERMS]])
        text.append(table(['SOC','负荷区间kW','状态数',*[label+' '+t for t in TERMS]],rows))
    rows=[]
    for soc in REQUESTED_SOC:
        for load in LOAD_BINS[:3]:
            for t in TERMS:
                a=og(soc,t,load)
                rows.append([soc,load,t,f'{int(a.winner_changed_count)}/{int(a.states)}',fmt(a.winner_change_rate*100),
                             fmt(a.get('changed_only_abs_fc0_difference_kw_p50',np.nan)),
                             fmt(a.get('changed_only_plan_rms_difference_kw_p50',np.nan))])
    text.append(table(['SOC','负荷区间kW','移除项','ID变化数','变化率%','变化后|首步差|P50','变化后六步RMS P50'],rows))
    text += ['在上述每个SOC/负荷分组，S都没有成为最大R或sigma项；平均A也没有超过H/F。个别S支持winner的近似并列并不构成跨组系统性主导。',
             '## 为什么绝对状态价值和动作选择可以不同',
             '令e0=SOC0-0.55，d_k(a)=SOC_k(a)-SOC0，则C_S=e0²/0.05²+2e0*mean(d_k(a))/0.05²+mean(d_k(a)²)/0.05²。第一项与动作无关，在同一状态的差值中完全抵消；后两项才会影响动作排序。在6秒内SOC变化有限，因而绝对SOC偏差可以很大，而动作间边际差异仍较小。',
             '对给定状态，给全部动作加相同常数不会改变总C排序；经单调的1/(1+sqrt(C))映射后也不改变即时winner，但会降低共同reward水平并改变reward间距。因此这里的“状态价值”指即时共同评分基线，不是已经训练得到的DQN Q/V值。',
             '**回答核心问题：0.05主要惩罚低SOC状态的绝对评分，额外提供适度的SOC恢复选择偏好；尚未垄断84动作选择。保持0.05，不启动尺度扫描。** 下一步若要改尺度，需要出现跨状态、跨负荷一致的S边际主导和显著控制效应；本轮证据不满足。结论仅限已审状态和当前6步MPC，不外推至未采样的>900 kW状态或长期DQN策略。',
             '## 可复核输出',
             '- state_marginals.csv：1440行，全部R/sigma、signed delta、A、a1/a2及恒等式残差。',
             '- omission_controls.csv：5760行，单项移除后的winner、signed/absolute首步FC差和六步轨迹RMS差。',
             '- group_marginals.csv / group_omission_controls.csv：总体、原840状态、匹配SOC组及负荷区间的完整分位数。',
             '- empty_groups.json：明确记录无样本的负荷/SOC组。',
             '- verification.json / provenance.json：逐项核验、原文件和输入哈希，新增求解/训练为0。',
             '- 复现：python -B src/main/audit_rms_marginals_84_train.py；仅核验可加--verify-only。']
    (OUT/'audit_report.md').write_text('\n\n'.join(text)+'\n',encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--verify-only',action='store_true');args=parser.parse_args()
    if not args.verify_only:
        run()
        report()
    verify()
