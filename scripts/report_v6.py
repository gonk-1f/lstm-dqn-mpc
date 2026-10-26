"""Summarize the existing V6 run without network instantiation or environment runs."""
import argparse
import csv
import gzip
import json
import math
import os
import time
from itertools import islice
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _cost_totals(rows):
    keys=('fc_h2_cost_cny','battery_h2eq_cost_cny','fc_degradation_cost_cny',
          'battery_degradation_cost_cny','shore_electricity_cost_cny','ledger_total_cost_cny')
    totals={key:math.fsum(float(row[key]) for row in rows) for key in keys}
    totals.update(episodes=len(rows),
        executed_duration_s=math.fsum(float(row['executed_duration_s']) for row in rows),
        fc_h2_kg=math.fsum(float(row['fc_h2_kg']) if 'fc_h2_kg' in row
                           else float(row['fc_h2_cost_cny'])/21.9 for row in rows),
        economic_cost_cny=math.fsum(float(row['executed_prefix_economic_cost_cny']) for row in rows))
    return totals


def _split_round_report(root, round_index, split, summary):
    path=(root/f'round_{round_index:02d}_train_samples.csv' if split=='train'
          else root/f'round_{round_index:02d}_validation/sample_metrics.csv')
    with path.open(encoding='utf-8-sig',newline='') as handle:
        rows=list(csv.DictReader(handle))
    complete=[row for row in rows if row['completed']=='True']
    failed=[row for row in rows if row['completed']!='True']
    boundaries={'upper':0,'lower':0,'interior':0}
    for row in failed:
        side=row.get('failure_boundary_side')
        if not side:
            soc=float(row['soc_terminal'])
            side='upper' if soc>=.79 else 'lower' if soc<=.21 else 'interior'
        boundaries[side]+=1
    result=dict(summary)
    result.update(failure_boundary_counts=boundaries,
        failure_classification_note='SOC boundary side and action/physical failure reason are overlapping classifications.',
        all_executed=_cost_totals(rows),failed_prefixes=_cost_totals(failed),
        completed_voyages=_cost_totals(complete))
    return result


def _fit_error(q_diagnostics):
    for name in ('checkpoint_replay_fit','training_td_fit'):
        group=q_diagnostics.get(name,{})
        measured=[(float(item['mae']),int(item.get('measured_count',item['sample_count'])))
                  for item in group.get('groups',{}).values() if item.get('mae') is not None]
        count=sum(n for _,n in measured)
        if count:
            return name,'sample_weighted_mae',math.fsum(value*n for value,n in measured)/count
        for metric in ('td_rmse','td_mae','td_mse'):
            if group.get(metric) is not None:
                return name,metric,float(group[metric])
    return None


def _q_report_lines(root, diagnostics):
    lines=[]
    def number(value):
        return '未记录' if value is None else f'{float(value):.6f}'
    def span(values):
        return f"{number(values.get('mean'))} [{number(values.get('min'))}, {number(values.get('max'))}]"
    for name,title in (('training_td_fit','在线训练 draw 统计'),('checkpoint_replay_fit','当前 checkpoint 对只读 Replay 样本的拟合'),
                       ('fixed_probes','固定全 13 动作 Q 排名'),('train_failure_value','既有 Train 失败轨迹实际回报 / 当前 Q / 采样次数')):
        group=diagnostics.get(name)
        lines.extend([f'### {title}',''])
        if group is None:
            lines.extend(['本运行未记录。',''])
            continue
        file=group.get('file') or group.get('path') or group.get('raw_samples')
        if file:
            lines.extend([f"[完整记录]({(root/file).resolve().as_posix()})",''])
        if 'groups' in group:
            lines.extend(['|经验类别|Replay类别总量|实际draw/诊断样本数|Q均值 [最小,最大]|TD目标均值 [最小,最大]|有符号残差均值 [最小,最大]|MAE|绝对误差p95|p95计数/方式|',
                '|---|---:|---:|---|---|---|---:|---:|---|'])
            for kind,stats in group['groups'].items():
                count=stats.get('measured_count',stats.get('sample_count',0))
                lines.append(f"|{kind}|{stats.get('replay_count','—')}|{count}|{span(stats.get('q',{}))}|{span(stats.get('target',{}))}|"
                    f"{span(stats.get('residual_target_minus_q',{}))}|"
                    f"{number(stats.get('mae'))}|{number(stats.get('absolute_residual_p95'))}|"
                    f"{stats.get('p95_sample_count',0)} / {stats.get('p95_method','未记录')}|")
            lines.extend(['',f"统计范围：{group.get('scope',name)}。"])
            if 'replay_count' in group:
                lines.append(f"Replay 经验 {group['replay_count']}；本次只读测量 {group.get('measured_count','未记录')}。")
            for key,label in (('failure_terminal_draws_window','本窗口失败终止实际 draw 次数'),
                              ('failure_terminal_draws_total','累计失败终止实际 draw 次数')):
                if key in group:
                    draws=group[key]
                    lines.append(f'{label}：{sum(draws.values())}；分 replay ID 的完整计数保留在本轮 JSON。')
            if group.get('prefix_label_note'):
                lines.append(group['prefix_label_note'])
            lines.append('')
        else:
            lines.extend(['```json',json.dumps(group,ensure_ascii=False,indent=2),'```',''])
    return lines


def write_round_summary(root, entry, previous_entry=None, config=None):
    """Write one round's recorded control, cost and Q evidence without inference."""
    root=Path(root)
    index=entry['round']
    train=_split_round_report(root,index,'train',entry['train'])
    valid=_split_round_report(root,index,'validation',entry['validation'])
    q_diagnostics=entry.get('q_diagnostics',{})
    notes=[
        'TD target 不是 Q*。训练时 Online/Target 参数变化会改变目标；拟合误差仅衡量对对应时刻 bootstrap 目标的拟合。',
        'Loss、TD 误差或单个状态 Q 值不能单独证明控制改善；同时检查完整航段完成数、边界失败和物理时间尺度的功率波动。',
        '固定 Validation 探针仅用于报告，不加入 Replay、不训练；Test 不读取。',
        '失败前缀成本只覆盖实际执行区间；完整航段子集成本不是固定全套成本。SOC/ramp/失败惩罚不计入经济账本。',
        'SOC 上/下边界计数与增量动作不可达计数是交叉分类，不可相加作为失败总数。',
        '在线 draw 与 checkpoint 分层诊断样本的分布不同；加权 MAE 仅在同一统计口径内作轮间比较，并同时保留各类别误差。',
        '分层诊断样本和 TD 目标均随轮次变化；weighted MAE 的变化仅作描述，不能解释为对 Q* 的拟合改善。',
        '实际失败轨迹回报来自当时行为策略，不是最优 Q 的真值；未知终局的历史 draw 分类不会在航段失败后追改。',
    ]
    trends={}
    current_error=_fit_error(q_diagnostics)
    previous_error=_fit_error(previous_entry.get('q_diagnostics',{})) if previous_entry else None
    terminal_fit=q_diagnostics.get('checkpoint_replay_fit',{}).get('groups',{}).get('failure_terminal',{})
    terminal_residual=terminal_fit.get('residual_target_minus_q',{}).get('mean')
    if terminal_residual is not None and terminal_residual<0:
        notes.append(f'失败终止诊断样本平均 target−Q={terminal_residual:.6f}；Q仍偏高，平均低估已观测失败代价。'
                     '该判断针对已执行失败动作，不将此前行为轨迹回报当作Q*。')
    for event in q_diagnostics.get('train_failure_value',{}).get('events',[]):
        notes.append(f"固定Train {event['boundary_side']}失败 {event['sample_id']} 的终止经验累计实际抽样 "
                     f"{event['actual_training_draws']} 次；终止 target−Q={event['terminal_signed_residual']:.6f}，"
                     f"终止MAE={event['terminal_mae']:.6f}。前缀与终止拟合及Validation需同时查看。")
    if previous_entry:
        prior=previous_entry['validation']
        previous_boundaries=prior.get('failure_boundary_counts')
        if previous_boundaries is None:
            previous_boundaries=_split_round_report(root,previous_entry['round'],'validation',prior)['failure_boundary_counts']
        trends.update(validation_completed_delta=valid['completed']-prior['completed'],
            validation_boundary_failures_delta=sum(valid['failure_boundary_counts'][side] for side in ('upper','lower'))
                -sum(previous_boundaries.get(side,0) for side in ('upper','lower')),
            validation_changes_per_hour_delta=valid['fc_changes_per_hour']-prior['fc_changes_per_hour'],
            validation_reversals_per_hour_delta=valid['fc_reversals_per_hour']-prior['fc_reversals_per_hour'])
        if current_error and previous_error and current_error[:2]==previous_error[:2]:
            trends.update(td_error_source=current_error[0],td_error_metric=current_error[1],
                td_error_delta=current_error[2]-previous_error[2])
            if trends['td_error_delta']<0:
                if trends['validation_completed_delta']<=0:
                    notes.append('TD 拟合误差下降，但 Greedy Validation 完成数未提高。')
                if trends['validation_boundary_failures_delta']>=0:
                    notes.append('TD 拟合误差下降，但上下 SOC 边界失败总数未减少。')
                if trends['validation_changes_per_hour_delta']>=0:
                    notes.append('TD 拟合误差下降，但 FC 每小时变载次数未改善。')
                if trends['validation_reversals_per_hour_delta']>=0:
                    notes.append('TD 拟合误差下降，但 FC 每小时相邻反向次数未改善。')
    else:
        notes.append('这是本运行首轮，没有轮间基线；不能据本轮 loss 或单状态 Q 宣称改善。')
    if not current_error:
        notes.append('本轮未记录可比较的 TD 拟合误差，不推断其是否下降。')
    summary=dict(round=index,configuration=config or {},train=train,validation=valid,
        epsilon_start=entry['epsilon_start'],epsilon_end=entry['epsilon_end'],
        adam_updates=entry['adam_updates'],target_copies=entry['target_copies'],
        cumulative_adam_updates=entry['cumulative_adam_updates'],
        cumulative_target_copies=entry['cumulative_target_copies'],q_diagnostics=q_diagnostics,
        trends_vs_previous_round=trends,interpretation_notes=notes)
    destination=root/f'round_{index:02d}_summary'
    destination.with_suffix('.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    text=[f'# V6 第 {index} 轮报告','',
        f"Train 完成 {train['completed']}/35；Greedy Validation 完成 {valid['completed']}/8。",
        f"实际 Adam 更新 {entry['adam_updates']}（累计 {entry['cumulative_adam_updates']}）；"
        f"Target 复制 {entry['target_copies']}（含初始化累计 {entry['cumulative_target_copies']}）。",
        f"Epsilon {entry['epsilon_start']:.6f} → {entry['epsilon_end']:.6f}。",'',
        '|划分|完成|上边界失败|下边界失败|内区失败|增量不可达|SOC不可行|结构功率失败|FC变载/h|相邻反向/h|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for label,item in (('Train',train),('Greedy Validation',valid)):
        b=item['failure_boundary_counts']; reasons=item['failure_reasons']
        text.append(f"|{label}|{item['completed']}/{item['episodes']}|{b['upper']}|{b['lower']}|{b['interior']}|"
            f"{reasons.get('increment_action_limited',0)}|{reasons.get('soc_limited',0)}|{reasons.get('structural_power',0)}|"
            f"{item['fc_changes_per_hour']:.3f}|{item['fc_reversals_per_hour']:.3f}|")
    text.extend(['','费率以实际执行秒数折算；失败前缀长度变化会影响跨轮比较。','',
        '|划分/成本范围|航段数|实际FC氢耗(kg)|氢成本(CNY)|FC退化(CNY)|电池退化(CNY)|四项账本总额(CNY)|电池等效氢修正(CNY)|修正后经济成本(CNY)|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|'])
    fixed_cost_notes=[]
    for label,item in (('Train',train),('Validation',valid)):
        for scope,title in (('all_executed','全部已执行区间'),('failed_prefixes','仅失败前缀'),('completed_voyages','仅完整航段子集')):
            cost=item[scope]
            text.append(f"|{label}/{title}|{cost['episodes']}|{cost['fc_h2_kg']:.6f}|{cost['fc_h2_cost_cny']:.6f}|"
                f"{cost['fc_degradation_cost_cny']:.6f}|{cost['battery_degradation_cost_cny']:.6f}|"
                f"{cost['ledger_total_cost_cny']:.6f}|{cost['battery_h2eq_cost_cny']:.6f}|{cost['economic_cost_cny']:.6f}|")
        fixed=item.get('fixed_complete_set_cost_cny')
        fixed_cost_notes.append(f"{label} 固定全套完整成本：{fixed:.6f} CNY。" if fixed is not None
                    else f"{label} 固定全套完整成本：未提供，存在未完成航段。")
    text.extend(['',*fixed_cost_notes,'','四项账本包括 FC 氢成本、FC 退化、电池退化和岸电成本；本任务岸电关闭。',
        '电池等效氢修正保留正负号，另列于实际账本之外。旧记录若未直接保存 FC 氢耗质量，由正式氢价 21.9 CNY/kg 换算。','',
        '## Q 与 TD 拟合证据',''])
    text.extend(_q_report_lines(root,q_diagnostics))
    text.extend(['## 解释限制与变化','',*[f'- {note}' for note in notes]])
    if trends:
        text.extend(['','相对上一轮的记录变化：','```json',json.dumps(trends,ensure_ascii=False,indent=2),'```'])
    destination.with_suffix('.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    return summary


def export_failure_details(root, report):
    if not report['rounds']:
        return
    round_index=report['rounds'][-1]['round']
    destination=root/f'round_{round_index:02d}_failure_details'
    if destination.with_suffix('.json').exists():
        return
    from v3.control import AccountState, EconomicMPC
    from v2.data.supervisory_rules import normalize_onboard_load_kw
    from v4.actions import candidate_fc_actions
    from v4.control import _v6_feasible_fc_actions

    with (root/f'round_{round_index:02d}_train_samples.csv').open(encoding='utf-8-sig',newline='') as handle:
        train_rows=list(csv.DictReader(handle))
    replay=None
    if any(row['status']!='completed' and 'previous_fc_terminal_kw' not in row for row in train_rows):
        import torch
        # Compatibility for old runs whose terminal FC was only saved in Replay.
        saved=torch.load(root/'models/training_state_latest.pt',map_location='cpu',weights_only=False)
        replay=saved['agent']['replay']
    data=Path(__file__).resolve().parents[1]/'data/processed/operating_dataset_v6_1s'
    metadata=json.loads((data/'metadata/active_split.json').read_text(encoding='utf-8'))
    with (data/metadata['manifest_relative_path']).open(encoding='utf-8-sig',newline='') as handle:
        manifest={r['candidate_id']:r for r in csv.DictReader(handle) if r['split'] in ('train','validation')}
    accountant=EconomicMPC(nominal_cost_cny=1.)
    failures=[]

    def append_failure(row, phase, previous_fc, provenance):
        index=int(row['executed_steps'])
        with gzip.open(data/manifest[row['sample_id']]['relative_path'],'rt',encoding='utf-8-sig',newline='') as handle:
            source=next(islice(csv.DictReader(handle),index,None))
        soc=float(row['soc_terminal'])
        raw_load=float(source['load_total_kw'])
        physical=_v6_feasible_fc_actions(AccountState(soc=soc,previous_fc_kw=previous_fc),
            raw_load,accountant,duration_s=1.)
        allowed=candidate_fc_actions(physical,previous_fc,'increment13')
        candidates=[dict(action_index=i,delta_kw=delta,fc_kw=previous_fc+delta,
            in_fc_bounds=0<=previous_fc+delta<=600,
            physical_legal=previous_fc+delta in physical,allowed=previous_fc+delta in allowed)
            for i,delta in enumerate(range(-60,61,10))]
        failures.append(dict(round=round_index,phase=phase,sample_id=row['sample_id'],
            failure_reason=row['status'],epsilon=float(row['epsilon']) if phase=='train_exploration' else 0.,
            executed_steps=index,failed_elapsed_time_s=float(source['time_s']),
            failed_source_time_s=float(source['source_time_s']),failed_timestamp=source['timestamp'],
            source_1s_row=int(source['source_1s_row']),soc=soc,raw_load_kw=raw_load,
            physical_load_kw=normalize_onboard_load_kw(raw_load),observed_load_over_600=raw_load/600.,
            previous_fc_kw=previous_fc,physical_legal_fc_kw=list(physical),
            physical_legal_fc_min_kw=min(physical) if physical else None,
            physical_legal_fc_max_kw=max(physical) if physical else None,
            increment_legal_fc_kw=list(allowed),all13_legal_mask=[c['allowed'] for c in candidates],
            all13_candidates=candidates,state_provenance=provenance))

    cursor=report['warmup_environment_steps']+sum(
        entry['train']['environment_steps'] for entry in report['rounds'] if entry['round']<round_index)
    for row in train_rows:
        steps=int(row['executed_steps'])
        if row['status']!='completed':
            if 'previous_fc_terminal_kw' in row:
                previous=int(float(row['previous_fc_terminal_kw']))
                provenance='SOC and previous FC: recorded float64 terminal sample metrics; load/time: recorded source row'
            else:
                slot=(cursor+steps-1)%replay['capacity']
                previous=round(float(replay['next_states'][slot,2])*600./10.)*10 if steps else 0
                provenance='SOC: float64 sample metric; previous FC: terminal Replay next_state decoded to exact 10 kW grid; load/time: recorded source row'
            append_failure(row,'train_exploration',previous,provenance)
        cursor+=steps
    folder=root/f'round_{round_index:02d}_validation'
    with (folder/'sample_metrics.csv').open(encoding='utf-8-sig',newline='') as handle:
        for row in csv.DictReader(handle):
            if row['status']=='completed':
                continue
            previous=int(float(row['previous_fc_terminal_kw'])) if 'previous_fc_terminal_kw' in row else 0
            if 'previous_fc_terminal_kw' not in row and int(row['executed_steps']):
                with gzip.open(folder/f"{row['sample_id']}.csv.gz",'rt',encoding='utf-8',newline='') as profile:
                    for last in csv.DictReader(profile):
                        previous=int(float(last['fc_kw']))
            append_failure(row,'greedy_validation',previous,
                'SOC: float64 sample metric; previous FC: last executed Validation profile row; load/time: recorded source row')
    fields=list(failures[0]) if failures else ['round','phase','sample_id','failure_reason','epsilon']
    with destination.with_suffix('.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader()
        writer.writerows({key:json.dumps(value,ensure_ascii=False) if isinstance(value,list) else value
                         for key,value in row.items()} for row in failures)
    notes=[
        'Postprocessed existing executed results only; no environment interval, policy inference or new experiment.',
        'Failure time is the next decision after executed_steps; no action was executed at that failed decision.',
        'train_exploration identifies the epsilon-greedy phase. Per-action random-choice flags were not stored, so these records do not establish that a random action caused the failure.',
        'Warmup failures are excluded because full-precision terminal SOC was not saved in warmup metrics.',
    ]
    destination.with_suffix('.json').write_text(json.dumps(dict(round=round_index,notes=notes,failures=failures),
        ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('run',type=Path)
    args=parser.parse_args()
    root=args.run
    report=json.loads((root/'report.json').read_text(encoding='utf-8'))
    if report['status'] not in ('completed','stopped_error') and (root/'progress.json').exists():
        progress=json.loads((root/'progress.json').read_text(encoding='utf-8'))
        report['status']=progress['status']
        report['warmup_environment_steps']=max(report['warmup_environment_steps'],progress['warmup_environment_steps'])
        report['training_environment_steps']=max(report['training_environment_steps'],progress['training_environment_steps'])
        report['cumulative_adam_updates']=progress['adam_updates']
        report['cumulative_target_copies']=progress['target_copies']
    export_failure_details(root,report)
    records=[]
    round_summaries=[]
    previous_entry=None
    for entry in report['rounds']:
        train,valid=entry['train'],entry['validation']
        summary=write_round_summary(root,entry,previous_entry,report.get('configuration',{}))
        round_summaries.append(summary)
        previous_entry=entry
        records.append(dict(round=entry['round'],train_completed=train['completed'],
            validation_completed=valid['completed'],train_environment_steps=train['environment_steps'],
            validation_environment_steps=valid['environment_steps'],adam_updates=entry['adam_updates'],
            target_copies=entry['target_copies'],cumulative_adam_updates=entry['cumulative_adam_updates'],
            cumulative_target_copies=entry['cumulative_target_copies'],epsilon_start=entry['epsilon_start'],
            epsilon_end=entry['epsilon_end'],train_failures=json.dumps(train['failure_reasons']),
            validation_failures=json.dumps(valid['failure_reasons']),
            validation_changes_per_hour=valid['fc_changes_per_hour'],
            validation_reversals_per_hour=valid['fc_reversals_per_hour'],
            validation_complete_set_cost_cny=valid['fixed_complete_set_cost_cny'],
            train_upper_boundary_failures=summary['train']['failure_boundary_counts']['upper'],
            train_lower_boundary_failures=summary['train']['failure_boundary_counts']['lower'],
            validation_upper_boundary_failures=summary['validation']['failure_boundary_counts']['upper'],
            validation_lower_boundary_failures=summary['validation']['failure_boundary_counts']['lower'],
            train_actual_fc_h2_kg=summary['train']['all_executed']['fc_h2_kg'],
            validation_actual_fc_h2_kg=summary['validation']['all_executed']['fc_h2_kg'],
            validation_failed_prefix_cost_cny=summary['validation']['failed_prefixes']['economic_cost_cny'],
            validation_complete_subset_cost_cny=summary['validation']['completed_voyages']['economic_cost_cny']))
    if records:
        with (root/'training_records.csv').open('w',encoding='utf-8-sig',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
        rounds=[r['round'] for r in records]
        fig,axes=plt.subplots(3,1,figsize=(9,8),sharex=True)
        axes[0].plot(rounds,[r['train_completed']/35 for r in records],'-o',label='Train /35')
        axes[0].plot(rounds,[r['validation_completed']/8 for r in records],'-o',label='Validation /8')
        axes[0].set_ylabel('Completion fraction');axes[0].set_ylim(0,1.05);axes[0].legend()
        axes[1].plot(rounds,[r['validation_changes_per_hour'] for r in records],'-o',label='FC changes/h')
        axes[1].plot(rounds,[r['validation_reversals_per_hour'] for r in records],'-o',label='Adjacent reversals/h')
        axes[1].set_ylabel('Events per physical hour');axes[1].legend()
        axes[2].plot(rounds,[r['cumulative_adam_updates'] for r in records],'-o')
        axes[2].set_ylabel('Actual cumulative Adam');axes[2].set_xlabel('Full Train traversal')
        fig.tight_layout();fig.savefig(root/'training_overview.png',dpi=150);plt.close(fig)
    comparison_path=root/'v5_v6_validation_comparison.csv'
    comparison=[]
    if records:
        latest=records[-1]['round']
        # A hold Q remains informative when hold is physically unavailable;
        # only its legal-action rank must then be left empty.
        for profile_path in (root/f'round_{latest:02d}_validation').glob('*.csv.gz'):
            with gzip.open(profile_path,'rt',encoding='utf-8',newline='') as handle:
                reader=csv.DictReader(handle)
                if 'hold_legal' in reader.fieldnames:
                    continue
                profile=list(reader)
                fields=[*reader.fieldnames,'hold_legal']
            for row in profile:
                previous=float(row['fc_kw'])-float(row['fc_delta_kw'])
                battery=max(0.,float(row['load_kw']))-previous
                cell=battery/.95 if battery>0. else battery*.95
                successor=float(row['soc_before'])-cell/3600./624.
                legal=0.<=previous<=600. and -624.<=battery<=1248. and .2<=successor<=.8
                row['hold_legal']=legal
                if not legal: row['hold_rank']=''
            temporary=profile_path.with_suffix('.tmp')
            with gzip.open(temporary,'wt',encoding='utf-8',newline='') as handle:
                writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(profile)
            temporary.replace(profile_path)
        old_path=root.resolve().parents[2]/'v5-economic4-ddqn/docs/results/v5_economic4_paused_20261021/sample_metrics.csv'
        with old_path.open(encoding='utf-8-sig',newline='') as handle:
            old={r['sample_id']:r for r in csv.DictReader(handle) if r['split']=='validation' and r['round']=='32'}
        with (root/f'round_{latest:02d}_validation/sample_metrics.csv').open(encoding='utf-8-sig',newline='') as handle:
            for row in csv.DictReader(handle):
                prior=old[row['sample_id']];duration=float(prior['executed_duration_s'])
                common=row['completed']=='True' and prior['completed']=='True'
                comparison.append(dict(sample_id=row['sample_id'],v5_completed=prior['completed']=='True',
                    v6_completed=row['completed']=='True',
                    v5_changes_per_hour=float(prior['fc_nonzero_changes'])*3600./duration,
                    v6_changes_per_hour=float(row['fc_changes_per_hour']),
                    v5_reversals_per_hour=float(prior['fc_adjacent_reversals'])*3600./duration,
                    v6_reversals_per_hour=float(row['fc_reversals_per_hour']),
                    v5_soc_terminal=float(prior['soc_terminal']),v6_soc_terminal=float(row['soc_terminal']),
                    same_complete_sample=common,
                    v5_complete_cost_cny=float(prior['complete_zhou_cost_cny']) if common else None,
                    v6_complete_cost_cny=float(row['complete_economic_cost_cny']) if common else None))
        with comparison_path.open('w',encoding='utf-8-sig',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(comparison[0]));writer.writeheader();writer.writerows(comparison)
        comparison=[]
    if comparison_path.exists():
        with comparison_path.open(encoding='utf-8-sig',newline='') as handle: comparison=list(csv.DictReader(handle))
    common=[r for r in comparison if r['same_complete_sample']=='True']
    text=['# V6 single-configuration training result','',
        f"Status: {report['status']}; completed traversals: {report['completed_rounds']}/8.",
        f"Warmup: {report['warmup_environment_steps']} genuine transitions, "
        f"{report.get('warmup_adam_updates',0)} Adam updates.",
        f"Formal training: {report['training_environment_steps']} executed transitions; "
        f"Adam updates: {report.get('cumulative_adam_updates',0)}; "
        f"Target copies (including initialization): {report.get('cumulative_target_copies',1)}.",
        f"Test payloads opened: {report.get('test_payloads_opened',0)}. This report describes this run's recorded configuration.",'',
        '|Traversal|Train /35|Validation /8|Train steps|Adam|Target copies|Epsilon start/end|',
        '|---|---:|---:|---:|---:|---:|---|']
    for row in records:
        text.append(f"|{row['round']}|{row['train_completed']}|{row['validation_completed']}|"
            f"{row['train_environment_steps']}|{row['adam_updates']}|{row['target_copies']}|"
            f"{row['epsilon_start']:.6f}/{row['epsilon_end']:.6f}|")
    text.extend(['','Research candidate: traversal '+str(report.get('best_research_candidate_round','none'))+'.',
        'Selection prioritizes Validation completion count, then duration coverage; full-set cost is used when all eight complete.',
        'Incomplete-prefix costs are never represented as a fixed complete-set cost.',
        'TD targets are bootstrapped training targets, not Q*. Online/Target updates change those targets during training.',
        'Loss, TD fitting error and a single probe Q do not establish control improvement; evaluate completion, boundary failures and FC fluctuations together.',
        'Each round_NN_summary.md/.json reports Train/Validation failure sides, actual hydrogen and economic components, failed prefixes, completed subsets and recorded Q diagnostics.',
        'Fixed Validation probes are reporting only and never enter Replay or training.','',
        '## Comparison with V5 round32','',
        'Different time resolution, action availability, state, objective and training budget prevent a single-factor causal comparison.',
        'Change/reversal rates use actual physical hours; per-step event counts are not compared directly.',
        f'Jointly completed Validation IDs at the final traversal: {len(common)}.'])
    if common:
        before=sum(float(r['v5_complete_cost_cny']) for r in common)
        after=sum(float(r['v6_complete_cost_cny']) for r in common)
        text.append(f'Same-ID complete economic cost: V5 {before:.6f} CNY; V6 {after:.6f} CNY.')
    if records:
        last=records[-1]
        text.extend(['',f"Final recorded Validation completion: {last['validation_completed']}/8; V5 round32:5/8.",
            f"V6 FC changes/hour: {last['validation_changes_per_hour']:.6f}; "
            f"adjacent reversals/hour: {last['validation_reversals_per_hour']:.6f}.",
            'These rates include actual failed prefixes and describe different policy paths; same-ID complete costs are reported separately.',
            'Each completed traversal preserves round_NN_failure_details.csv and .json with Train/Validation failure states and all 13 candidate legality flags.',
            'Train failures occurred under epsilon-greedy exploration; per-action random-choice flags were not stored, so a random action cannot be identified as their cause.'])
    text.extend(['','Validation CSV.gz and PNG files contain load, FC power, battery power, SOC, FC increments,',
        'all13 action Q values, legal hold rank, separate reward terms and cumulative reward terms.',
        'Fixed probes preserve V5 physical SOC/FC/time and add V6 causal history. They have a different input/output',
        'definition and reward unit from V5; raw numerical Q values must not be compared as identical targets.',
        'See fixed_probe_history.json and v5_v6_validation_comparison.csv.','',
        'Data reuse preserved every canonical original anchor exactly. No new real1Hz measurements were created.',
        'The ±60kW/s action capability remains a simulation assumption.'])
    if records:
        first={ (r['sample_id'],r['time_s']):r for r in report['fixed_probes'] if r['round']==0 }
        final=[r for r in report['fixed_probes'] if r['round']==records[-1]['round']]
        text.extend(['','## Existing fixed physical probes in V6 coordinates','',
            '|Sample|Time/s|Initial hold rank /13|Latest hold rank /13|Greedy increment/kW|Greedy minus hold Q|',
            '|---|---:|---:|---:|---:|---:|'])
        for row in final:
            initial=first[(row['sample_id'],row['time_s'])]
            text.append(f"|{row['sample_id']}|{row['time_s']}|{initial['hold_rank']}|{row['hold_rank']}|"
                f"{row['greedy_delta_kw']}|{row['greedy_minus_hold']:.6f}|")
    if round_summaries:
        text.extend(['','## Per-round Q fitting and control evidence',''])
        for summary in round_summaries:
            index=summary['round']
            text.extend([f'## Traversal {index}: recorded evidence','',
                f"[Control, cost and Q summary]({(root/f'round_{index:02d}_summary.md').resolve().as_posix()})",''])
            text.extend(_q_report_lines(root,summary['q_diagnostics']))
            text.extend(['Interpretation and control consistency:','',
                *[f'- {note}' for note in summary['interpretation_notes']],''])
            if summary['trends_vs_previous_round']:
                text.extend(['```json',json.dumps(summary['trends_vs_previous_round'],ensure_ascii=False,indent=2),'```',''])
    if report.get('error'): text.extend(['','Stopped reason: '+report['error']])
    (root/'training_result.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    request=root/'pause_after_round.json'
    if request.exists():
        requested_round=int(json.loads(request.read_text(encoding='utf-8'))['round'])
        if report['completed_rounds']>=requested_round:
            ready=dict(runpath=str(root.resolve()),round=report['completed_rounds'],
                checkpointpath=str((root/'models/training_state_latest.pt').resolve()),
                parentpid=os.getppid(),reporterpid=os.getpid(),status='ready_for_normal_stop')
            (root/'round_boundary_ready.json').write_text(
                json.dumps(ready,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
            print(f"PAUSE_READY round={ready['round']} runpath={ready['runpath']}",flush=True)
            while request.exists():
                time.sleep(.5)


if __name__=='__main__': main()
