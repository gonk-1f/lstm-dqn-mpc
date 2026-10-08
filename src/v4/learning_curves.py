"""Render already-recorded training diagnostics without loading any dataset."""
from __future__ import annotations

from itertools import groupby
import json
from math import isfinite
import os
from pathlib import Path

from .experiment_paths import unarchived_output_path


FIGURES = ('completion_rates.png', 'validation_economic_costs.png',
           'td_and_gradients.png', 'soc_and_epsilon.png', 'fc_and_update_counts.png')
SOC_BINS = ('below_0p4', '0p4_to_0p6', 'above_0p6_below_0p79', 'at_least_0p79')


def _image_filename(path):
    """Give Pillow an extended absolute Windows path without global settings."""
    name=str(Path(path).resolve())
    if os.name=='nt' and not name.startswith('\\\\?\\'):
        return ('\\\\?\\UNC\\'+name[2:]) if name.startswith('\\\\') else ('\\\\?\\'+name)
    return name


def _fraction(summary, key):
    return None if summary is None else summary.get(key)


def _complete_cost(summary, key):
    if summary is None or summary.get('completed') != summary.get('episodes'):
        return None
    return summary.get(key)


def _learning_series(report, scale):
    rows = report.get('rounds', [])
    def summary_values(split, key):
        return [_fraction(row.get(split),key) for row in rows]
    def td_values(key):
        return [row.get('td_statistics', {}).get(key) for row in rows]
    td = td_values('mean_absolute_td_error')
    failure_td = [row.get('td_statistics',{}).get('failure_terminal',{}).get(
        'mean_absolute_td_error') for row in rows]
    def original_error(statistics):
        key='mean_absolute_td_error'
        original=statistics.get('original_units',{})
        if key in original:
            return original[key]
        value=statistics.get(key)
        return None if value is None else value/scale
    def clipping_fraction(row):
        statistics=row.get('td_statistics',{})
        gradients=statistics.get('gradient_statistics',{})
        return gradients.get('clipped_fraction',statistics.get('gradient_clipping_fraction'))
    series = {
        'round': [row['round'] for row in rows],
        'epsilon': [row.get('epsilon') for row in rows],
        'exploratory_train_completed': summary_values('exploratory_train','completed'),
        'greedy_train_completed': summary_values('greedy_train','completed'),
        'greedy_validation_completed': summary_values('greedy_validation','completed'),
        'validation_comparable_cost_cny': [_complete_cost(row.get('greedy_validation'),'cost_cny') for row in rows],
        'validation_observed_cost_cny': [_complete_cost(row.get('greedy_validation'),'completed_observed_cost_cny') for row in rows],
        'validation_modeled_cost_cny': [_complete_cost(row.get('greedy_validation'),'completed_modeled_terminal_cost_cny') for row in rows],
        'td_mae_scaled_reward_units': td,
        'td_mae_original_reward_units': [original_error(row.get('td_statistics',{})) for row in rows],
        'failure_td_mae_scaled_reward_units': failure_td,
        'failure_td_mae_original_reward_units': [original_error(row.get('td_statistics',{}).get(
            'failure_terminal',{})) for row in rows],
        'mean_smooth_l1_loss': td_values('mean_smooth_l1_loss'),
        'gradient_clipping_fraction': [clipping_fraction(row) for row in rows],
        'economic_optimizer_updates_cumulative': [row.get('economic_optimizer_updates_cumulative') for row in rows],
        'economic_replay_insertions_cumulative': [row.get('economic_replay_insertions_cumulative') for row in rows],
        'training_environment_transitions_cumulative': [row.get('training_environment_transitions_cumulative') for row in rows],
        'target_sync_calls_cumulative': [row.get('target_sync_calls_cumulative') for row in rows],
    }
    for split in ('exploratory_train','greedy_train','greedy_validation'):
        for key in ('onboard_soc_mean','onboard_soc_min_including_failed_prefix',
                    'onboard_soc_max_including_failed_prefix','terminal_onboard_soc_mean','fc_zero_fraction'):
            series[f'{split}_{key}'] = summary_values(split,key)
        for name in SOC_BINS:
            series[f'{split}_soc_fraction_{name}'] = [None if row.get(split) is None else
                row[split].get('soc_time_occupancy',{}).get('fractions',{}).get(name) for row in rows]
        series[f'{split}_fc_mean_kw'] = [None if row.get(split) is None else
            row[split].get('fc_power_statistics_kw',{}).get('mean') for row in rows]
        series[f'{split}_fc_change_mean_absolute_kw'] = [None if row.get(split) is None else
            row[split].get('fc_change_statistics_kw',{}).get('mean_absolute') for row in rows]
    for split in ('exploratory_train','greedy_train','greedy_validation'):
        series[f'{split}_completion_fraction'] = [None if row.get(split) is None or not row[split].get('episodes') else
            row[split]['completed']/row[split]['episodes'] for row in rows]
    return series


def _captured_profiles(report, output):
    """Select existing best/final-round profiles; only read files in this run."""
    rows = report.get('rounds', [])
    if not rows:
        return []
    chosen = [('final', rows[-1])]
    best = report.get('best_checkpoint')
    if best is not None:
        best_row = next((row for row in rows if row['round'] == best['round']),None)
        if best_row is not None and best_row is not rows[-1]:
            chosen.insert(0,('best',best_row))
    profiles = []
    for role, row in chosen:
        for split in ('greedy_train','greedy_validation'):
            summary = row.get(split)
            name = None if summary is None else summary.get('trajectory_file')
            if not name:
                continue
            path = (output/name).resolve()
            if path.parent != output:
                raise ValueError('captured trajectory must belong to this output directory')
            if not path.exists():
                continue
            captured = json.loads(path.read_text(encoding='utf-8'))
            for outcome in ('completed','failed'):
                profile = next((item for item in captured.get(outcome,[]) if item.get('transitions')),None)
                if profile is not None:
                    profiles.append({
                        'figure':f'{role}_round_{row["round"]:03d}_{split}_{outcome}_trajectory.png',
                        'source':path.name,'selection':role,'round':row['round'],
                        'split':split,'outcome':outcome,'profile':profile,
                    })
    return profiles


def write_learning_curves(report: dict, output_dir: Path) -> dict:
    """Plot complete rounds, retaining gaps for skipped/incomplete evaluations.

    Economic curves remain CNY. TD curves explicitly distinguish scaled reward
    units and division by reward_scale. Loss is the actual native Smooth L1
    objective; its nonlinear value is not converted by division by the scale.
    Existing artifacts and archived directories are protected from overwrites.
    """
    output = unarchived_output_path(output_dir)
    scale = float(report.get('hyperparameters',{}).get('reward_scale',1.))
    if not isfinite(scale) or scale <= 0:
        raise ValueError('reward_scale must be finite and positive')
    profiles = _captured_profiles(report,output)
    names = [*FIGURES,*(item['figure'] for item in profiles),'learning_curves_metadata.json']
    if any(Path(_image_filename(output/name)).exists() for name in names):
        raise FileExistsError('learning curve artifacts already exist; preserving previous evidence')
    output.mkdir(parents=True,exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    series = _learning_series(report,scale)
    rounds = series['round']
    def plot(ax, key, label, **kwargs):
        values = [np.nan if value is None else value for value in series[key]]
        ax.plot(rounds,values,label=label,marker='.',**kwargs)
    def save(fig,name):
        try:
            for ax in fig.axes:
                ax.grid(alpha=.2)
                handles,_labels = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(fontsize=8)
                if not ax.get_xlabel():
                    ax.set_xlabel('Completed training round')
            fig.tight_layout()
            fig.savefig(_image_filename(output/name),dpi=160)
        finally:
            plt.close(fig)

    fig,ax=plt.subplots(figsize=(9,4))
    for split,label in (('exploratory_train','Exploratory Train'),('greedy_train','Greedy Train'),
                        ('greedy_validation','Greedy Validation')):
        plot(ax,f'{split}_completion_fraction',label)
    ax.set(ylabel='Completed / requested samples',ylim=(-.03,1.03),
           title='Completion history; skipped Validation remains missing')
    save(fig,FIGURES[0])

    fig,ax=plt.subplots(figsize=(9,4))
    for key,label in (('validation_comparable_cost_cny','Comparable: observed + modeled'),
                      ('validation_observed_cost_cny','Observed economic ledger'),
                      ('validation_modeled_cost_cny','MODELED terminal settlement')):
        plot(ax,key,label)
    ax.set(ylabel='CNY',title='Validation costs only for completely finished splits')
    save(fig,FIGURES[1])

    fig,axes=plt.subplots(2,2,figsize=(11,7))
    plot(axes[0,0],'td_mae_scaled_reward_units','All sampled transitions')
    plot(axes[0,0],'failure_td_mae_scaled_reward_units','Failure terminals')
    axes[0,0].set(ylabel='Scaled reward-equivalent CNY',title=f'TD MAE; reward_scale={scale:g}')
    plot(axes[0,1],'td_mae_original_reward_units','All sampled transitions')
    plot(axes[0,1],'failure_td_mae_original_reward_units','Failure terminals')
    axes[0,1].set(ylabel='Original reward-equivalent CNY',title='TD MAE / reward_scale')
    plot(axes[1,0],'mean_smooth_l1_loss','Actual optimization loss')
    axes[1,0].set(ylabel='Native Smooth L1 objective',title='Loss is not an economic cost')
    plot(axes[1,1],'gradient_clipping_fraction','Updates requiring clipping')
    axes[1,1].set(ylabel='Fraction',ylim=(-.03,1.03),title='Economic Q gradient clipping')
    save(fig,FIGURES[2])

    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for key,label in (('onboard_soc_mean','Mean actual post-action SOC'),
                      ('onboard_soc_min_including_failed_prefix','Minimum, including failed prefixes'),
                      ('onboard_soc_max_including_failed_prefix','Maximum, including failed prefixes'),
                      ('terminal_onboard_soc_mean','Completed sample terminal ONBOARD SOC')):
        plot(axes[0,0],f'greedy_train_{key}',label)
    axes[0,0].set(ylabel='SOC',title='Greedy Train SOC')
    for name,label in zip(SOC_BINS,('SOC < .4','.4 <= SOC <= .6','.6 < SOC < .79','SOC >= .79')):
        plot(axes[0,1],f'greedy_train_soc_fraction_{name}',label)
    axes[0,1].set(ylabel='ONBOARD time fraction',ylim=(-.03,1.03),title='Greedy Train SOC occupancy')
    for split,label in (('exploratory_train','Exploratory Train'),('greedy_train','Greedy Train'),
                        ('greedy_validation','Greedy Validation')):
        plot(axes[1,0],f'{split}_onboard_soc_mean',label)
    axes[1,0].set(ylabel='Mean SOC',title='Executed ONBOARD states; SHORE excluded')
    plot(axes[1,1],'epsilon','Training exploration probability')
    axes[1,1].set(ylabel='Epsilon',ylim=(-.03,1.03),title='Exploration schedule')
    save(fig,FIGURES[3])

    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for split,label in (('exploratory_train','Exploratory Train'),('greedy_train','Greedy Train'),
                        ('greedy_validation','Greedy Validation')):
        plot(axes[0,0],f'{split}_fc_zero_fraction',label)
        plot(axes[0,1],f'{split}_fc_mean_kw',label)
        plot(axes[1,0],f'{split}_fc_change_mean_absolute_kw',label)
    axes[0,0].set(ylabel='Fraction of ONBOARD steps',ylim=(-.03,1.03),title='FC = 0')
    axes[0,1].set(ylabel='FC power (kW)',title='Mean executed FC power')
    axes[1,0].set(ylabel='Absolute FC change (kW)',title='Change from observed previous FC; no added penalty')
    for key,label in (('economic_optimizer_updates_cumulative','Economic optimizer updates'),
                      ('economic_replay_insertions_cumulative','Economic replay insertions'),
                      ('training_environment_transitions_cumulative','Executed training transitions'),
                      ('target_sync_calls_cumulative','Target copies including initial copy')):
        plot(axes[1,1],key,label)
    axes[1,1].set(ylabel='Cumulative count',title='Recorded execution budgets')
    save(fig,FIGURES[4])

    for selected in profiles:
        profile=selected['profile']
        fig,axes=plt.subplots(3,1,figsize=(10,7),sharex=True)
        offset=0
        for _voyage,group in groupby(profile['transitions'],key=lambda item:item.get('voyage_index',0)):
            values=list(group)
            edges=np.arange(offset,offset+len(values)+1)
            fc=[item['fc_kw'] for item in values]
            battery=[item['battery_bus_kw'] for item in values]
            axes[0].stairs(fc,edges,baseline=None,label='FC' if offset==0 else None)
            axes[0].stairs([item['load_kw'] for item in values],edges,baseline=None,
                           label='Actual load' if offset==0 else None)
            axes[1].stairs(battery,edges,baseline=None,label='Battery bus' if offset==0 else None)
            axes[2].plot(edges,[values[0]['soc_before'],*(item['soc_after'] for item in values)],
                         marker='.',label='Actual SOC' if offset==0 else None)
            if offset:
                for ax in axes:
                    ax.axvline(offset,color='grey',linestyle=':',alpha=.5)
            offset+=len(values)
        axes[0].set(ylabel='Power (kW)',title=f'{selected["selection"]} round {selected["round"]}: '
                    f'{selected["split"]}, {selected["outcome"]}, sample {profile.get("sample_id")}')
        axes[1].set(ylabel='Battery power (kW)')
        axes[2].set(ylabel='SOC',xlabel='Executed ONBOARD interval index; each 30 s, SHORE omitted')
        for ax in axes:
            ax.set_xlabel('Executed ONBOARD interval index; SHORE omitted')
        save(fig,selected['figure'])
    metadata={
        'reward_scale':scale,'round_count':len(rounds),'source_commit':report.get('source_commit'),
        'hyperparameters':report.get('hyperparameters',{}),'execution_counts':report.get('execution_counts',{}),
        'test_payloads_opened':report.get('test_payloads_opened'),
        'units':{'economic_cost':'CNY','q_and_td_native':'scaled reward-equivalent CNY',
                 'q_and_td_converted':'original reward-equivalent CNY',
                 'smooth_l1':'native nonlinear objective, not converted economic cost',
                 'power':'kW','soc':'fraction'},
        'series':series,'figures':[*FIGURES,*(item['figure'] for item in profiles)],
        'representative_trajectories':[{key:value for key,value in item.items() if key!='profile'} for item in profiles],
        'scope':'Recorded completed rounds only; no data loading, optimization, or RNG sampling; missing values remain null',
        'representative_scope':'Existing captured actual ONBOARD transitions, including failed prefixes; no appended modeled charging control points',
    }
    temporary=output/'learning_curves_metadata.json.tmp'
    temporary.write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    temporary.replace(output/'learning_curves_metadata.json')
    return metadata
