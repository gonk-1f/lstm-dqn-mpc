"""Read bounded real Zoo artifacts; never train or open dataset payloads."""
import argparse
import json
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]


def verify(train, evaluation, optimization):
    import numpy as np
    import pandas as pd
    import torch
    import optuna
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from experiments.sb3_manual.zoo_vessel.algorithm import ZooMaskedDQN
    from v6_sb3.masked_dqn import masked_td_targets

    read = lambda path:json.loads(path.read_text(encoding='utf-8'))
    torch.set_num_threads(1)
    run = read(train/'run_summary.json')
    model = ZooMaskedDQN.load(run['model'], device='cpu')
    folder = Path(run['model']).parent
    with (folder/'replay_buffer.pkl').open('rb') as source:
        replay = pickle.load(source)
    total = model.num_timesteps
    assert total == run['n_timesteps'] == 4096
    assert model.learning_starts == 64 and model.formal_env_steps == 4032
    assert model._n_updates == 126 and model.target_sync_unit == 'formal_env_steps'
    assert model.target_sync_history == [dict(formal_env_steps=n, adam_updates=(n-1)//32)
                                        for n in range(128, 4033, 128)]
    assert total == replay.insertions and not replay.full and replay.pos == total
    assert np.isfinite(replay.rewards[:total]).all()
    actions = replay.actions[:total].reshape(-1).astype(int)
    continuing_previous = replay.dones[:total-1, 0] == 0
    positions = np.arange(1,total)[continuing_previous]
    assert replay.next_action_masks[positions-1, 0, actions[positions]].all()
    dones = replay.dones[:total, 0].astype(bool)
    assert not replay.next_action_masks[:total][dones].any()
    assert replay.next_action_masks[:total][~dones].any(axis=-1).all()
    assert not replay.failure_penalties[:total].any()
    names = np.array(replay.reason_names, dtype=object)[replay.reasons[:total]]
    failures = np.array([isinstance(name,str) and name.startswith('failure_') for name in names])
    assert (replay.rewards[:total, 0][failures] == -1.).all()
    assert (replay.rewards[:total, 0][~failures] > 0.).all()
    if failures.any():
        rewards = torch.tensor(replay.rewards[:total][failures])
        target = masked_td_targets(rewards, torch.full((len(rewards),13),1e12),
            torch.ones_like(rewards), torch.zeros((len(rewards),13),dtype=torch.bool))
        assert torch.equal(rewards,target)
    for parameter in model.policy.state_dict().values():
        assert torch.isfinite(parameter).all()

    def log_checks(directory, steps, updates, formal):
        csvs = list(directory.rglob('progress.csv'))
        assert len(csvs)==1, csvs
        frame = pd.read_csv(csvs[0])
        columns = ['rollout/exploration_rate','time/fps','time/total_timesteps',
            'train/loss','train/n_updates','train/td_mae','replay/buffer_size',
            'vessel/soc','vessel/fc_power_kw','vessel/fc_start_count','vessel/economic_cost_cny',
            'validation/completion_rate','validation/mean_duration_coverage','validation/economic_cost_cny']
        assert set(columns) <= set(frame.columns)
        times = frame['time/total_timesteps'].astype(int).tolist()
        assert times == sorted(set(times)), 'duplicate or unordered native dumps'
        assert times[-1] == steps and frame['train/n_updates'].iloc[-1] == updates
        assert frame['vessel/formal_env_steps'].iloc[-1] == formal
        assert frame['replay/buffer_size'].iloc[-1] == steps
        for column in columns:
            assert np.isfinite(frame[column].dropna().astype(float)).all(),column
        events = list(directory.rglob('events.out.tfevents.*'))
        assert events
        accumulator = EventAccumulator(str(events[0].parent));accumulator.Reload()
        tags = accumulator.Tags()['scalars']
        for key in ('train/loss','train/td_mae','train/n_updates','vessel/soc',
                    'validation/completion_rate','validation/mean_duration_coverage'):
            assert key in tags,key
        if frame['validation/economic_cost_cny'].notna().any():
            assert 'validation/economic_cost_cny' in tags
        else:
            assert 'validation/economic_cost_cny' not in tags
        assert accumulator.Scalars('train/n_updates')[-1].value == updates
        return dict(csv=str(csvs[0]),dump_steps=times,tensorboard_tags=tags)

    logs = log_checks(train,4096,126,4032)
    assert {'rollout/ep_len_mean','rollout/ep_rew_mean'} <= set(pd.read_csv(logs['csv']).columns)
    inline = read(folder/'best_validation.json')
    independent = read(evaluation/'validation.json')
    for summary in (inline,independent):
        assert summary['episodes']==8 and len({r['sample_id'] for r in summary['samples']})==8
        assert all(r['split']=='validation' for r in summary['samples'])
        if summary['completed'] < 8:
            assert summary['fixed_complete_set_cost_cny'] is None
        else:
            assert np.isfinite(summary['fixed_complete_set_cost_cny'])
        assert summary['evaluation_replay_insertions']==summary['evaluation_adam_updates']==0
        assert summary['test_payloads_opened']==0
    study = optuna.load_study(study_name='vessel_zoo_dqn', storage='sqlite:///'+str(optimization/'study.sqlite3'))
    assert len(study.trials)==1 and study.trials[0].state==optuna.trial.TrialState.COMPLETE
    trial = study.trials[0]
    assert trial.value==trial.user_attrs['validation']['score']
    assert trial.user_attrs['validation']['episodes']==8
    trial_logs = log_checks(optimization/'trials/trial_0',1024,30,960)
    return dict(passed=True,real_train_steps=total,warmup_steps=64,formal_steps=model.formal_env_steps,
        adam_updates=model._n_updates,replay_insertions=replay.insertions,
        target_sync_unit=model.target_sync_unit,target_sync_history=model.target_sync_history,
        predecessor_mask_actions_checked=len(positions),runtime_illegal_actions=0,
        train_episode_terminals=int(dones.sum()),train_failure_terminals=int(failures.sum()),
        all_weights_rewards_finite=True,terminal_bootstrap_zero=True,train_logs=logs,
        inline_validation=inline,independent_final_model_validation=independent,
        optuna=dict(state=trial.state.name,value=trial.value,seed=trial.user_attrs['model_seed'],
                    train_steps=1024,adam_updates=30,formal_steps=960,logs=trial_logs,
                    validation=trial.user_attrs['validation']),
        no_dataset_payloads_opened_by_this_check=True,
        limits=['Predecessor Replay masks independently cover non-reset actions; all actions also passed the environment runtime legality guard.',
                'Inline best Validation precedes the final Adam; independent evaluation uses the saved final model.',
                'Short bounded results validate integration, not learned policy quality.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    base=ROOT/'outputs/zoo_vessel_dev'
    parser.add_argument('--train',type=Path,default=base/'real_train_4096')
    parser.add_argument('--evaluation',type=Path,default=base/'independent_eval')
    parser.add_argument('--optimization',type=Path,default=base/'real_optuna_one_trial')
    parser.add_argument('--output',type=Path,default=base/'artifacts_verification.json')
    args=parser.parse_args()
    result=verify(args.train.resolve(),args.evaluation.resolve(),args.optimization.resolve())
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({key:result[key] for key in ('passed','real_train_steps','formal_steps','adam_updates',
        'replay_insertions','predecessor_mask_actions_checked','train_episode_terminals')}))


if __name__=='__main__':
    main()
