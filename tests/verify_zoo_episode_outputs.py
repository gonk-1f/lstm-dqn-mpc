"""Audit completed real Zoo runs without training, evaluating or opening datasets.

Only existing models, Replay, JSON/CSV/TensorBoard and read-only SQLite are read.
This supplements the synthetic boundary tests with actual real-voyage evidence.
"""
import argparse
import gc
import json
from pathlib import Path
import pickle
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def verify(train, optimization, timestep, evaluation):
    import numpy as np
    import pandas as pd
    import torch
    from optuna.distributions import json_to_distribution
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from experiments.sb3_manual.zoo_vessel.algorithm import ZooMaskedDQN
    from v6_sb3.masked_dqn import masked_td_targets

    torch.set_num_threads(1)

    def finite_model(model):
        for name, tensor in model.policy.state_dict().items():
            assert torch.isfinite(tensor).all(), ('nonfinite policy tensor', name)
        states = model.policy.optimizer.state_dict()['state']
        assert states or model._n_updates == 0, 'trained model has no Adam state'
        for state in states.values():
            for name, value in state.items():
                if torch.is_tensor(value):
                    assert torch.isfinite(value).all(), ('nonfinite Adam tensor', name)
            assert float(state['step']) == model._n_updates, 'Adam tensor step differs from model counter'

    def clocks(model, summary, *, episodes=None, warmup=None):
        total, formal = model.num_timesteps, model.formal_env_steps
        assert total == summary['n_timesteps'] == summary['replay_insertions']
        assert formal == summary['formal_env_steps'] == total - model.learning_starts
        assert model._n_updates == summary['adam_updates'] == formal // 32
        assert model.target_sync_unit == 'formal_env_steps'
        assert model.train_freq.frequency == 32 and model.train_freq.unit.value == 'step'
        assert model.gradient_steps == 1 and model.gamma == 1. and model.device.type == 'cpu'
        expected = [dict(formal_env_steps=n, adam_updates=(n - 1) // 32)
                    for n in range(model.target_update_interval, formal + 1, model.target_update_interval)]
        assert model.target_sync_history == summary['target_sync_history'] == expected
        assert model.target_sync_calls == summary['target_sync_calls_including_initial'] == len(expected) + 1
        assert summary['target_sync_count'] == len(expected)
        assert summary['test_payloads_opened'] == 0
        if warmup is not None:
            assert model.learning_starts == summary['warmup_env_steps'] == warmup
        if episodes is not None:
            assert summary['outcome'] == 'COMPLETE'
            assert model.budget_mode == summary['budget_mode'] == 'episode'
            assert model.n_episodes == summary['requested_n_episodes'] == episodes
            assert model.completed_formal_episodes == summary['completed_formal_episodes'] == episodes
            assert model.formal_successes + model.formal_failures == episodes
            assert summary['train_successes'] == model.formal_successes
            assert summary['train_failures'] == model.formal_failures
            assert model.exploration_schedule_mode == summary['exploration_schedule_mode'] == 'episode'
            assert model.exploration_rate == summary['final_epsilon'] == .05
            assert summary['internal_safety_limit'] >= total
            assert summary['budget_unit'] == 'complete_formal_train_episodes'
            history = model.formal_episode_history
            assert len(history) == episodes
            assert [row['episode'] for row in history] == list(range(1, episodes + 1))
            assert all(row['split'] == 'train' for row in history)
            assert sum(row['completed'] for row in history) == model.formal_successes
            start = history[0]['total_env_steps'] - history[0]['episode_env_steps']
            assert start >= warmup
            assert start - warmup == model.warmup_bridge_env_steps == summary['warmup_bridge_env_steps']
            for index, row in enumerate(history):
                assert row['episode_env_steps'] > 0
                assert row['total_env_steps'] == start + row['episode_env_steps']
                assert row['formal_env_steps'] == row['total_env_steps'] - warmup
                assert row['replay_insertions'] == row['total_env_steps']
                assert row['adam_updates'] == row['formal_env_steps'] // 32
                expected_epsilon = max(.05, 1. - .95 * index / (.8 * episodes))
                assert np.isclose(row['epsilon_used'], expected_epsilon, rtol=0, atol=1e-12)
                start = row['total_env_steps']
            assert history[-1]['total_env_steps'] == total, 'an action ran after the final formal episode'
        finite_model(model)

    def replay_checks(model, summary):
        path = Path(summary['final_model']).parent / 'replay_buffer.pkl'
        with path.open('rb') as source:
            replay = pickle.load(source)
        total = summary['n_timesteps']
        assert not replay.full and replay.pos == total == replay.insertions
        assert np.isfinite(replay.observations[:total]).all()
        assert np.isfinite(replay.next_observations[:total]).all()
        assert np.isfinite(replay.rewards[:total]).all()
        assert not replay.timeouts[:total].any(), 'a fake time-limit terminal was stored'
        dones = replay.dones[:total, 0].astype(bool)
        masks = replay.next_action_masks[:total, 0]
        assert not masks[dones].any()
        assert masks[~dones].any(axis=1).all()
        assert not replay.failure_penalties[:total].any(), 'legacy failure cost entered Replay'
        actions = replay.actions[:total].reshape(-1).astype(int)
        assert ((0 <= actions) & (actions < 13)).all()
        positions = np.arange(1, total)[~dones[:-1]]
        assert masks[positions - 1, actions[positions]].all(), 'illegal action after a continuing predecessor'
        reasons = np.asarray(replay.reason_names, dtype=object)[replay.reasons[:total]]
        failures = np.asarray([isinstance(name, str) and name.startswith('failure_') for name in reasons])
        assert (replay.rewards[:total, 0][failures] == -1.).all()
        assert (replay.rewards[:total, 0][~failures] > 0.).all()
        assert (dones[failures]).all()
        terminal_rewards = torch.tensor(replay.rewards[:total][dones])
        targets = masked_td_targets(terminal_rewards,
            torch.full((len(terminal_rewards), 13), 1e12), torch.ones_like(terminal_rewards),
            torch.zeros((len(terminal_rewards), 13), dtype=torch.bool))
        assert torch.equal(targets, terminal_rewards), 'a terminal transition bootstrapped'
        if summary['budget_mode'] == 'episode':
            for row in model.formal_episode_history:
                end = row['total_env_steps']
                start = end - row['episode_env_steps']
                assert start == 0 or dones[start - 1], 'formal voyage did not start at a natural reset'
                assert not dones[start:end - 1].any(), 'a voyage contains an earlier terminal'
                assert dones[end - 1]
                assert bool(failures[end - 1]) == (not row['completed'])
            assert dones[-1] and not masks[-1].any()
        return dict(path=str(path), insertions=total, terminals=int(dones.sum()),
            failure_terminals=int(failures.sum()), predecessor_mask_actions_checked=int(len(positions)),
            final_terminal=bool(dones[-1]), terminal_bootstrap_zero=True,
            all_observations_rewards_finite=True, legacy_failure_penalties_zero=True)

    def validation_checks(value):
        rows = value['samples']
        assert value['episodes'] == len(rows) == len({row['sample_id'] for row in rows}) == 8
        assert all(row['split'] == 'validation' for row in rows)
        assert value['completed'] == sum(bool(row['completed']) for row in rows)
        assert value['completion_rate'] == value['completed'] / 8
        assert np.isclose(value['mean_duration_coverage'], sum(row['duration_coverage'] for row in rows) / 8)
        if value['completed'] == 8:
            assert np.isfinite(value['fixed_complete_set_cost_cny'])
            assert np.isclose(value['fixed_complete_set_cost_cny'], sum(row['complete_economic_cost_cny'] for row in rows))
        else:
            assert value['fixed_complete_set_cost_cny'] is None
        assert value['evaluation_adam_updates'] == value['evaluation_replay_insertions'] == 0
        assert value['test_payloads_opened'] == 0

    def native_logs(directory, summary, *, validation_records):
        csvs = list(directory.rglob('progress.csv'))
        assert len(csvs) == 1, csvs
        frame = pd.read_csv(csvs[0])
        needed = {'rollout/ep_len_mean', 'rollout/ep_rew_mean', 'rollout/exploration_rate',
            'time/fps', 'time/total_timesteps', 'train/loss', 'train/n_updates', 'train/adam_updates',
            'train/td_mae', 'train/completed_formal_episodes', 'train/formal_env_steps',
            'train/warmup_env_steps', 'train/warmup_bridge_env_steps', 'train/budget_mode',
            'train/exploration_schedule_mode', 'replay/buffer_size', 'replay/insertions',
            'replay/pending_transition', 'vessel/sample_id', 'vessel/soc', 'vessel/fc_power_kw',
            'vessel/economic_cost_cny', 'validation/completion_rate',
            'validation/mean_duration_coverage', 'validation/economic_cost_cny'}
        assert needed <= set(frame.columns), sorted(needed - set(frame.columns))
        times = frame['time/total_timesteps'].astype(int).tolist()
        assert times == sorted(set(times)), 'duplicate or unordered SB3 tables'
        assert times[-1] == summary['n_timesteps']
        assert set(range(1000, summary['n_timesteps'] + 1, 1000)) <= set(times)
        strings = {'vessel/sample_id', 'train/budget_mode', 'train/exploration_schedule_mode'}
        for column in needed - strings:
            assert np.isfinite(frame[column].dropna().astype(float)).all(), column
        assert (frame['train/budget_mode'] == summary['budget_mode']).all()
        assert (frame['train/exploration_schedule_mode'] == summary['exploration_schedule_mode']).all()
        assert (frame['time/total_timesteps'] - frame['replay/insertions'] == frame['replay/pending_transition']).all()
        final = frame.iloc[-1]
        for column, key in [('train/n_updates', 'adam_updates'), ('train/adam_updates', 'adam_updates'),
            ('train/formal_env_steps', 'formal_env_steps'), ('train/warmup_env_steps', 'warmup_env_steps'),
            ('train/warmup_bridge_env_steps', 'warmup_bridge_env_steps'),
            ('train/completed_formal_episodes', 'completed_formal_episodes'), ('replay/insertions', 'replay_insertions')]:
            assert int(final[column]) == summary[key], (column, final[column], summary[key])
        assert final['replay/pending_transition'] == 0
        assert np.isclose(final['rollout/exploration_rate'], summary['final_epsilon'])
        events = list(directory.rglob('events.out.tfevents.*'))
        assert events
        folders = {event.parent for event in events}
        assert len(folders) == 1, folders
        accumulator = EventAccumulator(str(next(iter(folders))), size_guidance={'scalars': 0})
        accumulator.Reload()
        tags = set(accumulator.Tags()['scalars'])
        for key in ('train/loss', 'train/td_mae', 'train/n_updates', 'train/adam_updates',
                    'train/completed_formal_episodes', 'train/formal_env_steps', 'vessel/soc',
                    'validation/completion_rate', 'validation/mean_duration_coverage'):
            assert key in tags, key
            assert all(np.isfinite(event.value) for event in accumulator.Scalars(key)), key
        assert accumulator.Scalars('train/n_updates')[-1].value == summary['adam_updates']
        assert accumulator.Scalars('train/n_updates')[-1].step == summary['n_timesteps']
        known_costs = [value['fixed_complete_set_cost_cny'] for value in validation_records
                       if value['fixed_complete_set_cost_cny'] is not None]
        if known_costs:
            for value in frame['validation/economic_cost_cny'].dropna():
                assert any(np.isclose(value, cost) for cost in known_costs)
            assert 'validation/economic_cost_cny' in tags
        else:
            assert frame['validation/economic_cost_cny'].isna().all()
            assert 'validation/economic_cost_cny' not in tags, 'None cost became fake TensorBoard zero'
        return dict(csv=str(csvs[0]), dump_steps=times, tensorboard_scalar_tags=sorted(tags),
            no_duplicate_dumps=True, unknown_economic_cost_not_filled_with_zero=True)

    train_summary = read(train / 'run_summary.json')
    model = ZooMaskedDQN.load(train_summary['final_model'], device='cpu')
    clocks(model, train_summary, episodes=3, warmup=50000)
    assert model.policy.net_arch == [256, 256] and model.batch_size == 64
    assert model.buffer_size == 1000000 and model.target_update_interval == 10000
    replay_report = replay_checks(model, train_summary)
    history = model.formal_episode_history
    folder = Path(train_summary['final_model']).parent
    validations = read(folder / 'validation.json')
    assert [value['completed_formal_episodes'] for value in validations] == [3]
    final_validation = validations[-1]
    validation_checks(final_validation)
    assert final_validation == train_summary['final_validation']
    assert final_validation['training_env_steps'] == train_summary['n_timesteps']
    assert final_validation['adam_updates'] == train_summary['adam_updates']
    assert final_validation['replay_insertions'] == train_summary['replay_insertions']
    assert final_validation['evaluation_position'] == 'after terminal Replay insertion and due Adam update'
    assert Path(train_summary['best_model']).is_file()
    assert Path(train_summary['best_model']).resolve() != Path(train_summary['final_model']).resolve()
    best = read(folder / 'best_validation.json')
    assert best == final_validation
    train_logs = native_logs(train, train_summary, validation_records=validations)
    del model
    gc.collect()

    independent = read(evaluation / 'validation.json')
    validation_checks(independent)
    assert independent['samples'] == final_validation['samples'], 'saved final actor differs from final inline evaluation'
    assert independent['score'] == final_validation['score']
    assert independent['environment_steps'] == final_validation['environment_steps']

    step_summary = read(timestep / 'run_summary.json')
    step_model = ZooMaskedDQN.load(step_summary['final_model'], device='cpu')
    clocks(step_model, step_summary, warmup=64)
    assert step_model.num_timesteps == 4096 and step_model.formal_env_steps == 4032
    assert step_model._n_updates == 126 and step_model.budget_mode == 'timestep'
    assert step_model.exploration_schedule_mode == 'timestep' and step_model.exploration_rate == .05
    step_replay = replay_checks(step_model, step_summary)
    step_validations = read(Path(step_summary['final_model']).parent / 'validation.json')
    for value in step_validations:
        validation_checks(value)
    step_logs = native_logs(timestep, step_summary, validation_records=step_validations)
    del step_model
    gc.collect()

    database = optimization / 'study.sqlite3'
    connection = sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True)
    try:
        trials = connection.execute('SELECT trial_id,number,state FROM trials ORDER BY number').fetchall()
        assert len(trials) == 1 and trials[0][1:] == (0, 'COMPLETE'), trials
        trial_id = trials[0][0]
        attrs = {key: json.loads(value) for key, value in connection.execute(
            'SELECT key,value_json FROM trial_user_attributes WHERE trial_id=?', (trial_id,))}
        value, value_type = connection.execute(
            'SELECT value,value_type FROM trial_values WHERE trial_id=? AND objective=0', (trial_id,)).fetchone()
        assert value_type == 'FINITE' and np.isfinite(value)
        params = {name: json_to_distribution(distribution).to_external_repr(internal)
            for name, internal, distribution in connection.execute(
                'SELECT param_name,param_value,distribution_json FROM trial_params WHERE trial_id=?', (trial_id,))}
    finally:
        connection.close()
    trial_folder = optimization / 'trials' / 'trial_0'
    trial_summary = read(trial_folder / 'run_summary.json')
    assert trial_summary == attrs['training_summary']
    assert trial_summary['hyperparameters'] == params
    assert attrs['n_episodes'] == 1 and attrs['budget_unit'] == 'complete_formal_train_episodes'
    assert attrs['exploration_schedule_mode'] == 'episode'
    trial_model = ZooMaskedDQN.load(trial_summary['final_model'], device='cpu')
    clocks(trial_model, trial_summary, episodes=1, warmup=50000)
    assert value == attrs['validation']['score'] == trial_summary['final_validation']['score']
    trial_validations = read(trial_folder / 'validation.json')
    assert [row['completed_formal_episodes'] for row in trial_validations] == [1]
    trial_validation = trial_validations[-1]
    validation_checks(trial_validation)
    assert trial_validation['adam_updates'] == trial_model._n_updates
    assert trial_validation['training_env_steps'] == trial_model.num_timesteps
    trial_logs = native_logs(trial_folder, trial_summary, validation_records=trial_validations)
    return dict(passed=True, real_train_steps=train_summary['n_timesteps'],
        formal_episodes=3, formal_steps=train_summary['formal_env_steps'], warmup_steps=50000,
        warmup_bridge_steps=train_summary['warmup_bridge_env_steps'],
        adam_updates=train_summary['adam_updates'], replay=replay_report,
        formal_episode_history=history, target_sync_history=train_summary['target_sync_history'],
        train_successes=train_summary['train_successes'], train_failures=train_summary['train_failures'],
        train_logs=train_logs, final_validation=final_validation,
        independently_loaded_final_model_validation=independent,
        best_model=train_summary['best_model'], final_model=train_summary['final_model'],
        timestep_regression=dict(total_steps=4096, formal_steps=4032, adam_updates=126,
            replay=step_replay, logs=step_logs),
        official_optuna_trial=dict(state='COMPLETE', value=value, params=params,
            summary=trial_summary, logs=trial_logs, sqlite_opened_read_only=True),
        all_weights_optimizer_states_rewards_finite=True,
        no_dataset_payloads_opened_by_this_check=True,
        limits=['Independent predecessor-mask checks cover non-reset actions; resets are covered by the environment runtime legality guard.',
                'Trial model files do not include Replay; actual counters and saved episode history supplement native unit tests of storage/update timing.',
                'A short real run verifies execution semantics, not satisfactory learned vessel control.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('train', 'optimization', 'timestep', 'evaluation', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.train.resolve(), args.optimization.resolve(), args.timestep.resolve(), args.evaluation.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({key: result[key] for key in ('passed', 'real_train_steps', 'formal_episodes',
        'formal_steps', 'warmup_steps', 'warmup_bridge_steps', 'adam_updates', 'train_successes', 'train_failures')}))


if __name__ == '__main__':
    main()
