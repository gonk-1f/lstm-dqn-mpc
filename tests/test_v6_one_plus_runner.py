"""Reward identity at the real episode runner's public boundaries."""
import json
from pathlib import Path
import pickle

import numpy as np
import pytest
import torch

from v6_sb3 import episode_training as runner

MODE = 'inverse_one_plus_economic'


def options(**extra):
    return dict(smoke=True, episodes=2, learning_starts=64, buffer_size=512,
        batch_size=8, network_width=8, validation_every=2,
        target_update_interval=64, reward_mode=MODE, **extra)


def test_new_mode_exact_live_resume_validation_and_checkpoint(tmp_path):
    train, validation = runner._smoke_episodes()
    full = runner.run_episode_training(options(), train, validation, tmp_path/'full')
    paused = runner.run_episode_training(options(), train, validation, tmp_path/'resumed',
                                         stop_after_steps=97)
    assert paused['status'] == 'paused' and paused['formal_env_steps'] == 33
    resumed = runner.run_episode_training(options(), train, validation, tmp_path/'resumed', resume=True)
    for key in ('formal_env_steps', 'warmup_env_steps', 'adam_updates', 'sample_order',
                'sample_counts', 'validation_history', 'replay_insertions', 'best_score'):
        assert full[key] == resumed[key], key
    assert full['formal_env_steps'] == 128 and full['adam_updates'] == 4
    assert full['replay_insertions'] == 192
    summary = full['validation_history'][-1]
    assert summary['reward_mode'] == MODE
    assert summary['completed'] == 2
    assert summary['evaluation_adam_updates'] == summary['evaluation_replay_insertions'] == 0
    assert summary['test_payloads_opened'] == 0
    assert all(s['return_sum'] > 0 and s['complete_economic_cost_cny'] > 0
               for s in summary['samples'])
    from v6_sb3.masked_dqn import MaskedDQN
    a = MaskedDQN.load(Path(full['latest_checkpoint'])/'model.zip')
    b = MaskedDQN.load(Path(resumed['latest_checkpoint'])/'model.zip')
    assert a.reward_mode == b.reward_mode == MODE
    for key, value in a.policy.state_dict().items():
        assert torch.equal(value, b.policy.state_dict()[key]), key
    with (Path(full['latest_checkpoint'])/'replay.pkl').open('rb') as source:
        replay = pickle.load(source)
    assert np.isfinite(replay.rewards[:192]).all()
    assert (replay.rewards[:192] > 0).all()
    assert not replay.failure_penalties[:192].any()
    before = (tmp_path/'resumed'/'latest.json').read_bytes()
    with pytest.raises(ValueError, match='identity'):
        runner.run_episode_training({**options(), 'reward_mode':'zhou_economic_only'},
            train, validation, tmp_path/'resumed', resume=True)
    assert (tmp_path/'resumed'/'latest.json').read_bytes() == before


def test_new_identity_never_loads_legacy_failure_reference(monkeypatch):
    from v4.failure_replay import FailurePenalty
    from v6_sb3.identity import execution_identity
    def forbidden(*args, **kwargs):
        raise AssertionError('new reward must not load the old monetary penalty')
    monkeypatch.setattr(FailurePenalty, 'from_reference', forbidden)
    identity = execution_identity(runner.ROOT, {'train':[], 'validation':[]}, reward_mode=MODE)
    assert identity['failure_penalty'] is None
    assert identity['reward_contract']['cost_unit'] == 'CNY'


def test_parameter_import_rejects_old_reward_and_unlabelled_data(tmp_path):
    path = tmp_path/'parameters.json'
    for value in (
        {'learning_rate':1e-4},
        {'schema':'v6_episode_parameters_v1', 'parameters':{'learning_rate':1e-4},
         'reward_mode':'zhou_economic_only', 'target_sync_unit':'formal_env_steps'},
    ):
        path.write_text(json.dumps(value), encoding='utf-8')
        with pytest.raises(ValueError, match='reward|contract|schema'):
            runner.load_parameter_export(path, MODE)
    assert runner.load_parameter_export(path, 'zhou_economic_only') == {'learning_rate':1e-4}


def new_export():
    from v4.independent_economics import ZhouEvaluationConfig
    from v6_sb3.rewards import reward_contract
    from v6_sb3.identity import execution_identity
    return dict(schema='v6_episode_parameters_v2', parameters={'learning_rate':1e-4},
        reward_mode=MODE, reward_contract=reward_contract(MODE),
        economics=ZhouEvaluationConfig.from_formal_curve().identity(),
        target_sync_unit='formal_env_steps', selection_split='validation',
        source_files=execution_identity(runner.ROOT, {}, reward_mode=MODE)['source_files'])


@pytest.mark.parametrize('field,value', [
    ('reward_mode','zhou_economic_only'), ('target_sync_unit','adam_updates'),
    ('selection_split','test'), ('economics',{}), ('reward_contract',{}), ('source_files',{}),
])
def test_parameter_import_checks_the_export_envelope(tmp_path, field, value):
    path = tmp_path/'export.json'
    exported = new_export()
    path.write_text(json.dumps(exported), encoding='utf-8')
    assert runner.load_parameter_export(path, MODE) == {'learning_rate':1e-4}
    exported[field] = value
    path.write_text(json.dumps(exported), encoding='utf-8')
    with pytest.raises(ValueError):
        runner.load_parameter_export(path, MODE)


def test_comparison_rejects_old_mode_even_with_same_validation_data(tmp_path):
    from v4.independent_economics import ZhouEvaluationConfig
    from v6_sb3.rewards import reward_contract
    for name, mode in (('new', MODE), ('old', 'zhou_economic_only')):
        folder = tmp_path/name
        folder.mkdir()
        identity = dict(data={'validation':[{'sample_id':'same', 'sha256':'same'}]},
                        economics=ZhouEvaluationConfig.from_formal_curve().identity())
        if mode == MODE:
            identity['reward_contract'] = reward_contract(mode)
        for filename, value in (
            ('configuration.json', {'reward_mode':mode}),
            ('identity.json', identity), ('progress.json', {'status':'complete'}),
            ('best_validation.json', {'score':4., 'reward_mode':mode,
                                      'reward_contract':reward_contract(mode)}),
        ):
            (folder/filename).write_text(json.dumps(value), encoding='utf-8')
    with pytest.raises(ValueError, match='reward|mode'):
        runner.compare_runs(tmp_path/'new', tmp_path/'old')


def test_legacy_step_budget_executor_refuses_the_new_mode_before_execution():
    from v6_sb3.tuning import execute_experiment
    with pytest.raises(ValueError, match='episode|reward'):
        execute_experiment(None, None, None, None, None, MODE)


def test_failed_real_step_enters_replay_once_and_has_no_bootstrap(tmp_path):
    from types import SimpleNamespace
    from v6_sb3.masked_dqn import MaskedDQN, masked_td_targets
    train = (SimpleNamespace(sample_id='failure_train', split='train', task_mode='independent_voyage',
        operating_mode=('onboard',)*2, load_kw=(100., 1900.), time_s=(0., 1.),
        dt_s=(1., 1.), duration_s=2.),)
    _, validation = runner._smoke_episodes()
    config = {**options(), 'episodes':1, 'learning_starts':1}
    result = runner.run_episode_training(config, train, validation, tmp_path/'failed')
    assert result['formal_env_steps'] == 1 and result['replay_insertions'] == 2
    row = json.loads((tmp_path/'failed'/'episodes.jsonl').read_text().strip())
    assert row['return_sum'] == -1 and row['terminal_reason'] == 'structural_power'
    assert row['executed_prefix_economic_cost_cny'] > 0
    assert row['complete_economic_cost_cny'] is None
    model = MaskedDQN.load(Path(result['latest_checkpoint'])/'model.zip')
    model.load_replay_buffer(Path(result['latest_checkpoint'])/'replay.pkl')
    buffer = model.replay_buffer
    np.testing.assert_array_equal(buffer.rewards[:2], [[-1.], [-1.]])
    assert buffer.dones[:2].all() and not buffer.next_action_masks[:2].any()
    assert not buffer.failure_penalties[:2].any()
    target = masked_td_targets(torch.tensor(buffer.rewards[:2]), torch.full((2, 13), 1e12),
        torch.tensor(buffer.dones[:2]), torch.tensor(buffer.next_action_masks[:2, 0]))
    assert torch.equal(target, torch.full((2, 1), -1.))


@pytest.mark.parametrize('bad_cost', [-1., -2., float('nan'), float('inf')])
def test_bad_economics_stops_runner_before_replay_insertion(tmp_path, monkeypatch, bad_cost):
    from v6_sb3 import direct_environment
    from v6_sb3.masked_dqn import MaskedReplayBuffer
    inserted = []
    original = MaskedReplayBuffer.add
    def observe(buffer, *args, **kwargs):
        inserted.append(buffer.insertions)
        return original(buffer, *args, **kwargs)
    monkeypatch.setattr(MaskedReplayBuffer, 'add', observe)
    # FC-off is the first legal action: its ledger is zero, so the injected
    # SOC cost is exactly bad_cost rather than being offset by startup fees.
    monkeypatch.setattr(runner.random.Random, 'choice', lambda self, sequence:sequence[0])
    monkeypatch.setattr(direct_environment, 'onboard_reward_components',
                        lambda *args, **kwargs:(0., 0., 0., bad_cost))
    from types import SimpleNamespace
    train = (SimpleNamespace(sample_id='bad_train', split='train', task_mode='independent_voyage',
        operating_mode=('onboard',), load_kw=(0.,), time_s=(0.,), dt_s=(1.,), duration_s=1.),)
    _, validation = runner._smoke_episodes()
    with pytest.raises((ValueError, FloatingPointError)):
        runner.run_episode_training(options(), train, validation, tmp_path/'invalid')
    assert inserted == []
    error = json.loads((tmp_path/'invalid'/'error.json').read_text())
    assert error['formal_env_steps'] == error['adam_updates'] == 0


def test_cli_resume_rejects_explicit_mode_conflict_before_loading_data(tmp_path, monkeypatch):
    run = tmp_path/'run'
    run.mkdir()
    from dataclasses import asdict
    (run/'configuration.json').write_text(json.dumps(asdict(runner.EpisodeTrainingConfig(**options()))))
    monkeypatch.setattr('sys.argv', ['episode_training', '--run-dir', str(run), '--resume',
                                   '--reward-mode', 'zhou_economic_only'])
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2
    assert not (run/'latest.json').exists()


@pytest.mark.parametrize('mutation', ['unchanged', 'overwrite', 'delete'])
def test_new_mode_smoke_retraining_keeps_exported_model_parameters(tmp_path, monkeypatch, mutation):
    exported = new_export()
    exported['parameters'].update(buffer_size=256, batch_size=8, network_width=16,
                                  target_update_interval=32)
    path = tmp_path/'parameters.json'
    path.write_text(json.dumps(exported), encoding='utf-8')
    run = tmp_path/'retrained'
    observed = []
    def record(config, train, validation, directory, **kwargs):
        observed.append(config)
        Path(directory).mkdir()
        if mutation == 'overwrite':
            path.write_text('{}', encoding='utf-8')
        elif mutation == 'delete':
            path.unlink()
        return dict(status='complete', completed_episodes=3, formal_env_steps=192, adam_updates=6)
    monkeypatch.setattr(runner, 'run_episode_training', record)
    monkeypatch.setattr('sys.argv', ['episode_training', '--run-dir', str(run), '--smoke',
        '--reward-mode', MODE, '--parameters-json', str(path)])
    runner.main()
    assert len(observed) == 1
    for field, value in exported['parameters'].items():
        assert getattr(observed[0], field) == value
    assert observed[0].episodes == 3 and observed[0].learning_starts == 64
    assert json.loads((run/'parameter_source.json').read_text())['export'] == exported
