"""Energy reward routing with synthetic episodes; no formal payloads."""
import json

import pytest

from test_v4_per_training_entry import (
    _MetadataOnlyDataset, _StopAtAgent, _capture_agent_configuration, _capture_cli,
)
from test_v4_training import FakeDataset


@pytest.mark.parametrize('mode', ['full_economic', 'energy_only', 'energy_with_fc_ramp'])
def test_cli_routes_reward_cost_mode_without_changing_frozen_configuration(tmp_path, monkeypatch, mode):
    captured = _capture_cli(tmp_path, monkeypatch, ['--reward-cost-mode', mode,
                                                  '--replay-sampling', 'prioritized'])
    assert captured['reward_cost_mode'] == mode
    assert captured['replay_sampling'] == 'prioritized'
    assert captured['rounds'] == 100 and captured['n_step'] == 8
    assert captured['beta_soc'] == 200 and captured['reward_scale'] == .001
    assert captured['failure_terminal_quota'] == 2 and captured['cadence'] == 'replay32'
    assert (captured['per_alpha'], captured['per_beta'], captured['per_epsilon']) == (.6, 1., 1e-6)
    assert captured['batch_size'] == 64 and captured['seed'] == 42
    assert captured['unknown_replay_mode'] == 'observed_nstep'


@pytest.mark.parametrize('mode, expected_formula', [
    ('full_economic',
     '-(C_H2 + C_FC_deg + C_Bat_deg + C_shore_purchase + beta_soc * (SOC_next - soc_ref)**2)'),
    ('energy_only', '-(C_H2 + C_shore_purchase + beta_soc * (SOC_next - soc_ref)**2)'),
    ('energy_with_fc_ramp',
     '-(C_H2 + C_shore_purchase + beta_soc * (SOC_next - soc_ref)**2'
     ' + fc_ramp_reward_coefficient_cny_per_kw * abs(actual_fc_kw - previous_actual_fc_kw))'),
])
def test_reward_metadata_records_unscaled_fc_ramp_coefficient_and_formula(
        tmp_path, monkeypatch, mode, expected_formula):
    from v4.control import FC_RAMP_REWARD_CNY_PER_KW

    captured, metadata = _capture_agent_configuration(tmp_path, monkeypatch, {
        'reward_cost_mode': mode, 'replay_sampling': 'prioritized',
    })
    hp = metadata['hyperparameters']
    assert hp['reward_cost_mode'] == mode
    expected_coefficient = FC_RAMP_REWARD_CNY_PER_KW if mode == 'energy_with_fc_ramp' else 0.
    assert hp.get('fc_ramp_reward_coefficient_cny_per_kw') == expected_coefficient
    expected_degradation_weight = 1. if mode == 'full_economic' else 0.
    assert hp['fc_degradation_reward_weight'] == expected_degradation_weight
    assert hp['battery_degradation_reward_weight'] == expected_degradation_weight
    assert hp['reward_formula'] == metadata['reward_definition'] == expected_formula
    assert hp['reward_scale'] == captured['reward_scale'] == .001
    assert captured['replay_sampling'] == 'prioritized'
    assert (captured['per_alpha'], captured['per_beta'], captured['per_epsilon']) == (.6, 1., 1e-6)


@pytest.mark.parametrize('entry', ['cli', 'wrapper'])
def test_unknown_reward_cost_mode_is_rejected_before_data_or_output(tmp_path, monkeypatch, entry):
    from v4 import feedback_study, monitored_training
    output = tmp_path/'invalid'
    if entry == 'cli':
        monkeypatch.setattr(feedback_study.FormalTrainingDataset, 'open',
                            lambda *args: pytest.fail('must reject before dataset opening'))
        with pytest.raises(SystemExit):
            feedback_study.main(['--output-dir', str(output), '--reward-cost-mode', 'invalid'])
    else:
        with pytest.raises(ValueError, match='reward_cost_mode'):
            monitored_training.run_monitored_training(
                object(), output_dir=output, reward_cost_mode='invalid')
    assert not output.exists()


def test_energy_only_reaches_bootstrap_exploration_and_both_greedy_paths(tmp_path, monkeypatch):
    from v4 import monitored_training, train
    original = train.replay_episode
    calls = []

    def capture(episode, policy, **kwargs):
        assert kwargs['reward_cost_mode'] == 'energy_only'
        calls.append(episode.split)
        return original(episode, policy, **kwargs)

    monkeypatch.setattr(train, 'replay_episode', capture)
    monkeypatch.setattr(monitored_training, 'replay_episode', capture)
    agent, report = monitored_training.run_monitored_training(
        FakeDataset(), output_dir=tmp_path, rounds=1, beta_soc=200,
        reward_cost_mode='energy_only', replay_sampling='prioritized')
    assert calls == ['train', 'train', 'train', 'validation']
    assert report['test_payloads_opened'] == 0
    hp = report['hyperparameters']
    assert hp['reward_cost_mode'] == 'energy_only'
    assert hp['fc_degradation_reward_weight'] == hp['battery_degradation_reward_weight'] == 0.
    assert hp['economic_replay_format_version'] == agent.training_state()['format_version'] == 3
    assert hp['reward_formula'] == '-(C_H2 + C_shore_purchase + beta_soc * (SOC_next - soc_ref)**2)'
    assert report['reward_definition'] == hp['reward_formula']
    assert report['best_checkpoint'] is not None
    assert report['best_checkpoint']['validation_comparable_cost_cny'] == pytest.approx(
        report['rounds'][0]['greedy_validation']['comparable_evaluable_cost_cny'])


def test_reward_cost_mode_mismatch_cannot_resume_before_payload_loading(tmp_path, monkeypatch):
    from v4 import monitored_training

    def stop(**kwargs):
        raise _StopAtAgent('stop before rollout')

    monkeypatch.setattr(monitored_training, 'DirectPowerDDQN', stop)
    monkeypatch.setattr(monitored_training, 'write_learning_curves', lambda *args: {})
    with pytest.raises(_StopAtAgent):
        monitored_training.run_monitored_training(
            _MetadataOnlyDataset(), output_dir=tmp_path, reward_cost_mode='full_economic')
    saved = json.loads((tmp_path/'run_metadata.json').read_text(encoding='utf-8'))
    assert saved['hyperparameters']['fc_degradation_reward_weight'] == 1.
    assert saved['hyperparameters']['battery_degradation_reward_weight'] == 1.
    monkeypatch.setattr(monitored_training.torch, 'load',
                        lambda *args, **kwargs: {'format_version': 1, 'metadata': saved})
    dataset = _MetadataOnlyDataset()
    monkeypatch.setattr(dataset, 'load_train', lambda: pytest.fail('resume must fail before loading Train'))
    with pytest.raises(ValueError, match='resume source, architecture, hyperparameters'):
        monitored_training.run_monitored_training(dataset, output_dir=tmp_path,
            resume_from=tmp_path/'training_state_latest.pt', reward_cost_mode='energy_only')


@pytest.mark.parametrize('saved_mode, resume_mode, coefficient_mismatch', [
    ('energy_only', 'energy_with_fc_ramp', None),
    ('energy_with_fc_ramp', 'energy_only', None),
    ('energy_with_fc_ramp', 'energy_with_fc_ramp', 'different'),
    ('energy_with_fc_ramp', 'energy_with_fc_ramp', 'missing'),
])
def test_fc_ramp_resume_rejects_mode_or_coefficient_mismatch_before_payload_loading(
        tmp_path, monkeypatch, saved_mode, resume_mode, coefficient_mismatch):
    from v4 import monitored_training

    _, saved = _capture_agent_configuration(tmp_path, monkeypatch, {
        'reward_cost_mode': saved_mode, 'replay_sampling': 'prioritized',
    })
    if coefficient_mismatch == 'different':
        saved['hyperparameters']['fc_ramp_reward_coefficient_cny_per_kw'] = 0.
    elif coefficient_mismatch == 'missing':
        saved['hyperparameters'].pop('fc_ramp_reward_coefficient_cny_per_kw', None)
    (tmp_path/'run_metadata.json').write_text(json.dumps(saved), encoding='utf-8')
    monkeypatch.setattr(monitored_training.torch, 'load',
                        lambda *args, **kwargs: {'format_version': 1, 'metadata': saved})
    dataset = _MetadataOnlyDataset()
    for method in ('load_train', 'load_validation'):
        monkeypatch.setattr(dataset, method,
                            lambda: pytest.fail('resume must fail before loading dataset payloads'))
    with pytest.raises(ValueError, match='resume source, architecture, hyperparameters'):
        monitored_training.run_monitored_training(
            dataset, output_dir=tmp_path, resume_from=tmp_path/'training_state_latest.pt',
            reward_cost_mode=resume_mode, replay_sampling='prioritized')
