"""Small V5 configuration/identity checks; no dataset payloads or training."""
import json
from types import SimpleNamespace

import pytest


V5_ARGS = ['--dataset-mode', 'independent', '--state-mode', 'load_delta4',
    '--action-mode', 'increment7', '--reward-cost-mode', 'zhou_economic_only',
    '--beta-soc', '0', '--n-step', '1', '--replay-sampling', 'prioritized',
    '--update-timing', 'interleaved', '--episode-credit-scope', 'voyage',
    '--experiment-id', 'V5-ECONOMIC4-S42']


def test_v5_cli_routes_frozen_configuration_without_loading_payloads(tmp_path, monkeypatch):
    from v4 import feedback_study
    dataset = SimpleNamespace(opened_test_payloads=0, split_sizes=(35, 8),
        identity_hashes=lambda: {'split': 'frozen'})
    monkeypatch.setattr(feedback_study.IndependentTrainingDataset, 'open', lambda root: dataset)
    captured = {}
    def capture(selected, **kwargs):
        assert selected is dataset
        captured.update(kwargs)
        kwargs['output_dir'].mkdir()
        return None, {'rounds': [], 'best_checkpoint': None}
    monkeypatch.setattr(feedback_study, 'run_monitored_training', capture)
    assert feedback_study.main(['--output-dir', str(tmp_path/'run'), *V5_ARGS]) == 0
    assert captured['state_mode'] == 'load_delta4'
    assert captured['reward_cost_mode'] == 'zhou_economic_only'
    assert captured['beta_soc'] == 0 and captured['n_step'] == 1
    assert captured['update_timing'] == 'interleaved' and captured['cadence'] == 'replay32'
    assert captured['batch_size'] == 64 and captured['reward_scale'] == .001
    assert captured['replay_sampling'] == 'prioritized'
    assert (captured['per_alpha'], captured['per_beta'], captured['per_epsilon']) == (.6, 1., 1e-6)
    assert captured['required_split_sizes'] == (35, 8)


@pytest.mark.parametrize(('option', 'value'), [
    ('--state-mode', 'minimal3'), ('--reward-cost-mode', 'zhou_joint'),
    ('--beta-soc', '200'), ('--update-timing', 'episode_end'),
    ('--replay-sampling', 'uniform'), ('--n-step', '8')])
def test_v5_rejects_hybrid_configuration_before_payload_loading(tmp_path, monkeypatch, option, value):
    from v4 import feedback_study
    monkeypatch.setattr(feedback_study.IndependentTrainingDataset, 'open',
        lambda *args: pytest.fail('invalid V5 configuration must reject before payload loading'))
    args = list(V5_ARGS)
    args[args.index(option)+1] = value
    with pytest.raises(SystemExit) as exc:
        feedback_study.main(['--output-dir', str(tmp_path/'run'), *args])
    assert exc.value.code == 2
    assert not (tmp_path/'run').exists()


def test_v5_metadata_and_3d_checkpoint_identity_are_isolated(tmp_path, monkeypatch):
    from v4 import monitored_training
    from v4.independent_economics import ZhouEvaluationConfig
    class BeforeAgent(Exception):
        pass
    def stop(**kwargs):
        assert kwargs['state_mode'] == 'load_delta4'
        assert kwargs['action_mode'] == 'increment7' and kwargs['n_step'] == 1
        raise BeforeAgent()
    monkeypatch.setattr(monitored_training, 'DirectPowerDDQN', stop)
    monkeypatch.setattr(monitored_training, 'write_learning_curves', lambda *args: {})
    def episode(split):
        return SimpleNamespace(sample_id=split, split=split, task_mode='independent_voyage',
            operating_mode=('onboard',))
    dataset = SimpleNamespace(task_mode='independent_voyage', task_contract={'version': 'synthetic'},
        opened_test_payloads=0, load_train=lambda: (episode('train'),),
        load_validation=lambda: (episode('validation'),))
    config = dict(state_mode='load_delta4', action_mode='increment7',
        reward_cost_mode='zhou_economic_only', beta_soc=0., n_step=1,
        replay_sampling='prioritized', update_timing='interleaved',
        experiment_id='V5-ECONOMIC4-S42', episode_credit_scope='voyage',
        economic_evaluation_config=ZhouEvaluationConfig.from_formal_curve())
    with pytest.raises(BeforeAgent):
        monitored_training.run_monitored_training(dataset, output_dir=tmp_path, **config)
    metadata = json.loads((tmp_path/'run_metadata.json').read_text(encoding='utf-8'))
    hp = metadata['hyperparameters']
    assert hp['network_dims'] == [4, 128, 128, 7]
    assert hp['beta_soc'] == hp['fc_quadratic_ramp_reward_coefficient'] == 0.
    assert hp['fc_ramp_reward_coefficient_cny_per_kw'] == 0.
    assert hp['h2_reward_weight'] == hp['soc_equivalent_hydrogen_reward_weight'] == 1.
    assert hp['fc_degradation_reward_weight'] == hp['battery_degradation_reward_weight'] == 1.
    assert hp['shore_purchase_reward_weight'] == 0. and hp['terminal_charging'] is False
    assert metadata['framework_version'] == 'v5_economic4'
    assert 'economic_action_pair_diagnostics' in metadata['metrics_contract']
    incompatible = json.loads(json.dumps(metadata))
    incompatible['hyperparameters']['state_mode'] = 'minimal3'
    incompatible['hyperparameters']['network_dims'][0] = 3
    monkeypatch.setattr(monitored_training.torch, 'load', lambda *args, **kwargs:
        {'format_version': 1, 'metadata': incompatible})
    dataset.load_train = lambda: pytest.fail('old checkpoint must reject before payload loading')
    with pytest.raises(ValueError, match='resume source, architecture, hyperparameters'):
        monitored_training.run_monitored_training(dataset, output_dir=tmp_path,
            resume_from=tmp_path/'training_state_latest.pt', **config)
