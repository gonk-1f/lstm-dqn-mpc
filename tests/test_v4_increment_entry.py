"""Small interface checks for the isolated seven-increment experiment."""
import json
from types import SimpleNamespace

import pytest

from v4.independent_economics import ZhouEvaluationConfig


def test_increment_cli_routes_one_step_without_loading_payloads(tmp_path, monkeypatch):
    from v4 import feedback_study
    dataset = SimpleNamespace(opened_test_payloads=0, split_sizes=(35, 8),
        identity_hashes=lambda: {'split': 'frozen'})
    monkeypatch.setattr(feedback_study.IndependentTrainingDataset, 'open', lambda root: dataset)
    captured = {}
    def capture(selected, **kwargs):
        captured.update(kwargs)
        kwargs['output_dir'].mkdir()
        return None, {'rounds': [], 'best_checkpoint': None}
    monkeypatch.setattr(feedback_study, 'run_monitored_training', capture)
    assert feedback_study.main(['--output-dir', str(tmp_path/'run'),
        '--dataset-mode', 'independent', '--state-mode', 'minimal3',
        '--action-mode', 'increment7', '--reward-cost-mode', 'zhou_joint',
        '--n-step', '1', '--replay-sampling', 'prioritized']) == 0
    assert captured['action_mode'] == 'increment7'
    assert captured['n_step'] == 1 and captured['required_split_sizes'] == (35, 8)
    assert captured['reward_cost_mode'] == 'zhou_joint'
    assert captured['per_alpha'] == .6 and captured['per_beta'] == 1.


def test_increment_metadata_binds_action_and_reward_identity(tmp_path, monkeypatch):
    from v4 import monitored_training
    class BeforeAgent(Exception):
        pass
    def stop(**kwargs):
        assert kwargs['action_mode'] == 'increment7'
        assert kwargs['n_step'] == 1
        raise BeforeAgent()
    monkeypatch.setattr(monitored_training, 'DirectPowerDDQN', stop)
    monkeypatch.setattr(monitored_training, 'write_learning_curves', lambda *args: {})
    def episode(split):
        return SimpleNamespace(sample_id=split, split=split, task_mode='independent_voyage',
                               operating_mode=('onboard',))
    dataset = SimpleNamespace(task_mode='independent_voyage', task_contract={'version': 'synthetic'},
        opened_test_payloads=0, load_train=lambda: (episode('train'),),
        load_validation=lambda: (episode('validation'),))
    with pytest.raises(BeforeAgent):
        monitored_training.run_monitored_training(dataset, output_dir=tmp_path,
            state_mode='minimal3', action_mode='increment7', reward_cost_mode='zhou_joint',
            n_step=1, replay_sampling='prioritized',
            economic_evaluation_config=ZhouEvaluationConfig.from_formal_curve())
    metadata = json.loads((tmp_path/'run_metadata.json').read_text(encoding='utf-8'))
    hp = metadata['hyperparameters']
    assert hp['network_dims'] == [3, 128, 128, 7]
    assert hp['action_mode'] == 'increment7'
    assert hp['action_kw'] == [-60, -30, -10, 0, 10, 30, 60]
    assert hp['fc_quadratic_ramp_reward_coefficient'] == pytest.approx(.11025/60)
    assert hp['n_step'] == hp['gamma'] == 1
    assert hp['fc_degradation_reward_weight'] == hp['battery_degradation_reward_weight'] == 1.
    assert hp['terminal_charging'] is False and hp['shore_settlement'] == 'disabled_independent_voyage'
    incompatible = json.loads(json.dumps(metadata))
    incompatible['hyperparameters']['action_mode'] = 'absolute61'
    monkeypatch.setattr(monitored_training.torch, 'load', lambda *args, **kwargs:
        {'format_version': 1, 'metadata': incompatible})
    dataset.load_train = lambda: pytest.fail('identity must reject before reading any payload')
    with pytest.raises(ValueError, match='resume source, architecture, hyperparameters'):
        monitored_training.run_monitored_training(dataset, output_dir=tmp_path,
            state_mode='minimal3', action_mode='increment7', reward_cost_mode='zhou_joint',
            n_step=1, replay_sampling='prioritized', resume_from=tmp_path/'training_state_latest.pt',
            economic_evaluation_config=ZhouEvaluationConfig.from_formal_curve())


def test_increment_actual_q_diagnostics_use_delta_indices():
    from v3.control import EconomicMPC
    from v4.control import replay_episode
    from v4.dqn import DirectPowerDDQN
    from v4.decision_diagnostics import actual_greedy_q_records
    from v4.diagnostics import fixed_q_diagnostics
    agent = DirectPowerDDQN(state_mode='minimal3', action_mode='increment7', n_step=1)
    episode = SimpleNamespace(sample_id='synthetic', split='train', task_mode='independent_voyage',
        operating_mode=('onboard',)*2, load_kw=(100., 100.), time_s=(0., 30.),
        dt_s=(30., 12.), duration_s=42.)
    accountant = EconomicMPC(nominal_cost_cny=1.)
    result = replay_episode(episode, lambda state, feasible: agent.select_power(state, feasible),
        accountant=accountant, state_mode='minimal3', action_mode='increment7',
        reward_cost_mode='zhou_joint', economic_evaluation_config=ZhouEvaluationConfig.from_formal_curve())
    records = actual_greedy_q_records('synthetic', result.transitions, agent)
    assert records and all(len(item['q_values']) == 7 for item in records)
    assert all(item['selected_action_legal_rank'] == 1 for item in records)
    assert all(len(item['candidate_action_indices']) == len(item['candidate_actions_kw']) for item in records)
    assert all(len(item['q_values']) == 7 for item in fixed_q_diagnostics(agent, accountant,
        reward_cost_mode='zhou_joint'))
