"""Small independent-voyage routing checks; synthetic loads only."""
import json
import random
from types import SimpleNamespace

import pytest
import numpy as np

from v3.control import EconomicMPC
from v4.control import replay_episode
from v4.independent_economics import ZhouEvaluationConfig
from v4.monitored_training import greedy_evaluate, run_monitored_training
from v4.train import _summarize


def episode(sample_id='synthetic', split='validation', loads=(100., 100.)):
    return SimpleNamespace(sample_id=sample_id, split=split,
        task_mode='independent_voyage', operating_mode=('onboard',) * len(loads),
        load_kw=loads, time_s=tuple(30. * i for i in range(len(loads))),
        dt_s=(30.,) * (len(loads) - 1) + (12.,), duration_s=30. * (len(loads) - 1) + 12.)


def test_summary_uses_signed_three_cost_total_without_charging():
    config = ZhouEvaluationConfig.from_formal_curve()
    result = replay_episode(episode(), lambda state, mask: 200,
        accountant=EconomicMPC(nominal_cost_cny=1.), state_mode='minimal3',
        reward_cost_mode='control_only', economic_evaluation_config=config)
    summary = _summarize([result], 1, [], task_mode='independent_voyage')
    assert result.economic_evaluation.soc_equivalent_hydrogen_mass_kg < 0
    assert summary['cost_cny'] == result.comparable_cost_cny
    assert summary['completed_cost_cny'] == result.economic_evaluation.total_cost_cny
    assert summary['completed_three_cost_components_cny']['equivalent_hydrogen_cost_cny'] == (
        result.economic_evaluation.equivalent_hydrogen_cost_cny)
    assert summary['completed_modeled_terminal_cost_cny'] == 0


def test_greedy_evaluation_routes_independent_contract_and_preserves_rng(tmp_path):
    agent = SimpleNamespace(random=random.Random(42), reward_scale=.001,
        select_power=lambda state, feasible, epsilon: 100)
    before = agent.random.getstate()
    target = tmp_path/'trajectory.json'
    summary = greedy_evaluate([episode()], agent, EconomicMPC(nominal_cost_cny=1.),
        200., state_mode='minimal3', reward_cost_mode='control_only',
        economic_evaluation_config=ZhouEvaluationConfig.from_formal_curve(), trajectory_path=target)
    assert summary['evaluable_completed'] == summary['evaluable_episodes'] == 1
    assert summary['physical_failures'] == summary['unknown_episodes'] == 0
    assert summary['economic_evaluation_basis'] == 'zhou_soc_equivalent_hydrogen_v1'
    profile = json.loads(target.read_text(encoding='utf-8'))['completed'][0]
    assert profile['economic_evaluation']['total_cost_cny'] == summary['cost_cny']
    assert profile['executed_duration_s'] == 42.
    assert agent.random.getstate() == before


def test_independent_physical_failure_cannot_be_selected_by_partial_cost():
    agent = SimpleNamespace(random=random.Random(42), reward_scale=.001,
        select_power=lambda state, feasible, epsilon: max(feasible))
    summary = greedy_evaluate([episode(loads=(100., 5000.))], agent,
        EconomicMPC(nominal_cost_cny=1.), 200., state_mode='minimal3',
        reward_cost_mode='control_only', economic_evaluation_config=ZhouEvaluationConfig.from_formal_curve())
    assert summary['physical_failures'] == 1 and summary['data_truncated_episodes'] == 0
    assert summary['cost_cny'] is None and summary['completed'] == 0


class MetadataDataset:
    task_mode = 'independent_voyage'
    task_contract = {'task_mode': task_mode, 'split_version': 'synthetic_v2',
        'initial_soc': .6, 'shore_control': False, 'terminal_charging': False}
    opened_test_payloads = 0
    def load_train(self):
        return (episode(split='train'),)
    def load_validation(self):
        return (episode(),)


def test_task_and_economics_identity_saved_before_agent_initialization(tmp_path, monkeypatch):
    from v4 import monitored_training
    class StopBeforeRollout(Exception):
        pass
    def stop(**kwargs):
        raise StopBeforeRollout()
    monkeypatch.setattr(monitored_training, 'DirectPowerDDQN', stop)
    monkeypatch.setattr(monitored_training, 'write_learning_curves', lambda *args: {})
    with pytest.raises(StopBeforeRollout):
        run_monitored_training(MetadataDataset(), output_dir=tmp_path,
            state_mode='minimal3', reward_cost_mode='control_only', replay_sampling='prioritized',
            economic_evaluation_config=ZhouEvaluationConfig.from_formal_curve())
    metadata = json.loads((tmp_path/'run_metadata.json').read_text(encoding='utf-8'))
    hp = metadata['hyperparameters']
    assert hp['task_mode'] == 'independent_voyage'
    assert hp['shore_settlement'] == 'disabled_independent_voyage'
    assert hp['economic_evaluation']['fc_reference_count'] == 60
    assert hp['economic_evaluation']['soc_correction'].startswith('signed')
    assert hp['network_dims'] == [3, 128, 128, 61] and hp['learning_rate'] == .001
    assert metadata['task_contract'] == MetadataDataset.task_contract
    # An otherwise matching old checkpoint cannot infer this new task contract.
    saved_without_contract = dict(metadata)
    saved_without_contract.pop('task_contract')
    (tmp_path/'run_metadata.json').write_text(json.dumps(saved_without_contract), encoding='utf-8')
    monkeypatch.setattr(monitored_training.torch, 'load', lambda *args, **kwargs:
        {'format_version': 1, 'metadata': saved_without_contract})
    dataset = MetadataDataset()
    monkeypatch.setattr(dataset, 'load_train', lambda: pytest.fail('resume must reject before payload loading'))
    with pytest.raises(ValueError, match='resume source, architecture, hyperparameters'):
        run_monitored_training(dataset, output_dir=tmp_path,
            resume_from=tmp_path/'training_state_latest.pt',
            state_mode='minimal3', reward_cost_mode='control_only', replay_sampling='prioritized',
            economic_evaluation_config=ZhouEvaluationConfig.from_formal_curve())


def test_legacy_dataset_rejects_independent_evaluation_before_output(tmp_path):
    with pytest.raises(ValueError, match='independent'):
        run_monitored_training(object(), output_dir=tmp_path/'invalid',
            economic_evaluation_config=ZhouEvaluationConfig.from_formal_curve())
    assert not (tmp_path/'invalid').exists()


def test_independent_cli_uses_frozen_split_identity_and_curve_reference(tmp_path, monkeypatch):
    from v4 import feedback_study
    dataset = SimpleNamespace(opened_test_payloads=0, split_sizes=(35, 8),
        identity_hashes=lambda: {'metadata/new/sample_manifest.csv': 'frozen'},
        task_contract=MetadataDataset.task_contract)
    monkeypatch.setattr(feedback_study.IndependentTrainingDataset, 'open', lambda root: dataset)
    captured = {}
    def capture(selected, **kwargs):
        assert selected is dataset
        captured.update(kwargs)
        kwargs['output_dir'].mkdir()
        return None, {'rounds': [], 'best_checkpoint': None}
    monkeypatch.setattr(feedback_study, 'run_monitored_training', capture)
    assert feedback_study.main(['--output-dir', str(tmp_path/'run'),
        '--dataset-mode', 'independent', '--data-root', str(tmp_path/'dataset'),
        '--state-mode', 'minimal3', '--reward-cost-mode', 'control_only',
        '--replay-sampling', 'prioritized']) == 0
    assert captured['required_split_sizes'] == (35, 8)
    assert captured['manifest_sha256'] == dataset.identity_hashes()
    assert captured['economic_evaluation_config'].identity()['fc_reference_count'] == 60
    assert captured['rounds'] == 100 and captured['n_step'] == 8
    assert captured['per_alpha'] == .6 and captured['per_beta'] == 1.


def test_independent_qualification_cannot_hide_unknown_samples():
    from v4.experiment_schedule import fully_completed
    summary = {'task_mode': 'independent_voyage', 'episodes': 35, 'completed': 34,
        'evaluable_episodes': 34, 'evaluable_completed': 34, 'physical_failures': 0,
        'execution_errors': 0, 'data_truncated_episodes': 1, 'unknown_episodes': 1}
    assert not fully_completed(summary)


def test_independent_source_rows_are_replay_compatible_python_integers():
    from v4.monitored_training import _replay_source
    item = episode()
    item.source_1s_row = np.array([300, 330], dtype=np.int64)
    source = _replay_source(item, (object(), object()), 1)
    assert source['source_rows'] == (300, 330)
    assert all(type(row) is int for row in source['source_rows'])


def test_independent_diagnostic_artifact_uses_declared_sample_ids(tmp_path):
    import torch
    from v4.decision_diagnostics import save_diagnostic_snapshot
    model = SimpleNamespace(state_dict=lambda: {'weight': torch.zeros(1)})
    agent = SimpleNamespace(online=model, target=model, economic_optimizer_updates=0,
        target_sync_calls=1, economic_replay_insertions=0)
    metadata = {'source_commit': 'synthetic', 'source_worktree_dirty': True,
        'hyperparameters': {'task_mode': 'independent_voyage'},
        'manifest_sha256': {}, 'dataset_roots': [],
        'diagnostic_configuration': {'train_sample_ids': ['parent_015_seg01']}}
    artifact = save_diagnostic_snapshot(tmp_path, round_index=1, agent=agent, decisions=[],
        run_metadata=metadata, training_environment_transitions=0, test_payloads_opened=0)
    payload = json.loads((tmp_path/artifact['decisions']).read_text(encoding='utf-8'))
    assert payload['sample_ids'] == ['parent_015_seg01']


def test_independent_fixed_q_probe_labels_control_reward_units():
    import torch
    from v4.diagnostics import fixed_q_diagnostics
    agent = SimpleNamespace(state_mode='minimal3', reward_scale=.001,
        online=lambda state: torch.zeros((len(state), 61)))
    probes = fixed_q_diagnostics(agent, EconomicMPC(nominal_cost_cny=1.),
        reward_cost_mode='control_only')
    assert all(probe['q_unit'] == 'scaled control reward units' for probe in probes)
    assert all(probe['q_original_unit'] == 'original control reward units; not CNY expenditure'
               for probe in probes)
