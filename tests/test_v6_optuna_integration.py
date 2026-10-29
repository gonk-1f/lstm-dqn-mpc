"""A few synthetic seconds exercise storage, masks, pruning and exact resume."""
import json
from pathlib import Path
import pickle

import numpy as np
import optuna
import pytest
import torch

from v6_sb3.masked_dqn import MaskedDQN
from v6_sb3.tuning import make_smoke_episodes, train_trial
from v6_sb3 import tuning


def config():
    return dict(experiment_id='synthetic_v6_baseline_smoke', reward_mode='v6_energy_soc_ramp',
        reward_scale=1., seed=42, train_steps=4, learning_starts=2, eval_every=2,
        parameters=dict(learning_rate=.0001, batch_size=4, buffer_size=32,
                        target_update_interval=2, policy_kwargs=dict(net_arch=[16, 16])),
        dataset_identity='synthetic_only', baseline_commit='544908d')


@pytest.mark.parametrize('pause_at', [1, 2, 4])
def test_partial_trial_resume_matches_uninterrupted_train_only(tmp_path, pause_at):
    torch.set_num_threads(1)
    train, validation = make_smoke_episodes()
    full_study = optuna.create_study(direction='maximize')
    full = train_trial(full_study.ask(), config(), train, validation, tmp_path/'full')
    resumed_study = optuna.create_study(direction='maximize')
    trial = resumed_study.ask()
    paused = train_trial(trial, config(), train, validation, tmp_path/'resume',
                         stop_requested=lambda model:model._n_updates >= pause_at)
    assert paused['status'] == 'paused'
    resumed = train_trial(trial, config(), train, validation, tmp_path/'resume', resume=paused['checkpoint'])
    assert resumed['status'] == full['status'] == 'completed'
    assert resumed['adam_updates'] == full['adam_updates'] == 4
    assert resumed['test_payloads_opened'] == 0
    left = MaskedDQN.load(Path(full['checkpoint'])/'model.zip', device='cpu')
    right = MaskedDQN.load(Path(resumed['checkpoint'])/'model.zip', device='cpu')
    for key, value in left.policy.state_dict().items():
        torch.testing.assert_close(value, right.policy.state_dict()[key], rtol=0, atol=0)
    assert resumed['evaluation_history'] == full['evaluation_history']


def test_pruning_stops_after_a_saved_real_update(tmp_path):
    torch.set_num_threads(1)
    train, validation = make_smoke_episodes()
    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.ThresholdPruner(lower=100.))
    trial = study.ask()
    with pytest.raises(optuna.TrialPruned):
        train_trial(trial, config(), train, validation, tmp_path/'pruned')
    import json
    marker = json.loads((tmp_path/'pruned/latest_checkpoint.json').read_text(encoding='utf-8'))
    state = torch.load(marker['directory']+'/state.pt', map_location='cpu', weights_only=False)
    assert state['progress']['formal_steps'] == 2
    assert state['progress']['evaluation_history'][0]['evaluation_adam_updates'] == 0
    assert (Path(marker['directory'])/'replay.pkl').is_file()


@pytest.mark.parametrize('pause_at', [1, 4])
def test_warmup_pause_restores_exact_model_replay_and_validation(tmp_path, pause_at):
    train, validation = make_smoke_episodes()
    settings = dict(config(), learning_starts=4)
    full = train_trial(optuna.create_study(direction='maximize').ask(), settings,
                       train, validation, tmp_path/'full')
    study = optuna.create_study(direction='maximize')
    trial = study.ask()
    paused = train_trial(trial, settings, train, validation, tmp_path/'resume',
                         stop_requested=lambda model:model.num_timesteps >= pause_at)
    assert paused['status'] == 'paused'
    assert paused['adam_updates'] == 0
    assert paused['replay_insertions'] == pause_at
    resumed = train_trial(trial, settings, train, validation, tmp_path/'resume',
                          resume=paused['checkpoint'])
    assert resumed['evaluation_history'] == full['evaluation_history']
    assert resumed['adam_updates'] == full['adam_updates'] == 4
    assert resumed['replay_insertions'] == full['replay_insertions'] == 8
    left = MaskedDQN.load(Path(full['checkpoint'])/'model.zip', device='cpu')
    right = MaskedDQN.load(Path(resumed['checkpoint'])/'model.zip', device='cpu')
    for key, value in left.policy.state_dict().items():
        torch.testing.assert_close(value, right.policy.state_dict()[key], rtol=0, atol=0)
    left_replay = pickle.loads((Path(full['checkpoint'])/'replay.pkl').read_bytes())
    right_replay = pickle.loads((Path(resumed['checkpoint'])/'replay.pkl').read_bytes())
    for name in ('observations', 'next_observations', 'actions', 'rewards', 'dones',
                 'next_action_masks', 'outcomes', 'reasons', 'failure_penalties'):
        np.testing.assert_array_equal(getattr(left_replay, name), getattr(right_replay, name))


@pytest.mark.parametrize('interrupted_method', ['report', 'should_prune'])
def test_resume_replays_saved_validation_before_collecting_more(tmp_path, monkeypatch, interrupted_method):
    train, validation = make_smoke_episodes()
    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.ThresholdPruner(lower=100.))
    trial = study.ask()

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt('synthetic interruption after checkpoint')

    with monkeypatch.context() as context:
        context.setattr(trial, interrupted_method, interrupt)
        with pytest.raises(KeyboardInterrupt):
            train_trial(trial, config(), train, validation, tmp_path/'resume')
    marker = json.loads((tmp_path/'resume/latest_checkpoint.json').read_text(encoding='utf-8'))
    with pytest.raises(optuna.TrialPruned):
        train_trial(trial, config(), train, validation, tmp_path/'resume', resume=marker['directory'])
    result = json.loads((tmp_path/'resume/result.json').read_text(encoding='utf-8'))
    assert result['adam_updates'] == 2
    assert list(study.trials[0].intermediate_values) == [2]
    assert [row['formal_steps'] for row in result['evaluation_history']] == [2]


def test_pure_economic_reward_tiny_synthetic_trial(tmp_path, monkeypatch):
    train, validation = make_smoke_episodes()
    observed = []
    execute = tuning.execute_experiment

    def inspect(*args, **kwargs):
        values, status = execute(*args, **kwargs)
        observed.extend(values)
        return values, status

    monkeypatch.setattr(tuning, 'execute_experiment', inspect)
    settings = dict(config(), reward_mode='zhou_economic_only', train_steps=2)
    result = train_trial(optuna.create_study(direction='maximize').ask(), settings,
                         train, validation, tmp_path/'economic')
    assert result['status'] == 'completed'
    assert result['adam_updates'] == 2
    assert result['replay_insertions'] == 4
    assert result['test_payloads_opened'] == 0
    assert observed
    for transition in observed:
        assert transition.soc_soft_penalty_cny == transition.fc_ramp_penalty_cny == 0
        assert transition.excluded_degradation_cost_cny == 0
        assert transition.reward_cny == pytest.approx(
            transition.executed_ledger.reward_cny-transition.soc_equivalent_hydrogen_cost_cny)
