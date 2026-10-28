"""Eight synthetic physical actions check the real SB3 collection/resume loop.

No canonical Train, Validation or Test payload is loaded. Across this check:
2 warmup + 1 formal + 1 uninterrupted + 1 reconstructed + 1 resumed + 2 greedy
actions, with only 3 small artificial Adam updates.
"""
from copy import deepcopy
import csv
import gzip
import json
import random
from types import SimpleNamespace

import numpy as np
import torch
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

from v6_sb3.curriculum import TrainCurriculum
from v6_sb3.masked_dqn import MaskedDQN
from v6_sb3.reporting import quiet_evaluate
from v6_sb3.training import TrainingContext, resume_context


def synthetic_episode(name, load, split='train'):
    return SimpleNamespace(sample_id=name, split=split, task_mode='independent_voyage',
        operating_mode=('onboard', 'onboard'), load_kw=(load, load),
        time_s=(0., 1.), dt_s=(1., 1.), duration_s=2.)


def parameter_snapshot(model):
    return {k: v.detach().clone() for k, v in model.policy.state_dict().items()}


def replay_snapshot(model):
    b = model.replay_buffer
    return {name: getattr(b, name)[:b.size()].copy() for name in (
        'observations', 'next_observations', 'actions', 'rewards', 'dones',
        'next_action_masks', 'outcomes', 'reasons', 'failure_penalties')}


def test_real_collection_checkpoint_resume_and_readonly_greedy_artifacts(tmp_path):
    torch.set_num_threads(1)
    train = (synthetic_episode('fixture_zero', 0.), synthetic_episode('fixture_high', 1290.))
    validation = (synthetic_episode('fixture_validation', 80., split='validation'),)
    curriculum = TrainCurriculum(train)
    restored_curriculum = None
    model = MaskedDQN(env=DummyVecEnv([lambda: curriculum]), buffer_size=32, learning_starts=2)
    model.set_logger(configure(None, []))
    output = tmp_path/'run'
    (output/'models').mkdir(parents=True)
    context = TrainingContext(output, curriculum, validation, model)
    try:
        # The real collector must store the terminal before any stop/checkpoint.
        model.learn(2, callback=context.callback, reset_num_timesteps=False,
                    log_interval=None, progress_bar=False)
        assert model.num_timesteps == 2 and model._n_updates == 0
        assert model.replay.insertions == 2 and model.replay.done[1, 0] == 1
        assert not model.replay.masks[1].any()
        assert context.report['warmup_attempts'][0]['inserted'] == 2
        assert not curriculum.pending_completed
        # Random(42)'s first shuffled voyage has only +50/+60 legal at reset;
        # automatic reset now exposes the other voyage's seven legal actions.
        assert curriculum.episode.sample_id == 'fixture_zero'
        assert np.flatnonzero(curriculum.action_masks()).tolist() == list(range(6, 13))
        np.testing.assert_array_equal(model._last_obs[0],
            np.asarray(curriculum.current_decision_state, dtype=np.float32))

        curriculum.start_formal()
        model._last_obs = model.env.reset()
        context.report.update(warmup_complete=True, status='training')
        model.learn(1, callback=context.callback, reset_num_timesteps=False,
                    log_interval=None, progress_bar=False)
        assert model.num_timesteps == 3 and model._n_updates == 1
        assert model.replay.insertions == 3
        assert len(curriculum.episode_actions) == 1
        assert not model.replay.done[2, 0]
        checkpoint = context.save_checkpoint('pause_fixture')
        saved_physical_state = tuple(curriculum.current_decision_state)
        saved_actions = list(curriculum.episode_actions)
        assert (checkpoint/'checkpoint.json').is_file()

        # Continue exactly one real step without loading, retaining its result.
        model.learn(1, callback=context.callback, reset_num_timesteps=False,
                    log_interval=None, progress_bar=False)
        expected_parameters = parameter_snapshot(model)
        expected_optimizer = deepcopy(model.optimizer.state_dict())
        expected_replay = replay_snapshot(model)
        expected_numpy = np.random.get_state()
        expected_torch = torch.get_rng_state().clone()
        expected_python = random.getstate()
        expected_decision = tuple(curriculum.current_decision_state)
        expected_mask = curriculum.action_masks()
        assert model._n_updates == 2 and model.replay.insertions == 4
        assert len(context.pass_rows) == 1  # terminal formal voyage reached context
        assert model.replay.done[3, 0] and not model.replay.masks[3].any()
        curriculum.close()

        # Reconstruct only the saved one-action prefix; it must not re-enter Replay.
        restored_curriculum = TrainCurriculum(train)
        restored = resume_context(output, checkpoint, restored_curriculum, validation, probes=[])
        resumed = restored.model
        resumed.set_logger(configure(None, []))
        assert resumed.device.type == 'cpu'
        assert resumed.num_timesteps == 3 and resumed._n_updates == 1
        assert resumed.replay.insertions == 3
        assert restored_curriculum.episode_actions == saved_actions
        assert restored_curriculum.current_decision_state == saved_physical_state
        assert not restored_curriculum.pending_completed

        resumed.learn(1, callback=restored.callback, reset_num_timesteps=False,
                      log_interval=None, progress_bar=False)
        assert resumed.num_timesteps == 4 and resumed._n_updates == 2
        assert resumed.replay.insertions == 4 and len(restored.pass_rows) == 1
        for name, expected in expected_parameters.items():
            assert torch.equal(resumed.policy.state_dict()[name], expected), name
        for name, expected in expected_replay.items():
            np.testing.assert_array_equal(replay_snapshot(resumed)[name], expected, err_msg=name)
        actual_optimizer = resumed.optimizer.state_dict()
        assert actual_optimizer['param_groups'] == expected_optimizer['param_groups']
        for key, expected in expected_optimizer['state'].items():
            for name, value in expected.items():
                assert torch.equal(actual_optimizer['state'][key][name], value), (key, name)
        actual_numpy = np.random.get_state()
        assert actual_numpy[0] == expected_numpy[0] and actual_numpy[2:] == expected_numpy[2:]
        np.testing.assert_array_equal(actual_numpy[1], expected_numpy[1])
        assert torch.equal(torch.get_rng_state(), expected_torch)
        assert random.getstate() == expected_python
        assert restored_curriculum.current_decision_state == expected_decision
        np.testing.assert_array_equal(restored_curriculum.action_masks(), expected_mask)

        # Two greedy actions produce the real reporting artifacts, without Adam/Replay.
        before_replay = replay_snapshot(resumed)
        before_updates = resumed._n_updates
        result, rows, failures = quiet_evaluate(validation, resumed,
            restored_curriculum.accountant, restored_curriculum.economics,
            restored_curriculum.penalty, output, 99)
        assert result['completed'] == 1 and not failures and len(rows) == 1
        assert resumed.replay.insertions == 4 and resumed._n_updates == before_updates
        for name, expected in before_replay.items():
            np.testing.assert_array_equal(replay_snapshot(resumed)[name], expected)
        folder = output/'round_99_validation'
        with gzip.open(folder/'fixture_validation.csv.gz', 'rt', encoding='utf-8') as handle:
            trace = list(csv.DictReader(handle))
        assert len(trace) == 2 and trace[-1]['done'] == 'True'
        assert all(f'q_delta_{delta}' in trace[0] for delta in range(-60, 61, 10))
        assert json.loads(trace[0]['legal_action_ranking_delta_kw'])
        assert (folder/'fixture_validation.png').stat().st_size > 1000
        assert (folder/'sample_metrics.csv').is_file()
    finally:
        curriculum.close()
        if restored_curriculum is not None:
            restored_curriculum.close()
