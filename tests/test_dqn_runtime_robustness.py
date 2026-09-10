from __future__ import annotations

import contextlib
import io
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / 'src', ROOT / 'src/main'):
    sys.path.insert(0, str(path))

import train_dqn_mpc_mlp as training
import test_dqn_mpc_causal as evaluation
from dqn.agents.dqn_agent import DQNTrainConfig
from dqn.policies.epsilon_greedy import EpsilonGreedyPolicy
from dqn.utils.state_builder import build_dqn_mpc_state
from envs.dqn_mpc_weight_env import DqnMpcWeightEnv, MpcSolveFailure


class RuntimeRobustnessTests(unittest.TestCase):
    def test_retired_objectives_cannot_enter_current_qp_path(self):
        from dataclasses import replace
        from mpc_solvers.mpc_qp_formulation import build_qp_problem
        for variant in ('simplified_normalized_literature_v1', 'legacy_raw_h2_soc_batt_ramp_terminal'):
            with self.subTest(variant=variant), self.assertRaisesRegex(ValueError, 'objective_variant'):
                build_qp_problem(replace(training.build_formal_mpc_config(), objective_variant=variant),
                    load_forecast_kw=[220.] * 6, current_soc=0.55, prev_fc_kw=220., soc_reference=0.55)

    def test_physical_constants_have_a_single_source_without_value_changes(self):
        import importlib.util
        self.assertIsNotNone(importlib.util.find_spec('utils.physical_config'))
        from utils import physical_config as physical
        from dqn.utils import reward, state_builder
        config = training.build_formal_mpc_config()
        self.assertEqual(config.fuel_cell_max_kw, physical.FUEL_CELL_MAX_KW)
        self.assertEqual(config.fuel_cell_max_kw, 600.)
        self.assertEqual(config.battery_capacity_kwh, 624.)
        self.assertEqual(config.fuel_cell_ramp_rate_kw_per_s, 48.)
        self.assertEqual((config.soc_min, config.soc_max), (0.2, 0.8))
        self.assertEqual((config.soc_soft_min, config.soc_soft_max), (0.5, 0.6))
        self.assertEqual(reward.FUEL_CELL_MAX_KW, state_builder.FUEL_CELL_POWER_SCALE_KW)

    def test_loader_keeps_only_owned_float64_loads_and_uses_three_typed_columns(self):
        import pandas as pd
        import weakref
        from types import SimpleNamespace
        from utils import formal_operating_dataset as dataset
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'sample.csv'
            pd.DataFrame(dict(timestamp=pd.date_range('2024-01-01', periods=3, freq='s'),
                              time_s=[0., 1., 2.], load_total_kw=[0., 123.456789012345, 600.],
                              unused=['long text'] * 3)).to_csv(path, index=False)
            split = SimpleNamespace(train_segments=('fixture',), validation_segments=(),
                test_segments=(), dataset_root=root,
                manifest=pd.DataFrame([dict(segment_id='fixture', split='train', one_second_csv='sample.csv')]))
            references = []
            read = pd.read_csv
            def recording_read(*args, **kwargs):
                self.assertEqual(kwargs.get('usecols'), ['timestamp', 'time_s', 'load_total_kw'])
                self.assertEqual(kwargs.get('dtype'), {'time_s': np.float64, 'load_total_kw': np.float64})
                frame = read(*args, **kwargs)
                references.append(weakref.ref(frame))
                return frame
            with patch.object(pd, 'read_csv', side_effect=recording_read):
                for _ in range(5):
                    loads = dataset.load_operating_segment_loads('train', 'fixture', split=split)
            self.assertTrue(loads.flags.owndata)
            self.assertEqual(loads.dtype, np.float64)
            self.assertEqual(loads.tolist(), [0., 123.456789012345, 600.])
            self.assertTrue(all(reference() is None for reference in references))

    def runtime(self, **kwargs):
        return training.create_training_runtime(DQNTrainConfig(
            device='cpu', buffer_size=16, batch_size=2,
            log_window_steps=0, **kwargs,
        ))

    def test_random_policy_is_lazy_and_rng_sequence_is_unchanged(self):
        for warmup in (True, False):
            policy = EpsilonGreedyPolicy(0.6, 0.05, 0.99)
            np.random.seed(81)
            expected = [policy.select_action(3, 4, warmup) for _ in range(100)]
            expected_next_random = np.random.rand()
            calls = []
            np.random.seed(81)
            actual = [policy.select_action(lambda: calls.append(1) or 3, 4, warmup)
                      for _ in range(100)]
            self.assertEqual(actual, expected)
            self.assertEqual(np.random.rand(), expected_next_random)
            if warmup:
                self.assertEqual(calls, [])
            else:
                self.assertLess(len(calls), 100)

    def test_training_forward_counts_warmup_random_and_greedy(self):
        for warmup_steps, epsilon, expected_online, expected_target in (
            (100, 1.0, 0, 0), (0, 1.0, 1, 1), (0, 0.0, 3, 1),
        ):
            with self.subTest(warmup=warmup_steps, epsilon=epsilon):
                np.random.seed(123)
                runtime = self.runtime(warmup_steps=warmup_steps, epsilon_start=epsilon)
                online, target = [], []
                h1 = runtime.agent.q_net.register_forward_hook(lambda *args: online.append(1))
                h2 = runtime.agent.target_q_net.register_forward_hook(lambda *args: target.append(1))
                try:
                    training.run_training_episode(voyage_id='synthetic',
                        loads_kw=np.array([220., 221., 222.]),
                        base_config=training.build_formal_mpc_config(), runtime=runtime)
                finally:
                    h1.remove()
                    h2.remove()
                self.assertEqual(len(online), expected_online)
                self.assertEqual(len(target), expected_target)

    def test_validation_uses_one_q_forward_per_step(self):
        runtime = self.runtime(warmup_steps=100)
        with patch.object(runtime.agent.q_net, 'forward', wraps=runtime.agent.q_net.forward) as forward:
            result, trace = evaluation.run_test_episode(voyage_id='synthetic',
                loads_kw=np.array([220., 221., 222.]),
                base_config=training.build_formal_mpc_config(), agent=runtime.agent)
        self.assertTrue(result['completed'])
        self.assertEqual(forward.call_count, 2)
        self.assertEqual(trace.action_id.tolist(), trace[[f'q_A{i}' for i in range(4)]].to_numpy().argmax(1).tolist())

    def test_execution_checker_catches_all_existing_hard_constraints(self):
        import envs.dqn_mpc_weight_env as module
        self.assertTrue(hasattr(module, 'validate_executed_step'))
        config = training.build_formal_mpc_config()
        values = dict(p_fc_kw=300., p_batt_kw=0., next_soc=0.55,
                      load_kw=300., previous_fc_kw=300., config=config)
        module.validate_executed_step(**values)
        for change in (dict(p_fc_kw=601.), dict(p_batt_kw=1250.),
                       dict(p_batt_kw=-625.), dict(next_soc=0.19),
                       dict(next_soc=0.81), dict(load_kw=310.),
                       dict(previous_fc_kw=200.), dict(p_fc_kw=float('nan'))):
            with self.subTest(change=change), self.assertRaises(ValueError):
                module.validate_executed_step(**(values | change))

    def test_training_and_both_validation_paths_reject_same_bad_execution(self):
        import envs.dqn_mpc_weight_env as module
        original_step = DqnMpcWeightEnv.step

        def force_boundary(env, action):
            env.current_soc = 0.200001
            return original_step(env, action)

        # The forecast is 0 kW, actual next load 1000 kW: realized SOC violates
        # the existing hard lower bound despite a solved forecast QP.
        loads = np.array([0., 1000.])
        config = training.build_formal_mpc_config()
        with patch.object(DqnMpcWeightEnv, 'step', force_boundary):
            runtime = self.runtime(warmup_steps=100)
            train = training.run_training_episode(voyage_id='synthetic', loads_kw=loads,
                base_config=config, runtime=runtime)
            val = training.run_validation_episode(voyage_id='synthetic', loads_kw=loads,
                base_config=config, agent=runtime.agent)
            formal, trace = evaluation.run_test_episode(voyage_id='synthetic', loads_kw=loads,
                base_config=config, agent=runtime.agent)
        self.assertEqual(train['solver_failure_count'], 1)
        self.assertFalse(val['completed'])
        self.assertFalse(formal['completed'])
        self.assertEqual(train['episode_reward'], -620.)
        self.assertEqual(val['episode_reward'], -620.)
        self.assertEqual(formal['episode_reward'], -620.)
        self.assertEqual(len(trace), 0)
        self.assertTrue(runtime.replay_buffer.dones[-1])

    def test_state_hot_path_is_bounded_and_matches_full_history(self):
        import envs.dqn_mpc_weight_env as module
        loads = np.random.default_rng(42).uniform(0., 1000., 300)
        env = DqnMpcWeightEnv(loads_kw=loads, base_config=training.build_formal_mpc_config())
        lengths = []
        def recording_builder(**kwargs):
            lengths.append(len(kwargs['load_history_kw']))
            return build_dqn_mpc_state(**kwargs)
        with patch.object(module, 'build_dqn_mpc_state', side_effect=recording_builder):
            for index in range(len(loads)):
                env.decision_index = index
                expected = build_dqn_mpc_state(current_soc=env.current_soc,
                    previous_fc_kw=env.previous_fc_kw, previous_batt_kw=env.previous_batt_kw,
                    load_history_kw=loads[:index + 1])
                np.testing.assert_array_equal(env._build_state(), expected)
        self.assertLessEqual(max(lengths), 60)

    def test_segment_resume_matches_uninterrupted_updates_and_skips_completed(self):
        self.assertIn('on_segment_complete', __import__('inspect').signature(
            training.train_complete_voyage_rounds).parameters)
        config = DQNTrainConfig(device='cpu', buffer_size=5, batch_size=2,
            warmup_steps=1, target_sync_interval=3, log_window_steps=0)
        base = training.build_formal_mpc_config()
        ids = ('fixture_a', 'fixture_b', 'fixture_c')
        loads = np.array([220., 221., 222.])
        training.seed_training_rngs(config)
        full = training.create_training_runtime(config)
        training.train_complete_voyage_rounds(num_training_rounds=2, voyage_ids=ids,
            load_voyage=lambda _: loads, base_config=base, runtime=full)
        expected_random = (random.random(), np.random.rand(), torch.rand(1))
        training.seed_training_rngs(config)
        partial = training.create_training_runtime(config)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'training_state_latest.pt'
            def checkpoint(progress):
                training.save_training_state(runtime=partial, path=path,
                    completed_round=progress['completed_round'])
                if progress['completed_segment_count'] == 1:
                    raise InterruptedError('synthetic interruption')
            with self.assertRaises(InterruptedError):
                training.train_complete_voyage_rounds(num_training_rounds=2, voyage_ids=ids,
                    load_voyage=lambda _: loads, base_config=base, runtime=partial,
                    on_segment_complete=checkpoint)
            restored, completed, _ = training.load_training_state(path)
            calls = []
            training.train_complete_voyage_rounds(num_training_rounds=2, voyage_ids=ids,
                load_voyage=lambda identifier: calls.append(identifier) or loads,
                base_config=base, runtime=restored, first_round_id=training.next_round_id(completed))
        self.assertEqual(calls, ['fixture_b', 'fixture_c', *ids])
        self.assertEqual(full.global_step, restored.global_step)
        self.assertEqual(full.policy.epsilon, restored.policy.epsilon)
        self.assertEqual(full.gradient_update_count, restored.gradient_update_count)
        self.assertEqual(full.replay_buffer.write_position, restored.replay_buffer.write_position)
        for first, second in zip(full.agent.q_net.parameters(), restored.agent.q_net.parameters()):
            self.assertTrue(torch.equal(first, second))
        for first, second in zip(full.agent.target_q_net.parameters(), restored.agent.target_q_net.parameters()):
            self.assertTrue(torch.equal(first, second))
        self.assertEqual(full.target_sync_count, restored.target_sync_count)
        self.assertEqual(full.loss_accumulator.summary(), restored.loss_accumulator.summary())
        for key, value in full.replay_buffer.state_dict().items():
            np.testing.assert_array_equal(value, restored.replay_buffer.state_dict()[key])
        self.assertEqual(random.random(), expected_random[0])
        self.assertEqual(np.random.rand(), expected_random[1])
        self.assertTrue(torch.equal(torch.rand(1), expected_random[2]))

    def test_formal_orchestrator_saves_latest_after_every_segment(self):
        import run_dqn_mpc_causal_training as formal
        from types import SimpleNamespace
        runtime = self.runtime(warmup_steps=100)
        split = SimpleNamespace(train_segments=('fixture_a', 'fixture_b'), validation_segments=('fixture_val',))
        seen = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def load_train(identifier):
                if seen:
                    payload = torch.load(root / 'training_state_latest.pt', weights_only=False)
                    self.assertEqual(payload['round_progress']['completed_segment_ids'], list(split.train_segments[:1]))
                    self.assertEqual(payload['round_progress']['current_round'], 1)
                    self.assertEqual(payload['completed_round'], 0)
                seen.append(identifier)
                return np.array([220., 221.])
            # No formal validation/data access: return a small synthetic artifact.
            import pandas as pd
            with patch.object(formal.validation_artifacts, 'run_test_episode',
                    return_value=({'completed': True}, pd.DataFrame({'execution_index': [1]}))), \
                 patch.object(formal.validation_artifacts, 'plot_power_allocation'), \
                 patch.object(formal.validation_artifacts, 'plot_soc_trajectory'):
                formal.run_round_boundary_training(split=split, runtime=runtime,
                    base_config=training.build_formal_mpc_config(), output_dir=root,
                    load_train=load_train, load_validation=lambda _: np.array([220., 221.]),
                    num_training_rounds=1)
            latest = torch.load(root / 'training_state_latest.pt', weights_only=False)
            self.assertEqual(latest['round_progress']['completed_segment_count'], 2)
            self.assertEqual(latest['completed_round'], 1)
            self.assertTrue((root / 'round_1/model_round1.pt').is_file())
            self.assertTrue((root / 'round_1/training_state_round1.pt').is_file())
            self.assertFalse(list(root.rglob('*.tmp')))

    def test_resume_rejects_changed_segment_order_before_loading_any_data(self):
        runtime = self.runtime(warmup_steps=100)
        runtime.round_progress = {'train_segment_order': ['fixture_a']}
        calls = []
        with self.assertRaisesRegex(ValueError, 'order'):
            training.train_complete_voyage_rounds(num_training_rounds=2, voyage_ids=('fixture_b',),
                load_voyage=lambda identifier: calls.append(identifier),
                base_config=training.build_formal_mpc_config(), runtime=runtime)
        self.assertEqual(calls, [])

    def test_round_finalization_resume_never_retrains_completed_segments(self):
        import run_dqn_mpc_causal_training as formal
        import pandas as pd
        from types import SimpleNamespace
        for failure_at in ('model_save', 'validation'):
            with self.subTest(failure_at=failure_at), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                runtime = self.runtime(warmup_steps=100)
                split = SimpleNamespace(train_segments=('fixture_train',), validation_segments=('fixture_val',))
                loads = np.array([220., 221.])
                artifact = ({'completed': True}, pd.DataFrame({'execution_index': [1]}))
                with contextlib.ExitStack() as stack:
                    stack.enter_context(patch.object(formal.validation_artifacts, 'plot_power_allocation'))
                    stack.enter_context(patch.object(formal.validation_artifacts, 'plot_soc_trajectory'))
                    validate = stack.enter_context(patch.object(formal.validation_artifacts, 'run_test_episode', return_value=artifact))
                    if failure_at == 'model_save':
                        failure = patch.object(runtime.agent, 'save', side_effect=InterruptedError('fixture'))
                    else:
                        failure = patch.object(formal.validation_artifacts, 'run_test_episode', side_effect=InterruptedError('fixture'))
                    with failure, self.assertRaises(InterruptedError):
                        formal.run_round_boundary_training(split=split, runtime=runtime,
                            base_config=training.build_formal_mpc_config(), output_dir=root,
                            load_train=lambda _: loads, load_validation=lambda _: loads)
                    restored, completed, _ = training.load_training_state(root / 'training_state_latest.pt')
                    self.assertTrue(restored.round_progress.get('round_finalization_pending'))
                    self.assertEqual(formal.resume_round_id(restored, completed), 1)
                    train_calls = []
                    formal.run_round_boundary_training(split=split, runtime=restored,
                        base_config=training.build_formal_mpc_config(), output_dir=root,
                        load_train=lambda identifier: train_calls.append(identifier) or loads,
                        load_validation=lambda _: loads,
                        first_round_id=formal.resume_round_id(restored, completed))
                    self.assertEqual(train_calls, ['fixture_train'])  # Round 2 only.
                    self.assertEqual(restored.global_step, 2)
                    self.assertEqual(validate.call_count, 2)
                    for round_id in (1, 2):
                        self.assertTrue((root / f'round_{round_id}/model_round{round_id}.pt').is_file())
                        self.assertTrue((root / f'round_{round_id}/validation_summary.json').is_file())
                    final, completed, _ = training.load_training_state(root / 'training_state_latest.pt')
                    self.assertFalse(final.round_progress['round_finalization_pending'])
                    self.assertEqual(formal.resume_round_id(final, completed), 3)

    def test_progress_log_includes_round_segment_and_throughput(self):
        runtime = self.runtime(warmup_steps=100)
        runtime.config.log_window_steps = 1
        runtime.round_progress = {'current_round': 2, 'completed_segment_count': 73}
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            training.run_training_episode(voyage_id='fixture', loads_kw=np.array([220., 221.]),
                base_config=training.build_formal_mpc_config(), runtime=runtime)
        for field in ('step=1', 'round=2', 'segment=74', 'epsilon=', 'loss_mean=', 'reward_mean=', 'steps/s='):
            self.assertIn(field, output.getvalue())

    def test_loss_statistics_memory_is_bounded_and_mean_is_all_updates(self):
        runtime = self.runtime(warmup_steps=100)
        self.assertTrue(hasattr(runtime, 'record_loss'))
        for value in range(4001):
            runtime.record_loss(torch.tensor(float(value)))
        runtime.flush_loss_metrics()
        self.assertLessEqual(len(runtime.losses), 1000)
        self.assertEqual(len(runtime.pending_losses), 0)
        self.assertEqual(runtime.loss_accumulator.summary()['count'], 4001)
        self.assertEqual(runtime.loss_accumulator.summary()['mean'], 2000.)
        self.assertEqual(runtime.loss_accumulator.summary()['median_scope'], 'recent_1000')

    def test_update_safety_without_extra_forward_and_same_bellman_update(self):
        runtime = self.runtime(warmup_steps=0)
        reference = self.runtime(warmup_steps=0)
        reference.agent.q_net.load_state_dict(runtime.agent.q_net.state_dict())
        reference.agent.target_q_net.load_state_dict(runtime.agent.target_q_net.state_dict())
        batch = (np.ones((2, 7), np.float32), np.array([0, 1]),
                 np.array([-1., -2.], np.float32), np.array([0., 1.], np.float32),
                 np.full((2, 7), 0.5, np.float32))
        states, actions, rewards, dones, next_states = batch
        q = reference.agent.q_net(torch.tensor(states)).gather(1, torch.tensor(actions)[:, None]).squeeze(1)
        target = torch.tensor(rewards) + 0.9995 * reference.agent.target_q_net(
            torch.tensor(next_states)).detach().max(1).values * (1 - torch.tensor(dones))
        loss = torch.nn.functional.smooth_l1_loss(
            q,
            target,
            beta=1.0,
        )
        reference.agent.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(reference.agent.q_net.parameters(), 10.)
        reference.agent.optimizer.step()
        self.assertIn('defer_diagnostics', __import__('inspect').signature(runtime.agent.update).parameters)
        actual_loss = runtime.agent.update(batch, defer_diagnostics=True)
        self.assertIsInstance(actual_loss, torch.Tensor)
        self.assertFalse(actual_loss.requires_grad)
        self.assertEqual(float(actual_loss), float(loss.detach()))
        for left, right in zip(runtime.agent.q_net.parameters(), reference.agent.q_net.parameters()):
            self.assertTrue(torch.equal(left, right))
        bad_batch = (states, actions, np.array([np.nan, 0.], np.float32), dones, next_states)
        with self.assertRaisesRegex(RuntimeError, 'NaN|Inf'):
            runtime.agent.update(bad_batch, defer_diagnostics=True)


if __name__ == '__main__':
    unittest.main()
