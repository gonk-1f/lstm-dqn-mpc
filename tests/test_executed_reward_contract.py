from __future__ import annotations

import copy
from fractions import Fraction
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'src/main')]
from dqn.utils import action_mapper, reward
from dqn.agents.dqn_agent import DQNAgent, DQNTrainConfig
from dqn.memory.replay_buffer import ReplayBuffer
from envs.dqn_mpc_weight_env import DqnMpcWeightEnv, MpcSolveFailure
from mpc.solvers.fc_dp0_curve import h2_rate_gps_dp0
import train_dqn_mpc_mlp as training
import run_dqn_mpc_causal_training as entry


class ExecutedRewardContractTests(unittest.TestCase):
    def test_actions_are_exact_deterministic_integer_compositions(self):
        actions = action_mapper.DQN_MPC_WEIGHT_ACTIONS
        self.assertEqual(len(actions), 84)
        tuples = [a.as_tuple() for a in actions]
        self.assertEqual(tuples, sorted(set(tuples)))
        self.assertEqual([a.action_id for a in actions], list(range(84)))
        for values in tuples:
            self.assertTrue(all(.1 <= q <= .7 for q in values))
            self.assertEqual(sum(Fraction(str(q)) for q in values), 1)
        self.assertEqual(tuples[0], (.1, .1, .1, .7))
        self.assertEqual(tuples[-1], (.7, .1, .1, .1))
        agent = training.create_training_runtime(DQNTrainConfig(device='cpu')).agent
        self.assertEqual(agent.q_values(np.zeros(7)).shape, (84,))

    def score(self, **changes):
        values = dict(p_fc_kw=300., p_batt_kw=0., soc_after=.55, p_fc_prev_kw=300.)
        return reward.calculate_executed_reward(**(values | changes))

    def test_formula_and_actual_hydrogen_rate(self):
        r, info = self.score(p_fc_kw=600., p_fc_prev_kw=552., p_batt_kw=624., soc_after=.60)
        self.assertAlmostEqual(r, -2.5)
        self.assertAlmostEqual(info['hydrogen'], 1.)
        self.assertAlmostEqual(info['battery_power_stress_proxy'], 1.)
        r, _ = self.score()
        self.assertAlmostEqual(r, -float(h2_rate_gps_dp0(300.) / h2_rate_gps_dp0(600.)))

    def test_quadratic_symmetries(self):
        for left, right in [({'p_batt_kw': 200.}, {'p_batt_kw': -200.}),
                            ({'soc_after': .60}, {'soc_after': .50}),
                            ({'p_fc_prev_kw': 252.}, {'p_fc_prev_kw': 348.})]:
            self.assertAlmostEqual(self.score(**left)[0], self.score(**right)[0])

    def test_hydrogen_is_linear_in_rate(self):
        for rate in (2., 4.):
            with patch.object(reward, 'h2_rate_gps_dp0', side_effect=[np.array(rate), np.array(10.)]):
                self.assertAlmostEqual(self.score()[0], -rate / 10.)

    def test_invalid_execution_values_rejected(self):
        for key in ('p_fc_kw', 'p_batt_kw', 'soc_after', 'p_fc_prev_kw'):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.score(**{key: float('nan')})

    def test_future_plan_and_self_cost_cannot_change_reward(self):
        config = training.build_formal_mpc_config()
        env = DqnMpcWeightEnv(loads_kw=[300., 310.], base_config=config)
        result, ms = env.solver_bank.solve(action_id=0, load_forecast_kw=np.full(6,300.),
            current_soc=.55, prev_fc_kw=300., soc_reference=.55)
        changed = copy.copy(result)
        changed.x = result.x.copy()
        changed.x[1:6] += 99.
        changed.x[7:] += 9.
        changed.raw_mpc_objective += 999.
        results = []
        for plan in (result, changed):
            env.reset()
            with patch.object(env.solver_bank, 'solve', return_value=(plan, ms)):
                next_state, r, done, info = env.step(0)
            expected, _ = reward.calculate_executed_reward(p_fc_kw=info['p_fc_kw'],
                p_batt_kw=info['p_batt_kw'], soc_after=info['soc_after'], p_fc_prev_kw=info['p_fc_prev_kw'])
            self.assertAlmostEqual(r, expected)
            self.assertTrue(done)
            self.assertAlmostEqual(float(next_state[0]), (info['soc_after']-.55)/.05, places=6)
            results.append(r)
        self.assertEqual(results[0], results[1])

    def test_uncalibrated_formal_entry_stops_before_data_access(self):
        with patch.object(training, 'load_voyage_split') as load:
            with self.assertRaisesRegex(ValueError, 'terminal_failure_penalty.*calibrat'):
                entry.main([])
            load.assert_not_called()

    def test_numerical_failure_never_enters_replay(self):
        runtime = training.create_training_runtime(DQNTrainConfig(device='cpu', warmup_steps=100))
        error = MpcSolveFailure(action_id=0, decision_index=0, execution_index=1,
            solver_status='maximum iterations reached', solve_ms=1., current_soc=.55,
            previous_fc_kw=300., future_load_kw=[300.]*6,
            iterations=1, primal_residual=1., dual_residual=1.)
        with patch.object(DqnMpcWeightEnv, 'step', side_effect=error):
            with self.assertRaises(MpcSolveFailure):
                training.run_training_episode(voyage_id='synthetic', loads_kw=np.array([300.,300.]),
                    base_config=training.build_formal_mpc_config(), runtime=runtime)
        self.assertEqual(len(runtime.replay_buffer), 0)
        self.assertEqual(runtime.global_step, 0)

    def test_checkpoint_semantics_required_and_roundtrip(self):
        agent = training.create_training_runtime(DQNTrainConfig(device='cpu')).agent
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'model.pt'
            old = DQNAgent(7, 4, DQNTrainConfig(device='cpu'))
            torch.save(old.q_net.state_dict(), path)
            with self.assertRaisesRegex(ValueError, 'incompatible'):
                agent.load(path)
            agent.save(path)
            payload = torch.load(path, weights_only=False)
            self.assertEqual(len(payload['control_semantics']['actions']), 84)
            agent.load(path)
            payload['control_semantics']['actions'].reverse()
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, 'incompatible'):
                agent.load(path)

    def test_legacy_replay_rejected_even_with_low_action_ids(self):
        buffer = ReplayBuffer(5)
        buffer.push(np.zeros(7), 0, 1., False, np.ones(7))
        state = buffer.state_dict()
        state.pop('control_semantics', None)
        with self.assertRaisesRegex(ValueError, 'incompatible'):
            ReplayBuffer(5).load_state_dict(state)

    def test_inference_model_restores_calibrated_failure_policy(self):
        config = DQNTrainConfig(device='cpu', terminal_failure_penalty=123.,
                                failure_penalty_calibration='synthetic-test-only')
        agent = training.create_training_runtime(config).agent
        restored = training.create_training_runtime(DQNTrainConfig(device='cpu')).agent
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'model.pt'
            agent.save(path)
            restored.load(path)
        self.assertEqual(restored.config.terminal_failure_penalty, 123.)
        self.assertEqual(restored.config.failure_penalty_calibration, 'synthetic-test-only')

    def test_replay_rejects_different_failure_policy(self):
        one = ReplayBuffer(5, terminal_failure_penalty=123.,
                           failure_penalty_calibration='synthetic-test-only')
        two = ReplayBuffer(5, terminal_failure_penalty=456.,
                           failure_penalty_calibration='synthetic-test-only')
        with self.assertRaisesRegex(ValueError, 'incompatible.*failure'):
            two.load_state_dict(one.state_dict())


if __name__ == '__main__':
    unittest.main()
