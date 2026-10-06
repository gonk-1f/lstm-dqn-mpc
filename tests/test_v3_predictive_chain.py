import math
import unittest

import numpy as np
import torch

from v3.forecasting import DirectLSTM, FittedForecaster, make_windows, windows_from_episodes
from v3.control import EconomicMPC, MPCWeights, PredictiveController
from v3.training import load_fitted_forecaster, validation_windows_for_candidates


class ForecastingContractTests(unittest.TestCase):
    def test_windows_are_strictly_future_and_do_not_cross_shore(self):
        loads = np.arange(12, dtype=float)
        modes = ("ONBOARD",) * 7 + ("SHORE_CHARGING",) + ("ONBOARD",) * 4
        x, y = make_windows(loads, modes, history_steps=3, horizon=2)
        np.testing.assert_array_equal(x[:, :, 0], [[0, 1, 2], [1, 2, 3], [2, 3, 4]])
        np.testing.assert_array_equal(y, [[3, 4], [4, 5], [5, 6]])

    def test_lstm_maps_history_to_five_future_powers(self):
        model = DirectLSTM(history_steps=12, horizon=5, hidden_size=8)
        prediction = model(torch.zeros(2, 12, 1))
        self.assertEqual(tuple(prediction.shape), (2, 5))
        with self.assertRaises(ValueError):
            model(torch.zeros(2, 11, 1))

    def test_zero_residual_head_is_persistence_forecast(self):
        model = DirectLSTM(history_steps=3, horizon=5, hidden_size=8)
        torch.nn.init.zeros_(model.head.weight)
        torch.nn.init.zeros_(model.head.bias)
        history = torch.tensor([[[1.0], [2.0], [3.0]]])
        self.assertEqual(model(history).tolist(), [[3.0] * 5])

    def test_negative_shore_load_does_not_invalidate_onboard_windows(self):
        loads = np.array([1, 2, 3, 4, 5, -20, 7, 8, 9], dtype=float)
        modes = ("ONBOARD",) * 5 + ("SHORE_CHARGING",) + ("ONBOARD",) * 3
        x, y = make_windows(loads, modes, history_steps=2, horizon=2)
        self.assertEqual(len(x), 2)
        np.testing.assert_array_equal(y, [[3, 4], [4, 5]])

    def test_windows_from_episodes_never_cross_episode_boundary(self):
        from types import SimpleNamespace

        episodes = (
            SimpleNamespace(load_kw=np.array([1, 2, 3, 4], dtype=float), operating_mode=("ONBOARD",) * 4),
            SimpleNamespace(load_kw=np.array([5, 6, 7, 8], dtype=float), operating_mode=("ONBOARD",) * 4),
        )
        x, y = windows_from_episodes(episodes, history_steps=2, horizon=2)
        np.testing.assert_array_equal(x[:, :, 0], [[1, 2], [5, 6]])
        np.testing.assert_array_equal(y, [[3, 4], [7, 8]])

    def test_formal_lowercase_onboard_mode_produces_windows(self):
        x, y = make_windows(
            np.array([1, 2, 3, 4], dtype=float), ("onboard",) * 4,
            history_steps=2, horizon=2,
        )
        self.assertEqual(x.shape, (1, 2, 1))
        np.testing.assert_array_equal(y, [[3, 4]])

    def test_candidate_windows_share_identical_validation_origins(self):
        from types import SimpleNamespace

        episode = SimpleNamespace(
            load_kw=np.arange(15, dtype=float),
            operating_mode=("onboard",) * 15,
        )
        windows = validation_windows_for_candidates((episode,), (3, 6), horizon=2)
        x3, y3 = windows[3]
        x6, y6 = windows[6]
        np.testing.assert_array_equal(y3, y6)
        np.testing.assert_array_equal(x3, x6[:, -3:, :])

    def test_saved_forecaster_reloads_with_identical_five_step_output(self):
        import tempfile
        from pathlib import Path

        model = DirectLSTM(history_steps=3, horizon=5, hidden_size=8)
        original = FittedForecaster(model, mean_kw=10.0, scale_kw=2.0)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "lstm.pt"
            torch.save({
                "model_state": model.state_dict(), "history_steps": 3,
                "horizon_steps": 5, "hidden_size": 8,
                "train_mean_kw": 10.0, "train_scale_kw": 2.0,
                "architecture": "residual_direct_lstm_v1",
            }, path)
            reloaded = load_fitted_forecaster(path)
        self.assertEqual(reloaded((8.0, 9.0, 10.0)), original((8.0, 9.0, 10.0)))


class PredictiveControlContractTests(unittest.TestCase):
    def test_dqn_sees_forecast_and_revealed_errors_before_choosing_weights(self):
        controller = PredictiveController(
            forecaster=lambda history: (history[-1] + 10.0,) * 5,
            history_steps=2,
            mpc=EconomicMPC(nominal_cost_cny=1.0),
        )
        controller.observe(100.0)
        controller.observe(110.0)
        seen = []

        def policy(state):
            seen.append(state)
            return MPCWeights(0.5, 0.1)

        transition = controller.run_decision(policy, next_actual_load_kw=130.0)
        self.assertEqual(len(seen), 1)
        self.assertEqual(len(seen[0]), 15)
        self.assertEqual(seen[0][5:10], (120.0 / 600.0,) * 5)
        self.assertEqual(seen[0][10:13], (0.0, 0.0, 0.0))
        self.assertEqual(transition.state, seen[0])
        self.assertEqual(transition.action, MPCWeights(0.5, 0.1))
        self.assertEqual(transition.reward_cny, -transition.executed.actual_ledger.total_cost_cny)
        self.assertAlmostEqual(transition.next_state[10], 10.0 / 600.0)
        self.assertEqual(transition.next_state[13:15], (0.5, 0.1))

    def test_decision_and_actual_reveal_are_separate_calls(self):
        controller = PredictiveController(
            forecaster=lambda history: (100.0,) * 5,
            history_steps=2, mpc=EconomicMPC(nominal_cost_cny=1.0),
        )
        controller.observe(100.0)
        controller.observe(100.0)
        state, action, plan = controller.start_decision(lambda _: MPCWeights(0.1, 0.1))
        self.assertEqual(len(state), 15)
        self.assertEqual(action, MPCWeights(0.1, 0.1))
        self.assertEqual(len(plan.fc_power_kw), 5)
        self.assertEqual(controller.errors_kw, [])
        transition = controller.finish_decision(110.0)
        self.assertEqual(transition.state, state)
        self.assertAlmostEqual(transition.next_state[10], 10.0 / 600.0)

    def test_first_step_fc_hydrogen_and_degradation_match_actual(self):
        controller = PredictiveController(
            forecaster=lambda history: (250.0,) * 5,
            history_steps=3,
            mpc=EconomicMPC(nominal_cost_cny=1.0),
        )
        for value in (200.0, 210.0, 220.0):
            controller.observe(value)
        before_filter = controller.observed_base_kw
        plan = controller.plan(MPCWeights(lambda_ref=1.0, lambda_soc=0.1))
        self.assertEqual(controller.observed_base_kw, before_filter)
        self.assertEqual(len(plan.fc_power_kw), 5)
        result = controller.execute_next(actual_load_kw=plan.forecast_kw[0] + 40.0)
        self.assertEqual(result.actual_fc_power_kw, plan.fc_power_kw[0])
        self.assertAlmostEqual(result.actual_battery_power_kw - plan.battery_power_kw[0], 40.0)
        self.assertEqual(result.actual_ledger.h2_cost_cny, plan.first_predicted_ledger.h2_cost_cny)
        self.assertEqual(
            result.actual_ledger.fuel_cell_degradation_cost_cny,
            plan.first_predicted_ledger.fuel_cell_degradation_cost_cny,
        )
        self.assertNotEqual(
            result.actual_ledger.battery_degradation_cost_cny,
            plan.first_predicted_ledger.battery_degradation_cost_cny,
        )
        self.assertEqual(result.reward_cny, -result.actual_ledger.total_cost_cny)
        self.assertTrue(math.isfinite(result.actual_soc))

    def test_second_plan_uses_new_observation_once(self):
        controller = PredictiveController(
            forecaster=lambda history: (history[-1],) * 5,
            history_steps=2,
            mpc=EconomicMPC(nominal_cost_cny=1.0),
        )
        controller.observe(100.0)
        controller.observe(150.0)
        plan = controller.plan(MPCWeights(1.0, 0.1))
        controller.execute_next(180.0)
        alpha = math.exp(-30.0 / 180.0)
        expected = alpha * (alpha * 100.0 + (1 - alpha) * 150.0) + (1 - alpha) * 180.0
        self.assertAlmostEqual(controller.observed_base_kw, expected)
        second = controller.plan(MPCWeights(1.0, 0.1))
        self.assertEqual(second.forecast_kw, (180.0,) * 5)
        self.assertAlmostEqual(controller.observed_base_kw, expected)


if __name__ == "__main__":
    unittest.main()
