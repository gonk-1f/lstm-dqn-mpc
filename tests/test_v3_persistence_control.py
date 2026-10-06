import unittest

from v3.control import EconomicMPC, MPCWeights, PredictiveController
from v3.dqn import DoubleDQNAgent


class PersistenceControlTests(unittest.TestCase):
    def test_default_controller_holds_current_load_and_exposes_compact_state(self):
        controller = PredictiveController(mpc=EconomicMPC(nominal_cost_cny=1.0))
        controller.begin_voyage()
        controller.run_decision(lambda _: MPCWeights(0.2, 0.1), next_actual_load_kw=200.0)
        base_before = controller.observed_base_kw

        state, action, plan = controller.start_decision(lambda _: MPCWeights(0.2, 0.1))
        self.assertEqual(len(state), 10)
        self.assertEqual(state[1], 200.0 / 600.0)
        self.assertEqual(state[5:8], (200.0 / 600.0, 0.0, 0.0))
        self.assertEqual(state[8:10], (0.2, 0.1))
        self.assertEqual(plan.forecast_kw, (200.0,) * 5)
        self.assertEqual(plan.first_predicted_ledger.shore_cost_cny, 0.0)
        self.assertEqual(controller.observed_base_kw, base_before)

        transition = controller.finish_decision(220.0)
        self.assertEqual(transition.state, state)
        self.assertEqual(transition.action, action)
        self.assertEqual(transition.executed.prediction_error_kw, 20.0)
        self.assertAlmostEqual(
            transition.executed.actual_battery_power_kw - plan.battery_power_kw[0], 20.0
        )
        self.assertEqual(
            transition.executed.actual_ledger.h2_cost_cny,
            plan.first_predicted_ledger.h2_cost_cny,
        )
        self.assertEqual(
            transition.executed.actual_ledger.fuel_cell_degradation_cost_cny,
            plan.first_predicted_ledger.fuel_cell_degradation_cost_cny,
        )
        self.assertEqual(transition.reward_cny, -transition.executed.actual_ledger.total_cost_cny)
        self.assertEqual(transition.next_state[5:8], (20.0 / 600.0, 200.0 / 600.0, 0.0))
        self.assertEqual(transition.next_state[8:10], (0.2, 0.1))
        self.assertEqual(controller.plan(MPCWeights(0.2, 0.1)).forecast_kw, (220.0,) * 5)

    def test_double_dqn_accepts_persistence_state_and_realized_reward(self):
        actions = (MPCWeights(0.2, 0.1), MPCWeights(0.5, 0.2))
        agent = DoubleDQNAgent(actions, reward_scale_cny=1.0)
        self.assertEqual(agent.state_dim, 10)
        state = (0.0,) * 10
        action = agent.select_weights(state)
        agent.remember(state, action, -3.0, (1.0,) * 10, done=False)
        self.assertEqual(agent.replay[0].reward_cny, -3.0)

    def test_shore_closes_last_onboard_action_and_resets_only_control_history(self):
        controller = PredictiveController(mpc=EconomicMPC(nominal_cost_cny=1.0))
        controller.begin_voyage()
        state, action, plan = controller.start_decision(lambda _: MPCWeights(0.2, 0.1))
        transition = controller.finish_terminal_decision(100.0, (-50.0, -50.0))

        self.assertTrue(transition.done)
        self.assertEqual(transition.state, state)
        self.assertEqual(transition.action, action)
        self.assertIsNotNone(transition.shore_ledger)
        self.assertEqual(transition.shore_requested_battery_bus_kw, (-50.0, -50.0))
        self.assertEqual(transition.shore_accepted_battery_bus_kw, (-50.0, -50.0))
        self.assertAlmostEqual(
            transition.reward_cny,
            -(transition.executed.actual_ledger.total_cost_cny
              + transition.shore_ledger.total_cost_cny),
        )
        self.assertEqual(plan.first_predicted_ledger.shore_cost_cny, 0.0)
        self.assertEqual(controller.state.previous_fc_kw, 0.0)
        self.assertEqual(transition.next_state[0], controller.state.soc)
        self.assertEqual(controller.history, [])
        self.assertEqual(controller.errors_kw, [])
        self.assertIsNone(controller.observed_base_kw)
        self.assertIsNone(controller.previous_action)

        agent = DoubleDQNAgent((action,), reward_scale_cny=1.0)
        agent.remember_transition(transition)
        self.assertTrue(agent.replay[0].done)
        self.assertEqual(agent.replay[0].reward_cny, transition.reward_cny)

        restarted = controller.begin_voyage()
        self.assertEqual(restarted[0], transition.next_state[0])
        self.assertEqual(restarted[1:8], (0.0,) * 7)

        with self.assertRaises(ValueError):
            agent.remember_transition(transition, done=False)

    def test_invalid_shore_request_is_rejected_before_final_fc_execution(self):
        controller = PredictiveController(mpc=EconomicMPC(nominal_cost_cny=1.0))
        controller.begin_voyage()
        controller.start_decision(lambda _: MPCWeights(0.2, 0.1))
        with self.assertRaises(ValueError):
            controller.finish_terminal_decision(100.0, (10.0,))
        self.assertEqual(controller.state.previous_fc_kw, 0.0)
        self.assertIsNotNone(controller.pending_plan)

    def test_first_measured_onboard_load_uses_formal_zero_deadband(self):
        controller = PredictiveController(mpc=EconomicMPC(nominal_cost_cny=1.0))
        controller.begin_voyage()
        controller.start_decision(lambda _: MPCWeights(0.2, 0.1))
        transition = controller.finish_decision(-0.5)
        self.assertEqual(transition.executed.prediction_error_kw, 0.0)
        self.assertEqual(controller.history[-1], 0.0)
        with self.assertRaises(ValueError):
            controller.observe(-2.0)

    def test_each_persistence_voyage_requires_virtual_zero_boundary(self):
        controller = PredictiveController(mpc=EconomicMPC(nominal_cost_cny=1.0))
        with self.assertRaises(RuntimeError):
            controller.observe(100.0)
        controller.begin_voyage()
        controller.start_decision(lambda _: MPCWeights(0.2, 0.1))
        controller.finish_terminal_decision(100.0, (-10.0,))
        with self.assertRaises(RuntimeError):
            controller.observe(120.0)
        controller.begin_voyage()
        self.assertEqual(controller.history, [0.0])


if __name__ == "__main__":
    unittest.main()
