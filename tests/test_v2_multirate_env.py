from __future__ import annotations

from dataclasses import FrozenInstanceError
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class _StateProvider:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> tuple[float, ...]:
        self.calls += 1
        return (float(self.calls), 0.0)


class _Backend:
    def __init__(self, ledgers, *, done_at: int | None = None, fail_at: int | None = None):
        self.ledgers = tuple(ledgers)
        self.done_at = done_at
        self.fail_at = fail_at
        self.calls = 0
        self.weights = []

    def execute_mpc_step(self, weights):
        from v2.envs.multirate_weight_env import MPCExecutionResult

        self.calls += 1
        self.weights.append(weights)
        if self.calls == self.fail_at:
            raise RuntimeError("solver failed")
        return MPCExecutionResult(
            ledger=self.ledgers[self.calls - 1],
            done=self.calls == self.done_at,
        )


class MultiRateWeightEnvironmentTests(unittest.TestCase):
    @staticmethod
    def _candidate():
        from v2.dqn.action_space import ActionCandidate

        return ActionCandidate(2, 3, 5)

    @staticmethod
    def _ledgers(count: int):
        from v2.economics import RawCnyIntervalLedger

        return tuple(
            RawCnyIntervalLedger(float(i), 2.0, 3.0, 4.0)
            for i in range(1, count + 1)
        )

    def _environment(
        self,
        backend,
        provider,
        *,
        n: int = 5,
        m: int = 5,
        replay_sink=None,
        failure_policy=None,
    ):
        from v2.config import TimeScaleConfig
        from v2.envs.multirate_weight_env import MultiRateWeightEnvironment

        keywords = {}
        if failure_policy is not None:
            keywords["failure_policy"] = failure_policy
        return MultiRateWeightEnvironment(
            timescale=TimeScaleConfig(30.0, n, m),
            action_catalog=(self._candidate(),),
            backend=backend,
            state_provider=provider,
            synthetic_test_mode=True,
            replay_sink=replay_sink,
            **keywords,
        )

    def test_one_action_is_held_for_five_executions_and_one_boundary_transition(self) -> None:
        backend = _Backend(self._ledgers(5))
        provider = _StateProvider()
        environment = self._environment(backend, provider)

        self.assertEqual(environment.reset(), (1.0, 0.0))
        transition = environment.step("w_2_3_5")

        self.assertEqual(backend.calls, 5)
        self.assertEqual(provider.calls, 2)
        self.assertEqual(len({id(value) for value in backend.weights}), 5)
        self.assertTrue(
            all(
                (value.q_base, value.q_smooth, value.q_soc) == (0.2, 0.3, 0.5)
                for value in backend.weights
            )
        )
        self.assertEqual(transition.state, (1.0, 0.0))
        self.assertEqual(transition.next_state, (2.0, 0.0))
        self.assertEqual(transition.action_id, "w_2_3_5")
        self.assertEqual(transition.executed_mpc_steps, 5)
        self.assertEqual(transition.ledger.components_cny, (15.0, 10.0, 15.0, 20.0))
        self.assertEqual(transition.learning_reward, -60.0)
        self.assertEqual(transition.raw_economic_cost_cny, 60.0)
        self.assertEqual(transition.failure_penalty_score, 0.0)
        self.assertFalse(transition.failed)
        self.assertFalse(transition.done)
        self.assertEqual(environment.transitions, (transition,))
        with self.assertRaises(FrozenInstanceError):
            transition.learning_reward = 0.0  # type: ignore[misc]

    def test_backend_cannot_mutate_later_weights_or_prior_ledger_snapshots(self) -> None:
        from v2.economics import RawCnyIntervalLedger
        from v2.envs.multirate_weight_env import MPCExecutionResult

        source_ledgers = (
            RawCnyIntervalLedger(1.0, 2.0, 3.0, 4.0),
            RawCnyIntervalLedger(5.0, 6.0, 7.0, 8.0),
            RawCnyIntervalLedger(9.0, 10.0, 11.0, 12.0),
        )

        class MutatingBackend:
            def __init__(self):
                self.calls = 0
                self.received_values = []
                self.weights = []

            def execute_mpc_step(self, weights):
                if self.calls:
                    object.__setattr__(
                        source_ledgers[self.calls - 1],
                        "h2_cost_cny",
                        999.0,
                    )
                self.received_values.append(
                    (weights.q_base, weights.q_smooth, weights.q_soc)
                )
                self.weights.append(weights)
                object.__setattr__(weights, "q_base", 999.0)
                ledger = source_ledgers[self.calls]
                self.calls += 1
                return MPCExecutionResult(ledger, False)

        backend = MutatingBackend()
        environment = self._environment(
            backend,
            _StateProvider(),
            n=2,
            m=3,
        )
        environment.reset()

        transition = environment.step("w_2_3_5")

        self.assertEqual(
            backend.received_values,
            [(0.2, 0.3, 0.5)] * 3,
        )
        self.assertEqual(len({id(value) for value in backend.weights}), 3)
        self.assertEqual(
            transition.ledger.components_cny,
            (15.0, 18.0, 21.0, 24.0),
        )
        self.assertEqual(transition.action_id, "w_2_3_5")

    def test_loop_count_uses_m_not_prediction_horizon_n(self) -> None:
        backend = _Backend(self._ledgers(3))
        provider = _StateProvider()
        environment = self._environment(backend, provider, n=2, m=3)
        environment.reset()

        transition = environment.step("w_2_3_5")

        self.assertEqual(backend.calls, 3)
        self.assertEqual(transition.executed_mpc_steps, 3)

    def test_shore_intervals_are_drained_without_counting_as_mpc_steps(self) -> None:
        from v2.envs.multirate_weight_env import MPCExecutionResult

        class EventBackend:
            def __init__(self, ledgers):
                self.results = (
                    MPCExecutionResult(ledgers[0], False, True, True),
                    MPCExecutionResult(ledgers[1], False, True, True),
                    MPCExecutionResult(ledgers[2], False, True, False),
                    MPCExecutionResult(ledgers[3], False, False, False),
                    MPCExecutionResult(ledgers[4], False, False, True),
                )
                self.calls = 0

            def execute_mpc_step(self, weights):
                result = self.results[self.calls]
                self.calls += 1
                return result

        backend = EventBackend(self._ledgers(5))
        provider = _StateProvider()
        environment = self._environment(backend, provider)
        environment.reset()

        transition = environment.step("w_2_3_5")

        self.assertEqual(backend.calls, 5)
        self.assertEqual(transition.executed_mpc_steps, 3)
        self.assertEqual(transition.ledger.components_cny, (15.0, 10.0, 15.0, 20.0))
        self.assertEqual(provider.calls, 2)
        self.assertFalse(transition.done)

    def test_early_done_still_builds_one_boundary_state_and_one_transition(self) -> None:
        backend = _Backend(self._ledgers(5), done_at=2)
        provider = _StateProvider()
        environment = self._environment(backend, provider)
        environment.reset()

        transition = environment.step("w_2_3_5")

        self.assertEqual(backend.calls, 2)
        self.assertEqual(provider.calls, 2)
        self.assertEqual(transition.executed_mpc_steps, 2)
        self.assertTrue(transition.done)
        self.assertEqual(len(environment.transitions), 1)
        with self.assertRaisesRegex(RuntimeError, "done"):
            environment.step("w_2_3_5")

    def test_nonphysical_failure_reports_partial_count_and_emits_no_transition(self) -> None:
        from v2.envs.multirate_weight_env import MacroStepExecutionError

        backend = _Backend(self._ledgers(5), fail_at=3)
        provider = _StateProvider()
        replayed = []
        environment = self._environment(
            backend,
            provider,
            replay_sink=replayed.append,
        )
        environment.reset()

        with self.assertRaises(MacroStepExecutionError) as caught:
            environment.step("w_2_3_5")

        self.assertEqual(caught.exception.executed_mpc_steps, 2)
        self.assertEqual(provider.calls, 1)
        self.assertEqual(environment.transitions, ())
        self.assertEqual(replayed, [])

    def test_physical_infeasibility_emits_terminal_transition_with_partial_ledger(self) -> None:
        from v2.control.nonlinear_mpc import PhysicalInfeasibilityError
        from v2.envs.multirate_weight_env import MPCExecutionResult
        from v2.failure_policy import FORMAL_FAILURE_POLICY

        class PhysicalFailureBackend:
            def __init__(self, ledgers):
                self.ledgers = ledgers
                self.calls = 0

            def execute_mpc_step(self, weights):
                self.calls += 1
                if self.calls == 3:
                    raise PhysicalInfeasibilityError("no reachable SOC")
                return MPCExecutionResult(self.ledgers[self.calls - 1], False)

        backend = PhysicalFailureBackend(self._ledgers(5))
        provider = _StateProvider()
        replayed = []
        environment = self._environment(
            backend,
            provider,
            replay_sink=replayed.append,
            failure_policy=FORMAL_FAILURE_POLICY,
        )
        environment.reset()

        transition = environment.step("w_2_3_5")

        self.assertTrue(transition.done)
        self.assertTrue(transition.failed)
        self.assertFalse(transition.episode_completed)
        self.assertEqual(transition.executed_mpc_steps, 2)
        self.assertEqual(transition.ledger.components_cny, (3.0, 4.0, 6.0, 8.0))
        self.assertEqual(transition.raw_economic_cost_cny, 21.0)
        self.assertEqual(transition.failure_penalty_score, 50_000.0)
        self.assertEqual(transition.learning_reward, -50_021.0)
        self.assertEqual(transition.failure_kind, "physical_mpc_infeasibility")
        self.assertEqual(provider.calls, 2)
        self.assertEqual(environment.transitions, (transition,))
        self.assertEqual(replayed, [transition])
        with self.assertRaisesRegex(RuntimeError, "done"):
            environment.step("w_2_3_5")

    def test_physical_infeasibility_before_execution_emits_zero_cost_failure(self) -> None:
        from v2.control.nonlinear_mpc import PhysicalInfeasibilityError
        from v2.failure_policy import FORMAL_FAILURE_POLICY

        class ImmediatePhysicalFailureBackend:
            def execute_mpc_step(self, weights):
                raise PhysicalInfeasibilityError("initial horizon infeasible")

        provider = _StateProvider()
        environment = self._environment(
            ImmediatePhysicalFailureBackend(),
            provider,
            failure_policy=FORMAL_FAILURE_POLICY,
        )
        environment.reset()

        transition = environment.step("w_2_3_5")

        self.assertTrue(transition.failed)
        self.assertTrue(transition.done)
        self.assertEqual(transition.executed_mpc_steps, 0)
        self.assertEqual(transition.raw_economic_cost_cny, 0.0)
        self.assertEqual(transition.failure_penalty_score, 50_000.0)
        self.assertEqual(transition.learning_reward, -50_000.0)
        self.assertEqual(provider.calls, 2)

    def test_replay_observer_receives_a_detached_canonical_snapshot(self) -> None:
        observed = []

        def mutate_snapshot(transition):
            observed.append(transition)
            object.__setattr__(transition, "state", (999.0,))
            object.__setattr__(transition.action, "n_base", 9)
            object.__setattr__(transition.ledger, "h2_cost_cny", 999.0)

        backend = _Backend(self._ledgers(2))
        environment = self._environment(
            backend,
            _StateProvider(),
            n=2,
            m=1,
            replay_sink=mutate_snapshot,
        )
        environment.reset()

        committed = environment.step("w_2_3_5")

        self.assertIs(committed, environment.transitions[0])
        self.assertIsNot(observed[0], committed)
        self.assertIsNot(observed[0].state, committed.state)
        self.assertIsNot(observed[0].next_state, committed.next_state)
        self.assertIsNot(observed[0].action, committed.action)
        self.assertIsNot(observed[0].ledger, committed.ledger)
        self.assertEqual(committed.state, (1.0, 0.0))
        self.assertEqual(committed.action.numerators, (2, 3, 5))
        self.assertEqual(committed.ledger.components_cny, (1.0, 2.0, 3.0, 4.0))
        second = environment.step("w_2_3_5")
        self.assertEqual(second.action.numerators, (2, 3, 5))

    def test_replay_observer_failure_reports_already_committed_transition(self) -> None:
        from v2.envs.multirate_weight_env import ReplaySinkNotificationError

        observed = []

        def append_then_raise_once(transition):
            observed.append(transition)
            if len(observed) == 1:
                raise RuntimeError("observer storage failed after append")

        backend = _Backend(self._ledgers(2))
        provider = _StateProvider()
        environment = self._environment(
            backend,
            provider,
            n=2,
            m=1,
            replay_sink=append_then_raise_once,
        )
        environment.reset()

        with self.assertRaises(ReplaySinkNotificationError) as caught:
            environment.step("w_2_3_5")

        self.assertTrue(caught.exception.transition_committed)
        self.assertIs(caught.exception.transition, environment.transitions[0])
        self.assertIsNot(observed[0], environment.transitions[0])
        self.assertEqual(environment.current_state, (2.0, 0.0))
        self.assertEqual(backend.calls, 1)

        second = environment.step("w_2_3_5")
        self.assertIs(second, environment.transitions[1])
        self.assertEqual(backend.calls, 2)

    def test_strict_boundaries_and_integrated_preflight_status(self) -> None:
        import numpy as np

        from v2.envs.multirate_weight_env import (
            MPCExecutionResult,
            TRAINING_READINESS_STATUS,
        )

        self.assertEqual(TRAINING_READINESS_STATUS, "READY_FOR_INTEGRATED_PREFLIGHT")
        with self.assertRaises(TypeError):
            MPCExecutionResult(self._ledgers(1)[0], 0)  # type: ignore[arg-type]

        provider = lambda: np.asarray([1.0])
        environment = self._environment(_Backend(self._ledgers(5)), provider)
        with self.assertRaises(TypeError):
            environment.reset()

        with self.assertRaisesRegex(PermissionError, "exactly one"):
            from v2.config import TimeScaleConfig
            from v2.envs.multirate_weight_env import MultiRateWeightEnvironment

            MultiRateWeightEnvironment(
                timescale=TimeScaleConfig.formal_baseline(),
                action_catalog=(self._candidate(),),
                backend=_Backend(self._ledgers(5)),
                state_provider=_StateProvider(),
            )

        from v2.config import TimeScaleConfig
        from v2.envs.multirate_weight_env import MultiRateWeightEnvironment

        forged_timescale = TimeScaleConfig.formal_baseline()
        object.__setattr__(forged_timescale, "dqn_switch_steps", 0)
        with self.assertRaises(ValueError):
            MultiRateWeightEnvironment(
                timescale=forged_timescale,
                action_catalog=(self._candidate(),),
                backend=_Backend(self._ledgers(5)),
                state_provider=_StateProvider(),
                synthetic_test_mode=True,
            )

        overflowing_timescale = TimeScaleConfig(1e308, 2, 2)
        with self.assertRaises(ValueError):
            MultiRateWeightEnvironment(
                timescale=overflowing_timescale,
                action_catalog=(self._candidate(),),
                backend=_Backend(self._ledgers(5)),
                state_provider=_StateProvider(),
                synthetic_test_mode=True,
            )

    def test_constructor_rejects_candidate_with_injected_method(self) -> None:
        from v2.config import TimeScaleConfig
        from v2.control.nonlinear_mpc import MPCWeights
        from v2.envs.multirate_weight_env import MultiRateWeightEnvironment

        candidate = self._candidate()
        object.__setattr__(candidate, "to_mpc_weights", lambda: MPCWeights(0.2, 0.3, 0.5))
        with self.assertRaises(ValueError):
            MultiRateWeightEnvironment(
                timescale=TimeScaleConfig.formal_baseline(),
                action_catalog=(candidate,),
                backend=_Backend(self._ledgers(5)),
                state_provider=_StateProvider(),
                synthetic_test_mode=True,
            )

    def test_constructor_snapshots_canonical_timescale_and_action_values(self) -> None:
        from v2.config import TimeScaleConfig
        from v2.envs.multirate_weight_env import MultiRateWeightEnvironment

        timescale = TimeScaleConfig(30.0, 2, 3)
        canonical_candidate = self._candidate()
        backend = _Backend(self._ledgers(3))
        environment = MultiRateWeightEnvironment(
            timescale=timescale,
            action_catalog=(canonical_candidate,),
            backend=backend,
            state_provider=_StateProvider(),
            synthetic_test_mode=True,
        )
        object.__setattr__(timescale, "dqn_switch_steps", 1)
        object.__setattr__(canonical_candidate, "n_base", 9)

        environment.reset()
        transition = environment.step("w_2_3_5")

        self.assertEqual(backend.calls, 3)
        self.assertEqual(transition.action.numerators, (2, 3, 5))

    def test_duplicate_action_ids_and_malformed_backend_result_fail_closed(self) -> None:
        from v2.config import TimeScaleConfig
        from v2.envs.multirate_weight_env import (
            MacroStepExecutionError,
            MultiRateWeightEnvironment,
        )

        with self.assertRaises(ValueError):
            MultiRateWeightEnvironment(
                timescale=TimeScaleConfig.formal_baseline(),
                action_catalog=(self._candidate(), self._candidate()),
                backend=_Backend(self._ledgers(5)),
                state_provider=_StateProvider(),
                synthetic_test_mode=True,
            )

        malformed_ledger = self._ledgers(1)[0]

        class MalformedBackend:
            def execute_mpc_step(self, weights):
                return (malformed_ledger, False)

        provider = _StateProvider()
        environment = MultiRateWeightEnvironment(
            timescale=TimeScaleConfig.formal_baseline(),
            action_catalog=(self._candidate(),),
            backend=MalformedBackend(),
            state_provider=provider,
            synthetic_test_mode=True,
        )
        environment.reset()
        with self.assertRaises(MacroStepExecutionError) as caught:
            environment.step("w_2_3_5")
        self.assertEqual(caught.exception.executed_mpc_steps, 1)
        self.assertEqual(environment.transitions, ())


if __name__ == "__main__":
    unittest.main()
