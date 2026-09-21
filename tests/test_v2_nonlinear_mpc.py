from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class CausalBaseLoadTests(unittest.TestCase):
    def test_alpha_uses_physical_seconds_exactly(self) -> None:
        from v2.control.causal_base_load import CausalBaseLoadFilter

        estimator = CausalBaseLoadFilter(sample_seconds=30.0, tau_seconds=120.0)
        self.assertEqual(estimator.alpha, math.exp(-30.0 / 120.0))

    def test_forecast_is_persistence_and_base_reference_is_recursive(self) -> None:
        from v2.control.causal_base_load import CausalBaseLoadFilter

        estimator = CausalBaseLoadFilter(sample_seconds=30.0, tau_seconds=30.0)
        first = estimator.observe(100.0, horizon=3)
        second = estimator.observe(200.0, horizon=3)

        self.assertEqual(first.load_kw, (100.0, 100.0, 100.0))
        self.assertEqual(first.base_reference_kw, (100.0, 100.0, 100.0))
        alpha = math.exp(-1.0)
        observed_base = alpha * 100.0 + (1.0 - alpha) * 200.0
        expected = []
        state = observed_base
        for _ in range(3):
            state = alpha * state + (1.0 - alpha) * 200.0
            expected.append(state)
        np.testing.assert_allclose(second.base_reference_kw, expected, rtol=0.0, atol=1e-14)

    def test_future_actual_iterable_cannot_be_consumed_as_an_observation(self) -> None:
        from v2.control.causal_base_load import CausalBaseLoadFilter

        class PoisonFuture:
            def __iter__(self):
                raise AssertionError("future actual measurements were accessed")

        estimator = CausalBaseLoadFilter(sample_seconds=30.0, tau_seconds=120.0)
        with self.assertRaises(TypeError):
            estimator.observe(PoisonFuture(), horizon=5)  # type: ignore[arg-type]

    def test_filter_rejects_bool_text_nonfinite_and_invalid_horizon(self) -> None:
        from v2.control.causal_base_load import CausalBaseLoadFilter

        for kwargs in (
            {"sample_seconds": True, "tau_seconds": 1.0},
            {"sample_seconds": "30", "tau_seconds": 1.0},
            {"sample_seconds": 30.0, "tau_seconds": 0.0},
            {"sample_seconds": 30.0, "tau_seconds": float("inf")},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises((TypeError, ValueError)):
                CausalBaseLoadFilter(**kwargs)  # type: ignore[arg-type]

        estimator = CausalBaseLoadFilter(sample_seconds=30.0, tau_seconds=120.0)
        for load in (True, "1", float("nan"), float("inf")):
            with self.subTest(load=load), self.assertRaises((TypeError, ValueError)):
                estimator.observe(load, horizon=5)  # type: ignore[arg-type]
        for horizon in (True, 5.0, "5", 0):
            with self.subTest(horizon=horizon), self.assertRaises((TypeError, ValueError)):
                estimator.observe(100.0, horizon=horizon)  # type: ignore[arg-type]

    def test_forecast_copies_arrays_to_finite_immutable_equal_length_tuples(self) -> None:
        from v2.control.causal_base_load import CausalLoadForecast

        loads = np.asarray([100.0, 110.0])
        references = np.asarray([90.0, 95.0])
        forecast = CausalLoadForecast(load_kw=loads, base_reference_kw=references)
        loads[:] = 999.0
        references[:] = 999.0

        self.assertEqual(forecast.load_kw, (100.0, 110.0))
        self.assertEqual(forecast.base_reference_kw, (90.0, 95.0))
        self.assertIs(type(forecast.load_kw), tuple)
        for kwargs in (
            {"load_kw": (), "base_reference_kw": ()},
            {"load_kw": (1.0,), "base_reference_kw": (1.0, 2.0)},
            {"load_kw": (float("nan"),), "base_reference_kw": (1.0,)},
            {"load_kw": np.asarray([[1.0]]), "base_reference_kw": (1.0,)},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises((TypeError, ValueError)):
                CausalLoadForecast(**kwargs)


class ObjectiveTests(unittest.TestCase):
    def test_exact_three_components_and_weighted_sum(self) -> None:
        from v2.control.nonlinear_mpc import (
            DELTA_P_FC_OBJECTIVE_SCALE_KW,
            MPCWeights,
            P_FC_OBJECTIVE_SCALE_KW,
            SOC_OBJECTIVE_SCALE,
            SOC_WORKING_HIGH,
            SOC_WORKING_LOW,
            objective_components,
            weighted_objective,
        )

        components = objective_components(
            p_fc_kw=(100.0, 130.0, 110.0),
            base_reference_kw=(90.0, 110.0, 120.0),
            soc_path=(0.3, 0.5, 0.7),
            previous_executed_p_fc_kw=80.0,
            p_fc_scale_kw=P_FC_OBJECTIVE_SCALE_KW,
            delta_p_fc_scale_kw=DELTA_P_FC_OBJECTIVE_SCALE_KW,
            soc_deadband_low=SOC_WORKING_LOW,
            soc_deadband_high=SOC_WORKING_HIGH,
            soc_scale=SOC_OBJECTIVE_SCALE,
        )

        self.assertAlmostEqual(components.j_base, 1.0 / 1800.0)
        self.assertAlmostEqual(components.j_smooth, 17.0 / 10800.0)
        self.assertAlmostEqual(components.j_soc, 1.0 / 54.0)
        weights = MPCWeights(q_base=0.2, q_smooth=0.3, q_soc=0.5)
        self.assertAlmostEqual(
            weighted_objective(components, weights),
            0.2 / 1800.0 + 0.3 * 17.0 / 10800.0 + 0.5 / 54.0,
        )
        self.assertEqual(set(components.__dataclass_fields__), {"j_base", "j_smooth", "j_soc"})

    def test_soc_deadband_is_zero_on_inclusive_edges(self) -> None:
        from v2.control.nonlinear_mpc import (
            SOC_OBJECTIVE_SCALE,
            SOC_WORKING_HIGH,
            SOC_WORKING_LOW,
            soc_deadband_penalty,
        )

        self.assertEqual(SOC_OBJECTIVE_SCALE, 0.60)
        self.assertAlmostEqual(
            soc_deadband_penalty(0.3, SOC_WORKING_LOW, SOC_WORKING_HIGH, SOC_OBJECTIVE_SCALE),
            1.0 / 36.0,
        )
        self.assertEqual(soc_deadband_penalty(0.4, 0.4, 0.6, 0.6), 0.0)
        self.assertEqual(soc_deadband_penalty(0.5, 0.4, 0.6, 0.6), 0.0)
        self.assertEqual(soc_deadband_penalty(0.6, 0.4, 0.6, 0.6), 0.0)
        self.assertAlmostEqual(
            soc_deadband_penalty(0.7, SOC_WORKING_LOW, SOC_WORKING_HIGH, SOC_OBJECTIVE_SCALE),
            1.0 / 36.0,
        )

    def test_objective_components_are_horizon_means_not_sums(self) -> None:
        from v2.control.nonlinear_mpc import objective_components

        one = objective_components(
            p_fc_kw=(300.0,),
            base_reference_kw=(0.0,),
            soc_path=(0.2,),
            previous_executed_p_fc_kw=0.0,
            p_fc_scale_kw=600.0,
            delta_p_fc_scale_kw=600.0,
            soc_deadband_low=0.4,
            soc_deadband_high=0.6,
            soc_scale=0.6,
        )
        repeated = objective_components(
            p_fc_kw=(300.0,) * 5,
            base_reference_kw=(0.0,) * 5,
            soc_path=(0.2,) * 5,
            previous_executed_p_fc_kw=300.0,
            p_fc_scale_kw=600.0,
            delta_p_fc_scale_kw=600.0,
            soc_deadband_low=0.4,
            soc_deadband_high=0.6,
            soc_scale=0.6,
        )

        self.assertEqual(one.j_base, repeated.j_base)
        self.assertEqual(one.j_soc, repeated.j_soc)
        self.assertEqual(repeated.j_base, 0.25)
        self.assertAlmostEqual(repeated.j_soc, 1.0 / 9.0)
        self.assertEqual(repeated.j_smooth, 0.0)

    def test_objective_accepts_one_dimensional_numeric_numpy_vectors(self) -> None:
        from v2.control.nonlinear_mpc import objective_components

        components = objective_components(
            p_fc_kw=np.asarray([100.0, 100.0]),
            base_reference_kw=np.asarray([100.0, 100.0]),
            soc_path=np.asarray([0.5, 0.5]),
            previous_executed_p_fc_kw=100.0,
            p_fc_scale_kw=600.0,
            delta_p_fc_scale_kw=600.0,
            soc_deadband_low=0.4,
            soc_deadband_high=0.6,
            soc_scale=0.6,
        )
        self.assertEqual(components.j_base, 0.0)
        with self.assertRaises(TypeError):
            objective_components(
                p_fc_kw=np.asarray([[100.0, 100.0]]),
                base_reference_kw=(100.0, 100.0),
                soc_path=(0.5, 0.5),
                previous_executed_p_fc_kw=100.0,
                p_fc_scale_kw=600.0,
                delta_p_fc_scale_kw=600.0,
                soc_deadband_low=0.4,
                soc_deadband_high=0.6,
                soc_scale=0.6,
            )

    def test_weights_are_positive_finite_numeric_and_sum_to_one(self) -> None:
        from v2.control.nonlinear_mpc import MPCWeights

        self.assertEqual(MPCWeights(0.1, 0.2, 0.7).q_soc, 0.7)
        for values in (
            (0.0, 0.5, 0.5),
            (-0.1, 0.5, 0.6),
            (0.2, 0.3, 0.6),
            (float("nan"), 0.5, 0.5),
            (float("inf"), 0.5, 0.5),
            (True, 0.5, 0.5),
            ("0.2", 0.3, 0.5),
        ):
            with self.subTest(values=values), self.assertRaises((TypeError, ValueError)):
                MPCWeights(*values)  # type: ignore[arg-type]

    def test_result_leaf_dataclasses_reject_mutable_nonfinite_and_wrong_types(self) -> None:
        from v2.control.nonlinear_mpc import (
            MPCCommand,
            ObjectiveComponents,
            SolverDiagnostics,
        )

        components = ObjectiveComponents(1, 2.0, np.float64(3.0))
        self.assertEqual((components.j_base, components.j_smooth, components.j_soc), (1.0, 2.0, 3.0))
        command = MPCCommand(100, -20.0, np.float64(0.5))
        self.assertEqual((command.p_fc_kw, command.p_batt_bus_kw, command.predicted_next_soc), (100.0, -20.0, 0.5))
        diagnostics = SolverDiagnostics(True, np.int64(0), "ok", np.int64(2), 1)
        self.assertEqual((diagnostics.status, diagnostics.iterations), (0, 2))

        for constructor, args in (
            (ObjectiveComponents, (float("nan"), 0.0, 0.0)),
            (ObjectiveComponents, (-1.0, 0.0, 0.0)),
            (MPCCommand, (True, 0.0, 0.5)),
            (MPCCommand, (0.0, 0.0, float("inf"))),
            (SolverDiagnostics, ("True", 0, "ok", 1, 0.0)),
            (SolverDiagnostics, (True, True, "ok", 1, 0.0)),
            (SolverDiagnostics, (True, 0, "ok", -1, 0.0)),
            (SolverDiagnostics, (True, 0, "ok", 1, float("nan"))),
        ):
            with self.subTest(constructor=constructor, args=args), self.assertRaises((TypeError, ValueError)):
                constructor(*args)


class NonlinearMPCTests(unittest.TestCase):
    @staticmethod
    def _objects(**overrides: object):
        from v2.config import TimeScaleConfig
        from v2.control.causal_base_load import CausalBaseLoadFilter
        from v2.control.nonlinear_mpc import MPCConfig, NonlinearMPC
        from v2.models.battery_energy import formal_battery_efficiency

        values: dict[str, object] = {
            "timescale": TimeScaleConfig.provisional(),
            "fuel_cell_rated_kw": 600.0,
            "battery_capacity_kwh": 624.0,
            "battery_efficiency": formal_battery_efficiency(),
            "battery_charge_min_kw": -200.0,
            "battery_discharge_max_kw": 200.0,
            "fuel_cell_ramp_kw_per_step": 100.0,
            "soc_min": 0.2,
            "soc_max": 0.8,
            "soc_deadband_low": 0.4,
            "soc_deadband_high": 0.6,
            "p_fc_scale_kw": 600.0,
            "delta_p_fc_scale_kw": 600.0,
            "soc_scale": 0.6,
        }
        values.update(overrides)
        config = MPCConfig(**values)
        estimator = CausalBaseLoadFilter(
            sample_seconds=config.timescale.ts_mpc_seconds,
            tau_seconds=120.0,
        )
        return config, estimator, NonlinearMPC(config)

    def test_n5_plan_constraints_balance_formal_soc_and_first_step_only(self) -> None:
        from v2.control.nonlinear_mpc import MPCCommand, MPCWeights
        from v2.models.battery_energy import next_soc

        config, estimator, controller = self._objects()
        result = controller.solve(
            observed_load_kw=350.0,
            current_soc=0.5,
            previous_executed_p_fc_kw=300.0,
            weights=MPCWeights(0.5, 0.25, 0.25),
            base_load_filter=estimator,
        )

        self.assertEqual(len(result.p_fc_kw), 5)
        self.assertEqual(len(result.p_batt_bus_kw), 5)
        self.assertEqual(len(result.soc_path), 5)
        self.assertEqual(len(result.base_reference_kw), 5)
        self.assertEqual(len(result.load_forecast_kw), 5)
        np.testing.assert_allclose(
            np.asarray(result.p_batt_bus_kw),
            np.asarray(result.load_forecast_kw) - np.asarray(result.p_fc_kw),
            rtol=0.0,
            atol=1e-9,
        )
        self.assertTrue(all(0.0 <= p <= config.fuel_cell_rated_kw for p in result.p_fc_kw))
        self.assertTrue(
            all(
                config.battery_charge_min_kw <= p <= config.battery_discharge_max_kw
                for p in result.p_batt_bus_kw
            )
        )
        deltas = np.diff((300.0,) + result.p_fc_kw)
        self.assertTrue(np.all(np.abs(deltas) <= config.fuel_cell_ramp_kw_per_step + 1e-7))
        self.assertTrue(all(config.soc_min <= soc <= config.soc_max for soc in result.soc_path))

        state = 0.5
        expected_soc = []
        for power in result.p_batt_bus_kw:
            state = next_soc(
                state,
                power,
                config.timescale.ts_mpc_seconds,
                config.battery_capacity_kwh,
                efficiency=config.battery_efficiency,
            )
            expected_soc.append(state)
        np.testing.assert_allclose(result.soc_path, expected_soc, rtol=0.0, atol=1e-12)

        command = result.first_command()
        self.assertIsInstance(command, MPCCommand)
        self.assertEqual(set(command.__dataclass_fields__), {"p_fc_kw", "p_batt_bus_kw", "predicted_next_soc"})
        self.assertEqual(command.p_fc_kw, result.p_fc_kw[0])
        self.assertFalse(hasattr(command, "plan"))

    def test_cold_and_explicit_shifted_warm_starts_are_deterministic(self) -> None:
        from v2.control.nonlinear_mpc import MPCWeights, shifted_warm_start

        weights = MPCWeights(0.5, 0.25, 0.25)
        _, estimator_a, controller_a = self._objects()
        cold_a = controller_a.solve(350.0, 0.5, 300.0, weights, estimator_a)
        _, estimator_b, controller_b = self._objects()
        cold_b = controller_b.solve(350.0, 0.5, 300.0, weights, estimator_b)
        np.testing.assert_allclose(cold_a.p_fc_kw, cold_b.p_fc_kw, rtol=0.0, atol=1e-10)

        warm = shifted_warm_start(cold_a)
        self.assertEqual(warm, cold_a.p_fc_kw[1:] + (cold_a.p_fc_kw[-1],))
        _, estimator_c, controller_c = self._objects()
        warm_a = controller_c.solve(350.0, 0.5, 300.0, weights, estimator_c, warm_start=warm)
        _, estimator_d, controller_d = self._objects()
        warm_b = controller_d.solve(350.0, 0.5, 300.0, weights, estimator_d, warm_start=warm)
        np.testing.assert_allclose(warm_a.p_fc_kw, warm_b.p_fc_kw, rtol=0.0, atol=1e-10)

    def test_controller_plan_length_follows_n_mpc_not_dqn_switch_steps(self) -> None:
        from v2.config import TimeScaleConfig
        from v2.control.nonlinear_mpc import MPCWeights

        timescale = TimeScaleConfig(30.0, n_mpc=3, dqn_switch_steps=11)
        _, estimator, controller = self._objects(timescale=timescale)
        result = controller.solve(300.0, 0.5, 300.0, MPCWeights(0.5, 0.25, 0.25), estimator)
        self.assertEqual(len(result.p_fc_kw), 3)

    def test_obvious_physical_infeasibility_is_distinct_from_solver_failure(self) -> None:
        from v2.control.nonlinear_mpc import (
            MPCWeights,
            NumericalSolverError,
            PhysicalInfeasibilityError,
        )

        _, estimator, controller = self._objects(battery_discharge_max_kw=50.0)
        with self.assertRaises(PhysicalInfeasibilityError) as caught:
            controller.solve(800.0, 0.5, 300.0, MPCWeights(0.5, 0.25, 0.25), estimator)
        self.assertEqual(caught.exception.kind, "physical_infeasibility")

        def failed_optimizer(*args: object, **kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(
                success=False,
                status=9,
                message="iteration limit",
                nit=1,
                fun=float("nan"),
                x=np.full(5, 300.0),
            )

        _, estimator, _ = self._objects()
        _, _, numerical_controller = self._objects()
        numerical_controller = type(numerical_controller)(numerical_controller.config, optimizer=failed_optimizer)
        with self.assertRaises(NumericalSolverError) as caught:
            numerical_controller.solve(
                300.0, 0.5, 300.0, MPCWeights(0.5, 0.25, 0.25), estimator
            )
        self.assertEqual(caught.exception.kind, "numerical_solver_failure")
        self.assertEqual(caught.exception.status, 9)

    def test_successful_optimizer_output_is_physically_rechecked(self) -> None:
        from v2.control.nonlinear_mpc import MPCWeights, NumericalSolverError, NonlinearMPC

        def lying_optimizer(*args: object, **kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(
                success=True,
                status=7,
                message="claimed success",
                nit=1,
                fun=0.0,
                x=np.full(5, 1000.0),
            )

        config, estimator, _ = self._objects()
        with self.assertRaises(NumericalSolverError) as caught:
            NonlinearMPC(config, optimizer=lying_optimizer).solve(
                300.0, 0.5, 300.0, MPCWeights(0.5, 0.25, 0.25), estimator
            )
        self.assertEqual(caught.exception.status, 7)

    def test_nonnumeric_success_vector_is_a_structured_numerical_failure(self) -> None:
        from v2.control.nonlinear_mpc import MPCWeights, NumericalSolverError, NonlinearMPC

        def nonnumeric_optimizer(*args: object, **kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(
                success=True,
                status=8,
                message="claimed success",
                nit=1,
                fun=0.0,
                x="not-a-vector",
            )

        config, estimator, _ = self._objects()
        with self.assertRaises(NumericalSolverError) as caught:
            NonlinearMPC(config, optimizer=nonnumeric_optimizer).solve(
                300.0, 0.5, 300.0, MPCWeights(0.5, 0.25, 0.25), estimator
            )
        self.assertEqual(caught.exception.kind, "numerical_solver_failure")
        self.assertEqual(caught.exception.status, 8)

    def test_malformed_success_vector_is_a_structured_numerical_failure(self) -> None:
        from v2.control.nonlinear_mpc import MPCWeights, NumericalSolverError, NonlinearMPC

        for returned in ((300.0,), (300.0, 300.0, 300.0, 300.0, float("nan"))):
            def malformed_optimizer(*args: object, **kwargs: object) -> SimpleNamespace:
                return SimpleNamespace(
                    success=True,
                    status=0,
                    message="claimed success",
                    nit=1,
                    fun=0.0,
                    x=returned,
                )

            config, estimator, _ = self._objects()
            with self.subTest(returned=returned), self.assertRaises(NumericalSolverError):
                NonlinearMPC(config, optimizer=malformed_optimizer).solve(
                    300.0, 0.5, 300.0, MPCWeights(0.5, 0.25, 0.25), estimator
                )

    def test_optimizer_exception_is_a_structured_numerical_failure(self) -> None:
        from v2.control.nonlinear_mpc import MPCWeights, NumericalSolverError, NonlinearMPC

        def exploding_optimizer(*args: object, **kwargs: object) -> object:
            raise RuntimeError("backend crashed")

        config, estimator, _ = self._objects()
        with self.assertRaises(NumericalSolverError) as caught:
            NonlinearMPC(config, optimizer=exploding_optimizer).solve(
                300.0, 0.5, 300.0, MPCWeights(0.5, 0.25, 0.25), estimator
            )
        self.assertIsNone(caught.exception.status)
        self.assertIn("backend crashed", str(caught.exception))

    def test_malformed_optimizer_metadata_is_a_structured_failure(self) -> None:
        from v2.control.nonlinear_mpc import MPCWeights, NumericalSolverError, NonlinearMPC

        attacks = (
            ({"success": "False", "status": 4, "nit": 1}, 4),
            ({"success": True, "status": "not-an-int", "nit": 1}, None),
            ({"success": True, "status": 6, "nit": "one"}, 6),
            ({"success": True, "status": 7, "nit": -1}, 7),
        )
        for metadata, expected_status in attacks:
            def malicious_optimizer(*args: object, **kwargs: object) -> SimpleNamespace:
                return SimpleNamespace(
                    **metadata,
                    message="malformed metadata",
                    fun=0.0,
                    x=np.full(5, 300.0),
                )

            config, estimator, _ = self._objects()
            with self.subTest(metadata=metadata), self.assertRaises(NumericalSolverError) as caught:
                NonlinearMPC(config, optimizer=malicious_optimizer).solve(
                    300.0, 0.5, 300.0, MPCWeights(0.5, 0.25, 0.25), estimator
                )
            self.assertEqual(caught.exception.status, expected_status)

        def numpy_false_optimizer(*args: object, **kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(
                success=np.bool_(False),
                status=np.int64(-3),
                message="valid failure metadata",
                nit=np.int64(0),
                fun=float("nan"),
                x=np.full(5, 300.0),
            )

        config, estimator, _ = self._objects()
        with self.assertRaises(NumericalSolverError) as caught:
            NonlinearMPC(config, optimizer=numpy_false_optimizer).solve(
                300.0, 0.5, 300.0, MPCWeights(0.5, 0.25, 0.25), estimator
            )
        self.assertEqual(caught.exception.status, -3)

    def test_failed_solves_do_not_consume_filter_observation_and_success_commits_once(self) -> None:
        from v2.control.nonlinear_mpc import MPCWeights, NonlinearMPC, NumericalSolverError

        weights = MPCWeights(0.5, 0.25, 0.25)
        config, estimator, controller = self._objects()
        estimator.observe(100.0, horizon=1)
        before = estimator.observed_base_kw

        with self.assertRaises(ValueError):
            controller.solve(200.0, 0.5, 100.0, weights, estimator, warm_start=(1.0,))
        self.assertEqual(estimator.observed_base_kw, before)

        def failed_optimizer(*args: object, **kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(
                success=False,
                status=9,
                message="failed",
                nit=1,
                fun=float("nan"),
                x=np.full(5, 100.0),
            )

        with self.assertRaises(NumericalSolverError):
            NonlinearMPC(config, optimizer=failed_optimizer).solve(
                200.0, 0.5, 100.0, weights, estimator
            )
        self.assertEqual(estimator.observed_base_kw, before)

        def invalid_solution(*args: object, **kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(
                success=True,
                status=7,
                message="claimed success",
                nit=1,
                fun=0.0,
                x=np.full(5, 1000.0),
            )

        with self.assertRaises(NumericalSolverError):
            NonlinearMPC(config, optimizer=invalid_solution).solve(
                200.0, 0.5, 100.0, weights, estimator
            )
        self.assertEqual(estimator.observed_base_kw, before)

        result = controller.solve(200.0, 0.5, 100.0, weights, estimator)
        expected_once = estimator.alpha * float(before) + (1.0 - estimator.alpha) * 200.0
        self.assertAlmostEqual(float(estimator.observed_base_kw), expected_once)
        self.assertEqual(len(result.p_fc_kw), 5)

    def test_plan_defensively_copies_vectors_and_validates_nested_results(self) -> None:
        from v2.control.nonlinear_mpc import (
            MPCPlan,
            ObjectiveComponents,
            SolverDiagnostics,
            shifted_warm_start,
        )

        source = np.asarray([100.0, 110.0])
        components = ObjectiveComponents(1.0, 2.0, 3.0)
        diagnostics = SolverDiagnostics(True, 0, "ok", 2, 6.0)
        plan = MPCPlan(
            p_fc_kw=source,
            p_batt_bus_kw=np.asarray([10.0, 20.0]),
            soc_path=np.asarray([0.5, 0.49]),
            load_forecast_kw=np.asarray([110.0, 130.0]),
            base_reference_kw=np.asarray([90.0, 95.0]),
            components=components,
            objective_value=6.0,
            diagnostics=diagnostics,
        )
        source[:] = float("nan")
        self.assertEqual(plan.p_fc_kw, (100.0, 110.0))
        self.assertTrue(all(type(vector) is tuple for vector in (
            plan.p_fc_kw,
            plan.p_batt_bus_kw,
            plan.soc_path,
            plan.load_forecast_kw,
            plan.base_reference_kw,
        )))
        warm = shifted_warm_start(plan)
        self.assertEqual(warm, (110.0, 110.0))
        self.assertIsNot(warm, plan.p_fc_kw)

        valid = {
            "p_fc_kw": (100.0, 110.0),
            "p_batt_bus_kw": (10.0, 20.0),
            "soc_path": (0.5, 0.49),
            "load_forecast_kw": (110.0, 130.0),
            "base_reference_kw": (90.0, 95.0),
            "components": components,
            "objective_value": 6.0,
            "diagnostics": diagnostics,
        }
        attacks = (
            {"p_fc_kw": ()},
            {"soc_path": (0.5,)},
            {"p_batt_bus_kw": (10.0, float("nan"))},
            {"components": SimpleNamespace(j_base=1.0, j_smooth=2.0, j_soc=3.0)},
            {"diagnostics": SimpleNamespace(success=True)},
            {"objective_value": float("nan")},
        )
        for override in attacks:
            values = dict(valid)
            values.update(override)
            with self.subTest(override=override), self.assertRaises((TypeError, ValueError)):
                MPCPlan(**values)

    def test_config_and_runtime_inputs_are_strictly_validated(self) -> None:
        from v2.control.nonlinear_mpc import MPCWeights

        invalid_configs = (
            {"fuel_cell_rated_kw": True},
            {"battery_capacity_kwh": "624"},
            {"battery_charge_min_kw": 0.0},
            {"battery_discharge_max_kw": float("inf")},
            {"fuel_cell_ramp_kw_per_step": 0.0},
            {"soc_min": 0.8, "soc_max": 0.2},
            {"soc_deadband_low": 0.6, "soc_deadband_high": 0.6},
            {"soc_scale": float("nan")},
            {"soc_min": 0.1},
            {"soc_max": 0.9},
            {"soc_deadband_low": 0.3},
            {"soc_deadband_high": 0.7},
            {"p_fc_scale_kw": 599.0},
            {"delta_p_fc_scale_kw": 100.0},
            {"soc_scale": 0.05},
        )
        for override in invalid_configs:
            with self.subTest(override=override), self.assertRaises((TypeError, ValueError)):
                self._objects(**override)

        _, estimator, controller = self._objects()
        weights = MPCWeights(0.5, 0.25, 0.25)
        for name, value in (
            ("observed_load_kw", True),
            ("observed_load_kw", "300"),
            ("current_soc", float("nan")),
            ("previous_executed_p_fc_kw", float("inf")),
        ):
            kwargs = {
                "observed_load_kw": 300.0,
                "current_soc": 0.5,
                "previous_executed_p_fc_kw": 300.0,
                "weights": weights,
                "base_load_filter": estimator,
            }
            kwargs[name] = value
            with self.subTest(name=name, value=value), self.assertRaises((TypeError, ValueError)):
                controller.solve(**kwargs)  # type: ignore[arg-type]
        for warm in ((1.0,), (1.0, 2.0, 3.0, 4.0, float("nan")), "bad"):
            with self.subTest(warm=warm), self.assertRaises((TypeError, ValueError)):
                controller.solve(300.0, 0.5, 300.0, weights, estimator, warm_start=warm)  # type: ignore[arg-type]

    def test_mpc_module_has_no_economic_or_degradation_objective_surface(self) -> None:
        module = ROOT / "src" / "v2" / "control" / "nonlinear_mpc.py"
        text = module.read_text(encoding="utf-8").casefold()
        for forbidden in ("hydrogen", "price", "degradation", "reward", "economics"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, text)


if __name__ == "__main__":
    unittest.main()
