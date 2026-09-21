from __future__ import annotations

from dataclasses import FrozenInstanceError
import math
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class TrainOnlyTimeScaleAuditTests(unittest.TestCase):
    @staticmethod
    def _provenance(split=None):
        from v2.analysis.action_screening import DataSplit, DatasetProvenance
        from v2.contracts import DATASET_VERSION

        return DatasetProvenance(
            DATASET_VERSION,
            "sha256:synthetic-timescale-fixture",
            split if split is not None else DataSplit.TRAIN,
        )

    def test_every_method_selection_target_is_centrally_train_only(self) -> None:
        from v2.analysis.action_screening import DataSplit, HeldOutSelectionError
        from v2.analysis.timescale_audit import (
            SelectionParameter,
            load_train_selection_payload,
        )

        expected = {
            "n_mpc",
            "dqn_switch_steps",
            "tau_lpf_seconds",
            "soc_deadband",
            "state_schema",
            "action_catalog",
            "reward_scale",
        }
        self.assertEqual({item.value for item in SelectionParameter}, expected)
        for split in (DataSplit.VALIDATION, DataSplit.TEST, DataSplit.UNKNOWN):
            for parameter in SelectionParameter:
                reads = []

                def loader():
                    reads.append(parameter)
                    return (1.0, 2.0)

                with self.subTest(split=split, parameter=parameter):
                    with self.assertRaises(HeldOutSelectionError):
                        load_train_selection_payload(
                            provenance=self._provenance(split),
                            parameter=parameter,
                            payload_loader=loader,
                        )
                    self.assertEqual(reads, [])

    def test_synthetic_diagnostics_detect_autocorrelation_variance_changes_and_regimes(self) -> None:
        from v2.analysis.timescale_audit import run_timescale_audit

        source = [0.0] * 5 + [10.0] * 5 + [0.0] * 5 + [10.0] * 5
        result = run_timescale_audit(
            provenance=self._provenance(),
            payload_loader=lambda: source,
            sample_seconds=30.0,
            autocorrelation_lags=(1, 5),
            rolling_window_samples=5,
            change_threshold=5.0,
        )

        self.assertEqual(result.diagnostics.change_point_indices, (5, 10, 15))
        self.assertEqual(result.diagnostics.regime_durations_samples, (5, 5, 5, 5))
        self.assertEqual(result.diagnostics.regime_durations_seconds, (150.0,) * 4)
        self.assertEqual(len(result.diagnostics.rolling_variance), 16)
        self.assertEqual(result.diagnostics.rolling_variance[0], 0.0)
        self.assertEqual(result.diagnostics.rolling_variance[5], 0.0)
        self.assertEqual(result.diagnostics.autocorrelation_lags, (1, 5))
        self.assertEqual(len(result.diagnostics.autocorrelation), 2)
        self.assertTrue(all(math.isfinite(value) for value in result.diagnostics.autocorrelation))

        by_m = {item.dqn_switch_steps: item for item in result.switch_sensitivity}
        self.assertEqual(tuple(by_m), (5, 10))
        self.assertEqual(by_m[5].macro_means, (0.0, 10.0, 0.0, 10.0))
        self.assertEqual(by_m[5].mean_within_interval_variance, 0.0)
        self.assertEqual(by_m[10].macro_means, (5.0, 5.0))
        self.assertEqual(by_m[10].mean_within_interval_variance, 25.0)
        self.assertEqual(result.n_mpc, 5)
        self.assertEqual(result.sample_seconds, 30.0)
        self.assertEqual(result.sample_count, 20)
        self.assertEqual(result.candidate_switch_steps, (5, 10))
        self.assertEqual(result.formal_selection_status, "NO-GO")
        self.assertIsNone(result.selected_dqn_switch_steps)
        self.assertEqual(result.validate(), result)

        source[:] = [999.0]
        self.assertEqual(result.diagnostics.regime_durations_samples, (5, 5, 5, 5))

    def test_small_hand_computable_diagnostics_use_declared_definitions(self) -> None:
        from v2.analysis.timescale_audit import run_timescale_audit

        result = run_timescale_audit(
            provenance=self._provenance(),
            payload_loader=lambda: (0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 0.0, 0.0, 0.0, 10.0),
            sample_seconds=1.0,
            autocorrelation_lags=(1,),
            rolling_window_samples=3,
            change_threshold=5.0,
        )
        expected_first_mixture = 200.0 / 9.0
        self.assertAlmostEqual(result.diagnostics.rolling_variance[1], expected_first_mixture)
        self.assertEqual(result.diagnostics.change_point_indices, (3, 6, 9))
        self.assertEqual(result.diagnostics.regime_durations_samples, (3, 3, 3, 1))

    def test_timescale_domain_is_closed_and_exact(self) -> None:
        from v2.analysis.timescale_audit import run_timescale_audit

        base = dict(
            provenance=self._provenance(),
            payload_loader=lambda: tuple(float(i) for i in range(20)),
            sample_seconds=30.0,
            autocorrelation_lags=(1,),
            rolling_window_samples=5,
            change_threshold=2.0,
        )
        for override in (
            {"n_mpc": True},
            {"n_mpc": 4},
            {"candidate_switch_steps": [5, 10]},
            {"candidate_switch_steps": (10, 5)},
            {"candidate_switch_steps": (5.0, 10.0)},
            {"sample_seconds": float("nan")},
            {"autocorrelation_lags": (True,)},
            {"rolling_window_samples": 1.0},
            {"change_threshold": float("inf")},
        ):
            values = dict(base)
            values.update(override)
            with self.subTest(override=override), self.assertRaises((TypeError, ValueError)):
                run_timescale_audit(**values)

        for payload in ((1.0, float("nan")) + (1.0,) * 18, (True,) + (1.0,) * 19):
            values = dict(base)
            values["payload_loader"] = lambda payload=payload: payload
            with self.subTest(payload=payload[:2]), self.assertRaises((TypeError, ValueError)):
                run_timescale_audit(**values)

    def test_switch_sensitivity_records_every_discarded_tail_sample(self) -> None:
        from v2.analysis.timescale_audit import run_timescale_audit

        result = run_timescale_audit(
            provenance=self._provenance(),
            payload_loader=lambda: tuple(float(i) for i in range(21)),
            sample_seconds=30.0,
            autocorrelation_lags=(1,),
            rolling_window_samples=5,
            change_threshold=2.0,
        )
        self.assertEqual(
            tuple(item.discarded_tail_samples for item in result.switch_sensitivity),
            (1, 1),
        )

    def test_extreme_finite_constant_series_keeps_representable_statistics_finite(self) -> None:
        from v2.analysis.timescale_audit import run_timescale_audit

        result = run_timescale_audit(
            provenance=self._provenance(),
            payload_loader=lambda: (1e308,) * 20,
            sample_seconds=30.0,
            autocorrelation_lags=(1,),
            rolling_window_samples=5,
            change_threshold=1.0,
        )
        self.assertEqual(result.diagnostics.autocorrelation, (0.0,))
        self.assertEqual(result.diagnostics.rolling_variance, (0.0,) * 16)
        self.assertTrue(
            all(
                all(value == 1e308 for value in item.macro_means)
                for item in result.switch_sensitivity
            )
        )

    def test_provenance_must_be_exact_current_v2_and_results_are_sealed(self) -> None:
        from v2.analysis.action_screening import DataSplit, DatasetProvenance
        from v2.analysis.timescale_audit import TimeScaleAuditResult, run_timescale_audit

        with self.assertRaises(ValueError):
            run_timescale_audit(
                provenance=DatasetProvenance("synthetic", "forged", DataSplit.TRAIN),
                payload_loader=lambda: (0.0,) * 20,
                sample_seconds=30.0,
                autocorrelation_lags=(1,),
                rolling_window_samples=5,
                change_threshold=1.0,
            )
        with self.assertRaises(TypeError):
            TimeScaleAuditResult()  # type: ignore[call-arg]

        provenance = self._provenance()
        result = run_timescale_audit(
            provenance=provenance,
            payload_loader=lambda: tuple(float(i) for i in range(20)),
            sample_seconds=30.0,
            autocorrelation_lags=(1,),
            rolling_window_samples=5,
            change_threshold=2.0,
        )
        object.__setattr__(provenance, "provenance_id", "mutated")
        self.assertNotEqual(result.provenance.provenance_id, "mutated")
        with self.assertRaises(FrozenInstanceError):
            result.n_mpc = 4  # type: ignore[misc]
        object.__setattr__(result, "n_mpc", 4)
        with self.assertRaises(ValueError):
            result.validate()

        injected = run_timescale_audit(
            provenance=self._provenance(),
            payload_loader=lambda: tuple(float(i) for i in range(20)),
            sample_seconds=30.0,
            autocorrelation_lags=(1,),
            rolling_window_samples=5,
            change_threshold=2.0,
        )
        object.__setattr__(injected.diagnostics, "unsealed", True)
        with self.assertRaises(ValueError):
            injected.validate()


class PairedSolverAuditTests(unittest.TestCase):
    @staticmethod
    def _provenance(split=None):
        from v2.analysis.action_screening import DataSplit, DatasetProvenance
        from v2.contracts import DATASET_VERSION

        return DatasetProvenance(
            DATASET_VERSION,
            "sha256:synthetic-solver-fixture",
            split if split is not None else DataSplit.TRAIN,
        )

    @staticmethod
    def _cases():
        from v2.analysis.solver_audit import SolverAuditCase

        return (
            SolverAuditCase("case-a", seed=101, order=0, payload=(1.0, 2.0)),
            SolverAuditCase("case-b", seed=202, order=1, payload=(3.0, 4.0)),
        )

    def test_held_out_solver_audit_rejects_before_case_or_solver_access(self) -> None:
        from v2.analysis.action_screening import DataSplit, HeldOutSelectionError
        from v2.analysis.solver_audit import run_paired_solver_audit

        for split in (DataSplit.VALIDATION, DataSplit.TEST, DataSplit.UNKNOWN):
            events = []

            def cases_loader():
                events.append("cases")
                return self._cases()

            def runner(case, mode):
                events.append((case, mode))
                raise AssertionError("must not run")

            with self.subTest(split=split), self.assertRaises(HeldOutSelectionError):
                run_paired_solver_audit(
                    provenance=self._provenance(split),
                    cases_loader=cases_loader,
                    solver_runner=runner,
                )
            self.assertEqual(events, [])

    def test_paired_audit_preserves_case_seed_order_and_separates_success_from_speed(self) -> None:
        from v2.analysis.solver_audit import (
            SolverRunObservation,
            SolverStartMode,
            run_paired_solver_audit,
        )

        events = []

        def runner(case, mode):
            events.append((case.case_id, case.seed, case.order, mode))
            is_cold = mode is SolverStartMode.COLD
            success = not (case.case_id == "case-b" and is_cold)
            duration = {("case-a", True): 2.0, ("case-a", False): 1.0,
                        ("case-b", True): 4.0, ("case-b", False): 3.0}[(case.case_id, is_cold)]
            return SolverRunObservation(
                case_id=case.case_id,
                seed=case.seed,
                order=case.order,
                mode=mode,
                success=success,
                solve_seconds=duration,
                iterations=10 if is_cold else 6,
                status=0 if success else 9,
            )

        result = run_paired_solver_audit(
            provenance=self._provenance(),
            cases_loader=self._cases,
            solver_runner=runner,
        )

        self.assertEqual(
            events,
            [
                ("case-a", 101, 0, SolverStartMode.COLD),
                ("case-a", 101, 0, SolverStartMode.WARM),
                ("case-b", 202, 1, SolverStartMode.COLD),
                ("case-b", 202, 1, SolverStartMode.WARM),
            ],
        )
        self.assertEqual(tuple(pair.case.case_id for pair in result.pairs), ("case-a", "case-b"))
        self.assertEqual(result.cold_success_count, 1)
        self.assertEqual(result.warm_success_count, 2)
        self.assertEqual(result.both_success_count, 1)
        self.assertEqual(result.cold_success_rate, 0.5)
        self.assertEqual(result.warm_success_rate, 1.0)
        self.assertEqual(result.speed_comparison_case_ids, ("case-a",))
        self.assertEqual(result.median_cold_over_warm_solve_time, 2.0)
        self.assertEqual(result.formal_decision_status, "NO-GO")
        self.assertEqual(result.n_mpc, 5)
        self.assertEqual(result.validate(), result)
        object.__setattr__(result, "unsealed", True)
        with self.assertRaises(ValueError):
            result.validate()

    def test_runner_metadata_mismatch_is_rejected_instead_of_being_repaired(self) -> None:
        from v2.analysis.solver_audit import (
            SolverRunObservation,
            SolverStartMode,
            run_paired_solver_audit,
        )

        def wrong_runner(case, mode):
            return SolverRunObservation(
                case_id=case.case_id,
                seed=case.seed + (1 if mode is SolverStartMode.WARM else 0),
                order=case.order,
                mode=mode,
                success=True,
                solve_seconds=1.0,
                iterations=1,
                status=0,
            )

        with self.assertRaises(ValueError):
            run_paired_solver_audit(
                provenance=self._provenance(),
                cases_loader=self._cases,
                solver_runner=wrong_runner,
            )

        for invalid_n in (True, 4):
            reads = []
            with self.subTest(n_mpc=invalid_n), self.assertRaises((TypeError, ValueError)):
                run_paired_solver_audit(
                    provenance=self._provenance(),
                    cases_loader=lambda: reads.append("read") or self._cases(),
                    solver_runner=wrong_runner,
                    n_mpc=invalid_n,
                )
            self.assertEqual(reads, [])

    def test_solver_audit_rejects_mutation_nonfinite_bool_alias_and_bad_order(self) -> None:
        from v2.analysis.solver_audit import (
            PairedSolverAuditResult,
            SolverAuditCase,
            SolverRunObservation,
            SolverStartMode,
            run_paired_solver_audit,
        )

        with self.assertRaises(TypeError):
            PairedSolverAuditResult()  # type: ignore[call-arg]
        for constructor in (
            lambda: SolverAuditCase("x", True, 0, (1.0,)),
            lambda: SolverAuditCase("x", 1, 0, (float("nan"),)),
            lambda: SolverRunObservation("x", 1, 0, SolverStartMode.COLD, 1, 1.0, 1, 0),
            lambda: SolverRunObservation("x", 1, 0, SolverStartMode.COLD, True, float("nan"), 1, 0),
            lambda: SolverRunObservation("x", 1, 0, SolverStartMode.COLD, True, 1.0, True, 0),
        ):
            with self.assertRaises((TypeError, ValueError)):
                constructor()

        bad_cases = (
            SolverAuditCase("case-a", 1, 1, (1.0,)),
            SolverAuditCase("case-b", 2, 0, (2.0,)),
        )
        with self.assertRaises(ValueError):
            run_paired_solver_audit(
                provenance=self._provenance(),
                cases_loader=lambda: bad_cases,
                solver_runner=lambda case, mode: None,
            )


if __name__ == "__main__":
    unittest.main()
