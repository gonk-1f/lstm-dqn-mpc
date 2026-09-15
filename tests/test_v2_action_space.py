from __future__ import annotations

import unittest
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class V2CandidateActionSpaceTests(unittest.TestCase):
    def test_positive_tenth_grid_has_exact_deterministic_integer_catalog(self) -> None:
        from v2.contracts import ACTION_TABLE_VERSION as CONTRACT_VERSION
        from v2.dqn.action_space import (
            ACTION_TABLE_VERSION,
            CANDIDATE_ACTION_BANK,
            generate_candidate_action_bank,
        )

        expected_numerators = tuple(
            (n_base, n_smooth, 10 - n_base - n_smooth)
            for n_base in range(1, 9)
            for n_smooth in range(1, 10 - n_base)
        )
        expected_ids = tuple(
            f"w_{n_base}_{n_smooth}_{n_soc}"
            for n_base, n_smooth, n_soc in expected_numerators
        )

        first = generate_candidate_action_bank()
        second = generate_candidate_action_bank()
        self.assertEqual(ACTION_TABLE_VERSION, CONTRACT_VERSION)
        self.assertEqual(ACTION_TABLE_VERSION, "three_weight_simplex_behavior_filtered_v1")
        self.assertEqual(first, second)
        self.assertEqual(first, CANDIDATE_ACTION_BANK)
        self.assertEqual(len(first), 36)
        self.assertEqual(tuple(action.numerators for action in first), expected_numerators)
        self.assertEqual(tuple(action.action_id for action in first), expected_ids)
        self.assertEqual(len({action.numerators for action in first}), 36)
        self.assertEqual(len({action.action_id for action in first}), 36)

    def test_candidates_convert_exactly_to_task5_mpc_weights(self) -> None:
        from v2.control.nonlinear_mpc import MPCWeights
        from v2.dqn.action_space import CANDIDATE_ACTION_BANK

        for action in CANDIDATE_ACTION_BANK:
            with self.subTest(action=action.action_id):
                self.assertTrue(all(n >= 1 for n in action.numerators))
                self.assertEqual(sum(action.numerators), 10)
                self.assertTrue(all(weight >= 0.1 for weight in action.as_tuple()))
                self.assertAlmostEqual(sum(action.as_tuple()), 1.0)
                weights = action.to_mpc_weights()
                self.assertIs(type(weights), MPCWeights)
                self.assertEqual(
                    (weights.q_base, weights.q_smooth, weights.q_soc),
                    action.as_tuple(),
                )

    def test_candidate_bank_is_not_a_default_final_dqn_catalog(self) -> None:
        from v2.dqn.action_space import (
            ACTION_CATALOG_STATUS,
            CANDIDATE_ACTION_BANK,
            FINAL_DQN_ACTION_CATALOG,
            ActionCatalogUnavailableError,
            get_final_dqn_action_catalog,
        )

        self.assertEqual(len(CANDIDATE_ACTION_BANK), 36)
        self.assertEqual(ACTION_CATALOG_STATUS, "NO-GO")
        self.assertIsNone(FINAL_DQN_ACTION_CATALOG)
        with self.assertRaisesRegex(ActionCatalogUnavailableError, "NO-GO"):
            get_final_dqn_action_catalog()


class V2ActionScreeningTests(unittest.TestCase):
    def setUp(self) -> None:
        from v2.analysis.action_screening import (
            BehaviorFingerprint,
            CandidateScreeningRecord,
            DataSplit,
            DatasetProvenance,
            FeasibilityResult,
            MetricDefinition,
            MetricDirection,
            SolverReproducibilityResult,
        )

        self.train = DatasetProvenance(
            dataset_version="synthetic_train_v1",
            provenance_id="unit-test-fixture",
            split=DataSplit.TRAIN,
        )
        self.metrics = (
            MetricDefinition("tracking_error", MetricDirection.MINIMIZE, 1.0),
            MetricDefinition("smoothness", MetricDirection.MINIMIZE, 2.0),
        )

        def record(
            candidate_id: str,
            values: tuple[float, float],
            *,
            feasible: bool = True,
            reproducible: bool = True,
            provenance: object = self.train,
        ) -> object:
            feasibility = FeasibilityResult(
                candidate_id, provenance, feasible, "synthetic feasibility"
            )
            solver = SolverReproducibilityResult(
                candidate_id,
                provenance,
                reproducible,
                2,
                "synthetic repeated solve",
            )
            fingerprint = BehaviorFingerprint(candidate_id, self.metrics, values)
            return CandidateScreeningRecord(
                candidate_id, feasibility, solver, fingerprint
            )

        self.record = record

    def test_train_provenance_is_exact_and_mixed_or_held_out_batches_fail(self) -> None:
        from v2.analysis.action_screening import (
            DataSplit,
            DatasetProvenance,
            HeldOutSelectionError,
            pareto_front,
        )

        for split in ("Train", "train", True, 1, None):
            with self.subTest(split=split), self.assertRaises(TypeError):
                DatasetProvenance("v", "p", split)  # type: ignore[arg-type]

        for split in (DataSplit.VALIDATION, DataSplit.TEST, DataSplit.UNKNOWN):
            held_out = DatasetProvenance("synthetic_train_v1", "held", split)
            with self.subTest(split=split), self.assertRaises(HeldOutSelectionError):
                pareto_front((self.record("a", (1.0, 1.0), provenance=held_out),))

        other_train = DatasetProvenance(
            "synthetic_train_v1", "different-source", DataSplit.TRAIN
        )
        mixed = (
            self.record("a", (1.0, 1.0)),
            self.record("b", (2.0, 2.0), provenance=other_train),
        )
        with self.assertRaisesRegex(ValueError, "same.*provenance"):
            pareto_front(mixed)

        held_out = DatasetProvenance("synthetic_train_v1", "held", DataSplit.TEST)
        with self.assertRaises(HeldOutSelectionError):
            pareto_front(
                (
                    self.record("a", (1.0, 1.0)),
                    self.record(
                        "failed-held-out",
                        (99.0, 99.0),
                        feasible=False,
                        provenance=held_out,
                    ),
                )
            )

    def test_provenance_metadata_and_gate_results_are_strict(self) -> None:
        from v2.analysis.action_screening import (
            DataSplit,
            DatasetProvenance,
            FeasibilityResult,
            SolverReproducibilityResult,
        )

        for values in (("", "p"), ("v", " "), (1, "p"), ("v", True)):
            with self.subTest(values=values), self.assertRaises((TypeError, ValueError)):
                DatasetProvenance(values[0], values[1], DataSplit.TRAIN)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            FeasibilityResult("a", self.train, 1, "reason")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            SolverReproducibilityResult("a", self.train, True, True, "reason")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            SolverReproducibilityResult("a", self.train, True, 1, "reason")

    def test_fingerprints_are_immutable_finite_and_schema_safe(self) -> None:
        from dataclasses import FrozenInstanceError
        from v2.analysis.action_screening import (
            BehaviorFingerprint,
            MetricDefinition,
            MetricDirection,
        )

        source = np.asarray([1.0, 2.0])
        fingerprint = BehaviorFingerprint("a", self.metrics, source)
        source[:] = 99.0
        self.assertEqual(fingerprint.values, (1.0, 2.0))
        self.assertIs(type(fingerprint.values), tuple)
        with self.assertRaises(FrozenInstanceError):
            fingerprint.values = (3.0, 4.0)  # type: ignore[misc]

        with self.assertRaises(ValueError):
            BehaviorFingerprint("a", self.metrics, (float("nan"), 2.0))
        with self.assertRaises(ValueError):
            BehaviorFingerprint("a", self.metrics, (1.0,))
        duplicate = (
            MetricDefinition("same", MetricDirection.MINIMIZE, 1.0),
            MetricDefinition("same", MetricDirection.MAXIMIZE, 1.0),
        )
        with self.assertRaises(ValueError):
            BehaviorFingerprint("a", duplicate, (1.0, 2.0))
        with self.assertRaises(TypeError):
            MetricDefinition("x", "minimize", 1.0)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            MetricDefinition("x", MetricDirection.MINIMIZE, float("inf"))

        inconsistent = (
            self.record("a", (1.0, 1.0)),
            self.record(
                "b",
                (2.0, 2.0),
            ),
        )
        altered = inconsistent[1].fingerprint
        object.__setattr__(
            altered,
            "metrics",
            (
                MetricDefinition("different", MetricDirection.MINIMIZE, 1.0),
                self.metrics[1],
            ),
        )
        from v2.analysis.action_screening import pareto_front

        with self.assertRaisesRegex(ValueError, "schema"):
            pareto_front(inconsistent)

    def test_hard_gates_are_explicit_and_run_before_behavior_selection(self) -> None:
        from v2.analysis.action_screening import (
            apply_hard_gates,
            feasibility_gate,
            solver_reproducibility_gate,
        )

        records = (
            self.record("ok", (3.0, 3.0)),
            self.record("infeasible", (0.0, 0.0), feasible=False),
            self.record("unstable", (0.0, 0.0), reproducible=False),
        )
        self.assertEqual(
            tuple(record.candidate_id for record in apply_hard_gates(records)),
            ("ok",),
        )
        self.assertEqual(
            tuple(record.candidate_id for record in feasibility_gate(records)),
            ("ok", "unstable"),
        )
        self.assertEqual(
            tuple(
                record.candidate_id
                for record in solver_reproducibility_gate(records)
            ),
            ("infeasible", "ok"),
        )

    def test_pareto_direction_and_order_are_deterministic(self) -> None:
        from v2.analysis.action_screening import pareto_front

        records = (
            self.record("c", (2.0, 4.0)),
            self.record("a", (1.0, 3.0)),
            self.record("d", (3.0, 3.0)),
            self.record("b", (2.0, 2.0)),
        )
        expected = ("a", "b")
        self.assertEqual(tuple(item.candidate_id for item in pareto_front(records)), expected)
        self.assertEqual(
            tuple(item.candidate_id for item in pareto_front(tuple(reversed(records)))),
            expected,
        )

    def test_near_duplicate_removal_uses_explicit_threshold_and_stable_ties(self) -> None:
        from v2.analysis.action_screening import remove_near_duplicates

        records = (
            self.record("b", (0.03, 0.04)),
            self.record("c", (2.0, 2.0)),
            self.record("a", (0.0, 0.0)),
        )
        self.assertEqual(
            tuple(
                item.candidate_id
                for item in remove_near_duplicates(
                    records, 0.05, threshold_provenance=self.train
                )
            ),
            ("a", "c"),
        )
        self.assertEqual(
            tuple(
                item.candidate_id
                for item in remove_near_duplicates(
                    tuple(reversed(records)),
                    0.05,
                    threshold_provenance=self.train,
                )
            ),
            ("a", "c"),
        )
        for bad in (True, -1.0, float("nan"), "0.1"):
            with self.subTest(bad=bad), self.assertRaises((TypeError, ValueError)):
                remove_near_duplicates(
                    records, bad, threshold_provenance=self.train
                )  # type: ignore[arg-type]

        from v2.analysis.action_screening import DataSplit, DatasetProvenance

        held_out = DatasetProvenance("synthetic", "held", DataSplit.VALIDATION)
        with self.assertRaisesRegex(PermissionError, "Train-only"):
            remove_near_duplicates(
                records, 0.05, threshold_provenance=held_out
            )

    def test_threshold_clustering_and_medoids_are_deterministic(self) -> None:
        from v2.analysis.action_screening import (
            cluster_by_distance,
            select_cluster_medoids,
        )

        records = (
            self.record("c", (3.0, 0.0)),
            self.record("b", (0.2, 0.0)),
            self.record("a", (0.0, 0.0)),
        )
        clusters = cluster_by_distance(
            records,
            distance_threshold=0.21,
            threshold_provenance=self.train,
        )
        self.assertEqual(clusters, (("a", "b"), ("c",)))
        self.assertEqual(
            tuple(
                item.candidate_id
                for item in select_cluster_medoids(
                    records, clusters, cluster_provenance=self.train
                )
            ),
            ("a", "c"),
        )
        self.assertEqual(
            cluster_by_distance(
                tuple(reversed(records)),
                distance_threshold=0.21,
                threshold_provenance=self.train,
            ),
            clusters,
        )

    def test_catalog_finalization_requires_complete_train_data_and_solver_audit(self) -> None:
        from v2.analysis.action_screening import (
            CatalogFinalizationError,
            DataReadinessEvidence,
            SolverReproducibilityAudit,
            finalize_action_catalog,
        )
        from v2.dqn.action_space import CANDIDATE_ACTION_BANK

        records = tuple(
            self.record(action.action_id, (float(index), float(36 - index)))
            for index, action in enumerate(CANDIDATE_ACTION_BANK)
        )
        candidate_ids = tuple(action.action_id for action in CANDIDATE_ACTION_BANK)
        ready = DataReadinessEvidence(self.train, True, "synthetic-complete-data-audit")
        passed = SolverReproducibilityAudit(
            self.train,
            True,
            candidate_ids,
            2,
            "synthetic-solver-audit",
        )

        result = finalize_action_catalog(
            CANDIDATE_ACTION_BANK,
            records,
            selected_candidate_ids=(candidate_ids[3], candidate_ids[1]),
            selection_provenance=self.train,
            data_readiness=ready,
            solver_audit=passed,
        )
        self.assertEqual(
            tuple(action.action_id for action in result),
            (candidate_ids[1], candidate_ids[3]),
        )

        attacks = (
            {"records": records[:-1]},
            {
                "candidates": CANDIDATE_ACTION_BANK[:-1],
                "records": records[:-1],
                "solver_audit": SolverReproducibilityAudit(
                    self.train,
                    True,
                    candidate_ids[:-1],
                    2,
                    "synthetic-incomplete-bank-audit",
                ),
            },
            {
                "solver_audit": SolverReproducibilityAudit(
                    self.train,
                    False,
                    candidate_ids,
                    2,
                    "failed audit",
                )
            },
            {
                "data_readiness": DataReadinessEvidence(
                    self.train, False, "current raw-data NO-GO"
                )
            },
        )
        for override in attacks:
            kwargs = {
                "candidates": CANDIDATE_ACTION_BANK,
                "records": records,
                "selected_candidate_ids": (candidate_ids[1],),
                "selection_provenance": self.train,
                "data_readiness": ready,
                "solver_audit": passed,
            }
            kwargs.update(override)
            with self.subTest(override=override), self.assertRaises(CatalogFinalizationError):
                finalize_action_catalog(**kwargs)

    def test_finalization_rejects_held_out_evidence_before_selection(self) -> None:
        from v2.analysis.action_screening import (
            DataReadinessEvidence,
            DataSplit,
            DatasetProvenance,
            HeldOutSelectionError,
            SolverReproducibilityAudit,
            finalize_action_catalog,
        )
        from v2.dqn.action_space import CANDIDATE_ACTION_BANK

        test_provenance = DatasetProvenance("synthetic", "held", DataSplit.TEST)
        ids = tuple(action.action_id for action in CANDIDATE_ACTION_BANK)
        records = tuple(
            self.record(action_id, (float(index), 0.0), provenance=test_provenance)
            for index, action_id in enumerate(ids)
        )
        with self.assertRaises(HeldOutSelectionError):
            finalize_action_catalog(
                CANDIDATE_ACTION_BANK,
                records,
                selected_candidate_ids=(ids[0],),
                selection_provenance=test_provenance,
                data_readiness=DataReadinessEvidence(
                    test_provenance, True, "held-out data"
                ),
                solver_audit=SolverReproducibilityAudit(
                    test_provenance, True, ids, 2, "held-out audit"
                ),
            )

    def test_finalization_can_remove_failed_candidates_but_cannot_select_them(self) -> None:
        from v2.analysis.action_screening import (
            CatalogFinalizationError,
            DataReadinessEvidence,
            SolverReproducibilityAudit,
            finalize_action_catalog,
        )
        from v2.dqn.action_space import CANDIDATE_ACTION_BANK

        ids = tuple(action.action_id for action in CANDIDATE_ACTION_BANK)
        records = tuple(
            self.record(
                action_id,
                (float(index), 0.0),
                feasible=index != 0,
            )
            for index, action_id in enumerate(ids)
        )
        readiness = DataReadinessEvidence(self.train, True, "complete-data")
        audit = SolverReproducibilityAudit(
            self.train, True, ids, 2, "complete-solver-audit"
        )
        result = finalize_action_catalog(
            CANDIDATE_ACTION_BANK,
            records,
            selected_candidate_ids=(ids[1],),
            selection_provenance=self.train,
            data_readiness=readiness,
            solver_audit=audit,
        )
        self.assertEqual(tuple(item.action_id for item in result), (ids[1],))
        with self.assertRaises(CatalogFinalizationError):
            finalize_action_catalog(
                CANDIDATE_ACTION_BANK,
                records,
                selected_candidate_ids=(ids[0],),
                selection_provenance=self.train,
                data_readiness=readiness,
                solver_audit=audit,
            )


if __name__ == "__main__":
    unittest.main()
