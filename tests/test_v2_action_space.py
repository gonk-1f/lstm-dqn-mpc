from __future__ import annotations

from dataclasses import FrozenInstanceError
import inspect
import sys
import unittest
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

        self.assertEqual(ACTION_TABLE_VERSION, CONTRACT_VERSION)
        self.assertEqual(ACTION_TABLE_VERSION, "three_weight_simplex_behavior_filtered_v1")
        self.assertEqual(first, generate_candidate_action_bank())
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
            fingerprint_provenance: object | None = None,
        ) -> object:
            fingerprint_source = (
                provenance if fingerprint_provenance is None else fingerprint_provenance
            )
            return CandidateScreeningRecord(
                candidate_id,
                FeasibilityResult(
                    candidate_id, provenance, feasible, "synthetic feasibility"
                ),
                SolverReproducibilityResult(
                    candidate_id,
                    provenance,
                    reproducible,
                    2,
                    "synthetic repeated solve",
                ),
                BehaviorFingerprint(
                    candidate_id, fingerprint_source, self.metrics, values
                ),
            )

        self.record = record

    def pipeline(self, records: object) -> object:
        from v2.analysis.action_screening import (
            DistanceThresholdRule,
            apply_hard_gates,
            cluster_by_distance,
            derive_distance_threshold,
            pareto_front,
            remove_near_duplicates,
            select_cluster_medoids,
        )

        hard = apply_hard_gates(records, audit_id="hard-gate-audit")
        pareto = pareto_front(hard, audit_id="pareto-audit")
        duplicate_threshold = derive_distance_threshold(
            pareto, rule=DistanceThresholdRule.ZERO, audit_id="duplicate-threshold"
        )
        unique = remove_near_duplicates(
            pareto, duplicate_threshold, audit_id="duplicate-audit"
        )
        cluster_threshold = derive_distance_threshold(
            unique, rule=DistanceThresholdRule.ZERO, audit_id="cluster-threshold"
        )
        clusters = cluster_by_distance(
            unique, cluster_threshold, audit_id="cluster-audit"
        )
        return select_cluster_medoids(clusters, audit_id="medoid-audit")

    def test_train_provenance_is_exact_and_fingerprint_cannot_be_laundered(self) -> None:
        from v2.analysis.action_screening import (
            BehaviorFingerprint,
            CandidateScreeningRecord,
            DataSplit,
            DatasetProvenance,
            HeldOutSelectionError,
            apply_hard_gates,
        )

        for split in ("Train", "train", True, 1, None):
            with self.subTest(split=split), self.assertRaises(TypeError):
                DatasetProvenance("v", "p", split)  # type: ignore[arg-type]

        validation = DatasetProvenance("synthetic", "validation", DataSplit.VALIDATION)
        train_gates = self.record("a", (1.0, 1.0))
        held_out_fingerprint = BehaviorFingerprint(
            "a", validation, self.metrics, (1.0, 1.0)
        )
        with self.assertRaises(HeldOutSelectionError):
            CandidateScreeningRecord(
                "a",
                train_gates.feasibility,
                train_gates.reproducibility,
                held_out_fingerprint,
            )

        for split in (DataSplit.VALIDATION, DataSplit.TEST, DataSplit.UNKNOWN):
            provenance = DatasetProvenance("synthetic", "held", split)
            with self.subTest(split=split), self.assertRaises(HeldOutSelectionError):
                apply_hard_gates(
                    (self.record("a", (1.0, 1.0), provenance=provenance),),
                    audit_id="forbidden",
                )

        other_train = DatasetProvenance("synthetic", "different", DataSplit.TRAIN)
        with self.assertRaisesRegex(ValueError, "same.*provenance"):
            apply_hard_gates(
                (
                    self.record("a", (1.0, 1.0)),
                    self.record("b", (2.0, 2.0), provenance=other_train),
                ),
                audit_id="mixed",
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

    def test_fingerprints_are_immutable_finite_schema_safe_and_normalizable(self) -> None:
        from v2.analysis.action_screening import (
            BehaviorFingerprint,
            MetricDefinition,
            MetricDirection,
        )

        source = np.asarray([1.0, 2.0])
        fingerprint = BehaviorFingerprint("a", self.train, self.metrics, source)
        source[:] = 99.0
        self.assertEqual(fingerprint.values, (1.0, 2.0))
        self.assertEqual(fingerprint.provenance, self.train)
        self.assertIs(type(fingerprint.values), tuple)
        with self.assertRaises(FrozenInstanceError):
            fingerprint.values = (3.0, 4.0)  # type: ignore[misc]

        with self.assertRaises(ValueError):
            BehaviorFingerprint("a", self.train, self.metrics, (float("nan"), 2.0))
        with self.assertRaises(ValueError):
            BehaviorFingerprint("a", self.train, self.metrics, (1.0,))
        duplicate = (
            MetricDefinition("same", MetricDirection.MINIMIZE, 1.0),
            MetricDefinition("same", MetricDirection.MAXIMIZE, 1.0),
        )
        with self.assertRaises(ValueError):
            BehaviorFingerprint("a", self.train, duplicate, (1.0, 2.0))
        with self.assertRaises(TypeError):
            MetricDefinition("x", "minimize", 1.0)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            MetricDefinition("x", MetricDirection.MINIMIZE, float("inf"))

        overflow_schema = (
            MetricDefinition("tiny_scale", MetricDirection.MINIMIZE, 1e-308),
        )
        with self.assertRaisesRegex(ValueError, "normalized"):
            BehaviorFingerprint("a", self.train, overflow_schema, (1e308,))

    def test_extreme_finite_normalized_values_do_not_crash_distance_algorithms(self) -> None:
        from v2.analysis.action_screening import (
            BehaviorFingerprint,
            CandidateScreeningRecord,
            DistanceThresholdRule,
            FeasibilityResult,
            MetricDefinition,
            MetricDirection,
            SolverReproducibilityResult,
            apply_hard_gates,
            cluster_by_distance,
            derive_distance_threshold,
            pareto_front,
            remove_near_duplicates,
            select_cluster_medoids,
        )

        schema = (
            MetricDefinition("left", MetricDirection.MINIMIZE, 1.0),
            MetricDefinition("right", MetricDirection.MINIMIZE, 1.0),
        )

        def extreme(candidate_id: str, values: tuple[float, float]) -> object:
            return CandidateScreeningRecord(
                candidate_id,
                FeasibilityResult(candidate_id, self.train, True, "ok"),
                SolverReproducibilityResult(candidate_id, self.train, True, 2, "ok"),
                BehaviorFingerprint(candidate_id, self.train, schema, values),
            )

        hard = apply_hard_gates(
            (
                extreme("a", (-1e200, 1e200)),
                extreme("b", (1e200, -1e200)),
            ),
            audit_id="hard",
        )
        pareto = pareto_front(hard, audit_id="pareto")
        near_threshold = derive_distance_threshold(
            pareto, rule=DistanceThresholdRule.ZERO, audit_id="near-threshold"
        )
        unique = remove_near_duplicates(
            pareto, near_threshold, audit_id="near"
        )
        cluster_threshold = derive_distance_threshold(
            unique,
            rule=DistanceThresholdRule.MIN_POSITIVE_PAIRWISE,
            audit_id="cluster-threshold",
        )
        clusters = cluster_by_distance(unique, cluster_threshold, audit_id="cluster")
        selected = select_cluster_medoids(clusters, audit_id="medoid")
        self.assertEqual(tuple(item.candidate_id for item in selected.records), ("a",))

        overflow_hard = apply_hard_gates(
            (
                extreme("a", (-1e308, 1e308)),
                extreme("b", (1e308, -1e308)),
            ),
            audit_id="overflow-hard",
        )
        overflow_pareto = pareto_front(overflow_hard, audit_id="overflow-pareto")
        overflow_near_threshold = derive_distance_threshold(
            overflow_pareto,
            rule=DistanceThresholdRule.ZERO,
            audit_id="overflow-near-threshold",
        )
        overflow_unique = remove_near_duplicates(
            overflow_pareto, overflow_near_threshold, audit_id="overflow-near"
        )
        overflow_cluster_threshold = derive_distance_threshold(
            overflow_unique,
            rule=DistanceThresholdRule.MIN_POSITIVE_PAIRWISE,
            audit_id="overflow-cluster-threshold",
        )
        overflow_clusters = cluster_by_distance(
            overflow_unique,
            overflow_cluster_threshold,
            audit_id="overflow-cluster",
        )
        self.assertEqual(overflow_clusters.assignments, (("a",), ("b",)))

    def test_hard_gate_result_is_first_class_sealed_and_complete(self) -> None:
        from v2.analysis.action_screening import HardGateResult, apply_hard_gates

        records = (
            self.record("ok", (3.0, 3.0)),
            self.record("infeasible", (0.0, 0.0), feasible=False),
            self.record("unstable", (0.0, 0.0), reproducible=False),
        )
        result = apply_hard_gates(records, audit_id="gate-audit")
        self.assertIs(type(result), HardGateResult)
        self.assertEqual(result.provenance, self.train)
        self.assertEqual(result.audit_id, "gate-audit")
        self.assertEqual(len(result.source_records), 3)
        self.assertEqual(tuple(item.candidate_id for item in result.records), ("ok",))
        with self.assertRaises(TypeError):
            HardGateResult()  # type: ignore[call-arg]

    def test_medoid_scoring_resists_finite_distance_sum_overflow(self) -> None:
        from v2.analysis.action_screening import (
            BehaviorFingerprint,
            CandidateScreeningRecord,
            DistanceThresholdRule,
            FeasibilityResult,
            MetricDefinition,
            MetricDirection,
            SolverReproducibilityResult,
            apply_hard_gates,
            cluster_by_distance,
            derive_distance_threshold,
            pareto_front,
            remove_near_duplicates,
            select_cluster_medoids,
        )

        schema = (
            MetricDefinition("x", MetricDirection.MINIMIZE, 1.0),
            MetricDefinition("y", MetricDirection.MINIMIZE, 2.0),
        )

        def point(candidate_id: str, values: tuple[float, float]) -> object:
            return CandidateScreeningRecord(
                candidate_id,
                FeasibilityResult(candidate_id, self.train, True, "ok"),
                SolverReproducibilityResult(candidate_id, self.train, True, 2, "ok"),
                BehaviorFingerprint(candidate_id, self.train, schema, values),
            )

        hard = apply_hard_gates(
            (
                point("a", (-1e308, 1e308)),
                point("m", (0.0, 0.0)),
                point("z", (1e308, -1e308)),
            ),
            audit_id="hard",
        )
        pareto = pareto_front(hard, audit_id="pareto")
        zero = derive_distance_threshold(
            pareto, rule=DistanceThresholdRule.ZERO, audit_id="zero"
        )
        unique = remove_near_duplicates(pareto, zero, audit_id="near")
        median = derive_distance_threshold(
            unique, rule=DistanceThresholdRule.MEDIAN_PAIRWISE, audit_id="median"
        )
        clusters = cluster_by_distance(unique, median, audit_id="cluster")
        self.assertEqual(clusters.assignments, (("a", "m", "z"),))
        selected = select_cluster_medoids(clusters, audit_id="medoid")
        self.assertEqual(tuple(item.candidate_id for item in selected.records), ("m",))

    def test_pareto_result_has_direction_and_deterministic_parent_lineage(self) -> None:
        from v2.analysis.action_screening import apply_hard_gates, pareto_front

        records = (
            self.record("c", (2.0, 4.0)),
            self.record("a", (1.0, 3.0)),
            self.record("d", (3.0, 3.0)),
            self.record("b", (2.0, 2.0)),
        )
        forward = pareto_front(
            apply_hard_gates(records, audit_id="hard"), audit_id="pareto"
        )
        reverse = pareto_front(
            apply_hard_gates(tuple(reversed(records)), audit_id="hard"),
            audit_id="pareto",
        )
        self.assertEqual(tuple(item.candidate_id for item in forward.records), ("a", "b"))
        self.assertEqual(forward.digest, reverse.digest)
        with self.assertRaises(TypeError):
            pareto_front(records, audit_id="raw-records")  # type: ignore[arg-type]

    def test_near_duplicate_result_embeds_threshold_without_detached_label(self) -> None:
        from v2.analysis.action_screening import (
            DistanceThresholdEvidence,
            DistanceThresholdRule,
            apply_hard_gates,
            derive_distance_threshold,
            pareto_front,
            remove_near_duplicates,
        )

        records = (
            self.record("b", (0.03, 1.94)),
            self.record("c", (2.0, 0.0)),
            self.record("a", (0.0, 2.0)),
        )
        parent = pareto_front(
            apply_hard_gates(records, audit_id="hard"), audit_id="pareto"
        )
        threshold = derive_distance_threshold(
            parent,
            rule=DistanceThresholdRule.MIN_POSITIVE_PAIRWISE,
            audit_id="near-threshold",
        )
        first = remove_near_duplicates(parent, threshold, audit_id="near")
        reverse_parent = pareto_front(
            apply_hard_gates(tuple(reversed(records)), audit_id="hard"),
            audit_id="pareto",
        )
        reverse_threshold = derive_distance_threshold(
            reverse_parent,
            rule=DistanceThresholdRule.MIN_POSITIVE_PAIRWISE,
            audit_id="near-threshold",
        )
        second = remove_near_duplicates(
            reverse_parent, reverse_threshold, audit_id="near"
        )
        self.assertIs(type(threshold), DistanceThresholdEvidence)
        self.assertEqual(threshold.rule, DistanceThresholdRule.MIN_POSITIVE_PAIRWISE)
        self.assertGreater(threshold.value, 0.0)
        self.assertEqual(first.threshold, threshold.value)
        self.assertEqual(tuple(item.candidate_id for item in first.records), ("a", "c"))
        self.assertEqual(first.digest, second.digest)
        median = derive_distance_threshold(
            parent,
            rule=DistanceThresholdRule.MEDIAN_PAIRWISE,
            audit_id="median-threshold",
        )
        self.assertGreaterEqual(median.value, threshold.value)
        self.assertNotIn(
            "distance_threshold", inspect.signature(remove_near_duplicates).parameters
        )
        with self.assertRaises(TypeError):
            remove_near_duplicates(  # type: ignore[arg-type]
                parent, 0.05, audit_id="raw-threshold-laundering"
            )
        for bad in (True, "ZERO", 0):
            with self.subTest(bad=bad), self.assertRaises((TypeError, ValueError)):
                derive_distance_threshold(
                    parent, rule=bad, audit_id="bad"
                )  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            DistanceThresholdEvidence()  # type: ignore[call-arg]

    def test_clustering_and_medoids_only_consume_sealed_parent_results(self) -> None:
        from v2.analysis.action_screening import (
            ClusteringResult,
            DistanceThresholdRule,
            MedoidSelectionResult,
            apply_hard_gates,
            cluster_by_distance,
            derive_distance_threshold,
            pareto_front,
            remove_near_duplicates,
            select_cluster_medoids,
        )

        records = (
            self.record("c", (3.0, 0.0)),
            self.record("b", (0.2, 5.8)),
            self.record("a", (0.0, 6.0)),
        )
        pareto = pareto_front(
            apply_hard_gates(records, audit_id="hard"), audit_id="pareto"
        )
        near_threshold = derive_distance_threshold(
            pareto, rule=DistanceThresholdRule.ZERO, audit_id="near-threshold"
        )
        parent = remove_near_duplicates(
            pareto, near_threshold, audit_id="near"
        )
        cluster_threshold = derive_distance_threshold(
            parent,
            rule=DistanceThresholdRule.MIN_POSITIVE_PAIRWISE,
            audit_id="cluster-threshold",
        )
        clusters = cluster_by_distance(parent, cluster_threshold, audit_id="cluster")
        self.assertIs(type(clusters), ClusteringResult)
        self.assertEqual(clusters.assignments, (("a", "b"), ("c",)))
        medoids = select_cluster_medoids(clusters, audit_id="medoid")
        self.assertIs(type(medoids), MedoidSelectionResult)
        self.assertEqual(tuple(item.candidate_id for item in medoids.records), ("a", "c"))
        self.assertNotIn(
            "distance_threshold", inspect.signature(cluster_by_distance).parameters
        )
        with self.assertRaises(TypeError):
            cluster_by_distance(parent, 1000.0, audit_id="raw-threshold")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            select_cluster_medoids(  # type: ignore[arg-type]
                (("a", "b"), ("c",)), audit_id="arbitrary-clusters"
            )
        with self.assertRaises(TypeError):
            ClusteringResult()  # type: ignore[call-arg]
        with self.assertRaises(TypeError):
            MedoidSelectionResult()  # type: ignore[call-arg]

    def test_mutated_or_forged_stage_lineage_is_rejected(self) -> None:
        from v2.analysis.action_screening import (
            DataSplit,
            DatasetProvenance,
            DistanceThresholdRule,
            ScreeningLineageError,
            apply_hard_gates,
            cluster_by_distance,
            derive_distance_threshold,
            pareto_front,
            remove_near_duplicates,
        )

        hard = apply_hard_gates(
            (self.record("a", (0.0, 0.0)),), audit_id="hard"
        )
        pareto = pareto_front(hard, audit_id="pareto")
        threshold = derive_distance_threshold(
            pareto, rule=DistanceThresholdRule.ZERO, audit_id="threshold"
        )
        object.__setattr__(threshold, "value", 999.0)
        with self.assertRaises(ScreeningLineageError):
            remove_near_duplicates(pareto, threshold, audit_id="near")

        clean_threshold = derive_distance_threshold(
            pareto, rule=DistanceThresholdRule.ZERO, audit_id="clean-threshold"
        )
        other_pareto = pareto_front(
            apply_hard_gates(
                (self.record("other", (2.0, 2.0)),), audit_id="other-hard"
            ),
            audit_id="other-pareto",
        )
        wrong_parent_threshold = derive_distance_threshold(
            other_pareto,
            rule=DistanceThresholdRule.ZERO,
            audit_id="wrong-parent-threshold",
        )
        with self.assertRaises(ScreeningLineageError):
            remove_near_duplicates(
                pareto, wrong_parent_threshold, audit_id="wrong-parent"
            )

        near = remove_near_duplicates(pareto, clean_threshold, audit_id="near")
        cluster_threshold = derive_distance_threshold(
            near, rule=DistanceThresholdRule.ZERO, audit_id="cluster-threshold"
        )
        object.__setattr__(cluster_threshold, "audit_id", "mutated-audit")
        with self.assertRaises(ScreeningLineageError):
            cluster_by_distance(near, cluster_threshold, audit_id="cluster")

        tainted = self.record("tainted", (1.0, 1.0))
        tainted_hard = apply_hard_gates((tainted,), audit_id="hard")
        object.__setattr__(
            tainted.fingerprint,
            "provenance",
            DatasetProvenance("synthetic", "held", DataSplit.TEST),
        )
        with self.assertRaises(ScreeningLineageError):
            pareto_front(tainted_hard, audit_id="must-detect-nested-mutation")

    def test_catalog_finalization_consumes_only_complete_pipeline_result(self) -> None:
        from v2.analysis.action_screening import (
            CatalogFinalizationError,
            DataReadinessEvidence,
            ScreeningLineageError,
            SolverReproducibilityAudit,
            finalize_action_catalog,
        )
        from v2.dqn.action_space import CANDIDATE_ACTION_BANK

        records = tuple(
            self.record(action.action_id, (float(index), float(36 - index)))
            for index, action in enumerate(CANDIDATE_ACTION_BANK)
        )
        ids = tuple(action.action_id for action in CANDIDATE_ACTION_BANK)
        pipeline = self.pipeline(records)
        readiness = DataReadinessEvidence(
            self.train, True, "complete-data", "all Train rows loaded"
        )
        audit = SolverReproducibilityAudit(
            self.train,
            True,
            ids,
            2,
            "complete-solver-audit",
            "all candidates repeated",
        )
        result = finalize_action_catalog(
            CANDIDATE_ACTION_BANK,
            pipeline,
            data_readiness=readiness,
            solver_audit=audit,
        )
        self.assertEqual(
            tuple(item.action_id for item in result),
            tuple(item.candidate_id for item in pipeline.records),
        )
        self.assertNotIn(
            "selected_candidate_ids", inspect.signature(finalize_action_catalog).parameters
        )
        with self.assertRaises(TypeError):
            finalize_action_catalog(  # type: ignore[call-arg]
                CANDIDATE_ACTION_BANK,
                records,
                selected_candidate_ids=(ids[0],),
                selection_provenance=self.train,
                data_readiness=readiness,
                solver_audit=audit,
            )

        incomplete = self.pipeline(records[:-1])
        with self.assertRaisesRegex(CatalogFinalizationError, "complete"):
            finalize_action_catalog(
                CANDIDATE_ACTION_BANK,
                incomplete,
                data_readiness=readiness,
                solver_audit=audit,
            )
        failed_audit = SolverReproducibilityAudit(
            self.train, False, ids, 2, "failed-solver-audit", "solver mismatch"
        )
        with self.assertRaises(CatalogFinalizationError):
            finalize_action_catalog(
                CANDIDATE_ACTION_BANK,
                pipeline,
                data_readiness=readiness,
                solver_audit=failed_audit,
            )
        with self.assertRaises(CatalogFinalizationError):
            finalize_action_catalog(
                CANDIDATE_ACTION_BANK,
                pipeline,
                data_readiness=DataReadinessEvidence(
                    self.train, False, "current-NO-GO", "raw data unavailable"
                ),
                solver_audit=audit,
            )

        tampered_readiness = DataReadinessEvidence(
            self.train, False, "tampered-data", "data audit failed"
        )
        object.__setattr__(tampered_readiness, "passed", True)
        with self.assertRaises(ScreeningLineageError):
            finalize_action_catalog(
                CANDIDATE_ACTION_BANK,
                pipeline,
                data_readiness=tampered_readiness,
                solver_audit=audit,
            )

        tampered_ids = SolverReproducibilityAudit(
            self.train,
            True,
            ids[:-1],
            2,
            "tampered-ids",
            "one candidate missing",
        )
        object.__setattr__(tampered_ids, "candidate_ids", ids)
        with self.assertRaises(ScreeningLineageError):
            finalize_action_catalog(
                CANDIDATE_ACTION_BANK,
                pipeline,
                data_readiness=readiness,
                solver_audit=tampered_ids,
            )

        tampered_pass = SolverReproducibilityAudit(
            self.train,
            False,
            ids,
            2,
            "tampered-pass",
            "audit failed",
        )
        object.__setattr__(tampered_pass, "passed", True)
        with self.assertRaises(ScreeningLineageError):
            finalize_action_catalog(
                CANDIDATE_ACTION_BANK,
                pipeline,
                data_readiness=readiness,
                solver_audit=tampered_pass,
            )

    def test_failed_candidates_may_be_removed_but_never_selected(self) -> None:
        from v2.analysis.action_screening import (
            DataReadinessEvidence,
            SolverReproducibilityAudit,
            finalize_action_catalog,
        )
        from v2.dqn.action_space import CANDIDATE_ACTION_BANK

        ids = tuple(action.action_id for action in CANDIDATE_ACTION_BANK)
        records = tuple(
            self.record(
                action_id,
                (float(index), float(36 - index)),
                feasible=index != 0,
            )
            for index, action_id in enumerate(ids)
        )
        pipeline = self.pipeline(records)
        result = finalize_action_catalog(
            CANDIDATE_ACTION_BANK,
            pipeline,
            data_readiness=DataReadinessEvidence(
                self.train, True, "data", "all Train rows loaded"
            ),
            solver_audit=SolverReproducibilityAudit(
                self.train, True, ids, 2, "solver", "all candidates repeated"
            ),
        )
        self.assertNotIn(ids[0], tuple(item.action_id for item in result))

    def test_finalization_revalidates_fresh_canonical_bank_after_mutation(self) -> None:
        from types import SimpleNamespace

        import v2.dqn.action_space as action_space
        from v2.analysis.action_screening import (
            CatalogFinalizationError,
            DataReadinessEvidence,
            SolverReproducibilityAudit,
            finalize_action_catalog,
        )

        local_bank = action_space.generate_candidate_action_bank()
        object.__setattr__(local_bank[0], "n_soc", 9)
        with self.assertRaises(CatalogFinalizationError):
            finalize_action_catalog(  # lineage type is rejected after bank validation
                local_bank,
                object(),  # type: ignore[arg-type]
                data_readiness=object(),  # type: ignore[arg-type]
                solver_audit=object(),  # type: ignore[arg-type]
            )

        global_bank = action_space.CANDIDATE_ACTION_BANK
        first = global_bank[0]
        original_n_base = first.n_base
        try:
            object.__setattr__(first, "n_base", 9)
            corrupted_ids = tuple(action.action_id for action in global_bank)
            records = tuple(
                self.record(action_id, (float(index), float(36 - index)))
                for index, action_id in enumerate(corrupted_ids)
            )
            pipeline = self.pipeline(records)
            with self.assertRaises(CatalogFinalizationError):
                finalize_action_catalog(
                    global_bank,
                    pipeline,
                    data_readiness=DataReadinessEvidence(
                        self.train, True, "data", "all Train rows loaded"
                    ),
                    solver_audit=SolverReproducibilityAudit(
                        self.train,
                        True,
                        corrupted_ids,
                        2,
                        "solver",
                        "all candidates repeated",
                    ),
                )
        finally:
            object.__setattr__(first, "n_base", original_n_base)

        canonical = action_space.generate_candidate_action_bank()
        ids = tuple(action.action_id for action in canonical)
        pipeline = self.pipeline(
            tuple(
                self.record(action_id, (float(index), float(36 - index)))
                for index, action_id in enumerate(ids)
            )
        )
        poisoned = action_space.generate_candidate_action_bank()
        object.__setattr__(
            poisoned[0],
            "to_mpc_weights",
            lambda: SimpleNamespace(q_base=0.1, q_smooth=0.1, q_soc=0.8),
        )
        with self.assertRaises(CatalogFinalizationError):
            finalize_action_catalog(
                poisoned,
                pipeline,
                data_readiness=DataReadinessEvidence(
                    self.train, True, "data", "all Train rows loaded"
                ),
                solver_audit=SolverReproducibilityAudit(
                    self.train,
                    True,
                    ids,
                    2,
                    "solver",
                    "all candidates repeated",
                ),
            )
        result = finalize_action_catalog(
            canonical,
            pipeline,
            data_readiness=DataReadinessEvidence(
                self.train, True, "data", "all Train rows loaded"
            ),
            solver_audit=SolverReproducibilityAudit(
                self.train,
                True,
                ids,
                2,
                "solver",
                "all candidates repeated",
            ),
        )
        from v2.control.nonlinear_mpc import MPCWeights

        for action in result:
            weights = action.to_mpc_weights()
            self.assertIs(type(weights), MPCWeights)
            self.assertAlmostEqual(
                weights.q_base + weights.q_smooth + weights.q_soc, 1.0
            )

    def test_held_out_records_cannot_reach_any_derived_stage(self) -> None:
        from v2.analysis.action_screening import (
            DataSplit,
            DatasetProvenance,
            HeldOutSelectionError,
            apply_hard_gates,
        )
        from v2.dqn.action_space import CANDIDATE_ACTION_BANK

        held_out = DatasetProvenance("synthetic", "held", DataSplit.TEST)
        with self.assertRaises(HeldOutSelectionError):
            records = tuple(
                self.record(
                    action.action_id,
                    (float(index), 0.0),
                    provenance=held_out,
                )
                for index, action in enumerate(CANDIDATE_ACTION_BANK)
            )
            apply_hard_gates(records, audit_id="held-out")


if __name__ == "__main__":
    unittest.main()
