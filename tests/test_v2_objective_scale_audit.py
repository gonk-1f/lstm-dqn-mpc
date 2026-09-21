from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class ObjectiveScaleAuditTests(unittest.TestCase):
    @staticmethod
    def _provenance(split=None):
        from v2.analysis.action_screening import DataSplit, DatasetProvenance
        from v2.contracts import DATASET_VERSION

        return DatasetProvenance(
            DATASET_VERSION,
            "sha256:synthetic-objective-scale-fixture",
            split if split is not None else DataSplit.TRAIN,
        )

    @staticmethod
    def _cases():
        from v2.analysis.objective_scale_audit import ObjectiveAuditCase

        return (
            ObjectiveAuditCase("state-a", 0, (300.0, 0.5, 250.0)),
            ObjectiveAuditCase("state-b", 1, (450.0, 0.35, 300.0)),
        )

    @staticmethod
    def _plan(action, components, *, case_id):
        from v2.control.nonlinear_mpc import (
            MPCPlan,
            ObjectiveComponents,
            SolverDiagnostics,
            weighted_objective,
        )

        component_record = ObjectiveComponents(*components)
        objective = weighted_objective(component_record, action.to_mpc_weights())
        same_behavior = case_id == "state-a"
        first_fc = 300.0 if same_behavior else 320.0 + action.n_base
        first_batt = 50.0 if same_behavior else 130.0 - action.n_base
        soc = 0.5 if same_behavior else 0.49 + action.n_soc / 1000.0
        return MPCPlan(
            p_fc_kw=(first_fc,) * 5,
            p_batt_bus_kw=(first_batt,) * 5,
            soc_path=(soc,) * 5,
            load_forecast_kw=(first_fc + first_batt,) * 5,
            base_reference_kw=(300.0,) * 5,
            components=component_record,
            objective_value=objective,
            diagnostics=SolverDiagnostics(True, 0, "ok", 3, objective),
        )

    def test_held_out_split_rejects_before_case_loader_or_solver(self) -> None:
        from v2.analysis.action_screening import DataSplit, HeldOutSelectionError
        from v2.analysis.objective_scale_audit import (
            BehaviorTolerance,
            run_objective_scale_audit,
        )
        from v2.dqn.action_space import ActionCandidate

        for split in (DataSplit.VALIDATION, DataSplit.TEST, DataSplit.UNKNOWN):
            events = []

            def loader():
                events.append("loader")
                return self._cases()

            def runner(case, action):
                events.append((case, action))
                raise AssertionError("held-out solver must not run")

            with self.subTest(split=split), self.assertRaises(HeldOutSelectionError):
                run_objective_scale_audit(
                    provenance=self._provenance(split),
                    cases_loader=loader,
                    actions=(ActionCandidate(1, 2, 7),),
                    solver_runner=runner,
                    behavior_tolerance=BehaviorTolerance(1e-12, 1e-9, 1e-12),
                )
            self.assertEqual(events, [])

    def test_statistics_dominance_and_behavioral_redundancy_are_explicit(self) -> None:
        from v2.analysis.objective_scale_audit import (
            BehaviorTolerance,
            ObjectiveScaleStatus,
            run_objective_scale_audit,
        )
        from v2.dqn.action_space import CANDIDATE_ACTION_BANK

        actions = CANDIDATE_ACTION_BANK
        events = []

        def runner(case, action):
            events.append((case.case_id, action.action_id))
            components = (1.0, 1.0, 1.0) if case.case_id == "state-a" else (1.0, 10.0, 0.0)
            return self._plan(action, components, case_id=case.case_id)

        result = run_objective_scale_audit(
            provenance=self._provenance(),
            cases_loader=self._cases,
            actions=actions,
            solver_runner=runner,
            behavior_tolerance=BehaviorTolerance(1e-12, 1e-9, 1e-12),
        )

        self.assertEqual(len(events), 72)
        self.assertEqual(events[0], ("state-a", "w_1_1_8"))
        self.assertEqual(events[-1], ("state-b", "w_8_1_1"))
        by_term = {item.term: item for item in result.objective_statistics}
        self.assertEqual(by_term["base"].count, 72)
        self.assertEqual(by_term["base"].mean, 1.0)
        self.assertEqual(by_term["smooth"].mean, 5.5)
        self.assertEqual(by_term["smooth"].std, 4.5)
        self.assertEqual(by_term["smooth"].p50, 5.5)
        self.assertEqual(by_term["smooth"].p95, 10.0)
        self.assertEqual(result.soc_active.positive_count, 36)
        self.assertEqual(result.soc_active.probability_positive, 0.5)
        self.assertEqual(result.soc_active.p95_positive, 1.0)
        self.assertEqual(result.active_p95, (1.0, 10.0, 1.0))
        self.assertEqual(result.scale_ratio, 10.0)
        self.assertIs(result.status, ObjectiveScaleStatus.NO_GO)
        self.assertEqual(result.recommended_fixed_normalization_constants, (1.0, 10.0, 1.0))

        dominance = {
            (item.dominant_term, item.dominated_term): item
            for item in result.dominance_statistics
        }
        self.assertEqual(dominance[("smooth", "base")].count, 36)
        self.assertEqual(dominance[("smooth", "base")].rate, 0.5)
        self.assertEqual(len(dominance[("smooth", "base")].case_action_ids), 36)
        self.assertEqual(
            dominance[("smooth", "base")].case_action_ids[0],
            "state-b/w_1_1_8",
        )
        self.assertEqual(result.behavioral_redundancy[0].case_id, "state-a")
        self.assertEqual(len(result.behavioral_redundancy[0].action_pairs), 630)
        self.assertEqual(
            result.behavioral_redundancy[0].action_pairs[0],
            ("w_1_1_8", "w_1_2_7"),
        )
        self.assertEqual(result.validate(), result)

    def test_scale_ratio_thresholds_are_exact_project_rules(self) -> None:
        from v2.analysis.objective_scale_audit import (
            ObjectiveScaleStatus,
            classify_objective_scale_ratio,
        )

        self.assertIs(classify_objective_scale_ratio(5.0), ObjectiveScaleStatus.GO)
        self.assertIs(classify_objective_scale_ratio(5.000001), ObjectiveScaleStatus.WARNING)
        self.assertIs(classify_objective_scale_ratio(9.999999), ObjectiveScaleStatus.WARNING)
        self.assertIs(classify_objective_scale_ratio(10.0), ObjectiveScaleStatus.NO_GO)

    def test_train_audit_requires_the_complete_canonical_candidate_bank(self) -> None:
        from v2.analysis.objective_scale_audit import (
            BehaviorTolerance,
            run_objective_scale_audit,
        )
        from v2.dqn.action_space import ActionCandidate

        with self.assertRaises(ValueError):
            run_objective_scale_audit(
                provenance=self._provenance(),
                cases_loader=self._cases,
                actions=(ActionCandidate(1, 2, 7),),
                solver_runner=lambda case, candidate: self._plan(
                    candidate, (1.0, 1.0, 1.0), case_id=case.case_id
                ),
                behavior_tolerance=BehaviorTolerance(1e-12, 1e-9, 1e-12),
            )

    def test_missing_positive_soc_evidence_is_no_go_without_a_ratio(self) -> None:
        from v2.analysis.objective_scale_audit import (
            BehaviorTolerance,
            ObjectiveScaleStatus,
            run_objective_scale_audit,
        )
        from v2.dqn.action_space import CANDIDATE_ACTION_BANK

        result = run_objective_scale_audit(
            provenance=self._provenance(),
            cases_loader=lambda: (self._cases()[0],),
            actions=CANDIDATE_ACTION_BANK,
            solver_runner=lambda case, candidate: self._plan(
                candidate, (1.0, 1.0, 0.0), case_id=case.case_id
            ),
            behavior_tolerance=BehaviorTolerance(1e-12, 1e-9, 1e-12),
        )
        self.assertEqual(result.soc_active.positive_count, 0)
        self.assertIsNone(result.soc_active.p95_positive)
        self.assertIsNone(result.scale_ratio)
        self.assertIs(result.status, ObjectiveScaleStatus.NO_GO)
        self.assertIsNone(result.recommended_fixed_normalization_constants)

    def test_result_is_frozen_and_detects_tampering(self) -> None:
        from v2.analysis.objective_scale_audit import (
            BehaviorTolerance,
            run_objective_scale_audit,
        )
        from v2.dqn.action_space import CANDIDATE_ACTION_BANK

        result = run_objective_scale_audit(
            provenance=self._provenance(),
            cases_loader=lambda: (self._cases()[0],),
            actions=CANDIDATE_ACTION_BANK,
            solver_runner=lambda case, candidate: self._plan(
                candidate, (1.0, 1.0, 1.0), case_id=case.case_id
            ),
            behavior_tolerance=BehaviorTolerance(1e-12, 1e-9, 1e-12),
        )
        with self.assertRaises(FrozenInstanceError):
            result.scale_ratio = 2.0  # type: ignore[misc]
        object.__setattr__(result, "scale_ratio", 2.0)
        with self.assertRaises(ValueError):
            result.validate()
        object.__setattr__(result, "scale_ratio", 1.0)
        object.__setattr__(result.objective_statistics[0], "injected", True)
        with self.assertRaises(ValueError):
            result.validate()


if __name__ == "__main__":
    unittest.main()
