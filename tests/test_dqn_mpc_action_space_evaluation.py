from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
MAIN_ROOT = SRC / "main"

for path in (SRC, MAIN_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


from dqn.utils.action_mapper import (  # noqa: E402
    DQN_MPC_WEIGHT_ACTIONS,
    MPCWeightAction,
)
from envs.dqn_mpc_weight_env import DqnMpcWeightEnv  # noqa: E402
from evaluate_dqn_mpc_action_space import (  # noqa: E402
    RepresentativeState,
    build_representative_states,
    classify_solver_status,
    evaluate_action_space,
    evaluate_fixed_action_coverage,
    evaluate_state_probes,
    summarize_action_space,
    write_action_space_evaluation,
)
from mpc_solvers.dqn_mpc_solver_bank import (  # noqa: E402
    MpcWeightSolverBank,
)
from train_dqn_mpc_mlp import (  # noqa: E402
    DEFAULT_SPLIT_JSON,
    FORMAL_DATA_DIRECTORY,
    FORMAL_SAMPLE_INTERVAL_SECONDS,
    FORMAL_TARGET_LOAD,
    VoyageSplit,
    build_formal_mpc_config,
)


CUSTOM_ACTIONS = (
    MPCWeightAction(
        0,
        0.25,
        0.40,
        12.0,
        20.0,
        "candidate_C",
    ),
    MPCWeightAction(
        1,
        0.45,
        0.25,
        8.0,
        8.0,
        "moderate_hydrogen_economy",
    ),
)


class TestDqnMpcActionSpaceEvaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = build_formal_mpc_config()

    def test_custom_actions_are_injected_without_mutating_global_table(
        self,
    ) -> None:
        original_table = DQN_MPC_WEIGHT_ACTIONS
        original_len = len(original_table)
        bank = MpcWeightSolverBank(
            self.config,
            actions=CUSTOM_ACTIONS,
        )

        self.assertEqual(tuple(bank._entries), (0, 1))
        self.assertEqual(
            tuple(
                entry.action
                for entry in bank._entries.values()
            ),
            CUSTOM_ACTIONS,
        )
        self.assertEqual(
            tuple(
                (
                    entry.config.q_h2,
                    entry.config.q_batt,
                    entry.config.q_soc,
                    entry.config.q_fc_var,
                )
                for entry in bank._entries.values()
            ),
            tuple(action.as_tuple() for action in CUSTOM_ACTIONS),
        )

        env = DqnMpcWeightEnv(
            loads_kw=np.asarray(
                [200.0, 210.0, 220.0],
                dtype=float,
            ),
            base_config=self.config,
            actions=CUSTOM_ACTIONS,
        )
        self.assertEqual(tuple(env.solver_bank._entries), (0, 1))
        self.assertIs(DQN_MPC_WEIGHT_ACTIONS, original_table)
        self.assertEqual(
            len(DQN_MPC_WEIGHT_ACTIONS),
            original_len,
        )

    def test_representative_states_are_deterministic_and_cover_regimes(
        self,
    ) -> None:
        loads = np.asarray(
            [
                90.0,
                100.0,
                160.0,
                155.0,
                105.0,
                280.0,
                300.0,
                365.0,
                355.0,
                290.0,
                480.0,
                500.0,
                560.0,
                550.0,
                490.0,
            ],
            dtype=float,
        )
        voyage_loads = {
            ("train", "voyage_001"): loads,
            ("validation", "voyage_047"): loads[::-1].copy(),
        }

        first = build_representative_states(
            voyage_loads,
            base_config=self.config,
        )
        second = build_representative_states(
            voyage_loads,
            base_config=self.config,
        )

        self.assertEqual(first, second)
        self.assertEqual(len(first), 81)
        self.assertTrue(
            all(
                isinstance(state, RepresentativeState)
                for state in first
            )
        )
        self.assertEqual(
            {state.load_regime for state in first},
            {"low", "medium", "high"},
        )
        self.assertEqual(
            {state.load_delta_regime for state in first},
            {"rapidly_falling", "near_steady", "rapidly_rising"},
        )
        self.assertEqual(
            {state.soc_regime for state in first},
            {"low", "reference", "high"},
        )
        self.assertEqual(
            {state.previous_fc_regime for state in first},
            {"low", "medium", "high"},
        )
        self.assertTrue(
            all(len(state.future_load_kw) == 6 for state in first)
        )
        self.assertEqual(
            len(
                {
                    (
                        state.source_split,
                        state.source_voyage_id,
                        state.decision_index,
                    )
                    for state in first
                }
            ),
            9,
        )
        self.assertFalse(
            {
                state.source_voyage_id
                for state in first
            }
            & {
                f"voyage_{index:03d}"
                for index in range(60, 67)
            }
        )

    def test_test_split_is_rejected_before_trajectory_io(
        self,
    ) -> None:
        with patch(
            "evaluate_dqn_mpc_action_space.load_voyage_loads",
            side_effect=AssertionError(
                "test trajectory loader must not be called"
            ),
        ) as loader:
            with self.assertRaisesRegex(
                ValueError,
                "test",
            ):
                evaluate_action_space(
                    actions=CUSTOM_ACTIONS,
                    split_names=("test",),
                    split_path=DEFAULT_SPLIT_JSON,
                )

        loader.assert_not_called()

    def test_locked_test_voyage_is_rejected_before_io_even_if_split_is_wrong(
        self,
    ) -> None:
        corrupt_split = VoyageSplit(
            train_voyages=("voyage_060",),
            validation_voyages=(),
            test_voyages=("voyage_999",),
            excluded_voyages=(),
            formal_1s_directory=FORMAL_DATA_DIRECTORY,
            target_load=FORMAL_TARGET_LOAD,
            sample_interval_seconds=FORMAL_SAMPLE_INTERVAL_SECONDS,
        )
        with (
            patch(
                "evaluate_dqn_mpc_action_space.load_voyage_split",
                return_value=corrupt_split,
            ),
            patch(
                "evaluate_dqn_mpc_action_space.load_voyage_loads",
                side_effect=AssertionError(
                    "locked test trajectory loader must not be called"
                ),
            ) as loader,
        ):
            with self.assertRaisesRegex(
                ValueError,
                "voyage_060",
            ):
                evaluate_action_space(
                    actions=CUSTOM_ACTIONS,
                    split_names=("train",),
                )

        loader.assert_not_called()

    def test_status_classes_are_not_collapsed_into_one_failure(
        self,
    ) -> None:
        expected = {
            "solved": "solved",
            "solved inaccurate": "solved_inaccurate",
            "primal infeasible": "primal_infeasible",
            "primal infeasible inaccurate": "primal_infeasible",
            "maximum iterations reached": "maximum_iterations",
            "dual infeasible": "other_failure",
        }

        self.assertEqual(
            {
                status: classify_solver_status(status)
                for status in expected
            },
            expected,
        )

    def test_probe_rows_expose_physics_horizons_and_fixed_reward_terms(
        self,
    ) -> None:
        state = RepresentativeState(
            state_id="probe_001",
            source_split="train",
            source_voyage_id="voyage_001",
            decision_index=4,
            current_soc=0.55,
            previous_fc_kw=200.0,
            previous_batt_kw=0.0,
            current_load_kw=200.0,
            previous_load_kw=195.0,
            future_load_kw=(
                215.0,
                230.0,
                245.0,
                235.0,
                225.0,
                220.0,
            ),
            load_regime="medium",
            load_delta_regime="near_steady",
            soc_regime="reference",
            previous_fc_regime="medium",
        )

        rows = evaluate_state_probes(
            states=(state,),
            actions=CUSTOM_ACTIONS,
            base_config=self.config,
        )

        self.assertEqual(len(rows), 2)
        required = {
            "state_id",
            "action_id",
            "action_name",
            "solver_status",
            "solver_status_class",
            "solve_ms",
            "p_fc_first_kw",
            "p_batt_first_kw",
            "delta_p_fc_kw",
            "soc_predicted",
            "soc_next",
            "h2_kg",
            "reward",
            "weighted_h2",
            "weighted_batt",
            "weighted_soc",
            "weighted_fc_var",
            "p_fc_horizon_kw",
            "p_batt_horizon_kw",
            "soc_horizon",
        }

        for row in rows:
            with self.subTest(action_id=row["action_id"]):
                self.assertTrue(required.issubset(row))
                self.assertEqual(
                    row["solver_status_class"],
                    "solved",
                )
                self.assertEqual(len(row["p_fc_horizon_kw"]), 6)
                self.assertEqual(len(row["p_batt_horizon_kw"]), 6)
                self.assertEqual(len(row["soc_horizon"]), 7)
                self.assertAlmostEqual(
                    row["p_fc_first_kw"]
                    + row["p_batt_first_kw"],
                    state.future_load_kw[0],
                    places=6,
                )
                self.assertAlmostEqual(
                    row["reward"],
                    -(
                        row["weighted_h2"]
                        + row["weighted_batt"]
                        + row["weighted_soc"]
                        + row["weighted_fc_var"]
                    ),
                    places=10,
                )

    def test_invalid_solved_vectors_are_recorded_as_probe_failures(
        self,
    ) -> None:
        state = RepresentativeState(
            state_id="probe_invalid",
            source_split="train",
            source_voyage_id="voyage_001",
            decision_index=1,
            current_soc=0.55,
            previous_fc_kw=200.0,
            previous_batt_kw=0.0,
            current_load_kw=200.0,
            previous_load_kw=195.0,
            future_load_kw=(210.0, 220.0, 230.0, 220.0, 210.0, 200.0),
            load_regime="medium",
            load_delta_regime="near_steady",
            soc_regime="reference",
            previous_fc_regime="medium",
        )

        class InvalidSolutionBank:
            solution = None

            def __init__(self, *_: object, **__: object) -> None:
                pass

            def solve(self, **_: object):
                return (
                    SimpleNamespace(
                        info=SimpleNamespace(
                            status="solved",
                            iter=1,
                            prim_res=0.0,
                            dual_res=0.0,
                        ),
                        x=self.solution,
                    ),
                    0.1,
                )

        invalid_vectors = (
            None,
            np.zeros(2, dtype=float),
            np.full(19, np.nan, dtype=float),
        )
        for invalid_vector in invalid_vectors:
            with self.subTest(invalid_vector=repr(invalid_vector)):
                InvalidSolutionBank.solution = invalid_vector
                with patch(
                    "evaluate_dqn_mpc_action_space.MpcWeightSolverBank",
                    InvalidSolutionBank,
                ):
                    rows = evaluate_state_probes(
                        states=(state,),
                        actions=(CUSTOM_ACTIONS[0],),
                        base_config=self.config,
                    )

                self.assertEqual(len(rows), 1)
                self.assertEqual(
                    rows[0]["solver_status_class"],
                    "other_failure",
                )
                self.assertIsNone(rows[0]["p_fc_first_kw"])

    def test_short_coverage_stops_at_failure_without_action_fallback(
        self,
    ) -> None:
        calls: list[int] = []
        failing_action = CUSTOM_ACTIONS[1]

        class FailingEnv:
            def __init__(self, **_: object) -> None:
                self.current_soc = 0.55
                self.previous_fc_kw = 200.0
                self.previous_batt_kw = 0.0
                self.decision_index = 0

            def reset(self) -> np.ndarray:
                return np.zeros(11, dtype=np.float32)

            def step(self, action_id: int):
                calls.append(action_id)
                error = RuntimeError(
                    "MPC solve failed: "
                    f"action_id={action_id}, "
                    "decision_index=0, "
                    "status=maximum iterations reached"
                )
                error.solver_status = (
                    "maximum iterations reached"
                )
                error.solve_ms = 1.25
                raise error

        voyage_loads = {
            ("train", "voyage_001"): np.asarray(
                [200.0, 215.0, 230.0],
                dtype=float,
            )
        }

        with patch(
            "evaluate_dqn_mpc_action_space.DqnMpcWeightEnv",
            FailingEnv,
        ):
            rows = evaluate_fixed_action_coverage(
                voyage_loads=voyage_loads,
                actions=(failing_action,),
                base_config=self.config,
            )

        self.assertEqual(calls, [failing_action.action_id])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["action_id"], 1)
        self.assertFalse(rows[0]["completed"])
        self.assertEqual(rows[0]["successful_steps"], 0)
        self.assertEqual(
            rows[0]["solver_status_class"],
            "maximum_iterations",
        )
        self.assertEqual(
            rows[0]["first_failure_decision_index"],
            0,
        )

    def test_coverage_never_classifies_a_raised_failure_as_solved(
        self,
    ) -> None:
        class InvalidSolvedEnv:
            def __init__(self, **_: object) -> None:
                self.current_soc = 0.55
                self.decision_index = 0

            def reset(self) -> np.ndarray:
                return np.zeros(11, dtype=np.float32)

            def step(self, action_id: int):
                error = RuntimeError(
                    "invalid solution after solved status"
                )
                error.solver_status = "solved"
                error.solve_ms = 0.1
                error.decision_index = 0
                raise error

        with patch(
            "evaluate_dqn_mpc_action_space.DqnMpcWeightEnv",
            InvalidSolvedEnv,
        ):
            rows = evaluate_fixed_action_coverage(
                voyage_loads={
                    ("train", "voyage_001"): np.asarray(
                        [200.0, 215.0, 230.0],
                        dtype=float,
                    )
                },
                actions=(CUSTOM_ACTIONS[0],),
                base_config=self.config,
            )

        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0]["completed"])
        self.assertEqual(
            rows[0]["solver_status_class"],
            "other_failure",
        )

    def test_summary_and_writer_preserve_statuses_and_reward_terms(
        self,
    ) -> None:
        probe_rows = evaluate_state_probes(
            states=(
                RepresentativeState(
                    state_id="probe_001",
                    source_split="train",
                    source_voyage_id="voyage_001",
                    decision_index=1,
                    current_soc=0.55,
                    previous_fc_kw=200.0,
                    previous_batt_kw=0.0,
                    current_load_kw=200.0,
                    previous_load_kw=195.0,
                    future_load_kw=(
                        210.0,
                        220.0,
                        230.0,
                        220.0,
                        210.0,
                        200.0,
                    ),
                    load_regime="medium",
                    load_delta_regime="near_steady",
                    soc_regime="reference",
                    previous_fc_regime="medium",
                ),
            ),
            actions=CUSTOM_ACTIONS,
            base_config=self.config,
        )
        coverage_rows = [
            {
                "split": "train",
                "voyage_id": "voyage_001",
                "action_id": 0,
                "action_name": "candidate_C",
                "completed": True,
                "solver_status_class": "solved",
                "successful_steps": 2,
                "mean_solve_ms": 0.2,
                "p99_solve_ms": 0.3,
                "episode_reward": -1.0,
                "weighted_h2": 0.1,
                "weighted_batt": 0.2,
                "weighted_soc": 0.3,
                "weighted_fc_var": 0.4,
            },
            {
                "split": "train",
                "voyage_id": "voyage_001",
                "action_id": 1,
                "action_name": "moderate_hydrogen_economy",
                "completed": False,
                "solver_status_class": "maximum_iterations",
                "successful_steps": 1,
                "mean_solve_ms": 0.4,
                "p99_solve_ms": 0.5,
                "episode_reward": -2.0,
                "weighted_h2": 0.2,
                "weighted_batt": 0.3,
                "weighted_soc": 0.4,
                "weighted_fc_var": 0.5,
            },
        ]
        result = {
            "actions": [
                {
                    "action_id": action.action_id,
                    "name": action.name,
                    "weights": list(action.as_tuple()),
                }
                for action in CUSTOM_ACTIONS
            ],
            "representative_states": [],
            "probe_rows": probe_rows,
            "coverage_rows": coverage_rows,
            "data_access": {
                "test_voyages_locked": [
                    f"voyage_{index:03d}"
                    for index in range(60, 67)
                ],
                "test_trajectories_accessed": [],
            },
        }
        summary = summarize_action_space(result)
        self.assertEqual(
            summary["coverage"]["status_counts"],
            {"solved": 1, "maximum_iterations": 1},
        )
        self.assertEqual(
            summary["coverage"]["per_action"]["A0"]["success"],
            1,
        )
        self.assertEqual(
            summary["coverage"]["per_action"]["A1"][
                "maximum_iterations"
            ],
            1,
        )
        self.assertTrue(
            any(
                "maximum-iterations" in reason
                for reason in summary["acceptance"]["reasons"]
            )
        )

        with tempfile.TemporaryDirectory() as directory:
            paths = write_action_space_evaluation(
                result,
                output_dir=Path(directory),
                prefix="final",
            )
            self.assertEqual(set(paths), {"probes", "coverage", "summary"})
            self.assertTrue(all(path.is_file() for path in paths.values()))

    def test_all_action_failure_voyage_fails_acceptance(self) -> None:
        result = {
            "actions": [
                {
                    "action_id": action.action_id,
                    "name": action.name,
                    "weights": list(action.as_tuple()),
                }
                for action in CUSTOM_ACTIONS
            ],
            "probe_rows": [],
            "coverage_rows": [
                {
                    "voyage_id": "voyage_001",
                    "action_id": action.action_id,
                    "completed": False,
                    "solver_status_class": "primal_infeasible",
                    "mean_solve_ms": 0.2,
                }
                for action in CUSTOM_ACTIONS
            ],
            "data_access": {
                "test_trajectories_accessed": [],
            },
        }

        summary = summarize_action_space(result)

        self.assertEqual(
            summary["coverage"]["all_action_failure_voyages"],
            ["voyage_001"],
        )
        self.assertEqual(
            summary["acceptance"]["decision"],
            "FAIL",
        )
        self.assertTrue(
            any(
                "all-action-failure" in reason
                for reason in summary["acceptance"]["reasons"]
            )
        )

    def test_redundancy_and_dominant_winner_fail_acceptance(self) -> None:
        probe_rows = []
        for state_index in range(10):
            for action in CUSTOM_ACTIONS:
                probe_rows.append(
                    {
                        "state_id": f"probe_{state_index:03d}",
                        "action_id": action.action_id,
                        "action_name": action.name,
                        "p_fc_first_kw": (
                            200.0 + 0.2 * action.action_id
                        ),
                        "p_fc_horizon_kw": [
                            200.0 + 0.2 * action.action_id
                        ]
                        * 6,
                        "soc_horizon": [0.55] * 7,
                        "reward": -1.0 - action.action_id,
                        "load_regime": "medium",
                        "load_delta_regime": "near_steady",
                        "soc_regime": "reference",
                        "previous_fc_regime": "medium",
                        "solver_status_class": "solved",
                    }
                )
        coverage_rows = [
            {
                "voyage_id": "voyage_001",
                "action_id": action.action_id,
                "completed": True,
                "solver_status_class": "solved",
                "mean_solve_ms": 0.2,
            }
            for action in CUSTOM_ACTIONS
        ]
        result = {
            "actions": [
                {
                    "action_id": action.action_id,
                    "name": action.name,
                    "weights": list(action.as_tuple()),
                }
                for action in CUSTOM_ACTIONS
            ],
            "probe_rows": probe_rows,
            "coverage_rows": coverage_rows,
            "data_access": {
                "test_trajectories_accessed": [],
            },
        }

        summary = summarize_action_space(result)

        self.assertEqual(
            summary["probe"]["redundant_pairs_at_1kw"],
            ["A0-A1"],
        )
        self.assertEqual(
            summary["probe"]["dominant_winner_fraction"],
            1.0,
        )
        self.assertEqual(
            summary["acceptance"]["decision"],
            "FAIL",
        )
        self.assertTrue(
            any(
                "redundant" in reason
                for reason in summary["acceptance"]["reasons"]
            )
        )
        self.assertTrue(
            any(
                "90%" in reason
                for reason in summary["acceptance"]["reasons"]
            )
        )


if __name__ == "__main__":
    unittest.main()
