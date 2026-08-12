from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dqn.utils.action_mapper import DQN_MPC_WEIGHT_ACTIONS, get_weight_action  # noqa: E402


class TestDqnMpcWeightActionTable(unittest.TestCase):
    def test_action_table_is_not_empty(self) -> None:
        self.assertGreater(len(DQN_MPC_WEIGHT_ACTIONS), 0)

    def test_action_ids_are_contiguous_from_zero(self) -> None:
        self.assertEqual(
            [action.action_id for action in DQN_MPC_WEIGHT_ACTIONS],
            list(range(len(DQN_MPC_WEIGHT_ACTIONS))),
        )

    def test_weight_tuples_are_unique(self) -> None:
        weight_tuples = {action.as_tuple() for action in DQN_MPC_WEIGHT_ACTIONS}
        self.assertEqual(
            len(weight_tuples),
            len(DQN_MPC_WEIGHT_ACTIONS),
        )

    def test_action_zero_is_nominal(self) -> None:
        action = get_weight_action(0)
        self.assertEqual(action.action_id, 0)
        self.assertEqual(action.as_tuple(), (0.25, 0.40, 12.0, 20.0))
        self.assertEqual(action.name, "nominal")

    def test_v3_action_definitions(self) -> None:
        expected = {
            0: ((0.25, 0.40, 12.0, 20.0), "nominal"),
            1: ((0.40, 0.25, 8.0, 8.0), "hydrogen_economy"),
            2: ((0.25, 0.45, 36.0, 15.0), "soc_regulation"),
            3: ((0.15, 0.80, 12.0, 1.0), "fast_fc_response"),
        }

        for action_id, (weights, name) in expected.items():
            with self.subTest(action_id=action_id):
                action = get_weight_action(action_id)
                self.assertEqual(action.as_tuple(), weights)
                self.assertEqual(action.name, name)

    def test_as_tuple_uses_required_weight_order(self) -> None:
        action = get_weight_action(1)
        self.assertEqual(
            action.as_tuple(),
            (action.q_h2, action.q_batt, action.q_soc, action.q_fc_var),
        )

    def test_invalid_action_id_raises(self) -> None:
        for action_id in (-1, len(DQN_MPC_WEIGHT_ACTIONS)):
            with self.subTest(action_id=action_id):
                with self.assertRaises((IndexError, ValueError)):
                    get_weight_action(action_id)


if __name__ == "__main__":
    unittest.main()
