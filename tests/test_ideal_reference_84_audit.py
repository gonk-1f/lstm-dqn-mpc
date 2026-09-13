from pathlib import Path
import sys
import unittest
import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / 'src'),
                str(Path(__file__).resolve().parents[1] / 'src/main')]
from audit_ideal_reference_84_train import (
    weight_grid, reference_reward, pareto_mask, physical_feasibility,
    direct_terms, CONFIG,
)


class AuditTests(unittest.TestCase):
    def test_grid_endpoints_and_unique_integer_contract(self):
        actions = weight_grid()
        rows = [a.as_tuple() for a in actions]
        self.assertEqual(len(set(rows)), 84)
        self.assertEqual(rows[0], (.1, .1, .1, .7))
        self.assertEqual(rows[-1], (.7, .1, .1, .1))
        self.assertTrue(all(sum(round(10*x) for x in r) == 10 for r in rows))

    def test_reward_is_action_independent_and_retains_outside_reference(self):
        m = np.ones((4, 4)) - np.eye(4)
        h, r = reference_reward(np.array([[2., 0., 0., 0.], [2., 0., 0., 0.]]), m)
        np.testing.assert_allclose(h[:, 0], 2.)
        np.testing.assert_allclose(r, 1/3)

    def test_degenerate_reference_is_rejected_without_epsilon(self):
        with self.assertRaises(ValueError):
            reference_reward(np.ones((1, 4)), np.ones((4, 4)))

    def test_negative_coordinate_is_exposed_for_diagnosis(self):
        m = np.ones((4, 4)) - np.eye(4)
        h, _ = reference_reward(np.array([[-.1, 0., 0., 0.]]), m)
        self.assertLess(h[0, 0], 0.)

    def test_pareto_ties_and_dominance(self):
        mask = pareto_mask(np.array([[0, 1], [1, 0], [1, 1], [0, 1]]), 0.)
        np.testing.assert_array_equal(mask, [True, True, False, True])

    def test_feasibility_separates_capacity_from_numerical_status(self):
        self.assertEqual(physical_feasibility(np.full(6, 700.), .22, 600.)['classification'], 'feasible')
        self.assertEqual(physical_feasibility(np.full(6, 700.), .20, 600.)['classification'], 'physical_infeasible')
        self.assertEqual(physical_feasibility(np.full(6, 1900.), .55, 600.)['classification'], 'physical_infeasible')

    def test_exact_soc_reconstruction_and_fc_ramp_term(self):
        f, soc, pb = direct_terms(np.full(6, 300.), np.full(6, 300.), .55, 300.)
        np.testing.assert_allclose(soc, .55)
        np.testing.assert_allclose(f[1:], 0., atol=1e-20)
        np.testing.assert_allclose(pb, 0.)


if __name__ == '__main__':
    unittest.main()
