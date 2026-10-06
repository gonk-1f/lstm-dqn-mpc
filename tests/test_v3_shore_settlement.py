import unittest

from v3.control import AccountState, EconomicMPC
from v3.shore import settle_shore_segment


class ShoreSettlementTests(unittest.TestCase):
    def test_accepted_charge_sets_soc_grid_cost_and_battery_degradation(self):
        mpc = EconomicMPC(nominal_cost_cny=1.0)
        before = AccountState(soc=0.5, previous_fc_kw=100.0)

        result = settle_shore_segment(mpc, before, (-50.0, -50.0))

        bus_energy_kwh = 50.0 * 30.0 / 3600.0
        self.assertAlmostEqual(result.end_state.soc, 0.5 + 2 * bus_energy_kwh / 624.0)
        self.assertAlmostEqual(result.ledger.shore_cost_cny, 2 * bus_energy_kwh / 0.95 * 1.10)
        self.assertEqual(result.requested_battery_bus_kw, (-50.0, -50.0))
        self.assertEqual(result.accepted_battery_bus_kw, (-50.0, -50.0))
        self.assertEqual(result.end_state.previous_fc_kw, 0.0)
        self.assertEqual(result.ledger.h2_cost_cny, 0.0)
        self.assertGreater(result.ledger.battery_degradation_cost_cny, 0.0)
        self.assertGreater(result.steps[0].fuel_cell_degradation_cost_cny, 0.0)
        self.assertEqual(result.steps[1].fuel_cell_degradation_cost_cny, 0.0)
        self.assertEqual(result.ledger.reward_cny, -result.ledger.total_cost_cny)

    def test_shore_settlement_rejects_discharge_and_caps_charge_at_target_soc(self):
        mpc = EconomicMPC(nominal_cost_cny=1.0)
        with self.assertRaises(ValueError):
            settle_shore_segment(mpc, AccountState(soc=0.5), (10.0,))
        full = settle_shore_segment(mpc, AccountState(soc=0.6), (-50.0,))
        self.assertEqual(full.accepted_battery_bus_kw, (0.0,))
        self.assertEqual(full.end_state.soc, 0.6)
        self.assertEqual(full.ledger.shore_cost_cny, 0.0)

        almost_full = settle_shore_segment(mpc, AccountState(soc=0.5999), (-50.0,))
        expected_accepted = -(0.6 - 0.5999) * 624.0 * 3600.0 / 30.0
        self.assertAlmostEqual(almost_full.accepted_battery_bus_kw[0], expected_accepted)
        self.assertAlmostEqual(almost_full.end_state.soc, 0.6)
        self.assertAlmostEqual(
            almost_full.ledger.shore_cost_cny,
            -expected_accepted * 30.0 / 3600.0 / 0.95 * 1.10,
        )

        limited = settle_shore_segment(mpc, AccountState(soc=0.2), (-1000.0,))
        self.assertEqual(limited.requested_battery_bus_kw, (-1000.0,))
        self.assertEqual(limited.accepted_battery_bus_kw, (-624.0,))


if __name__ == "__main__":
    unittest.main()
