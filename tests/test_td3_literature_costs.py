import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC / "scripts"))

from td3_config import evaluate_profiles, make_env
from ship_ems.battery import BatteryConfig, LithiumBattery
from ship_ems.costs import CostCalculator
from ship_ems.fuel_cell import D_START_STOP_UV_PER_CYCLE, FuelCellParameters


class LiteratureCostsTest(unittest.TestCase):
    def test_fuel_cell_eol(self):
        fc = FuelCellParameters()
        self.assertEqual(fc.initial_cell_voltage_v, 0.95)
        self.assertEqual(fc.eol_voltage_drop_ratio, 0.15)
        self.assertAlmostEqual(fc.eol_voltage_loss_uv, 142_500.0)

    def test_soc_factor_and_signed_mass(self):
        calculator = CostCalculator()
        battery = BatteryConfig()
        lhv = calculator.config.hydrogen_lhv_mj_per_kg / 3.6
        price = calculator.config.hydrogen_price_rmb_per_kg
        for soc, expected_factor in (
            (0.20, 8.0), (0.35, 3.375), (0.50, 1.0),
            (0.65, 0.125), (0.70, 1.0 / 27.0), (0.80, 0.0),
        ):
            for power in (90.0, -90.0):
                with self.subTest(soc=soc, power=power):
                    kg, cost, factor, equivalence = calculator._battery_equivalent_hydrogen(
                        soc, power, battery, dt_s=1.0,
                    )
                    expected_s = 1.828869163806 if power > 0 else 1.650554420335
                    self.assertAlmostEqual(factor, expected_factor)
                    self.assertAlmostEqual(equivalence, expected_s)
                    self.assertAlmostEqual(kg, expected_s * power / 3600 / lhv * expected_factor)
                    self.assertAlmostEqual(cost, kg * price)
                    if expected_factor:
                        self.assertGreater(kg * power, 0.0)
                    else:
                        self.assertEqual(kg, 0.0)

    def test_matches_previous_model_at_target_soc(self):
        calculator = CostCalculator()
        battery = LithiumBattery()
        lhv = calculator.config.hydrogen_lhv_mj_per_kg / 3.6
        for power in (90.0, -90.0):
            with self.subTest(power=power):
                battery.reset(initial_soc=0.50)
                step = battery.step(power_kw=power, dt_s=1.0)
                old_mass = (step.soc_before - step.soc_after) * battery.config.capacity_kwh / (
                    calculator.eta_fc_avg * lhv
                )
                new_mass, *_ = calculator._battery_equivalent_hydrogen(
                    step.soc_before, step.power_kw, battery.config, dt_s=1.0,
                )
                self.assertAlmostEqual(new_mass, old_mass, places=13)

    def test_environment_reward_and_voyage_ledger(self):
        env = make_env([[100.0, 100.0, 0.0]], ["short"])
        self.addCleanup(env.close)
        env.reset(seed=42)
        mass = []
        costs = []
        for action, completed in ((0.0, False), (1.0, True)):
            _, reward, terminated, truncated, info = env.step([action])
            self.assertEqual(terminated, completed)
            self.assertFalse(truncated)
            self.assertAlmostEqual(info["soc_after"], info["soc"])
            self.assertAlmostEqual(info["step_soc_penalty_unscaled"], 100 * (info["soc"] - 0.5) ** 2)
            self.assertAlmostEqual(
                info["battery_equivalent_soc_factor"],
                (1 - (info["soc_before"] - 0.5) / 0.3) ** 3,
            )
            mass.append(info["step_battery_equivalent_hydrogen_kg"])
            costs.append(info["step_battery_equivalent_hydrogen_cost_rmb"])
            self.assertAlmostEqual(reward, -(
                info["step_reward_cost_rmb"] + costs[-1] + info["step_soc_penalty_unscaled"]
            ))
        self.assertGreater(costs[0], 0.0)
        self.assertLess(costs[1], 0.0)
        self.assertAlmostEqual(info["battery_equivalent_hydrogen_kg"], math.fsum(mass))
        self.assertAlmostEqual(info["battery_equivalent_hydrogen_cost_rmb"], math.fsum(costs))
        fc = env.fuel_cell.params
        self.assertAlmostEqual(
            info["fc_operating_degradation_cost_rmb"],
            env.fuel_cell.voltage_loss_uv / 142_500 * fc.replacement_cost_rmb,
        )
        self.assertAlmostEqual(
            info["fc_start_stop_degradation_cost_rmb"],
            D_START_STOP_UV_PER_CYCLE / 142_500 * fc.replacement_cost_rmb,
        )

    def test_evaluation_step_columns_keep_terminal_costs_separate(self):
        profiles = [[100.0, 100.0, 0.0]]
        env = make_env(profiles, ["short"])
        self.addCleanup(env.close)
        model = SimpleNamespace(
            observation_space=env.observation_space,
            action_space=env.action_space,
            policy=SimpleNamespace(set_training_mode=lambda _: None),
            predict=lambda obs, deterministic: (np.array([1.0]), None),
        )
        summaries, steps = evaluate_profiles(model, profiles, ["short"], collect_step_rows=True)
        last = steps[-1]
        self.assertEqual(last["soc_after"], last["soc"])
        self.assertEqual(
            last["battery_equivalent_hydrogen_cost_rmb"],
            last["step_battery_equivalent_hydrogen_cost_rmb"],
        )
        self.assertAlmostEqual(
            last["voyage_battery_equivalent_hydrogen_cost_rmb"],
            summaries[0]["battery_equivalent_hydrogen_cost_rmb"],
        )

    def test_failure_reward(self):
        env = make_env([[5000.0, 0.0]], ["infeasible"])
        self.addCleanup(env.close)
        env.reset(seed=42)
        _, reward, terminated, truncated, info = env.step([0.5])
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["failure"])
        self.assertEqual(reward, 0.0)
        self.assertEqual(env.episode_step_costs, [])


if __name__ == "__main__":
    unittest.main()
