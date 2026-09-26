from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class TestFormalModeInterlock(unittest.TestCase):
    @staticmethod
    def _solver():
        from v2.models.battery_energy import formal_battery_efficiency, next_soc

        class Solver:
            def __init__(self):
                self.calls = 0

            def solve(self, **kwargs):
                self.calls += 1
                load = float(kwargs["observed_load_kw"])
                soc = float(kwargs["current_soc"])
                p_fc = min(60.0, load)
                p_batt = load - p_fc
                predicted = next_soc(
                    soc,
                    p_batt,
                    30.0,
                    624.0,
                    efficiency=formal_battery_efficiency(),
                )

                class Command:
                    p_fc_kw = p_fc
                    p_batt_bus_kw = p_batt
                    predicted_next_soc = predicted

                class Plan:
                    @staticmethod
                    def first_command():
                        return Command()

                return Plan()

        return Solver()

    def test_explicit_modes_replace_blanket_negative_power_classification(self) -> None:
        from v2.envs.formal_episode import FormalEpisodeBackend

        backend = FormalEpisodeBackend(
            load_kw=np.asarray([120.0, -70.0]),
            speed_kn=np.asarray([5.0, 0.0]),
            fc_power_kw=np.asarray([100.0, 0.0]),
            battery_bus_kw=np.asarray([20.0, -70.0]),
            operating_mode=("onboard", "unresolved"),
            mpc=self._solver(),
        )
        backend.reset()
        from v2.dqn.action_space import FINAL_DQN_ACTION_CATALOG

        backend.execute_mpc_step(FINAL_DQN_ACTION_CATALOG[0].to_mpc_weights())
        with self.assertRaisesRegex(ValueError, "unresolved"):
            backend.execute_mpc_step(FINAL_DQN_ACTION_CATALOG[0].to_mpc_weights())

    def test_onboard_deadband_negative_load_reaches_mpc_as_zero(self) -> None:
        from v2.dqn.action_space import FINAL_DQN_ACTION_CATALOG
        from v2.envs.formal_episode import FormalEpisodeBackend

        class RecordingSolver:
            def __init__(self) -> None:
                self.observed_load_kw = None

            def solve(self, **kwargs):
                self.observed_load_kw = float(kwargs["observed_load_kw"])
                soc = float(kwargs["current_soc"])

                class Command:
                    p_fc_kw = 0.0
                    p_batt_bus_kw = 0.0
                    predicted_next_soc = soc

                class Plan:
                    @staticmethod
                    def first_command():
                        return Command()

                return Plan()

        solver = RecordingSolver()
        backend = FormalEpisodeBackend(
            load_kw=np.asarray([-0.5]),
            speed_kn=np.asarray([0.2]),
            fc_power_kw=np.asarray([0.0]),
            battery_bus_kw=np.asarray([0.0]),
            operating_mode=("onboard",),
            mpc=solver,
        )

        backend.execute_mpc_step(FINAL_DQN_ACTION_CATALOG[0].to_mpc_weights())

        self.assertEqual(solver.observed_load_kw, 0.0)

    def test_shore_pauses_mpc_but_updates_soc_degradation_cost_and_reentry_s8(self) -> None:
        from v2.config import TimeScaleConfig
        from v2.data.supervisory_rules import OperatingMode
        from v2.dqn.action_space import FINAL_DQN_ACTION_CATALOG
        from v2.envs.formal_episode import FormalEpisodeBackend
        from v2.envs.multirate_weight_env import MultiRateWeightEnvironment

        solver = self._solver()
        modes = (
            OperatingMode.ONBOARD.value,
            OperatingMode.ONBOARD.value,
            OperatingMode.ONBOARD.value,
            OperatingMode.SHORE_PENDING.value,
            OperatingMode.SHORE_PENDING.value,
            OperatingMode.SHORE_CHARGING.value,
            OperatingMode.ONBOARD.value,
        )
        backend = FormalEpisodeBackend(
            load_kw=np.asarray([120.0, 120.0, 120.0, -70.0, -70.0, -70.0, 100.0]),
            speed_kn=np.asarray([5.0, 5.0, 1.0, 0.0, 0.0, 0.0, 4.0]),
            fc_power_kw=np.asarray([100.0, 100.0, 100.0, 0.0, 0.0, 0.0, 90.0]),
            battery_bus_kw=np.asarray([20.0, 20.0, 20.0, -70.0, -70.0, -70.0, 10.0]),
            operating_mode=modes,
            mpc=solver,
        )
        environment = MultiRateWeightEnvironment(
            timescale=TimeScaleConfig.formal_baseline(),
            action_catalog=FINAL_DQN_ACTION_CATALOG,
            backend=backend,
            state_provider=backend.state,
            formal_training_mode=True,
        )

        initial = environment.reset()
        self.assertEqual(len(initial), 8)
        transition = environment.step(FINAL_DQN_ACTION_CATALOG[0].action_id)

        self.assertFalse(transition.done)
        self.assertEqual(transition.executed_mpc_steps, 3)
        self.assertEqual(solver.calls, 3)
        self.assertEqual(backend.index, 6)
        self.assertEqual(backend.mode_counts[OperatingMode.SHORE_PENDING], 2)
        self.assertEqual(backend.mode_counts[OperatingMode.SHORE_CHARGING], 1)
        self.assertGreater(transition.ledger.shore_cost_cny, 0.0)
        self.assertGreater(transition.ledger.battery_degradation_cost_cny, 0.0)
        self.assertEqual(backend.executed_fc_power_kw[3:6], [0.0, 0.0, 0.0])
        self.assertLessEqual(backend.soc, backend.INITIAL_SOC)
        self.assertAlmostEqual(transition.next_state[1], 100.0 / 600.0)
        self.assertEqual(transition.next_state[2:7], (0.0, 0.0, 0.0, 0.0, 0.0))
        self.assertAlmostEqual(transition.next_state[7], 4.0 / 20.0)

    def test_shore_at_target_accepts_no_power_and_does_not_charge_cost(self) -> None:
        from v2.data.supervisory_rules import OperatingMode
        from v2.dqn.action_space import FINAL_DQN_ACTION_CATALOG
        from v2.envs.formal_episode import FormalEpisodeBackend

        backend = FormalEpisodeBackend(
            load_kw=np.asarray([0.0, -70.0, -70.0, -70.0]),
            speed_kn=np.zeros(4),
            fc_power_kw=np.zeros(4),
            battery_bus_kw=np.asarray([0.0, -70.0, -70.0, -70.0]),
            operating_mode=(
                OperatingMode.ONBOARD.value,
                OperatingMode.SHORE_PENDING.value,
                OperatingMode.SHORE_PENDING.value,
                OperatingMode.SHORE_CHARGING.value,
            ),
            mpc=self._solver(),
        )
        backend.reset()
        result = backend.execute_mpc_step(
            FINAL_DQN_ACTION_CATALOG[0].to_mpc_weights()
        )
        self.assertTrue(result.mpc_solve_executed)
        shore_ledgers = []
        while not result.done:
            result = backend.execute_mpc_step(
                FINAL_DQN_ACTION_CATALOG[0].to_mpc_weights()
            )
            if not result.mpc_solve_executed:
                shore_ledgers.append(result.ledger)
        self.assertTrue(shore_ledgers)
        self.assertTrue(all(item.shore_cost_cny == 0.0 for item in shore_ledgers[:-1]))

    def test_terminal_transition_retains_finite_s8_for_replay_storage(self) -> None:
        from v2.config import TimeScaleConfig
        from v2.data.supervisory_rules import OperatingMode
        from v2.dqn.action_space import FINAL_DQN_ACTION_CATALOG
        from v2.envs.formal_episode import FormalEpisodeBackend
        from v2.envs.multirate_weight_env import MultiRateWeightEnvironment

        backend = FormalEpisodeBackend(
            load_kw=np.asarray([100.0]),
            speed_kn=np.asarray([3.0]),
            fc_power_kw=np.asarray([90.0]),
            battery_bus_kw=np.asarray([10.0]),
            operating_mode=(OperatingMode.ONBOARD.value,),
            mpc=self._solver(),
        )
        environment = MultiRateWeightEnvironment(
            timescale=TimeScaleConfig.formal_baseline(),
            action_catalog=FINAL_DQN_ACTION_CATALOG,
            backend=backend,
            state_provider=backend.state,
            formal_training_mode=True,
        )
        environment.reset()

        transition = environment.step(FINAL_DQN_ACTION_CATALOG[0].action_id)

        self.assertTrue(transition.done)
        self.assertEqual(len(transition.next_state), 8)
        self.assertTrue(np.isfinite(np.asarray(transition.next_state)).all())

    def test_real_train_segment_015_emits_terminal_failure_transition(self) -> None:
        from v2.data.formal_training_dataset import FormalTrainingDataset
        from v2.failure_policy import FORMAL_FAILURE_KIND
        from v2.main.train_formal_dqn import (
            DEFAULT_AIS_ROOT,
            DEFAULT_MODE_ROOT,
            DEFAULT_POWER_ROOT,
            _environment,
        )

        dataset = FormalTrainingDataset.open(
            DEFAULT_POWER_ROOT, DEFAULT_AIS_ROOT, DEFAULT_MODE_ROOT
        )
        episode = next(
            value for value in dataset.load_train()
            if value.sample_id == "zero_boundary_015"
        )
        _, environment = _environment(episode)
        environment.reset()

        for _ in range(episode.step_count):
            transition = environment.step("w_5_4_1")
            if transition.done:
                break
        else:  # pragma: no cover - defensive bound
            self.fail("zero_boundary_015 did not reach a terminal transition")

        self.assertTrue(transition.failed)
        self.assertFalse(transition.episode_completed)
        self.assertEqual(transition.failure_kind, FORMAL_FAILURE_KIND)
        self.assertEqual(transition.failure_penalty_score, 50_000.0)
        self.assertEqual(
            transition.learning_reward,
            -transition.raw_economic_cost_cny - transition.failure_penalty_score,
        )


if __name__ == "__main__":
    unittest.main()
