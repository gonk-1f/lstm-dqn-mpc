from __future__ import annotations

import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class V2ContractTests(unittest.TestCase):
    def test_version_contracts_are_frozen(self) -> None:
        from v2 import contracts

        expected = {
            "METHOD_VERSION": "multiscale_dqn_wmpc_v2",
            "MPC_OBJECTIVE_VERSION": "fc_base_smooth_soc_deadband_v1",
            "ACTION_TABLE_VERSION": "three_weight_simplex_behavior_filtered_v1",
            "REWARD_VERSION": "macro_interval_real_economic_cost_v1",
            "FC_ENERGY_VERSION": "eta_fc_lhv_h2_v1",
            "FC_DEGRADATION_VERSION": "aggregate_four_condition_voltage_loss_v1",
            "BATTERY_DEGRADATION_VERSION": "soc_current_weighted_throughput_v1",
            "DATASET_VERSION": "mode_aware_operating_cycle_v2",
        }

        self.assertEqual(
            {name: getattr(contracts, name) for name in expected},
            expected,
        )

    def test_provisional_timescales_have_separate_semantics(self) -> None:
        from v2.config import TimeScaleConfig

        config = TimeScaleConfig.provisional()

        self.assertEqual(config.ts_mpc_seconds, 30.0)
        self.assertEqual(config.n_mpc, 5)
        self.assertEqual(config.dqn_switch_steps, 5)
        self.assertEqual(config.prediction_seconds, 150.0)
        self.assertEqual(config.switch_seconds, 150.0)
        self.assertEqual(config.mpc_solves_per_action, 5)
        self.assertNotEqual(config.n_mpc_semantics, config.dqn_switch_steps_semantics)

    def test_timescale_validation_does_not_couple_n_and_m(self) -> None:
        from v2.config import TimeScaleConfig

        config = TimeScaleConfig(
            ts_mpc_seconds=30.0,
            n_mpc=7,
            dqn_switch_steps=11,
        )

        self.assertEqual(config.prediction_seconds, 210.0)
        self.assertEqual(config.switch_seconds, 330.0)

    def test_v1_semantics_are_rejected(self) -> None:
        from v2.contracts import IncompatibleArtifactError, require_v2_semantics

        with self.assertRaises(IncompatibleArtifactError):
            require_v2_semantics(
                {
                    "method_version": "executed_closed_loop_reward",
                    "action_table_version": "positive_integer_simplex_10_lexicographic_v1",
                }
            )

    def test_any_metadata_mismatch_is_rejected(self) -> None:
        from v2.contracts import (
            IncompatibleArtifactError,
            control_semantics,
            require_v2_semantics,
        )

        mismatch = control_semantics()
        mismatch["reward_version"] = "macro_interval_real_economic_cost_v0"
        with self.assertRaises(IncompatibleArtifactError):
            require_v2_semantics(mismatch)

        extra = control_semantics()
        extra["legacy_switch_seconds"] = 1.0
        with self.assertRaises(IncompatibleArtifactError):
            require_v2_semantics(extra)

    def test_current_semantics_round_trip(self) -> None:
        from v2.contracts import control_semantics, require_v2_semantics

        semantics = control_semantics()
        require_v2_semantics(dict(semantics))
        self.assertEqual(semantics["n_mpc"], 5)
        self.assertEqual(semantics["dqn_switch_steps"], 5)
        self.assertEqual(semantics["ts_mpc_seconds"], 30.0)


if __name__ == "__main__":
    unittest.main()
