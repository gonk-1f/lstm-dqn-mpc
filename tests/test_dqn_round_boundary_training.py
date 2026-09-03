from __future__ import annotations

import inspect
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
MAIN_ROOT = SRC / "main"

for path in (SRC, MAIN_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


import run_dqn_mpc_causal_training as formal_training  # noqa: E402
import train_dqn_mpc_mlp as training  # noqa: E402
from dqn.agents.dqn_agent import DQNTrainConfig  # noqa: E402


class _FakeAgent:
    def __init__(self, events: list[str], network_type: str) -> None:
        self.events = events
        self.network_type = network_type
        self.weight = torch.tensor([0.0])
        self.q_net = object()
        self.target_q_net = object()
        self.optimizer = object()
        self.update_call_count = 0

    def save(self, model_path: str | Path) -> Path:
        path = Path(model_path)
        self.events.append(
            f"{self.network_type}:save:{int(self.weight.item())}"
        )
        torch.save({"weight": self.weight.clone()}, path)
        return path

    def update(self, batch: object) -> float:
        self.update_call_count += 1
        raise AssertionError("validation must not call agent.update")


class RoundBoundaryTrainingTests(unittest.TestCase):
    def test_each_round_is_saved_and_validated_before_next_round(
        self,
    ) -> None:
        self.assertTrue(
            hasattr(formal_training, "run_round_boundary_training"),
            "formal training needs a shared round-boundary orchestrator",
        )
        self.assertIn(
            "on_round_complete",
            inspect.signature(
                training.train_complete_voyage_rounds
            ).parameters,
        )

        for network_type in ("mlp", "kan"):
            with self.subTest(network_type=network_type):
                events: list[str] = []
                agent = _FakeAgent(events, network_type)
                runtime = SimpleNamespace(
                    agent=agent,
                    global_step=0,
                    policy=SimpleNamespace(epsilon=1.0),
                    replay_buffer=[],
                    update_steps=[],
                    target_sync_steps=[],
                )
                runtime_id = id(runtime)
                agent_id = id(runtime.agent)
                q_net_id = id(runtime.agent.q_net)
                target_q_net_id = id(runtime.agent.target_q_net)
                optimizer_id = id(runtime.agent.optimizer)
                train_call_count = 0

                def fake_training_episode(
                    *, voyage_id, loads_kw, base_config, runtime
                ):
                    nonlocal train_call_count
                    self.assertEqual(id(runtime), runtime_id)
                    self.assertEqual(id(runtime.agent), agent_id)
                    self.assertEqual(id(runtime.agent.q_net), q_net_id)
                    self.assertEqual(
                        id(runtime.agent.target_q_net),
                        target_q_net_id,
                    )
                    self.assertEqual(
                        id(runtime.agent.optimizer),
                        optimizer_id,
                    )
                    train_call_count += 1
                    events.append(
                        f"{network_type}:train:{train_call_count}:"
                        f"step={runtime.global_step}:"
                        f"epsilon={runtime.policy.epsilon:.1f}:"
                        f"replay={len(runtime.replay_buffer)}"
                    )
                    runtime.agent.weight += 1.0
                    runtime.global_step += 1
                    runtime.policy.epsilon -= 0.1
                    runtime.replay_buffer.append(train_call_count)
                    runtime.update_steps.append(runtime.global_step)
                    return {
                        "voyage_id": voyage_id,
                        "episode_steps": len(loads_kw) - 1,
                    }

                def fake_validation_episode(
                    *, voyage_id, loads_kw, base_config, agent
                ):
                    events.append(
                        f"{network_type}:validate:"
                        f"{int(agent.weight.item())}"
                    )
                    return (
                        {
                            "voyage_id": voyage_id,
                            "episode_reward": -1.0,
                        },
                        pd.DataFrame(
                            {
                                "execution_index": [1],
                                "load_kw": [float(loads_kw[-1])],
                                "soc_after": [0.55],
                            }
                        ),
                    )

                split = SimpleNamespace(
                    train_voyages=("voyage_001",),
                    validation_voyages=("voyage_047",),
                )
                loads = {
                    "voyage_001": np.asarray([200.0, 205.0]),
                    "voyage_047": np.asarray([220.0, 225.0]),
                }

                with tempfile.TemporaryDirectory() as temp_dir:
                    output_dir = Path(temp_dir) / network_type
                    output_dir.mkdir()
                    with (
                        patch.object(
                            training,
                            "run_training_episode",
                            side_effect=fake_training_episode,
                        ),
                        patch.object(
                            formal_training.validation_artifacts,
                            "run_test_episode",
                            side_effect=fake_validation_episode,
                        ),
                        patch.object(
                            formal_training.validation_artifacts,
                            "plot_power_allocation",
                        ),
                        patch.object(
                            formal_training.validation_artifacts,
                            "plot_soc_trajectory",
                        ),
                    ):
                        rounds = (
                            formal_training.run_round_boundary_training(
                                split=split,
                                runtime=runtime,
                                base_config=object(),
                                output_dir=output_dir,
                                load_train=loads.__getitem__,
                                load_validation=loads.__getitem__,
                                num_training_rounds=2,
                            )
                        )

                    round1 = torch.load(
                        output_dir / "round_1" / "model_round1.pt",
                        map_location="cpu",
                        weights_only=True,
                    )
                    round2 = torch.load(
                        output_dir / "round_2" / "model_round2.pt",
                        map_location="cpu",
                        weights_only=True,
                    )

                self.assertEqual(
                    events,
                    [
                        f"{network_type}:train:1:step=0:epsilon=1.0:replay=0",
                        f"{network_type}:save:1",
                        f"{network_type}:validate:1",
                        f"{network_type}:train:2:step=1:epsilon=0.9:replay=1",
                        f"{network_type}:save:2",
                        f"{network_type}:validate:2",
                    ],
                )
                self.assertFalse(
                    torch.equal(round1["weight"], round2["weight"])
                )
                self.assertEqual(runtime.global_step, 2)
                self.assertAlmostEqual(runtime.policy.epsilon, 0.8)
                self.assertEqual(runtime.replay_buffer, [1, 2])
                self.assertEqual(runtime.update_steps, [1, 2])
                self.assertEqual(agent.update_call_count, 0)
                self.assertEqual(
                    [round_summary["global_step"] for round_summary in rounds],
                    [1, 2],
                )

    def test_validation_does_not_call_agent_update(self) -> None:
        config = DQNTrainConfig(
            device="cpu",
            batch_size=2,
            warmup_steps=10,
            buffer_size=100,
        )
        runtime = training.create_training_runtime(config)
        base_config = training.build_formal_mpc_config()
        loads = np.asarray([220.0, 225.0, 230.0])

        with patch.object(
            runtime.agent,
            "update",
            wraps=runtime.agent.update,
        ) as update:
            training.validate_voyages(
                voyage_ids=("fixture_validation",),
                load_voyage=lambda _: loads,
                base_config=base_config,
                agent=runtime.agent,
            )

        update.assert_not_called()


if __name__ == "__main__":
    unittest.main()
