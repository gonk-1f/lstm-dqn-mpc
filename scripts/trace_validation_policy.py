from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]

TRAIN_SCRIPT = (
    ROOT
    / "src"
    / "main"
    / "train_dqn_mpc_mlp.py"
)

MODEL_PATH = (
    ROOT
    / "outputs"
    / "dqn_mpc_mlp_one_epoch_20260812_180309"
    / "model_final.pt"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "dqn_mpc_mlp_one_epoch_20260812_180309"
    / "policy_trace"
)

TARGET_VOYAGES = (
    "voyage_053",
    "voyage_054",
)


def load_training_module():
    spec = importlib.util.spec_from_file_location(
        "train_dqn_mpc_trace",
        TRAIN_SCRIPT,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Cannot load training script: {TRAIN_SCRIPT}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module


def q_values_for_state(agent, state):
    state_tensor = torch.as_tensor(
        np.asarray(state, dtype=np.float32),
        dtype=agent.tensor_dtype,
        device=agent.device,
    ).reshape(1, -1)

    with torch.no_grad():
        values = agent.q_net(state_tensor)

    return (
        values
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
        .astype(np.float64)
    )


def trace_voyage(
    module,
    *,
    voyage_id,
    loads_kw,
    base_config,
    agent,
):
    env = module.DqnMpcWeightEnv(
        loads_kw=loads_kw,
        base_config=base_config,
        initial_soc=0.55,
    )

    state = env.reset()
    rows = []

    while not env.done:
        decision_index = int(env.decision_index)

        current_load_kw = float(
            env.loads_kw[decision_index]
        )

        previous_load_kw = float(
            env.loads_kw[
                decision_index - 1
                if decision_index > 0
                else decision_index
            ]
        )

        load_delta_kw = (
            current_load_kw - previous_load_kw
        )

        q_values = q_values_for_state(
            agent,
            state,
        )

        action = int(np.argmax(q_values))

        base_row = {
            "voyage_id": voyage_id,
            "decision_index": decision_index,
            "current_load_kw": current_load_kw,
            "previous_load_kw": previous_load_kw,
            "load_delta_kw": load_delta_kw,
            "soc_before": float(
                env.current_soc
            ),
            "prev_fc_kw": float(
                env.previous_fc_kw
            ),
            "prev_batt_kw": float(
                env.previous_batt_kw
            ),
            "selected_action": action,
            "selected_action_name": (
                module.DQN_MPC_WEIGHT_ACTIONS[
                    action
                ].name
            ),
        }

        for action_id, value in enumerate(
            q_values
        ):
            base_row[
                f"q_A{action_id}"
            ] = float(value)

        try:
            next_state, reward, done, info = (
                env.step(action)
            )

        except module.MpcSolveFailure as error:
            base_row.update(
                {
                    "reward": float(
                        agent.config
                        .solver_failure_reward
                    ),
                    "solver_status": str(
                        error.solver_status
                    ),
                    "solver_failed": True,
                    "p_fc_kw": float(
                        env.previous_fc_kw
                    ),
                    "p_batt_kw": float(
                        env.previous_batt_kw
                    ),
                    "soc_after": float(
                        env.current_soc
                    ),
                    "solve_ms": float(
                        error.solve_ms
                    ),
                }
            )

            rows.append(base_row)

            print(
                f"[trace] {voyage_id} "
                f"FAILED at step "
                f"{decision_index} "
                f"SOC={env.current_soc:.6f} "
                f"action=A{action}",
                flush=True,
            )

            break

        base_row.update(
            {
                "reward": float(reward),
                "solver_status": str(
                    info["solver_status"]
                ),
                "solver_failed": False,
                "p_fc_kw": float(
                    info["p_fc_kw"]
                ),
                "p_batt_kw": float(
                    info["p_batt_kw"]
                ),
                "soc_after": float(
                    info["soc_after"]
                ),
                "fc_delta_kw": float(
                    info["p_fc_kw"]
                    - info["p_fc_prev_kw"]
                ),
                "solve_ms": float(
                    info["solve_ms"]
                ),
            }
        )

        rows.append(base_row)
        state = next_state

        if len(rows) % 1000 == 0:
            print(
                f"[trace] {voyage_id} "
                f"step={len(rows)} "
                f"SOC={env.current_soc:.4f} "
                f"action=A{action}",
                flush=True,
            )

    return pd.DataFrame(rows)


def main():
    module = load_training_module()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    split = module.load_voyage_split()

    base_config = (
        module.build_formal_mpc_config()
    )

    config = module.DQNTrainConfig(
        device="cpu",
    )

    runtime = (
        module.create_training_runtime(
            config
        )
    )

    agent = runtime.agent
    agent.load(MODEL_PATH)
    agent.q_net.eval()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Loaded model: {MODEL_PATH}",
        flush=True,
    )

    for voyage_id in TARGET_VOYAGES:
        print()
        print(
            f"Tracing {voyage_id}...",
            flush=True,
        )

        loads_kw = (
            module.load_voyage_loads(
                "validation",
                voyage_id,
                split=split,
            )
        )

        frame = trace_voyage(
            module,
            voyage_id=voyage_id,
            loads_kw=loads_kw,
            base_config=base_config,
            agent=agent,
        )

        output_path = (
            OUTPUT_DIR
            / f"{voyage_id}_trace.csv"
        )

        frame.to_csv(
            output_path,
            index=False,
        )

        print(
            f"Saved: {output_path}",
            flush=True,
        )

        print(
            "rows=",
            len(frame),
            "final_soc=",
            frame.iloc[-1]["soc_after"],
            "failed=",
            bool(
                frame.iloc[-1][
                    "solver_failed"
                ]
            ),
        )


if __name__ == "__main__":
    main()