from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dqn.agents.dqn_agent import DQNTrainConfig
from formal_paths import formal_output_dir
from utils.physical_config import FUEL_CELL_RAMP_KW_PER_S
import train_dqn_mpc_mlp as training
import test_dqn_mpc_causal as validation_artifacts


NUM_TRAINING_ROUNDS = 2
NETWORK_TYPE = "mlp"
FORMAL_OUTPUT_DIR = formal_output_dir(NETWORK_TYPE)
Q_GAP_NEAR_ZERO_ATOL = 1.0e-6
RAPID_LOAD_DELTA_KW = FUEL_CELL_RAMP_KW_PER_S


def resume_round_id(runtime, completed_round: int) -> int:
    progress = getattr(runtime, 'round_progress', {})
    if progress.get('round_finalization_pending', False):
        return int(progress['current_round'])
    return training.next_round_id(completed_round)


def _action_fractions(frame: pd.DataFrame) -> dict[str, float]:
    total = len(frame)
    return {
        f"A{action_id}": float((frame["action_id"] == action_id).sum() / total)
        if total else 0.0
        for action_id in range(training.ACTION_DIM)
    }


def summarize_validation_traces(traces: list[pd.DataFrame]) -> dict[str, object]:
    """Summarize read-only greedy validation Q values and state regimes."""
    if not traces:
        raise ValueError("validation trace summary requires at least one trace")
    combined = pd.concat(traces, ignore_index=True)
    required = {"action_id", "q_gap", "soc_before", "load_delta_kw", "current_load_kw"}
    if not required.issubset(combined.columns):
        return {"available": False, "reason": "trace lacks Q-value diagnostic columns"}
    q_gap = pd.to_numeric(combined["q_gap"], errors="raise").to_numpy(dtype=float)
    if not np.all(np.isfinite(q_gap)):
        raise RuntimeError("validation Q-gap contains NaN or Inf")
    loads = pd.to_numeric(combined["current_load_kw"], errors="raise")
    low_load, high_load = (float(value) for value in loads.quantile([1 / 3, 2 / 3]))
    soc = pd.to_numeric(combined["soc_before"], errors="raise")
    delta = pd.to_numeric(combined["load_delta_kw"], errors="raise")
    regimes = {
        "soc": {
            "soc_lt_0_50": _action_fractions(combined.loc[soc < training.SOC_SOFT_MIN]),
            "soc_0_50_to_0_60": _action_fractions(combined.loc[(soc >= training.SOC_SOFT_MIN) & (soc <= training.SOC_SOFT_MAX)]),
            "soc_gt_0_60": _action_fractions(combined.loc[soc > training.SOC_SOFT_MAX]),
        },
        "transition": {
            "rapid_rise": _action_fractions(combined.loc[delta >= RAPID_LOAD_DELTA_KW]),
            "rapid_fall": _action_fractions(combined.loc[delta <= -RAPID_LOAD_DELTA_KW]),
            "stable": _action_fractions(combined.loc[delta.abs() < RAPID_LOAD_DELTA_KW]),
        },
        "load": {
            "low": _action_fractions(combined.loc[loads <= low_load]),
            "medium": _action_fractions(combined.loc[(loads > low_load) & (loads < high_load)]),
            "high": _action_fractions(combined.loc[loads >= high_load]),
        },
    }
    action_counts = {
        f"A{action_id}": int((combined["action_id"] == action_id).sum())
        for action_id in range(training.ACTION_DIM)
    }
    return {
        "available": True,
        "action_counts": action_counts,
        "action_fractions": _action_fractions(combined),
        "dominant_action_share": float(max(action_counts.values()) / len(combined)),
        "q_gap": {
            "mean": float(np.mean(q_gap)),
            "median": float(np.median(q_gap)),
            "p10": float(np.quantile(q_gap, 0.10)),
            "p90": float(np.quantile(q_gap, 0.90)),
            "near_zero_atol": Q_GAP_NEAR_ZERO_ATOL,
            "near_zero_count": int((q_gap <= Q_GAP_NEAR_ZERO_ATOL).sum()),
            "near_zero_fraction": float((q_gap <= Q_GAP_NEAR_ZERO_ATOL).mean()),
        },
        "load_tertiles_kw": {"low_high_boundary": low_load, "medium_high_boundary": high_load},
        "rapid_load_delta_kw": RAPID_LOAD_DELTA_KW,
        "regime_action_fractions": regimes,
    }


def run_round_boundary_training(
    *,
    split,
    runtime,
    base_config,
    output_dir: Path,
    load_train,
    load_validation,
    num_training_rounds: int = NUM_TRAINING_ROUNDS,
    first_round_id: int = 1,
    training_metadata: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    """Checkpoint every segment; save/evaluate immediately at round boundaries."""
    progress = getattr(runtime, 'round_progress', {})
    pending_round = int(progress['current_round']) if progress.get('round_finalization_pending') else None

    def save_segment(progress: dict[str, object]) -> None:
        training.save_training_state(
            runtime=runtime, path=output_dir / 'training_state_latest.pt',
            completed_round=int(progress['completed_round']), metadata=training_metadata,
        )

    def save_and_validate_round(
        round_summary: dict[str, object],
    ) -> None:
        round_dir = output_dir / f"round_{round_summary['round_id']}"
        # Only an explicitly resumed, unfinished round may reuse its artifacts.
        finishing_pending_round = pending_round == int(round_summary['round_id'])
        round_dir.mkdir(exist_ok=finishing_pending_round)
        trace_dir = round_dir / "traces"
        plot_dir = round_dir / "plots"
        trace_dir.mkdir(exist_ok=finishing_pending_round)
        plot_dir.mkdir(exist_ok=finishing_pending_round)
        runtime.agent.save(round_dir / f"model_round{round_summary['round_id']}.pt")
        state_path = training.save_training_state(
            runtime=runtime,
            path=round_dir / f"training_state_round{round_summary['round_id']}.pt",
            completed_round=int(round_summary["round_id"]),
            metadata=training_metadata,
        )
        round_summary["training_state_path"] = str(state_path)
        validation_artifacts.PLOT_DIR = plot_dir
        validation_voyages = []
        validation_traces = []
        validation_segments = getattr(
            split,
            "effective_validation_segments",
            split.validation_segments,
        )
        for voyage_id in validation_segments:
            result, trace = validation_artifacts.run_test_episode(
                voyage_id=voyage_id,
                loads_kw=load_validation(voyage_id),
                base_config=base_config,
                agent=runtime.agent,
            )
            trace.to_csv(trace_dir / f"{voyage_id}_trace.csv", index=False)
            validation_artifacts.plot_power_allocation(voyage_id, trace)
            validation_artifacts.plot_soc_trajectory(voyage_id, trace)
            validation_voyages.append(result)
            # Full per-second artifacts are on disk. Keep only columns needed
            # for exact validation quantiles/regime counts, never full traces.
            diagnostic_columns = ['action_id', 'q_gap', 'soc_before', 'load_delta_kw', 'current_load_kw']
            validation_traces.append(trace.loc[:, trace.columns.intersection(diagnostic_columns)].copy())
            del trace
        validation = {
            "voyages": validation_voyages,
            "q_value_diagnostics": summarize_validation_traces(validation_traces),
        }
        pd.DataFrame(validation["voyages"]).to_csv(round_dir / "validation_by_voyage.csv", index=False)
        (round_dir / "validation_summary.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        round_summary["validation"] = validation
        runtime.round_progress['round_finalization_pending'] = False
        training.save_training_state(
            runtime=runtime, path=state_path,
            completed_round=int(round_summary['round_id']), metadata=training_metadata,
        )
        save_segment(runtime.round_progress)

    rounds = training.train_complete_voyage_rounds(
        num_training_rounds=num_training_rounds,
        voyage_ids=getattr(
            split,
            "effective_train_segments",
            split.train_segments,
        ),
        load_voyage=load_train,
        base_config=base_config,
        runtime=runtime,
        on_round_complete=save_and_validate_round,
        on_segment_complete=save_segment,
        first_round_id=first_round_id,
    )
    (output_dir / "training_summary.json").write_text(
        json.dumps(
            {
                "num_training_rounds": num_training_rounds,
                "first_round_id": first_round_id,
                "rounds": rounds,
                "formal_configuration": training_metadata,
                "test_voyages": [],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return rounds


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Formal causal DQN-MPC training with segment-boundary resume",
    )
    parser.add_argument(
        "--resume-training-state",
        type=Path,
        default=None,
        help="atomic training_state_latest.pt or training_state_roundX.pt; resumes at the next segment",
    )
    args = parser.parse_args(argv)
    split = training.load_voyage_split()
    if args.resume_training_state is None:
        config = DQNTrainConfig(network_type=NETWORK_TYPE)
        training.seed_training_rngs(config)
        runtime = training.create_training_runtime(config)
        first_round_id = 1
        output_dir = FORMAL_OUTPUT_DIR
        if output_dir.exists():
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = output_dir.with_name(f"{output_dir.name}_{stamp}")
    else:
        runtime, completed_round, saved_metadata = training.load_training_state(
            args.resume_training_state,
        )
        if runtime.config.network_type != NETWORK_TYPE:
            raise ValueError("resume checkpoint network type does not match formal entrypoint")
        if runtime.config != DQNTrainConfig(network_type=NETWORK_TYPE):
            raise ValueError('resume checkpoint training configuration differs from the formal configuration')
        first_round_id = resume_round_id(runtime, completed_round)
        if first_round_id > NUM_TRAINING_ROUNDS:
            raise ValueError("resume checkpoint already completed all formal rounds")
        state_parent = args.resume_training_state.resolve().parent
        output_dir = state_parent.parent if state_parent.name.startswith('round_') else state_parent
    base_config = training.build_formal_mpc_config()
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    if (output_dir / f"round_{first_round_id}").exists() and not runtime.round_progress.get('round_finalization_pending'):
        raise FileExistsError(output_dir / f"round_{first_round_id}")
    metadata = training.formal_training_metadata(runtime.config, split)
    if args.resume_training_state is not None and saved_metadata != metadata:
        raise ValueError('resume checkpoint formal metadata differs from the current configuration/data split')

    def load_train(voyage_id: str):
        return training.load_operating_segment_loads(
            "train",
            voyage_id,
            split=split,
        )

    def load_validation(voyage_id: str):
        return training.load_operating_segment_loads(
            "validation",
            voyage_id,
            split=split,
        )

    run_round_boundary_training(
        split=split,
        runtime=runtime,
        base_config=base_config,
        output_dir=output_dir,
        load_train=load_train,
        load_validation=load_validation,
        first_round_id=first_round_id,
        training_metadata=metadata,
    )


if __name__ == "__main__":
    main()
