
from __future__ import annotations

import base64
import io
import json
import os
import random
import time
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from dqn.agents.dqn_agent import (  # noqa: E402
    DQNAgent,
    DQNTrainConfig,
)
from dqn.memory.replay_buffer import ReplayBuffer  # noqa: E402
from dqn.utils.training_metrics import LossAccumulator  # noqa: E402
from dqn.policies.epsilon_greedy import (  # noqa: E402
    EpsilonGreedyPolicy,
)
from dqn.utils.action_mapper import (  # noqa: E402
    DQN_MPC_WEIGHT_ACTIONS,
)
from dqn.utils.reward import (  # noqa: E402
    REWARD_Q_BATT,
    REWARD_Q_FC_VAR,
    REWARD_Q_H2,
    REWARD_Q_SOC,
    SOC_SOFT_MAX,
    SOC_SOFT_MIN,
)
from dqn.utils.state_builder import (  # noqa: E402
    DQN_MPC_STATE_DIM,
    SOC_REFERENCE,
)
from envs.dqn_mpc_weight_env import (  # noqa: E402
    DqnMpcWeightEnv,
    MpcSolveFailure,
)
from mpc_solvers.mpc_qp_formulation import (  # noqa: E402
    QpMpcConfig,
)
from mpc_solvers.formal_config import (  # noqa: E402
    SOC_SOFT_MAX as MPC_SOC_SOFT_MAX,
    SOC_SOFT_MIN as MPC_SOC_SOFT_MIN,
    build_formal_mpc_config,
)
from utils.formal_operating_dataset import (  # noqa: E402
    DEFAULT_OPERATING_DATASET_ROOT as FORMAL_OPERATING_DATASET_ROOT,
    DEFAULT_SPLIT_MANIFEST as FORMAL_SPLIT_MANIFEST,
    LOAD_COLUMN,
    OperatingSegmentSplit,
    PHYSICAL_INFEASIBLE_STRESS_CASES,
    effective_segments_for_split,
    load_formal_operating_split,
    load_operating_segment_loads as _load_operating_segment_loads,
)


DEFAULT_OPERATING_DATASET_ROOT = FORMAL_OPERATING_DATASET_ROOT
DEFAULT_SPLIT_MANIFEST = FORMAL_SPLIT_MANIFEST
DEFAULT_MLP_SINGLE_PASS_OUTPUT_DIR = (
    REPO_ROOT
    / "outputs"
    / "dqn_mpc_mlp_causal_soc_deadband_single_pass"
)

ALLOWED_RUNTIME_SPLITS = ("train", "validation")
STATE_DIM = DQN_MPC_STATE_DIM
ACTION_DIM = len(DQN_MPC_WEIGHT_ACTIONS)
FORMAL_DATA_DIRECTORY = DEFAULT_OPERATING_DATASET_ROOT.relative_to(REPO_ROOT).as_posix()
FORMAL_TARGET_LOAD = LOAD_COLUMN
FORMAL_SAMPLE_INTERVAL_SECONDS = 1.0
SAVE_REPLAY_BUFFER = True
TRAINING_STATE_FORMAT_VERSION = 2
VoyageSplit = OperatingSegmentSplit


@dataclass
class TrainingRuntime:
    agent: DQNAgent
    replay_buffer: ReplayBuffer
    policy: EpsilonGreedyPolicy
    config: DQNTrainConfig
    global_step: int = 0
    losses: list[float] = field(default_factory=list)
    update_steps: list[int] = field(default_factory=list)
    target_sync_steps: list[int] = field(
        default_factory=list
    )
    gradient_update_count: int = 0
    target_sync_count: int = 0
    loss_accumulator: LossAccumulator = field(default_factory=LossAccumulator)
    round_loss_accumulator: LossAccumulator = field(default_factory=LossAccumulator)
    pending_losses: list[torch.Tensor] = field(default_factory=list)
    round_progress: dict[str, object] = field(default_factory=dict)

    def record_loss(self, loss: torch.Tensor) -> None:
        self.pending_losses.append(loss.detach())
        if len(self.pending_losses) >= 1000:
            self.flush_loss_metrics()

    def flush_loss_metrics(self) -> None:
        if not self.pending_losses:
            return
        values = torch.stack(self.pending_losses).cpu().numpy()
        self.loss_accumulator.add(values)
        self.round_loss_accumulator.add(values)
        self.losses = list(self.loss_accumulator.recent)
        self.pending_losses.clear()


class _ValidationMpcFailure(RuntimeError):
    def __init__(
        self,
        diagnostic: dict[str, object],
    ) -> None:
        super().__init__(
            "validation MPC solve failed"
        )
        self.diagnostic = diagnostic


def load_voyage_split(
    split_path: str | Path = DEFAULT_SPLIT_MANIFEST,
) -> VoyageSplit:
    path = Path(split_path).resolve()
    if path.name == "sample_manifest.csv" and path.parent.name == "metadata":
        dataset_root = path.parent.parent
    else:
        raise ValueError(
            "formal DQN-MPC requires metadata/sample_manifest.csv"
        )
    return load_formal_operating_split(dataset_root)


def load_operating_segment_loads(
    split_name: str,
    segment_id: str,
    *,
    split: VoyageSplit,
    allow_test: bool = False,
) -> np.ndarray:
    return _load_operating_segment_loads(
        split_name,
        segment_id,
        split=split,
        allow_test=allow_test,
    )


def effective_split_statistics(
    split: VoyageSplit,
) -> dict[str, dict[str, int]]:
    """Scan normal-use train/validation segments without altering the manifest."""

    result: dict[str, dict[str, int]] = {}
    for split_name in ("train", "validation"):
        identifiers = effective_segments_for_split(split, split_name)
        point_count = sum(
            int(
                load_operating_segment_loads(
                    split_name,
                    segment_id,
                    split=split,
                ).size
            )
            for segment_id in identifiers
        )
        result[split_name] = {
            "segment_count": int(len(identifiers)),
            "point_count": int(point_count),
        }
    return result


def estimate_full_replay_bytes(config: DQNTrainConfig) -> int:
    """Estimate dense replay payload bytes for the fixed seven-state DQN."""

    capacity = int(config.buffer_size)
    state_bytes = 2 * capacity * STATE_DIM * np.dtype(np.float32).itemsize
    action_bytes = capacity * np.dtype(np.int64).itemsize
    reward_bytes = capacity * np.dtype(np.float32).itemsize
    done_bytes = capacity * np.dtype(np.bool_).itemsize
    return int(state_bytes + action_bytes + reward_bytes + done_bytes)


def formal_training_metadata(
    config: DQNTrainConfig,
    split: VoyageSplit,
) -> dict[str, object]:
    """Return self-describing metadata for formal summaries/checkpoints."""

    return {
        "gamma": float(config.gamma),
        "random_seed": int(config.seed),
        "training_config": asdict(config),
        "mpc_config": asdict(build_formal_mpc_config()),
        "common_reward": {
            "q_h2": float(REWARD_Q_H2),
            "q_batt": float(REWARD_Q_BATT),
            "q_soc": float(REWARD_Q_SOC),
            "q_fcvar": float(REWARD_Q_FC_VAR),
            "soc_deadband": [float(SOC_SOFT_MIN), float(SOC_SOFT_MAX)],
        },
        "mpc_soc_deadband": [
            float(MPC_SOC_SOFT_MIN),
            float(MPC_SOC_SOFT_MAX),
        ],
        "actions": [
            {
                "action_id": int(action.action_id),
                "name": str(action.name),
                "weights": list(action.as_tuple()),
            }
            for action in DQN_MPC_WEIGHT_ACTIONS
        ],
        "physical_infeasible_stress_cases": PHYSICAL_INFEASIBLE_STRESS_CASES,
        "effective_split_statistics": effective_split_statistics(split),
        "save_replay_buffer": bool(SAVE_REPLAY_BUFFER),
        "estimated_full_replay_bytes": estimate_full_replay_bytes(config),
    }


def _cpu_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        str(key): value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }


def _move_optimizer_state_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _runtime_update_count(runtime: TrainingRuntime) -> int:
    return max(
        int(getattr(runtime, "gradient_update_count", 0)),
        int(len(runtime.update_steps)),
    )


def _runtime_target_sync_count(runtime: TrainingRuntime) -> int:
    return max(
        int(getattr(runtime, "target_sync_count", 0)),
        int(len(runtime.target_sync_steps)),
    )


def save_training_state(
    *,
    runtime: TrainingRuntime,
    path: str | Path,
    completed_round: int,
    metadata: dict[str, object] | None = None,
    save_replay_buffer: bool = SAVE_REPLAY_BUFFER,
) -> Path:
    """Atomically save all state needed to continue formal training exactly."""

    if int(completed_round) < 0:
        raise ValueError("completed_round must be nonnegative")
    runtime.flush_loss_metrics()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    payload: dict[str, object] = {
        "format_version": TRAINING_STATE_FORMAT_VERSION,
        "completed_round": int(completed_round),
        "round_progress": runtime.round_progress,
        "loss_accumulator": runtime.loss_accumulator.state_dict(),
        "round_loss_accumulator": runtime.round_loss_accumulator.state_dict(),
        "latest_update_diagnostics": (
            _validate_latest_q_diagnostics(runtime.agent)
            if runtime.agent.latest_update_diagnostics else {}
        ),
        "online_q_network_state_dict": _cpu_state_dict(runtime.agent.q_net),
        "target_q_network_state_dict": _cpu_state_dict(runtime.agent.target_q_net),
        "optimizer_state_dict": runtime.agent.optimizer.state_dict(),
        "epsilon": float(runtime.policy.epsilon),
        "global_step": int(runtime.global_step),
        "gradient_update_count": _runtime_update_count(runtime),
        "target_sync_count": _runtime_target_sync_count(runtime),
        "random_seed": int(runtime.config.seed),
        "training_config": asdict(runtime.config),
        "training_metadata": dict(metadata or {"gamma": float(runtime.config.gamma)}),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
        "replay_buffer_saved": bool(save_replay_buffer),
    }
    if save_replay_buffer:
        payload["replay_buffer"] = runtime.replay_buffer.state_dict()

    temporary.unlink(missing_ok=True)
    try:
        with temporary.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def load_training_state(
    path: str | Path,
) -> tuple[TrainingRuntime, int, dict[str, object]]:
    """Restore a complete formal training runtime from an atomic checkpoint."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("training state checkpoint must contain a dictionary")
    required = {
        "format_version", "completed_round", "online_q_network_state_dict",
        "target_q_network_state_dict", "optimizer_state_dict", "epsilon",
        "global_step", "gradient_update_count", "target_sync_count",
        "training_config", "training_metadata", "python_rng_state",
        "numpy_rng_state", "torch_cpu_rng_state", "torch_cuda_rng_state_all",
        "replay_buffer_saved",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"training state checkpoint is missing keys: {sorted(missing)}")
    # Version 1 round-boundary states are used by existing checkpoints/tests.
    if int(payload["format_version"]) not in (1, TRAINING_STATE_FORMAT_VERSION):
        raise ValueError("unsupported training state checkpoint format")
    config_data = payload["training_config"]
    if not isinstance(config_data, dict):
        raise ValueError("training state configuration must be a dictionary")
    runtime = create_training_runtime(DQNTrainConfig(**config_data))
    runtime.agent.q_net.load_state_dict(payload["online_q_network_state_dict"])
    runtime.agent.target_q_net.load_state_dict(payload["target_q_network_state_dict"])
    runtime.agent.optimizer.load_state_dict(payload["optimizer_state_dict"])
    _move_optimizer_state_to_device(runtime.agent.optimizer, runtime.agent.device)
    runtime.policy.epsilon = float(payload["epsilon"])
    runtime.global_step = int(payload["global_step"])
    runtime.gradient_update_count = int(payload["gradient_update_count"])
    runtime.target_sync_count = int(payload["target_sync_count"])
    if int(payload['format_version']) >= 2:
        runtime.round_progress = dict(payload['round_progress'])
        runtime.loss_accumulator.load_state_dict(payload['loss_accumulator'])
        runtime.round_loss_accumulator.load_state_dict(payload['round_loss_accumulator'])
        runtime.losses = list(runtime.loss_accumulator.recent)
        runtime.agent.latest_update_diagnostics = dict(payload.get('latest_update_diagnostics', {}))
    if not bool(payload['replay_buffer_saved']) and runtime.global_step:
        raise ValueError('cannot resume training without the replay buffer')
    if bool(payload["replay_buffer_saved"]):
        replay_state = payload.get("replay_buffer")
        if not isinstance(replay_state, dict):
            raise ValueError("training state declares replay buffer but omits it")
        runtime.replay_buffer.load_state_dict(replay_state)
    random.setstate(payload["python_rng_state"])
    np.random.set_state(payload["numpy_rng_state"])
    torch.set_rng_state(payload["torch_cpu_rng_state"])
    cuda_rng = payload["torch_cuda_rng_state_all"]
    if cuda_rng is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_rng)
    metadata = payload["training_metadata"]
    if not isinstance(metadata, dict):
        raise ValueError("training state metadata must be a dictionary")
    return runtime, int(payload["completed_round"]), metadata


def next_round_id(completed_round: int) -> int:
    """Return the first not-yet-completed formal round after resume."""

    if int(completed_round) < 0:
        raise ValueError("completed_round must be nonnegative")
    return int(completed_round) + 1

def _validate_fixed_dqn_design(
    config: DQNTrainConfig,
) -> None:
    if STATE_DIM != 7:
        raise ValueError("DQN-MPC state dimension must be 7")



    network_type = str(config.network_type).strip().lower()
    if network_type not in {"mlp", "kan"}:
        raise ValueError("DQN-MPC network_type must be 'mlp' or 'kan'")

    if network_type == "mlp" and tuple(config.mlp_hidden_dims) != (128, 64):
        raise ValueError(
            "formal MLP hidden dimensions must be (128, 64)"
        )

    if config.state_normalization_enabled:
        raise ValueError(
            "DQN-MPC state normalization must remain disabled"
        )

    if config.double_dqn:
        raise ValueError("Double DQN must remain disabled")

    if config.dueling_dqn:
        raise ValueError("dueling DQN must remain disabled")

    if int(config.target_sync_interval) <= 0:
        raise ValueError(
            "target_sync_interval must be a positive integer"
        )


def create_training_runtime(
    config: DQNTrainConfig | None = None,
) -> TrainingRuntime:
    resolved = config or DQNTrainConfig()
    _validate_fixed_dqn_design(resolved)

    agent = DQNAgent(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        config=resolved,
    )
    replay_buffer = ReplayBuffer(resolved.buffer_size)
    policy = EpsilonGreedyPolicy(
        resolved.epsilon_start,
        resolved.epsilon_min,
        resolved.epsilon_decay,
    )

    return TrainingRuntime(
        agent=agent,
        replay_buffer=replay_buffer,
        policy=policy,
        config=resolved,
    )


def seed_training_rngs(config: DQNTrainConfig) -> None:
    """Seed every RNG captured by a formal resumable training state."""

    seed = int(config.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _require_finite_scalar(
    value: object,
    label: str,
) -> float:
    number = float(value)

    if not np.isfinite(number):
        raise RuntimeError(
            f"{label} must be finite, got {number}"
        )

    return number


def _validate_environment_step(
    *,
    action: int,
    reward: float,
    next_state: np.ndarray,
    info: dict[str, object],
) -> None:
    if not 0 <= int(action) < ACTION_DIM:
        raise RuntimeError(
            f"action_id must be in 0..{ACTION_DIM - 1}, "
            f"got {action}"
        )

    _require_finite_scalar(reward, "reward")

    state_array = np.asarray(
        next_state,
        dtype=np.float64,
    )
    if not np.all(np.isfinite(state_array)):
        raise RuntimeError(
            "next DQN state contains NaN or Inf"
        )

    solver_status = str(info.get("solver_status", ""))
    if not solver_status.lower().startswith("solved"):
        raise RuntimeError(
            f"MPC solve failed: status={solver_status!r}"
        )

    # Physical execution is checked once inside the shared environment, before
    # state commit, for training and both validation entrypoints alike.


def _validate_latest_q_diagnostics(
    agent: DQNAgent,
) -> dict[str, float]:
    required = (
        "q_value_mean",
        "q_value_std",
        "target_q_mean",
        "target_q_std",
    )
    diagnostics = agent.latest_update_diagnostics
    result: dict[str, float] = {}

    if any(key not in diagnostics for key in required):
        raise RuntimeError('missing DQN update diagnostic')
    values = [diagnostics[key] for key in required]
    if isinstance(values[0], torch.Tensor):
        values = torch.stack(values).detach().cpu().numpy()
    for key, value in zip(required, values):
        result[key] = _require_finite_scalar(value, key)

    return result


def _loss_statistics(
    losses: Sequence[float],
) -> dict[str, float | int | None]:
    values = np.asarray(losses, dtype=np.float64)

    if values.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "last": None,
            "recent_1000_mean": None,
        }

    if not np.all(np.isfinite(values)):
        raise RuntimeError(
            "training loss contains NaN or Inf"
        )

    result = {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "last": float(values[-1]),
        "recent_1000_mean": float(
            np.mean(values[-1000:])
        ),
    }
    return result


def _snapshot_online_parameters(
    agent: DQNAgent,
) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter
        in agent.q_net.named_parameters()
    }


def _online_parameters_changed(
    before: dict[str, torch.Tensor],
    agent: DQNAgent,
) -> bool:
    after = dict(agent.q_net.named_parameters())

    if before.keys() != after.keys():
        raise RuntimeError(
            "online Q-network parameter structure changed"
        )

    return any(
        not torch.equal(
            before[name],
            after[name].detach().cpu(),
        )
        for name in before
    )


def _optional_finite_float(
    value: object,
) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    return number if np.isfinite(number) else None


def _solver_result_diagnostic(
    result: object,
) -> dict[str, object]:
    info = getattr(result, "info", None)

    return {
        "status": str(
            getattr(info, "status", "")
        ),
        "primal_residual": _optional_finite_float(
            getattr(info, "prim_res", None)
        ),
        "dual_residual": _optional_finite_float(
            getattr(info, "dual_res", None)
        ),
        "iterations": int(
            getattr(info, "iter", 0)
        ),
    }


def _persistent_same_state_diagnostics(
    *,
    env: DqnMpcWeightEnv,
    load_forecast_kw: np.ndarray,
    current_soc: float,
    prev_fc_kw: float,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []

    for action in DQN_MPC_WEIGHT_ACTIONS:
        record: dict[str, object] = {
            "action_id": int(action.action_id),
            "name": str(action.name),
            "weights": {
                "q_h2": float(action.q_h2),
                "q_batt": float(action.q_batt),
                "q_soc": float(action.q_soc),
                "q_fc_var": float(action.q_fc_var),
            },
        }

        try:
            result, solve_ms = env.solver_bank.solve(
                action_id=action.action_id,
                load_forecast_kw=load_forecast_kw,
                current_soc=current_soc,
                prev_fc_kw=prev_fc_kw,
                soc_reference=SOC_REFERENCE,
            )
            record.update(
                _solver_result_diagnostic(result)
            )
            record["solve_ms"] = (
                _optional_finite_float(solve_ms)
            )
        except Exception as error:
            record["diagnostic_exception"] = (
                f"{type(error).__name__}: {error}"
            )

        results.append(record)

    return results


def _serialize_online_state_dict(
    agent: DQNAgent,
) -> str:
    buffer = io.BytesIO()
    state_dict = {
        key: value.detach().cpu()
        for key, value
        in agent.q_net.state_dict().items()
    }
    torch.save(state_dict, buffer)
    return base64.b64encode(
        buffer.getvalue()
    ).decode("ascii")


def _write_failure_diagnostic(
    diagnostic: dict[str, object],
    *,
    network_type: str,
) -> Path:
    backend = str(network_type).strip().lower()
    if backend not in {"mlp", "kan"}:
        raise ValueError("network_type must be 'mlp' or 'kan'")
    directory = (
        REPO_ROOT
        / "outputs"
        / f"dqn_mpc_{backend}_causal_soc_deadband_single_pass_failure"
    )

    if directory.exists():
        raise FileExistsError(
            "failure diagnostic output directory "
            f"already exists: {directory}"
        )

    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "failure_diagnostic.json"
    path.write_text(
        json.dumps(
            diagnostic,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def run_training_episode(
    *,
    voyage_id: str,
    loads_kw: Sequence[float] | np.ndarray,
    base_config: QpMpcConfig,
    runtime: TrainingRuntime,
) -> dict[str, object]:
    env = DqnMpcWeightEnv(
        loads_kw=loads_kw,
        base_config=base_config,
        initial_soc=SOC_REFERENCE,
    )
    state = env.reset()
    if not np.all(
        np.isfinite(
            np.asarray(state, dtype=np.float64)
        )
    ):
        raise RuntimeError(
            "initial DQN state contains NaN or Inf"
        )

    episode_reward = 0.0
    episode_steps = 0
    solver_failure_count = 0
    log_interval = int(
        runtime.config.log_window_steps
    )
    episode_start_time = time.perf_counter()
    action_counts = np.zeros(
        ACTION_DIM,
        dtype=np.int64,
    )
    min_soc = float(env.current_soc)
    done = False

    while not done:
        step_before = int(runtime.global_step)
        warmup = (
            step_before
            < int(runtime.config.warmup_steps)
        )
        action = runtime.policy.select_action(
            greedy_action=lambda: runtime.agent.greedy_action(state),
            action_dim=ACTION_DIM,
            warmup=warmup,
        )


        try:
            next_state, reward, done, info = env.step(action)

            _validate_environment_step(
                action=action,
                reward=reward,
                next_state=next_state,
                info=info,
            )

        except MpcSolveFailure as error:
            solver_failure_count += 1


            # The failed MPC decision was not executed, so the
            # environment remains at the pre-decision state.
            next_state = np.asarray(
                state,
                dtype=np.float32,
            ).copy()

            reward = float(
                runtime.config.solver_failure_reward
            )
            done = True

            info = {
                "solver_status": str(error.solver_status),
                "solver_failed": True,
                "action_id": int(action),
                "decision_index": int(
                    error.decision_index
                ),
            }

        runtime.replay_buffer.push(
            state,
            action,
            reward,
            done,
            next_state,
        )
        runtime.global_step += 1
        episode_steps += 1
        episode_reward += float(reward)
        action_counts[action] += 1
        min_soc = min(min_soc, float(env.current_soc))

        if not warmup:
            runtime.policy.step()

            if (
                len(runtime.replay_buffer)
                >= int(runtime.config.batch_size)
            ):
                batch = runtime.replay_buffer.sample(
                    runtime.config.batch_size
                )
                loss = runtime.agent.update(batch, defer_diagnostics=True)
                runtime.record_loss(loss)
                runtime.update_steps.append(
                    int(runtime.global_step)
                )
                del runtime.update_steps[:-1000]
                runtime.gradient_update_count += 1

        sync_interval = int(
            runtime.config.target_sync_interval
        )
        if (
            sync_interval > 0
            and runtime.global_step > 0
            and runtime.global_step % sync_interval == 0
        ):
            runtime.agent.sync_target_network()
            runtime.target_sync_steps.append(
                int(runtime.global_step)
            )
            del runtime.target_sync_steps[:-1000]
            runtime.target_sync_count += 1
        log_interval = int(
            runtime.config.log_window_steps
        )

        if (
            log_interval > 0
            and runtime.global_step % log_interval == 0
        ):
            runtime.flush_loss_metrics()
            if runtime.agent.latest_update_diagnostics:
                _validate_latest_q_diagnostics(runtime.agent)
            recent_losses = runtime.losses[
                -log_interval:
            ]

            if recent_losses:
                loss_text = (
                    f"{float(np.mean(recent_losses)):.6g}"
                )
            else:
                loss_text = "NA"

            reward_mean = float(
                episode_reward / episode_steps
            )

            elapsed_s = (
                    time.perf_counter()
                    - episode_start_time
            )

            steps_per_s = float(
                episode_steps / max(elapsed_s, 1.0e-12)
            )

            action_text = " ".join(
                f"A{action_id}={int(action_counts[action_id])}"
                for action_id in range(ACTION_DIM)
            )

            print(
                "[train] "
                f"step={runtime.global_step} "
                f"round={runtime.round_progress.get('current_round', 1)} "
                f"segment={int(runtime.round_progress.get('completed_segment_count', 0)) + 1} "
                f"voyage={voyage_id} "
                f"epsilon={runtime.policy.epsilon:.6f} "
                f"buffer={len(runtime.replay_buffer)} "
                f"loss_mean={loss_text} "
                f"reward_mean={reward_mean:.6f} "
                f"steps/s={steps_per_s:.2f} "
                f"{action_text} "
                f"solver_failures={solver_failure_count}",
                flush=True,
            )

        state = next_state

    runtime.flush_loss_metrics()
    result = {
        "voyage_id": str(voyage_id),
        "episode_reward": float(episode_reward),
        "mean_reward_per_step": float(
            episode_reward / episode_steps
        ),
        "episode_steps": int(episode_steps),
        "solver_failure_count": int(
            solver_failure_count
        ),
        "min_soc": float(min_soc),
        "final_soc": float(env.current_soc),
        "global_step": int(runtime.global_step),
    }
    for action_id in range(ACTION_DIM):
        result[f"action_count_A{action_id}"] = int(action_counts[action_id])
    return result


def train_to_budget(
    *,
    voyage_ids: Sequence[str],
    load_voyage: Callable[[str], np.ndarray],
    base_config: QpMpcConfig,
    runtime: TrainingRuntime,
) -> list[dict[str, object]]:
    ordered_voyages = tuple(
        str(voyage_id)
        for voyage_id in voyage_ids
    )

    if not ordered_voyages:
        raise ValueError(
            "training requires at least one voyage"
        )

    if int(runtime.config.max_steps) <= 0:
        raise ValueError("max_steps must be positive")

    episodes: list[dict[str, object]] = []
    voyage_index = 0

    while runtime.global_step < int(
        runtime.config.max_steps
    ):
        voyage_id = ordered_voyages[
            voyage_index % len(ordered_voyages)
        ]
        loads_kw = load_voyage(voyage_id)
        episodes.append(
            run_training_episode(
                voyage_id=voyage_id,
                loads_kw=loads_kw,
                base_config=base_config,
                runtime=runtime,
            )
        )
        voyage_index += 1

    return episodes


def train_complete_voyage_rounds(
    *,
    num_training_rounds: int,
    voyage_ids: Sequence[str],
    load_voyage: Callable[[str], np.ndarray],
    base_config: QpMpcConfig,
    runtime: TrainingRuntime,
    first_round_id: int = 1,
    on_round_complete: (
        Callable[[dict[str, object]], None] | None
    ) = None,
    on_segment_complete: Callable[[dict[str, object]], None] | None = None,
) -> list[dict[str, object]]:
    """Run each ordered voyage as one complete episode per training round."""
    if int(num_training_rounds) <= 0:
        raise ValueError("num_training_rounds must be positive")
    if int(first_round_id) <= 0:
        raise ValueError("first_round_id must be positive")
    ordered_voyages = tuple(str(voyage_id) for voyage_id in voyage_ids)
    if not ordered_voyages:
        raise ValueError("training rounds require at least one voyage")

    round_summaries: list[dict[str, object]] = []
    progress = getattr(runtime, 'round_progress', {})
    if progress and tuple(progress['train_segment_order']) != ordered_voyages:
        raise ValueError('resume train segment order does not match current split')
    completed_episodes = (int(first_round_id) - 1) * len(ordered_voyages)
    for round_id in range(int(first_round_id), int(num_training_rounds) + 1):
        update_start = _runtime_update_count(runtime)
        progress = getattr(runtime, 'round_progress', {})
        if progress and int(progress['current_round']) == round_id:
            episodes = list(progress['episodes'])
            if progress['completed_segment_ids'] != list(ordered_voyages[:len(episodes)]):
                raise ValueError('resume completed segment prefix is inconsistent')
            update_start = int(progress['round_start_update_count'])
        else:
            episodes = []
            runtime.round_loss_accumulator = LossAccumulator()
        completed_episodes += len(episodes)
        runtime.round_progress = {
            'current_round': round_id, 'completed_round': round_id - 1,
            'completed_segment_count': len(episodes),
            'completed_segment_ids': list(ordered_voyages[:len(episodes)]),
            'train_segment_order': list(ordered_voyages),
            'round_start_update_count': update_start, 'episodes': episodes,
            'round_finalization_pending': bool(on_round_complete and len(episodes) == len(ordered_voyages)),
        }
        for voyage_id in ordered_voyages[len(episodes):]:
            episode = run_training_episode(
                voyage_id=voyage_id,
                loads_kw=load_voyage(voyage_id),
                base_config=base_config,
                runtime=runtime,
            )
            episodes.append(episode)
            completed_episodes += 1
            runtime.round_progress.update(
                completed_segment_count=len(episodes),
                completed_segment_ids=list(ordered_voyages[:len(episodes)]),
                completed_round=round_id if len(episodes) == len(ordered_voyages) else round_id - 1,
                round_finalization_pending=bool(on_round_complete and len(episodes) == len(ordered_voyages)),
            )
            if on_segment_complete is not None:
                on_segment_complete(dict(runtime.round_progress))
        training_action_counts = {
            f"A{action_id}": int(
                sum(
                    int(episode.get(f"action_count_A{action_id}", 0))
                    for episode in episodes
                )
            )
            for action_id in range(ACTION_DIM)
        }
        round_q_diagnostics = (
            _validate_latest_q_diagnostics(runtime.agent)
            if (
                _runtime_update_count(runtime) > update_start
                and hasattr(runtime.agent, "latest_update_diagnostics")
            )
            else None
        )
        round_summary: dict[str, object] = {
            "round_id": round_id,
            "episodes": episodes,
            "completed_training_episodes": completed_episodes,
            "global_step": int(runtime.global_step),
            "epsilon": float(runtime.policy.epsilon),
            "gradient_update_count": _runtime_update_count(runtime),
            "gradient_update_count_this_round": _runtime_update_count(runtime) - update_start,
            "loss_statistics": runtime.round_loss_accumulator.summary(),
            "q_value_diagnostics": round_q_diagnostics,
            "training_solver_failure_count": int(
                sum(int(episode.get("solver_failure_count", 0)) for episode in episodes)
            ),
            "training_action_counts": training_action_counts,
            "training_min_soc": float(min(float(episode.get("min_soc", 0.55)) for episode in episodes)),
            "training_final_soc": {
                "mean": float(np.mean([float(episode.get("final_soc", 0.55)) for episode in episodes])),
                "min": float(np.min([float(episode.get("final_soc", 0.55)) for episode in episodes])),
                "max": float(np.max([float(episode.get("final_soc", 0.55)) for episode in episodes])),
            },
        }
        round_summaries.append(round_summary)
        if on_round_complete is not None:
            on_round_complete(round_summary)
    return round_summaries


def run_validation_episode(
    *,
    voyage_id: str,
    loads_kw: Sequence[float] | np.ndarray,
    base_config: QpMpcConfig,
    agent: DQNAgent,
) -> dict[str, object]:
    env = DqnMpcWeightEnv(
        loads_kw=loads_kw,
        base_config=base_config,
        initial_soc=SOC_REFERENCE,
    )
    state = env.reset()
    if not np.all(
        np.isfinite(
            np.asarray(state, dtype=np.float64)
        )
    ):
        raise RuntimeError(
            "initial validation state contains NaN or Inf"
        )

    episode_reward = 0.0
    episode_steps = 0
    solver_failure_count = 0
    completed = True
    failure_diagnostic: dict[str, object] | None = None

    action_counts = np.zeros(
        ACTION_DIM,
        dtype=np.int64,
    )
    recent_steps: list[dict[str, object]] = []
    done = False

    while not done:
        action = agent.greedy_action(state)
        try:
            next_state, reward, done, info = env.step(
                action
            )

        except MpcSolveFailure as error:
            decision_index = int(env.decision_index)
            execution_index = decision_index + 1

            load_forecast_kw = env._future_window(
                decision_index
            )
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

            failure_reward = float(
                agent.config.solver_failure_reward
            )

            diagnostic: dict[str, object] = {
                "voyage_id": str(voyage_id),
                "decision_index": decision_index,
                "execution_index": execution_index,
                "action_id": int(action),
                "current_soc": float(
                    env.current_soc
                ),
                "prev_fc_kw": float(
                    env.previous_fc_kw
                ),
                "previous_batt_kw": float(
                    env.previous_batt_kw
                ),
                "current_load_kw": current_load_kw,
                "previous_load_kw": previous_load_kw,
                "future_load_kw": [
                    float(value)
                    for value in load_forecast_kw
                ],
                "state": [
                    float(value)
                    for value in np.asarray(
                        state,
                        dtype=np.float64,
                    )
                ],
                "solver_status": str(
                    getattr(
                        error,
                        "solver_status",
                        str(error),
                    )
                ),
                "solver_failure_reward": (
                    failure_reward
                ),
                "solve_ms": getattr(
                    error,
                    "solve_ms",
                    None,
                ),
                "primal_residual": getattr(
                    error,
                    "primal_residual",
                    None,
                ),
                "dual_residual": getattr(
                    error,
                    "dual_residual",
                    None,
                ),
                "iterations": getattr(
                    error,
                    "iterations",
                    None,
                ),
                "original_error": str(error),
                "last_20_successful_steps": list(
                    recent_steps
                ),
            }

            diagnostic[
                "persistent_same_state_actions"
            ] = _persistent_same_state_diagnostics(
                env=env,
                load_forecast_kw=load_forecast_kw,
                current_soc=float(env.current_soc),
                prev_fc_kw=float(env.previous_fc_kw),
            )

            # The failed action counts as one terminal
            # validation decision. No fallback action is used.
            action_counts[action] += 1
            episode_reward += failure_reward
            episode_steps += 1
            solver_failure_count += 1
            completed = False
            failure_diagnostic = diagnostic
            done = True
            break

        _validate_environment_step(
            action=action,
            reward=reward,
            next_state=next_state,
            info=info,
        )
        action_counts[action] += 1
        episode_reward += float(reward)
        episode_steps += 1
        recent_steps.append(
            {
                "decision_index": int(
                    info["decision_index"]
                ),
                "execution_index": int(
                    info["execution_index"]
                ),
                "action_id": int(action),
                "load_kw": float(
                    info["load_actual_kw"]
                ),
                "p_fc_kw": float(info["p_fc_kw"]),
                "p_batt_kw": float(
                    info["p_batt_kw"]
                ),
                "soc_before": float(
                    info["soc_before"]
                ),
                "soc_after": float(
                    info["soc_after"]
                ),
                "fc_delta_kw": float(
                    info["p_fc_kw"]
                    - info["p_fc_prev_kw"]
                ),
            }
        )
        if len(recent_steps) > 20:
            del recent_steps[0]
        state = next_state

    result: dict[str, object] = {
        "voyage_id": str(voyage_id),
        "completed": bool(completed),
        "episode_reward": float(episode_reward),
        "mean_reward_per_step": float(
            episode_reward / episode_steps
        ),
        "episode_steps": int(episode_steps),
        "solver_failure_count": int(
            solver_failure_count
        ),
        "failure_diagnostic": failure_diagnostic,
        "final_soc": float(env.current_soc),
    }

    for action_id in range(ACTION_DIM):
        result[f"action_count_A{action_id}"] = int(
            action_counts[action_id]
        )

    return result


def validate_voyages(
    *,
    voyage_ids: Sequence[str],
    load_voyage: Callable[[str], np.ndarray],
    base_config: QpMpcConfig,
    agent: DQNAgent,
) -> dict[str, object]:
    ordered_voyages = tuple(
        str(voyage_id)
        for voyage_id in voyage_ids
    )

    if not ordered_voyages:
        raise ValueError(
            "validation requires at least one voyage"
        )

    voyages: list[dict[str, object]] = []

    for index, voyage_id in enumerate(
            ordered_voyages,
            start=1,
    ):
        voyage = run_validation_episode(
            voyage_id=voyage_id,
            loads_kw=load_voyage(voyage_id),
            base_config=base_config,
            agent=agent,
        )

        voyages.append(voyage)

        print(
            "[validation] "
            f"{index}/{len(ordered_voyages)} "
            f"voyage={voyage_id} "
            f"completed={voyage['completed']} "
            f"steps={voyage['episode_steps']} "
            f"reward_mean="
            f"{float(voyage['mean_reward_per_step']):.6f} "
            f"solver_failures="
            f"{voyage['solver_failure_count']}",
            flush=True,
        )

    total_reward = float(
        sum(
            float(voyage["episode_reward"])
            for voyage in voyages
        )
    )
    total_steps = int(
        sum(
            int(voyage["episode_steps"])
            for voyage in voyages
        )
    )

    result: dict[str, object] = {
        "voyages": voyages,
        "mean_episode_reward": float(
            np.mean(
                [
                    float(voyage["episode_reward"])
                    for voyage in voyages
                ]
            )
        ),
        "mean_reward_per_step": float(
            total_reward / total_steps
        ),
    }

    for action_id in range(ACTION_DIM):
        action_total = sum(
            int(
                voyage[
                    f"action_count_A{action_id}"
                ]
            )
            for voyage in voyages
        )
        result[f"action_fraction_A{action_id}"] = float(
            action_total / total_steps
        )

    return result


def train_dqn_mpc_mlp(
    *,
    config: DQNTrainConfig | None = None,
    split_path: str | Path = DEFAULT_SPLIT_MANIFEST,
) -> tuple[TrainingRuntime, dict[str, object]]:
    resolved_config = config or DQNTrainConfig()
    _validate_fixed_dqn_design(resolved_config)

    seed_training_rngs(resolved_config)

    split = load_voyage_split(split_path)
    runtime = create_training_runtime(resolved_config)
    base_config = build_formal_mpc_config()
    online_parameters_before = (
        _snapshot_online_parameters(runtime.agent)
    )
    loaded_train_voyages: list[str] = []
    loaded_validation_voyages: list[str] = []

    def load_train_voyage(
        voyage_id: str,
    ) -> np.ndarray:
        loads = load_operating_segment_loads(
            "train",
            voyage_id,
            split=split,
        )
        loaded_train_voyages.append(str(voyage_id))
        return loads

    def load_validation_voyage(
        voyage_id: str,
    ) -> np.ndarray:
        loads = load_operating_segment_loads(
            "validation",
            voyage_id,
            split=split,
        )
        loaded_validation_voyages.append(
            str(voyage_id)
        )
        return loads

    train_episodes = train_to_budget(
        voyage_ids=split.effective_train_segments,
        load_voyage=load_train_voyage,
        base_config=base_config,
        runtime=runtime,
    )
    validation_start_state_dict = (
        _serialize_online_state_dict(runtime.agent)
    )

    try:
        validation = validate_voyages(
            voyage_ids=split.effective_validation_segments,
            load_voyage=load_validation_voyage,
            base_config=base_config,
            agent=runtime.agent,
        )
    except _ValidationMpcFailure as failure:
        failure_diagnostic: dict[str, object] = {
            "training": {
                "seed": int(resolved_config.seed),
                "requested_max_steps": int(
                    resolved_config.max_steps
                ),
                "actual_global_step": int(
                    runtime.global_step
                ),
                "training_voyages": [
                    str(episode["voyage_id"])
                    for episode in train_episodes
                ],
                "gradient_update_count": _runtime_update_count(runtime),
                "target_sync_steps": list(
                    runtime.target_sync_steps
                ),
                "final_epsilon": float(
                    runtime.policy.epsilon
                ),
                "replay_buffer_size": len(
                    runtime.replay_buffer
                ),
            },
            "failure_snapshot": failure.diagnostic,
            (
                "validation_start_online_q_net_"
                "state_dict_torch_base64"
            ): validation_start_state_dict,
            "data_access": {
                "train_voyages": (
                    loaded_train_voyages
                ),
                "validation_voyages": (
                    loaded_validation_voyages
                ),
                "test_voyages": [],
            },
        }
        diagnostic_path = _write_failure_diagnostic(
            failure_diagnostic,
            network_type=resolved_config.network_type,
        )
        raise RuntimeError(
            "validation MPC failure diagnostic saved "
            f"to {diagnostic_path}"
        ) from failure
    loss_statistics = runtime.loss_accumulator.summary()
    final_q_diagnostics = (
        _validate_latest_q_diagnostics(runtime.agent)
    )
    online_network_updated = (
        _online_parameters_changed(
            online_parameters_before,
            runtime.agent,
        )
    )

    summary: dict[str, object] = {
        "seed": int(resolved_config.seed),
        "requested_max_steps": int(
            resolved_config.max_steps
        ),
        "actual_global_step": int(
            runtime.global_step
        ),
        "warmup_steps": int(
            resolved_config.warmup_steps
        ),
        "train_voyage_count": len(split.effective_train_segments),
        "validation_voyage_count": len(
            split.effective_validation_segments
        ),
        "test_voyage_count": len(split.test_segments),
        "training_episodes": train_episodes,
        "global_step": int(runtime.global_step),
        "gradient_update_count": _runtime_update_count(runtime),
        "target_sync_steps": list(
            runtime.target_sync_steps
        ),
        "final_epsilon": float(
            runtime.policy.epsilon
        ),
        "replay_buffer_size": len(
            runtime.replay_buffer
        ),
        "loss_statistics": loss_statistics,
        "final_update_diagnostics": (
            final_q_diagnostics
        ),
        "online_network_updated": bool(
            online_network_updated
        ),
        "training_voyages": [
            str(episode["voyage_id"])
            for episode in train_episodes
        ],
        "data_access": {
            "train_voyages": loaded_train_voyages,
            "validation_voyages": (
                loaded_validation_voyages
            ),
            "test_voyages": [],
        },
        "formal_configuration": formal_training_metadata(
            resolved_config,
            split,
        ),
        "validation": validation,
    }

    return runtime, summary


def write_baseline_outputs(
    *,
    runtime: TrainingRuntime,
    summary: dict[str, object],
    output_dir: str | Path,
) -> dict[str, Path]:
    directory = Path(output_dir)

    if directory.exists():
        raise FileExistsError(
            "baseline output directory already exists: "
            f"{directory}"
        )

    validation = summary.get("validation")
    if not isinstance(validation, dict):
        raise RuntimeError(
            "training summary is missing validation"
        )

    validation_voyages = validation.get("voyages")
    if not isinstance(validation_voyages, list):
        raise RuntimeError(
            "training summary is missing validation voyages"
        )

    validation_columns = [
        "voyage_id",
        "completed",
        "solver_failure_count",
        "episode_steps",
        "episode_reward",
        "mean_reward_per_step",
        "final_soc",
        *[
            f"action_count_A{action_id}"
            for action_id in range(ACTION_DIM)
        ],
    ]
    validation_frame = pd.DataFrame(
        validation_voyages,
        columns=validation_columns,
    )

    if len(validation_frame) != 13:
        raise RuntimeError(
            "formal validation output must contain 13 voyages"
        )

    numeric_columns = [
        column
        for column in validation_columns
        if column != "voyage_id"
    ]
    numeric_values = (
        validation_frame[numeric_columns]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=np.float64)
    )
    if not np.all(np.isfinite(numeric_values)):
        raise RuntimeError(
            "validation output contains NaN or Inf"
        )

    action_columns = [
        f"action_count_A{action_id}"
        for action_id in range(ACTION_DIM)
    ]
    if not np.array_equal(
        validation_frame[action_columns]
        .sum(axis=1)
        .to_numpy(dtype=np.int64),
        validation_frame["episode_steps"]
        .to_numpy(dtype=np.int64),
    ):
        raise RuntimeError(
            "validation action counts do not match "
            "episode steps"
        )

    directory.mkdir(parents=True, exist_ok=False)
    model_path = directory / "model_single_pass.pt"
    validation_path = (
        directory / "validation_by_voyage.csv"
    )
    summary_path = directory / "training_summary.json"

    runtime.agent.save(model_path)
    validation_frame.to_csv(
        validation_path,
        index=False,
    )
    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "training_summary": summary_path,
        "validation_by_voyage": validation_path,
        "model_single_pass": model_path,
    }


def main() -> None:
    runtime, summary = train_dqn_mpc_mlp()
    output_paths = write_baseline_outputs(
        runtime=runtime,
        summary=summary,
        output_dir=DEFAULT_MLP_SINGLE_PASS_OUTPUT_DIR,
    )
    print(
        json.dumps(
            {
                "summary": summary,
                "output_files": {
                    key: str(path)
                    for key, path in output_paths.items()
                },
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
