from __future__ import annotations

import base64
import io
import json
import sys
from dataclasses import dataclass, field
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
from dqn.policies.epsilon_greedy import (  # noqa: E402
    EpsilonGreedyPolicy,
)
from dqn.utils.action_mapper import (  # noqa: E402
    DQN_MPC_WEIGHT_ACTIONS,
)
from dqn.utils.state_builder import (  # noqa: E402
    DQN_MPC_STATE_DIM,
    SOC_REFERENCE,
)
from envs.dqn_mpc_weight_env import (  # noqa: E402
    DqnMpcWeightEnv,
)
from mpc_solvers.mpc_qp_formulation import (  # noqa: E402
    QpMpcConfig,
)
from run_mpc_1s_n6_four_objective_sensitivity import (  # noqa: E402
    N6_STATE_COMMIT_TOLERANCES,
    build_sensitivity_cases,
    four_objective_config,
)


DEFAULT_SPLIT_JSON = (
    REPO_ROOT
    / "outputs"
    / "config"
    / "voyage_split_total_load_721.json"
)
DEFAULT_VOYAGE_DATA_DIR = (
    REPO_ROOT
    / "outputs"
    / "spline_1s_diagnostics"
    / "data"
    / "natural_clipped_by_voyage"
)
DEFAULT_BASELINE_OUTPUT_DIR = (
    REPO_ROOT
    / "outputs"
    / "dqn_mpc_mlp_10k_baseline"
)

ALLOWED_RUNTIME_SPLITS = ("train", "validation")
EXPECTED_SPLIT_COUNTS = (46, 13, 7)
STATE_DIM = DQN_MPC_STATE_DIM
ACTION_DIM = len(DQN_MPC_WEIGHT_ACTIONS)
FORMAL_DATA_DIRECTORY = (
    DEFAULT_VOYAGE_DATA_DIR
    .relative_to(REPO_ROOT)
    .as_posix()
)
FORMAL_TARGET_LOAD = "load_total_kw"
FORMAL_SAMPLE_INTERVAL_SECONDS = 1.0


@dataclass(frozen=True)
class VoyageSplit:
    train_voyages: tuple[str, ...]
    validation_voyages: tuple[str, ...]
    test_voyages: tuple[str, ...]
    excluded_voyages: tuple[str, ...]
    formal_1s_directory: str
    target_load: str
    sample_interval_seconds: float


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


class _ValidationMpcFailure(RuntimeError):
    def __init__(
        self,
        diagnostic: dict[str, object],
    ) -> None:
        super().__init__(
            "validation MPC solve failed"
        )
        self.diagnostic = diagnostic


def _tuple_of_unique_strings(
    payload: dict[str, object],
    key: str,
) -> tuple[str, ...]:
    values = payload.get(key)

    if not isinstance(values, list):
        raise ValueError(
            f"active split is missing list: {key}"
        )

    result = tuple(str(value) for value in values)

    if len(result) != len(set(result)):
        raise ValueError(
            f"active split contains duplicate values: {key}"
        )

    return result


def load_voyage_split(
    split_path: str | Path = DEFAULT_SPLIT_JSON,
) -> VoyageSplit:
    path = Path(split_path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    train = _tuple_of_unique_strings(
        payload,
        "train_voyages",
    )
    validation = _tuple_of_unique_strings(
        payload,
        "validation_voyages",
    )
    test = _tuple_of_unique_strings(
        payload,
        "test_voyages",
    )
    excluded = _tuple_of_unique_strings(
        payload,
        "excluded_voyages",
    )
    formal_1s_directory = str(
        payload.get("formal_1s_directory", "")
    ).replace("\\", "/")
    target_load = str(payload.get("target_load", ""))

    try:
        sample_interval_seconds = float(
            payload["sample_interval_seconds"]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "active split must define sample_interval_seconds"
        ) from error

    if formal_1s_directory != FORMAL_DATA_DIRECTORY:
        raise ValueError(
            "active split formal_1s_directory changed: "
            f"{formal_1s_directory!r}"
        )

    if target_load != FORMAL_TARGET_LOAD:
        raise ValueError(
            "active split target_load changed: "
            f"{target_load!r}"
        )

    if not np.isclose(
        sample_interval_seconds,
        FORMAL_SAMPLE_INTERVAL_SECONDS,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError(
            "active split sample interval must remain 1 s"
        )

    counts = (len(train), len(validation), len(test))
    if counts != EXPECTED_SPLIT_COUNTS:
        raise ValueError(
            "formal split must remain 46/13/7, "
            f"got {counts}"
        )

    train_set = set(train)
    validation_set = set(validation)
    test_set = set(test)

    if (
        train_set.intersection(validation_set)
        or train_set.intersection(test_set)
        or validation_set.intersection(test_set)
    ):
        raise ValueError(
            "train, validation, and test voyages must be disjoint"
        )

    all_voyages = train_set | validation_set | test_set
    if len(all_voyages) != sum(EXPECTED_SPLIT_COUNTS):
        raise ValueError(
            "formal split must contain 66 unique voyages"
        )

    if excluded:
        raise ValueError(
            "formal split currently requires excluded_voyages=[]"
        )

    for alias, expected in (
        ("train", train),
        ("validation", validation),
        ("test", test),
    ):
        if alias in payload:
            alias_values = tuple(
                str(value)
                for value in payload[alias]
            )
            if alias_values != expected:
                raise ValueError(
                    f"split alias does not match {alias}_voyages"
                )

    return VoyageSplit(
        train_voyages=train,
        validation_voyages=validation,
        test_voyages=test,
        excluded_voyages=excluded,
        formal_1s_directory=formal_1s_directory,
        target_load=target_load,
        sample_interval_seconds=(
            sample_interval_seconds
        ),
    )


def _allowed_voyages(
    split_name: str,
    split: VoyageSplit,
) -> tuple[str, ...]:
    name = str(split_name).strip().lower()

    if name not in ALLOWED_RUNTIME_SPLITS:
        raise ValueError(
            "DQN-MPC training loader only permits train or "
            f"validation; {name!r} data access is prohibited"
        )

    if name == "train":
        return split.train_voyages

    return split.validation_voyages


def load_voyage_loads(
    split_name: str,
    voyage_id: str,
    *,
    split: VoyageSplit,
    data_dir: str | Path = DEFAULT_VOYAGE_DATA_DIR,
) -> np.ndarray:
    allowed_voyages = _allowed_voyages(
        split_name,
        split,
    )
    voyage = str(voyage_id)

    if voyage not in allowed_voyages:
        raise ValueError(
            f"{voyage} does not belong to {split_name}"
        )

    root = Path(data_dir).resolve()
    manifest_path = root / "manifest.csv"
    manifest = pd.read_csv(
        manifest_path,
        usecols=[
            "voyage_id",
            "split",
            "output_csv",
        ],
    )
    rows = manifest.loc[
        manifest["voyage_id"].astype(str) == voyage
    ]

    if len(rows) != 1:
        raise ValueError(
            f"manifest must contain exactly one row for {voyage}"
        )

    manifest_split = str(
        rows.iloc[0]["split"]
    ).strip().lower()
    expected_split = str(split_name).strip().lower()

    if manifest_split != expected_split:
        raise ValueError(
            f"manifest split mismatch for {voyage}: "
            f"{manifest_split!r} != {expected_split!r}"
        )

    raw_path = Path(str(rows.iloc[0]["output_csv"]))
    voyage_path = (
        raw_path
        if raw_path.is_absolute()
        else REPO_ROOT / raw_path
    ).resolve()

    try:
        voyage_path.relative_to(root)
    except ValueError as error:
        raise ValueError(
            f"manifest path escapes formal data directory: "
            f"{voyage_path}"
        ) from error

    frame = pd.read_csv(
        voyage_path,
        usecols=[
            "voyage_id",
            "split",
            "time_s",
            split.target_load,
        ],
    )

    if len(frame) < 2:
        raise ValueError(
            f"{voyage} must contain at least two 1 s samples"
        )

    file_voyages = set(
        frame["voyage_id"].astype(str).unique()
    )
    if file_voyages != {voyage}:
        raise ValueError(
            f"{voyage_path} contains unexpected voyage IDs"
        )

    file_splits = set(
        frame["split"]
        .astype(str)
        .str.strip()
        .str.lower()
        .unique()
    )
    if file_splits != {expected_split}:
        raise ValueError(
            f"{voyage_path} contains unexpected split labels"
        )

    time_s = pd.to_numeric(
        frame["time_s"],
        errors="coerce",
    ).to_numpy(dtype=np.float64)
    loads_kw = pd.to_numeric(
        frame[split.target_load],
        errors="coerce",
    ).to_numpy(dtype=np.float64)

    if (
        not np.all(np.isfinite(time_s))
        or not np.all(np.isfinite(loads_kw))
    ):
        raise ValueError(
            f"{voyage_path} contains non-finite time or load"
        )

    if not np.allclose(
        np.diff(time_s),
        split.sample_interval_seconds,
        rtol=0.0,
        atol=1.0e-9,
    ):
        raise ValueError(
            f"{voyage_path} is not a continuous 1 s voyage"
        )

    return loads_kw


def build_formal_mpc_config() -> QpMpcConfig:
    cases = build_sensitivity_cases()

    if len(cases) != 1:
        raise ValueError(
            "formal Candidate C runner must expose one fixed case"
        )

    return four_objective_config(cases[0])


def _validate_fixed_dqn_design(
    config: DQNTrainConfig,
) -> None:
    if STATE_DIM != 11:
        raise ValueError("DQN-MPC state dimension must be 11")

    if ACTION_DIM != 7:
        raise ValueError("DQN-MPC action dimension must be 7")

    if str(config.network_type).strip().lower() != "mlp":
        raise ValueError("formal DQN-MPC network must be MLP")

    if tuple(config.mlp_hidden_dims) != (128, 64):
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
    base_config: QpMpcConfig,
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

    soc_after = _require_finite_scalar(
        info.get("soc_after"),
        "SOC",
    )
    soc_tolerance = float(
        N6_STATE_COMMIT_TOLERANCES["soc"]
    )
    if (
        soc_after < float(base_config.soc_min) - soc_tolerance
        or soc_after
        > float(base_config.soc_max) + soc_tolerance
    ):
        raise RuntimeError(
            "SOC crossed MPC hard bounds: "
            f"{soc_after} not in "
            f"[{base_config.soc_min}, {base_config.soc_max}]"
        )


def _validate_online_q_values(
    *,
    agent: DQNAgent,
    state: np.ndarray,
    context: str,
) -> None:
    state_tensor = torch.as_tensor(
        np.asarray(state, dtype=np.float32),
        dtype=agent.tensor_dtype,
        device=agent.device,
    ).reshape(1, -1)

    with torch.no_grad():
        q_values = agent.q_net(state_tensor)

    if not bool(torch.isfinite(q_values).all().item()):
        raise RuntimeError(
            f"{context} online Q values contain NaN or Inf"
        )


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

    for key in required:
        if key not in diagnostics:
            raise RuntimeError(
                f"missing DQN update diagnostic: {key}"
            )
        result[key] = _require_finite_scalar(
            diagnostics[key],
            key,
        )

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

    return {
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
) -> Path:
    directory = DEFAULT_BASELINE_OUTPUT_DIR

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
        initial_soc=0.55,
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
    done = False

    while not done:
        step_before = int(runtime.global_step)
        warmup = (
            step_before
            < int(runtime.config.warmup_steps)
        )
        _validate_online_q_values(
            agent=runtime.agent,
            state=state,
            context="training",
        )
        greedy_action = runtime.agent.greedy_action(state)
        action = runtime.policy.select_action(
            greedy_action=greedy_action,
            action_dim=ACTION_DIM,
            warmup=warmup,
        )
        next_state, reward, done, info = env.step(action)
        _validate_environment_step(
            action=action,
            reward=reward,
            next_state=next_state,
            info=info,
            base_config=base_config,
        )

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

        if not warmup:
            runtime.policy.step()

            if (
                len(runtime.replay_buffer)
                >= int(runtime.config.batch_size)
            ):
                batch = runtime.replay_buffer.sample(
                    runtime.config.batch_size
                )
                loss = runtime.agent.update(batch)
                _require_finite_scalar(loss, "training loss")
                _validate_latest_q_diagnostics(
                    runtime.agent
                )
                _validate_online_q_values(
                    agent=runtime.agent,
                    state=next_state,
                    context="post-update",
                )
                runtime.losses.append(float(loss))
                runtime.update_steps.append(
                    int(runtime.global_step)
                )

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

        state = next_state

    return {
        "voyage_id": str(voyage_id),
        "episode_reward": float(episode_reward),
        "mean_reward_per_step": float(
            episode_reward / episode_steps
        ),
        "episode_steps": int(episode_steps),
        "final_soc": float(env.current_soc),
        "global_step": int(runtime.global_step),
    }


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
        initial_soc=0.55,
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
    action_counts = np.zeros(
        ACTION_DIM,
        dtype=np.int64,
    )
    recent_steps: list[dict[str, object]] = []
    done = False

    while not done:
        _validate_online_q_values(
            agent=agent,
            state=state,
            context="validation",
        )
        action = agent.greedy_action(state)
        try:
            next_state, reward, done, info = env.step(
                action
            )
        except RuntimeError as error:
            if "MPC solve failed" not in str(error):
                raise

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
                "original_solver_status": (
                    "primal infeasible"
                    if "primal infeasible" in str(error)
                    else str(error)
                ),
                "original_error": str(error),
                "original_primal_residual": None,
                "original_dual_residual": None,
                "original_iterations": None,
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

            raise _ValidationMpcFailure(
                diagnostic
            ) from error

        _validate_environment_step(
            action=action,
            reward=reward,
            next_state=next_state,
            info=info,
            base_config=base_config,
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
        "episode_reward": float(episode_reward),
        "mean_reward_per_step": float(
            episode_reward / episode_steps
        ),
        "episode_steps": int(episode_steps),
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

    voyages = [
        run_validation_episode(
            voyage_id=voyage_id,
            loads_kw=load_voyage(voyage_id),
            base_config=base_config,
            agent=agent,
        )
        for voyage_id in ordered_voyages
    ]

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
    split_path: str | Path = DEFAULT_SPLIT_JSON,
    data_dir: str | Path = DEFAULT_VOYAGE_DATA_DIR,
) -> tuple[TrainingRuntime, dict[str, object]]:
    resolved_config = config or DQNTrainConfig()
    _validate_fixed_dqn_design(resolved_config)

    np.random.seed(int(resolved_config.seed))
    torch.manual_seed(int(resolved_config.seed))

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
        loads = load_voyage_loads(
            "train",
            voyage_id,
            split=split,
            data_dir=data_dir,
        )
        loaded_train_voyages.append(str(voyage_id))
        return loads

    def load_validation_voyage(
        voyage_id: str,
    ) -> np.ndarray:
        loads = load_voyage_loads(
            "validation",
            voyage_id,
            split=split,
            data_dir=data_dir,
        )
        loaded_validation_voyages.append(
            str(voyage_id)
        )
        return loads

    train_episodes = train_to_budget(
        voyage_ids=split.train_voyages,
        load_voyage=load_train_voyage,
        base_config=base_config,
        runtime=runtime,
    )
    validation_start_state_dict = (
        _serialize_online_state_dict(runtime.agent)
    )

    try:
        validation = validate_voyages(
            voyage_ids=split.validation_voyages,
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
                "gradient_update_count": len(
                    runtime.update_steps
                ),
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
            failure_diagnostic
        )
        raise RuntimeError(
            "validation MPC failure diagnostic saved "
            f"to {diagnostic_path}"
        ) from failure
    loss_statistics = _loss_statistics(runtime.losses)
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
        "train_voyage_count": len(split.train_voyages),
        "validation_voyage_count": len(
            split.validation_voyages
        ),
        "test_voyage_count": len(split.test_voyages),
        "training_episodes": train_episodes,
        "global_step": int(runtime.global_step),
        "gradient_update_count": len(
            runtime.update_steps
        ),
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
        "validation": validation,
    }

    return runtime, summary


def write_baseline_outputs(
    *,
    runtime: TrainingRuntime,
    summary: dict[str, object],
    output_dir: str | Path = (
        DEFAULT_BASELINE_OUTPUT_DIR
    ),
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
    model_path = directory / "model_final.pt"
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
        "model_final": model_path,
    }


def main() -> None:
    runtime, summary = train_dqn_mpc_mlp()
    output_paths = write_baseline_outputs(
        runtime=runtime,
        summary=summary,
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
