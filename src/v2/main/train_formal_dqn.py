"""Independent formal v2 DQN training entrypoint.

This module deliberately does not import or delegate to the legacy trainer.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Sequence

import numpy as np

from ..config import TAU_LPF_SECONDS, TimeScaleConfig
from ..control.causal_base_load import CausalBaseLoadFilter
from ..data.formal_training_dataset import FormalEpisode, FormalTrainingDataset
from ..data.supervisory_rules import OperatingMode, normalize_onboard_load_kw
from ..dqn.action_space import FINAL_DQN_ACTION_CATALOG
from ..dqn.history import FormalStateHistory
from ..dqn.state import (
    FORMAL_FRAME_DIMENSION,
    FORMAL_STATE_DIMENSION,
    FORMAL_STATE_HISTORY_LENGTH,
    OperatingHistorySample,
    build_formal_operating_frame,
)
from ..envs.formal_episode import FormalEpisodeBackend, build_formal_nonlinear_mpc
from ..evaluation.formal_policy import build_formal_environment as _environment
from ..preflight import assess_formal_training_preflight, require_formal_training_ready
from ..economics import RewardScaleCalibration
from ..training.checkpoint import load_checkpoint, save_checkpoint
from ..training.diagnostics import (
    DqnOptimizationDiagnostics,
    summarize_actions,
)
from ..training.dqn import DqnAgent, DqnTrainingConfig, epsilon_at_global_step
from ..training.experiments import DqnExperimentProfile, history_study_profile
from ..training.schedule import EpisodeShuffleSchedule


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_POWER_ROOT = REPOSITORY_ROOT / "data" / "processed" / "operating_dataset_zero_boundary_v2"
DEFAULT_AIS_ROOT = REPOSITORY_ROOT / "data" / "processed" / "operating_dataset_zero_boundary_v2_ais"
DEFAULT_MODE_ROOT = REPOSITORY_ROOT / "data" / "processed" / "operating_dataset_zero_boundary_v2_modes"
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "outputs" / "v2_formal_dqn_v3"


@dataclass(frozen=True)
class ValidationSummary:
    raw_economic_cost_cny: float
    failure_penalty_score: float
    learning_reward: float
    transitions: int
    completed_episodes: int
    failed_episodes: int
    greedy_action_indices: tuple[int, ...] = ()

    @property
    def completion_rate(self) -> float:
        total = self.completed_episodes + self.failed_episodes
        return float(self.completed_episodes / total) if total else 0.0

    @classmethod
    def empty(cls) -> "ValidationSummary":
        return cls(0.0, 0.0, 0.0, 0, 0, 0, ())


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Formal v2 S8/36-action DQN trainer")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--preflight-only", action="store_true")
    modes.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--power-root", type=Path, default=DEFAULT_POWER_ROOT)
    parser.add_argument("--ais-root", type=Path, default=DEFAULT_AIS_ROOT)
    parser.add_argument("--mode-root", type=Path, default=DEFAULT_MODE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--rounds", type=_positive_int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log-every", type=_positive_int, default=1)
    parser.add_argument("--resume", type=Path)
    return parser


def _validate_all_states(episodes: tuple[FormalEpisode, ...]) -> int:
    checked = 0
    for episode in episodes:
        history: list[OperatingHistorySample] = []
        state_history = FormalStateHistory()
        for time_s, load_kw, speed_kn, fc_kw, mode_value in zip(
            episode.time_s,
            episode.load_kw,
            episode.speed_kn,
            episode.fc_power_kw,
            episode.operating_mode,
        ):
            mode = OperatingMode(mode_value)
            if mode is not OperatingMode.ONBOARD:
                history.clear()
                state_history.reset()
                continue
            normalized_load = normalize_onboard_load_kw(float(load_kw))
            sample = OperatingHistorySample(
                float(time_s),
                0.60,
                float(fc_kw),
                0.0,
                normalized_load,
                normalized_load,
            )
            history.append(sample)
            # Only the frozen 150 s causal window can affect S8.
            history = history[-6:]
            frame = build_formal_operating_frame(
                tuple(history),
                current_time_seconds=float(time_s),
                speed_kn=float(speed_kn),
            )
            state = state_history.encode(frame)
            if (
                len(state) != FORMAL_STATE_DIMENSION
                or not np.isfinite(np.asarray(state)).all()
            ):
                raise ValueError(f"{episode.sample_id}: nonfinite formal state")
            state_history.commit(frame)
            checked += 1
    return checked


def _solver_reproducibility_check(train: tuple[FormalEpisode, ...]) -> int:
    representative: float | None = None
    for episode in train:
        values = episode.load_kw[episode.load_kw > 1.0]
        if len(values):
            representative = float(values[0])
            break
    if representative is None:
        raise ValueError("Train contains no positive sailing/islanded load")

    checked = 0
    for action_index in (0, len(FINAL_DQN_ACTION_CATALOG) // 2, len(FINAL_DQN_ACTION_CATALOG) - 1):
        action = FINAL_DQN_ACTION_CATALOG[action_index]
        commands: list[np.ndarray] = []
        for _ in range(2):
            base_filter = CausalBaseLoadFilter(
                sample_seconds=30.0, tau_seconds=TAU_LPF_SECONDS
            )
            plan = build_formal_nonlinear_mpc().solve(
                observed_load_kw=representative,
                current_soc=0.60,
                previous_executed_p_fc_kw=0.0,
                weights=action.to_mpc_weights(),
                base_load_filter=base_filter,
            )
            command = plan.first_command()
            commands.append(
                np.asarray(
                    (command.p_fc_kw, command.p_batt_bus_kw, command.predicted_next_soc),
                    dtype=float,
                )
            )
        if not np.allclose(commands[0], commands[1], rtol=0.0, atol=1.0e-9):
            raise RuntimeError(f"solver is not reproducible for {action.action_id}")
        checked += 1
    return checked


def _live_preflight(
    args: argparse.Namespace,
    *,
    timescale: TimeScaleConfig | None = None,
) -> tuple[object, FormalTrainingDataset, tuple[FormalEpisode, ...], tuple[FormalEpisode, ...]]:
    report = assess_formal_training_preflight()
    dataset = FormalTrainingDataset.open(args.power_root, args.ais_root, args.mode_root)
    train = dataset.load_train()
    validation = dataset.load_validation()
    scale = TimeScaleConfig.formal_baseline() if timescale is None else timescale
    if type(scale) is not TimeScaleConfig:
        raise TypeError("timescale must be an exact TimeScaleConfig or None")
    train_macro_transitions = dataset.train_macro_transition_count(
        scale.dqn_switch_steps
    )
    state_count = _validate_all_states(train)
    solver_cases = _solver_reproducibility_check(train)
    mode_counts = {mode: 0 for mode in OperatingMode}
    for episode in train:
        for mode_value in episode.operating_mode:
            mode_counts[OperatingMode(mode_value)] += 1
    if dataset.opened_test_payloads != 0:
        raise RuntimeError("Test payload was opened during formal preflight")
    print(
        f"FORMAL_TRAINING={report.formal_training} "
        f"train_episodes={len(train)} validation_episodes={len(validation)} "
        f"train_steps={dataset.train_supervisory_steps} "
        f"train_macro_transitions={train_macro_transitions} "
        f"finite_states={state_count} solver_reproducibility_cases={solver_cases} "
        f"train_onboard_steps={mode_counts[OperatingMode.ONBOARD]} "
        f"train_shore_pending_steps={mode_counts[OperatingMode.SHORE_PENDING]} "
        f"train_shore_steps={mode_counts[OperatingMode.SHORE_CHARGING]} "
        f"train_unresolved_steps={dataset.unresolved_mode_counts['train']} "
        f"validation_unresolved_steps={dataset.unresolved_mode_counts['validation']} "
        f"test_payloads_opened={dataset.opened_test_payloads}",
        flush=True,
    )
    return report, dataset, train, validation


def _smoke(
    train: tuple[FormalEpisode, ...],
    *,
    timescale: TimeScaleConfig | None = None,
) -> None:
    selected: tuple[FormalEpisode, int, int] | None = None
    for episode in train:
        modes = tuple(OperatingMode(value) for value in episode.operating_mode)
        for shore_index, mode in enumerate(modes):
            if mode is not OperatingMode.SHORE_PENDING:
                continue
            start = max(0, int(shore_index) - 4)
            if not all(value is OperatingMode.ONBOARD for value in modes[start:shore_index]):
                continue
            reentry = shore_index
            while reentry < len(modes) and modes[reentry] in {
                OperatingMode.SHORE_PENDING,
                OperatingMode.SHORE_CHARGING,
            }:
                reentry += 1
            if shore_index > start and (
                reentry == len(modes) or modes[reentry] is OperatingMode.ONBOARD
            ):
                selected = (episode, start, min(len(modes), reentry + 1))
                break
        if selected is not None:
            break
    if selected is None:
        raise ValueError("Train has no bounded window covering both sailing and shore")
    episode, start, stop = selected
    bounded = FormalEpisode(
        episode.parent,
        f"{episode.sample_id}__smoke",
        episode.split,
        episode.timestamp[start:stop],
        episode.time_s[start:stop].copy(),
        episode.load_kw[start:stop].copy(),
        episode.speed_kn[start:stop].copy(),
        episode.speed_provenance[start:stop],
        episode.fc_power_kw[start:stop].copy(),
        episode.battery_bus_kw[start:stop].copy(),
        episode.operating_mode[start:stop],
        episode.mode_reason[start:stop],
    )
    if timescale is None:
        backend, environment = _environment(bounded)
    else:
        backend, environment = _environment(bounded, timescale=timescale)
    environment.reset()
    transition = environment.step(FINAL_DQN_ACTION_CATALOG[0].action_id)
    if (
        backend.mode_counts[OperatingMode.ONBOARD] == 0
        or (
            backend.mode_counts[OperatingMode.SHORE_PENDING]
            + backend.mode_counts[OperatingMode.SHORE_CHARGING]
        ) == 0
        or any(
            power != 0.0
            for power, mode in zip(
                backend.executed_fc_power_kw,
                bounded.operating_mode[: backend.index],
            )
            if OperatingMode(mode) in {
                OperatingMode.SHORE_PENDING,
                OperatingMode.SHORE_CHARGING,
            }
        )
    ):
        raise RuntimeError("integrated smoke did not exercise the shore FC interlock")
    epsilon = epsilon_at_global_step(0)
    print(
        f"SMOKE=PASS episode={episode.sample_id} steps={transition.executed_mpc_steps} "
        f"mpc_solves={backend.mpc_solve_count} "
        f"raw_economic_cost_cny={transition.raw_economic_cost_cny:.9f} "
        f"failure_penalty_score={transition.failure_penalty_score:.9f} "
        f"learning_reward={transition.learning_reward:.9f} "
        f"episode_completed={'YES' if transition.episode_completed else 'NO'} "
        f"failure_kind={transition.failure_kind or 'NONE'} "
        f"epsilon={epsilon:.6f} greedy_rate={1.0 - epsilon:.6f} "
        f"paused_shore_steps={backend.mode_counts[OperatingMode.SHORE_PENDING] + backend.mode_counts[OperatingMode.SHORE_CHARGING]} "
        "gradient_updates=0 checkpoint=NONE",
        flush=True,
    )


def _progress_line(
    *,
    round_index: int,
    rounds: int,
    episode_position: int,
    episode_count: int,
    episode: FormalEpisode,
    episode_macro: int,
    global_step: int,
    action_index: int,
    epsilon: float,
    transition: object,
    episode_raw_economic_cost_cny: float,
    episode_failure_penalty_score: float,
    episode_learning_reward: float,
    replay_reward: float,
    optimization_diagnostics: tuple[DqnOptimizationDiagnostics, ...],
    replay_size: int,
    backend: FormalEpisodeBackend,
    mode_delta: dict[OperatingMode, int],
    elapsed_s: float,
) -> None:
    action = FINAL_DQN_ACTION_CATALOG[action_index]
    last_index = max(0, backend.index - 1)
    load = float(backend.load_kw[last_index])
    state = transition.next_state
    frame_start = FORMAL_FRAME_DIMENSION * (FORMAL_STATE_HISTORY_LENGTH - 1)
    current_frame = state[frame_start : frame_start + FORMAL_FRAME_DIMENSION]
    terminal = "episode_end" if transition.done else "continue"
    if optimization_diagnostics:
        names = tuple(DqnOptimizationDiagnostics.__dataclass_fields__)
        averaged = {
            name: float(
                np.mean([getattr(value, name) for value in optimization_diagnostics])
            )
            for name in names
        }
        diagnostic_text = " ".join(
            (
                f"loss={averaged['loss']:.9g}",
                f"td_abs_p50={averaged['td_abs_p50']:.9g}",
                f"td_abs_p95={averaged['td_abs_p95']:.9g}",
                f"td_abs_max={averaged['td_abs_max']:.9g}",
                f"q_common_mean={averaged['q_common_mean']:.9g}",
                f"q_advantage_std={averaged['q_advantage_std']:.9g}",
                f"q_margin_p50={averaged['q_margin_p50']:.9g}",
                f"gradient_norm_preclip={averaged['gradient_norm_preclip']:.9g}",
            )
        )
    else:
        diagnostic_text = " ".join(
            f"{name}=NA"
            for name in (
                "loss",
                "td_abs_p50",
                "td_abs_p95",
                "td_abs_max",
                "q_common_mean",
                "q_advantage_std",
                "q_margin_p50",
                "gradient_norm_preclip",
            )
        )
    print(
        f"round={round_index + 1}/{rounds} episode={episode_position + 1}/{episode_count} "
        f"segment={episode.sample_id} macro={episode_macro} global={global_step} "
        f"action_index={action_index} action_id={action.action_id} weights={action.as_tuple()} "
        f"epsilon={epsilon:.6f} greedy_rate={1.0 - epsilon:.6f} "
        f"raw_economic_cost_cny={transition.raw_economic_cost_cny:.9f} "
        f"failure_penalty_score={transition.failure_penalty_score:.9f} "
        f"learning_reward={transition.learning_reward:.9f} "
        f"replay_reward={replay_reward:.9f} "
        f"episode_raw_economic_cost_cny={episode_raw_economic_cost_cny:.9f} "
        f"episode_failure_penalty_score={episode_failure_penalty_score:.9f} "
        f"episode_learning_reward={episode_learning_reward:.9f} "
        f"episode_completed={'YES' if transition.episode_completed else 'NO'} "
        f"failure_kind={transition.failure_kind or 'NONE'} "
        f"{diagnostic_text} replay={replay_size} soc={current_frame[0]:.9f} load_kw={load:.6f} "
        f"base_kw={current_frame[1] * 600.0:.6f} fc_kw={backend.previous_fc_kw:.6f} "
        f"delta_fc_kw={current_frame[6] * 600.0:.6f} executed_steps={transition.executed_mpc_steps} "
        f"onboard_steps={mode_delta[OperatingMode.ONBOARD]} "
        f"shore_pending_steps={mode_delta[OperatingMode.SHORE_PENDING]} "
        f"shore_steps={mode_delta[OperatingMode.SHORE_CHARGING]} "
        f"unresolved_steps={mode_delta[OperatingMode.UNRESOLVED]} terminal={terminal} "
        f"elapsed_s={elapsed_s:.1f}",
        flush=True,
    )


def _evaluate_validation(
    agent: DqnAgent,
    validation: tuple[FormalEpisode, ...],
    *,
    timescale: TimeScaleConfig | None = None,
) -> ValidationSummary:
    raw_economic_cost_cny = 0.0
    failure_penalty_score = 0.0
    learning_reward = 0.0
    transitions = 0
    completed_episodes = 0
    failed_episodes = 0
    greedy_actions: list[int] = []
    for episode in validation:  # Manifest order is fixed; never shuffled.
        if timescale is None:
            _, environment = _environment(episode)
        else:
            _, environment = _environment(episode, timescale=timescale)
        state = environment.reset()
        done = False
        while not done:
            action_index = agent.greedy_action(np.asarray(state, dtype=np.float32))
            greedy_actions.append(action_index)
            transition = environment.step(FINAL_DQN_ACTION_CATALOG[action_index].action_id)
            raw_economic_cost_cny += transition.raw_economic_cost_cny
            failure_penalty_score += transition.failure_penalty_score
            learning_reward += transition.learning_reward
            transitions += 1
            state = transition.next_state
            done = transition.done
        if transition.failed:
            failed_episodes += 1
        elif transition.episode_completed:
            completed_episodes += 1
        else:  # pragma: no cover - environment contract guard
            raise RuntimeError("validation episode ended without terminal outcome")
    return ValidationSummary(
        raw_economic_cost_cny,
        failure_penalty_score,
        learning_reward,
        transitions,
        completed_episodes,
        failed_episodes,
        tuple(greedy_actions),
    )


def _train(
    args: argparse.Namespace,
    train: tuple[FormalEpisode, ...],
    validation: tuple[FormalEpisode, ...],
    *,
    profile: DqnExperimentProfile,
    calibration: RewardScaleCalibration | None,
    timescale: TimeScaleConfig | None = None,
) -> None:
    if type(profile) is not DqnExperimentProfile:
        raise TypeError("profile must be an exact DqnExperimentProfile")
    explicit_timescale = timescale is not None
    scale = profile.timescale if timescale is None else timescale
    if type(scale) is not TimeScaleConfig:
        raise TypeError("timescale must be an exact TimeScaleConfig or None")
    if scale != profile.timescale:
        raise ValueError("timescale differs from the experiment profile")
    config = profile.dqn_config(rounds=args.rounds)
    agent = DqnAgent(config, seed=args.seed, device=args.device)
    ordered_ids = tuple(episode.sample_id for episode in train)
    by_id = {episode.sample_id: episode for episode in train}
    schedule = EpisodeShuffleSchedule(ordered_ids, seed=args.seed)
    global_step = 0
    round_index = 0
    episode_position = 0
    permutation = schedule.next_round()

    if args.resume is not None:
        if explicit_timescale:
            metadata = load_checkpoint(
                args.resume,
                agent=agent,
                schedule=schedule,
                timescale=scale,
            )
        else:
            metadata = load_checkpoint(
                args.resume,
                agent=agent,
                schedule=schedule,
            )
        global_step = metadata.global_macro_step
        round_index = metadata.round_index
        episode_position = metadata.episode_position
        permutation = metadata.current_permutation

    args.output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = args.output_dir / "latest.pt"
    started = time.perf_counter()
    while round_index < config.rounds:
        round_raw_economic_cost_cny = 0.0
        round_failure_penalty_score = 0.0
        round_learning_reward = 0.0
        round_completed_episodes = 0
        round_failed_episodes = 0
        round_behavior_actions: list[int] = []
        diagnostic_window: list[DqnOptimizationDiagnostics] = []
        for position in range(episode_position, len(permutation)):
            episode = by_id[permutation[position]]
            if explicit_timescale:
                backend, environment = _environment(episode, timescale=scale)
            else:
                backend, environment = _environment(episode)
            state = environment.reset()
            done = False
            episode_macro = 0
            episode_raw_economic_cost_cny = 0.0
            episode_failure_penalty_score = 0.0
            episode_learning_reward = 0.0
            while not done:
                epsilon = epsilon_at_global_step(global_step, config=config)
                action_index = agent.select_action(
                    np.asarray(state, dtype=np.float32), epsilon=epsilon
                )
                before_modes = dict(backend.mode_counts)
                transition = environment.step(
                    FINAL_DQN_ACTION_CATALOG[action_index].action_id
                )
                replay_reward = profile.replay_reward(transition, calibration)
                agent.replay.append(
                    np.asarray(transition.state, dtype=np.float32),
                    action_index,
                    replay_reward,
                    np.asarray(transition.next_state, dtype=np.float32),
                    transition.done,
                )
                global_step += 1
                episode_macro += 1
                round_behavior_actions.append(action_index)
                episode_raw_economic_cost_cny += transition.raw_economic_cost_cny
                episode_failure_penalty_score += transition.failure_penalty_score
                episode_learning_reward += transition.learning_reward
                if (
                    global_step >= config.warmup_steps
                    and len(agent.replay) >= config.batch_size
                ):
                    diagnostic_window.append(agent.optimize())
                mode_delta = {
                    mode: backend.mode_counts[mode] - before_modes[mode]
                    for mode in OperatingMode
                }
                if global_step % args.log_every == 0 or transition.done:
                    _progress_line(
                        round_index=round_index,
                        rounds=config.rounds,
                        episode_position=position,
                        episode_count=len(permutation),
                        episode=episode,
                        episode_macro=episode_macro,
                        global_step=global_step,
                        action_index=action_index,
                        epsilon=epsilon,
                        transition=transition,
                        episode_raw_economic_cost_cny=episode_raw_economic_cost_cny,
                        episode_failure_penalty_score=episode_failure_penalty_score,
                        episode_learning_reward=episode_learning_reward,
                        replay_reward=replay_reward,
                        optimization_diagnostics=tuple(diagnostic_window),
                        replay_size=len(agent.replay),
                        backend=backend,
                        mode_delta=mode_delta,
                        elapsed_s=time.perf_counter() - started,
                    )
                    diagnostic_window.clear()
                state = transition.next_state
                done = transition.done

            round_raw_economic_cost_cny += episode_raw_economic_cost_cny
            round_failure_penalty_score += episode_failure_penalty_score
            round_learning_reward += episode_learning_reward
            if transition.failed:
                round_failed_episodes += 1
            elif transition.episode_completed:
                round_completed_episodes += 1
            else:  # pragma: no cover - environment contract guard
                raise RuntimeError("training episode ended without terminal outcome")

            save_checkpoint(
                latest_path,
                agent=agent,
                schedule=schedule,
                global_macro_step=global_step,
                round_index=round_index,
                episode_position=position + 1,
                current_permutation=permutation,
                timescale=scale,
            )
            print(
                f"checkpoint={latest_path} round={round_index + 1} "
                f"episode={position + 1}/{len(permutation)} global={global_step}",
                flush=True,
            )

        if explicit_timescale:
            validation_summary = _evaluate_validation(
                agent,
                validation,
                timescale=scale,
            )
        else:
            validation_summary = _evaluate_validation(agent, validation)
        greedy_diagnostics = (
            summarize_actions(
                validation_summary.greedy_action_indices,
                action_dim=len(FINAL_DQN_ACTION_CATALOG),
            )
            if validation_summary.greedy_action_indices
            else None
        )
        print(
            f"validation round={round_index + 1}/{config.rounds} "
            f"episodes={len(validation)} transitions={validation_summary.transitions} "
            f"raw_economic_cost_cny={validation_summary.raw_economic_cost_cny:.9f} "
            f"failure_penalty_score={validation_summary.failure_penalty_score:.9f} "
            f"learning_reward={validation_summary.learning_reward:.9f} "
            f"completed_episodes={validation_summary.completed_episodes} "
            f"failed_episodes={validation_summary.failed_episodes} "
            f"completion_rate={validation_summary.completion_rate:.6f} "
            f"greedy_unique_actions={greedy_diagnostics.unique_action_count if greedy_diagnostics else 'NA'} "
            f"greedy_max_share={f'{greedy_diagnostics.max_action_share:.9g}' if greedy_diagnostics else 'NA'} "
            f"greedy_entropy={f'{greedy_diagnostics.shannon_entropy:.9g}' if greedy_diagnostics else 'NA'} "
            f"shuffle=NO replay_updates=0 "
            f"optimizer_updates=0 test_payloads_opened=0",
            flush=True,
        )
        completed_round = round_index + 1
        permutation = schedule.next_round()
        round_index = completed_round
        episode_position = 0
        save_checkpoint(
            latest_path,
            agent=agent,
            schedule=schedule,
            global_macro_step=global_step,
            round_index=round_index,
            episode_position=episode_position,
            current_permutation=permutation,
            timescale=scale,
        )
        periodic_path = args.output_dir / f"round_{completed_round:03d}.pt"
        save_checkpoint(
            periodic_path,
            agent=agent,
            schedule=schedule,
            global_macro_step=global_step,
            round_index=round_index,
            episode_position=episode_position,
            current_permutation=permutation,
            timescale=scale,
        )
        behavior_diagnostics = summarize_actions(
            tuple(round_behavior_actions),
            action_dim=len(FINAL_DQN_ACTION_CATALOG),
        )
        print(
            f"round_complete={completed_round}/{config.rounds} global={global_step} "
            f"raw_economic_cost_cny={round_raw_economic_cost_cny:.9f} "
            f"failure_penalty_score={round_failure_penalty_score:.9f} "
            f"learning_reward={round_learning_reward:.9f} "
            f"completed_episodes={round_completed_episodes} "
            f"failed_episodes={round_failed_episodes} "
            f"completion_rate={round_completed_episodes / len(permutation):.6f} "
            f"behavior_unique_actions={behavior_diagnostics.unique_action_count} "
            f"behavior_max_share={behavior_diagnostics.max_action_share:.9g} "
            f"behavior_entropy={behavior_diagnostics.shannon_entropy:.9g} "
            f"checkpoint={latest_path} periodic_checkpoint={periodic_path} "
            f"elapsed_s={time.perf_counter() - started:.1f}",
            flush=True,
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.resume is not None and (args.preflight_only or args.smoke_only):
        _parser().error("--resume is available only in normal training mode")
    report, dataset, train, validation = _live_preflight(args)
    if args.preflight_only:
        return 0 if report.ready else 2
    if args.smoke_only:
        _smoke(train)
        return 0
    require_formal_training_ready()
    _train(
        args,
        train,
        validation,
        profile=history_study_profile("H1", None),
        calibration=None,
    )
    if dataset.opened_test_payloads != 0:
        raise RuntimeError("Test payload was opened by formal training")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
