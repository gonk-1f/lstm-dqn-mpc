"""Independent formal v2 DQN training entrypoint.

This module deliberately does not import or delegate to the legacy trainer.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import time
from typing import Sequence

import numpy as np

from ..config import TAU_LPF_SECONDS, TimeScaleConfig
from ..control.causal_base_load import CausalBaseLoadFilter
from ..data.formal_training_dataset import FormalEpisode, FormalTrainingDataset
from ..data.supervisory_rules import OperatingMode
from ..dqn.action_space import FINAL_DQN_ACTION_CATALOG
from ..dqn.state import OperatingHistorySample, build_formal_operating_state
from ..envs.formal_episode import (
    FormalEpisodeBackend,
    build_formal_nonlinear_mpc,
)
from ..envs.multirate_weight_env import MultiRateWeightEnvironment
from ..preflight import assess_formal_training_preflight, require_formal_training_ready
from ..training.checkpoint import load_checkpoint, save_checkpoint
from ..training.dqn import DqnAgent, DqnTrainingConfig, epsilon_at_global_step
from ..training.schedule import EpisodeShuffleSchedule


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_POWER_ROOT = REPOSITORY_ROOT / "data" / "processed" / "operating_dataset_zero_boundary_v2"
DEFAULT_AIS_ROOT = REPOSITORY_ROOT / "data" / "processed" / "operating_dataset_zero_boundary_v2_ais"
DEFAULT_MODE_ROOT = REPOSITORY_ROOT / "data" / "processed" / "operating_dataset_zero_boundary_v2_modes"
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "outputs" / "v2_formal_dqn"


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
    parser.add_argument("--rounds", type=_positive_int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log-every", type=_positive_int, default=1)
    parser.add_argument("--resume", type=Path)
    return parser


def _environment(episode: FormalEpisode) -> tuple[FormalEpisodeBackend, MultiRateWeightEnvironment]:
    backend = FormalEpisodeBackend(
        load_kw=episode.load_kw,
        speed_kn=episode.speed_kn,
        fc_power_kw=episode.fc_power_kw,
        battery_bus_kw=episode.battery_bus_kw,
        operating_mode=episode.operating_mode,
        mpc=build_formal_nonlinear_mpc(),
    )
    environment = MultiRateWeightEnvironment(
        timescale=TimeScaleConfig.formal_baseline(),
        action_catalog=FINAL_DQN_ACTION_CATALOG,
        backend=backend,
        state_provider=backend.state,
        formal_training_mode=True,
    )
    return backend, environment


def _validate_all_states(episodes: tuple[FormalEpisode, ...]) -> int:
    checked = 0
    for episode in episodes:
        history: list[OperatingHistorySample] = []
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
                continue
            sample = OperatingHistorySample(
                float(time_s), 0.60, float(fc_kw), 0.0, float(load_kw), float(load_kw)
            )
            history.append(sample)
            # Only the frozen 150 s causal window can affect S8.
            history = history[-6:]
            state = build_formal_operating_state(
                tuple(history),
                current_time_seconds=float(time_s),
                speed_kn=float(speed_kn),
            )
            if len(state) != 8 or not np.isfinite(np.asarray(state)).all():
                raise ValueError(f"{episode.sample_id}: nonfinite formal state")
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


def _live_preflight(args: argparse.Namespace) -> tuple[object, FormalTrainingDataset, tuple[FormalEpisode, ...], tuple[FormalEpisode, ...]]:
    report = assess_formal_training_preflight()
    dataset = FormalTrainingDataset.open(args.power_root, args.ais_root, args.mode_root)
    train = dataset.load_train()
    validation = dataset.load_validation()
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
        f"train_macro_transitions={dataset.train_macro_transitions} "
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


def _smoke(train: tuple[FormalEpisode, ...]) -> None:
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
    backend, environment = _environment(bounded)
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
        f"mpc_solves={backend.mpc_solve_count} reward_cny={transition.reward_cny:.9f} "
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
    episode_reward: float,
    loss: float | None,
    replay_size: int,
    backend: FormalEpisodeBackend,
    mode_delta: dict[OperatingMode, int],
    elapsed_s: float,
) -> None:
    action = FINAL_DQN_ACTION_CATALOG[action_index]
    last_index = max(0, backend.index - 1)
    load = float(backend.load_kw[last_index])
    state = transition.next_state
    terminal = "episode_end" if transition.done else "continue"
    loss_text = "NA" if loss is None else f"{loss:.9g}"
    print(
        f"round={round_index + 1}/{rounds} episode={episode_position + 1}/{episode_count} "
        f"segment={episode.sample_id} macro={episode_macro} global={global_step} "
        f"action_index={action_index} action_id={action.action_id} weights={action.as_tuple()} "
        f"epsilon={epsilon:.6f} greedy_rate={1.0 - epsilon:.6f} "
        f"reward_cny={transition.reward_cny:.9f} episode_reward_cny={episode_reward:.9f} "
        f"loss={loss_text} replay={replay_size} soc={state[0]:.9f} load_kw={load:.6f} "
        f"base_kw={state[1] * 600.0:.6f} fc_kw={backend.previous_fc_kw:.6f} "
        f"delta_fc_kw={state[6] * 600.0:.6f} executed_steps={transition.executed_mpc_steps} "
        f"onboard_steps={mode_delta[OperatingMode.ONBOARD]} "
        f"shore_pending_steps={mode_delta[OperatingMode.SHORE_PENDING]} "
        f"shore_steps={mode_delta[OperatingMode.SHORE_CHARGING]} "
        f"unresolved_steps={mode_delta[OperatingMode.UNRESOLVED]} terminal={terminal} "
        f"elapsed_s={elapsed_s:.1f}",
        flush=True,
    )


def _evaluate_validation(agent: DqnAgent, validation: tuple[FormalEpisode, ...]) -> tuple[float, int]:
    total_reward = 0.0
    transitions = 0
    for episode in validation:  # Manifest order is fixed; never shuffled.
        _, environment = _environment(episode)
        state = environment.reset()
        done = False
        while not done:
            action_index = agent.greedy_action(np.asarray(state, dtype=np.float32))
            transition = environment.step(FINAL_DQN_ACTION_CATALOG[action_index].action_id)
            total_reward += transition.reward_cny
            transitions += 1
            state = transition.next_state
            done = transition.done
    return total_reward, transitions


def _train(
    args: argparse.Namespace,
    train: tuple[FormalEpisode, ...],
    validation: tuple[FormalEpisode, ...],
) -> None:
    config = replace(DqnTrainingConfig.formal_baseline(), rounds=args.rounds)
    agent = DqnAgent(config, seed=args.seed, device=args.device)
    ordered_ids = tuple(episode.sample_id for episode in train)
    by_id = {episode.sample_id: episode for episode in train}
    schedule = EpisodeShuffleSchedule(ordered_ids, seed=args.seed)
    global_step = 0
    round_index = 0
    episode_position = 0
    permutation = schedule.next_round()

    if args.resume is not None:
        metadata = load_checkpoint(args.resume, agent=agent, schedule=schedule)
        global_step = metadata.global_macro_step
        round_index = metadata.round_index
        episode_position = metadata.episode_position
        permutation = metadata.current_permutation

    args.output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = args.output_dir / "latest.pt"
    started = time.perf_counter()
    while round_index < config.rounds:
        for position in range(episode_position, len(permutation)):
            episode = by_id[permutation[position]]
            backend, environment = _environment(episode)
            state = environment.reset()
            done = False
            episode_macro = 0
            episode_reward = 0.0
            while not done:
                epsilon = epsilon_at_global_step(global_step)
                action_index = agent.select_action(
                    np.asarray(state, dtype=np.float32), epsilon=epsilon
                )
                before_modes = dict(backend.mode_counts)
                transition = environment.step(
                    FINAL_DQN_ACTION_CATALOG[action_index].action_id
                )
                agent.replay.append(
                    np.asarray(transition.state, dtype=np.float32),
                    action_index,
                    transition.reward_cny,
                    np.asarray(transition.next_state, dtype=np.float32),
                    transition.done,
                )
                global_step += 1
                episode_macro += 1
                episode_reward += transition.reward_cny
                loss: float | None = None
                if (
                    global_step >= config.warmup_steps
                    and len(agent.replay) >= config.batch_size
                ):
                    loss = agent.optimize()
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
                        episode_reward=episode_reward,
                        loss=loss,
                        replay_size=len(agent.replay),
                        backend=backend,
                        mode_delta=mode_delta,
                        elapsed_s=time.perf_counter() - started,
                    )
                state = transition.next_state
                done = transition.done

            save_checkpoint(
                latest_path,
                agent=agent,
                schedule=schedule,
                global_macro_step=global_step,
                round_index=round_index,
                episode_position=position + 1,
                current_permutation=permutation,
            )
            print(
                f"checkpoint={latest_path} round={round_index + 1} "
                f"episode={position + 1}/{len(permutation)} global={global_step}",
                flush=True,
            )

        validation_reward, validation_transitions = _evaluate_validation(agent, validation)
        print(
            f"validation round={round_index + 1}/{config.rounds} "
            f"episodes={len(validation)} transitions={validation_transitions} "
            f"reward_cny={validation_reward:.9f} shuffle=NO replay_updates=0 "
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
        )
        print(
            f"round_complete={completed_round}/{config.rounds} global={global_step} "
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
    _train(args, train, validation)
    if dataset.opened_test_payloads != 0:
        raise RuntimeError("Test payload was opened by formal training")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
