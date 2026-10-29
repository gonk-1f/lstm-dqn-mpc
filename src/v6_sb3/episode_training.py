"""Single-thread, complete-episode SB3 DQN with exact safe-point recovery.

Only the explicit CLI starts work. Formal training requires acknowledgement
of the unchanged, uncalibrated legacy failure penalty and early-failure risk.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, fields
import csv
import hashlib
import json
from math import fsum
from pathlib import Path
import pickle
import random
import shutil
import signal
from time import perf_counter

from filelock import FileLock
import numpy as np
from stable_baselines3.common.logger import configure
import torch

from .episode_metrics import EpisodeMetrics
from .episode_schedule import EpisodeSchedule, epsilon_at_episode
from .environment import InfeasibleReset
from .identity import execution_identity
from .masked_dqn import MaskedDQN
from .search import validation_objective


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class EpisodeTrainingConfig:
    episodes: int = 500
    learning_starts: int = 50_000
    train_freq: int = 32
    gradient_steps: int = 1
    target_update_interval: int = 10_000
    learning_rate: float = 1e-4
    batch_size: int = 64
    buffer_size: int = 1_000_000
    network_width: int = 256
    seed: int = 42
    order_seed: int = 42
    warmup_seed: int = 43
    validation_every: int = 50
    reward_mode: str = 'zhou_economic_only'
    acknowledge_failure_penalty: bool = False
    checkpoint_keep: int = 2
    epsilon_initial: float = 1.
    epsilon_final: float = .05
    epsilon_decay_fraction: float = .8
    max_env_steps: int | None = None
    smoke: bool = False

    def __post_init__(self):
        if self.train_freq != 32 or self.gradient_steps != 1:
            raise ValueError('the episode runner requires train_freq=32 and gradient_steps=1')
        if min(self.episodes, self.batch_size, self.network_width, self.target_update_interval,
               self.validation_every, self.checkpoint_keep) < 1 or self.learning_starts < 0:
            raise ValueError('positive training budgets and nonnegative learning_starts are required')
        if self.buffer_size < max(self.batch_size, 32):
            raise ValueError('buffer_size must accommodate the minibatch and collection block')
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError('learning_rate must be positive and finite')
        if self.max_env_steps is not None and self.max_env_steps < 1:
            raise ValueError('max_env_steps must be positive')
        if self.reward_mode not in ('zhou_economic_only', 'v6_energy_soc_ramp'):
            raise ValueError('unsupported audited reward mode')
        epsilon_at_episode(1, self.episodes, initial=self.epsilon_initial, final=self.epsilon_final,
                           decay_fraction=self.epsilon_decay_fraction)


def _configuration(value):
    if isinstance(value, EpisodeTrainingConfig):
        return value
    options = dict(value)
    policy = options.pop('policy_kwargs', None)
    architecture = options.pop('net_arch', None)
    if policy is not None:
        if set(policy) != {'net_arch'}:
            raise ValueError('only policy_kwargs.net_arch may be configured')
        architecture = policy['net_arch']
    if architecture is not None:
        if len(architecture) != 2 or architecture[0] != architecture[1]:
            raise ValueError('network must contain two equal hidden layers')
        options['network_width'] = architecture[0]
    return EpisodeTrainingConfig(**options)


def _make_env(config):
    from .direct_environment import DirectVesselEnv
    return DirectVesselEnv(reward_mode=config.reward_mode)


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def _append_row(path, row):
    with Path(path).open('a', encoding='utf-8', newline='') as output:
        output.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')


def _csv_append(path, row):
    exists = path.exists() and path.stat().st_size > 0
    with path.open('a', encoding='utf-8', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _repair_logs(directory, state):
    """Discard post-checkpoint rows before replaying an interrupted suffix."""
    for filename, count in (('episodes', state['completed_episodes']),
                            ('validation', len(state['validation_history']))):
        path = directory/f'{filename}.jsonl'
        lines = path.read_text(encoding='utf-8').splitlines() if path.exists() else []
        if len(lines) < count:
            raise ValueError(f'{filename} log has fewer rows than the checkpoint')
        with path.open('w', encoding='utf-8', newline='') as output:
            output.write(''.join(line+'\n' for line in lines[:count]))
        if filename == 'episodes':
            csv_path = directory/'episodes.csv'
            csv_path.write_text('', encoding='utf-8')
            for line in lines[:count]:
                _csv_append(csv_path, json.loads(line))


def _episode_identity(train, validation):
    return {split:[dict(sample_id=e.sample_id,
        sha256=hashlib.sha256(pickle.dumps(e, protocol=5)).hexdigest()) for e in episodes]
        for split, episodes in (('train', train), ('validation', validation))}


def _rng_snapshot():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state())


def _restore_rng(value):
    random.setstate(value['python'])
    np.random.set_state(value['numpy'])
    torch.set_rng_state(value['torch'])


def _remove_checkpoint_tree(path, checkpoints):
    """Check resolved deletion targets, including Windows junctions/symlinks."""
    resolved = path.resolve()
    if resolved.parent != checkpoints.resolve():
        raise ValueError('checkpoint cleanup target escapes the checkpoint directory')
    shutil.rmtree(resolved)


def _save_checkpoint(directory, model, state, schedule, warmup_rng, env, active, metrics, config, identity):
    checkpoints = directory/'checkpoints'
    checkpoints.mkdir(exist_ok=True)
    if checkpoints.resolve().parent != directory.resolve():
        raise ValueError('checkpoint directory escapes the run directory')
    # A killed process may leave a partial generation, or a complete generation
    # whose latest pointer was not yet committed. Never reuse either name.
    generations = [state.get('checkpoint_sequence', 0)]
    for path in checkpoints.iterdir():
        if path.name.startswith(('checkpoint_', '.writing_')):
            suffix = path.name.rsplit('_', 1)[-1]
            if suffix.isdigit():
                generations.append(int(suffix))
    sequence = max(generations)+1
    state['checkpoint_sequence'] = sequence
    destination = checkpoints/f'checkpoint_{sequence:06d}'
    temporary = checkpoints/f'.writing_{sequence:06d}'
    temporary.mkdir(exist_ok=False)
    try:
        model.save(temporary/'model.zip')
        model.save_replay_buffer(temporary/'replay.pkl')
        saved = dict(state=state, config=asdict(config), identity=identity,
            schedule=schedule.snapshot(), warmup_rng=warmup_rng.getstate(), rng=_rng_snapshot(),
            active=active, metrics=metrics.snapshot(), environment=env.snapshot() if active else None)
        with (temporary/'state.pkl').open('wb') as output:
            pickle.dump(saved, output, protocol=5)
        _write_json(temporary/'summary.json', dict(status=state['status'],
            completed_episodes=state['completed_episodes'], formal_env_steps=model.formal_env_steps,
            warmup_env_steps=state['warmup_env_steps'], adam_updates=model._n_updates,
            total_env_steps=model.num_timesteps))
        temporary.replace(destination)
        _write_json(directory/'latest.json', dict(checkpoint=destination.relative_to(directory).as_posix()))
    except BaseException:
        # A previous latest pointer remains valid if this atomic generation fails.
        if temporary.exists():
            _remove_checkpoint_tree(temporary, checkpoints)
        raise
    # Hold the run lock, and commit a usable latest pointer before removing
    # abandoned large Replay writes or pruning old generations.
    for abandoned in checkpoints.glob('.writing_*'):
        _remove_checkpoint_tree(abandoned, checkpoints)
    previous = sorted(checkpoints.glob('checkpoint_*'))
    for obsolete in previous[:-config.checkpoint_keep]:
        _remove_checkpoint_tree(obsolete, checkpoints)
    return destination


def _validation(config, validation, model):
    """Same physical environment; no Replay, optimizer or training RNG changes."""
    rng = _rng_snapshot()
    was_training = model.policy.training
    before = (model.num_timesteps, model.formal_env_steps, model._n_updates, model.replay_buffer.insertions)
    rows = []
    env = _make_env(config)
    try:
        model.policy.set_training_mode(False)
        with torch.inference_mode():
            for episode in validation:
                metrics = EpisodeMetrics()
                try:
                    observation, _ = env.reset(options=dict(episode=episode))
                except InfeasibleReset as failure:
                    rows.append(metrics.row(episode, failure.status))
                    continue
                while True:
                    action, _ = model.masked_predict(observation, env.action_masks(), deterministic=True)
                    observation, reward, terminated, truncated, _ = env.step(int(action))
                    if truncated:
                        raise RuntimeError('Validation must not artificially truncate a voyage')
                    metrics.observe_transition(env, reward)
                    if terminated:
                        rows.append(metrics.row(episode, env.episode_status))
                        break
    finally:
        env.close()
        model.policy.set_training_mode(was_training)
        _restore_rng(rng)
    after = (model.num_timesteps, model.formal_env_steps, model._n_updates, model.replay_buffer.insertions)
    if before != after:
        raise RuntimeError('Validation changed training counters or Replay')
    complete = sum(row['completed'] for row in rows)
    summary = dict(episodes=len(rows), completed=complete,
        mean_duration_coverage=fsum(row['duration_coverage'] for row in rows)/len(rows),
        fixed_complete_set_cost_cny=(fsum(row['complete_economic_cost_cny'] for row in rows)
                                     if complete == len(rows) else None),
        executed_prefix_cost_cny=fsum(row['executed_prefix_economic_cost_cny'] for row in rows),
        environment_steps=sum(row['episode_env_steps'] for row in rows),
        failure_reasons=dict(Counter(row['terminal_reason'] for row in rows if not row['completed'])),
        evaluation_epsilon=0., evaluation_adam_updates=0, evaluation_replay_insertions=0,
        test_payloads_opened=0, samples=rows)
    summary['score'] = validation_objective(summary)
    return summary


def run_episode_training(config, train, validation, directory, *, resume=False,
                         stop_requested=None, on_validation=None, stop_after_steps=None):
    """Shared baseline/Optuna runner; stop_after_steps pauses an exact live prefix.

    ``max_env_steps`` is a soft formal budget checked only at natural voyage
    boundaries. ``stop_after_steps`` counts real transitions in this invocation,
    including warmup, and saves resumable state without creating a terminal.
    Validation callbacks receive ``(summary, progress)`` after checkpointing;
    Optuna's TrialPruned may propagate without losing the completed evaluation.
    """
    config = _configuration(config)
    if not config.smoke and not config.acknowledge_failure_penalty:
        raise ValueError('formal training requires acknowledge_failure_penalty=True: the unchanged '
            'legacy 10132.66 CNY penalty is uncalibrated for current Train and may reward early failure')
    train, validation = tuple(train), tuple(validation)
    if not train or not validation:
        raise ValueError('nonempty Train and Validation sets are required')
    for split, episodes in (('train', train), ('validation', validation)):
        if any(e.split != split for e in episodes):
            raise ValueError(f'{split} split isolation violation')
    if len({e.sample_id for e in (*train, *validation)}) != len(train)+len(validation):
        raise ValueError('Train and Validation sample IDs must be distinct')
    if stop_after_steps is not None and stop_after_steps < 0:
        raise ValueError('stop_after_steps must be nonnegative')
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(directory/'.run.lock'), timeout=0):
        return _run_locked(config, train, validation, directory, resume, stop_requested,
                           on_validation, stop_after_steps)


def _run_locked(config, train, validation, directory, resume, stop_requested, on_validation, stop_after_steps):
    identity = execution_identity(ROOT, _episode_identity(train, validation))
    torch.set_num_threads(1)
    env = _make_env(config)
    by_id = {episode.sample_id: episode for episode in train}
    latest = None
    if resume:
        pointer = json.loads((directory/'latest.json').read_text(encoding='utf-8'))
        latest = directory/pointer['checkpoint']
        with (latest/'state.pkl').open('rb') as source:
            saved = pickle.load(source)
        if saved['config'] != asdict(config) or saved['identity'] != identity:
            env.close()
            raise ValueError('resume config, data, code or dependency identity differs')
        model = MaskedDQN.load(latest/'model.zip', env=env, device='cpu')
        model.load_replay_buffer(latest/'replay.pkl', truncate_last_traj=False)
        state = saved['state']
        schedule = EpisodeSchedule.restore(saved['schedule'])
        warmup_rng = random.Random()
        warmup_rng.setstate(saved['warmup_rng'])
        active = saved['active']
        metrics = EpisodeMetrics(**saved['metrics'])
        observation = env.restore(saved['environment'], by_id[active['sample_id']])[0] if active else None
        _repair_logs(directory, state)
        _restore_rng(saved['rng'])
    else:
        if any(path.name != '.run.lock' for path in directory.iterdir()):
            env.close()
            raise ValueError('fresh training needs a new empty run directory; use resume for a checkpoint')
        _write_json(directory/'configuration.json', asdict(config))
        _write_json(directory/'identity.json', identity)
        schedule = EpisodeSchedule(tuple(by_id), config.episodes, seed=config.order_seed)
        warmup_rng = random.Random(config.warmup_seed)
        model = MaskedDQN(env=env, td_diagnostics=False, target_sync_unit='formal_env_steps',
            learning_rate=config.learning_rate, learning_starts=config.learning_starts,
            buffer_size=config.buffer_size, batch_size=config.batch_size, train_freq=32,
            gradient_steps=1, target_update_interval=config.target_update_interval, gamma=1.,
            policy_kwargs=dict(net_arch=[config.network_width]*2), seed=config.seed, device='cpu')
        state = dict(status='running', completed_episodes=0, warmup_env_steps=0,
            validation_history=[], best_score=None, best_checkpoint=None, checkpoint_sequence=0,
            elapsed_seconds=0., validation_seconds=0., checkpoint_seconds=0.)
        active, observation, metrics = None, None, EpisodeMetrics()
    model.set_logger(configure(folder=None, format_strings=[]))
    # The observers retain scalars for the current episode, never minibatch tensors.
    last_td_mae = [None]
    def observe_td(agent, batch, online, target):
        last_td_mae[0] = float((target-online).abs().mean().item())
    def observe_update(agent):
        metrics.observe_update(agent.last_training_loss, last_td_mae[0])
    model.train_step_observer = observe_td
    model.after_update_observer = observe_update
    invocation_steps = 0
    checkpoint_signature = None
    started = perf_counter()
    previous_elapsed = state['elapsed_seconds']

    def progress():
        return dict(status=state['status'], completed_episodes=state['completed_episodes'],
            planned_episodes=config.episodes, formal_env_steps=model.formal_env_steps,
            warmup_env_steps=state['warmup_env_steps'], total_env_steps=model.num_timesteps,
            adam_updates=model._n_updates, pending_collection_steps=model.formal_env_steps%32,
            target_sync_history=list(model.target_sync_history), replay_insertions=model.replay_buffer.insertions,
            replay_entries=model.replay_buffer.size(), sample_order=list(schedule.order),
            sample_counts=dict(schedule.counts), validation_history=state['validation_history'],
            best_checkpoint=state['best_checkpoint'], best_score=state['best_score'],
            latest_checkpoint=str(latest) if latest else None,
            active_sample_id=active['sample_id'] if active else None,
            test_payloads_opened=0, elapsed_seconds=previous_elapsed+perf_counter()-started,
            validation_seconds=state['validation_seconds'], checkpoint_seconds=state['checkpoint_seconds'])

    def checkpoint():
        nonlocal latest, checkpoint_signature
        signature = (model.num_timesteps, state['completed_episodes'], state['status'],
                     len(state['validation_history']))
        if signature == checkpoint_signature:
            return
        state['elapsed_seconds'] = previous_elapsed+perf_counter()-started
        start = perf_counter()
        latest = _save_checkpoint(directory, model, state, schedule, warmup_rng, env,
                                  active, metrics, config, identity)
        state['checkpoint_seconds'] += perf_counter()-start
        checkpoint_signature = signature
        _write_json(directory/'progress.json', progress())

    def validate():
        start = perf_counter()
        summary = _validation(config, validation, model)
        state['validation_seconds'] += perf_counter()-start
        summary['episode'] = state['completed_episodes']
        is_best = state['best_score'] is None or summary['score'] > state['best_score']
        if is_best:
            temporary = directory/'best_model.tmp.zip'
            model.save(temporary)
            temporary.replace(directory/'best_model.zip')
            state['best_score'] = summary['score']
            state['best_checkpoint'] = str(directory/'best_model.zip')
            _write_json(directory/'best_validation.json', summary)
        summary['best_checkpoint'] = is_best
        state['validation_history'].append(summary)
        _append_row(directory/'validation.jsonl', summary)
        print(f'Validation | Episode {summary["episode"]} | {summary["completed"]}/{summary["episodes"]}'
              f' | Coverage {summary["mean_duration_coverage"]:.6f}'
              f' | EconomicCost {summary["fixed_complete_set_cost_cny"]} | BestCheckpoint {is_best}', flush=True)
        checkpoint()
        if on_validation is not None:
            on_validation(summary, progress())

    def finish_episode(status):
        nonlocal active, metrics, observation
        row = metrics.row(by_id[active['sample_id']], status)
        state['completed_episodes'] += 1
        if state['completed_episodes'] == config.episodes:
            state['status'] = 'complete'
        row = dict(episode=state['completed_episodes'], env_steps=model.formal_env_steps,
            adam_updates=model._n_updates, epsilon=active['epsilon'], **row)
        _append_row(directory/'episodes.jsonl', row)
        _csv_append(directory/'episodes.csv', row)
        print(f'Episode {row["episode"]} | EnvSteps {row["env_steps"]} | AdamUpdates {row["adam_updates"]}'
              f' | Epsilon {row["epsilon"]:.6f} | LossMean {row["loss_mean"]} | TDMAE {row["td_mae"]}'
              f' | Return {row["return_sum"]:.6f} | Completed {row["completed"]}'
              f' | SOC_Final {row["soc_final"]:.6f}', flush=True)
        active, observation, metrics = None, None, EpisodeMetrics()
        _write_json(directory/'sample_schedule.json', dict(order=schedule.order, counts=schedule.counts))
        if (state['completed_episodes']%config.validation_every == 0 or
                state['completed_episodes'] == config.episodes):
            validate()

    try:
        state['status'] = 'running'
        while state['completed_episodes'] < config.episodes:
            if ((stop_requested is not None and stop_requested()) or
                    (stop_after_steps is not None and invocation_steps >= stop_after_steps)):
                state['status'] = 'paused'
                break
            warming = state['warmup_env_steps'] < config.learning_starts
            if active is None:
                if not warming and config.max_env_steps and model.formal_env_steps >= config.max_env_steps:
                    state['status'] = 'budget_complete'
                    break
                sample_id = warmup_rng.choice(tuple(by_id)) if warming else schedule.draw()
                epsilon = (1. if warming else epsilon_at_episode(state['completed_episodes']+1,
                    config.episodes, initial=config.epsilon_initial, final=config.epsilon_final,
                    decay_fraction=config.epsilon_decay_fraction))
                active = dict(sample_id=sample_id, warmup=warming, epsilon=epsilon)
                metrics = EpisodeMetrics()
                try:
                    observation, _ = env.reset(options=dict(episode=by_id[sample_id]))
                except InfeasibleReset as failure:
                    if warming:
                        state['infeasible_warmup_resets'] = state.get('infeasible_warmup_resets', 0)+1
                        active = None
                        if state['infeasible_warmup_resets'] > max(100, len(train)*10):
                            raise RuntimeError('warmup cannot collect transitions: repeated infeasible resets')
                    else:
                        finish_episode(failure.status)
                    continue
            mask = env.action_masks()
            if warming:
                action = warmup_rng.choice(np.flatnonzero(mask).tolist())
            else:
                model.exploration_rate = active['epsilon']
                action = int(model.masked_predict(observation, mask, epsilon=active['epsilon'])[0])
            successor, reward, terminated, truncated, info = env.step(action)
            if truncated:
                raise RuntimeError('formal/warmup voyages must not receive artificial truncation')
            metrics.observe_transition(env, reward)
            model.replay_buffer.add(np.asarray(observation)[None], np.asarray(successor)[None],
                np.asarray([action]), np.asarray([reward]), np.asarray([terminated]), [info])
            observation = successor
            model.num_timesteps += 1
            invocation_steps += 1
            if warming:
                state['warmup_env_steps'] += 1
                # Warmup owns its stream; its unfinished final voyage is never
                # counted as or continued into a formal episode.
                if terminated or state['warmup_env_steps'] == config.learning_starts:
                    active, observation, metrics = None, None, EpisodeMetrics()
            else:
                # Copy immediately on the true environment-step clock, before
                # an Adam update if both events share the same safe point.
                model.record_formal_transition()
                if model.formal_env_steps%32 == 0:
                    model.train(gradient_steps=1, batch_size=config.batch_size)
                if terminated:
                    finish_episode(env.episode_status)
            if model.num_timesteps%5000 == 0:
                _write_json(directory/'progress.json', progress())
        else:
            state['status'] = 'complete'
        if state['status'] == 'budget_complete' and state['completed_episodes'] and (
                not state['validation_history'] or
                state['validation_history'][-1]['episode'] != state['completed_episodes']):
            validate()
        checkpoint()
        result = progress()
        _write_json(directory/'result.json', result)
        return result
    except BaseException as error:
        # Callback pruning has already checkpointed at a stable Validation
        # boundary. Other errors are visible immediately and never called success.
        _write_json(directory/'error.json', dict(type=type(error).__name__, message=str(error),
            formal_env_steps=model.formal_env_steps, adam_updates=model._n_updates))
        raise
    finally:
        model.train_step_observer = None
        model.after_update_observer = None
        env.close()


def _smoke_episodes():
    """Tiny synthetic independent voyages, with actual direct physical steps."""
    from types import SimpleNamespace
    def make(sample_id, split, count, load):
        return SimpleNamespace(sample_id=sample_id, split=split, task_mode='independent_voyage',
            operating_mode=('onboard',)*count, load_kw=(load,)*count,
            time_s=tuple(float(i) for i in range(count)), dt_s=(1.,)*count, duration_s=float(count))
    return (tuple(make(f'smoke_train_{i}', 'train', 64, 60.+i*20.) for i in range(2)),
            tuple(make(f'smoke_validation_{i}', 'validation', 8, 60.+i*20.) for i in range(2)))


def inspect_run(directory):
    directory = Path(directory).resolve()
    progress = json.loads((directory/'progress.json').read_text(encoding='utf-8'))
    best = json.loads((directory/'best_validation.json').read_text(encoding='utf-8')) if (
        directory/'best_validation.json').exists() else None
    return dict(directory=str(directory), progress=progress, best_validation=best)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, help='new output directory, or existing directory with --resume')
    parser.add_argument('--data-root', type=Path, default=ROOT/'data/processed/operating_dataset_v6_1s')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='tiny synthetic voyages through real direct physics')
    parser.add_argument('--parameters-json', type=Path, help='exported Optuna parameters; fresh directory required')
    parser.add_argument('--inspect', action='store_true', help='print current counters and best Validation; no training')
    parser.add_argument('--compare-runs', nargs=2, type=Path, metavar=('BASELINE', 'RETRAINED'))
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--learning-starts', type=int, default=50_000)
    parser.add_argument('--train-freq', type=int, default=32)
    parser.add_argument('--gradient-steps', type=int, default=1)
    parser.add_argument('--target-update-interval', type=int, default=10_000)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--buffer-size', type=int, default=1_000_000)
    parser.add_argument('--network-width', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--order-seed', type=int, default=42)
    parser.add_argument('--warmup-seed', type=int, default=43)
    parser.add_argument('--validation-every', type=int, default=50)
    parser.add_argument('--checkpoint-keep', type=int, default=2)
    parser.add_argument('--epsilon-initial', type=float, default=1.)
    parser.add_argument('--epsilon-final', type=float, default=.05)
    parser.add_argument('--epsilon-decay-fraction', type=float, default=.8)
    parser.add_argument('--reward-mode', choices=('zhou_economic_only', 'v6_energy_soc_ramp'),
                        default='zhou_economic_only')
    parser.add_argument('--max-env-steps', type=int, help='soft formal budget; stop only after a complete episode')
    parser.add_argument('--stop-after-steps', type=int, help='pause after N real transitions, preserving live state')
    parser.add_argument('--acknowledge-failure-penalty', action='store_true', help='accept unchanged legacy '
        '10132.66 CNY penalty: not calibrated on current Train; early failure may beat completion')
    args = parser.parse_args()
    if args.compare_runs:
        reports = [inspect_run(path) for path in args.compare_runs]
        if any(report['best_validation'] is None for report in reports):
            parser.error('both runs need a saved Validation result')
        identities = [json.loads((path/'identity.json').read_text(encoding='utf-8'))['data']['validation']
                      for path in args.compare_runs]
        if identities[0] != identities[1]:
            parser.error('runs use different Validation voyages/data')
        print(json.dumps(dict(runs=reports,
            preferred=str(args.compare_runs[int(reports[1]['best_validation']['score'] >
                                                 reports[0]['best_validation']['score'])])), indent=2))
        return
    if args.run_dir is None:
        parser.error('--run-dir is required')
    if args.inspect:
        print(json.dumps(inspect_run(args.run_dir), indent=2))
        return
    options = {field.name:getattr(args, field.name) for field in fields(EpisodeTrainingConfig)}
    if args.resume:
        if args.parameters_json:
            parser.error('--parameters-json requires a fresh run and cannot be combined with --resume')
        options = json.loads((args.run_dir/'configuration.json').read_text(encoding='utf-8'))
        if not options['smoke'] and not args.acknowledge_failure_penalty:
            parser.error('formal resume requires --acknowledge-failure-penalty')
    elif args.parameters_json:
        selected = json.loads(args.parameters_json.read_text(encoding='utf-8'))
        selected = selected.get('model_parameters', selected.get('parameters', selected))
        allowed = {'learning_rate', 'batch_size', 'target_update_interval', 'buffer_size', 'network_width', 'policy_kwargs'}
        if set(selected)-allowed:
            parser.error('parameter export contains unsupported model parameter names')
        options.update(selected)
    if args.smoke and not args.resume:
        options.update(episodes=3, learning_starts=64, buffer_size=512, batch_size=8,
                       network_width=16, target_update_interval=64, validation_every=1)
    config = _configuration(options)
    if not config.smoke and not config.acknowledge_failure_penalty:
        parser.error('--acknowledge-failure-penalty is required before formal training')
    if config.smoke:
        train, validation = _smoke_episodes()
    else:
        from v4.v6_dataset import V6TrainingDataset
        dataset = V6TrainingDataset.open(args.data_root)
        if dataset.split_sizes != (35, 8):
            raise ValueError('formal training requires the frozen 35 Train / 8 Validation split')
        train, validation = dataset.load_train(), dataset.load_validation()
    stopping = [False]
    def request_pause(signum, frame):
        stopping[0] = True
    old_handler = signal.signal(signal.SIGINT, request_pause)
    try:
        result = run_episode_training(config, train, validation, args.run_dir, resume=args.resume,
            stop_requested=lambda:stopping[0], stop_after_steps=args.stop_after_steps)
        print(f'{result["status"]} | episodes {result["completed_episodes"]}/{config.episodes}'
              f' | formal steps {result["formal_env_steps"]} | Adam {result["adam_updates"]}', flush=True)
    finally:
        signal.signal(signal.SIGINT, old_handler)


if __name__ == '__main__':
    main()
