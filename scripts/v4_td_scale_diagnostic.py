"""Reproducible synthetic reward-scale diagnosis; never loads official routes.

Run from any directory: python scripts/v4_td_scale_diagnostic.py
The default comparison uses raw rewards to reproduce the controlled failure-TD
test. Use --reward-mode redistributed for the production feedback-study timing.
Both reward timing modes use the same fixed corpus and penalty reference.
Production learn(), physics, prices, and default algorithms are unchanged.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import random
import struct
import sys
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

import torch
from v3.control import AccountState, EconomicMPC
from v4.control import ACTION_KW, DirectTransition, ReplayExecutionError, feasible_fc_actions, replay_episode
import v4.dqn as dqn_module
from v4.dqn import DirectPowerDDQN
from v4.experiment_schedule import EconomicUpdateSchedule
from v4.experiment_paths import unarchived_output_path
from v4.failure_replay import FAILURE_TERMINALS, FailurePenalty, prepare_failure_replay
from v4.reward_feedback import BatteryEnergyValue

INITIAL_SOC = .2135
LOADS_KW = (600., 1000., 0.)
BETA_SOC = 500.
SEED = 42
CORPUS_REPEATS = 82
ALPHAS = (1., .01, .001)


class SyntheticAccountant(EconomicMPC):
    """The existing physical accountant, with an explicit no-solver guard."""
    def __init__(self):
        super().__init__(nominal_cost_cny=1.)

    def solve(self, *args, **kwargs):
        raise AssertionError('synthetic direct-power diagnosis must never solve MPC')


@dataclass(frozen=True)
class SyntheticBranch:
    first_power_kw: int
    failed: bool
    raw: tuple[DirectTransition, ...]
    redistributed: tuple[DirectTransition, ...]


@dataclass(frozen=True)
class SyntheticCorpus:
    branches: tuple[SyntheticBranch, ...]
    penalty: FailurePenalty

    @property
    def safe_actions_kw(self):
        return tuple(b.first_power_kw for b in self.branches if not b.failed)

    def trajectories(self, redistributed=False):
        return tuple(values for b in self.branches
                     for values in [(b.redistributed if redistributed else b.raw)] * (16 if b.failed else 1))


def synthetic_episode():
    return SimpleNamespace(sample_id='synthetic_td_scale_600_1000_0', split='synthetic',
                           operating_mode=('onboard',) * 3, load_kw=LOADS_KW,
                           battery_bus_kw=(0.,) * 3)


def build_corpus():
    """Same61 branches and failure multiplicity as the controlled failure test."""
    accountant = SyntheticAccountant()
    penalty = FailurePenalty.from_reference()  # Frozen Train cost scalar JSON only.
    energy_value = BatteryEnergyValue.from_accountant(accountant)
    branches = []
    for power in ACTION_KW:
        results = []
        outcomes = []
        for feedback in (False, True):
            powers = iter((power, 600, 0))
            try:
                # Retain the frozen pre-Scheme-A synthetic corpus: the final
                # OFF command is physically legal but now strategy-gated.
                with patch('v4.control.policy_candidate_fc_actions',
                           side_effect=lambda physical, _previous: physical):
                    result = replay_episode(synthetic_episode(), lambda _s, _m: next(powers),
                        accountant=accountant, initial_state=AccountState(soc=INITIAL_SOC),
                        beta_soc=BETA_SOC, redistribute_battery_energy=feedback)
                results.append(result.transitions)
                outcomes.append(False)
            except ReplayExecutionError as error:
                if error.failure_kind != 'no_feasible_action' or error.failure_cause != 'soc_limited':
                    raise
                results.append(prepare_failure_replay(error, penalty=penalty,
                    energy_value=energy_value, redistribute_battery_energy=feedback).transitions)
                outcomes.append(True)
        if outcomes[0] != outcomes[1]:
            raise AssertionError('reward redistribution changed physical outcome')
        branches.append(SyntheticBranch(power, outcomes[0], *results))
    corpus = SyntheticCorpus(tuple(branches), penalty)
    assert [b.first_power_kw for b in branches if b.failed] == [0, 10, 20, 30]
    assert sum(map(len, corpus.trajectories())) == 235
    return corpus


def reward_components(transition):
    """Original unit components; all enter reward with their explicit signs."""
    return {
        'executed_economic_reward': -transition.executed_ledger.total_cost_cny,
        'observed_shore_reward': -(transition.shore_ledger.total_cost_cny if transition.shore_ledger else 0.),
        'modeled_terminal_settlement_reward': -(transition.modeled_terminal_ledger.total_cost_cny
                                                if transition.modeled_terminal_ledger else 0.),
        'soc_shaping_reward': -transition.soc_soft_penalty_cny,
        'immediate_battery_energy_adjustment': transition.immediate_battery_energy_adjustment,
        'terminal_correction': transition.terminal_correction,
        'failure_penalty_reward': -transition.failure_penalty_equivalent_cny,
    }


def ledger_fingerprint(corpus):
    values = [[{
        'state': t.state, 'next_state': t.next_state, 'action_kw': t.action_kw,
        'actual_soc': t.actual_soc, 'actual_battery_kw': t.actual_battery_kw,
        'executed_ledger': asdict(t.executed_ledger),
        'shore_ledger': asdict(t.shore_ledger) if t.shore_ledger else None,
        'modeled_terminal_ledger': asdict(t.modeled_terminal_ledger) if t.modeled_terminal_ledger else None,
        'original_reward': t.original_reward,
    } for t in b.raw] for b in corpus.branches]
    return hashlib.sha256(json.dumps(values, sort_keys=True, allow_nan=False).encode()).hexdigest()


def remember_scaled_trajectory(agent, values, *, alpha):
    """Scale the final replay reward exactly once; original transitions stay intact.

    Component addition happens in original units before multiplication. Penalty
    metadata is also multiplied once; remember_trajectory does not add it again.
    These are diagnostic reward units despite production ReplayItem field names.
    """
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError('alpha must be positive finite')
    count = agent.remember_trajectory(values)
    for offset in range(1, min(count, len(agent.replay)) + 1):
        item = agent.replay[-offset]
        agent.replay[-offset] = replace(item, reward_cny=alpha * item.reward_cny,
            failure_penalty_equivalent_cny=alpha * item.failure_penalty_equivalent_cny)
    return count


def distribution(values):
    tensor = torch.as_tensor(values, dtype=torch.float64).flatten()
    if not tensor.numel():
        return {'count': 0}
    if not torch.isfinite(tensor).all():
        raise FloatingPointError('non-finite diagnostic distribution')
    quantiles = torch.quantile(tensor, torch.tensor([0., .05, .5, .95, 1.], dtype=torch.float64))
    return {'count': tensor.numel(), 'mean': tensor.mean().item(),
            'std': tensor.std(unbiased=False).item(),
            **{name: value.item() for name, value in zip(('min', 'p05', 'p50', 'p95', 'max'), quantiles)}}


def probe(agent, corpus):
    state = corpus.branches[0].raw[0].state
    with torch.no_grad():
        q = agent.online(torch.tensor([state], dtype=torch.float32))[0]
        target_q = agent.target(torch.tensor([state], dtype=torch.float32))[0]
    safe = [ACTION_KW.index(power) for power in corpus.safe_actions_kw]
    order = sorted(range(len(ACTION_KW)), key=lambda i: (-q[i].item(), i))
    return {'q_values': q.tolist(), 'target_q_values': target_q.tolist(),
            'order_kw_descending': [ACTION_KW[i] for i in order],
            'greedy_power_kw': ACTION_KW[order[0]], 'fc0_rank': order.index(0) + 1,
            'q_fc0_minus_best_safe': q[0].item() - q[safe].max().item(),
            'q_fc0_minus_mean_safe': q[0].item() - q[safe].mean().item(),
            'q_fc0_minus_fc600': q[0].item() - q[60].item()}


def model_fingerprint(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


class SamplingObserver(random.Random):
    def __init__(self, previous):
        super().__init__()
        self.setstate(previous.getstate())
        self.last_batch = None
        self.last_indices = None
        self.digest = hashlib.sha256()

    def sample(self, population, k, *, counts=None):
        batch = super().sample(population, k, counts=counts)
        indices = {id(item): index for index, item in enumerate(population)}
        self.last_indices = [indices[id(item)] for item in batch]
        self.last_batch = batch
        self.digest.update(struct.pack(f'<{len(batch)}I', *self.last_indices))
        return batch


def _actual_greedy(agent, *, redistributed):
    accountant = SyntheticAccountant()
    saved_rng = agent.random.getstate()
    try:
        try:
            result = replay_episode(synthetic_episode(), lambda s, m: agent.select_power(s, m),
                accountant=accountant, initial_state=AccountState(soc=INITIAL_SOC),
                beta_soc=BETA_SOC, redistribute_battery_energy=redistributed)
            values = result.transitions
            return {'completed': True, 'failure_cause': None,
                    'powers_kw': [t.action_kw for t in values],
                    'soc': [INITIAL_SOC, *[t.actual_soc for t in values]],
                    'reward_sum_original_units': math.fsum(t.reward_cny for t in values),
                    'actual_ledger_cost_cny': result.total_cost_cny,
                    'modeled_terminal_cost_cny': result.comparable_cost_cny - result.total_cost_cny}
        except ReplayExecutionError as error:
            if error.failure_kind != 'no_feasible_action':
                raise
            return {'completed': False, 'failure_cause': error.failure_cause,
                    'powers_kw': [t.action_kw for t in error.executed_transitions],
                    'soc': [INITIAL_SOC, *[t.actual_soc for t in error.executed_transitions]]}
    finally:
        agent.random.setstate(saved_rng)


def run_scale(corpus, *, alpha, repeats=CORPUS_REPEATS, redistributed=False):
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError('alpha must be positive finite')
    if type(repeats) is not int or repeats < 1:
        raise ValueError('repeats must be a positive integer')
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            return _run_scale(corpus, alpha=alpha, repeats=repeats, redistributed=redistributed)
    finally:
        torch.set_num_threads(threads)


def _run_scale(corpus, *, alpha, repeats, redistributed):
    ledger_before = ledger_fingerprint(corpus)
    agent = DirectPowerDDQN(seed=SEED, gamma=1., learning_rate=1e-4, n_step=1)
    with torch.no_grad():
        agent.online.layers[-1].bias[0] += 1.  # Exact test-only initial preference.
    agent.sync_target()
    initial_copies = agent.target_sync_calls
    outcome_before = model_fingerprint(agent.outcome_model)
    before = probe(agent, corpus)
    mask = feasible_fc_actions(AccountState(soc=INITIAL_SOC), LOADS_KW[0], SyntheticAccountant())
    # Match the existing controlled test's one initial select_power RNG draw.
    assert agent.select_power(corpus.branches[0].raw[0].state, mask) == 0
    agent.random = SamplingObserver(agent.random)
    schedule = EconomicUpdateSchedule('replay32', target_mode='optimizer', target_interval=500)
    trajectories = corpus.trajectories(redistributed)
    records = []
    target_sync_at = []
    original_target = dqn_module.masked_double_dqn_targets
    original_clip = torch.nn.utils.clip_grad_norm_
    original_sync = agent.sync_target
    pending = {}

    def target_observer(*args, **kwargs):
        target = original_target(*args, **kwargs)
        batch = agent.random.last_batch
        with torch.no_grad():
            states = torch.tensor([item.state for item in batch], dtype=torch.float32)
            actions = torch.tensor([[item.action_index] for item in batch], dtype=torch.long)
            q = agent.online(states).gather(1, actions)
            error = (target - q).flatten()
            failed = torch.tensor([item.experience_outcome == 'failure' for item in batch])
            terminal = torch.tensor([item.terminal_reason in FAILURE_TERMINALS for item in batch])
            pending.clear()
            pending.update({
                'update': agent.economic_optimizer_updates + 1, 'alpha': alpha,
                'replay_size': len(agent.replay), 'sample_count': len(batch),
                'sample_success_count': sum(item.experience_outcome == 'success' for item in batch),
                'sample_failure_count': int(failed.sum().item()),
                'sample_failure_terminal_count': int(terminal.sum().item()),
                'sample_success_fraction': 1 - failed.float().mean().item(),
                'sample_failure_fraction': failed.float().mean().item(),
                'sample_failure_terminal_fraction': terminal.float().mean().item(),
                'sample_indices_sha256': hashlib.sha256(struct.pack('<64I', *agent.random.last_indices)).hexdigest(),
                'td_signed_mean': error.mean().item(),
                'td_mae': error.abs().mean().item(),
                'td_rmse': error.square().mean().sqrt().item(),
                'smooth_l1_loss': torch.nn.functional.smooth_l1_loss(q, target).item(),
                'huber_linear_fraction': (error.abs() > 1.).float().mean().item(),
                'td_signed_mean_original_units': error.mean().item() / alpha,
                'td_mae_original_units': error.abs().mean().item() / alpha,
                'td_rmse_original_units': error.square().mean().sqrt().item() / alpha,
                'td_mae_over_common_failure_reference': error.abs().mean().item() / (alpha * corpus.penalty.amount),
                'smooth_l1_original_units_beta1': torch.nn.functional.smooth_l1_loss(q / alpha, target / alpha).item(),
                'failure_terminal_td_mae': error[terminal].abs().mean().item() if terminal.any() else None,
                'success_td_mae': error[~failed].abs().mean().item() if (~failed).any() else None,
            })
            selected = args[1].masked_fill(~kwargs['next_action_masks'], -torch.inf).argmax(dim=1, keepdim=True)
            continuing = args[3].flatten() == 0
            bootstrap_online_q = args[1].gather(1, selected).flatten()[continuing]
            bootstrap_target_q = args[2].gather(1, selected).flatten()[continuing]
            for name, values in (('target', target), ('predicted_q', q), ('td_signed', error),
                                 ('td_absolute', error.abs()), ('reward', args[0]),
                                 ('bootstrap_online_q', bootstrap_online_q),
                                 ('bootstrap_target_q', bootstrap_target_q)):
                pending.update({f'{name}_{key}': value for key, value in distribution(values).items() if key != 'count'})
        return target

    def clip_observer(*args, **kwargs):
        norm = original_clip(*args, **kwargs)
        value = float(norm.item())
        if not math.isfinite(value):
            raise FloatingPointError('non-finite preclip gradient norm')
        pending.update(preclip_gradient_norm=value, clipping_triggered=int(value > 10.))
        records.append(dict(pending))
        return norm

    def sync_observer():
        original_sync()
        target_sync_at.append(agent.economic_optimizer_updates)

    agent.sync_target = sync_observer
    with patch.object(dqn_module, 'masked_double_dqn_targets', target_observer), \
         patch.object(torch.nn.utils, 'clip_grad_norm_', clip_observer):
        for _ in range(repeats):
            count = sum(remember_scaled_trajectory(agent, values, alpha=alpha) for values in trajectories)
            schedule.grant(insertions=count, completed_episodes=0)
            schedule.consume(agent, batch_size=64)
    after = probe(agent, corpus)
    actual = _actual_greedy(agent, redistributed=redistributed)
    assert ledger_fingerprint(corpus) == ledger_before
    assert len(records) == agent.economic_optimizer_updates == agent.economic_replay_insertions // 32
    assert agent.target_sync_calls == initial_copies + agent.economic_optimizer_updates // 500
    assert target_sync_at == list(range(500, agent.economic_optimizer_updates + 1, 500))
    return {
        'alpha': alpha, 'reward_mode': 'redistributed' if redistributed else 'raw',
        'sampling_sha256': agent.random.digest.hexdigest(),
        'ledger_sha256_before': ledger_before, 'ledger_sha256_after': ledger_fingerprint(corpus),
        'outcome_weights_unchanged': outcome_before == model_fingerprint(agent.outcome_model),
        'counts': {
            'replay_insertions': agent.economic_replay_insertions,
            'success_insertions': agent.economic_success_replay_insertions,
            'failure_insertions': agent.economic_failure_replay_insertions,
            'failure_terminal_insertions': agent.economic_failure_terminal_insertions,
            'optimizer_updates': agent.economic_optimizer_updates,
            'outcome_optimizer_updates': agent.outcome_optimizer_updates,
            'remaining_transition_credit': schedule.remaining_transition_credit,
            'initial_target_copies': initial_copies,
            'initial_copy_breakdown': {'constructor': 1, 'after_synthetic_fc0_bias': 1},
            'scheduled_target_copies': len(target_sync_at),
            'scheduled_target_sync_at_updates': target_sync_at,
            'target_sync_calls_total': agent.target_sync_calls,
        },
        'probe_before': before, 'probe_after': after, 'actual_greedy': actual,
        'aggregate': summarize_updates(records), 'production_td_statistics': agent.td_statistics(),
        'updates': records,
    }


def summarize_updates(records):
    metrics = ('td_mae', 'smooth_l1_loss', 'td_signed_mean', 'td_mae_original_units',
               'td_mae_over_common_failure_reference', 'smooth_l1_original_units_beta1',
               'preclip_gradient_norm', 'sample_failure_terminal_fraction', 'huber_linear_fraction')
    result = {name: distribution([r[name] for r in records]) for name in metrics}
    result['clipping_trigger_fraction'] = sum(r['clipping_triggered'] for r in records) / len(records)
    result['clipping_trigger_count'] = sum(r['clipping_triggered'] for r in records)
    result['last_50_updates'] = {name: distribution([r[name] for r in records[-50:]]) for name in metrics}
    return result


def corpus_metadata(corpus):
    identities = []
    for b in corpus.branches:
        raw_sum = math.fsum(t.reward_cny for t in b.raw)
        shifted_sum = math.fsum(t.reward_cny for t in b.redistributed)
        assert math.isclose(raw_sum, shifted_sum, rel_tol=1e-12, abs_tol=1e-8)
        for raw, shifted in zip(b.raw, b.redistributed):
            assert raw.state == shifted.state and raw.next_state == shifted.next_state
            assert raw.executed_ledger == shifted.executed_ledger
            assert raw.shore_ledger == shifted.shore_ledger
            assert raw.modeled_terminal_ledger == shifted.modeled_terminal_ledger
            for t in (raw, shifted):
                assert math.isclose(math.fsum(reward_components(t).values()), t.reward_cny,
                                    rel_tol=1e-12, abs_tol=1e-8)
        identities.append({'first_power_kw': b.first_power_kw, 'failed': b.failed,
                           'raw_return': raw_sum, 'redistributed_return': shifted_sum,
                           'return_difference': shifted_sum - raw_sum,
                           'raw_reward_components': [reward_components(t) for t in b.raw],
                           'redistributed_reward_components': [reward_components(t) for t in b.redistributed]})
    profiles = {}
    for mode, feedback in (('raw', False), ('redistributed', True)):
        unique = [t for b in corpus.branches for t in (b.redistributed if feedback else b.raw)]
        normal_steps = [abs(t.reward_cny) for t in unique if not t.done]
        successful_terminal = [abs(t.reward_cny) for t in unique if t.is_successful_terminal]
        failed_terminal = [abs(t.reward_cny) for t in unique if t.terminal_reason in FAILURE_TERMINALS]
        success_returns = [abs(math.fsum(t.reward_cny for t in (b.redistributed if feedback else b.raw)))
                           for b in corpus.branches if not b.failed]
        profiles[mode] = {
            'ordinary_nonterminal_abs_reward': distribution(normal_steps),
            'successful_terminal_abs_reward': distribution(successful_terminal),
            'failed_terminal_abs_reward': distribution(failed_terminal),
            'completed_segment_abs_return': distribution(success_returns),
            'cf_over_median_ordinary_step': corpus.penalty.amount / distribution(normal_steps)['p50'],
            'cf_over_median_complete_segment_return': corpus.penalty.amount / distribution(success_returns)['p50'],
        }
    reference_path = ROOT / 'src/v4/failure_penalty_reference.json'
    frozen_reference = json.loads(reference_path.read_text(encoding='utf-8'))
    frozen_costs = list(frozen_reference['training_costs_cny'].values())
    scaling = [{
        'alpha': alpha, 'cf_scaled': alpha * corpus.penalty.amount,
        'mode': mode, 'max_component_sum_identity_error': max(abs(
            math.fsum(alpha * value for value in reward_components(t).values()) - alpha * t.reward_cny)
            for b in corpus.branches for t in (b.redistributed if mode == 'redistributed' else b.raw)),
        'completed_return_range_scaled': [alpha * profiles[mode]['completed_segment_abs_return'][name]
                                          for name in ('min', 'max')],
    } for alpha in ALPHAS for mode in ('raw', 'redistributed')]
    return {
        'scope': 'Synthetic fixed three-row branches; no Train/Validation/Test trajectories or payloads.',
        'initial_soc': INITIAL_SOC, 'loads_kw': LOADS_KW, 'prescribed_powers_after_first': [600, 0],
        'beta_soc': BETA_SOC, 'first_actions_kw': list(ACTION_KW),
        'safe_first_actions_kw': corpus.safe_actions_kw,
        'failed_first_actions_kw': [b.first_power_kw for b in corpus.branches if b.failed],
        'unique_branch_count': len(corpus.branches), 'failure_branch_multiplicity': 16,
        'corpus_trajectory_count': len(corpus.trajectories()),
        'economic_transitions_per_cycle': sum(map(len, corpus.trajectories())),
        'successful_transitions_per_cycle': 171, 'failed_terminal_transitions_per_cycle': 64,
        'corpus_cycles': CORPUS_REPEATS, 'ledger_sha256': ledger_fingerprint(corpus),
        'failure_reference': asdict(corpus.penalty), 'cf_original_units': corpus.penalty.amount,
        'allowed_frozen_cost_json': str(reference_path),
        'allowed_frozen_cost_json_sha256': hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        'frozen_train_completed_economic_cost_distribution': distribution(frozen_costs),
        'cf_over_frozen_train_median_episode_cost': corpus.penalty.amount / distribution(frozen_costs)['p50'],
        'frozen_cost_scope': frozen_reference['cost_scope'],
        'raw_versus_redistributed_identity': identities,
        'reward_scale_profiles': profiles, 'scaling_identity_checks': scaling,
        'ledger_units': 'Original actual/modelled accounting CNY; no scaled ledger copies.',
        'scaled_reward_units': 'alpha times original CNY-equivalent reward, including SOC shaping and CF.',
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        default=ROOT / 'outputs/v4_td_scale_diagnostic_20261008')
    parser.add_argument('--reward-mode', choices=('raw', 'redistributed'), default='raw',
                        help='synthetic reward timing mode; does not change production defaults')
    args = parser.parse_args(argv)
    args.output_dir = unarchived_output_path(args.output_dir, require_empty=True)
    corpus = build_corpus()
    metadata = corpus_metadata(corpus)
    runs = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for alpha in ALPHAS:
        run = run_scale(corpus, alpha=alpha, redistributed=args.reward_mode == 'redistributed')
        records = run.pop('updates')
        destination = args.output_dir / f'alpha_{alpha:g}_updates.csv'
        with destination.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        run['per_update_csv'] = str(destination.resolve())
        runs.append(run)
        print(json.dumps({'alpha': alpha, 'reward_mode': args.reward_mode,
            'updates': run['counts']['optimizer_updates'],
            'clipping_fraction': run['aggregate']['clipping_trigger_fraction'],
            'greedy_first_power_kw': run['probe_after']['greedy_power_kw'],
            'greedy_complete': run['actual_greedy']['completed']}, allow_nan=False), flush=True)
    assert len({run['sampling_sha256'] for run in runs}) == 1
    summary = {
        'scope': metadata['scope'], 'formal_training_started': False,
        'official_trajectory_payload_reads': 0,
        'production_parameters': {'network': [8, 128, 64, 61], 'optimizer': 'Adam',
            'learning_rate': 1e-4, 'loss': 'SmoothL1 beta=1', 'clip_norm': 10,
            'gamma': 1., 'n_step': 1, 'batch_size': 64, 'cadence': 'replay32',
            'target_mode': 'optimizer', 'target_interval': 500, 'seed': SEED},
        'synthetic_protocol': {'cycles': CORPUS_REPEATS, 'transitions_per_cycle': 235,
            'total_insertions': 19270, 'optimizer_updates': 602,
            'initial_fc0_bias_increment': 1., 'main_reward_mode': args.reward_mode,
            'mode_reason': ('Matches the existing controlled failure-TD test.' if args.reward_mode == 'raw'
                            else 'Matches current feedback-study timing, using the same fixed corpus and penalty as raw diagnosis.'),
            'thread_setting': '1 during each synthetic run only, original count restored afterward',
            'replay_capacity': 100000},
        'common_reference_for_dimensionless_td_error': corpus.penalty.amount,
        'comparison_note': 'Smaller native loss alone is numeric shrinkage. Compare TD/alpha, TD/(alpha*CF), Q ranking and actual full trajectory completion.',
        'huber_note': 'SmoothL1 beta=1 has derivative magnitude capped at1 for residuals beyond1; huge TD residuals alone do not imply proportional gradients or divergence. Scaling changes the quadratic/linear loss regime. The fixed+1 initial bias, Adam epsilon, and clip threshold also make optimizer trajectories non-invariant under scaling.',
        'sampling_identical_across_scales': True,
        'corpus_metadata_file': str((args.output_dir / 'corpus_metadata.json').resolve()),
        'results': runs,
        'interpretation_limit': 'One synthetic scenario, one seed,602updates; no convergence or formal-route performance claim.',
    }
    for filename, content in (('summary.json', summary), ('corpus_metadata.json', metadata)):
        (args.output_dir / filename).write_text(json.dumps(content, indent=2, allow_nan=False), encoding='utf-8')
    print(f'Saved synthetic diagnosis to {args.output_dir.resolve()}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
