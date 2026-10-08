"""Synthetic diagnosis must preserve economic contracts and update accounting."""
from importlib.util import module_from_spec, spec_from_file_location
from functools import lru_cache
from math import fsum, isfinite
from pathlib import Path
import sys

import pytest
import torch


@lru_cache()
def load_diagnostic():
    path = Path(__file__).resolve().parents[1] / 'scripts/v4_td_scale_diagnostic.py'
    assert path.is_file(), 'the standalone synthetic TD-scale diagnostic is missing'
    spec = spec_from_file_location('v4_td_scale_diagnostic', path)
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@lru_cache()
def load_corpus():
    return load_diagnostic().build_corpus()


def test_fixed_corpus_preserves_raw_and_redistributed_returns():
    diagnostic = load_diagnostic()
    corpus = load_corpus()
    assert len(corpus.branches) == 61
    assert [b.first_power_kw for b in corpus.branches if b.failed] == [0, 10, 20, 30]
    assert len(corpus.trajectories(False)) == 121
    assert sum(map(len, corpus.trajectories(False))) == 235
    for branch in corpus.branches:
        assert len(branch.raw) == (1 if branch.failed else 3)
        assert fsum(t.reward_cny for t in branch.raw) == pytest.approx(
            fsum(t.reward_cny for t in branch.redistributed), abs=1e-8)
        for raw, redistributed in zip(branch.raw, branch.redistributed):
            assert raw.state == redistributed.state
            assert raw.next_state == redistributed.next_state
            assert raw.executed_ledger == redistributed.executed_ledger
            assert raw.modeled_terminal_ledger == redistributed.modeled_terminal_ledger


@pytest.mark.parametrize('alpha', [1., .01, .001])
def test_scale_entire_reward_once_without_mutating_real_ledgers(alpha):
    diagnostic = load_diagnostic()
    from v4.dqn import DirectPowerDDQN
    corpus = load_corpus()
    agent = DirectPowerDDQN(seed=42)
    originals = corpus.trajectories(True)
    ledger_hash = diagnostic.ledger_fingerprint(corpus)
    for values in originals:
        diagnostic.remember_scaled_trajectory(agent, values, alpha=alpha)
    flattened = [t for values in originals for t in values]
    assert len(agent.replay) == 235
    for raw, item in zip(flattened, agent.replay):
        components = diagnostic.reward_components(raw)
        assert fsum(components.values()) == pytest.approx(raw.reward_cny, abs=1e-8)
        assert fsum(alpha * x for x in components.values()) == pytest.approx(item.reward_cny, abs=1e-8)
        assert item.reward_cny == alpha * raw.reward_cny
        assert item.failure_penalty_equivalent_cny == alpha * raw.failure_penalty_equivalent_cny
    assert diagnostic.ledger_fingerprint(corpus) == ledger_hash


def test_shared_sampling_602_updates_target500_and_threads_restore():
    diagnostic = load_diagnostic()
    corpus = load_corpus()
    threads = torch.get_num_threads()
    fingerprint = diagnostic.ledger_fingerprint(corpus)
    runs = [diagnostic.run_scale(corpus, alpha=alpha) for alpha in (1., .01, .001)]
    assert torch.get_num_threads() == threads
    assert diagnostic.ledger_fingerprint(corpus) == fingerprint
    assert len({run['sampling_sha256'] for run in runs}) == 1
    for run in runs:
        assert run['counts']['optimizer_updates'] == 602
        assert run['counts']['replay_insertions'] == 19270
        assert run['counts']['success_insertions'] == 14022
        assert run['counts']['failure_terminal_insertions'] == 5248
        assert run['counts']['failure_insertions'] == 5248
        assert run['counts']['remaining_transition_credit'] == 6
        assert run['counts']['initial_target_copies'] == 2
        assert run['counts']['scheduled_target_copies'] == 1
        assert run['counts']['target_sync_calls_total'] == 3
        assert run['counts']['scheduled_target_sync_at_updates'] == [500]
        assert run['counts']['outcome_optimizer_updates'] == 0
        assert run['outcome_weights_unchanged'] is True
        assert run['probe_before']['greedy_power_kw'] == 0
        assert len(run['probe_after']['q_values']) == 61
        assert sorted(run['probe_after']['order_kw_descending']) == list(range(0, 601, 10))
        assert len(run['updates']) == 602
        assert all(isfinite(row['preclip_gradient_norm']) and row['preclip_gradient_norm'] >= 0 for row in run['updates'])
        assert all(row['sample_count'] == 64 for row in run['updates'])
        assert all(row['sample_success_count'] + row['sample_failure_count'] == 64 for row in run['updates'])
        assert all(row['sample_failure_terminal_count'] == row['sample_failure_count'] for row in run['updates'])
        assert all(row['td_mae_original_units'] == pytest.approx(row['td_mae'] / run['alpha']) for row in run['updates'])
    assert runs[0]['actual_greedy']['completed'] is True
    assert runs[0]['probe_after']['greedy_power_kw'] in corpus.safe_actions_kw


@pytest.mark.parametrize('alpha', [1., .01, .001])
@pytest.mark.parametrize('feedback', [False, True])
def test_nonzero_actual_shore_reward_scales_without_duplicate_settlement(alpha, feedback):
    from test_v4_reward_feedback import CASES, fixed_replay
    from v4.dqn import DirectPowerDDQN
    from dataclasses import asdict
    diagnostic = load_diagnostic()
    case = next(c for c in CASES if c[0] == 'C_actual_shore')
    result, _ = fixed_replay(case, feedback=feedback)
    ledgers_before = asdict(result.total_ledger)
    assert result.total_ledger.shore_cost_cny > 0
    assert result.modeled_terminal_settlement is None
    assert result.transitions[-1].shore_ledger.total_cost_cny > 0
    agent = DirectPowerDDQN(seed=42)
    count = diagnostic.remember_scaled_trajectory(agent, result.transitions, alpha=alpha)
    assert count == len(result.transitions)
    for actual, item in zip(result.transitions, agent.replay):
        components = diagnostic.reward_components(actual)
        assert item.reward_cny == pytest.approx(fsum(alpha*v for v in components.values()), abs=1e-8)
        assert item.reward_cny == alpha*actual.reward_cny
        assert actual.modeled_terminal_ledger is None
    assert diagnostic.reward_components(result.transitions[-1])['observed_shore_reward'] < 0
    assert asdict(result.total_ledger) == ledgers_before
    assert agent.economic_optimizer_updates == agent.outcome_optimizer_updates == 0


def test_records_huber_regime_and_bootstrap_target_q_before_updates():
    diagnostic = load_diagnostic()
    run = diagnostic.run_scale(load_corpus(), alpha=.001, repeats=1)
    assert run['probe_before']['target_q_values'] == run['probe_before']['q_values']
    for row in run['updates']:
        assert 0 <= row['huber_linear_fraction'] <= 1
        assert isfinite(row['bootstrap_target_q_p50'])
        assert isfinite(row['bootstrap_online_q_p50'])


def test_cli_rejects_nonempty_output_before_building_corpus(tmp_path, monkeypatch):
    diagnostic = load_diagnostic()
    sentinel = tmp_path / 'original.txt'
    sentinel.write_text('preserve', encoding='utf-8')
    monkeypatch.setattr(diagnostic, 'build_corpus', lambda: pytest.fail('output guard must precede corpus construction'))
    with pytest.raises(FileExistsError):
        diagnostic.main(['--output-dir', str(tmp_path)])
    assert sentinel.read_text(encoding='utf-8') == 'preserve'


def test_cli_rejects_archive_output_before_building_corpus(tmp_path, monkeypatch):
    diagnostic = load_diagnostic()
    destination = tmp_path / 'docs/results/old_raw'
    monkeypatch.setattr(diagnostic, 'build_corpus', lambda: pytest.fail('archive guard must precede corpus construction'))
    with pytest.raises(ValueError, match='archive paths are read-only'):
        diagnostic.main(['--output-dir', str(destination)])
    assert not destination.exists()


@pytest.mark.parametrize('alpha', [0., -1., float('nan'), float('inf')])
def test_rejects_invalid_scale_before_replay_insertion(alpha):
    diagnostic = load_diagnostic()
    from v4.dqn import DirectPowerDDQN
    agent = DirectPowerDDQN(seed=42)
    with pytest.raises(ValueError, match='positive finite'):
        diagnostic.remember_scaled_trajectory(agent, (), alpha=alpha)
    assert agent.economic_replay_insertions == 0


@pytest.mark.parametrize('reward_mode', [None, 'redistributed'])
def test_cli_forwards_reward_mode_to_every_scale(tmp_path, monkeypatch, reward_mode):
    import json
    from types import SimpleNamespace
    diagnostic = load_diagnostic()
    observed = []
    corpus = SimpleNamespace(penalty=SimpleNamespace(amount=10132.660087898294))
    monkeypatch.setattr(diagnostic, 'build_corpus', lambda: corpus)
    monkeypatch.setattr(diagnostic, 'corpus_metadata', lambda _corpus: {'scope': 'synthetic CLI routing test'})

    def observe_scale(actual_corpus, *, alpha, redistributed=False):
        assert actual_corpus is corpus
        observed.append((alpha, redistributed))
        return {'updates': [{'update': 1, 'alpha': alpha}],
                'counts': {'optimizer_updates': 0},
                'aggregate': {'clipping_trigger_fraction': 0},
                'probe_after': {'greedy_power_kw': 0},
                'actual_greedy': {'completed': False},
                'sampling_sha256': 'same-synthetic-sampling-sequence'}

    monkeypatch.setattr(diagnostic, 'run_scale', observe_scale)
    destination = tmp_path / 'new-diagnostic-output'
    argv = ['--output-dir', str(destination)]
    if reward_mode is not None:
        argv += ['--reward-mode', reward_mode]
    assert diagnostic.main(argv) == 0
    assert observed == [(alpha, reward_mode == 'redistributed') for alpha in diagnostic.ALPHAS]
    summary = json.loads((destination / 'summary.json').read_text(encoding='utf-8'))
    assert summary['synthetic_protocol']['main_reward_mode'] == (reward_mode or 'raw')
