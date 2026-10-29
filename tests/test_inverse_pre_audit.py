"""Reproducible pre-change evidence; no alternate reward is enabled."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def audit():
    path = Path(__file__).parents[1] / 'scripts/audit_inverse_economics.py'
    assert path.is_file(), 'Train-only inverse economic audit script must exist'
    spec = importlib.util.spec_from_file_location('inverse_pre_audit', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_distribution_reports_signed_values_and_missing_without_imputation():
    result = audit().distribution([-2., 0., 1., 3., None])
    assert result == pytest.approx(dict(count=4, missing=1, min=-2., p1=-1.94, median=.5,
                          p99=2.94, max=3., mean=.5, zero=1, negative=1))
    assert audit().distribution([None])['min'] is None


def test_distribution_rejects_nonfinite_instead_of_silently_filtering():
    with pytest.raises(ValueError, match='finite'):
        audit().distribution([float('inf')])


def test_boundary_audit_proves_legal_zero_and_negative_cost():
    module = audit()
    cases = {row['case']: row for row in module.boundary_cases()}
    idle = cases['idle_zero']
    assert idle['physical_legal'] and idle['increment13_legal']
    assert idle['total_cost_cny'] == 0.
    assert idle['h2_cost_cny'] == idle['fc_degradation_cost_cny'] == idle['battery_degradation_cost_cny'] == idle['soc_equivalent_cost_cny'] == 0.
    eol = cases['efficient_charge_both_eol']
    assert eol['physical_legal'] and eol['increment13_legal']
    assert eol['total_cost_cny'] == pytest.approx(-.0012402552831242142, abs=1e-14)
    assert eol['inverse_reward'] is None
    fresh = cases['efficient_charge_fresh']
    assert fresh['fc_loss_before_uv'] > 0.  # Constructed by actual legal warmup.
    assert fresh['soc_equivalent_cost_cny'] < -fresh['h2_cost_cny']
    assert fresh['total_cost_cny'] > 0.


def test_step_audit_uses_executed_cost_and_one_telescoping_soc_correction():
    episode = SimpleNamespace(sample_id='audit_fixture', split='train',
        task_mode='independent_voyage', operating_mode=('onboard',) * 4,
        load_kw=(100., 40., 80., 100.), time_s=(0., 1., 2., 3.),
        dt_s=(1.,) * 4, duration_s=4.)
    rows, result = audit().run_trajectory(episode, policy='load_follow', max_steps=4)
    assert result['completed']
    assert result['steps'] == 4
    assert abs(result['soc_telescope_residual_cny']) < 1e-12
    assert all(abs(row['power_balance_residual_kw']) < 1e-12 for row in rows)
    assert result['economic_cost_cny'] == pytest.approx(sum(row['total_cost_cny'] for row in rows))


def test_sparse_archive_does_not_invent_degradation_or_complete_cost():
    result = audit().summarize_sparse_rows([
        dict(source_split='train', time_s='0', fc_h2='1', battery_h2eq='-2'),
        dict(source_split='train', time_s='10', fc_h2='2', battery_h2eq='0')])
    assert result['complete_step_economic_distribution_cny'] is None
    assert result['fc_degradation_distribution_cny'] is None
    assert result['battery_degradation_distribution_cny'] is None
    assert not result['contiguous_from_reset']
    assert result['soc_equivalent_distribution_cny']['negative'] == 1


def test_trajectory_rejects_nontrain_before_execution():
    with pytest.raises(ValueError, match='Train'):
        audit().run_trajectory(SimpleNamespace(split='validation'), policy='load_follow', max_steps=1)


def test_equal_soc_cycle_cancels_fixed_value_and_keeps_actual_losses():
    cycle = audit().round_trip_cycle()
    assert cycle['soc_after'] == pytest.approx(cycle['soc_before'], abs=1e-15)
    assert cycle['discharge_bus_kwh'] / cycle['charge_bus_kwh'] == pytest.approx(.95**2)
    assert abs(cycle['soc_correction_sum_cny']) < 1e-11
    assert cycle['total_cost_cny'] == pytest.approx(cycle['actual_ledger_cost_cny'], abs=1e-11)
    assert cycle['actual_ledger_cost_cny'] > 0
