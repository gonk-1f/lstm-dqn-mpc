"""Audit evidence must distinguish complete summaries from sparse observations."""
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def summary(*, completed=True, cost=100.0, penalty=0.0):
    return dict(sample_id="parent_train", status="completed" if completed else "soc_limited",
        completed=str(completed), executed_steps="32", voyage_duration_s="100",
        executed_duration_s="32", soc_terminal="0.6", duration_coverage="0.32",
        fc_h2_cost_cny="10", fc_degradation_cost_cny=str(cost-12),
        battery_degradation_cost_cny="2", battery_h2eq_cost_cny="0",
        shore_electricity_cost_cny="0", executed_prefix_economic_cost_cny=str(cost),
        complete_economic_cost_cny=str(cost) if completed else "",
        failure_penalty_reward_units=str(penalty))


def test_archived_summary_reconstructs_economic_cost_separately_from_failure():
    audit = load_script("audit_episode_failure_penalty")
    row = audit.summarize_train_row(summary(completed=False, cost=100, penalty=50))
    assert row["economic_cost_cny"] == 100
    assert row["economic_only_return"] == -150
    assert row["complete_economic_cost_cny"] is None
    assert row["economic_ledger_identity_residual_cny"] == 0


def test_sparse_rows_do_not_invent_prefix_cost_or_step_degradation():
    audit = load_script("audit_episode_failure_penalty")
    rows = [dict(source_split="train", sample_id="parent_train", origin_round="1",
        time_s=str(t), fc_h2="1", battery_h2eq="0", failure_penalty="0",
        soc_before="0.6", soc_after="0.6", fc_kw="0", delta_kw="0") for t in (0, 600)]
    result = audit.summarize_sparse_group(rows)
    assert not result["contiguous_from_reset"]
    assert result["economic_prefix_cost_cny"] is None
    assert result["complete_economic_cost_cny"] is None
    assert result["step_economic_cost_distribution_cny"] is None


def test_distribution_uses_the_standard_even_sample_median():
    audit = load_script("audit_episode_failure_penalty")
    assert audit.distribution([1, 3])["median"] == 2


def test_inconsistent_archived_economic_ledger_is_rejected():
    audit = load_script("audit_episode_failure_penalty")
    row = summary(cost=100)
    row["fc_h2_cost_cny"] = "11"
    with pytest.raises(ValueError, match="inconsistent"):
        audit.summarize_train_row(row)


@pytest.mark.parametrize("path", ["test/p.csv", "round_01_validation/p.csv", "test_summary.json"])
def test_nontrain_payload_path_rejected_before_open(path):
    audit = load_script("audit_episode_failure_penalty")
    with pytest.raises(ValueError, match="Train-only"):
        audit.train_only_path(path)


def test_only_same_id_complete_failed_behaviors_are_compared():
    audit = load_script("audit_episode_failure_penalty")
    rows = [audit.summarize_train_row(summary(cost=1000)),
            audit.summarize_train_row(summary(completed=False, cost=20, penalty=50))]
    comparisons = audit.compare_same_sample(rows, penalty=50)
    assert len(comparisons) == 1
    assert comparisons[0]["break_even_failure_penalty_cny"] == 980
    assert comparisons[0]["failed_return_exceeds_completed_return"]
    rows[1]["sample_id"] = "different_train"
    assert audit.compare_same_sample(rows, penalty=50) == []


def test_benchmark_limits_reject_large_or_nonpositive_budgets():
    benchmark = load_script("benchmark_v6_environment")
    for sizes in ([257], [0], [32, 64, 128, 256, 256, 256, 256, 256, 256]):
        with pytest.raises(ValueError):
            benchmark.validate_budget(sizes, repeats=3)
    assert benchmark.validate_budget([32, 64, 128], repeats=3) is None
    with pytest.raises(ValueError):
        benchmark.validate_budget([32], repeats=11)


def test_benchmark_equivalence_reports_real_divergence():
    benchmark = load_script("benchmark_v6_environment")
    import numpy as np
    a = dict(observation=np.zeros(5), reward=1., terminated=False, truncated=False,
             mask=np.ones(13, dtype=bool), transition=dict(actual_soc=.6))
    b = dict(a, transition=dict(actual_soc=.61))
    with pytest.raises(AssertionError, match="actual_soc"):
        benchmark.assert_equivalent(a, b)


def test_budget_uses_only_metadata_durations_and_excludes_first_adam():
    benchmark = load_script("benchmark_v6_environment")
    manifest = [dict(split="train",candidate_id=f"train_{i:02}",duration_s=100+i) for i in range(35)]
    manifest += [dict(split="validation",candidate_id=f"val_{i}",duration_s=100) for i in range(8)]
    # No Test duration exists: accessing it would fail this test.
    manifest += [dict(split="test",candidate_id="unopened_test")]
    measured = dict(economic_adam_microbenchmark=dict(step_mean_s=.001,
        adam_single_update_seconds=[100.,.01,.03]))
    result = benchmark.estimate_work_budget(manifest,measured)
    assert result["adam_remaining_median_s"] == .02
    assert result["Validation_metadata_only_duration_sum_steps"] == 800
    scenario = result["scenarios"][-1]
    assert sum(scenario["seed42_voyage_counts"].values()) == 500
    assert scenario["uniform_replacement_expected"]["formal_steps"] == 500*117
    assert scenario["validation_full_completion_steps"] == 8000
