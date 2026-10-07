import pytest

from test_v4_training import FakeDataset
from v4.train import run_train_validation


def test_soc_time_bins_are_disjoint_and_include_exact_boundaries():
    from v4.telemetry import soc_time_occupancy

    result = soc_time_occupancy((0.2, 0.4, 0.6, 0.6001, 0.7899, 0.79, 0.8))
    assert list(result["counts"].values()) == [1, 2, 2, 2]
    assert sum(result["fractions"].values()) == pytest.approx(1.0)
    assert result["executed_seconds"] == 7 * 30
    assert soc_time_occupancy(())["fractions"] == dict.fromkeys(result["counts"], None)


def test_training_counts_real_updates_and_transitions_even_without_progress():
    _, report = run_train_validation(
        FakeDataset(), rounds=2, batch_size=1, updates_per_episode=1,
        seed=1, beta_soc=250.0, progress_every_steps=0,
    )
    counts = report["execution_counts"]
    assert counts["training_onboard_executed_transitions"] == 2
    assert counts["bootstrap_onboard_executed_transitions"] == 1
    assert counts["greedy_evaluation_onboard_executed_transitions"] == 2
    assert counts["economic_q_optimizer_updates"] == 3
    assert counts["target_sync_calls_including_initial_copy"] == 4
    assert counts["target_sync_calls_after_initialization"] == 3
    assert [row["economic_q_optimizer_updates"] for row in report["training_rounds"]] == [1, 1]
    assert report["validation"]["soc_time_occupancy"]["executed_steps"] == 1


def test_failure_prefix_counts_as_executed_without_an_economic_update():
    from types import SimpleNamespace

    class PrefixFailureDataset(FakeDataset):
        def load_train(self):
            return (SimpleNamespace(
                sample_id="failed_prefix", split="train",
                operating_mode=("onboard", "onboard"),
                load_kw=(100.0, 2000.0), battery_bus_kw=(0.0, 0.0),
            ),)

    _, report = run_train_validation(
        PrefixFailureDataset(), rounds=1, batch_size=1, updates_per_episode=1,
        seed=1, beta_soc=250.0,
    )
    counts = report["execution_counts"]
    assert counts["training_onboard_executed_transitions"] == 1
    assert counts["economic_q_optimizer_updates"] == 0
    row = report["training_rounds"][0]
    assert row["transitions"] == 0
    assert row["executed_onboard_transitions"] == 1
    assert row["soc_time_occupancy"]["executed_steps"] == 1
    assert row["onboard_soc_min_including_failed_prefix"] is not None
    assert report["test_payloads_opened"] == 0


def test_fixed_review_keeps_diagnostic_checkpoint_and_cost_categories(tmp_path, monkeypatch):
    import json
    from v4 import review

    class IncompleteDataset(FakeDataset):
        def load_validation(self):
            row = super().load_validation()[0]
            row.load_kw = (2000.0,)
            return (row,)

    dataset = IncompleteDataset()
    monkeypatch.setattr(review.FormalTrainingDataset, "open", lambda *roots: dataset)
    monkeypatch.setattr(review, "_default_data_root", lambda name: tmp_path / name)
    monkeypatch.setattr(review, "FIXED_CONFIGURATION", {
        "rounds": 1, "seed": 1, "batch_size": 1, "updates_per_episode": 1,
        "beta_soc": 250.0,
    })
    monkeypatch.setattr(review, "_trajectory_plot", lambda *args: None)
    monkeypatch.setattr(review, "_training_plot", lambda *args: None)
    assert review.main(["--output-dir", str(tmp_path / "output")]) == 0
    report = json.loads((tmp_path / "output" / "report.json").read_text())
    assert report["selection_eligible"] is False
    assert report["test_payloads_opened"] == 0
    assert report["dataset_manifests_unchanged"] is True
    assert (tmp_path / "output" / "diagnostic_final_agent.pt").exists()
    assert not (tmp_path / "output" / "selected_agent.pt").exists()
    train = report["train_greedy_evaluation"]
    assert sum(train["completed_observed_components_cny"].values()) == pytest.approx(
        train["completed_observed_cost_cny"]
    )
    assert train["completed_cost_cny"] == pytest.approx(
        train["completed_observed_cost_cny"] + train["completed_modeled_terminal_cost_cny"]
    )
