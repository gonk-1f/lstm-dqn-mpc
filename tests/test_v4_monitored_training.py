import json
from types import SimpleNamespace

import pytest
import torch

from test_v4_training import FakeDataset
from v4.dqn import DirectPowerDDQN


class UnknownPrefixDataset(FakeDataset):
    """A real observed prefix with SHORE rows and an unverified final boundary."""
    def load_train(self):
        from v2.data.supervisory_rules import OperatingMode
        modes = ("shore_charging", OperatingMode.ONBOARD, "onboard", "shore_charging",
                 *(("onboard",) * 50), "unknown")
        return (SimpleNamespace(sample_id="unknown_prefix", split="train",
            operating_mode=modes, load_kw=(0.,) * len(modes),
            battery_bus_kw=(0.,) * len(modes)),)


@pytest.mark.parametrize("mode,retained", [("observed_nstep", 44), ("discard", 0)])
def test_unknown_replay_retains_observed_windows_with_sources_and_actual_credit(tmp_path, monkeypatch,
                                                                            mode, retained):
    from v4 import monitored_training
    monkeypatch.setattr(monitored_training, "write_learning_curves", lambda *args: {})
    agent, report = monitored_training.run_monitored_training(UnknownPrefixDataset(),
        output_dir=tmp_path, rounds=1, seed=42, batch_size=1000, n_step=8,
        unknown_replay_mode=mode, episode_credit_scope="voyage")
    discarded = 52 - retained
    assert report["bootstrap"]["unknown_retained_economic_q_transitions"] == retained
    assert report["bootstrap"]["unknown_discarded_economic_q_transitions"] == discarded
    row = report["rounds"][0]
    assert row["unknown_retained_economic_q_transitions"] == retained
    assert row["unknown_discarded_economic_q_transitions"] == discarded
    assert row["unknown_retained_economic_q_transitions_cumulative"] == 2 * retained
    assert row["unknown_discarded_economic_q_transitions_cumulative"] == 2 * discarded
    assert report["execution_counts"]["unknown_retained_economic_q_transitions"] == 2 * retained
    assert report["execution_counts"]["unknown_discarded_economic_q_transitions"] == 2 * discarded
    assert row["completed_voyages_entering_economic_replay"] == 0
    assert agent.economic_replay_insertions == 2 * retained
    assert row["remaining_transition_credit"] == 2 * retained
    assert agent.economic_success_replay_insertions == agent.economic_failure_replay_insertions == 0
    assert agent.economic_unresolved_replay_insertions == 2 * retained
    assert agent.normal_terminal_replay_count == agent.failure_terminal_replay_count == 0
    assert report["economic_replay_outcome_counts"]["unresolved"] == 2 * retained
    assert report["economic_replay_outcome_counts"]["normal_terminals_in_pool"] == 0
    assert row["greedy_train"]["data_truncated_episodes"] == 1
    assert row["greedy_train"]["physical_failures"] == 0
    assert row["greedy_train"]["evaluable_completed"] == 0
    assert row["greedy_validation"]["evaluable_completed"] == 1
    assert row["validation_skip_reason"] is None and not row["qualified_checkpoint"]
    assert report["best_checkpoint"] is None and report["test_payloads_opened"] == 0
    expected_rows = (1, 2, *range(4, 54))[:retained]
    for round_index in (0, 1):
        items = [item for item in agent.replay if item.source_round == round_index]
        assert tuple(item.source_row_index for item in items) == expected_rows
        assert all(item.source_sample_id == "unknown_prefix" and not item.done
                   and item.experience_outcome == "unresolved" and item.bootstrap_steps == 8
                   for item in items)
    assert report["hyperparameters"]["unknown_replay_mode"] == mode
    assert report["hyperparameters"]["economic_replay_format_version"] == 2
    checkpoint = torch.load(tmp_path / "training_state_latest.pt", weights_only=False)
    assert checkpoint["agent"]["format_version"] == 2
    assert len(checkpoint["agent"]["replay"]) == 2 * retained


def test_monitored_unknown_replay_default_is_observed_nstep(tmp_path, monkeypatch):
    from v4 import monitored_training
    monkeypatch.setattr(monitored_training, "write_learning_curves", lambda *args: {})
    agent, report = monitored_training.run_monitored_training(UnknownPrefixDataset(),
        output_dir=tmp_path, rounds=1, batch_size=1000, n_step=8)
    assert report["hyperparameters"]["unknown_replay_mode"] == "observed_nstep"
    assert agent.economic_unresolved_replay_insertions == 88


def test_monitored_success_and_failure_replay_have_actual_source_rows(tmp_path, monkeypatch):
    from v4 import monitored_training
    monkeypatch.setattr(monitored_training, "write_learning_curves", lambda *args: {})
    class SourceDataset(FakeDataset):
        def load_train(self):
            complete = super().load_train()[0]
            complete.operating_mode = ("shore_charging", "onboard", "shore_charging")
            complete.load_kw = (0., 100., 0.)
            complete.battery_bus_kw = (0., 0., 0.)
            failed = SimpleNamespace(sample_id="failed_source", split="train",
                operating_mode=("shore_charging", *(("onboard",) * 13)),
                load_kw=(0., *((0.,) * 12), 3000.), battery_bus_kw=(0.,) * 14)
            return complete, failed
    agent, report = monitored_training.run_monitored_training(SourceDataset(), output_dir=tmp_path,
        rounds=1, n_step=8, batch_size=1000)
    for round_index in (0, 1):
        successful = [t for t in agent.replay if t.source_round == round_index
                      and t.source_sample_id == "train_1"]
        failed = [t for t in agent.replay if t.source_round == round_index
                  and t.source_sample_id == "failed_source"]
        assert len(successful) == 1 and successful[0].source_row_index == 1
        assert tuple(t.source_row_index for t in failed) == tuple(range(1, 13))
        assert failed[-1].done and failed[-1].experience_outcome == "failure"
    assert agent.economic_replay_insertions == 26
    assert report["rounds"][0]["remaining_transition_credit"] == 26


def test_invalid_unknown_replay_mode_is_rejected_before_dataset_loading(tmp_path):
    from v4.monitored_training import run_monitored_training
    class UnopenedDataset(FakeDataset):
        def load_train(self):
            pytest.fail("invalid replay semantics must be rejected before data loading")
    with pytest.raises(ValueError, match="unknown_replay_mode"):
        run_monitored_training(UnopenedDataset(), output_dir=tmp_path,
                               unknown_replay_mode="invented_successor")


def test_prior_replay_metadata_contract_cannot_resume_before_dataset_loading(tmp_path, monkeypatch):
    from v4 import monitored_training
    monkeypatch.setattr(monitored_training, "write_learning_curves", lambda *args: {})
    monitored_training.run_monitored_training(FakeDataset(), output_dir=tmp_path, rounds=1)
    path = tmp_path / "training_state_latest.pt"
    checkpoint = torch.load(path, weights_only=False)
    saved = checkpoint["metadata"]
    saved["hyperparameters"].pop("economic_replay_format_version")
    saved["hyperparameters"].pop("unknown_replay_mode")
    torch.save(checkpoint, path)
    (tmp_path / "run_metadata.json").write_text(json.dumps(saved), encoding="utf-8")
    monkeypatch.setattr(FakeDataset, "load_train",
                        lambda self: pytest.fail("old replay contract must fail before data loading"))
    with pytest.raises(ValueError, match="resume source, architecture, hyperparameters, experiment identity or data manifest mismatch"):
        monitored_training.run_monitored_training(FakeDataset(), output_dir=tmp_path,
            rounds=1, resume_from=path)


def test_economic_replay_insertions_count_new_entries_after_capacity_saturation():
    a = DirectPowerDDQN(seed=1, replay_capacity=2)
    for _ in range(5):
        a.remember((0.6,)*7, 0, -1, (0.6,)*7, done=True, next_feasible_actions=())
    assert len(a.replay) == 2
    assert a.economic_replay_insertions == 5


def test_greedy_evaluation_restores_training_random_stream_and_never_learns():
    from v3.control import EconomicMPC
    from v4.monitored_training import greedy_evaluate

    a = DirectPowerDDQN(seed=1)
    before = a.random.getstate()
    summary = greedy_evaluate(FakeDataset().load_train(), a, EconomicMPC(nominal_cost_cny=1.0), 250.0)
    assert summary["completed"] == 1
    assert a.random.getstate() == before
    assert len(a.replay) == 0 and a.economic_optimizer_updates == 0
    assert "fc_zero_fraction" in summary and "onboard_soc_mean" in summary


def test_failed_greedy_train_still_runs_validation_and_cannot_save_checkpoint(tmp_path):
    from v4.monitored_training import run_monitored_training

    class ImpossibleDataset(FakeDataset):
        def load_train(self):
            return (SimpleNamespace(sample_id="bad", split="train", operating_mode=("onboard",), load_kw=(2000.0,), battery_bus_kw=(0.0,)),)

    _, report = run_monitored_training(ImpossibleDataset(), output_dir=tmp_path, rounds=2, beta_soc=250, seed=1, batch_size=1)
    assert all(row["greedy_validation"]["evaluable_completed"] == 1
               and row["validation_skip_reason"] is None for row in report["rounds"])
    assert report["best_checkpoint"] is None
    assert not (tmp_path / "best_agent.pt").exists()
    assert report["test_payloads_opened"] == 0


def test_unknown_prefix_is_observed_but_never_used_as_a_terminal_q_target():
    from v3.control import EconomicMPC
    from v4.control import ReplayExecutionError, replay_episode
    from v4.experiment_schedule import fully_completed
    from v4.monitored_training import greedy_evaluate

    accountant = EconomicMPC(nominal_cost_cny=1.0)
    complete = SimpleNamespace(sample_id="complete", split="train",
        operating_mode=("onboard",), load_kw=(100.0,), battery_bus_kw=(0.0,))
    truncated = SimpleNamespace(sample_id="truncated", split="train",
        operating_mode=("onboard", "unknown"), load_kw=(100.0, 0.0),
        battery_bus_kw=(0.0, 0.0))
    agent = DirectPowerDDQN(seed=42, n_step=8)
    summary = greedy_evaluate((complete, truncated), agent, accountant, 0.0)
    assert summary["evaluable_completed"] == summary["evaluable_episodes"] == 1
    assert summary["data_truncated_episodes"] == summary["unknown_episodes"] == 1
    assert summary["unknown_prefix_executed_onboard_transitions"] == 1
    assert summary["cost_cny"] == summary["completed_cost_cny"]
    assert fully_completed(summary)
    assert not agent.replay
    failed_before_unknown = SimpleNamespace(sample_id="failed_prefix", split="train",
        operating_mode=("onboard", "unknown"), load_kw=(3000.0, 0.0),
        battery_bus_kw=(0.0, 0.0))
    failed_summary = greedy_evaluate((complete, failed_before_unknown), agent, accountant, 0.0)
    assert failed_summary["unknown_prefix_physical_failures"] == 1
    assert failed_summary["data_truncated_episodes"] == 0
    assert not fully_completed(failed_summary)
    try:
        replay_episode(truncated, lambda _state, feasible: min(feasible), accountant=accountant)
    except ReplayExecutionError as error:
        assert error.failure_kind == "data_truncation"
        assert error.executed_transitions[-1].terminal_reason == "data_truncation"
        assert not error.executed_transitions[-1].done
        assert error.executed_transitions[-1].modeled_terminal_ledger is None
        assert error.executed_transitions[-1].failure_penalty_equivalent_cny == 0
        assert agent.remember_completed_prefix(error.executed_transitions) == 0
        assert not agent.replay
    else:
        raise AssertionError("UNKNOWN must truncate the observed trajectory")


def test_monitored_runs_are_reproducible_without_legacy_runner_equivalence(tmp_path):
    from v4.monitored_training import run_monitored_training
    kwargs = dict(rounds=2, seed=42, batch_size=1, beta_soc=200.0)
    old, reference = run_monitored_training(FakeDataset(), output_dir=tmp_path/'first', **kwargs)
    new, report = run_monitored_training(FakeDataset(), output_dir=tmp_path/'second', **kwargs)
    assert old.random.getstate() == new.random.getstate()
    assert tuple(old.replay) == tuple(new.replay)
    for name in ('online', 'target'):
        assert all(torch.equal(value, getattr(new,name).state_dict()[key])
                   for key,value in getattr(old,name).state_dict().items())
    assert reference['rounds'] == report['rounds']


def test_optimizer_target_mode_has_no_bootstrap_or_round_extra_sync(tmp_path):
    from v4.monitored_training import run_monitored_training

    class FiftySteps(FakeDataset):
        def load_train(self):
            return (SimpleNamespace(sample_id="fifty",split="train",operating_mode=("onboard",)*50,
                                    load_kw=(100.0,)*50,battery_bus_kw=(0.0,)*50),)
    for cadence,stride in (("replay16",16),("replay8",8)):
        agent,report=run_monitored_training(FiftySteps(),output_dir=tmp_path/cadence,rounds=1,
                                           beta_soc=500,seed=42,batch_size=64,cadence=cadence,
                                           target_mode="optimizer",target_interval=4)
        assert agent.economic_replay_insertions==100
        assert agent.economic_optimizer_updates==100//stride
        assert agent.target_sync_calls==1+agent.economic_optimizer_updates//4
        assert report["execution_counts"]["remaining_transition_credit"]==100%stride


def test_stage1_gate_selects_lowest_qualified_beta_and_worker_verifies_best(tmp_path,monkeypatch):
    from v4 import staged_study

    assert staged_study.select_beta({"250":{"best_checkpoint":None}}) is None
    assert staged_study.select_beta({
        "250":{"best_checkpoint":None},
        "500":{"best_checkpoint":{"validation_comparable_cost_cny":100}},
        "1000":{"best_checkpoint":{"validation_comparable_cost_cny":80}},
    })=="1000"

    class FormalSizedFake(FakeDataset):
        def load_train(self):
            return tuple(SimpleNamespace(sample_id=f"t_{i}",split="train",operating_mode=("onboard",),load_kw=(100.,),battery_bus_kw=(0.,)) for i in range(30))
        def load_validation(self):
            return tuple(SimpleNamespace(sample_id=f"v_{i}",split="validation",operating_mode=("onboard",),load_kw=(80.,),battery_bus_kw=(0.,)) for i in range(8))
    dataset=FormalSizedFake()
    monkeypatch.setattr(staged_study.FormalTrainingDataset,"open",lambda *roots:dataset)
    monkeypatch.setattr(staged_study,"_default_data_root",lambda name:tmp_path/name)
    monkeypatch.setattr(staged_study,"_trajectory_plot",lambda *args:None)
    assert staged_study.main(["--worker","--output-dir",str(tmp_path/"worker"),"--beta","500","--rounds","1"])==0
    import json
    report=json.loads((tmp_path/"worker"/"report.json").read_text())
    assert report["best_checkpoint_replay_verified"]
    assert report["dataset_manifests_unchanged"] and report["test_payloads_opened"]==0
    assert (tmp_path/"worker"/"best_agent.pt").exists()
