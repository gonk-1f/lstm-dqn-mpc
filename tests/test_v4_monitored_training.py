from types import SimpleNamespace

import torch

from test_v4_training import FakeDataset
from v4.dqn import DirectPowerDDQN


def test_economic_replay_insertions_count_new_entries_after_capacity_saturation():
    a = DirectPowerDDQN(seed=1, replay_capacity=2)
    for _ in range(5):
        a.remember((0.6,)*8, 0, -1, (0.6,)*8, done=True, next_feasible_actions=())
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


def test_round_monitoring_does_not_change_legacy_training_trajectory(tmp_path, monkeypatch):
    from v4.train import run_train_validation
    from v4.monitored_training import run_monitored_training

    kwargs = dict(rounds=3,seed=42,batch_size=1,updates_per_episode=16,beta_soc=250.0,epsilon_start=1.0,epsilon_end=.05)
    original_select = DirectPowerDDQN.select_power
    def isolate_reference_final_evaluation(self, state, feasible, *, epsilon=0.0):
        if epsilon != 0.0:
            return original_select(self, state, feasible, epsilon=epsilon)
        saved = self.random.getstate()
        try:
            return original_select(self, state, feasible, epsilon=epsilon)
        finally:
            self.random.setstate(saved)
    # Keep the legacy runner's greedy decisions isolated while comparing the
    # actual training streams; both runners now evaluate after every round.
    with monkeypatch.context() as patch:
        patch.setattr(DirectPowerDDQN, "select_power", isolate_reference_final_evaluation)
        old, _ = run_train_validation(FakeDataset(), **kwargs)
    new, report = run_monitored_training(FakeDataset(), output_dir=tmp_path, **kwargs)
    assert old.random.getstate() == new.random.getstate()
    assert old.outcome_random.getstate() == new.outcome_random.getstate()
    assert tuple(old.replay) == tuple(new.replay)
    for name in ("online", "target", "outcome_model"):
        assert all(torch.equal(value, getattr(new,name).state_dict()[key]) for key,value in getattr(old,name).state_dict().items())
    assert len(report["rounds"]) == 3


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
