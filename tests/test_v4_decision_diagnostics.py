"""Diagnostic artifacts must observe, never alter, Scheme A training."""

import json
from types import SimpleNamespace

import pytest
import torch

from dqn.networks.mlp_qnet import MLPQNetwork
from v3.control import AccountState, EconomicMPC
from v4.control import replay_episode
from v4.decision_diagnostics import actual_greedy_q_records, save_diagnostic_snapshot
from v4.dqn import DirectPowerDDQN
from v4.monitored_training import run_monitored_training


class DiagnosticDataset:
    opened_test_payloads = 0

    def load_train(self):
        return tuple(SimpleNamespace(
            sample_id=f"zero_boundary_{index:03d}", split="train",
            operating_mode=("onboard",) * 12,
            load_kw=(100.,) * 12, battery_bus_kw=(0.,) * 12,
        ) for index in (15, 17, 44))

    def load_validation(self):
        return (SimpleNamespace(sample_id="synthetic_validation", split="validation",
            operating_mode=("onboard",) * 4, load_kw=(100.,) * 4,
            battery_bus_kw=(0.,) * 4),)

    def load_test(self):
        raise AssertionError("Test must remain sealed")


def _run(dataset, output, diagnostic_rounds):
    return run_monitored_training(dataset, output_dir=output, rounds=1, beta_soc=500.,
        cadence="replay32", target_mode="optimizer", target_interval=500,
        seed=42, batch_size=8, epsilon_start=1., epsilon_end=.05,
        reward_scale=.001, failure_terminal_quota=2, n_step=1,
        redistribute_battery_energy=True, episode_credit_scope="voyage",
        required_split_sizes=(3, 1), capture_trajectories=False,
        manifest_sha256={"synthetic_train_manifest": "fixed-hash"},
        diagnostic_rounds=diagnostic_rounds)


def test_diagnostics_do_not_change_actions_updates_networks_or_selection(tmp_path):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        dataset = DiagnosticDataset()
        plain, plain_report = _run(dataset, tmp_path / "plain", ())
        plain_torch_rng = torch.random.get_rng_state().clone()
        observed, report = _run(dataset, tmp_path / "observed", (1,))
        observed_torch_rng = torch.random.get_rng_state().clone()
    finally:
        torch.set_num_threads(old_threads)
    assert dataset.opened_test_payloads == 0
    assert tuple(plain.replay) == tuple(observed.replay)
    assert plain.random.getstate() == observed.random.getstate()
    assert plain.outcome_random.getstate() == observed.outcome_random.getstate()
    assert torch.equal(plain_torch_rng, observed_torch_rng)
    assert plain.economic_optimizer_updates == observed.economic_optimizer_updates > 0
    assert plain.target_sync_calls == observed.target_sync_calls
    assert plain.outcome_optimizer_updates == observed.outcome_optimizer_updates
    for name in ("online", "target", "outcome_model"):
        assert all(torch.equal(value, getattr(observed, name).state_dict()[key])
                   for key, value in getattr(plain, name).state_dict().items())
    assert plain_report["rounds"][0]["greedy_train"] == report["rounds"][0]["greedy_train"]
    assert plain_report["rounds"][0]["greedy_validation"] == report["rounds"][0]["greedy_validation"]
    assert plain_report["best_checkpoint"] == report["best_checkpoint"]
    assert report["test_payloads_opened"] == 0
    assert not (tmp_path / "plain" / "diagnostic_checkpoints").exists()
    assert (tmp_path / "observed" / "best_agent.pt").exists() == (
        tmp_path / "plain" / "best_agent.pt").exists()

    checkpoint_path = tmp_path / "observed" / "diagnostic_checkpoints" / "round_001.pt"
    assert checkpoint_path.is_file()
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    assert checkpoint["purpose"] == "diagnostic_only"
    assert checkpoint["round"] == 1
    assert checkpoint["hyperparameters"]["failure_terminal_quota"] == 2
    assert checkpoint["hyperparameters"]["n_step"] == 1
    assert checkpoint["manifest_sha256"] == {"synthetic_train_manifest": "fixed-hash"}
    assert checkpoint["economic_optimizer_updates"] == observed.economic_optimizer_updates
    assert all(torch.equal(value, checkpoint["online_state"][key])
               for key, value in observed.online.state_dict().items())
    assert all(torch.equal(value, checkpoint["target_state"][key])
               for key, value in observed.target.state_dict().items())
    records = json.loads((tmp_path / "observed" / "diagnostic_checkpoints" /
                          "round_001_decisions.json").read_text(encoding="utf-8"))
    assert records["round"] == 1 and len(records["decisions"]) <= 48
    assert {row["sample_id"] for row in records["decisions"]} == {
        "zero_boundary_015", "zero_boundary_017", "zero_boundary_044"}
    restored = MLPQNetwork(8, 61, (128, 64))
    restored.load_state_dict(checkpoint["online_state"])
    with torch.inference_mode():
        for row in records["decisions"]:
            q = restored(torch.tensor([row["state"]], dtype=torch.float32))[0].tolist()
            assert q == pytest.approx(row["q_values"])
            assert len(q) == 61
            assert row["selected_action_kw"] in row["candidate_actions_kw"]
            assert set(row["candidate_actions_kw"]).issubset(row["physical_actions_kw"])
            assert row["selected_action_legal_rank"] == 1
            assert row["legal_action_order_kw"][0] == max(
                row["candidate_actions_kw"], key=lambda action: (q[action // 10], -action))


def test_actual_state_probe_prioritizes_forced_stop_without_rng_or_replay():
    accountant = EconomicMPC(nominal_cost_cny=1.)
    agent = DirectPowerDDQN(seed=42)
    rng = (agent.random.getstate(), agent.outcome_random.getstate(),
           torch.random.get_rng_state().clone())
    data = SimpleNamespace(sample_id="zero_boundary_015", split="train",
        operating_mode=("onboard",), load_kw=(0.,), battery_bus_kw=(0.,))
    replay = replay_episode(data, lambda _state, mask: min(mask),
        accountant=accountant, initial_state=AccountState(soc=.79995, previous_fc_kw=100.))
    rows = actual_greedy_q_records(data.sample_id, replay.transitions, agent)
    assert len(rows) == 1
    assert "forced_stop" in rows[0]["priority_tags"]
    assert rows[0]["physical_actions_kw"] == rows[0]["candidate_actions_kw"] == [0]
    assert rows[0]["selected_action_kw"] == 0
    assert agent.random.getstate() == rng[0]
    assert agent.outcome_random.getstate() == rng[1]
    assert torch.equal(torch.random.get_rng_state(), rng[2])
    assert not agent.replay and agent.economic_optimizer_updates == 0


def test_actual_state_probe_tags_low_soc_low_fc_run_and_high_soc():
    accountant = EconomicMPC(nominal_cost_cny=1.)
    agent = DirectPowerDDQN(seed=42)
    low = SimpleNamespace(sample_id="zero_boundary_017", split="train",
        operating_mode=("onboard",) * 5, load_kw=(700.,) * 5,
        battery_bus_kw=(0.,) * 5)
    low_run = replay_episode(low, lambda _state, mask: min(mask),
        accountant=accountant, initial_state=AccountState(soc=.3))
    low_records = actual_greedy_q_records(low.sample_id, low_run.transitions, agent)
    assert any("low_soc_high_load" in row["priority_tags"] for row in low_records)
    assert any("consecutive_low_fc" in row["priority_tags"] for row in low_records)

    high = SimpleNamespace(sample_id="zero_boundary_044", split="train",
        operating_mode=("onboard",), load_kw=(100.,), battery_bus_kw=(0.,))
    high_run = replay_episode(high, lambda _state, mask: min(mask),
        accountant=accountant, initial_state=AccountState(soc=.7999))
    high_records = actual_greedy_q_records(high.sample_id, high_run.transitions, agent)
    assert "near_soc_ceiling" in high_records[0]["priority_tags"]


def test_formal_entry_passes_explicit_diagnostic_rounds_without_running_training(tmp_path, monkeypatch):
    from v4 import feedback_study
    dataset = DiagnosticDataset()
    captured = {}
    monkeypatch.setattr(feedback_study.FormalTrainingDataset, "open", lambda *_roots: dataset)
    monkeypatch.setattr(feedback_study, "_default_data_root", lambda name: tmp_path / name)

    def fake_run(_dataset, **kwargs):
        captured.update(kwargs)
        kwargs["output_dir"].mkdir(parents=True)
        return None, {"rounds": [], "best_checkpoint": None}

    monkeypatch.setattr(feedback_study, "run_monitored_training", fake_run)
    assert feedback_study.main(["--output-dir", str(tmp_path / "diagnostic_cli"),
        "--rounds", "40", "--diagnostic-rounds", "1,10,20,30,40",
        "--reward-scale", "0.001", "--failure-terminal-quota", "2",
        "--n-step", "1", "--cadence", "replay32", "--target-interval", "500"]) == 0
    assert captured["diagnostic_rounds"] == (1, 10, 20, 30, 40)
    assert captured["failure_terminal_quota"] == 2
    assert captured["reward_scale"] == .001
    assert captured["required_split_sizes"] == (30, 8)
    assert dataset.opened_test_payloads == 0


def test_diagnostic_snapshot_refuses_open_test_and_unqualified_round_stays_unselected(tmp_path):
    agent = DirectPowerDDQN(seed=1)
    metadata = {"source_commit": "synthetic", "source_worktree_dirty": False,
        "hyperparameters": {"n_step": 1}, "manifest_sha256": {"train": "synthetic"},
        "dataset_roots": []}
    with pytest.raises(RuntimeError, match="Test"):
        save_diagnostic_snapshot(tmp_path, round_index=1, agent=agent,
            decisions=[], run_metadata=metadata, training_environment_transitions=0,
            test_payloads_opened=1)
    assert not (tmp_path / "diagnostic_checkpoints").exists()

    class Impossible(DiagnosticDataset):
        def load_train(self):
            return (SimpleNamespace(sample_id="zero_boundary_015", split="train",
                operating_mode=("onboard",), load_kw=(2000.,),
                battery_bus_kw=(0.,)),)

    _, report = run_monitored_training(Impossible(), output_dir=tmp_path / "failed",
        rounds=1, beta_soc=500., batch_size=8, required_split_sizes=(1, 1),
        diagnostic_rounds=(1,))
    assert report["best_checkpoint"] is None
    assert report["rounds"][0]["greedy_validation"] is None
    assert not (tmp_path / "failed" / "best_agent.pt").exists()
    assert (tmp_path / "failed" / "diagnostic_checkpoints" / "round_001.pt").exists()
    decisions = json.loads((tmp_path / "failed" / "diagnostic_checkpoints" /
                            "round_001_decisions.json").read_text(encoding="utf-8"))
    assert decisions["decisions"] == []
    assert report["test_payloads_opened"] == 0
