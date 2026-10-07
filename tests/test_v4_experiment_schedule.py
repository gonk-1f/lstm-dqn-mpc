import pytest


class FakeAgent:
    def __init__(self, replay_size=100):
        self.replay = [None] * replay_size
        self.economic_optimizer_updates = 0
        self.target_sync_calls = 1
        self.sync_at = []

    def learn(self, *, batch_size):
        if len(self.replay) < batch_size:
            return None
        self.economic_optimizer_updates += 1
        return 1.0

    def sync_target(self):
        self.sync_at.append(self.economic_optimizer_updates)
        self.target_sync_calls += 1


def test_transition_credit_is_independent_of_episode_partition_and_retains_remainder():
    from v4.experiment_schedule import EconomicUpdateSchedule

    for parts in ((7, 9, 17), (33,)):
        schedule = EconomicUpdateSchedule("replay16", target_mode="optimizer", target_interval=1000)
        agent = FakeAgent()
        for n in parts:
            schedule.grant(insertions=n, completed_episodes=1)
            schedule.consume(agent, batch_size=64)
        assert agent.economic_optimizer_updates == 2
        assert schedule.remaining_transition_credit == 1


def test_credit_survives_warmup_and_target_sync_counts_only_actual_updates():
    from v4.experiment_schedule import EconomicUpdateSchedule

    schedule = EconomicUpdateSchedule("replay8", target_mode="optimizer", target_interval=1000)
    agent = FakeAgent(replay_size=2)
    schedule.grant(insertions=8008, completed_episodes=0)
    assert schedule.consume(agent, batch_size=64) == []
    assert schedule.remaining_transition_credit == 8008
    agent.replay.extend([None] * 62)
    assert len(schedule.consume(agent, batch_size=64)) == 1001
    assert agent.sync_at == [1000]
    assert schedule.remaining_transition_credit == 0


def test_episode_baseline_has_no_budget_for_failed_episode_valid_prefix():
    from v4.experiment_schedule import EconomicUpdateSchedule

    schedule = EconomicUpdateSchedule("episode16", target_mode="round")
    agent = FakeAgent()
    schedule.grant(insertions=300, completed_episodes=0)
    assert schedule.consume(agent, batch_size=64) == []
    schedule.grant(insertions=1000, completed_episodes=1)
    assert len(schedule.consume(agent, batch_size=64)) == 16
    assert agent.sync_at == []


def test_best_checkpoint_rejects_failures_and_uses_only_comparable_economic_cost():
    from v4.experiment_schedule import BestCheckpoint

    best = BestCheckpoint()
    def summary(done, requested, cost):
        return {"completed": done, "episodes": requested, "failed": [] if done == requested else ["failed"], "cost_cny": cost}
    assert not best.consider(1, summary(29,30,1), summary(8,8,1))
    assert not best.consider(2, summary(30,30,1), summary(7,8,1))
    assert best.consider(3, summary(30,30,1), summary(8,8,100))
    assert best.consider(5, summary(30,30,1), summary(8,8,90))
    assert not best.consider(40, summary(10,30,1), None)
    assert best.round == 5 and best.cost_cny == 90
