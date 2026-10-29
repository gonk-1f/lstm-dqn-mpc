import random

import pytest


def test_iid_schedule_uses_dedicated_rng_and_restores_exactly():
    from v6_sb3.episode_schedule import EpisodeSchedule
    ids = tuple(f'train_{i}' for i in range(35))
    random.seed(123)
    global_state = random.getstate()
    schedule = EpisodeSchedule(ids, 500, seed=77)
    first = [schedule.draw() for _ in range(83)]
    restored = EpisodeSchedule.restore(schedule.snapshot())
    rest = [schedule.draw() for _ in range(417)]
    assert rest == [restored.draw() for _ in range(417)]
    expected_rng = random.Random(77)
    assert first + rest == [expected_rng.choice(ids) for _ in range(500)]
    assert sum(schedule.counts.values()) == 500
    assert random.getstate() == global_state
    with pytest.raises(StopIteration):
        schedule.draw()


def test_epsilon_counts_formal_episodes_and_holds_final_value():
    from v6_sb3.episode_schedule import epsilon_at_episode
    assert epsilon_at_episode(1, 500) == 1.0
    assert epsilon_at_episode(400, 500) == pytest.approx(.05)
    assert epsilon_at_episode(401, 500) == pytest.approx(.05)
    assert epsilon_at_episode(500, 500) == pytest.approx(.05)
