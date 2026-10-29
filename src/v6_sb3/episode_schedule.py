"""Independent replacement sampling and episode-count exploration."""
from math import ceil
import random


class EpisodeSchedule:
    def __init__(self, sample_ids, episodes=500, *, seed=42):
        self.sample_ids = tuple(sample_ids)
        if not self.sample_ids or len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError('schedule needs distinct Train sample IDs')
        if episodes < 1:
            raise ValueError('episodes must be positive')
        self.episodes = episodes
        self.rng = random.Random(seed)
        self.order = []
        self.counts = dict.fromkeys(self.sample_ids, 0)

    def draw(self):
        if len(self.order) >= self.episodes:
            raise StopIteration
        sample_id = self.rng.choice(self.sample_ids)
        self.order.append(sample_id)
        self.counts[sample_id] += 1
        return sample_id

    def snapshot(self):
        return dict(sample_ids=self.sample_ids, episodes=self.episodes,
            order=list(self.order), counts=dict(self.counts), rng=self.rng.getstate())

    @classmethod
    def restore(cls, state):
        schedule = cls(state['sample_ids'], state['episodes'])
        schedule.order = list(state['order'])
        schedule.counts = dict(state['counts'])
        schedule.rng.setstate(state['rng'])
        return schedule


def epsilon_at_episode(index, episodes=500, *, initial=1., final=.05, decay_fraction=.8):
    """One-based: episode 1 starts at 1; episode ceil(.8*N) reaches .05."""
    if not 1 <= index <= episodes:
        raise ValueError('formal episode index is outside the budget')
    if not 0 < decay_fraction <= 1 or not 0 <= final <= initial <= 1:
        raise ValueError('invalid epsilon schedule')
    denominator = max(1, ceil(episodes*decay_fraction)-1)
    return initial+(final-initial)*min((index-1)/denominator, 1.)
