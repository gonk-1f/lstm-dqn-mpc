"""Pure Python PER checks: these tests never import the training stack."""
from collections import Counter
from copy import deepcopy
from functools import lru_cache
import importlib.util
from math import isfinite
from pathlib import Path
from random import Random

import pytest


@lru_cache(maxsize=1)
def priority_class():
    path = Path(__file__).resolve().parents[1] / 'src/v4/prioritized_replay.py'
    assert path.is_file(), 'the PriorityIndex implementation is missing'
    spec = importlib.util.spec_from_file_location('pure_python_per', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PriorityIndex


def index(capacity=8, **kwargs):
    return priority_class()(capacity, **kwargs)


def test_rare_large_error_has_more_exposure_with_replacement():
    replay = index(20)
    for replay_id in range(20):
        replay.add(replay_id, 20 if replay_id == 19 else 1)
    ids, probabilities = replay.sample(Random(73), 20000)
    counts = Counter(ids)
    assert counts[19] > 4000
    assert counts[19] > 4 * max(counts[i] for i in range(19))
    assert len(ids) == len(probabilities) == 20000
    assert len(set(ids)) < len(ids)


def test_probabilities_follow_positive_epsilon_adjusted_mass():
    replay = index(3, alpha=.6, epsilon=.01)
    raw = (0, 1, 9)
    for replay_id, priority in enumerate(raw):
        replay.add(replay_id, priority)
    masses = tuple((value + .01) ** .6 for value in raw)
    ids, probabilities = replay.sample(Random(9), 1000)
    assert set(ids) == {0, 1, 2}
    for replay_id, probability in zip(ids, probabilities):
        assert isfinite(probability) and probability > 0
        assert probability == pytest.approx(masses[replay_id] / sum(masses))
    weights = replay.importance_weights(probabilities, .8)
    assert all(isfinite(weight) and 0 < weight <= 1 for weight in weights)


def test_beta_one_weighted_expectation_recovers_uniform_objective():
    replay = index(3, alpha=1, epsilon=1)
    for replay_id, priority in enumerate((0, 1, 3)):
        replay.add(replay_id, priority)
    probabilities = (1 / 7, 2 / 7, 4 / 7)
    values = (2, -5, 13)
    normalizer = replay.importance_normalizer(1)
    weights = replay.importance_weights(probabilities, 1)
    assert normalizer == pytest.approx(7 / 3)
    assert sum(p * w * v for p, w, v in zip(probabilities, weights, values)) == pytest.approx(
        sum(values) / len(values) / normalizer)
    assert replay.importance_weights(probabilities, 0) == (1, 1, 1)


def test_normalization_uses_global_minimum_even_when_batch_misses_it():
    replay = index(3, alpha=1, epsilon=1)
    for replay_id, priority in enumerate((0, 9, 99)):
        replay.add(replay_id, priority)
    assert replay.importance_weights((100 / 111,), 1) == pytest.approx((.01,))
    replay.remove(0)
    assert replay.importance_normalizer(1) == pytest.approx(110 / 20)


def test_fifo_slot_reuse_has_no_stale_ids_or_priority():
    replay = index(3, alpha=1, epsilon=1)
    replay.add(0, 100)
    replay.add(1, 2)
    replay.add(2, 3)
    assert len(replay) == 3 and 0 in replay
    with pytest.raises(ValueError, match='slot'):
        replay.add(3, 8)
    assert replay.remove(0) is True
    assert replay.remove(0) is False
    replay.add(3, 0)
    assert 0 not in replay and 3 in replay and len(replay) == 3
    with pytest.raises(KeyError):
        replay.update(0, 4)
    ids, probabilities = replay.sample(Random(52), 1000)
    assert set(ids) == {1, 2, 3}
    expected = {1: 3 / 8, 2: 4 / 8, 3: 1 / 8}
    assert all(p == pytest.approx(expected[i]) for i, p in zip(ids, probabilities))


def test_default_insertion_uses_historical_maximum_after_removal():
    replay = index(4)
    replay.add(0)
    replay.update(0, 17)
    replay.remove(0)
    replay.add(4)
    replay.update(4, 0)
    replay.add(1)
    state = replay.state_dict()
    assert state['max_priority'] == 17
    assert dict(state['entries']) == {4: 0, 1: 17}
    with pytest.raises(ValueError, match='already'):
        replay.add(1, 9)


def test_alpha_zero_is_uniform_despite_errors_and_inactive_slots():
    replay = index(8, alpha=0)
    for replay_id, priority in ((0, 0), (2, 1), (7, 1000000)):
        replay.add(replay_id, priority)
    ids, probabilities = replay.sample(Random(11), 6000)
    counts = Counter(ids)
    assert set(ids) == {0, 2, 7}
    assert all(abs(count - 2000) < 150 for count in counts.values())
    assert probabilities == pytest.approx((1 / 3,) * 6000)
    assert replay.importance_normalizer(1) == 1


def test_seed_reproducibility_and_identical_checkpoint_sampling():
    replay = index(8, epsilon=.001)
    for replay_id in (8, 1, 2, 7):
        replay.add(replay_id, replay_id)
    replay.update(2, 100)
    replay.remove(2)
    replay.add(10, 3)
    assert replay.sample(Random(103), 500) == replay.sample(Random(103), 500)
    restored = index(8, epsilon=.001)
    restored.load_state_dict(deepcopy(replay.state_dict()))
    assert restored.state_dict() == replay.state_dict()
    assert restored.sample(Random(103), 500) == replay.sample(Random(103), 500)
    assert restored.importance_normalizer(.7) == replay.importance_normalizer(.7)
    restored.add(3)
    assert dict(restored.state_dict()['entries'])[3] == 100


def test_empty_index_zero_count_and_remove_are_rng_free():
    replay = index()
    rng = Random(8)
    state = rng.getstate()
    assert replay.sample(rng, 0) == ((), ())
    assert rng.getstate() == state
    assert len(replay) == 0 and 0 not in replay
    assert replay.importance_normalizer(1) == 1
    assert replay.importance_weights((), .5) == ()
    with pytest.raises(ValueError, match='empty'):
        replay.sample(rng, 1)


@pytest.mark.parametrize('kwargs', (
    {'capacity': 0}, {'capacity': -1}, {'capacity': 2.5}, {'capacity': True},
    {'capacity': 3, 'alpha': -.1}, {'capacity': 3, 'alpha': 1.1},
    {'capacity': 3, 'alpha': float('nan')},
    {'capacity': 3, 'epsilon': 0}, {'capacity': 3, 'epsilon': -1},
    {'capacity': 3, 'epsilon': float('inf')},
))
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        priority_class()(**kwargs)


@pytest.mark.parametrize('replay_id', (-1, 1.5, True, '1'))
def test_invalid_ids_are_rejected(replay_id):
    replay = index()
    for operation in (replay.add, replay.remove, lambda value: replay.update(value, 1)):
        with pytest.raises(ValueError):
            operation(replay_id)


@pytest.mark.parametrize('priority', (-1, float('inf'), float('nan')))
def test_invalid_raw_errors_are_rejected_without_mutation(priority):
    replay = index()
    replay.add(0, 1)
    state = replay.state_dict()
    with pytest.raises(ValueError):
        replay.add(1, priority)
    with pytest.raises(ValueError):
        replay.update(0, priority)
    assert replay.state_dict() == state


@pytest.mark.parametrize('beta', (-.1, 1.1, float('nan'), float('inf')))
def test_invalid_beta_is_rejected(beta):
    replay = index()
    replay.add(0)
    with pytest.raises(ValueError):
        replay.importance_normalizer(beta)
    with pytest.raises(ValueError):
        replay.importance_weights((1,), beta)


@pytest.mark.parametrize('count', (-1, 1.5, True))
def test_invalid_sample_count_is_rejected(count):
    with pytest.raises(ValueError):
        index().sample(Random(1), count)


@pytest.mark.parametrize('probability', (0, -1, 1.1, float('inf'), float('nan')))
def test_invalid_sample_probability_is_rejected(probability):
    replay = index()
    replay.add(0)
    with pytest.raises(ValueError):
        replay.importance_weights((probability,), .5)


@pytest.mark.parametrize('change', (
    {'capacity': 9}, {'alpha': .7}, {'epsilon': .001},
    {'entries': ((0, 1), (0, 2))}, {'entries': ((0, 1), (8, 2))},
    {'entries': ((-1, 1),)}, {'entries': ((0, -1),)},
    {'entries': ((0, float('nan')),)}, {'entries': ((0, None),)}, {'entries': ((True, 1),)},
    {'max_priority': .5}, {'max_priority': float('inf')},
))
def test_malformed_checkpoint_is_rejected_atomically(change):
    replay = index()
    replay.add(0, 1)
    before = deepcopy(replay.state_dict())
    malformed = {**before, **change}
    with pytest.raises(ValueError):
        replay.load_state_dict(malformed)
    assert replay.state_dict() == before
