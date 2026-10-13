"""Proportional prioritized replay over externally owned, stable replay IDs."""
from array import array
from math import isfinite, nextafter


class PriorityIndex:
    """Sum/min trees with replacement; raw priorities are absolute TD errors.

    Remove an evicted ID before reusing its ``id % capacity`` slot. New entries
    inherit the historical maximum (initially 1), including after an eviction.
    The supplied RNG is used only by sample; checkpoint restoration is RNG free.
    """

    def __init__(self, capacity, alpha=.6, epsilon=1e-6):
        self.capacity = self._integer(capacity, 'capacity', minimum=1)
        self.alpha = self._number(alpha, 'alpha', maximum=1)
        self.epsilon = self._number(epsilon, 'epsilon')
        if self.epsilon == 0:
            raise ValueError('epsilon must be positive')
        self._base = 1 << (self.capacity - 1).bit_length()
        self._sum = array('d', [0.]) * (2 * self._base)
        self._min = array('d', [float('inf')]) * (2 * self._base)
        self._raw = array('d', [0.]) * self.capacity
        self._ids = [None] * self.capacity
        self._size = 0
        self._max_priority = 1.

    @staticmethod
    def _integer(value, name, minimum=0):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
        return value

    @staticmethod
    def _number(value, name, maximum=None):
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f'{name} must be finite and nonnegative') from error
        if not isfinite(value) or value < 0 or (maximum is not None and value > maximum):
            raise ValueError(f'{name} must be finite and in the allowed range')
        return value

    def __len__(self):
        return self._size

    def __contains__(self, replay_id):
        return (isinstance(replay_id, int) and not isinstance(replay_id, bool)
                and replay_id >= 0 and self._ids[replay_id % self.capacity] == replay_id)

    def contains(self, replay_id):
        return replay_id in self

    def _set_mass(self, slot, mass):
        node = self._base + slot
        self._sum[node] = mass
        self._min[node] = mass if mass > 0 else float('inf')
        while node > 1:
            node //= 2
            self._sum[node] = self._sum[2 * node] + self._sum[2 * node + 1]
            self._min[node] = min(self._min[2 * node], self._min[2 * node + 1])

    def _write_priority(self, slot, raw):
        mass = (raw + self.epsilon) ** self.alpha
        total = self._sum[1] - self._sum[self._base + slot] + mass
        if not isfinite(mass) or not isfinite(total) or mass <= 0:
            raise ValueError('priority mass and total must be finite and positive')
        self._raw[slot] = raw
        self._set_mass(slot, mass)
        self._max_priority = max(self._max_priority, raw)

    def add(self, replay_id, priority=None):
        replay_id = self._integer(replay_id, 'replay_id')
        slot = replay_id % self.capacity
        if self._ids[slot] is not None:
            message = 'replay_id is already active' if replay_id in self else 'replay slot collision'
            raise ValueError(message)
        raw = self._max_priority if priority is None else self._number(priority, 'priority')
        self._write_priority(slot, raw)
        self._ids[slot] = replay_id
        self._size += 1

    def remove(self, replay_id):
        replay_id = self._integer(replay_id, 'replay_id')
        if replay_id not in self:
            return False
        slot = replay_id % self.capacity
        self._ids[slot] = None
        self._raw[slot] = 0.
        self._set_mass(slot, 0.)
        self._size -= 1
        return True

    def update(self, replay_id, absolute_td_error):
        replay_id = self._integer(replay_id, 'replay_id')
        if replay_id not in self:
            raise KeyError(replay_id)
        self._write_priority(replay_id % self.capacity, self._number(absolute_td_error, 'TD error'))

    def sample(self, rng, count):
        count = self._integer(count, 'count')
        if count and not self._size:
            raise ValueError('cannot sample an empty priority index')
        ids, probabilities = [], []
        total = self._sum[1]
        for _ in range(count):
            draw = min(rng.random() * total, nextafter(total, 0.))
            node = 1
            while node < self._base:
                node *= 2
                if draw >= self._sum[node]:
                    draw -= self._sum[node]
                    node += 1
            ids.append(self._ids[node - self._base])
            probabilities.append(self._sum[node] / total)
        return tuple(ids), tuple(probabilities)

    def importance_normalizer(self, beta):
        beta = self._number(beta, 'beta', maximum=1)
        if not self._size:
            return 1.
        return max(1., (self._size * (self._min[1] / self._sum[1])) ** -beta)

    def importance_weights(self, probabilities, beta):
        beta = self._number(beta, 'beta', maximum=1)
        normalizer = self.importance_normalizer(beta)
        weights = []
        for probability in probabilities:
            probability = self._number(probability, 'probability', maximum=1)
            if probability == 0 or not self._size:
                raise ValueError('probability requires an active entry and must be positive')
            weights.append((self._size * probability) ** -beta / normalizer)
        return tuple(weights)

    def state_dict(self):
        """Export only configuration, live raw priorities and historical maximum."""
        return dict(capacity=self.capacity, alpha=self.alpha, epsilon=self.epsilon,
                    max_priority=self._max_priority,
                    entries=tuple((replay_id, self._raw[slot])
                                  for slot, replay_id in enumerate(self._ids) if replay_id is not None))

    def load_state_dict(self, state):
        """Validate and rebuild atomically, without consuming any random numbers."""
        try:
            restored = type(self)(state['capacity'], state['alpha'], state['epsilon'])
            if (restored.capacity, restored.alpha, restored.epsilon) != (
                    self.capacity, self.alpha, self.epsilon):
                raise ValueError('priority index checkpoint configuration mismatch')
            for replay_id, raw in state['entries']:
                restored.add(replay_id, self._number(raw, 'priority'))
            historical = self._number(state['max_priority'], 'max_priority')
            if historical < restored._max_priority:
                raise ValueError('max_priority is smaller than historical or active priorities')
            restored._max_priority = historical
        except (TypeError, KeyError, OverflowError) as error:
            raise ValueError('malformed priority index checkpoint') from error
        self.__dict__.update(restored.__dict__)
