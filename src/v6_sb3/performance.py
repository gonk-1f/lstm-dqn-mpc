"""Optional wall-time instrumentation, removed before checkpoint serialization."""
from collections import defaultdict
from contextlib import contextmanager
from functools import wraps
from threading import local
from time import perf_counter


class PerformanceRecorder:
    def __init__(self):
        self.stages = defaultdict(lambda: dict(calls=0, inclusive_seconds=0., exclusive_seconds=0.))
        self._local = local()

    @contextmanager
    def measure(self, name):
        stack = getattr(self._local, 'stack', None)
        if stack is None:
            stack = self._local.stack = []
        frame = [perf_counter(), 0.]
        stack.append(frame)
        try:
            yield
        finally:
            elapsed = perf_counter()-frame[0]
            stack.pop()
            if stack:
                stack[-1][1] += elapsed
            stage = self.stages[name]
            stage['calls'] += 1
            stage['inclusive_seconds'] += elapsed
            stage['exclusive_seconds'] += max(0., elapsed-frame[1])

    @contextmanager
    def instrument(self, bindings):
        originals = []
        try:
            for owner, attribute, stage in bindings:
                original = getattr(owner, attribute)
                had_local = attribute in vars(owner)
                originals.append((owner, attribute, original, had_local))

                def make_wrapper(method, name):
                    @wraps(method)
                    def timed(*args, **kwargs):
                        with self.measure(name):
                            return method(*args, **kwargs)
                    return timed

                setattr(owner, attribute, make_wrapper(original, stage))
            yield
        finally:
            for owner, attribute, original, had_local in reversed(originals):
                if had_local:
                    setattr(owner, attribute, original)
                else:
                    delattr(owner, attribute)

    def model_bindings(self, model, environment):
        return [(environment, 'step', 'environment_step'),
                (model, 'masked_predict', 'policy_prediction'),
                (model.replay_buffer, 'add', 'replay_store'),
                (model.replay_buffer, 'sample', 'replay_sample'),
                (model.optimizer, 'step', 'adam_step'),
                (model.online, 'forward', 'online_forward'),
                (model.target, 'forward', 'target_forward'),
                (model, 'train', 'training_update_total')]

    def snapshot(self):
        return dict(clock='perf_counter wall time',
            note=('Nested inclusive timings must not be added. Exclusive timings subtract nested scopes. '
                  'Only exited scopes are included; failed/paused calls also count. Restore continues cumulative '
                  'totals without counting downtime or state reconstruction. environment_step includes physical '
                  'execution and queue waiting. policy_prediction includes observation/mask preparation and '
                  'epsilon-random choices, which need no NN forward. online_forward includes both action '
                  'inference and training forwards; target_forward is the TD-target forward. '
                  'replay_store/sample are buffer insertion/sampling only. adam_step excludes backward/clip. '
                  'training_update_total includes forward/backward/clipping/diagnostics and Replay sampling; '
                  'its calls equal Adam updates only for the fixed one-update-per-call runner on successful updates. '
                  'validation_executor includes validation_prediction (Greedy selection plus NN inference). '
                  'report_write measures Validation CSV/JSON writes only; checkpoint_write measures model, '
                  'Replay and state serialization, excluding the subsequent JSON markers and performance file.'),
            stages={name:dict(values) for name, values in sorted(self.stages.items())})

    def restore(self, saved):
        for name, values in saved.get('stages', {}).items():
            self.stages[name].update(values)
