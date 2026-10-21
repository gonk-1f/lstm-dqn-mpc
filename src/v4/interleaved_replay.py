"""Closed one-step experience insertion using the existing replay and scheduler."""
from .failure_replay import FAILURE_TERMINALS


class InterleavedReplayWriter:
    def __init__(self, agent, schedule, episode, *, source_round, batch_size, losses):
        if (agent.n_step != 1 or agent.action_mode != 'increment7'
                or getattr(episode, 'task_mode', None) != 'independent_voyage'
                or schedule.cadence != 'replay32' or schedule.target_mode != 'soft'):
            raise ValueError('interleaved replay requires independent increment7/1-step/replay32/soft')
        self.agent, self.schedule, self.episode = agent, schedule, episode
        self.source_round, self.batch_size, self.losses = source_round, batch_size, losses
        self.inserted_count = 0
        self.replay_ids = []
        self.callback_optimizer_updates = self.finalization_optimizer_updates = 0
        self.closed = False

    def _insert(self, transition, outcome):
        row = int(self.episode.source_1s_row[self.inserted_count])
        self.agent.remember(transition.state, transition.action_kw, transition.reward_cny,
            transition.next_state, done=transition.done,
            next_feasible_actions=transition.next_feasible_actions,
            experience_outcome=outcome, terminal_reason=transition.terminal_reason,
            failure_penalty_equivalent_cny=transition.failure_penalty_equivalent_cny,
            source_sample_id=str(self.episode.sample_id), source_round=self.source_round,
            source_row_index=row)
        self.replay_ids.append(self.agent.replay[-1].replay_id)
        self.inserted_count += 1
        self.schedule.grant(insertions=1, completed_episodes=0)
        before = self.agent.economic_optimizer_updates
        self.losses.extend(self.schedule.consume(self.agent, batch_size=self.batch_size))
        return self.agent.economic_optimizer_updates - before

    def observe(self, row_index, transition):
        if self.closed or row_index != self.inserted_count:
            raise ValueError('streamed transitions must arrive once, in original order')
        if (not transition.done and not transition.next_feasible_actions
                or transition.terminal_reason in FAILURE_TERMINALS):
            raise ValueError('failure tail must be finalized before insertion')
        self.callback_optimizer_updates += self._insert(transition, 'success')

    def finish_success(self, transitions):
        if (self.closed or len(transitions) != self.inserted_count
                or not transitions or not transitions[-1].is_successful_terminal):
            raise ValueError('streamed success must close exactly the executed voyage')
        self.closed = True

    def finish_failure(self, transitions):
        if not self.closed and not transitions and self.inserted_count == 0:
            self.closed = True
            return 0  # No valid action was executed: no synthetic terminal item.
        if (self.closed or len(transitions) != self.inserted_count + 1
                or not transitions[-1].done
                or transitions[-1].terminal_reason not in FAILURE_TERMINALS):
            raise ValueError('exactly one withheld executed action must become the failure terminal')
        self.agent.finalize_streamed_failure_prefix(self.replay_ids)
        self.finalization_optimizer_updates += self._insert(transitions[-1], 'failure')
        self.closed = True
        return self.inserted_count
