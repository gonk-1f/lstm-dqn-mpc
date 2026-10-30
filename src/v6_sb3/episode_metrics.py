"""Constant-memory episode and minibatch summaries; no tensor histories."""
from dataclasses import dataclass, asdict
from math import isfinite


@dataclass
class EpisodeMetrics:
    steps: int = 0
    duration: float = 0.
    return_sum: float = 0.
    economic_cost: float = 0.
    soc_final: float = .6
    loss_sum: float = 0.
    td_mae_sum: float = 0.
    updates: int = 0

    def observe_transition(self, env, reward):
        transition = env.last_transition
        ledger = transition.executed_ledger
        cost = (ledger.h2_cost_cny+ledger.fuel_cell_degradation_cost_cny+
                ledger.battery_degradation_cost_cny+ledger.shore_cost_cny+
                transition.soc_equivalent_hydrogen_cost_cny)
        if getattr(env, 'reward_mode', None) == 'inverse_one_plus_economic':
            from .rewards import economic_cost_cny
            cost = economic_cost_cny(ledger, transition.soc_equivalent_hydrogen_cost_cny)
        if not all(isfinite(float(value)) for value in (reward, cost, transition.actual_soc)):
            raise FloatingPointError('non-finite reward, economic cost or SOC')
        self.steps += 1
        self.duration += float(transition.duration_s)
        self.return_sum += float(reward)
        self.economic_cost += float(cost)
        self.soc_final = float(transition.actual_soc)

    def observe_update(self, loss, td_mae):
        if not isfinite(loss) or not isfinite(td_mae):
            raise FloatingPointError('non-finite minibatch loss or TDMAE')
        self.loss_sum += loss
        self.td_mae_sum += td_mae
        self.updates += 1

    def row(self, episode, status):
        complete = status == 'completed'
        return dict(sample_id=episode.sample_id, completed=complete, terminal_reason=status,
            episode_env_steps=self.steps, duration_coverage=self.duration/episode.duration_s,
            return_sum=self.return_sum, soc_final=self.soc_final,
            loss_mean=self.loss_sum/self.updates if self.updates else None,
            td_mae=self.td_mae_sum/self.updates if self.updates else None,
            episode_adam_updates=self.updates, executed_prefix_economic_cost_cny=self.economic_cost,
            complete_economic_cost_cny=self.economic_cost if complete else None)

    def snapshot(self):
        return asdict(self)
