"""Economic replay credit and qualified checkpoint selection for v4 studies."""

from dataclasses import dataclass, field
import math


class EconomicUpdateSchedule:
    def __init__(self, cadence: str, *, target_mode: str, target_interval: int = 1000):
        if cadence not in {"episode16", "replay32", "replay16", "replay8"}:
            raise ValueError("unknown economic update cadence")
        if (target_mode not in {"round", "optimizer", "soft"}
                or type(target_interval) is not int or target_interval < 1):
            raise ValueError("invalid target schedule")
        self.cadence = cadence
        self.target_mode = target_mode
        self.target_interval = target_interval
        self.remaining_transition_credit = 0
        self.pending_episode_updates = 0

    def grant(self, *, insertions: int, completed_episodes: int) -> None:
        if any(type(value) is not int or value < 0 for value in (insertions, completed_episodes)):
            raise ValueError("credit counters must be nonnegative integers")
        if self.cadence == "episode16":
            self.pending_episode_updates += 16 * completed_episodes
        else:
            self.remaining_transition_credit += insertions

    def consume(self, agent, *, batch_size: int) -> list[float]:
        stride = int(self.cadence.removeprefix('replay')) if self.cadence != 'episode16' else 0
        losses = []
        while (
            self.pending_episode_updates > 0 if self.cadence == "episode16"
            else self.remaining_transition_credit >= stride
        ):
            loss = agent.learn(batch_size=batch_size)
            if loss is None:
                break
            losses.append(loss)
            if self.cadence == "episode16":
                self.pending_episode_updates -= 1
            else:
                self.remaining_transition_credit -= stride
            if self.target_mode == "optimizer" and agent.economic_optimizer_updates % self.target_interval == 0:
                agent.sync_target()
        return losses


def fully_completed(summary: dict | None) -> bool:
    if summary is not None and 'evaluable_episodes' in summary:
        return bool(summary['evaluable_episodes'] > 0
                    and summary['evaluable_completed'] == summary['evaluable_episodes']
                    and summary['physical_failures'] == 0
                    and summary['execution_errors'] == 0
                    and summary['data_truncated_episodes'] == summary['unknown_episodes'])
    return bool(summary and summary["episodes"] > 0 and not summary["failed"]
                and summary["completed"] == summary["episodes"])


def annotate_evaluation(summary: dict, episodes, results, errors) -> dict:
    """Keep fixed complete-sample costs separate from UNKNOWN prefix diagnostics."""
    episodes, results, errors = tuple(episodes), tuple(results), tuple(errors)
    complete_ids = {str(item.sample_id) for item in episodes
                    if 'unknown' not in item.operating_mode}
    unknown_ids = {str(item.sample_id) for item in episodes
                   if 'unknown' in item.operating_mode}
    result_ids = {str(item.sample_id) for item in results}
    error_ids = {str(item.sample_id) for item in errors}
    if (len(complete_ids) + len(unknown_ids) != len(episodes)
            or len(result_ids) != len(results) or len(error_ids) != len(errors)
            or result_ids & error_ids or result_ids | error_ids != complete_ids | unknown_ids
            or result_ids & unknown_ids):
        raise ValueError('evaluation sample outcomes do not match the fixed split')
    physical = tuple(error for error in errors if error.failure_kind == 'no_feasible_action')
    truncations = tuple(error for error in errors if error.failure_kind == 'data_truncation')
    execution = tuple(error for error in errors
                      if error.failure_kind not in ('no_feasible_action', 'data_truncation'))
    complete_cost = (summary['completed_cost_cny']
                     if complete_ids and complete_ids <= result_ids and not execution else None)
    summary.update(
        evaluable_episodes=len(complete_ids),
        evaluable_sample_ids=sorted(complete_ids),
        evaluable_completed=len(result_ids & complete_ids),
        unknown_episodes=len(unknown_ids),
        unknown_sample_ids=sorted(unknown_ids),
        data_truncated_episodes=len(truncations),
        physical_failures=len(physical),
        failed_samples=len(physical),
        noncompleted_samples=len(errors),
        evaluable_physical_failures=sum(error.sample_id in complete_ids for error in physical),
        unknown_prefix_physical_failures=sum(error.sample_id in unknown_ids for error in physical),
        execution_errors=len(execution),
        unknown_prefix_executed_onboard_transitions=sum(
            len(error.executed_transitions) for error in errors if error.sample_id in unknown_ids),
        cost_cny=complete_cost,
        comparable_evaluable_cost_cny=complete_cost,
        cost_comparison_scope='fixed complete samples only; UNKNOWN prefixes excluded',
    )
    return summary


@dataclass
class BestCheckpoint:
    round: int | None = None
    cost_cny: float | None = None
    first_eligible_round: int | None = None
    eligible_rounds: list[int] = field(default_factory=list)

    def consider(self, round_index: int, train: dict, validation: dict | None) -> bool:
        if not fully_completed(train) or not fully_completed(validation):
            return False
        cost = validation["cost_cny"]
        if cost is None or not math.isfinite(cost):
            raise ValueError("qualified Validation needs a finite comparable economic cost")
        self.eligible_rounds.append(round_index)
        if self.first_eligible_round is None:
            self.first_eligible_round = round_index
        if self.cost_cny is not None and cost >= self.cost_cny:
            return False
        self.round = round_index
        self.cost_cny = cost
        return True
