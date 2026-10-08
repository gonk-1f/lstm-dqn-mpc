"""Economic replay credit and qualified checkpoint selection for v4 studies."""

from dataclasses import dataclass, field
import math


class EconomicUpdateSchedule:
    def __init__(self, cadence: str, *, target_mode: str, target_interval: int = 1000):
        if cadence not in {"episode16", "replay32", "replay16", "replay8"}:
            raise ValueError("unknown economic update cadence")
        if (target_mode not in {"round", "optimizer"}
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
    return bool(summary and summary["episodes"] > 0 and not summary["failed"]
                and summary["completed"] == summary["episodes"])


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
