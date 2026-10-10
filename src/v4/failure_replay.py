"""Explicit economic-TD failure views; no changes to executed physical ledgers."""
from dataclasses import dataclass, replace
import json
from math import ceil, isfinite, isclose
from pathlib import Path

from .control import DirectTransition, NoFeasibleFCActionError, ReplayExecutionError
from .reward_feedback import BatteryEnergyValue


FAILURE_TERMINALS = frozenset(('failure_soc_limited','failure_structural_power'))


@dataclass(frozen=True)
class FailurePenalty:
    reference_amount: float
    scale: float
    reference_split: str
    reference_count: int
    reference_statistic: str
    unit: str
    source_file: str
    source_sha256: str

    def __post_init__(self):
        if (not isfinite(self.reference_amount) or self.reference_amount<=0 or
                not isfinite(self.scale) or self.scale<0 or not isfinite(self.amount)):
            raise ValueError('failure reference, scale and resulting penalty must be finite')

    @property
    def amount(self) -> float:
        return self.reference_amount*self.scale

    @classmethod
    def from_reference(cls, *, scale: float = 1.0):
        if not isfinite(scale) or scale < 0:
            raise ValueError('failure penalty scale must be finite and nonnegative')
        reference=json.loads(Path(__file__).with_name('failure_penalty_reference.json').read_text(encoding='utf-8'))
        costs=sorted(reference['training_costs_cny'].values())
        if (reference['reference_split'] != 'train' or reference['quantile'] != .95 or len(costs) != 30
                or any(not isfinite(c) or c<=0 for c in costs)):
            raise ValueError('failure penalty reference must use30 positive completed Train costs')
        amount=costs[ceil(reference['quantile']*len(costs))-1]
        return cls(amount,float(scale),'train',len(costs),reference['reference_statistic'],
                   reference['unit'],reference['source_file'],reference['source_sha256'])


@dataclass(frozen=True)
class FailureReplay:
    transitions: tuple[DirectTransition,...]
    successful_prefix_transitions: int
    failed_suffix_transitions: int
    failure_cause: str
    event_only: bool
    controllable_by_power_policy: bool | None


def prepare_failure_replay(error: ReplayExecutionError, *, penalty: FailurePenalty,
                           energy_value: BatteryEnergyValue,
                           redistribute_battery_energy: bool) -> FailureReplay:
    """Convert only a genuine no-action failure, preserving every real state."""
    if error.failure_kind != 'no_feasible_action' or not isinstance(error.cause,NoFeasibleFCActionError):
        raise ValueError('only genuine no_feasible_action errors qualify for economic failure TD')
    if error.failure_cause not in ('soc_limited','structural_power'):
        raise ValueError('physical failure cause must be classified explicitly')
    values=tuple(error.executed_transitions)
    if any(t.terminal_reason in FAILURE_TERMINALS for t in values):
        raise ValueError('trajectory is already failure-terminated')
    start=1+max((i for i,t in enumerate(values) if t.is_successful_terminal),default=-1)
    suffix=values[start:]
    if not suffix:
        return FailureReplay(values,start,0,error.failure_cause,True,
                             False if error.failure_cause=='structural_power' else None)
    last=suffix[-1]
    if last.done or last.next_feasible_actions or last.modeled_terminal_ledger is not None:
        raise ValueError('failure suffix must end at an executed nonterminal infeasible-next-state transition')
    if last.terminal_correction or last.failure_penalty_equivalent_cny:
        raise ValueError('failure suffix is already corrected or penalized')
    for item in suffix:
        expected=-(energy_value(item.actual_soc)-energy_value(item.state[0])) if redistribute_battery_energy else 0.
        if not isclose(item.immediate_battery_energy_adjustment,expected,rel_tol=1e-12,abs_tol=1e-9):
            raise ValueError('redistribution configuration does not match the executed reward fields')
    correction=(energy_value.terminal_correction(suffix[0].state[0],last.actual_soc)
                if redistribute_battery_energy else 0.)
    terminal=replace(last,reward_cny=last.reward_cny+correction-penalty.amount,
        done=True,next_feasible_actions=(),terminal_correction=correction,
        terminal_reason='failure_'+error.failure_cause,failure_penalty_equivalent_cny=penalty.amount,
        original_reward=last.reward_cny if last.original_reward is None else last.original_reward)
    return FailureReplay((*values[:-1],terminal),start,len(suffix),error.failure_cause,False,
                         False if error.failure_cause=='structural_power' else None)
