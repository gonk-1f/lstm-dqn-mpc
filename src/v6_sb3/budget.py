"""Estimate a future budget from an existing trial profile; never runs a model."""
import argparse
import json
from math import ceil
from pathlib import Path


def estimate(trial_dir, formal_steps, eval_every):
    if formal_steps <= 0 or eval_every <= 0:
        raise ValueError('Formal steps and evaluation interval must be positive.')
    trial_dir = Path(trial_dir)
    configuration = json.loads((trial_dir/'configuration.json').read_text(encoding='utf-8'))
    stages = json.loads((trial_dir/'performance.json').read_text(encoding='utf-8'))['stages']
    training_calls = stages.get('training_update_total', {}).get('calls', 0)
    formal_time = stages.get('formal_collect_total', {}).get('inclusive_seconds', 0.)
    summaries = list(trial_dir.glob('validation_step_*/summary.json'))
    validation_count = len(summaries)
    try:
        result = json.loads((trial_dir/'result.json').read_text(encoding='utf-8'))
    except (FileNotFoundError, json.JSONDecodeError):
        result = {}
    updates = result.get('adam_updates') if isinstance(result, dict) else None
    if (not isinstance(result, dict) or result.get('status') not in ('completed', 'paused', 'pruned')
            or type(updates) is not int or updates < 0):
        return dict(can_estimate=False, measured_formal_updates=None,
                    completed_evaluations=validation_count,
                    reason='Need a trustworthy result.json with completed/paused/pruned status and integer adam_updates.')
    # Timers count calls even when train raises before Adam. A previous paused
    # result can survive a failed resume, so also require matching snapshots.
    if updates != training_calls:
        return dict(can_estimate=False, measured_formal_updates=updates,
                    completed_evaluations=validation_count,
                    reason='Actual Adam updates do not match recorded training calls; use a consistent successful or paused trial profile.')
    if not updates or not formal_time or not validation_count:
        return dict(can_estimate=False, measured_formal_updates=updates,
                    completed_evaluations=validation_count,
                    reason='Need both formal update measurements and at least one full evaluation; warmup/paused prefixes are insufficient.')
    required = ('validation_executor', 'report_write', 'checkpoint_write')
    missing = [name for name in required if not stages.get(name, {}).get('calls', 0)
               or stages.get(name, {}).get('inclusive_seconds', 0.) <= 0.]
    if missing:
        return dict(can_estimate=False, measured_formal_updates=updates,
                    completed_evaluations=validation_count,
                    reason='Missing completed timing measurements: '+', '.join(missing))
    if stages['report_write']['calls'] != validation_count:
        return dict(can_estimate=False, measured_formal_updates=updates,
                    completed_evaluations=validation_count,
                    reason='Validation summaries do not match recorded report writes; use a consistent trial profile.')
    validation_time = stages.get('validation_executor', {}).get('inclusive_seconds', 0.)/validation_count
    report_time = stages.get('report_write', {}).get('inclusive_seconds', 0.)/validation_count
    checkpoint = stages.get('checkpoint_write', {})
    checkpoint_time = checkpoint.get('inclusive_seconds', 0.)/max(1, checkpoint.get('calls', 0))
    evaluations = ceil(formal_steps/eval_every)
    estimate_seconds = (stages.get('warmup_collect_total', {}).get('inclusive_seconds', 0.)
        + formal_steps*formal_time/updates + evaluations*(validation_time+report_time+checkpoint_time))
    return dict(can_estimate=True, measured_formal_updates=updates, measured_formal_updates_per_second=updates/formal_time,
        measured_validation_seconds_per_evaluation=validation_time,
        requested_formal_steps=formal_steps, requested_evaluations=evaluations,
        estimated_seconds_per_trial=estimate_seconds,
        scope=('synthetic smoke only; NOT a real-voyage runtime prediction'
               if configuration['experiment_id'].startswith('synthetic_') else
               'same machine, model, data scope and warmup; extrapolation, not a guarantee'),
        caveats=('Inclusive nested stages are not summed twice. The estimate covers measured collection, '
                 'Validation execution/report output and checkpoint serialization only; data loading, model '
                 'construction, resume reconstruction, study bookkeeping and final output overhead are excluded. '
                 'Adam updates come from result.json and must match timed training calls for the fixed gradient_steps=1 runner. '
                 'Failures, voyage length, Replay size, network/batch size, report detail and machine load '
                 'alter runtime. A small synthetic smoke cannot establish a full-dataset budget.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial-dir', type=Path, required=True)
    parser.add_argument('--formal-steps', type=int, required=True)
    parser.add_argument('--eval-every', type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(estimate(args.trial_dir,args.formal_steps,args.eval_every),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
