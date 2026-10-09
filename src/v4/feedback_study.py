"""Explicit, single-configuration v4 reward-feedback verification/training entry.

No parallel jobs or implicit 40-round budget. Formal experiments require an
explicit command and user authorization; this change was verified synthetically.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import traceback
from typing import Sequence

from v2.data.formal_training_dataset import FormalTrainingDataset
from .monitored_training import _write_json, run_monitored_training
from .review import _manifest_hashes
from .train import _default_data_root
from .experiment_paths import unarchived_output_path


def fresh_output_path(path: Path) -> Path:
    return unarchived_output_path(path,require_empty=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--rounds',type=int,required=True)
    parser.add_argument('--beta-soc',type=float,default=500.)
    parser.add_argument('--reward-feedback',choices=('original','redistributed'),default='redistributed')
    parser.add_argument('--reward-scale',type=float,default=1.,
                        help='uniform multiplier applied once to final economic-Q replay rewards; ledgers remain CNY')
    parser.add_argument('--failure-terminal-quota',type=int,default=0,
                        help='failure terminal samples per economic Q batch; 0 preserves uniform replay')
    parser.add_argument('--cadence',choices=('episode16','replay32','replay16'),default='replay32')
    parser.add_argument('--target-interval',type=int,choices=(250,500,1000),default=500)
    parser.add_argument('--n-step',type=int,choices=(1,8),default=1)
    parser.add_argument('--failure-penalty-scale',type=float,default=1.,
                        help='multiplier of frozen Train-only P95 economic cost, in reward-equivalent CNY')
    parser.add_argument('--episode-credit-scope',choices=('sample','voyage'),default='voyage',
                        help='voyage counts completed ONBOARD segments; sample reproduces legacy credit')
    parser.add_argument('--diagnostic-rounds',default='',
                        help='comma-separated read-only network/actual-Train-Q snapshots, e.g. 1,10,20,30,40')
    args = parser.parse_args(argv)
    if not 1 <= args.rounds <= 40:
        parser.error('rounds must be explicitly chosen in [1,40]')
    if not math.isfinite(args.reward_scale) or args.reward_scale <= 0:
        parser.error('reward-scale must be finite and positive')
    if args.failure_terminal_quota < 0:
        parser.error('failure-terminal-quota must be nonnegative')
    try:
        diagnostic_rounds = tuple(int(value.strip()) for value in args.diagnostic_rounds.split(',')
                                  if value.strip())
    except ValueError:
        parser.error('diagnostic-rounds must contain comma-separated integers')
    if (len(set(diagnostic_rounds)) != len(diagnostic_rounds)
            or any(not 1 <= value <= args.rounds for value in diagnostic_rounds)):
        parser.error('diagnostic-rounds must be distinct and within the requested rounds')
    output = fresh_output_path(args.output_dir)
    roots = tuple(_default_data_root(name) for name in (
        'operating_dataset_zero_boundary_v2','operating_dataset_zero_boundary_v2_ais',
        'operating_dataset_zero_boundary_v2_modes'))
    before = _manifest_hashes(roots)
    dataset = FormalTrainingDataset.open(*roots)
    _, report = run_monitored_training(dataset,output_dir=output,rounds=args.rounds,
        beta_soc=args.beta_soc,cadence=args.cadence,target_mode='optimizer',
        target_interval=args.target_interval,seed=42,batch_size=64,epsilon_start=1.,epsilon_end=.05,
        redistribute_battery_energy=args.reward_feedback == 'redistributed',
        episode_credit_scope=args.episode_credit_scope,n_step=args.n_step,
        required_split_sizes=(30,8),capture_trajectories=True,progress_every_steps=50,
        learn_no_feasible_failures=True,failure_penalty_scale=args.failure_penalty_scale,
        reward_scale=args.reward_scale,failure_terminal_quota=args.failure_terminal_quota,
        manifest_sha256=before,
        dataset_roots=tuple(str(root.resolve()) for root in roots),
        abort_on_execution_error=True,capture_log=True,
        diagnostic_rounds=diagnostic_rounds)
    try:
        after = _manifest_hashes(roots)
        if dataset.opened_test_payloads != 0 or after != before:
            raise RuntimeError('Test opened or dataset manifests changed')
    except BaseException as exc:
        report.update(completed_training=False,run_status='aborted',
                      dataset_manifests_unchanged=False,checkpoint_selection_valid=False)
        report['test_payloads_opened'] = dataset.opened_test_payloads
        if 'after' in locals():
            report['final_manifest_sha256'] = after
        report['abort'] = {'exception_type':type(exc).__name__,'message':str(exc),
            'phase':'final_dataset_guard','completed_rounds_retained':len(report['rounds']),
            'automatic_resume':False,'checkpoint_artifact_preserved_as_invalid_evidence':True}
        if report['best_checkpoint'] is not None:
            report['best_checkpoint']['selection_valid'] = False
            report['best_checkpoint']['invalidation_reason'] = 'final_dataset_guard_failed'
        _write_json(output/'report.json',report)
        _write_json(output/'round_history.json',report)
        detail = traceback.format_exc()
        with (output/'train.log').open('a',encoding='utf-8') as handle:
            handle.write(detail)
        sys.stderr.write(detail)
        raise
    report['manifest_sha256'] = before
    report['final_manifest_sha256'] = after
    report['dataset_manifests_unchanged'] = True
    report['checkpoint_selection_valid'] = report['best_checkpoint'] is not None
    _write_json(output/'report.json',report)
    _write_json(output/'round_history.json',report)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
