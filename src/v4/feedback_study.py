"""Explicit, single-configuration v4 reward-feedback verification/training entry.

No parallel jobs or implicit 40-round budget. Formal experiments require an
explicit command and user authorization; this change was verified synthetically.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from v2.data.formal_training_dataset import FormalTrainingDataset
from .monitored_training import _write_json, run_monitored_training
from .review import _manifest_hashes
from .staged_study import _csv_history
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
    parser.add_argument('--cadence',choices=('episode16','replay32','replay16'),default='replay32')
    parser.add_argument('--target-interval',type=int,choices=(250,500,1000),default=500)
    parser.add_argument('--n-step',type=int,choices=(1,8),default=1)
    parser.add_argument('--episode-credit-scope',choices=('sample','voyage'),default='voyage',
                        help='voyage counts completed ONBOARD segments; sample reproduces legacy credit')
    args = parser.parse_args(argv)
    if not 1 <= args.rounds <= 40:
        parser.error('rounds must be explicitly chosen in [1,40]')
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
        required_split_sizes=(30,8),capture_trajectories=True,progress_every_steps=50)
    if dataset.opened_test_payloads != 0 or _manifest_hashes(roots) != before:
        raise RuntimeError('Test opened or dataset manifests changed')
    report['manifest_sha256'] = before
    report['dataset_manifests_unchanged'] = True
    _write_json(output/'report.json',report)
    _csv_history(report,output/'round_metrics.csv')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
