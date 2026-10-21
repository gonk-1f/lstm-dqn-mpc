"""Frozen V4 studies and V5 causal-state, pure-economic interleaved DDQN."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import traceback
from typing import Sequence

from v2.data.formal_training_dataset import FormalTrainingDataset
from .control import DEFAULT_BETA_SOC, DEFAULT_STATE_MODE, REWARD_COST_MODES, STATE_MODES
from .monitored_training import (
    _write_json, run_monitored_training, validate_experiment_identity, validate_replay_configuration,
)
from .review import _manifest_hashes
from .train import _default_data_root
from .experiment_paths import unarchived_output_path
from .independent_dataset import IndependentTrainingDataset
from .independent_economics import ZhouEvaluationConfig
from .actions import FC_ACTION_MODES


def fresh_output_path(path: Path) -> Path:
    return unarchived_output_path(path,require_empty=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--resume-from',type=Path,default=None)
    parser.add_argument('--dataset-mode',choices=('legacy','independent'),default='legacy',
                        help='independent uses the active frozen voyage split, no SHORE or terminal charging')
    parser.add_argument('--data-root',type=Path,default=None,
                        help='independent dataset root; legacy retains its three frozen roots')
    parser.add_argument('--rounds',type=int,default=100)
    parser.add_argument('--seed',type=int,default=42,
                        help='paired runs use the same seed; subsequent repeats may use43 and44')
    parser.add_argument('--experiment-id',default=None,
                        help='report/checkpoint identifier only; does not change learning')
    parser.add_argument('--beta-soc',type=float,default=DEFAULT_BETA_SOC,
                        help='fixed-reference SOC auxiliary reward coefficient; not an economic expense')
    parser.add_argument('--reward-feedback',choices=('original','redistributed'),default='original')
    parser.add_argument('--state-mode',choices=STATE_MODES,default=DEFAULT_STATE_MODE,
                        help='history7 retains FC/load differences; minimal3 uses SOC/load/previous FC; load_delta4 appends the causal observed load change for V5')
    parser.add_argument('--action-mode',choices=FC_ACTION_MODES,default='absolute61',
                        help='absolute61 preserves historical runs; increment7 executes previous actual FC plus one of -60,-30,-10,0,10,30,60 kW without clipping or fallback')
    parser.add_argument('--reward-cost-mode',choices=REWARD_COST_MODES,default='full_economic',
                        help='historical V4 reward modes remain available; zhou_economic_only is the V5 signed SOC-equivalent hydrogen plus actual hydrogen/FC degradation/battery degradation reward without auxiliary penalties')
    parser.add_argument('--reward-scale',type=float,default=.001,
                        help='uniform multiplier applied once to final economic-Q replay rewards; ledgers remain CNY')
    parser.add_argument('--failure-terminal-quota',type=int,default=2,
                        help='failure terminal samples per economic Q batch; 0 preserves uniform replay')
    parser.add_argument('--replay-sampling',choices=('uniform','prioritized'),default='uniform',
                        help='economic replay sampling; prioritized replay is an explicit candidate option')
    parser.add_argument('--per-alpha',type=float,default=.6,
                        help='proportional replay priority exponent in [0,1]')
    parser.add_argument('--per-beta',type=float,default=1.,
                        help='importance-sampling correction exponent in [0,1]; 1 applies full correction')
    parser.add_argument('--per-epsilon',type=float,default=1e-6,
                        help='finite positive offset keeping proportional priorities nonzero')
    parser.add_argument('--cadence',choices=('episode16','replay32','replay16'),default='replay32',
                        help='replay32 baseline; replay16 requires independent increment7 and an explicit experiment-id')
    parser.add_argument('--update-timing',choices=('episode_end','interleaved'),default='episode_end',
                        help='episode_end retains concentrated updates; interleaved is an independent increment7 replay32 study with an explicit experiment-id')
    parser.add_argument('--n-step',type=int,choices=(1,8),default=8)
    parser.add_argument('--unknown-replay-mode',choices=('discard','observed_nstep'),
                        default='observed_nstep',
                        help='retain full observed n-step windows before UNKNOWN; discard reproduces prior replay semantics')
    parser.add_argument('--failure-penalty-scale',type=float,default=1.,
                        help='multiplier of frozen Train-only P95 economic cost, in reward-equivalent CNY')
    parser.add_argument('--episode-credit-scope',choices=('sample','voyage'),default='voyage',
                        help='voyage counts completed ONBOARD segments; sample reproduces legacy credit')
    parser.add_argument('--diagnostic-rounds',default='',
                        help='comma-separated read-only network/actual-Train-Q snapshots, e.g. 1,10,20,30,40')
    args = parser.parse_args(argv)
    try:
        validate_experiment_identity(args.seed, args.experiment_id)
        validate_replay_configuration(args.replay_sampling, args.per_alpha, args.per_beta, args.per_epsilon)
    except ValueError as exc:
        parser.error(str(exc))
    incremental = args.action_mode == 'increment7'
    v5 = args.state_mode == 'load_delta4' or args.reward_cost_mode == 'zhou_economic_only'
    if v5 and (
            args.dataset_mode != 'independent' or not incremental
            or args.state_mode != 'load_delta4' or args.reward_cost_mode != 'zhou_economic_only'
            or args.n_step != 1 or args.beta_soc != 0.
            or args.update_timing != 'interleaved' or args.cadence != 'replay32'
            or args.experiment_id is None or args.unknown_replay_mode != 'observed_nstep'):
        parser.error('V5 requires independent/load_delta4/zhou_economic_only/increment7, n-step=1, beta-soc=0, interleaved/replay32, observed_nstep and an explicit experiment-id')
    replay16_study = (args.cadence == 'replay16' and args.dataset_mode == 'independent'
                      and incremental and args.experiment_id is not None)
    if args.cadence != 'replay32' and not replay16_study:
        parser.error('replay16 is only available for an independent increment7 study with an explicit experiment-id')
    if args.update_timing == 'interleaved' and (
            args.dataset_mode != 'independent' or not incremental
            or args.cadence != 'replay32' or args.experiment_id is None):
        parser.error('interleaved requires independent increment7/replay32 and an explicit experiment-id')
    if (args.rounds != 100 or args.beta_soc != (0. if v5 else DEFAULT_BETA_SOC) or args.reward_feedback != 'original'
            or args.reward_scale != .001 or args.failure_terminal_quota != 2
            or args.failure_penalty_scale != 1 or args.episode_credit_scope != 'voyage'):
        parser.error('formal 100-round run uses the frozen DDQN configuration')
    if args.dataset_mode == 'legacy' and args.data_root is not None:
        parser.error('--data-root is only supported with --dataset-mode independent')
    if incremental and not v5 and (args.dataset_mode != 'independent' or args.state_mode != 'minimal3'
                        or args.reward_cost_mode != 'zhou_joint' or args.n_step != 1):
        parser.error('increment7 requires independent/minimal3/zhou_joint and n-step=1')
    if not incremental and (args.n_step != 8 or args.reward_cost_mode in ('zhou_joint','zhou_economic_only')):
        parser.error('historical absolute61 formal configuration retains n-step=8 and its original rewards')
    if args.dataset_mode == 'independent' and (
            args.state_mode != ('load_delta4' if v5 else 'minimal3')
            or args.reward_cost_mode != ('zhou_economic_only' if v5 else 'zhou_joint' if incremental else 'control_only')
            or args.replay_sampling != 'prioritized'
            or (args.per_alpha,args.per_beta,args.per_epsilon) != (.6,1.,1e-6)):
        parser.error('independent runs require minimal3 and unchanged prioritized .6/1/1e-6 with the selected experiment reward')
    try:
        diagnostic_rounds = tuple(int(value.strip()) for value in args.diagnostic_rounds.split(',')
                                  if value.strip())
    except ValueError:
        parser.error('diagnostic-rounds must contain comma-separated integers')
    if (len(set(diagnostic_rounds)) != len(diagnostic_rounds)
            or any(not 1 <= value <= args.rounds for value in diagnostic_rounds)):
        parser.error('diagnostic-rounds must be distinct and within the requested rounds')
    output = (fresh_output_path(args.output_dir) if args.resume_from is None
              else unarchived_output_path(args.output_dir))
    if args.dataset_mode == 'independent':
        roots = (args.data_root or _default_data_root('operating_dataset_independent_v2'),)
        dataset = IndependentTrainingDataset.open(roots[0])
        before = dataset.identity_hashes()
        required_sizes = dataset.split_sizes
        economic_evaluation_config = ZhouEvaluationConfig.from_formal_curve()
    else:
        roots = tuple(_default_data_root(name) for name in (
            'operating_dataset_zero_boundary_v2','operating_dataset_zero_boundary_v2_ais',
            'operating_dataset_zero_boundary_v2_modes_v3'))
        before = _manifest_hashes(roots)
        dataset = FormalTrainingDataset.open(*roots)
        required_sizes = (30,8)
        economic_evaluation_config = None
    _, report = run_monitored_training(dataset,output_dir=output,rounds=args.rounds,
        beta_soc=args.beta_soc,cadence=args.cadence,target_mode='soft',
        update_timing=args.update_timing,
        reward_cost_mode=args.reward_cost_mode,
        state_mode=args.state_mode,
        action_mode=args.action_mode,
        target_interval=500,seed=args.seed,batch_size=64,epsilon_start=1.,epsilon_end=.05,
        redistribute_battery_energy=False,
        episode_credit_scope=args.episode_credit_scope,n_step=args.n_step,
        required_split_sizes=required_sizes,capture_trajectories=True,progress_every_steps=50,
        learn_no_feasible_failures=True,failure_penalty_scale=args.failure_penalty_scale,
        reward_scale=args.reward_scale,failure_terminal_quota=args.failure_terminal_quota,
        replay_sampling=args.replay_sampling,per_alpha=args.per_alpha,
        per_beta=args.per_beta,per_epsilon=args.per_epsilon,
        unknown_replay_mode=args.unknown_replay_mode,
        experiment_id=args.experiment_id,
        manifest_sha256=before,
        dataset_roots=tuple(str(root.resolve()) for root in roots),
        economic_evaluation_config=economic_evaluation_config,
        abort_on_execution_error=True,capture_log=True,
        diagnostic_rounds=diagnostic_rounds,resume_from=args.resume_from)
    try:
        after = (dataset.identity_hashes() if args.dataset_mode == 'independent'
                 else _manifest_hashes(roots))
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
