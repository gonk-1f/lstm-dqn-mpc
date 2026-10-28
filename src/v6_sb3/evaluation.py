"""User-launched final Test evaluation of the frozen Validation-selected model."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from v3.control import EconomicMPC
from v4.failure_replay import FailurePenalty
from v4.independent_economics import ZhouEvaluationConfig
from v4.v6_dataset import V6TrainingDataset
from v4.v6_training import CONFIG, write_csv, write_json
from .masked_dqn import MaskedDQN
from .reporting import quiet_evaluate


ROOT = Path(__file__).resolve().parents[2]


def load_test_episodes(data_root):
    """Read the four frozen source payloads using the existing 1-second loader.

    The V6 data directory copied only Train/Validation. Its Test manifest rows
    retain the original canonical 1-second payload locations; no resampling or
    data copying is needed. The training loader's public Test prohibition stays
    unchanged: this explicit final-evaluation entry is the only caller here.
    """
    dataset = V6TrainingDataset.open(data_root)
    rows = dataset._manifest.loc[dataset._manifest['split'].eq('test')].sort_values('candidate_id')
    episodes = []
    for row in rows.itertuples(index=False):
        source = V6TrainingDataset(Path(row.source_dataset_path), dataset._manifest, dataset._metadata)
        source_row = row._replace(relative_path=row.source_payload_relative_path)
        episodes.append(source._load_episode(source_row))
    return tuple(episodes)


def run(run_dir, data_root, output_dir=None):
    run_dir = Path(run_dir).resolve()
    selection = json.loads((run_dir/'analysis/model_selection.json').read_text(encoding='utf-8'))
    configuration = json.loads((run_dir/'configuration.json').read_text(encoding='utf-8'))
    if any(configuration[key] != value for key, value in CONFIG.items()):
        raise ValueError('The environment configuration differs from the selected training run.')
    round_index = selection['selected_round']
    checkpoint = run_dir/selection['checkpoint_relative_path']
    output = Path(output_dir).resolve() if output_dir else run_dir/f'test_round_{round_index:02d}'
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'model_selection.json', selection)
    write_json(output/'configuration.json', configuration)

    torch.set_num_threads(1)
    model = MaskedDQN.load(checkpoint, device='cpu')
    model.policy.set_training_mode(False)
    model.online.requires_grad_(False)
    initial_updates = model._n_updates
    accountant = EconomicMPC(nominal_cost_cny=1.)
    economics = ZhouEvaluationConfig.from_formal_curve()
    penalty = FailurePenalty.from_reference()
    print(f'Test | checkpoint round {round_index:02d} | Greedy | CPU | output: {output}', flush=True)
    episodes = load_test_episodes(data_root)
    with torch.inference_mode():
        summary, rows, failures = quiet_evaluate(
            episodes, model, accountant, economics, penalty, output, round_index, split='test')
    changes = sum(row['fc_nonzero_changes'] for row in rows)
    summary.update(
        split='test', selected_round=round_index, checkpoint=str(checkpoint),
        selection_basis=selection['selection_rule'], evaluation_epsilon=0.,
        test_payloads_opened=len(episodes), evaluation_adam_updates=model._n_updates-initial_updates,
        mean_episode_return=math.fsum(row['training_return'] for row in rows)/len(rows),
        mean_nonzero_fc_delta_kw=(summary['fc_mean_absolute_ramp_kw_per_s']
            * summary['executed_duration_s']/changes if changes else None))
    write_json(output/'test_summary.json', summary)
    write_csv(output/'test_sample_metrics.csv', rows)
    write_json(output/'test_failure_details.json', dict(failures=failures))
    if failures:
        failure_rows = [{key:json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list, tuple))
                        else value for key, value in row.items()} for row in failures]
        write_csv(output/'test_failure_details.csv', failure_rows)
    cost = summary['fixed_complete_set_cost_cny']
    cost_text = f'{cost:,.2f} CNY' if cost is not None else 'not available: some voyages failed'
    lines = [f'# SB3 DQN Test: checkpoint round {round_index:02d}', '',
        f'Completed voyages: {summary["completed"]}/{summary["episodes"]}.',
        f'Full-set economic cost: {cost_text}.',
        f'Actual FC hydrogen consumption: {summary["fc_h2_kg"]:.6f} kg.',
        f'FC degradation cost over executed trajectories: {summary["fc_degradation_cost_cny"]:.2f} CNY.',
        f'Battery degradation cost over executed trajectories: {summary["battery_degradation_cost_cny"]:.2f} CNY.',
        f'Mean absolute nonzero FC adjustment: {summary["mean_nonzero_fc_delta_kw"]} kW.',
        f'FC changes/hour: {summary["fc_changes_per_hour"]:.3f}.',
        f'Adjacent reversals/hour: {summary["fc_reversals_per_hour"]:.3f}.', '',
        'Model selection used Validation only and was frozen before Test payload access.',
        'Greedy inference only; Test transitions are not stored in Replay or used for parameter updates.',
        'Per-voyage CSV.gz contains all 13 Q values, legal rankings, power, SOC and reward components.',
        'When a voyage fails, its cost and hydrogen figures cover only its executed prefix.']
    (output/'test_result.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(f'Test finished | completed {summary["completed"]}/{summary["episodes"]} | '
          f'FC changes {summary["fc_changes_per_hour"]:.2f}/h | '
          f'reversals {summary["fc_reversals_per_hour"]:.2f}/h | cost: {cost_text}', flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True, help='completed SB3 run with analysis/model_selection.json')
    parser.add_argument('--data-root', type=Path, default=ROOT/'data/processed/operating_dataset_v6_1s')
    parser.add_argument('--output-dir', type=Path, help='new output directory; default: run-dir/test_round_NN')
    args = parser.parse_args()
    run(args.run_dir, args.data_root, args.output_dir)


if __name__ == '__main__':
    main()
