import csv
import json
from pathlib import Path

root = Path(__file__).resolve().parent
reports = {}
for beta in (250, 500, 1000, 2000):
    run = root / 'stage1' / f'beta_{beta}'
    report = json.loads((run / 'report.json').read_text(encoding='utf-8'))
    assert report['completed_training'] and len(report['rounds']) == 40
    assert report['dataset_manifests_unchanged'] and report['test_payloads_opened'] == 0
    if report['best_checkpoint']:
        assert report['best_checkpoint_replay_verified']
    reports[run.name] = report

from v4.staged_study import _plots, compact_report, select_beta
from v4.monitored_training import _write_json

_plots(reports, root, 'stage1')
chosen = select_beta(reports)
summary = {'stage1': {key: compact_report(value) for key, value in reports.items()},
           'chosen_beta_label': chosen, 'stage2': None,
           'stage2_status': 'not_started' if chosen else 'SKIPPED_no_qualified_beta',
           'test_payloads_opened': 0}
_write_json(root / 'stage1_summary.json', summary)
rows = []
for label, report in reports.items():
    best = report['best_checkpoint']
    row = {'beta': report['hyperparameters']['beta_soc'],
           'final_greedy_train_completed': report['rounds'][-1]['greedy_train']['completed'],
           'eligible_rounds': ','.join(map(str, report['eligible_rounds'])),
           'best_round': None if best is None else best['round'],
           'validation_observed_cny': None if best is None else best['validation']['completed_observed_cost_cny'],
           'validation_modeled_cny': None if best is None else best['validation']['completed_modeled_terminal_cost_cny'],
           'validation_comparable_cny': None if best is None else best['validation_comparable_cost_cny'],
           'validation_soc_soft_penalty': None if best is None else best['validation']['completed_soc_soft_penalty_cny'],
           'economic_optimizer_updates': report['execution_counts']['economic_optimizer_updates']}
    rows.append(row)
    print(row)
with (root / 'stage1_comparison.csv').open('w', encoding='utf-8', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print('CHOSEN', chosen)
