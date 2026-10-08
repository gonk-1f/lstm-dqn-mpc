import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
audits = {}
for stage in ('stage1', 'stage2'):
    for run in sorted((root / stage).glob('*')):
        if not run.is_dir():
            continue
        path = run / 'round_history.json'
        if not path.exists():
            lines = (run / 'train.log').read_text(encoding='utf-8').splitlines() if (run / 'train.log').exists() else []
            print(stage, run.name, 'startup', lines[-1:])
            continue
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except (PermissionError, json.JSONDecodeError):
            continue
        row = data['rounds'][-1]
        validation = row['greedy_validation']
        best = data['best_checkpoint']
        print(stage, run.name, 'round', row['round'], 'explore', row['exploratory_train']['completed'],
              'greedy', row['greedy_train']['completed'], 'Val', None if validation is None else validation['completed'],
              'best', None if best is None else (best['round'], round(best['validation_comparable_cost_cny'], 2)),
              'updates', row['economic_optimizer_updates_cumulative'], 'done', (run / 'report.json').exists())
        archive = root / 'interruptions' / (stage + '_' + run.name)
        if (archive / 'round_history.json').exists():
            old = json.loads((archive / 'round_history.json').read_text(encoding='utf-8'))
            count = min(len(data['rounds']), len(old['rounds']))
            audit = {'checked_rounds': count, 'interrupted_completed_rounds': len(old['rounds']),
                     'all_round_rows_exactly_equal': data['rounds'][:count] == old['rounds'][:count]}
            if best and old['best_checkpoint'] and best['round'] == old['best_checkpoint']['round']:
                audit['matching_best_checkpoint_bytes_equal'] = (
                    hashlib.sha256((run / 'best_agent.pt').read_bytes()).hexdigest()
                    == hashlib.sha256((archive / 'best_agent.pt').read_bytes()).hexdigest())
            audits[stage + '_' + run.name] = audit
            print('RECOVERY', run.name, audit)
        if not (run / 'report.json').exists():
            print('LIVE', (run / 'train.log').read_text(encoding='utf-8').splitlines()[-1])
if audits:
    (root / 'reconstruction_audit.json').write_text(json.dumps(audits, indent=2), encoding='utf-8')
