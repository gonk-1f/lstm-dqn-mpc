"""Archive existing study evidence without running or evaluating any agent."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

from v4.review import _manifest_hashes
from v4.staged_study import _csv_history, _plots, compact_report
from v4.train import _default_data_root


root = Path(__file__).resolve().parent
worktree = root.parents[1]
archive = worktree / 'docs/results/v4_staged_beta_cadence_seed42'
figures = worktree / 'docs/figures/v4_staged_beta_cadence_seed42'
archive.mkdir(parents=True, exist_ok=True)
figures.mkdir(parents=True, exist_ok=True)

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

stage1 = {f'beta_{beta}': read(root / f'stage1/beta_{beta}/report.json')
          for beta in (250, 500, 1000, 2000)}
roots = tuple(_default_data_root(name) for name in (
    'operating_dataset_zero_boundary_v2', 'operating_dataset_zero_boundary_v2_ais',
    'operating_dataset_zero_boundary_v2_modes'))
current_manifests = _manifest_hashes(roots)
assert all(r['manifest_sha256'] == current_manifests for r in stage1.values())
assert all(r['completed_training'] and len(r['rounds']) == 40
           and r['test_payloads_opened'] == 0 and r['best_checkpoint_replay_verified']
           for r in stage1.values())

stage2 = {}
partial = {}
for label, cadence, pid in [('A', 'episode16', 20124), ('B', 'replay16', 13632)]:
    directory = root / f'stage2/{label}_{cadence}'
    report = read(directory / 'round_history.json')
    assert report['completed_training'] is False and report['test_payloads_opened'] == 0
    last_progress = next(line for line in reversed((directory / 'train.log').read_text(
        encoding='utf-8').splitlines()) if line.startswith('progress '))
    report['pause_metadata'] = {
        'status': 'PAUSED_BY_USER', 'stopped_worker_pid': pid,
        'completed_monitored_rounds': len(report['rounds']), 'planned_rounds': 40,
        'last_logged_progress': last_progress,
        'execution_count_scope': 'completed monitored rounds only; interrupted-round totals unknown',
        'inference_checkpoint_only': True,
        'optimizer_replay_rng_resume_state_saved': False,
        'best_checkpoint_independent_replay_verified': False,
        'dataset_manifests_match_completed_stage1_at_archive': True,
    }
    write(directory / 'paused_report.json', report)
    _csv_history(report, directory / 'round_metrics.csv')
    partial[label] = report
    stage2[label] = {
        'status': 'PAUSED_BY_USER', 'hyperparameters': report['hyperparameters'],
        'completed_monitored_rounds': len(report['rounds']), 'planned_rounds': 40,
        'best_checkpoint': report['best_checkpoint'],
        'first_qualification': report['first_qualification'], 'eligible_rounds': report['eligible_rounds'],
        'last_completed_round': report['rounds'][-1],
        'execution_counts_at_last_completed_round': report['execution_counts'],
        'pause_metadata': report['pause_metadata'], 'test_payloads_opened': 0,
    }
    shutil.copy2(directory / 'paused_report.json', archive / f'stage2_{label}_paused.json')
    shutil.copy2(directory / 'round_metrics.csv', archive / f'stage2_{label}_rounds.csv')
stage2['C'] = {'status': 'NOT_RUN', 'cadence': 'replay8', 'completed_monitored_rounds': 0}
_plots(partial, figures, 'stage2_partial')

summary = {
    'status': 'PAUSED_BY_USER', 'archive_timestamp_utc': datetime.now(timezone.utc).isoformat(),
    'source_commit': stage1['beta_500']['source_commit'],
    'stage1': {label: compact_report(report) for label, report in stage1.items()},
    'chosen_beta_label': 'beta_500', 'stage2': stage2,
    'stage2_status': 'PARTIAL_PAUSED_BY_USER', 'completed_three_way_cadence_comparison': False,
    'training_restarted_during_archive': False, 'additional_model_evaluation_during_archive': False,
    'test_payloads_opened': 0, 'manifest_sha256_at_archive': current_manifests,
    'counter_scope': 'canonical completed-round counters; aborted and interrupted work kept separately in raw logs',
    'checkpoint_scope': 'qualified online-network weights only; no optimizer/replay/RNG continuation checkpoints',
}
write(root / 'study_summary.json', summary)
write(archive / 'study_summary.json', summary)

# Preserve every existing output, including interruption/resource-failure evidence.
raw = archive / 'raw'
source_files = sorted(p for p in root.rglob('*') if p.is_file())
inventory = []
for source in source_files:
    relative = source.relative_to(root)
    destination = raw / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    assert hashlib.sha256(destination.read_bytes()).hexdigest() == digest
    inventory.append({'path': relative.as_posix(), 'bytes': source.stat().st_size, 'sha256': digest})
write(archive / 'raw_archive_inventory.json', {
    'files': inventory, 'file_count': len(inventory),
    'total_bytes': sum(row['bytes'] for row in inventory), 'all_copies_sha256_verified': True,
})
print(json.dumps({'archived_files': len(inventory), 'megabytes': round(sum(
    row['bytes'] for row in inventory) / 1e6, 2), 'stage2_rounds': {
    k: v['completed_monitored_rounds'] for k, v in stage2.items()},
    'Test': 0, 'manifest_hashes_match': True}, ensure_ascii=False))
