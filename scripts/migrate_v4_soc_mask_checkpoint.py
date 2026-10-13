"""Fork a complete v4 round checkpoint for the single strict SOC-mask repair.

This does not train or open route data. The original output is read-only. Run
with --source-output OLD --output-dir NEW_EMPTY --experiment-id NEW_ID, then
resume feedback_study using NEW_EMPTY/training_state_latest.pt. Historical
rounds were produced by the original mask; only future bootstrap masks and
future execution use the strict contract. No historical rewards are repaired.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

import torch
from v3.control import AccountState, EconomicMPC
from v4.control import ACTION_KW, feasible_fc_actions
from v4.dqn import ReplayItem
from v4.experiment_paths import unarchived_output_path

OLD_MASK_LINE = b'        if SOC_MIN - 1e-9 <= soc <= SOC_MAX + 1e-9:\n'
STRICT_MASK_LINE = b'        if SOC_MIN <= soc <= SOC_MAX:\n'
CONTRACT = 'soc-mask-strict-v1'
SOURCE_FILES = ('training_state_latest.pt', 'run_metadata.json', 'report.json', 'round_history.json')
LOAD_LIMITATION = (
    'The original measured load is not stored in ReplayItem. Reconstruct current '
    'load as float(next_state[4]) * 600 using the stored seven-feature state; '
    'division/multiplication can change the last floating-point bits at a physical '
    'boundary. Intersect with the original mask only: never add candidates. Abort '
    'if a nonterminal intersection is empty. No route payload is opened.'
)


def source_tree_hashes(source_root: Path) -> tuple[str, str]:
    """Use the trainer's CRLF-normalized hash, and invert exactly one repaired line."""
    current, reconstructed = hashlib.sha256(), hashlib.sha256()
    control_seen = False
    for path in sorted((source_root / 'src').rglob('*')):
        if not path.is_file() or path.suffix not in ('.py', '.json'):
            continue
        relative = path.relative_to(source_root).as_posix()
        data = path.read_bytes().replace(b'\r\n', b'\n')
        prior = data
        if relative == 'src/v4/control.py':
            if data.count(STRICT_MASK_LINE) != 1 or OLD_MASK_LINE in data:
                raise ValueError('control.py must contain exactly one strict SOC-mask line')
            prior = data.replace(STRICT_MASK_LINE, OLD_MASK_LINE, 1)
            control_seen = True
        prefix = relative.encode('utf-8') + b'\0'
        current.update(prefix + data + b'\0')
        reconstructed.update(prefix + prior + b'\0')
    if not control_seen:
        raise ValueError('source tree lacks src/v4/control.py')
    return current.hexdigest(), reconstructed.hexdigest()


def verify_source_contract(source_root: Path, metadata: dict, current_commit: str) -> tuple[str, str]:
    if current_commit != metadata['source_commit']:
        raise ValueError('current source_commit differs from the saved checkpoint')
    current, reconstructed = source_tree_hashes(source_root)
    if reconstructed != metadata['source_tree_sha256']:
        raise ValueError('current src must differ from saved src by only the strict SOC-mask line')
    return current, reconstructed


def migrate_replay_state(state: dict, accountant: EconomicMPC) -> tuple[dict, dict]:
    """Replace masks only; persisted pool IDs restore references to the new items."""
    replay, ids = state['replay'], state['replay_ids']
    failure, ordinary = state['failure_pool_ids'], state['ordinary_pool_ids']
    if (state.get('format_version') != 2 or not isinstance(replay, tuple)
            or len(replay) != len(ids) or len(set(ids)) != len(ids)
            or len(failure) + len(ordinary) != len(ids)
            or len(set(failure)) != len(failure) or len(set(ordinary)) != len(ordinary)
            or set(failure).intersection(ordinary) or set(failure).union(ordinary) != set(ids)):
        raise ValueError('unsupported or inconsistent replay IDs and sampling pools')
    counts = {'replay_items': len(replay), 'nonterminal_items': 0, 'terminal_items': 0,
              'narrowed_nonterminal_masks': 0, 'removed_action_indices': 0,
              'added_action_indices': 0}

    @lru_cache(maxsize=8192)
    def strict_indices(soc: float, load_feature: float) -> frozenset[int]:
        actions = feasible_fc_actions(AccountState(soc=soc), load_feature * 600., accountant)
        return frozenset(power // 10 for power in actions)

    migrated = []
    for ordinal, (entry_id, item) in enumerate(zip(ids, replay), 1):
        if ordinal > 1 and ordinal % 50_000 == 1:
            print(f'MIGRATION replay_checked={ordinal - 1}/{len(replay)} '
                  f'narrowed_masks={counts["narrowed_nonterminal_masks"]} '
                  f'removed_actions={counts["removed_action_indices"]}', flush=True)
        if not isinstance(item, ReplayItem) or item.replay_id != entry_id or type(item.done) is not bool:
            raise ValueError(f'inconsistent ReplayItem at replay ID {entry_id}')
        old = item.next_feasible_indices
        if (not isinstance(old, tuple) or len(set(old)) != len(old)
                or any(type(index) is not int or not 0 <= index < len(ACTION_KW) for index in old)):
            raise ValueError(f'invalid saved mask at replay ID {entry_id}')
        if item.done:
            if old:
                raise ValueError(f'terminal replay mask must be empty: replay ID {entry_id}')
            counts['terminal_items'] += 1
            migrated.append(item)
            continue
        if len(item.next_state) != 7 or not all(math.isfinite(value) for value in item.next_state):
            raise ValueError(f'nonterminal next_state must have seven finite features: {entry_id}')
        strict = strict_indices(float(item.next_state[0]), float(item.next_state[4]))
        narrowed = tuple(index for index in old if index in strict)
        if not narrowed:
            raise ValueError(f'empty strict nonterminal mask at replay ID {entry_id}; migration aborted')
        counts['nonterminal_items'] += 1
        if narrowed != old:
            counts['narrowed_nonterminal_masks'] += 1
            counts['removed_action_indices'] += len(old) - len(narrowed)
            item = replace(item, next_feasible_indices=narrowed)
        migrated.append(item)
    # Online/Target/Adam/RNG/counters and pool ID sequences remain the same objects.
    return state | {'replay': tuple(migrated)}, counts


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def require_no_source_writer(source: Path) -> None:
    """Conservative Windows check for Python processes using this output argument."""
    if os.name != 'nt':
        raise ValueError('this narrowly scoped migration requires Windows process preflight')
    command = (
        'Get-CimInstance Win32_Process | Where-Object { $_.Name -match "^(python|pythonw|py).*\\.exe$" } '
        '| Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress'
    )
    result = subprocess.run(['powershell', '-NoProfile', '-NonInteractive', '-Command', command],
                            check=True, capture_output=True, text=True, timeout=30)
    processes = json.loads(result.stdout.strip() or '[]')
    if isinstance(processes, dict):
        processes = [processes]
    for process in processes:
        text = (process.get('CommandLine') or '').replace('\\', '/').casefold()
        if (process['ProcessId'] != os.getpid() and '--output-dir' in text
                and (source.name.casefold() in text or source.as_posix().casefold() in text)):
            raise ValueError(f'Python process {process["ProcessId"]} may write the source output; stop it first')


def validate_complete_boundary(checkpoint: dict, metadata: dict, report: dict, history: dict,
                               source: Path) -> int:
    if checkpoint.get('format_version') != 1 or checkpoint.get('metadata') != metadata:
        raise ValueError('run_metadata must exactly equal complete-round checkpoint metadata')
    saved = checkpoint['report']
    completed = checkpoint['completed_round']
    if (type(completed) is not int or not 0 < completed < metadata['hyperparameters']['rounds']
            or report != history or saved['rounds'] != report['rounds']
            or len(saved['rounds']) != completed
            or [row['round'] for row in saved['rounds']] != list(range(1, completed + 1))):
        raise ValueError('checkpoint, report and round_history must agree at the last complete round')
    for document in (saved, report):
        if any(document.get(key) != value for key, value in metadata.items()):
            raise ValueError('report metadata differs from run_metadata')
        if document.get('test_payloads_opened') != 0:
            raise ValueError('source report must record zero Test payloads opened')
    if (checkpoint.get('best_checkpoint_bytes') is not None or saved.get('best_checkpoint') is not None
            or report.get('best_checkpoint') is not None or (source / 'best_agent.pt').exists()):
        raise ValueError('migration of a selected best checkpoint is outside this narrow tool')
    parameters = metadata['hyperparameters']
    if (parameters['state_dim'] != 7 or parameters['action_kw'] != list(ACTION_KW)
            or parameters.get('economic_replay_format_version') != 2):
        raise ValueError('only the saved seven-feature, 61-action v4 replay contract is supported')
    return completed


def link_historical_artifacts(report: dict, source: Path, output: Path) -> dict:
    """Relocate JSON path strings without reading or copying full trajectories."""
    def link(value: str) -> str:
        path = (source / value).resolve()
        if not path.is_relative_to(source) or not path.is_file():
            raise ValueError(f'historical artifact is missing or outside source output: {value}')
        return Path(os.path.relpath(path, output)).as_posix()

    def walk(value):
        if isinstance(value, dict):
            linked = {}
            for key, item in value.items():
                if key == 'trajectory_file' and isinstance(item, str):
                    # Old traces remain provenance links. The plotter only reads this run's files.
                    linked[key] = None
                    linked['historical_trajectory_file'] = link(item)
                elif key in ('checkpoint', 'decisions') and isinstance(item, str):
                    linked[key] = link(item)
                else:
                    linked[key] = walk(item)
            return linked
        if isinstance(value, list):
            return [walk(item) for item in value]
        return value

    return walk(report)


def write_json(path: Path, document: dict) -> None:
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(document, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-output', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--experiment-id', required=True)
    args = parser.parse_args(argv)
    started = time.monotonic()
    source = args.source_output.resolve()
    output = unarchived_output_path(args.output_dir, require_empty=True)
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('new output must be separate from the original output tree')
    if (not args.experiment_id.strip() or len(args.experiment_id) > 128
            or any(character in args.experiment_id for character in '\r\n')):
        raise ValueError('experiment-id must be a nonempty single line of at most 128 characters')
    require_no_source_writer(source)
    source_digests = {name: file_sha256(source / name) for name in SOURCE_FILES}
    metadata = json.loads((source / 'run_metadata.json').read_text(encoding='utf-8'))
    if args.experiment_id == metadata['experiment_id']:
        raise ValueError('migration requires a new experiment-id')
    current_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    current_hash, original_hash = verify_source_contract(ROOT, metadata, current_commit)
    # Load once. Do not construct an agent, reseed or load formal dataset payloads.
    checkpoint = torch.load(source / 'training_state_latest.pt', map_location='cpu', weights_only=False)
    report = json.loads((source / 'report.json').read_text(encoding='utf-8'))
    history = json.loads((source / 'round_history.json').read_text(encoding='utf-8'))
    completed = validate_complete_boundary(checkpoint, metadata, report, history, source)
    print(f'MIGRATION complete_round={completed} replay_items={len(checkpoint["agent"]["replay"])}', flush=True)
    migrated_agent, counts = migrate_replay_state(checkpoint['agent'], EconomicMPC(nominal_cost_cny=1.))
    lineage = {
        'contract': CONTRACT, 'source_output_dir': str(source),
        'source_checkpoint': str(source / 'training_state_latest.pt'),
        'source_checkpoint_sha256': source_digests['training_state_latest.pt'],
        'source_file_sha256': source_digests, 'original_experiment_id': metadata['experiment_id'],
        'original_source_tree_sha256': original_hash, 'current_source_tree_sha256': current_hash,
        'completed_rounds': completed, 'resume_from_round': completed + 1,
        'historical_rounds_use_original_permissive_mask': True,
        'historical_trajectory_links': 'historical_trajectory_file links to preserved original files; '
                                     'historical trajectory_file is null to retain the plotter output-directory guard',
        'replay_mask_migration': counts, 'normalized_load_reconstruction_limitation': LOAD_LIMITATION,
    }
    new_metadata = metadata | {
        'source_tree_sha256': current_hash, 'experiment_id': args.experiment_id,
        'initialization': f'resume original complete round {completed}; strict SOC-mask migration; no new initialization',
        'resume_lineage': lineage,
    }
    new_report = link_historical_artifacts(checkpoint['report'], source, output) | new_metadata
    new_checkpoint = checkpoint | {'metadata': new_metadata, 'report': new_report, 'agent': migrated_agent}
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.soc-mask-migration-', dir=output.parent))
    published = False
    try:
        write_json(stage / 'run_metadata.json', new_metadata)
        write_json(stage / 'report.json', new_report)
        write_json(stage / 'round_history.json', new_report)
        write_json(stage / 'checkpoint_migration.json', lineage)
        temporary = stage / 'training_state_latest.pt.tmp'
        with temporary.open('xb') as handle:
            torch.save(new_checkpoint, handle)
            handle.flush()
            os.fsync(handle.fileno())
        require_no_source_writer(source)
        final_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        if ({name: file_sha256(source / name) for name in SOURCE_FILES} != source_digests
                or verify_source_contract(ROOT, metadata, final_commit)[0] != current_hash):
            raise ValueError('source output or runtime source changed during migration')
        temporary.rename(stage / 'training_state_latest.pt')
        # Publish the complete set as one directory rename; a failed save has no final checkpoint.
        if output.exists():
            if not output.is_dir() or any(output.iterdir()):
                raise FileExistsError('new output became nonempty during migration')
            output.rmdir()
        stage.rename(output)
        published = True
    finally:
        if not published and stage.exists():
            if stage.resolve().parent != output.parent.resolve() or not stage.name.startswith('.soc-mask-migration-'):
                raise ValueError('refusing cleanup outside migration staging directory')
            shutil.rmtree(stage)
    print(json.dumps({'output_dir': str(output), 'experiment_id': args.experiment_id,
                      'resume_round': completed + 1, 'migration': counts,
                      'elapsed_seconds': round(time.monotonic() - started, 3)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
