"""Losslessly archive finished-run trajectories, or inspect JSON/gzip by sample.

Examples (from the repository root):
  python scripts/v4_trajectory_archive.py inventory --run-dir outputs/RUN
  python scripts/v4_trajectory_archive.py compress --run-dir outputs/RUN --remove-originals
  python scripts/v4_trajectory_archive.py inspect outputs/RUN/round_100_train_trajectories.json --sample-id parent_015_seg01

Only immediate round_*_trajectories.json files are compressed. Checkpoints,
datasets and other directories are never removed. Compression defaults to
keeping originals; explicit --remove-originals requires byte-identical SHA256
verification of both the archive and the unchanged original first.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from v4.trajectory_io import iter_trajectory_records

PATTERN = re.compile(r'round_\d+_(?:exploratory|train|validation)_trajectories\.json(?:\.gz)?\Z')
MANIFEST = 'trajectory_compression_manifest.json'
BLOCK_BYTES = 1024 * 1024


def _safe_path(path: Path) -> Path:
    """Reject symlinks and Windows junctions before resolving any path."""
    path = Path(os.path.abspath(path))
    for item in (*reversed(path.parents), path):
        try:
            info = item.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_reparse_tag', 0) in (
                0xA0000003, 0xA000000C):
            raise ValueError(f'refusing symlink/junction: {item}')
    return path


def _run_root(path: Path) -> Path:
    root = _safe_path(path)
    metadata = _safe_path(root / 'run_metadata.json')
    if not root.is_dir() or not metadata.is_file():
        raise ValueError('explicit run directory containing run_metadata.json is required')
    return root


def _ensure_no_active_training(root: Path) -> None:
    if os.name != 'nt':
        raise RuntimeError('active-training check supports Windows only; refusing compression')
    command = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe' OR Name='pythonw.exe'\" "
               "| Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress")
    result = subprocess.run(['powershell', '-NoProfile', '-NonInteractive', '-Command', command],
                            capture_output=True, text=True, check=True)
    processes = json.loads(result.stdout) if result.stdout.strip() else []
    if isinstance(processes, dict):
        processes = [processes]
    for process in processes:
        command_line = process.get('CommandLine') or ''
        if 'v4.feedback_study' in command_line and root.name.casefold() in command_line.casefold():
            raise RuntimeError(f'run may still be active (PID {process["ProcessId"]}); refusing compression')


def inventory(root: Path) -> list[dict]:
    names = sorted({path.name.removesuffix('.gz') for path in root.iterdir()
                    if PATTERN.fullmatch(path.name)})
    rows = []
    for name in names:
        source, target = _safe_path(root / name), _safe_path(root / (name + '.gz'))
        if source.parent != root or target.parent != root:
            raise ValueError('trajectory must belong directly to the explicit run directory')
        for path in (source, target):
            if path.exists() and not path.is_file():
                raise ValueError(f'expected a regular file: {path}')
        rows.append(dict(file=name, original_bytes=source.stat().st_size if source.exists() else None,
                         gzip_bytes=target.stat().st_size if target.exists() else None))
    return rows


def _digest(path: Path, *, compressed=False) -> tuple[str, int]:
    digest, count = hashlib.sha256(), 0
    with (gzip.open(path, 'rb') if compressed else path.open('rb')) as handle:
        for block in iter(lambda: handle.read(BLOCK_BYTES), b''):
            digest.update(block)
            count += len(block)
    return digest.hexdigest(), count


def _signature(path: Path):
    value = path.stat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _write_manifest(root: Path, manifest: dict) -> None:
    destination = _safe_path(root / MANIFEST)
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=root,
                                     prefix='.trajectory-manifest-', suffix='.tmp', delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    temporary.replace(destination)


def compress_run(root: Path, *, remove_originals=False, level=6) -> dict:
    root = _run_root(root)
    _ensure_no_active_training(root)
    rows = inventory(root)  # Validate every file boundary before mutating anything.
    manifest_path = _safe_path(root / MANIFEST)
    manifest = (json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists()
                else dict(format_version=1, run_directory=str(root), files={}))
    if manifest.get('format_version') != 1 or manifest.get('run_directory') != str(root):
        raise ValueError('archive manifest identity mismatch')
    for row in rows:
        source, target = root / row['file'], root / (row['file'] + '.gz')
        temporary = None
        if source.exists():
            before = _signature(source)
            try:
                if not target.exists():
                    with tempfile.NamedTemporaryFile('wb', dir=root, prefix='.trajectory-',
                                                     suffix='.gz.tmp', delete=False) as raw:
                        temporary = Path(raw.name)
                        with source.open('rb') as original, gzip.GzipFile(
                                filename='', mode='wb', compresslevel=level, fileobj=raw, mtime=0) as archive:
                            for block in iter(lambda: original.read(BLOCK_BYTES), b''):
                                archive.write(block)
                candidate = temporary if temporary is not None else target
                original_identity = _digest(source)
                if _digest(candidate, compressed=True) != original_identity or _signature(source) != before:
                    raise RuntimeError(f'archive differs from original or source changed: {source.name}')
                if temporary is not None:
                    # Hard-link publication is atomic and fails rather than overwriting a concurrent archive.
                    try:
                        os.link(temporary, target)
                    except FileExistsError:
                        if _digest(target, compressed=True) != original_identity:
                            raise RuntimeError(f'existing gzip differs; preserved both: {target.name}')
                    temporary.unlink()
                    temporary = None
                if _signature(source) != before:
                    raise RuntimeError(f'original changed before completion: {source.name}')
                _safe_path(target)
                archive_before = _signature(target)
                if _digest(target, compressed=True) != original_identity:
                    raise RuntimeError(f'published archive differs from original: {target.name}')
                archive_hash, archive_bytes = _digest(target)
                if _signature(target) != archive_before:
                    raise RuntimeError(f'archive changed during verification: {target.name}')
                entry = dict(original_sha256=original_identity[0], original_bytes=original_identity[1],
                             gzip_sha256=archive_hash, gzip_bytes=archive_bytes,
                             verification='sha256_and_bytes_match', original_removed=False,
                             result='verified_original_kept')
                manifest['files'][source.name] = entry
                _write_manifest(root, manifest)  # Preserve verification evidence before optional deletion.
                if remove_originals:
                    _safe_path(source)
                    _safe_path(target)
                    if _signature(source) != before:
                        raise RuntimeError(f'original changed before removal: {source.name}')
                    if _signature(target) != archive_before:
                        raise RuntimeError(f'archive changed before removal: {target.name}')
                    source.unlink()  # Only this verified file; never delete a directory or use a wildcard.
                    entry['original_removed'] = True
                    entry['result'] = 'verified_original_removed'
            finally:
                if temporary is not None and temporary.exists():
                    temporary.unlink()
        else:
            entry = manifest['files'].get(source.name)
            if entry is None:
                raise ValueError(f'archive-only file has no original verification manifest: {target.name}')
            if (_digest(target, compressed=True) != (entry['original_sha256'], entry['original_bytes'])
                    or _digest(target) != (entry['gzip_sha256'], entry['gzip_bytes'])):
                raise RuntimeError(f'previously archived file changed: {target.name}')
            entry['original_removed'] = True
            entry['result'] = 'verified_existing_archive'
        _write_manifest(root, manifest)
        print(json.dumps(dict(file=source.name, **entry)), flush=True)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    for action in ('inventory', 'compress'):
        command = actions.add_parser(action)
        command.add_argument('--run-dir', type=Path, required=True)
        if action == 'compress':
            command.add_argument('--remove-originals', action='store_true')
            command.add_argument('--level', type=int, choices=range(1, 10), default=6)
    inspect = actions.add_parser('inspect')
    inspect.add_argument('path', type=Path)
    inspect.add_argument('--sample-id')
    args = parser.parse_args()
    if args.action == 'inspect':
        for section, record in iter_trajectory_records(args.path, sample_id=args.sample_id):
            print(json.dumps(dict(section=section, sample_id=record.get('sample_id'),
                                  transitions=len(record.get('transitions', [])),
                                  completed=record.get('completed'), failure=record.get('failure')),
                             ensure_ascii=False))
        return 0
    root = _run_root(args.run_dir)
    if args.action == 'inventory':
        print(json.dumps(dict(run_directory=str(root), files=inventory(root)), ensure_ascii=False, indent=2))
    else:
        compress_run(root, remove_originals=args.remove_originals, level=args.level)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
