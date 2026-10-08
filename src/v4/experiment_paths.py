"""Prevent training entry points from writing archived experiment evidence."""
from pathlib import Path


def unarchived_output_path(path: Path, *, require_empty: bool = False) -> Path:
    resolved = Path(path).resolve()
    parts = tuple(part.casefold() for part in resolved.parts)
    archived = ('v4_staged_beta_cadence_seed42' in parts or any(
        left == 'docs' and right in ('results','figures')
        for left,right in zip(parts,parts[1:])))
    if archived:
        raise ValueError('experiment archive paths are read-only; choose a fresh outputs directory')
    if resolved.exists() and (not resolved.is_dir() or (require_empty and any(resolved.iterdir()))):
        raise FileExistsError('training requires an empty output directory')
    return resolved
