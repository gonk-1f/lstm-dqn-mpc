from __future__ import annotations

from pathlib import Path


def resolve_checkpoint_path(preferred_path: str | Path) -> Path:
    path = Path(preferred_path)
    if path.exists():
        return path

    parent = path.parent
    name = path.name
    if name == "last_ship_dqn.pt":
        best_path = parent / "best_ship_dqn.pt"
        if best_path.exists():
            return best_path
    if name == "best_ship_dqn.pt":
        last_path = parent / "last_ship_dqn.pt"
        if last_path.exists():
            return last_path
    return path
