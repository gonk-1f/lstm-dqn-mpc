from __future__ import annotations

from pathlib import Path
from typing import Any


DEFAULT_CONFIG_FILES = [
    "configs/base.yaml",
    "configs/ship_system.yaml",
    "configs/mpc.yaml",
    "configs/dqn.yaml",
    "configs/plotting.yaml",
    "configs/timing.yaml",
]

FALLBACK_PROJECT_CONFIG: dict[str, Any] = {
    "ship": {
        "topology": "dual_side_hydrogen_hybrid",
        "sides": ["left", "right"],
        "environment_type": "vessel_microgrid",
        "original_microgrid_buy_sell_logic_reused": False,
        "controlled_sources": [
            "left_fuel_cell_group",
            "right_fuel_cell_group",
            "left_battery_cluster",
            "right_battery_cluster",
        ],
        "observed_devices": [
            "left_inverter",
            "right_inverter",
            "dcdc_converters",
            "propulsion_speed",
            "propulsion_load",
        ],
    }
}


def _deep_merge(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in incoming.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return ""
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    if value.startswith("[") and value.endswith("]"):
        items = value[1:-1].strip()
        if not items:
            return []
        return [_parse_scalar(item) for item in items.split(",")]
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _load_simple_yaml_file(path: Path) -> dict[str, Any]:
    """Small fallback parser for the project's simple config YAML files."""

    entries: list[tuple[int, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            if not raw_line.strip() or raw_line.lstrip().startswith("#"):
                continue
            indent = len(raw_line) - len(raw_line.lstrip(" "))
            entries.append((indent, raw_line.strip()))

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any] | list[Any]]] = [(-1, root)]
    for idx, (indent, content) in enumerate(entries):
        while indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if content.startswith("- "):
            if not isinstance(parent, list):
                raise ValueError(f"YAML list item has no list parent in {path}: {content}")
            parent.append(_parse_scalar(content[2:]))
            continue

        key, sep, value = content.partition(":")
        if not sep:
            raise ValueError(f"Unsupported YAML line in {path}: {content}")
        key = key.strip()
        value = value.strip()
        if value:
            if not isinstance(parent, dict):
                raise ValueError(f"YAML key/value has no mapping parent in {path}: {content}")
            parent[key] = _parse_scalar(value)
            continue

        next_is_list = idx + 1 < len(entries) and entries[idx + 1][0] > indent and entries[idx + 1][1].startswith("- ")
        child: dict[str, Any] | list[Any] = [] if next_is_list else {}
        if not isinstance(parent, dict):
            raise ValueError(f"Nested YAML mapping has no mapping parent in {path}: {content}")
        parent[key] = child
        stack.append((indent, child))
    return root


def _load_yaml_file(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError:
        return _load_simple_yaml_file(path)

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_project_config(project_root: str | Path, config_files: list[str] | None = None) -> dict[str, Any]:
    root = Path(project_root)
    config_files = config_files or DEFAULT_CONFIG_FILES
    merged: dict[str, Any] = dict(FALLBACK_PROJECT_CONFIG)
    for rel_path in config_files:
        path = root / rel_path
        if path.exists():
            merged = _deep_merge(merged, _load_yaml_file(path))
    return merged


def get_project_root(current_file: str | Path) -> Path:
    path = Path(current_file).resolve()
    for parent in [path, *path.parents]:
        if (parent / "src").exists() and (parent / "configs").exists():
            return parent
    raise FileNotFoundError("Could not infer project root from current file path.")
