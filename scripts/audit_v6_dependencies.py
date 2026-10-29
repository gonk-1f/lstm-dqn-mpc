"""Static source/dependency inventory; never imports project code or opens data.

Only Python/source entrypoints and dependency configuration are read. Historical
docs/results source snapshots are named but not opened. Git worktree checks read
status/path names only. Output is confined to this checkout's docs/audits tree.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict, deque
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs/audits/v6_optuna_20261028"
ARCHIVES = {
    "tests/test_v2_history_study_cli.py": "tests/history/test_v2_history_study_cli.py",
    "scripts/v4_per_frozen_probe.py": "scripts/history/v4_per_frozen_probe.py",
    "scripts/v4_per_jointfit_state_probe.py": "scripts/history/v4_per_jointfit_state_probe.py",
}
CONFIG_NAMES = {
    "pyproject.toml", "setup.py", "setup.cfg", "pytest.ini", "tox.ini",
    "Pipfile", "Pipfile.lock", "poetry.lock", "uv.lock", "environment.yml",
    "environment.yaml", "requirements.txt", "requirements-sb3.txt",
    "requirements-optuna.txt",
}
EDGE_FIELDS = ("source", "line", "kind", "reference", "target", "resolution", "detail")


def git(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.check_output(
        ["git", "-C", str(cwd), *args], encoding="utf-8", errors="replace"
    )


def is_config(path: str) -> bool:
    name = Path(path).name
    return name in CONFIG_NAMES or (name.startswith("requirements") and name.endswith(".txt"))


def is_entry(path: str) -> bool:
    return (path.endswith(".py") or is_config(path) or
            (path.startswith("scripts/") and Path(path).suffix in {".ps1", ".bat", ".sh"}))


def worktree_snapshot() -> list[dict]:
    snapshots = []
    for block in git("worktree", "list", "--porcelain").strip().split("\n\n"):
        fields = dict(line.split(" ", 1) for line in block.splitlines() if " " in line)
        worktree = Path(fields["worktree"])
        tokens = git("status", "--porcelain=v1", "-z", "--untracked-files=all", cwd=worktree).split("\0")
        changes, position = [], 0
        while position < len(tokens) and tokens[position]:
            token = tokens[position]
            row = {"status": token[:2], "path": token[3:]}
            position += 1
            if "R" in row["status"] or "C" in row["status"]:
                row["original_path"] = tokens[position]
                position += 1
            changes.append(row)
        protected = [row["path"] for row in changes
                     if "ddpg" in row["path"].lower() or "td3" in row["path"].lower()]
        snapshots.append(dict(path=worktree.as_posix(), head=fields.get("HEAD"),
            branch=fields.get("branch", "detached"), changes=changes,
            changed_count=len(changes), untracked_count=sum(r["status"] == "??" for r in changes),
            protected_ddpg_td3_paths=protected, content_opened=False))
    return snapshots


def module_aliases(paths: list[str]) -> dict[str, str]:
    aliases = {}
    for path in paths:
        if not path.endswith(".py"):
            continue
        name = path[:-3].replace("/", ".")
        if name.endswith(".__init__"):
            name = name[:-len(".__init__")]
        aliases[name] = path
        if path.startswith("src/"):
            aliases[name[4:]] = path
        if path.startswith("src/main/"):
            aliases[name[len("src.main."):]] = path
        if path.startswith(("tests/", "scripts/")):
            aliases[Path(path).stem] = path
    return aliases


def scan_python(path: str, source: str, aliases: dict[str, str], available: set[str]) -> tuple[list[dict], list[dict], bool]:
    tree = ast.parse(source, filename=path)
    edges, dynamic = [], []
    current = path.removeprefix("src/")[:-3].replace("/", ".")
    package = current.removesuffix(".__init__") if path.endswith("/__init__.py") else current.rpartition(".")[0]

    def add_module(module: str, line: int, kind: str, detail: str = "") -> None:
        if not module:
            return
        target = aliases.get(module, "")
        namespace = any(name.startswith(module + ".") for name in aliases)
        resolution = "local" if target else ("local_namespace" if namespace else
            ("stdlib" if module.split(".")[0] in sys.stdlib_module_names else "external_or_unresolved"))
        edges.append(dict(source=path, line=line, kind=kind, reference=module,
                          target=target, resolution=resolution, detail=detail))
        if target or namespace:
            pieces = module.split(".")
            for length in range(1, len(pieces)):
                parent = aliases.get(".".join(pieces[:length]))
                if parent and parent.endswith("/__init__.py"):
                    edges.append(dict(source=path, line=line, kind="package_init",
                        reference=".".join(pieces[:length]), target=parent, resolution="local", detail=detail))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                add_module(alias.name, node.lineno, "ast_import")
        elif isinstance(node, ast.ImportFrom):
            prefix = package.split(".") if package else []
            if node.level:
                prefix = prefix[:len(prefix) - node.level + 1]
                module = ".".join([*prefix, *([node.module] if node.module else [])])
            else:
                module = node.module or ""
            add_module(module, node.lineno, "ast_from_import")
            for alias in node.names:
                candidate = f"{module}.{alias.name}" if module else alias.name
                if candidate in aliases:
                    add_module(candidate, node.lineno, "ast_imported_submodule")
        elif isinstance(node, ast.Call):
            call = ast.unparse(node.func)
            if call.split(".")[-1] in {"__import__", "import_module", "spec_from_file_location", "run_module", "run_path", "Popen", "run", "call", "check_call", "check_output", "exec", "eval"}:
                expression = ast.unparse(node)
                dynamic.append(dict(source=path, line=node.lineno, call=call,
                    expression=expression[:700], resolution="manual_review_required"))
                if call.split(".")[-1] in {"__import__", "import_module", "run_module"} and node.args:
                    first = node.args[0]
                    if isinstance(first, ast.Constant) and isinstance(first.value, str):
                        module = first.value
                        if module.startswith("."):
                            level = len(module) - len(module.lstrip("."))
                            parent = package.split(".")[:len(package.split(".")) - level + 1]
                            module = ".".join([*parent, module.lstrip(".")]).rstrip(".")
                        add_module(module, node.lineno, "dynamic_literal_import", call)
        elif isinstance(node, (ast.List, ast.Tuple)):
            for position, value in enumerate(node.elts[:-1]):
                following = node.elts[position + 1]
                if (isinstance(value, ast.Constant) and value.value == "-m" and
                        isinstance(following, ast.Constant) and isinstance(following.value, str)):
                    add_module(following.value, node.lineno, "literal_module_argument_vector",
                               "Literal -m argument; command execution is not proven.")
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            literal = node.value.replace("\\", "/")
            candidates = set(re.findall(r"(?:src|scripts|tests)/[A-Za-z0-9_./-]+\.py", literal))
            if literal.endswith(".py") and "\n" not in literal and " " not in literal:
                candidates.update((literal, (Path(path).parent / literal).as_posix()))
            for candidate in sorted(candidates & available):
                edges.append(dict(source=path, line=node.lineno, kind="literal_script_reference",
                    reference=literal[:250], target=candidate, resolution="local_literal_candidate",
                    detail="String/path reference; execution is not proven."))
            for module in re.findall(r"\bpython(?:\.exe)?\s+(?:-\w+\s+)*-m\s+([A-Za-z_]\w*(?:\.\w+)*)", literal):
                add_module(module, node.lineno, "literal_module_command", "Literal command; execution is not proven.")
    main_guard = any(isinstance(node, ast.If) and "__name__" in ast.unparse(node.test)
                     and "__main__" in ast.unparse(node.test) for node in tree.body)
    return edges, dynamic, main_guard


def scan_script(path: str, source: str, aliases: dict[str, str], available: set[str]) -> list[dict]:
    edges = []
    for number, line in enumerate(source.splitlines(), 1):
        modules = re.findall(r"['\"]-m['\"]\s*,\s*['\"]([\w.]+)['\"]|\s-m\s+([\w.]+)", line)
        for match in modules:
            module = next(value for value in match if value)
            edges.append(dict(source=path, line=number, kind="shell_module_command", reference=module,
                target=aliases.get(module, ""), resolution="local" if module in aliases else "external_or_unresolved", detail=line.strip()))
        for literal in re.findall(r"['\"]([^'\"]+\.py)['\"]", line):
            normalized = literal.replace("\\", "/")
            for candidate in (normalized, (Path(path).parent / normalized).as_posix()):
                if candidate in available:
                    edges.append(dict(source=path, line=number, kind="shell_script_reference", reference=literal,
                        target=candidate, resolution="local_literal_candidate", detail=line.strip()))
    return edges


def closure(roots: list[str], edges: list[dict]) -> set[str]:
    links = defaultdict(set)
    for edge in edges:
        if edge["target"]:
            links[edge["source"]].add(edge["target"])
    reached, queue = set(roots), deque(roots)
    while queue:
        for target in links[queue.popleft()] - reached:
            reached.add(target)
            queue.append(target)
    return reached


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(phase: str, output: Path) -> dict:
    output = output.resolve()
    if not output.is_relative_to((ROOT / "docs/audits").resolve()):
        raise ValueError("Audit outputs must stay inside this checkout's docs/audits directory")
    output.mkdir(parents=True, exist_ok=True)
    tracked = sorted(filter(None, git("ls-files", "-z").split("\0")))
    untracked = sorted(filter(None, git("ls-files", "--others", "--exclude-standard", "-z", "--", "src", "scripts", "tests", "requirements*.txt", "pytest.ini").split("\0")))
    artifact_paths = [p for p in tracked if p.startswith("docs/results/") and is_entry(p)]
    tracked_entries = [p for p in tracked if is_entry(p) and not p.startswith("docs/results/")]
    extra_entries = [p for p in untracked if is_entry(p) and not p.startswith("docs/results/")]
    existing = [p for p in sorted(set(tracked_entries + extra_entries)) if (ROOT / p).is_file()]
    available = set(existing)
    aliases = module_aliases(existing)
    edges, dynamics, errors, entrypoints, configs = [], [], [], [], []
    for path in existing:
        source = (ROOT / path).read_text(encoding="utf-8-sig")
        if path.endswith(".py"):
            try:
                found, dynamic, main = scan_python(path, source, aliases, available)
                edges.extend(found)
                dynamics.extend(dynamic)
                if main:
                    entrypoints.append(path)
            except SyntaxError as error:
                errors.append(dict(path=path, line=error.lineno, message=error.msg))
        elif is_config(path):
            configs.append(dict(path=path, entries=[line.strip() for line in source.splitlines()
                            if line.strip() and not line.lstrip().startswith("#")]))
        else:
            entrypoints.append(path)
            edges.extend(scan_script(path, source, aliases, available))
    unique = {tuple(row[key] for key in EDGE_FIELDS): row for row in edges}
    edges = sorted(unique.values(), key=lambda row: (row["source"], row["line"], row["kind"], row["reference"], row["target"]))
    roots = sorted(p for p in existing if p.startswith("src/v6_sb3/"))
    roots.extend(p for p in ("scripts/start_v6_sb3.ps1", "scripts/build_v6_dataset.py",
                            "scripts/report_v6.py", "scripts/finish_v6_report.ps1") if p in available)
    reached = closure(roots, edges)
    incoming = Counter(row["target"] for row in edges if row["target"])
    files = []
    for path in existing:
        if path in reached:
            category, reason = "v6_production_closure", "Conservative static reachability from current V6 entrypoints."
        elif path.startswith("tests/history/"):
            category, reason = "historical_experiment", "Approved reversible archive; excluded from default pytest."
        elif path.startswith("tests/"):
            category, reason = "current_regression", "Retained regression coverage; historical shared contracts are intentionally preserved."
        elif path.startswith("scripts/history/") or path in ARCHIVES or path.startswith("src/main/train_") or path.startswith("src/main/run_dqn_"):
            category, reason = "historical_experiment", "Historical DQN experiment/training entrypoint; keep dependencies and source."
        elif path == "scripts/audit_v6_dependencies.py":
            category, reason = "audit_tool", "Static inspection only; no project imports or payload reads."
        elif is_config(path):
            category, reason = "dependency_entry", "Dependency/test configuration is retained."
        else:
            category, reason = "unused_unconfirmed", "Outside selected V6 closure does not prove unused; retained without deletion."
        files.append(dict(path=path, tracked=path in tracked, category=category, reason=reason,
                          local_incoming_edges=incoming[path], has_main_guard=path in entrypoints))
    snapshots = worktree_snapshot()
    metadata = dict(phase=phase, captured_at=datetime.now(timezone.utc).isoformat(),
        root=str(ROOT), head=git("rev-parse", "HEAD").strip(), branch=git("branch", "--show-current").strip(),
        tracked_python_count=sum(p.endswith(".py") for p in tracked),
        scanned_python_count=sum(p.endswith(".py") for p in existing),
        tracked_python_scanned_count=sum(p.endswith(".py") and p in tracked for p in existing),
        historical_artifact_entry_count=len(artifact_paths), files=files,
        historical_artifacts=[dict(path=p, content_opened=False, action="retain_immutable") for p in artifact_paths],
        missing_tracked_entries=[p for p in tracked_entries if p not in available],
        untracked_source_entries=extra_entries, source_entrypoints=sorted(entrypoints),
        v6_roots=sorted(set(roots)), v6_closure=sorted(reached),
        category_counts=dict(Counter(row["category"] for row in files)),
        edge_count=len(edges), edge_kind_counts=dict(Counter(row["kind"] for row in edges)),
        dynamic_calls=dynamics, parse_errors=errors, dependency_entries=configs,
        archive_mapping=ARCHIVES, deletion_count=0,
        limitations=[
            "All AST imports, including function-local, optional and TYPE_CHECKING imports, are conservative edges; reachability does not prove execution.",
            "Literal script paths and -m commands are candidate dynamic edges; computed strings, aliases, plugins, exec/eval and environment-dependent sys.path need manual review.",
            "Package aliases include src, src/main legacy and tests conventions; namespace collisions and runtime monkeypatching can differ.",
            "No project module is imported; no dataset/Test payload, checkpoint, replay or result contents are opened; no large hashes are calculated.",
            "docs/results source/config snapshots are inventory-only immutable artifacts, excluded from AST parsing.",
            "Untracked code in this checkout is inventoried separately to include changes in progress; untracked contents of other worktrees are never opened.",
            "No-incoming-edge or outside-V6-closure status is not evidence that a file is safe to delete.",
            "Regression execution and tiny smoke workloads are coordinated by the main agent; this utility performs static inspection only.",
        ])
    write_json(output / f"inventory_{phase}.json", metadata)
    write_json(output / f"dependency_edges_{phase}.json", edges)
    write_json(output / f"worktrees_{phase}.json", snapshots)
    with (output / f"dependency_edges_{phase}.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=EDGE_FIELDS)
        writer.writeheader()
        writer.writerows(edges)
    with (output / f"file_inventory_{phase}.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("path", "tracked", "category", "reason", "local_incoming_edges", "has_main_guard"))
        writer.writeheader()
        writer.writerows(files)
    summary = {key: metadata[key] for key in ("phase", "tracked_python_count", "scanned_python_count", "tracked_python_scanned_count", "historical_artifact_entry_count", "category_counts", "edge_count", "parse_errors")}
    summary["worktrees"] = [dict(path=row["path"], changed_count=row["changed_count"],
        untracked_count=row["untracked_count"], protected_ddpg_td3_count=len(row["protected_ddpg_td3_paths"])) for row in snapshots]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("before", "after"), required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    run(arguments.phase, arguments.output_dir)
