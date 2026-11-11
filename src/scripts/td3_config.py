from __future__ import annotations

import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


def _find_project_layout() -> tuple[Path, Path]:
    """
    Return:
        PROJECT_ROOT: repository/worktree root
        SOURCE_ROOT: src directory
    """

    script_dir = Path(__file__).resolve().parent

    for candidate in (script_dir, *script_dir.parents):

        # Standard src layout:
        # project_root/src/ship_ems/env.py
        src_package = candidate / "src" / "ship_ems" / "env.py"

        if src_package.is_file():
            return candidate, candidate / "src"

        # Current candidate is the src directory:
        # project_root/src/ship_ems/env.py
        root_package = candidate / "ship_ems" / "env.py"

        if root_package.is_file():

            if candidate.name.lower() == "src":
                return candidate.parent, candidate

            return candidate, candidate

    raise RuntimeError(
        "未找到 ship_ems 包，请检查项目目录结构。"
    )


PROJECT_ROOT, SOURCE_ROOT = _find_project_layout()
for _path in (PROJECT_ROOT, SOURCE_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from ship_ems.config import CONFIG
from ship_ems.data import load_voyage_csv
from ship_ems.env import ShipEMSEnv


# ---------------------------------------------------------------------
# Fixed project settings. No CSV filenames are entered manually.
# ---------------------------------------------------------------------

DATA_ROOT = Path(
    r"C:\Users\20883\OneDrive\Desktop\lstm-dqn-mpc\lstm-dqn-mpc\.worktrees\v6-sb3-baseline\data\processed\operating_dataset_v6_1s"
)
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "td3_ship_ems_fixed_parent005_n32_500ep"

TRAIN_VOYAGE_ID = "parent_005_seg01"
EXPECTED_VALIDATION_COUNT = 8
EXPECTED_INTERVAL_S = 1.0

MAX_EPISODES = 500
REWARD_PLOT_INTERVAL = 10
TRAIN_DIAGNOSTIC_INTERVAL = 20
VALIDATION_INTERVAL = 100

SEED = 42
DEVICE = "auto"

DT_S = 1.0
MAX_FC_DELTA_KW = 60.0
REWARD_ALPHA = 0.001
SOC_TERMINAL_WEIGHT = 20000.0
FAILURE_PENALTY = -20.0


@dataclass
class DatasetBundle:
    train_profile: pd.DataFrame
    train_id: str
    validation_profiles: list[pd.DataFrame]
    validation_ids: list[str]
    manifest_path: str | None
    train_sample_count: int


def _norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).strip().lower())


def _strip_data_suffix(value: str) -> str:
    name = Path(str(value).replace("\\", "/")).name
    low = name.lower()
    if low.endswith(".csv.gz"):
        return name[:-7]
    if low.endswith(".csv"):
        return name[:-4]
    return name


def _classify_split(value: Any) -> str | None:
    token = _norm(value)
    if token.startswith("train"):
        return "train"
    if token.startswith("val") or token.startswith("valid"):
        return "validation"
    # Test rows are intentionally ignored and their data files are never loaded.
    return None


def _find_manifest_columns(df: pd.DataFrame):
    columns = {_norm(c): c for c in df.columns}

    split_col = next(
        (columns[k] for k in (
            "split", "datasetsplit", "partition", "subset", "set", "datasplit"
        ) if k in columns),
        None,
    )
    id_col = next(
        (columns[k] for k in (
            "sampleid", "voyageid", "segmentid", "tripid", "sample", "voyage"
        ) if k in columns),
        None,
    )
    path_col = next(
        (columns[k] for k in (
            "filepath", "relativepath", "filename", "file", "csvpath",
            "processedfile", "outputfile", "datafile", "csvfile", "path"
        ) if k in columns),
        None,
    )
    return split_col, id_col, path_col


def _manifest_candidates(data_root: Path) -> list[Path]:
    candidates = [
        data_root / "metadata" / "data_processing_report.csv",
        data_root / "metadata" / "split_manifest.csv",
        data_root / "metadata" / "manifest.csv",
        data_root / "data_processing_report.csv",
        data_root / "split_manifest.csv",
        data_root / "manifest.csv",
    ]

    metadata_dir = data_root / "metadata"
    if metadata_dir.is_dir():
        candidates.extend(sorted(metadata_dir.glob("*.csv")))

    unique = []
    seen = set()
    for path in candidates:
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        if resolved not in seen and path.is_file():
            unique.append(path)
            seen.add(resolved)
    return unique


def _rows_from_manifest(path: Path):
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig").fillna("")
    split_col, id_col, path_col = _find_manifest_columns(df)
    if split_col is None or (id_col is None and path_col is None):
        return None

    train_rows = []
    validation_rows = []

    for record in df.to_dict(orient="records"):
        split = _classify_split(record.get(split_col, ""))
        if split is None:
            continue

        raw_path = str(record.get(path_col, "")).strip() if path_col else ""
        sample_id = str(record.get(id_col, "")).strip() if id_col else ""
        if not sample_id and raw_path:
            sample_id = _strip_data_suffix(raw_path)

        sample_id = _strip_data_suffix(sample_id)
        if not sample_id:
            continue

        row = {"sample_id": sample_id, "file_value": raw_path}
        if split == "train":
            train_rows.append(row)
        else:
            validation_rows.append(row)

    if not train_rows and not validation_rows:
        return None
    return train_rows, validation_rows


def _list_payload_files(data_root: Path) -> list[Path]:
    files_found = []
    for path in data_root.rglob("*"):
        if not path.is_file():
            continue
        if "metadata" in {_norm(part) for part in path.parts}:
            continue
        low = path.name.lower()
        if low.endswith(".csv") or low.endswith(".csv.gz"):
            files_found.append(path)
    return sorted(files_found)


def _resolve_manifest_file(
    data_root: Path,
    sample_id: str,
    file_value: str,
    payload_files: Sequence[Path],
) -> Path:
    raw = str(file_value or "").strip().replace("\\", "/")

    if raw:
        raw_path = Path(raw)
        direct_candidates = []
        if raw_path.is_absolute():
            direct_candidates.append(raw_path)
        else:
            direct_candidates.extend((data_root / raw_path, PROJECT_ROOT / raw_path))
        for candidate in direct_candidates:
            if candidate.is_file():
                return candidate.resolve()

        basename = Path(raw).name
        by_basename = [p for p in payload_files if p.name == basename]
        if len(by_basename) == 1:
            return by_basename[0].resolve()
        if len(by_basename) > 1:
            raise RuntimeError(
                f"文件名 {basename!r} 在数据集中不唯一，请检查 manifest 的路径列。"
            )

    exact = [
        p for p in payload_files
        if _strip_data_suffix(p.name) == sample_id
    ]
    if len(exact) == 1:
        return exact[0].resolve()
    if len(exact) > 1:
        raise RuntimeError(f"航段 {sample_id} 对应多个 CSV 文件：{exact}")

    fuzzy = [
        p for p in payload_files
        if sample_id in _strip_data_suffix(p.name)
    ]
    if len(fuzzy) == 1:
        return fuzzy[0].resolve()
    if len(fuzzy) > 1:
        raise RuntimeError(
            f"航段 {sample_id} 匹配到多个 CSV 文件：{fuzzy}。"
            "请检查 manifest 中的文件路径列。"
        )

    raise FileNotFoundError(
        f"在数据目录 {data_root} 中找不到航段 {sample_id!r} 对应的 CSV/CSV.GZ。"
    )


def _select_manifest(data_root: Path, payload_files: Sequence[Path]):
    valid = []
    diagnostics = []

    for manifest in _manifest_candidates(data_root):
        parsed = _rows_from_manifest(manifest)
        if parsed is None:
            continue

        train_rows, validation_rows = parsed
        has_fixed_train = any(
            row["sample_id"] == TRAIN_VOYAGE_ID
            or TRAIN_VOYAGE_ID in row["file_value"]
            for row in train_rows
        )
        diagnostics.append(
            (manifest, len(train_rows), len(validation_rows), has_fixed_train)
        )

        if has_fixed_train and len(validation_rows) == EXPECTED_VALIDATION_COUNT:
            valid.append((manifest, train_rows, validation_rows))

    if len(valid) == 1:
        return valid[0]

    if len(valid) > 1:
        preferred = [
            item for item in valid
            if item[0].name.lower() == "data_processing_report.csv"
            and item[0].parent.name.lower() == "metadata"
        ]
        if len(preferred) == 1:
            return preferred[0]
        raise RuntimeError(
            "发现多个同时包含固定训练航段和8条验证航段的 manifest，"
            f"无法安全判断应使用哪一个：{[str(x[0]) for x in valid]}"
        )

    if diagnostics:
        detail = [
            {
                "manifest": str(p),
                "train_rows": n_train,
                "validation_rows": n_val,
                "contains_parent_005_seg01": has_fixed,
            }
            for p, n_train, n_val, has_fixed in diagnostics
        ]
        raise RuntimeError(
            "找到了数据划分文件，但其中没有同时满足“Train 中含 "
            "parent_005_seg01 且 Validation 恰好8条”的划分。"
            f"检测结果：{detail}"
        )

    return None


def _directory_split(data_root: Path, payload_files: Sequence[Path]):
    train_dirs = [
        p for p in data_root.iterdir()
        if p.is_dir() and _norm(p.name) in {"train", "training"}
    ]
    validation_dirs = [
        p for p in data_root.iterdir()
        if p.is_dir() and _norm(p.name) in {
            "validation", "valid", "val", "validationset"
        }
    ]

    if len(train_dirs) != 1 or len(validation_dirs) != 1:
        raise FileNotFoundError(
            "未找到可识别的数据划分 manifest，也未找到 data_root/Train "
            "与 data_root/Validation 两个目录。请确认正式数据集目录结构；"
            "脚本不会随机挑选8条航段，也不会读取 Test 数据。"
        )

    train_files = [
        p for p in payload_files if train_dirs[0] in p.parents
    ]
    validation_files = [
        p for p in payload_files if validation_dirs[0] in p.parents
    ]

    train_matches = [
        p for p in train_files
        if _strip_data_suffix(p.name) == TRAIN_VOYAGE_ID
    ]
    if len(train_matches) != 1:
        raise RuntimeError(
            f"Train 目录中应当恰好有一个 {TRAIN_VOYAGE_ID}，"
            f"实际匹配到 {len(train_matches)} 个。"
        )

    validation_rows = [
        {"sample_id": _strip_data_suffix(p.name), "file_value": str(p)}
        for p in validation_files
    ]
    if len(validation_rows) != EXPECTED_VALIDATION_COUNT:
        raise RuntimeError(
            f"Validation 目录应有8条航段，实际发现 {len(validation_rows)} 条。"
        )

    train_rows = [
        {"sample_id": _strip_data_suffix(p.name), "file_value": str(p)}
        for p in train_files
    ]
    return None, train_rows, validation_rows


def load_dataset_bundle(data_root: Path = DATA_ROOT) -> DatasetBundle:
    data_root = Path(data_root).resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(
            f"正式数据集目录不存在：{data_root}\n"
            "默认目录为项目根目录下 data/processed/operating_dataset_v6_1s。"
        )

    payload_files = _list_payload_files(data_root)
    if not payload_files:
        raise FileNotFoundError(f"数据目录中没有 CSV/CSV.GZ 文件：{data_root}")

    selected = _select_manifest(data_root, payload_files)
    if selected is None:
        manifest_path, train_rows, validation_rows = _directory_split(
            data_root, payload_files
        )
    else:
        manifest_path, train_rows, validation_rows = selected

    train_matches = [
        row for row in train_rows
        if row["sample_id"] == TRAIN_VOYAGE_ID
        or TRAIN_VOYAGE_ID in row["file_value"]
    ]
    if len(train_matches) != 1:
        raise RuntimeError(
            f"正式 Train 划分中应当恰好包含一个 {TRAIN_VOYAGE_ID}，"
            f"实际匹配到 {len(train_matches)} 个。"
        )

    if len(validation_rows) != EXPECTED_VALIDATION_COUNT:
        raise RuntimeError(
            f"Validation 必须恰好8条，实际为 {len(validation_rows)} 条。"
        )

    validation_ids = [row["sample_id"] for row in validation_rows]
    if len(set(validation_ids)) != EXPECTED_VALIDATION_COUNT:
        raise RuntimeError("Validation8 中存在重复 sample_id，已停止加载。")

    train_path = _resolve_manifest_file(
        data_root,
        TRAIN_VOYAGE_ID,
        train_matches[0]["file_value"],
        payload_files,
    )
    validation_paths = [
        _resolve_manifest_file(data_root, row["sample_id"], row["file_value"], payload_files)
        for row in validation_rows
    ]

    # Only the fixed Train voyage and the eight Validation voyages are opened.
    # Test payloads are not loaded.
    train_profile = load_voyage_csv(
        train_path, expected_interval_s=EXPECTED_INTERVAL_S
    )
    validation_profiles = [
        load_voyage_csv(path, expected_interval_s=EXPECTED_INTERVAL_S)
        for path in validation_paths
    ]

    if len(train_profile) < 2:
        raise RuntimeError("固定训练航段少于两个样本。")

    for sample_id, profile in zip(validation_ids, validation_profiles):
        if len(profile) < 2:
            raise RuntimeError(f"验证航段 {sample_id} 少于两个样本。")

    bundle = DatasetBundle(
        train_profile=train_profile,
        train_id=TRAIN_VOYAGE_ID,
        validation_profiles=validation_profiles,
        validation_ids=validation_ids,
        manifest_path=str(manifest_path) if manifest_path else None,
        train_sample_count=len(train_rows),
    )

    print("=" * 72)
    print("Ship EMS TD3 数据加载完成")
    print(f"数据目录：{data_root}")
    print(f"划分文件：{bundle.manifest_path or 'Train/Validation 文件夹'}")
    print(f"Train 样本数：{bundle.train_sample_count}")
    print(
        f"固定训练航段：{bundle.train_id} | "
        f"{len(train_profile)} 个采样点 | {len(train_profile) - 1} 个环境步"
    )
    print(f"Validation 航段数：{len(validation_ids)}")
    for i, sample_id in enumerate(validation_ids, 1):
        print(f"  Val {i:02d}: {sample_id} | {len(validation_profiles[i-1])} 个采样点")
    print("Test 数据：未加载")
    print("=" * 72)

    return bundle


def make_env(
    load_profiles: Sequence,
    sample_ids: Sequence[str],
) -> ShipEMSEnv:
    return ShipEMSEnv(
        load_profiles=load_profiles,
        sample_ids=sample_ids,
        dt_s=DT_S,
        max_fc_delta_kw=MAX_FC_DELTA_KW,
        random_voyage=False,
        reward_alpha=REWARD_ALPHA,
        soc_terminal_weight=SOC_TERMINAL_WEIGHT,
        soc_target=CONFIG.soc_target,
        failure_penalty=FAILURE_PENALTY,
    )


def load_huber_td3_class():
    try:
        from experiments.sb3_manual.td3_aggregate.huber_td3 import HuberTD3
    except Exception as exc:
        raise ImportError(
            "无法导入项目现有的 HuberTD3。需要仓库中的 "
            "experiments/sb3_manual/td3_aggregate/huber_td3.py；"
            "这里不自动退回普通 TD3，以免悄悄改变 Critic 损失函数。"
        ) from exc
    return HuberTD3


def _python_value(value: Any):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _python_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_python_value(v) for v in value]
    return value


def write_csv(path: Path, rows: list[dict]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return

    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _python_value(v) for k, v in row.items()})


def save_json(path: Path, payload: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_python_value(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def evaluate_profiles(
    model,
    profiles: Sequence,
    sample_ids: Sequence[str],
    seed: int = SEED,
    collect_step_rows: bool = False,
):
    """Greedy evaluation only: no exploration, replay insertion, or gradient update."""
    from stable_baselines3.common.utils import check_for_correct_spaces

    if len(profiles) != len(sample_ids):
        raise ValueError("profiles 与 sample_ids 长度不一致。")

    env = make_env(profiles, sample_ids)
    check_for_correct_spaces(env, model.observation_space, model.action_space)

    summaries = []
    step_rows = []

    try:
        model.policy.set_training_mode(False)

        for voyage_index, sample_id in enumerate(sample_ids):
            obs, reset_info = env.reset(
                seed=seed if voyage_index == 0 else None,
                options={"voyage_index": voyage_index},
            )

            expected_steps = int(reset_info["episode_length"])
            total_reward = 0.0
            calls = 0
            executed_steps = 0
            final_info = {}
            soc_values = [float(reset_info["initial_soc"])]
            fc_values = []
            battery_values = []
            delta_fc_values = []

            while True:
                action, _ = model.predict(obs, deterministic=True)
                action_value = float(np.asarray(action).reshape(-1)[0])

                obs, reward, terminated, truncated, info = env.step(action)
                calls += 1
                total_reward += float(reward)
                final_info = info

                if not info.get("failure", False) and info.get("fc_actual_kw") is not None:
                    executed_steps += 1

                if info.get("soc") is not None:
                    soc_values.append(float(info["soc"]))
                if info.get("fc_actual_kw") is not None:
                    fc_values.append(float(info["fc_actual_kw"]))
                if info.get("battery_actual_kw") is not None:
                    battery_values.append(float(info["battery_actual_kw"]))
                if info.get("delta_fc_kw") is not None:
                    delta_fc_values.append(float(info["delta_fc_kw"]))

                if collect_step_rows:
                    row = {
                        "sample_id": sample_id,
                        "step": calls,
                        "actor_action_delta_kw": action_value,
                        "reward": float(reward),
                    }
                    row.update({k: _python_value(v) for k, v in info.items()})
                    step_rows.append(row)

                if terminated or truncated:
                    break

                if calls > expected_steps + 2:
                    raise RuntimeError(
                        f"航段 {sample_id} 的步数超过预期，环境未正常终止。"
                    )

            completed = (
                final_info.get("terminal_reason") == "voyage_end"
                and not final_info.get("failure", False)
                and bool(final_info.get("economic_cost_ready", False))
            )

            summaries.append({
                "sample_id": sample_id,
                "completed": bool(completed),
                "failure": bool(final_info.get("failure", False)),
                "terminal_reason": final_info.get("terminal_reason"),
                "expected_steps": expected_steps,
                "environment_steps": calls,
                "executed_steps": executed_steps,
                "coverage": executed_steps / expected_steps if expected_steps else 0.0,
                "episode_reward": float(total_reward),
                "mean_reward_per_step": float(total_reward / max(calls, 1)),
                "initial_soc": float(reset_info["initial_soc"]),
                "final_soc": final_info.get("final_soc", final_info.get("soc")),
                "soc_min": float(min(soc_values)) if soc_values else None,
                "soc_max": float(max(soc_values)) if soc_values else None,
                "mean_fc_kw": float(np.mean(fc_values)) if fc_values else None,
                "mean_battery_kw": float(np.mean(battery_values)) if battery_values else None,
                "mean_abs_delta_fc_kw": (
                    float(np.mean(np.abs(delta_fc_values))) if delta_fc_values else None
                ),
                "voyage_hydrogen_kg": final_info.get("voyage_hydrogen_kg"),
                "voyage_hydrogen_cost_rmb": final_info.get("voyage_hydrogen_cost_rmb"),
                "fc_operating_degradation_cost_rmb": final_info.get(
                    "fc_operating_degradation_cost_rmb"
                ),
                "fc_start_stop_degradation_cost_rmb": final_info.get(
                    "fc_start_stop_degradation_cost_rmb"
                ),
                "fc_total_degradation_cost_rmb": final_info.get(
                    "fc_total_degradation_cost_rmb"
                ),
                "voyage_battery_degradation_cost_rmb": final_info.get(
                    "voyage_battery_degradation_cost_rmb"
                ),
                "battery_equivalent_hydrogen_cost_rmb": final_info.get(
                    "battery_equivalent_hydrogen_cost_rmb"
                ),
                "total_economic_cost_rmb": final_info.get("total_economic_cost_rmb"),
                "soc_terminal_error": final_info.get("soc_terminal_error"),
            })

    finally:
        env.close()

    return summaries, step_rows


def summarize_validation(rows: list[dict]) -> dict:
    completed_rows = [r for r in rows if r["completed"]]
    rewards = [float(r["episode_reward"]) for r in rows]
    completed_costs = [
        float(r["total_economic_cost_rmb"])
        for r in completed_rows
        if r.get("total_economic_cost_rmb") is not None
    ]

    return {
        "expected_voyages": EXPECTED_VALIDATION_COUNT,
        "evaluated_voyages": len(rows),
        "completed_voyages": len(completed_rows),
        "completion_rate": (
            len(completed_rows) / len(rows) if rows else 0.0
        ),
        "validation_set_complete": (
            len(rows) == EXPECTED_VALIDATION_COUNT
            and len(completed_rows) == EXPECTED_VALIDATION_COUNT
        ),
        "mean_episode_reward": float(np.mean(rewards)) if rewards else None,
        "mean_total_economic_cost_completed_only_rmb": (
            float(np.mean(completed_costs)) if completed_costs else None
        ),
        "completed_cost_sample_count": len(completed_costs),
    }
