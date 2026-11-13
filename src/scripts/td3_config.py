
from __future__ import annotations

import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


def _find_project_layout() -> tuple[Path, Path]:
    """Return (project_root, source_root) for either src/ship_ems or ship_ems."""
    script_dir = Path(__file__).resolve().parent

    for candidate in (script_dir, *script_dir.parents):
        src_package = candidate / "src" / "ship_ems" / "env.py"
        root_package = candidate / "ship_ems" / "env.py"

        if src_package.is_file():
            return candidate, candidate / "src"

        if root_package.is_file():
            project_root = candidate.parent if candidate.name == "src" else candidate
            return project_root, candidate

    raise RuntimeError(
        "未找到 ship_ems 包。请将本组脚本放在项目的 src/scripts/ 目录，"
        "并确认项目中存在 src/ship_ems/env.py。"
    )


PROJECT_ROOT, SOURCE_ROOT = _find_project_layout()

for _path in (PROJECT_ROOT, SOURCE_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


from ship_ems.config import CONFIG
from ship_ems.data import load_voyage_csv
from ship_ems.env import ShipEMSEnv


# =============================================================
# 1. Fixed project settings
# =============================================================

DATA_ROOT = Path(
    os.environ.get(
        "SHIP_EMS_DATA_ROOT",
        str(PROJECT_ROOT / "data" / "processed" / "operating_dataset_v6_1s"),
    )
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "td3_ship_ems_parent005_obs3_1s_eq_h2_soc50_no_failure_penalty_20ep"
)

ACTION_MAPPING = "relative_feasible_fc_fraction"

DIAGNOSTIC_VOYAGE_ID = "parent_005_seg01"

EXPECTED_TRAIN_COUNT = 35
EXPECTED_VALIDATION_COUNT = 8


MAX_EPISODES = 20

REWARD_PLOT_INTERVAL = 10

TRAIN_DIAGNOSTIC_INTERVAL = 10

VALIDATION_INTERVAL = 100

SEED = 42

DEVICE = "auto"


# =============================================================
# 2. Physical and training parameters
# =============================================================

DT_S = 1.0

# Existing onboard control deadband from v2/data/supervisory_rules.py.
SOURCE_LOAD_DEADBAND_KW = 1.0


# =============================================================
# 3. Reward parameters
# =============================================================

REWARD_VERSION = "economic4_plus_soc_step_unscaled_no_failure_penalty"

# 逐步 SOC 惩罚权重
SOC_STEP_WEIGHT = 50.0

# SOC 目标值
SOC_TARGET = 0.50

# 失败惩罚
FAILURE_PENALTY = 0.0


# =============================================================
# 4. Dataset bundle
# =============================================================

@dataclass
class DatasetBundle:

    train_profiles: list[pd.DataFrame]

    train_ids: list[str]

    validation_profiles: list[pd.DataFrame]

    validation_ids: list[str]

    manifest_path: str | None


# =============================================================
# 5. Dataset utilities
# =============================================================

def _norm(value: Any) -> str:

    return re.sub(
        r"[^a-z0-9]",
        "",
        str(value).strip().lower()
    )


def _strip_data_suffix(value: str) -> str:

    name = Path(
        str(value).replace("\\", "/")
    ).name

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

    return None


def _find_manifest_columns(df: pd.DataFrame):

    columns = {
        _norm(c): c
        for c in df.columns
    }

    split_col = next(
        (
            columns[k]
            for k in (
                "split",
                "datasetsplit",
                "partition",
                "subset",
                "set",
                "datasplit",
            )
            if k in columns
        ),
        None,
    )

    id_col = next(
        (
            columns[k]
            for k in (
                "sampleid",
                "candidateid",
                "voyageid",
                "segmentid",
                "tripid",
                "sample",
                "voyage",
            )
            if k in columns
        ),
        None,
    )

    path_col = next(
        (
            columns[k]
            for k in (
                "filepath",
                "relativepath",
                "filename",
                "file",
                "csvpath",
                "processedfile",
                "outputfile",
                "datafile",
                "csvfile",
                "path",
            )
            if k in columns
        ),
        None,
    )

    return split_col, id_col, path_col


def _manifest_candidates(data_root: Path) -> list[Path]:
    active_split = data_root / "metadata" / "active_split.json"
    if active_split.is_file():
        split = json.loads(active_split.read_text(encoding="utf-8-sig"))
        return [data_root / split["manifest_relative_path"]]

    candidates = [

        data_root
        / "metadata"
        / "data_processing_report.csv",

        data_root
        / "metadata"
        / "split_manifest.csv",

        data_root
        / "metadata"
        / "manifest.csv",

        data_root
        / "data_processing_report.csv",

        data_root
        / "split_manifest.csv",

        data_root
        / "manifest.csv",
    ]

    metadata_dir = data_root / "metadata"

    if metadata_dir.is_dir():

        candidates.extend(
            sorted(metadata_dir.glob("*.csv"))
        )

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

    df = pd.read_csv(
        path,
        dtype=str,
        encoding="utf-8-sig"
    ).fillna("")

    split_col, id_col, path_col = (
        _find_manifest_columns(df)
    )

    if split_col is None or (
        id_col is None and path_col is None
    ):
        return None

    train_rows = []

    validation_rows = []

    for record in df.to_dict(orient="records"):

        split = _classify_split(
            record.get(split_col, "")
        )

        if split is None:
            continue

        raw_path = (
            str(record.get(path_col, "")).strip()
            if path_col
            else ""
        )

        sample_id = (
            str(record.get(id_col, "")).strip()
            if id_col
            else ""
        )

        if not sample_id and raw_path:

            sample_id = _strip_data_suffix(
                raw_path
            )

        sample_id = _strip_data_suffix(
            sample_id
        )

        if not sample_id:
            continue

        row = {
            "sample_id": sample_id,
            "file_value": raw_path,
        }

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

        if "metadata" in {
            _norm(part)
            for part in path.parts
        }:
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

    raw = str(
        file_value or ""
    ).strip().replace("\\", "/")

    if raw:

        raw_path = Path(raw)

        direct_candidates = []

        if raw_path.is_absolute():

            direct_candidates.append(raw_path)

        else:

            direct_candidates.extend(
                (
                    data_root / raw_path,
                    PROJECT_ROOT / raw_path,
                )
            )

        for candidate in direct_candidates:

            if candidate.is_file():
                return candidate.resolve()

        basename = Path(raw).name

        by_basename = [
            p
            for p in payload_files
            if p.name == basename
        ]

        if len(by_basename) == 1:
            return by_basename[0].resolve()

        if len(by_basename) > 1:

            raise RuntimeError(
                f"文件名 {basename!r} 在数据集中不唯一，"
                "请检查 manifest 的路径列。"
            )

    exact = [
        p
        for p in payload_files
        if _strip_data_suffix(p.name) == sample_id
    ]

    if len(exact) == 1:
        return exact[0].resolve()

    if len(exact) > 1:

        raise RuntimeError(
            f"航段 {sample_id} 对应多个 CSV 文件：{exact}"
        )

    fuzzy = [
        p
        for p in payload_files
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
        f"在数据目录 {data_root} 中找不到航段 "
        f"{sample_id!r} 对应的 CSV/CSV.GZ。"
    )


# =============================================================
# 6. Dataset split selection
# =============================================================

def _select_manifest(
    data_root: Path,
    payload_files: Sequence[Path],
):

    valid = []

    diagnostics = []

    for manifest in _manifest_candidates(data_root):

        parsed = _rows_from_manifest(manifest)

        if parsed is None:
            continue

        train_rows, validation_rows = parsed

        diagnostics.append(
            (
                manifest,
                len(train_rows),
                len(validation_rows),
            )
        )

        if (
            len(train_rows) == EXPECTED_TRAIN_COUNT
            and len(validation_rows)
            == EXPECTED_VALIDATION_COUNT
        ):

            valid.append(
                (
                    manifest,
                    train_rows,
                    validation_rows,
                )
            )

    if len(valid) == 1:
        return valid[0]

    if len(valid) > 1:

        preferred = [
            item
            for item in valid
            if item[0].name.lower()
            == "data_processing_report.csv"
            and item[0].parent.name.lower()
            == "metadata"
        ]

        if len(preferred) == 1:
            return preferred[0]

        raise RuntimeError(
            "发现多个同时包含35条训练航段和8条验证航段的 manifest，"
            f"无法安全判断应使用哪一个：{[str(x[0]) for x in valid]}"
        )

    if diagnostics:

        detail = [
            {
                "manifest": str(p),
                "train_rows": n_train,
                "validation_rows": n_val,
            }
            for p, n_train, n_val in diagnostics
        ]

        raise RuntimeError(
            "找到了数据划分文件，但其中没有同时满足"
            "“Train35 和 Validation8”的划分。"
            f"检测结果：{detail}"
        )

    return None


def _directory_split(
    data_root: Path,
    payload_files: Sequence[Path],
):

    train_dirs = [
        p
        for p in data_root.iterdir()
        if p.is_dir()
        and _norm(p.name) in {
            "train",
            "training",
        }
    ]

    validation_dirs = [
        p
        for p in data_root.iterdir()
        if p.is_dir()
        and _norm(p.name) in {
            "validation",
            "valid",
            "val",
            "validationset",
        }
    ]

    if (
        len(train_dirs) != 1
        or len(validation_dirs) != 1
    ):

        raise FileNotFoundError(
            "未找到可识别的数据划分 manifest，也未找到 "
            "data_root/Train 与 data_root/Validation 两个目录。"
            "请确认正式数据集目录结构；"
            "脚本不会随机挑选8条航段，也不会读取 Test 数据。"
        )

    train_files = [
        p
        for p in payload_files
        if train_dirs[0] in p.parents
    ]

    validation_files = [
        p
        for p in payload_files
        if validation_dirs[0] in p.parents
    ]

    train_matches = [
        p
        for p in train_files
        if _strip_data_suffix(p.name)
        == TRAIN_VOYAGE_ID
    ]

    if len(train_matches) != 1:

        raise RuntimeError(
            f"Train 目录中应当恰好有一个 {TRAIN_VOYAGE_ID}，"
            f"实际匹配到 {len(train_matches)} 个。"
        )

    validation_rows = [
        {
            "sample_id": _strip_data_suffix(p.name),
            "file_value": str(p),
        }
        for p in validation_files
    ]

    if len(validation_rows) != EXPECTED_VALIDATION_COUNT:

        raise RuntimeError(
            f"Validation 目录应有8条航段，"
            f"实际发现 {len(validation_rows)} 条。"
        )

    train_rows = [
        {
            "sample_id": _strip_data_suffix(p.name),
            "file_value": str(p),
        }
        for p in train_files
    ]

    return None, train_rows, validation_rows


# =============================================================
# 7. Load dataset
# =============================================================

def load_dataset_bundle(
    data_root: Path = DATA_ROOT,
) -> DatasetBundle:

    data_root = Path(data_root).resolve()

    if not data_root.is_dir():

        raise FileNotFoundError(
            f"正式数据集目录不存在：{data_root}\n"
            "默认目录为项目根目录下 "
            "data/processed/operating_dataset_v6_1s。"
        )

    payload_files = _list_payload_files(
        data_root
    )

    if not payload_files:

        raise FileNotFoundError(
            f"数据目录中没有 CSV/CSV.GZ 文件：{data_root}"
        )

    selected = _select_manifest(
        data_root,
        payload_files,
    )

    if selected is None:

        (
            manifest_path,
            train_rows,
            validation_rows,
        ) = _directory_split(
            data_root,
            payload_files,
        )

    else:

        (
            manifest_path,
            train_rows,
            validation_rows,
        ) = selected

    train_ids = [row["sample_id"] for row in train_rows]
    if len(train_ids) != EXPECTED_TRAIN_COUNT or len(set(train_ids)) != len(train_ids):
        raise RuntimeError("Train 划分必须恰好包含35条不同航段。")

    if len(validation_rows) != EXPECTED_VALIDATION_COUNT:

        raise RuntimeError(
            f"Validation 必须恰好8条，"
            f"实际为 {len(validation_rows)} 条。"
        )

    validation_ids = [
        row["sample_id"]
        for row in validation_rows
    ]

    if len(set(validation_ids)) != EXPECTED_VALIDATION_COUNT:

        raise RuntimeError(
            "Validation8 中存在重复 sample_id，已停止加载。"
        )

    def load_profiles(rows):
        return [
            load_voyage_csv(
                _resolve_manifest_file(
                    data_root,
                    row["sample_id"],
                    row["file_value"],
                    payload_files,
                ),
                expected_interval_s=DT_S,
                negative_load_tolerance_kw=SOURCE_LOAD_DEADBAND_KW,
            )
            for row in rows
        ]

    train_profiles = load_profiles(train_rows)
    validation_profiles = load_profiles(validation_rows)

    bundle = DatasetBundle(

        train_profiles=train_profiles,

        train_ids=train_ids,

        validation_profiles=validation_profiles,

        validation_ids=validation_ids,

        manifest_path=(
            str(manifest_path)
            if manifest_path
            else None
        ),

    )

    print("=" * 72)

    print("Ship EMS TD3 数据加载完成")

    print(f"数据目录：{data_root}")

    print(
        f"划分文件："
        f"{bundle.manifest_path or 'Train/Validation 文件夹'}"
    )

    print(
        f"Train 航段数：{len(train_ids)} | "
        f"环境步范围：{min(map(len, train_profiles)) - 1}–"
        f"{max(map(len, train_profiles)) - 1}"
    )

    print(
        f"Validation 航段数：{len(validation_ids)}"
    )

    for i, sample_id in enumerate(
        validation_ids,
        1,
    ):

        print(
            f"  Val {i:02d}: {sample_id} | "
            f"{len(validation_profiles[i-1])} 个采样点"
        )

    print("Test 数据：未加载")

    print("=" * 72)

    return bundle


# =============================================================
# 8. Step-wise reward function
# =============================================================

def calculate_step_reward(info: dict) -> float:
    """
    每个环境步计算一次奖励。

    r_t = -(
        C_H2
        + C_FC_degradation
        + C_LB_degradation
        + C_LB_equivalent_hydrogen
        + SOC_STEP_WEIGHT * (SOC_next - SOC_target)^2
    )

    SOC 使用当前步更新后的 SOC。
    """

    if not isinstance(info, dict):
        raise TypeError("info 必须是 dict。")

    required_keys = (
        "step_reward_cost_rmb",
        "step_battery_equivalent_hydrogen_cost_rmb",
        "soc",
    )

    missing = [
        key
        for key in required_keys
        if key not in info
    ]

    if missing:

        raise KeyError(
            f"奖励计算缺少必要字段：{missing}"
        )

    economic_cost = float(
        info["step_reward_cost_rmb"]
    )

    soc_next = float(
        info["soc"]
    )

    soc_error = (
        soc_next - SOC_TARGET
    )

    soc_step_cost = (
        SOC_STEP_WEIGHT
        * soc_error ** 2
    )

    total_step_cost = (
        economic_cost
        + float(info["step_battery_equivalent_hydrogen_cost_rmb"])
        + soc_step_cost
    )

    reward = -total_step_cost

    if not np.isfinite(reward):

        raise ValueError(
            "计算得到非有限奖励。"
        )

    info["step_soc_penalty_unscaled"] = soc_step_cost
    return float(reward)


# =============================================================
# 9. Environment factory
# =============================================================

def make_env(
    load_profiles: Sequence,
    sample_ids: Sequence[str],
    *,
    random_voyage: bool = False,
) -> ShipEMSEnv:

    return ShipEMSEnv(

        load_profiles=load_profiles,

        sample_ids=sample_ids,

        dt_s=DT_S,

        random_voyage=random_voyage,

        # 使用自定义逐步奖励函数。
        reward_fn=calculate_step_reward,

        # 关闭环境内置奖励计算分支。
        # SOC 惩罚完全由 calculate_step_reward 逐步计算。
        reward_alpha=None,

        # None 表示不启用环境内置 SOC 惩罚项。
        soc_step_weight=None,

        soc_target=SOC_TARGET,

        failure_penalty=FAILURE_PENALTY,
    )


# =============================================================
# 10. Load HuberTD3
# =============================================================

def load_huber_td3_class():

    try:

        from experiments.sb3_manual.td3_aggregate.huber_td3 import (
            HuberTD3
        )

    except Exception as exc:

        raise ImportError(
            "无法导入项目现有的 HuberTD3。"
            "需要仓库中的 "
            "experiments/sb3_manual/td3_aggregate/huber_td3.py；"
            "这里不自动退回普通 TD3，"
            "以免悄悄改变 Critic 损失函数。"
        ) from exc

    return HuberTD3


# =============================================================
# 11. Serialization utilities
# =============================================================

def _python_value(value: Any):

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):

        return {
            str(k): _python_value(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):

        return [
            _python_value(v)
            for v in value
        ]

    return value


def write_csv(
    path: Path,
    rows: list[dict],
):

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:

        path.write_text(
            "",
            encoding="utf-8-sig",
        )

        return

    fieldnames = []

    seen = set()

    for row in rows:

        for key in row:

            if key not in seen:

                seen.add(key)

                fieldnames.append(key)

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()

        for row in rows:

            writer.writerow(
                {
                    k: _python_value(v)
                    for k, v in row.items()
                }
            )


def save_json(
    path: Path,
    payload: dict,
):

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            _python_value(payload),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


# =============================================================
# 12. Validation evaluation
# =============================================================

def evaluate_profiles(
    model,
    profiles: Sequence,
    sample_ids: Sequence[str],
    seed: int = SEED,
    collect_step_rows: bool = False,
):
    """
    Greedy evaluation only.

    No exploration.
    No replay insertion.
    No gradient update.

    Training and validation use the same step-wise reward.
    """

    from stable_baselines3.common.utils import (
        check_for_correct_spaces
    )

    if len(profiles) != len(sample_ids):

        raise ValueError(
            "profiles 与 sample_ids 长度不一致。"
        )

    env = make_env(
        profiles,
        sample_ids,
    )

    check_for_correct_spaces(
        env,
        model.observation_space,
        model.action_space,
    )

    summaries = []

    step_rows = []

    try:

        model.policy.set_training_mode(False)

        for voyage_index, sample_id in enumerate(
            sample_ids
        ):

            obs, reset_info = env.reset(

                seed=(
                    seed
                    if voyage_index == 0
                    else None
                ),

                options={
                    "voyage_index": voyage_index
                },
            )

            expected_steps = int(
                reset_info["episode_length"]
            )

            total_reward = 0.0

            calls = 0

            executed_steps = 0

            final_info = {}

            soc_values = [
                float(reset_info["initial_soc"])
            ]

            fc_values = []

            battery_values = []

            delta_fc_values = []

            while True:

                action, _ = model.predict(
                    obs,
                    deterministic=True,
                )

                action_value = float(
                    np.asarray(action).reshape(-1)[0]
                )

                (
                    obs,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = env.step(action)

                calls += 1

                total_reward += float(reward)

                final_info = info

                if (
                    not info.get("failure", False)
                    and info.get("fc_actual_kw") is not None
                ):

                    executed_steps += 1

                if info.get("soc") is not None:

                    soc_values.append(
                        float(info["soc"])
                    )

                if info.get("fc_actual_kw") is not None:

                    fc_values.append(
                        float(info["fc_actual_kw"])
                    )

                if info.get("battery_actual_kw") is not None:

                    battery_values.append(
                        float(info["battery_actual_kw"])
                    )

                if info.get("delta_fc_kw") is not None:

                    delta_fc_values.append(
                        float(info["delta_fc_kw"])
                    )

                if collect_step_rows:

                    row = {

                        "sample_id": sample_id,

                        "step": calls,

                        "actor_action_fraction": action_value,

                        "reward": float(reward),
                    }

                    row.update(
                        {
                            k: _python_value(v)
                            for k, v in info.items()
                        }
                    )

                    step_rows.append(row)

                if terminated or truncated:
                    break

                if calls > expected_steps + 2:

                    raise RuntimeError(
                        f"航段 {sample_id} 的步数超过预期，"
                        "环境未正常终止。"
                    )

            completed = (

                final_info.get("terminal_reason")
                == "voyage_end"

                and not final_info.get("failure", False)

                and bool(
                    final_info.get("economic_cost_ready", False)
                )
            )

            summaries.append({

                "sample_id": sample_id,

                "completed": bool(completed),

                "failure": bool(
                    final_info.get("failure", False)
                ),

                "terminal_reason": (
                    final_info.get("terminal_reason")
                ),

                "expected_steps": expected_steps,

                "environment_steps": calls,

                "executed_steps": executed_steps,

                "coverage": (
                    executed_steps / expected_steps
                    if expected_steps
                    else 0.0
                ),

                "episode_reward": float(total_reward),

                "mean_reward_per_step": (
                    float(total_reward / max(calls, 1))
                ),

                "initial_soc": float(
                    reset_info["initial_soc"]
                ),

                "final_soc": final_info.get(
                    "final_soc",
                    final_info.get("soc"),
                ),

                "soc_min": (
                    float(min(soc_values))
                    if soc_values
                    else None
                ),

                "soc_max": (
                    float(max(soc_values))
                    if soc_values
                    else None
                ),

                "mean_fc_kw": (
                    float(np.mean(fc_values))
                    if fc_values
                    else None
                ),

                "mean_battery_kw": (
                    float(np.mean(battery_values))
                    if battery_values
                    else None
                ),

                "mean_abs_delta_fc_kw": (

                    float(
                        np.mean(
                            np.abs(delta_fc_values)
                        )
                    )

                    if delta_fc_values
                    else None
                ),

                "voyage_hydrogen_kg": (
                    final_info.get("voyage_hydrogen_kg")
                ),

                "voyage_hydrogen_cost_rmb": (
                    final_info.get("voyage_hydrogen_cost_rmb")
                ),

                "fc_operating_degradation_cost_rmb": (
                    final_info.get(
                        "fc_operating_degradation_cost_rmb"
                    )
                ),

                "fc_start_stop_degradation_cost_rmb": (
                    final_info.get(
                        "fc_start_stop_degradation_cost_rmb"
                    )
                ),

                "fc_total_degradation_cost_rmb": (
                    final_info.get(
                        "fc_total_degradation_cost_rmb"
                    )
                ),

                "voyage_battery_degradation_cost_rmb": (
                    final_info.get(
                        "voyage_battery_degradation_cost_rmb"
                    )
                ),

                "battery_equivalent_hydrogen_cost_rmb": (
                    final_info.get(
                        "battery_equivalent_hydrogen_cost_rmb"
                    )
                ),

                "total_economic_cost_rmb": (
                    final_info.get(
                        "total_economic_cost_rmb"
                    )
                ),

                "soc_terminal_error": (
                    final_info.get("soc_terminal_error")
                ),
            })

    finally:

        env.close()

    return summaries, step_rows


# =============================================================
# 13. Validation summary
# =============================================================

def summarize_validation(
    rows: list[dict],
) -> dict:

    completed_rows = [
        r
        for r in rows
        if r["completed"]
    ]

    rewards = [
        float(r["episode_reward"])
        for r in rows
    ]

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

            len(completed_rows) / len(rows)
            if rows
            else 0.0
        ),

        "validation_set_complete": (

            len(rows) == EXPECTED_VALIDATION_COUNT

            and len(completed_rows) == EXPECTED_VALIDATION_COUNT
        ),

        "mean_episode_reward": (

            float(np.mean(rewards))
            if rewards
            else None
        ),

        "mean_total_economic_cost_completed_only_rmb": (

            float(np.mean(completed_costs))
            if completed_costs
            else None
        ),

        "completed_cost_sample_count": len(
            completed_costs
        ),
    }
