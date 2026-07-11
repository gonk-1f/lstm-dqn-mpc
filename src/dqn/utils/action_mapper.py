from __future__ import annotations

import csv
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import yaml


def build_action_table(fc_grid_kw: tuple[float, ...], batt_grid_kw: tuple[float, ...]) -> list[tuple[float, float]]:
    """Map discrete action ids to fuel-cell and battery power increments."""
    return [(fc_delta, batt_delta) for fc_delta in fc_grid_kw for batt_delta in batt_grid_kw]


@dataclass(frozen=True)
class WeightAction:
    action_id: int
    delta_q_soc: float
    delta_q_fc: float
    delta_q_batt: float
    description: str
    delta_q_ramp: float = 0.0

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.delta_q_soc, self.delta_q_fc, self.delta_q_batt, self.delta_q_ramp)


@dataclass(frozen=True)
class WeightActionConfig:
    q_soc_base: float = 6000.0
    q_fc_base: float = 0.001
    q_batt_base: float = 0.004
    q_ramp_base: float = 0.18
    dq_soc_1: float = 1000.0
    dq_soc_2: float = 3000.0
    dq_fc_1: float = 0.0003
    dq_batt_1: float = 0.001
    dq_soc_values: tuple[float, ...] | None = None
    dq_fc_values: tuple[float, ...] | None = None
    dq_batt_values: tuple[float, ...] | None = None
    dq_ramp_values: tuple[float, ...] | None = None
    eps: float = 1e-8


def build_weight_action_table(config: WeightActionConfig) -> list[WeightAction]:
    """Build discrete additive MPC-weight actions."""
    if (
        config.dq_soc_values is not None
        or config.dq_fc_values is not None
        or config.dq_batt_values is not None
        or config.dq_ramp_values is not None
    ):
        dq_soc_values = tuple(float(v) for v in (config.dq_soc_values or (0.0,)))
        dq_fc_values = tuple(float(v) for v in (config.dq_fc_values or (0.0,)))
        dq_batt_values = tuple(float(v) for v in (config.dq_batt_values or (0.0,)))
        dq_ramp_values = tuple(float(v) for v in (config.dq_ramp_values or (0.0,)))
        table: list[WeightAction] = []
        for action_id, (d_soc, d_fc, d_batt, d_ramp) in enumerate(
            product(dq_soc_values, dq_fc_values, dq_batt_values, dq_ramp_values)
        ):
            description = (
                "direct delta "
                f"dq_soc={d_soc:g}, dq_fc={d_fc:g}, dq_batt={d_batt:g}, dq_ramp={d_ramp:g}"
            )
            table.append(WeightAction(action_id, d_soc, d_fc, d_batt, description, d_ramp))
        return table

    ds1 = float(config.dq_soc_1)
    ds2 = float(config.dq_soc_2)
    dfc = float(config.dq_fc_1)
    db = float(config.dq_batt_1)
    return [
        # action 0: 默认权重，不调整。
        WeightAction(0, 0.0, 0.0, 0.0, "default weights"),
        # action 1: 轻度 SOC 保护；适用于 SOC 略低或下降趋势，提高 q_soc、降低 q_fc。
        WeightAction(1, ds1, -dfc, 0.0, "mild SOC protection"),
        # action 2: 强 SOC 保护；适用于 SOC 明显偏低或未来负荷较高，进一步提高 q_soc、降低 q_fc。
        WeightAction(2, ds2, -dfc, 0.0, "strong SOC protection"),
        # action 3: 充电恢复；提高 q_soc、降低 q_fc 和 q_batt，鼓励燃料电池多出力并回充电池。
        WeightAction(3, ds2, -dfc, -db, "charge recovery"),
        # action 4: 高 SOC 经济放电；降低 q_soc、提高 q_fc、降低 q_batt，允许电池供能以降低氢耗。
        WeightAction(4, -ds1, dfc, -db, "high-SOC economic discharge"),
        # action 5: 省氢模式；SOC 安全且未来负荷较低时提高 q_fc，抑制燃料电池输出。
        WeightAction(5, 0.0, dfc, 0.0, "fuel-cell saving"),
        # action 6: 电池保护；电池功率较大或 SOC 变化较快时提高 q_soc 和 q_batt。
        WeightAction(6, ds1, 0.0, db, "battery protection"),
        # action 7: SOC 保护 + 电池保护；SOC 偏低且电池压力较大时提高 q_soc、降低 q_fc、提高 q_batt。
        WeightAction(7, ds1, -dfc, db, "SOC and battery protection"),
    ]


def apply_weight_action(action_id: int, config: WeightActionConfig) -> dict[str, float | int]:
    table = build_weight_action_table(config)
    if int(action_id) < 0 or int(action_id) >= len(table):
        raise IndexError(f"Invalid weight action id {action_id}; expected 0..{len(table) - 1}.")
    action = table[int(action_id)]
    eps = float(config.eps)
    q_soc = max(float(config.q_soc_base) + action.delta_q_soc, eps)
    q_fc = max(float(config.q_fc_base) + action.delta_q_fc, eps)
    q_batt = max(float(config.q_batt_base) + action.delta_q_batt, eps)
    q_ramp = max(float(config.q_ramp_base) + action.delta_q_ramp, eps)
    return {
        "action_id": int(action.action_id),
        "delta_q_soc": float(action.delta_q_soc),
        "delta_q_fc": float(action.delta_q_fc),
        "delta_q_batt": float(action.delta_q_batt),
        "delta_q_ramp": float(action.delta_q_ramp),
        "q_soc_new": float(q_soc),
        "q_fc_new": float(q_fc),
        "q_batt_new": float(q_batt),
        "q_ramp_new": float(q_ramp),
    }


def build_dqn_weight_state(
    soc: float,
    soc_ref: float,
    d_soc: float,
    current_load: float,
    pred_mu: np.ndarray,
    pred_sigma: np.ndarray,
    p_fc_prev: float,
    p_batt_prev: float,
    sigma_multipliers: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    pred_mu = np.asarray(pred_mu, dtype=float).reshape(-1)
    pred_sigma = np.asarray(pred_sigma, dtype=float).reshape(-1)
    if pred_mu.shape[0] != 18 or pred_sigma.shape[0] != 18:
        raise ValueError("DQN weight state requires 18 future load predictions and 18 sigma values.")
    state_sigma = pred_sigma
    if sigma_multipliers is not None:
        multipliers = np.asarray(sigma_multipliers, dtype=float).reshape(-1)
        if multipliers.shape[0] != 18:
            raise ValueError("DQN sigma calibration requires 18 horizon multipliers.")
        state_sigma = pred_sigma * multipliers
    ramps = np.diff(pred_mu)
    pred_load_ramp_max = float(np.max(np.abs(ramps))) if ramps.size else 0.0
    pred_load_ramp_mean = float(np.mean(np.abs(ramps))) if ramps.size else 0.0
    info = {
        "SOC": float(soc),
        "SOC_ref_minus_SOC": float(soc_ref - soc),
        "dSOC": float(d_soc),
        "current_load": float(current_load),
        "pred_load_mean": float(np.mean(pred_mu)),
        "pred_load_max": float(np.max(pred_mu)),
        "pred_load_ramp_max": pred_load_ramp_max,
        "pred_load_ramp_mean": pred_load_ramp_mean,
        "sigma_mean": float(np.mean(state_sigma)),
        "sigma_max": float(np.max(state_sigma)),
        "P_fc_prev": float(p_fc_prev),
        "P_batt_prev": float(p_batt_prev),
    }
    state = np.array(
        [
            info["SOC"],
            info["SOC_ref_minus_SOC"],
            info["dSOC"],
            info["current_load"],
            info["pred_load_mean"],
            info["pred_load_max"],
            info["pred_load_ramp_max"],
            info["pred_load_ramp_mean"],
            info["sigma_mean"],
            info["sigma_max"],
            info["P_fc_prev"],
            info["P_batt_prev"],
        ],
        dtype=np.float32,
    )
    return state, info


def load_horizon_sigma_multipliers(path: str | Path, target_coverage: float = 0.90) -> np.ndarray:
    """Load 18 horizon-specific sigma multipliers from a calibration CSV."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Sigma calibration CSV not found: {csv_path}")
    selected: dict[int, float] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"target_coverage", "horizon", "sigma_multiplier_k"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Sigma calibration CSV missing columns: {sorted(missing)}")
        for row in reader:
            if abs(float(row["target_coverage"]) - float(target_coverage)) > 1.0e-9:
                continue
            horizon = int(row["horizon"])
            if 1 <= horizon <= 18:
                selected[horizon] = float(row["sigma_multiplier_k"])
    missing_horizons = [idx for idx in range(1, 19) if idx not in selected]
    if missing_horizons:
        raise ValueError(
            f"Sigma calibration CSV lacks target_coverage={float(target_coverage):g} "
            f"for horizons: {missing_horizons}"
        )
    return np.array([selected[idx] for idx in range(1, 19)], dtype=float)


def load_weight_action_config(path: str | Path) -> WeightActionConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    section = data.get("dqn", {}).get("mpc_weight_actions", {})
    return WeightActionConfig(**section)
