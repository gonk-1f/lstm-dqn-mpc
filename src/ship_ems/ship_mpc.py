from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from casadi import SX, nlpsol, vertcat
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "casadi is required for the ship MPC. Install it in the active interpreter."
    ) from exc


@dataclass
class MPCCommand:
    battery_ref_kw: float
    fuel_cell_ref_kw: float
    slack_pos_kw: float
    slack_neg_kw: float
    predicted_soc: float


class ShipEnergyMPC:
    def __init__(self, config: dict):
        self.cfg = config
        self.horizon = int(config["prediction_horizon"])
        self.dt_hours = float(config["dt_hours"])
        self.battery_capacity_kwh = float(config["battery_capacity_kwh"])
        self.fc_min_kw = float(config["fuel_cell_min_kw"])
        self.fc_max_kw = float(config["fuel_cell_max_kw"])
        self.fc_ramp_kw = float(config["fuel_cell_ramp_kw"])
        self.batt_charge_max_kw = float(config["battery_charge_max_kw"])
        self.batt_discharge_max_kw = float(config["battery_discharge_max_kw"])
        self.soc_min = float(config["soc_min"])
        self.soc_max = float(config["soc_max"])
        self.soc_target = float(config["soc_target"])
        self.weights = config["weights"]
        self._build_solver()

    def _build_solver(self) -> None:
        N = self.horizon
        self.soc = SX.sym("soc", N + 1)
        self.p_batt = SX.sym("p_batt", N)
        self.p_fc = SX.sym("p_fc", N)
        self.slack_pos = SX.sym("slack_pos", N)
        self.slack_neg = SX.sym("slack_neg", N)
        self.load = SX.sym("load", N)
        self.prev_fc = SX.sym("prev_fc")
        self.soc0 = SX.sym("soc0")

        x = vertcat(self.soc, self.p_batt, self.p_fc, self.slack_pos, self.slack_neg)
        p = vertcat(self.load, self.prev_fc, self.soc0)

        g = []
        lbg = []
        ubg = []
        J = 0
        w = self.weights

        g.append(self.soc[0] - self.soc0)
        lbg.append(0.0)
        ubg.append(0.0)

        for k in range(N):
            next_soc = self.soc[k] - (self.p_batt[k] * self.dt_hours / self.battery_capacity_kwh)
            g.append(self.soc[k + 1] - next_soc)
            lbg.append(0.0)
            ubg.append(0.0)

            g.append(self.p_fc[k] + self.p_batt[k] + self.slack_pos[k] - self.slack_neg[k] - self.load[k])
            lbg.append(0.0)
            ubg.append(0.0)

            delta_fc = self.p_fc[k] - (self.prev_fc if k == 0 else self.p_fc[k - 1])
            g.append(delta_fc)
            lbg.append(-self.fc_ramp_kw)
            ubg.append(self.fc_ramp_kw)

            J += w["fuel_cell"] * self.p_fc[k] ** 2
            J += w["battery"] * self.p_batt[k] ** 2
            J += w["tracking"] * (self.slack_pos[k] ** 2 + self.slack_neg[k] ** 2)
            J += w["ramp"] * delta_fc ** 2
            J += w["soc"] * (self.soc[k + 1] - self.soc_target) ** 2

        lbx = []
        ubx = []
        lbx.extend([self.soc_min] * (N + 1))
        ubx.extend([self.soc_max] * (N + 1))
        lbx.extend([-self.batt_charge_max_kw] * N)
        ubx.extend([self.batt_discharge_max_kw] * N)
        lbx.extend([self.fc_min_kw] * N)
        ubx.extend([self.fc_max_kw] * N)
        lbx.extend([0.0] * N)
        ubx.extend([1.0e4] * N)
        lbx.extend([0.0] * N)
        ubx.extend([1.0e4] * N)

        opts = {"ipopt.print_level": 0, "print_time": 0}
        self.solver = nlpsol("ship_mpc", "ipopt", {"f": J, "x": x, "g": vertcat(*g), "p": p}, opts)
        self.lbx = np.asarray(lbx, dtype=float)
        self.ubx = np.asarray(ubx, dtype=float)
        self.lbg = np.asarray(lbg, dtype=float)
        self.ubg = np.asarray(ubg, dtype=float)

    def solve(self, current_soc: float, prev_fc_kw: float, load_forecast_kw: np.ndarray) -> MPCCommand:
        load_forecast_kw = np.asarray(load_forecast_kw, dtype=float).reshape(-1)
        if load_forecast_kw.shape[0] != self.horizon:
            raise ValueError(f"Expected horizon {self.horizon}, got {load_forecast_kw.shape[0]}")

        x0 = np.zeros_like(self.lbx)
        x0[0 : self.horizon + 1] = current_soc
        x0[self.horizon + 1 : self.horizon + 1 + self.horizon] = np.clip(
            load_forecast_kw * 0.3, -self.batt_charge_max_kw, self.batt_discharge_max_kw
        )
        x0[self.horizon + 1 + self.horizon : self.horizon + 1 + 2 * self.horizon] = np.clip(
            load_forecast_kw * 0.7, self.fc_min_kw, self.fc_max_kw
        )

        params = np.concatenate([load_forecast_kw, [prev_fc_kw, current_soc]])
        solution = self.solver(x0=x0, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg, p=params)
        values = np.asarray(solution["x"]).reshape(-1)

        offset = 0
        soc = values[offset : offset + self.horizon + 1]
        offset += self.horizon + 1
        p_batt = values[offset : offset + self.horizon]
        offset += self.horizon
        p_fc = values[offset : offset + self.horizon]
        offset += self.horizon
        slack_pos = values[offset : offset + self.horizon]
        offset += self.horizon
        slack_neg = values[offset : offset + self.horizon]

        return MPCCommand(
            battery_ref_kw=float(p_batt[0]),
            fuel_cell_ref_kw=float(p_fc[0]),
            slack_pos_kw=float(slack_pos[0]),
            slack_neg_kw=float(slack_neg[0]),
            predicted_soc=float(soc[1]),
        )
