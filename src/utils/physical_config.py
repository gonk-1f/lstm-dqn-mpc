"""Frozen physical and normalization constants shared by formal DQN-MPC.

These values are not tunable action weights. Keep this module dependency-free.
"""
FUEL_CELL_MIN_KW = 0.0
FUEL_CELL_MAX_KW = 600.0
FUEL_CELL_RAMP_KW_PER_S = 48.0
BATTERY_CAPACITY_KWH = 624.0
BATTERY_CHARGE_MAX_KW = 624.0
BATTERY_DISCHARGE_MAX_KW = 1248.0
BATTERY_POWER_REF_KW = 624.0
SOC_MIN = 0.20
SOC_MAX = 0.80
SOC_REFERENCE = 0.55
SOC_SOFT_MIN = 0.50
SOC_SOFT_MAX = 0.60
SOC_SOFT_SCALE = 0.05
DT_SECONDS = 1.0
MPC_HORIZON = 6
