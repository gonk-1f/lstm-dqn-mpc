"""Public pure helpers for the rebuilt operating-segment data pipeline.

The historical natural-spline and load-clipping rules were removed.  The
canonical implementation is ``rebuilt_operating_dataset``.
"""

from utils.rebuilt_operating_dataset import (
    align_ais_to_power,
    chronological_parent_splits,
    collapse_battery_records,
    collapse_fc_records,
    collapse_scalar_records,
    find_contiguous_intervals,
    match_nearest_without_reuse,
    pchip_to_one_second,
    select_shore_intervals,
)

__all__ = [
    "align_ais_to_power", "chronological_parent_splits", "collapse_battery_records",
    "collapse_fc_records", "collapse_scalar_records", "find_contiguous_intervals",
    "match_nearest_without_reuse", "pchip_to_one_second", "select_shore_intervals",
]
