"""Read-only diagnostics for fixed-duration, actually executed ONBOARD steps."""

from math import isfinite
from typing import Iterable


SOC_BINS = ("below_0p4", "0p4_to_0p6", "above_0p6_below_0p79", "at_least_0p79")


def soc_time_occupancy(values: Iterable[float]) -> dict[str, object]:
    counts = dict.fromkeys(SOC_BINS, 0)
    for soc in values:
        if not isfinite(soc):
            raise ValueError("SOC diagnostic needs finite values")
        index = 0 if soc < 0.4 else 1 if soc <= 0.6 else 2 if soc < 0.79 else 3
        counts[SOC_BINS[index]] += 1
    steps = sum(counts.values())
    return {
        "counts": counts,
        "fractions": {name: count / steps if steps else None for name, count in counts.items()},
        "executed_steps": steps,
        "executed_seconds": steps * 30,
        "scope": "actual post-action SOC; all executed ONBOARD steps including failed prefixes; SHORE excluded",
        "boundaries": "SOC=0.4 and 0.6 belong to working band; SOC=0.79 belongs to top bin",
    }
