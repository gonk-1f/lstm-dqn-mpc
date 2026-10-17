"""Read-only diagnostics for actually executed ONBOARD intervals."""

from math import fsum, isfinite
from typing import Iterable


SOC_BINS = ("below_0p4", "0p4_to_0p6", "above_0p6_below_0p79", "at_least_0p79")


def duration_weighted_mean(values: Iterable[float], durations_s: Iterable[float]) -> float | None:
    values, durations = tuple(values), tuple(durations_s)
    if (len(values) != len(durations)
            or any(not isfinite(value) for value in values)
            or any(not isfinite(duration) or duration <= 0.0 for duration in durations)):
        raise ValueError('time statistics need finite values and matching positive durations')
    if not values:
        return None
    # Preserve the historical arithmetic exactly when every interval is equal.
    if all(duration == durations[0] for duration in durations):
        return fsum(values) / len(values)
    return fsum(value * duration for value, duration in zip(values, durations)) / fsum(durations)


def soc_time_occupancy(values: Iterable[float], durations_s: Iterable[float] | None = None) -> dict[str, object]:
    values = tuple(values)
    durations = None if durations_s is None else tuple(durations_s)
    if durations is not None and (len(durations) != len(values)
            or any(not isfinite(duration) or duration <= 0.0 for duration in durations)):
        raise ValueError('SOC time diagnostics need matching finite positive durations')
    counts = dict.fromkeys(SOC_BINS, 0)
    bin_durations = {name: [] for name in SOC_BINS}
    for index, soc in enumerate(values):
        if not isfinite(soc):
            raise ValueError("SOC diagnostic needs finite values")
        bin_index = 0 if soc < 0.4 else 1 if soc <= 0.6 else 2 if soc < 0.79 else 3
        counts[SOC_BINS[bin_index]] += 1
        if durations is not None:
            bin_durations[SOC_BINS[bin_index]].append(durations[index])
    steps = sum(counts.values())
    result = {
        "counts": counts,
        "fractions": {name: count / steps if steps else None for name, count in counts.items()},
        "executed_steps": steps,
        "executed_seconds": steps * 30,
        "scope": "actual post-action SOC; all executed ONBOARD steps including failed prefixes; SHORE excluded",
        "boundaries": "SOC=0.4 and 0.6 belong to working band; SOC=0.79 belongs to top bin",
    }
    if durations is not None:
        seconds = fsum(durations)
        seconds_by_bin = {name: fsum(values) for name, values in bin_durations.items()}
        result.update(
            executed_seconds=seconds,
            seconds_by_bin=seconds_by_bin,
            fractions=({name: value / seconds if seconds else None for name, value in seconds_by_bin.items()}
                       if durations and any(duration != durations[0] for duration in durations) else
                       {name: count / steps if steps else None for name, count in counts.items()}),
        )
    return result
