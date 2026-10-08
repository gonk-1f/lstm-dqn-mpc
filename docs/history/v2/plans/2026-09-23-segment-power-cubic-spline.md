# Segment Power Cubic-Spline Filling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fill all 140 diagnosed internal gaps in the review-only segment power curves with auditable 30-second natural-cubic estimates while preserving raw telemetry and the source-power identity.

**Architecture:** Put the interpolation math and audit metadata in a small tracked v2 data module. The existing ignored review generator consumes that module to redraw PNG/HTML artifacts and add interpolation metadata without changing measured-only statistics or `gap_diagnostics.csv`.

**Tech Stack:** Python 3, NumPy, SciPy `CubicSpline`, pandas/CSV, Matplotlib, `unittest`.

---

## File Structure

- Create `src/v2/data/power_gap_interpolation.py`: pure, testable interpolation and warning logic.
- Create `tests/test_v2_power_gap_interpolation.py`: tracked unit tests for timestamps, identities, failures, and warnings.
- Modify `.codex_tmp/segment_power_review.py`: review-only plotting and index integration; remains intentionally gitignored.
- Modify `.codex_tmp/test_segment_power_review.py`: focused generator regression tests; remains intentionally gitignored.
- Regenerate `outputs/v2_segment_power_review/`: ignored PNG, CSV, and HTML review artifacts.
- Preserve `outputs/v2_segment_power_review/gap_diagnostics.csv` byte-for-byte.

### Task 1: Add the pure gap interpolator

**Files:**
- Create: `src/v2/data/power_gap_interpolation.py`
- Create: `tests/test_v2_power_gap_interpolation.py`

- [ ] **Step 1: Write failing tests for the nominal grid, identity, and no extrapolation**

```python
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from v2.data.power_gap_interpolation import interpolate_power_gaps


def test_interpolates_only_nominal_interior_slots_and_preserves_identity():
    t0 = datetime(2024, 5, 11, 8, 20, tzinfo=timezone.utc)
    times = tuple(t0 + timedelta(seconds=s) for s in (0, 30, 60, 450, 480, 510))
    fc = np.array([90.0, 100.0, 110.0, 180.0, 190.0, 200.0])
    batt = np.array([-10.0, -12.0, -14.0, -30.0, -32.0, -34.0])

    result = interpolate_power_gaps(times, fc, batt)

    expected = tuple(times[2] + timedelta(seconds=30 * n) for n in range(1, 13))
    assert tuple(t for t, flag in zip(result.timestamps, result.is_interpolated) if flag) == expected
    np.testing.assert_allclose(result.source_kw, result.fc_kw - result.battery_raw_kw)
    assert result.timestamps[0] == times[0]
    assert result.timestamps[-1] == times[-1]
    assert result.interpolation_gap_count == 1
    assert result.interpolated_point_count == 12
    assert result.max_interpolated_gap_seconds == 390.0


def test_preserves_observed_values_exactly():
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    times = tuple(t0 + timedelta(seconds=s) for s in (0, 30, 60, 120, 150, 180))
    fc = np.array([1.0, 2.0, 3.0, 5.0, 6.0, 7.0])
    batt = np.array([-1.0, -2.0, -3.0, -5.0, -6.0, -7.0])

    result = interpolate_power_gaps(times, fc, batt)
    observed = ~result.is_interpolated

    np.testing.assert_array_equal(result.fc_kw[observed], fc)
    np.testing.assert_array_equal(result.battery_raw_kw[observed], batt)


def test_rejects_gap_with_fewer_than_four_distinct_anchors():
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    times = (t0, t0 + timedelta(seconds=390), t0 + timedelta(seconds=420))
    with pytest.raises(ValueError, match="fewer than four anchors"):
        interpolate_power_gaps(times, [10.0, 20.0, 30.0], [-1.0, -2.0, -3.0])


def test_reports_negative_fc_or_local_range_overshoot_without_clipping():
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    times = tuple(t0 + timedelta(seconds=s) for s in (0, 30, 60, 450, 480, 510))
    result = interpolate_power_gaps(
        times,
        [0.0, 200.0, 0.0, 200.0, 0.0, 200.0],
        [0.0, -100.0, 0.0, -100.0, 0.0, -100.0],
    )
    assert result.warning_messages
    assert any("OVERSHOOT" in value for value in result.warning_messages)
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```powershell
python -m pytest tests/test_v2_power_gap_interpolation.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'v2.data.power_gap_interpolation'`.

- [ ] **Step 3: Implement the minimal typed result and local natural splines**

Create `src/v2/data/power_gap_interpolation.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Sequence

import numpy as np
from scipy.interpolate import CubicSpline


@dataclass(frozen=True)
class InterpolatedPowerSeries:
    timestamps: tuple[datetime, ...]
    fc_kw: np.ndarray
    battery_raw_kw: np.ndarray
    source_kw: np.ndarray
    is_interpolated: np.ndarray
    gap_intervals: tuple[tuple[datetime, datetime], ...]
    interpolation_gap_count: int
    interpolated_point_count: int
    max_interpolated_gap_seconds: float
    warning_messages: tuple[str, ...]


def _nominal_missing_count(gap_seconds: float, step_seconds: float) -> int:
    return max(1, round(gap_seconds / step_seconds) - 1)


def interpolate_power_gaps(
    timestamps: Sequence[datetime],
    fc_kw: Sequence[float],
    battery_raw_kw: Sequence[float],
    *,
    gap_threshold_seconds: float = 45.0,
    step_seconds: float = 30.0,
    anchor_points_each_side: int = 3,
) -> InterpolatedPowerSeries:
    times = tuple(timestamps)
    fc = np.asarray(fc_kw, dtype=float)
    battery = np.asarray(battery_raw_kw, dtype=float)
    if len(times) != len(fc) or len(times) != len(battery):
        raise ValueError("timestamp and power lengths differ")
    if any(right <= left for left, right in zip(times, times[1:])):
        raise ValueError("timestamps must be strictly increasing")
    if not np.all(np.isfinite(fc)) or not np.all(np.isfinite(battery)):
        raise ValueError("observed power must be finite")

    rows: list[tuple[datetime, float, float, bool]] = []
    gaps: list[tuple[datetime, datetime]] = []
    warnings: list[str] = []
    max_gap = 0.0
    for left_index in range(len(times) - 1):
        rows.append((times[left_index], fc[left_index], battery[left_index], False))
        gap_seconds = (times[left_index + 1] - times[left_index]).total_seconds()
        if gap_seconds <= gap_threshold_seconds:
            continue
        anchor_start = max(0, left_index - anchor_points_each_side + 1)
        anchor_end = min(len(times), left_index + 1 + anchor_points_each_side)
        anchor_times = times[anchor_start:anchor_end]
        if len(anchor_times) < 4:
            raise ValueError(
                f"gap {times[left_index].isoformat()} has fewer than four anchors"
            )
        origin = anchor_times[0]
        x = np.array([(value - origin).total_seconds() for value in anchor_times])
        fc_anchor = fc[anchor_start:anchor_end]
        battery_anchor = battery[anchor_start:anchor_end]
        fc_spline = CubicSpline(x, fc_anchor, bc_type="natural")
        battery_spline = CubicSpline(x, battery_anchor, bc_type="natural")
        missing_count = _nominal_missing_count(gap_seconds, step_seconds)
        generated = tuple(
            times[left_index] + timedelta(seconds=step_seconds * number)
            for number in range(1, missing_count + 1)
        )
        generated = tuple(value for value in generated if value < times[left_index + 1])
        generated_x = np.array([(value - origin).total_seconds() for value in generated])
        generated_fc = np.asarray(fc_spline(generated_x), dtype=float)
        generated_battery = np.asarray(battery_spline(generated_x), dtype=float)
        if not np.all(np.isfinite(generated_fc)) or not np.all(np.isfinite(generated_battery)):
            raise ValueError("cubic spline produced non-finite power")
        anchor_fc_min, anchor_fc_max = float(np.min(fc_anchor)), float(np.max(fc_anchor))
        anchor_batt_min = float(np.min(battery_anchor))
        anchor_batt_max = float(np.max(battery_anchor))
        for timestamp, fc_value, batt_value in zip(
            generated, generated_fc, generated_battery
        ):
            if fc_value < 0.0 or not anchor_fc_min <= fc_value <= anchor_fc_max:
                warnings.append(f"{timestamp.isoformat()}:FC_OVERSHOOT")
            if not anchor_batt_min <= batt_value <= anchor_batt_max:
                warnings.append(f"{timestamp.isoformat()}:BATTERY_OVERSHOOT")
            rows.append((timestamp, float(fc_value), float(batt_value), True))
        gaps.append((times[left_index], times[left_index + 1]))
        max_gap = max(max_gap, gap_seconds)
    if times:
        rows.append((times[-1], fc[-1], battery[-1], False))
    rows.sort(key=lambda row: row[0])
    out_times = tuple(row[0] for row in rows)
    out_fc = np.array([row[1] for row in rows])
    out_battery = np.array([row[2] for row in rows])
    mask = np.array([row[3] for row in rows], dtype=bool)
    return InterpolatedPowerSeries(
        timestamps=out_times,
        fc_kw=out_fc,
        battery_raw_kw=out_battery,
        source_kw=out_fc - out_battery,
        is_interpolated=mask,
        gap_intervals=tuple(gaps),
        interpolation_gap_count=len(gaps),
        interpolated_point_count=int(np.count_nonzero(mask)),
        max_interpolated_gap_seconds=max_gap,
        warning_messages=tuple(warnings),
    )
```

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run:

```powershell
python -m pytest tests/test_v2_power_gap_interpolation.py -q
```

Expected: `4 passed`.

- [ ] **Step 5: Commit the pure interpolator**

```powershell
git add src/v2/data/power_gap_interpolation.py tests/test_v2_power_gap_interpolation.py
git commit -m "feat(v2): add review gap interpolation"
```

### Task 2: Integrate interpolation into the review generator

**Files:**
- Modify: `.codex_tmp/test_segment_power_review.py`
- Modify: `.codex_tmp/segment_power_review.py`

- [ ] **Step 1: Add a failing integration-focused test**

Add the imports and test:

```python
from v2.data.power_gap_interpolation import interpolate_power_gaps
from segment_power_review import interpolation_metadata


def test_filled_review_series_marks_generated_points_and_preserves_source_identity(self):
    start = datetime(2024, 5, 11, 8, 20, tzinfo=timezone.utc)
    timestamps = tuple(start + timedelta(seconds=s) for s in (0, 30, 60, 450, 480, 510))
    result = interpolate_power_gaps(
        timestamps,
        [90.0, 100.0, 110.0, 180.0, 190.0, 200.0],
        [-10.0, -12.0, -14.0, -30.0, -32.0, -34.0],
    )

    self.assertEqual(result.interpolation_gap_count, 1)
    self.assertEqual(result.interpolated_point_count, 12)
    np.testing.assert_allclose(result.source_kw, result.fc_kw - result.battery_raw_kw)
    self.assertEqual(
        interpolation_metadata(result)["interpolation_method"],
        "local_natural_cubic_fc_battery_30s_source_derived",
    )
```

- [ ] **Step 2: Run the temporary focused suite and verify RED before changing the generator import path**

Run:

```powershell
python .codex_tmp/test_segment_power_review.py
```

Expected: import fails because `segment_power_review.interpolation_metadata` does not exist.

- [ ] **Step 3: Replace endpoint-only gap bridging with filled-series plotting**

In `.codex_tmp/segment_power_review.py`:

```python
from v2.data.power_gap_interpolation import interpolate_power_gaps


def interpolation_metadata(filled):
    return {
        "interpolated_point_count": filled.interpolated_point_count,
        "interpolation_gap_count": filled.interpolation_gap_count,
        "max_interpolated_gap_seconds": _number(filled.max_interpolated_gap_seconds),
        "interpolation_method": "local_natural_cubic_fc_battery_30s_source_derived",
        "interpolation_warning_count": len(filled.warning_messages),
        "gap_bridge_policy": "cubic_spline_imputed_review_only",
    }
```

Immediately after `battery_raw, source = power_conventions(fc, battery_bus)`, add:

```python
filled = interpolate_power_gaps(timestamps, fc, battery_raw)
```

Keep `_stats(source)`, `_stats(fc)`, and `_stats(battery_raw)` before any reassignment so index statistics remain measured-only. Replace the NaN-break and dashed-bridge loop with:

```python
for gap_index, (left, right) in enumerate(filled.gap_intervals):
    ax.axvspan(
        left,
        right,
        color="#7F8C8D",
        alpha=0.08,
        linewidth=0,
        label="Cubic-spline imputed interval" if gap_index == 0 else "_nolegend_",
    )

for values, label, color, width, alpha in (
    (filled.source_kw, "Source total", "#0B3C5D", 1.8, 1.0),
    (filled.fc_kw, "FC total", "#D95F02", 1.15, 0.85),
    (
        filled.battery_raw_kw,
        "Battery raw total (negative=discharge, positive=charge)",
        "#1B9E77",
        1.05,
        0.85,
    ),
):
    ax.plot(filled.timestamps, values, label=label, color=color, linewidth=width, alpha=alpha)
    ax.scatter(
        np.asarray(filled.timestamps, dtype=object)[filled.is_interpolated],
        values[filled.is_interpolated],
        facecolors="none",
        edgecolors=color,
        linewidths=0.65,
        s=13,
        alpha=0.8,
    )
```

Update the summary text to include measured and interpolated counts. Remove calls to `break_long_gaps` and `long_gap_bridges` from `_plot_parent`; leave the helpers in place only until their legacy unit tests are replaced.

- [ ] **Step 4: Add index metadata and update the HTML notice**

Add to each OK row:

```python
**interpolation_metadata(filled),
```

Add the five new names next to the existing gap fields in `FIELDNAMES`. Replace the HTML notice with:

```html
<p class="notice">HUMAN SEGMENT REVIEW ONLY — hollow markers and shaded intervals are local natural-cubic estimates on a nominal 30 s grid; they are not measured telemetry and are not a formal dataset.</p>
```

- [ ] **Step 5: Run both focused suites**

Run:

```powershell
python -m pytest tests/test_v2_power_gap_interpolation.py -q
python .codex_tmp/test_segment_power_review.py
```

Expected: interpolation tests pass; generator tests pass after obsolete NaN-break/bridge expectations are removed or rewritten for the filled-series contract.

### Task 3: Regenerate all review artifacts and prove accounting

**Files:**
- Regenerate: `outputs/v2_segment_power_review/2024-*/**.png`
- Regenerate: `outputs/v2_segment_power_review/segment_power_index.csv`
- Regenerate: `outputs/v2_segment_power_review/index.html`
- Preserve: `outputs/v2_segment_power_review/gap_diagnostics.csv`

- [ ] **Step 1: Hash the raw-evidence diagnostics before regeneration**

```powershell
Get-FileHash outputs\v2_segment_power_review\gap_diagnostics.csv -Algorithm SHA256
```

Record the hash in the execution notes.

- [ ] **Step 2: Regenerate all 66 parent plots**

```powershell
python .codex_tmp\segment_power_review.py `
  --raw-root 'C:\Users\20883\OneDrive\Desktop\氢舟一号' `
  --output-root 'outputs\v2_segment_power_review'
```

Expected final line: `parents=66 png_ok=66 failed=0`.

- [ ] **Step 3: Validate all 140 gaps and expected nominal point count**

Run a read-only CSV check that asserts:

```python
assert len(rows) == 66
assert sum(int(row["interpolation_gap_count"]) for row in rows) == 140
assert sum(int(row["interpolated_point_count"]) for row in rows) == 1655
assert all(row["interpolation_method"] == "local_natural_cubic_fc_battery_30s_source_derived" for row in rows)
assert all(row["plot_status"] == "OK" for row in rows)
```

The expected 1,655 points are derived from the diagnosed gap durations with `round(gap_seconds / 30) - 1`, preventing a synthetic point from being placed only 1–10 seconds before an observed right boundary.

- [ ] **Step 4: Verify the diagnostics file is unchanged**

```powershell
Get-FileHash outputs\v2_segment_power_review\gap_diagnostics.csv -Algorithm SHA256
```

Expected: identical SHA-256 to Step 1.

- [ ] **Step 5: Inspect representative short and long gaps**

Open the regenerated 5 May 8 plot containing the 60-second alignment gap, the May 11 plot containing the 420-second gap, and one 600/630-second example. Confirm hollow markers appear, the curve is connected, and the legend labels estimates as non-measured.

### Task 4: Run final regression and repository checks

**Files:**
- Verify only; no new files.

- [ ] **Step 1: Run focused tests**

```powershell
python -m pytest tests/test_v2_power_gap_interpolation.py -q
python .codex_tmp/test_segment_power_review.py
python .codex_tmp/test_gap_diagnostics.py
```

Expected: all pass.

- [ ] **Step 2: Run compile/import checks**

```powershell
python -m py_compile `
  src/v2/data/power_gap_interpolation.py `
  tests/test_v2_power_gap_interpolation.py `
  .codex_tmp/segment_power_review.py `
  .codex_tmp/test_segment_power_review.py
python -c "from v2.data.power_gap_interpolation import interpolate_power_gaps; print('import-ok')"
```

Expected: `import-ok` and no compile errors.

- [ ] **Step 3: Run repository hygiene checks**

```powershell
git diff --check
git status --short --branch
```

Expected: only intentional tracked interpolation code/tests/plan commits; ignored review artifacts do not enter Git.

- [ ] **Step 4: Report without expanding scope**

Report parent/PNG counts, 140 gaps, 1,655 generated points, warning count, representative visual checks, output paths, tests, and commit SHA. Explicitly state that raw telemetry, diagnostics, dataset split, and preflight were not modified.
