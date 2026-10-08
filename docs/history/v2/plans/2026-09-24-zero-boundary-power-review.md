# Zero-Boundary Power Review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate 53 independently scaled total-power figures and a split-grouped HTML review for the frozen zero-boundary v2 dataset without changing any source data.

**Architecture:** A focused `v2.data` module validates the frozen manifest, verifies every source hash and time-series contract, computes per-segment statistics and y-limits, and renders PNG/CSV/HTML review artifacts. A thin `src/main` command fixes the production paths and guards replacement to the exact repository output subtree. Tests use a three-segment synthetic dataset so rendering and failure contracts remain fast and deterministic.

**Tech Stack:** Python 3, pandas, NumPy, Matplotlib Agg, `unittest`, HTML/CSS.

---

## File structure

- Create `src/v2/data/zero_boundary_power_review.py`: validation, statistics, per-segment plotting, review manifest, and HTML index.
- Create `src/main/build_zero_boundary_power_review.py`: repository-path CLI and safe replacement guard.
- Create `tests/test_v2_zero_boundary_power_review.py`: synthetic contract and rendering tests.
- Create `outputs/v2_zero_boundary_power_review/`: 53 PNGs, `review_manifest.csv`, and `index.html` generated only after tests pass.
- Preserve `data/processed/operating_dataset_zero_boundary_v2/` and `outputs/v2_segment_power_review/` byte-for-byte.

### Task 1: Validate sources and compute independent y-limits

**Files:**
- Create: `tests/test_v2_zero_boundary_power_review.py`
- Create: `src/v2/data/zero_boundary_power_review.py`

- [ ] **Step 1: Write failing validation and scale tests**

Create the test fixture and first tests:

```python
from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from v2.data.zero_boundary_power_review import (  # noqa: E402
    axis_limits,
    load_review_entries,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


class ZeroBoundaryPowerReviewTests(unittest.TestCase):
    def make_dataset(self, root: Path) -> Path:
        dataset = root / "dataset"
        rows = []
        cases = (
            ("train", "zero_boundary_001", "parent_train", [0.0, 10.0, -2.0, 0.0]),
            ("validation", "zero_boundary_002", "parent_validation", [0.0, 100.0, 50.0, 0.0]),
            ("test", "zero_boundary_003", "parent_test", [0.0, 400.0, 200.0, 0.0]),
        )
        for split, sample_id, parent, loads in cases:
            relative = Path(split) / f"{sample_id}.csv"
            path = dataset / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {
                    "timestamp": pd.date_range("2024-01-01", periods=4, freq="s", tz="Asia/Shanghai"),
                    "time_s": np.arange(4, dtype=float),
                    "load_total_kw": loads,
                }
            ).to_csv(path, index=False)
            rows.append(
                {
                    "parent": parent,
                    "sample_id": sample_id,
                    "relative_path": relative.as_posix(),
                    "split": split,
                    "point_count_1s": 4,
                    "start_timestamp": "2024-01-01T00:00:00+08:00",
                    "end_timestamp": "2024-01-01T00:00:03+08:00",
                    "duration_s": 3.0,
                    "sha256": sha256(path),
                }
            )
        metadata = dataset / "metadata"
        metadata.mkdir()
        pd.DataFrame(rows).to_csv(metadata / "sample_manifest.csv", index=False)
        return dataset

    def test_axis_limits_use_each_segments_own_range_and_include_zero(self):
        low = axis_limits(np.asarray([0.0, 10.0, -2.0, 0.0]))
        high = axis_limits(np.asarray([0.0, 400.0, 200.0, 0.0]))
        self.assertEqual(low, (-2.6, 10.6))
        self.assertEqual(high, (-20.0, 420.0))
        self.assertNotEqual(low, high)
        self.assertEqual(axis_limits(np.zeros(3)), (-1.0, 1.0))

    def test_load_entries_verifies_counts_hashes_and_retains_negative_values(self):
        with tempfile.TemporaryDirectory() as temp:
            dataset = self.make_dataset(Path(temp))
            entries = load_review_entries(
                dataset,
                expected_split_counts={"train": 1, "validation": 1, "test": 1},
            )
            self.assertEqual([entry.split for entry in entries], ["train", "validation", "test"])
            self.assertEqual(entries[0].minimum_kw, -2.0)
            self.assertEqual(entries[0].point_count, 4)
            self.assertEqual(entries[0].duration_s, 3.0)

            source = dataset / entries[0].source_relative_path
            source.write_text(source.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_review_entries(
                    dataset,
                    expected_split_counts={"train": 1, "validation": 1, "test": 1},
                )
```

- [ ] **Step 2: Run the tests and verify the module is missing**

Run:

```powershell
python -X utf8 -B -m unittest tests.test_v2_zero_boundary_power_review -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'v2.data.zero_boundary_power_review'`.

- [ ] **Step 3: Implement source validation and y-limits**

Create `src/v2/data/zero_boundary_power_review.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd


FROZEN_SPLIT_COUNTS = {"train": 38, "validation": 10, "test": 5}
SPLIT_ORDER = ("train", "validation", "test")
REQUIRED_COLUMNS = ("timestamp", "time_s", "load_total_kw")


@dataclass(frozen=True)
class ReviewEntry:
    parent: str
    sample_id: str
    split: str
    source_relative_path: str
    source_sha256: str
    point_count: int
    duration_s: float
    minimum_kw: float
    maximum_kw: float
    mean_kw: float
    y_min_kw: float
    y_max_kw: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def axis_limits(values: np.ndarray) -> tuple[float, float]:
    data = np.asarray(values, dtype=float)
    if data.ndim != 1 or data.size == 0 or not np.isfinite(data).all():
        raise ValueError("power values must be a non-empty finite vector")
    lower = min(0.0, float(data.min()))
    upper = max(0.0, float(data.max()))
    span = upper - lower
    if span == 0.0:
        return (-1.0, 1.0)
    pad = 0.05 * span
    return (lower - pad, upper + pad)


def _load_series(path: Path, expected_points: int, expected_duration: float) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=list(REQUIRED_COLUMNS))
    time_s = pd.to_numeric(frame.time_s, errors="coerce").to_numpy(dtype=float)
    load = pd.to_numeric(frame.load_total_kw, errors="coerce").to_numpy(dtype=float)
    if len(frame) != expected_points or len(frame) < 2:
        raise ValueError(f"point count mismatch: {path}")
    if not np.isfinite(time_s).all() or not np.isfinite(load).all():
        raise ValueError(f"non-finite series: {path}")
    if time_s[0] != 0.0 or not np.allclose(np.diff(time_s), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError(f"time_s is not an exact one-second axis: {path}")
    if time_s[-1] != expected_duration:
        raise ValueError(f"duration mismatch: {path}")
    if load[0] != 0.0 or load[-1] != 0.0:
        raise ValueError(f"zero endpoint contract failed: {path}")
    return frame


def load_review_entries(
    dataset_root: Path,
    *,
    expected_split_counts: dict[str, int] | None = None,
) -> list[ReviewEntry]:
    root = Path(dataset_root)
    expected = FROZEN_SPLIT_COUNTS if expected_split_counts is None else expected_split_counts
    manifest = pd.read_csv(root / "metadata" / "sample_manifest.csv")
    if manifest.groupby("split").size().to_dict() != expected:
        raise ValueError("manifest split counts do not match the frozen contract")
    entries: list[ReviewEntry] = []
    for split in SPLIT_ORDER:
        for row in manifest.loc[manifest.split.eq(split)].itertuples(index=False):
            path = root / str(row.relative_path)
            if _sha256(path) != str(row.sha256):
                raise ValueError(f"source SHA-256 mismatch: {path}")
            frame = _load_series(path, int(row.point_count_1s), float(row.duration_s))
            values = frame.load_total_kw.to_numpy(dtype=float)
            y_min, y_max = axis_limits(values)
            entries.append(ReviewEntry(
                parent=str(row.parent), sample_id=str(row.sample_id), split=split,
                source_relative_path=str(row.relative_path), source_sha256=str(row.sha256),
                point_count=len(frame), duration_s=float(row.duration_s),
                minimum_kw=float(values.min()), maximum_kw=float(values.max()),
                mean_kw=float(values.mean()), y_min_kw=y_min, y_max_kw=y_max,
            ))
    return entries
```

- [ ] **Step 4: Run focused tests**

Run the same unittest command. Expected: 2 tests pass.

- [ ] **Step 5: Commit validation layer**

```powershell
git add -- tests/test_v2_zero_boundary_power_review.py src/v2/data/zero_boundary_power_review.py
git commit -m "feat(v2): validate power review sources"
```

### Task 2: Render PNGs, review CSV, and grouped HTML

**Files:**
- Modify: `tests/test_v2_zero_boundary_power_review.py`
- Modify: `src/v2/data/zero_boundary_power_review.py`

- [ ] **Step 1: Add failing artifact-generation test**

Import `build_review` and append:

```python
    def test_build_review_writes_split_plots_manifest_and_index(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset = self.make_dataset(root)
            output = root / "review"
            rows = build_review(
                dataset,
                output,
                expected_split_counts={"train": 1, "validation": 1, "test": 1},
            )
            self.assertEqual(len(rows), 3)
            self.assertEqual(len(list(output.rglob("*.png"))), 3)
            for split, sample_id in (
                ("train", "zero_boundary_001"),
                ("validation", "zero_boundary_002"),
                ("test", "zero_boundary_003"),
            ):
                self.assertTrue((output / split / f"{sample_id}.png").is_file())
            review = pd.read_csv(output / "review_manifest.csv")
            self.assertEqual(review.split.tolist(), ["train", "validation", "test"])
            self.assertEqual(float(review.loc[0, "minimum_kw"]), -2.0)
            self.assertNotEqual(float(review.loc[0, "y_max_kw"]), float(review.loc[2, "y_max_kw"]))
            html = (output / "index.html").read_text(encoding="utf-8")
            self.assertLess(html.index('id="train"'), html.index('id="validation"'))
            self.assertLess(html.index('id="validation"'), html.index('id="test"'))
            self.assertIn("每个航段采用独立纵轴", html)
```

- [ ] **Step 2: Run the test and verify `build_review` is absent**

Run the focused test file. Expected: import failure for `build_review`.

- [ ] **Step 3: Implement rendering**

Add imports `csv`, `html.escape`, `shutil`, and `matplotlib.pyplot as plt`. Add these functions:

```python
def _plot_entry(dataset_root: Path, output_root: Path, entry: ReviewEntry) -> str:
    frame = pd.read_csv(dataset_root / entry.source_relative_path, usecols=list(REQUIRED_COLUMNS))
    image_relative = Path(entry.split) / f"{entry.sample_id}.png"
    image_path = output_root / image_relative
    image_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12.0, 4.8), constrained_layout=True)
    ax.plot(frame.time_s, frame.load_total_kw, color="#0B3C5D", linewidth=1.0)
    ax.axhline(0.0, color="#666666", linewidth=0.8, linestyle="--")
    ax.scatter(
        [float(frame.time_s.iloc[0]), float(frame.time_s.iloc[-1])],
        [float(frame.load_total_kw.iloc[0]), float(frame.load_total_kw.iloc[-1])],
        color="#A61B1B", s=18, zorder=3, label="Zero-power endpoints",
    )
    ax.set_xlim(0.0, entry.duration_s)
    ax.set_ylim(entry.y_min_kw, entry.y_max_kw)
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel("Total power (kW)")
    ax.grid(True, linewidth=0.45, alpha=0.25)
    ax.set_title(f"{entry.parent} | {entry.sample_id} | {entry.split.upper()}", loc="left")
    fig.suptitle(
        f"duration={entry.duration_s:.0f} s | points={entry.point_count:,} | "
        f"min={entry.minimum_kw:.2f} kW | max={entry.maximum_kw:.2f} kW | "
        f"mean={entry.mean_kw:.2f} kW",
        fontsize=9, x=0.99, ha="right", color="#4B5563",
    )
    fig.savefig(image_path, dpi=140, metadata={"Software": "zero-boundary power review"})
    plt.close(fig)
    return image_relative.as_posix()


def _write_index(output_root: Path, rows: list[dict[str, object]]) -> None:
    sections = []
    for split in SPLIT_ORDER:
        cards = []
        for row in (item for item in rows if item["split"] == split):
            image = escape(str(row["image_relative_path"]), quote=True)
            title = escape(f'{row["parent"]} | {row["sample_id"]}')
            cards.append(
                f'<article><h3>{title}</h3><p>duration={row["duration_s"]:.0f} s | '
                f'points={row["point_count"]:,} | min={row["minimum_kw"]:.2f} kW | '
                f'max={row["maximum_kw"]:.2f} kW | mean={row["mean_kw"]:.2f} kW</p>'
                f'<a href="{image}"><img src="{image}" loading="lazy" alt="{title}"></a></article>'
            )
        sections.append(f'<section id="{split}"><h2>{split.upper()}</h2>{"".join(cards)}</section>')
    document = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>Zero-boundary v2 total-power review</title><style>
body{font-family:Arial,"Microsoft YaHei",sans-serif;margin:24px;background:#f4f6f8;color:#17202a}
nav a{margin-right:18px}section{margin-top:30px}article{background:#fff;padding:14px;margin:16px 0;border-radius:7px;box-shadow:0 1px 5px #ccd1d1}
h3{font-size:17px;margin:0 0 4px}p{font-size:13px;color:#566573}img{width:100%;height:auto;border:1px solid #ddd}
</style></head><body><h1>Zero-boundary v2 total-power review</h1>
<p>每个航段采用独立纵轴；图像高度不可用于跨航段比较功率幅值。数据按正式清单原值绘制。</p>
<nav><a href="#train">Train</a><a href="#validation">Validation</a><a href="#test">Test</a></nav>
""" + "".join(sections) + "</body></html>"
    (output_root / "index.html").write_text(document, encoding="utf-8")


def build_review(
    dataset_root: Path,
    output_root: Path,
    *,
    expected_split_counts: dict[str, int] | None = None,
) -> list[dict[str, object]]:
    output = Path(output_root)
    if output.exists():
        raise FileExistsError(f"review output already exists: {output}")
    entries = load_review_entries(dataset_root, expected_split_counts=expected_split_counts)
    output.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    try:
        for entry in entries:
            image = _plot_entry(Path(dataset_root), output, entry)
            rows.append({**entry.__dict__, "image_relative_path": image})
        pd.DataFrame(rows).to_csv(output / "review_manifest.csv", index=False, encoding="utf-8-sig")
        _write_index(output, rows)
    except Exception:
        shutil.rmtree(output)
        raise
    return rows
```

- [ ] **Step 4: Run focused tests**

Run the focused file. Expected: 3 tests pass and Matplotlib closes all figures.

- [ ] **Step 5: Commit renderer**

```powershell
git add -- tests/test_v2_zero_boundary_power_review.py src/v2/data/zero_boundary_power_review.py
git commit -m "feat(v2): render final power review"
```

### Task 3: Add the guarded production command

**Files:**
- Modify: `tests/test_v2_zero_boundary_power_review.py`
- Create: `src/main/build_zero_boundary_power_review.py`

- [ ] **Step 1: Add failing output-path and overwrite tests**

Import `validated_output_path` and append:

```python
    def test_production_output_guard_and_overwrite_refusal(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            outputs = root / "outputs"
            outputs.mkdir()
            accepted = validated_output_path(root, outputs / "review")
            self.assertEqual(accepted, (outputs / "review").resolve())
            with self.assertRaisesRegex(ValueError, "repository outputs"):
                validated_output_path(root, root / "data" / "review")

            dataset = self.make_dataset(root)
            target = outputs / "review"
            build_review(
                dataset,
                target,
                expected_split_counts={"train": 1, "validation": 1, "test": 1},
            )
            with self.assertRaises(FileExistsError):
                build_review(
                    dataset,
                    target,
                    expected_split_counts={"train": 1, "validation": 1, "test": 1},
                )
```

- [ ] **Step 2: Run the focused test and verify the CLI module is absent**

Expected: import failure for `main.build_zero_boundary_power_review`.

- [ ] **Step 3: Implement the CLI**

Create:

```python
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from v2.data.zero_boundary_power_review import build_review  # noqa: E402


DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "processed" / "operating_dataset_zero_boundary_v2"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "v2_zero_boundary_power_review"


def validated_output_path(project_root: Path, output_root: Path) -> Path:
    root = Path(project_root).resolve()
    outputs = (root / "outputs").resolve()
    target = Path(output_root).resolve()
    try:
        relative = target.relative_to(outputs)
    except ValueError as exc:
        raise ValueError("review output must stay under repository outputs") from exc
    if not relative.parts:
        raise ValueError("repository outputs itself cannot be replaced")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Build final zero-boundary total-power review")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    target = validated_output_path(PROJECT_ROOT, args.output_root)
    if target.exists() and args.replace:
        shutil.rmtree(target)
    rows = build_review(args.dataset_root, target)
    print(f"generated {len(rows)} segment figures at {target}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run focused tests and compile/import checks**

```powershell
python -X utf8 -B -m unittest tests.test_v2_zero_boundary_power_review -v
python -X utf8 -B -m compileall -q src\v2\data\zero_boundary_power_review.py src\main\build_zero_boundary_power_review.py tests\test_v2_zero_boundary_power_review.py
python -X utf8 -B -c "import sys; from pathlib import Path; sys.path.insert(0,str(Path.cwd()/'src')); from v2.data.zero_boundary_power_review import build_review; from main.build_zero_boundary_power_review import validated_output_path; print('power_review_imports_ok')"
```

Expected: 4 tests pass, compilation exits zero, and import prints `power_review_imports_ok`.

- [ ] **Step 5: Commit CLI**

```powershell
git add -- tests/test_v2_zero_boundary_power_review.py src/main/build_zero_boundary_power_review.py
git commit -m "feat(v2): add power review command"
```

### Task 4: Generate and independently verify the 53-figure review

**Files:**
- Create: `outputs/v2_zero_boundary_power_review/index.html`
- Create: `outputs/v2_zero_boundary_power_review/review_manifest.csv`
- Create: `outputs/v2_zero_boundary_power_review/train/*.png` (38)
- Create: `outputs/v2_zero_boundary_power_review/validation/*.png` (10)
- Create: `outputs/v2_zero_boundary_power_review/test/*.png` (5)

- [ ] **Step 1: Hash protected source trees**

```powershell
New-Item -ItemType Directory -Force .codex_tmp | Out-Null
Get-ChildItem data\processed\operating_dataset_zero_boundary_v2 -Recurse -File | Get-FileHash -Algorithm SHA256 | Export-Csv .codex_tmp\zero_boundary_dataset_before_review.csv -NoTypeInformation -Encoding utf8
Get-ChildItem outputs\v2_segment_power_review -Recurse -File | Get-FileHash -Algorithm SHA256 | Export-Csv .codex_tmp\old_power_review_before.csv -NoTypeInformation -Encoding utf8
```

Expected: both baseline hash manifests are written outside tracked output.

- [ ] **Step 2: Generate the formal review once**

```powershell
python -X utf8 -B src\main\build_zero_boundary_power_review.py
```

Expected: `generated 53 segment figures` and no existing-output error.

- [ ] **Step 3: Independently validate counts, hashes, endpoints, statistics, and grouping**

```powershell
python -X utf8 -B -c "import hashlib,pandas as pd; from pathlib import Path; root=Path('data/processed/operating_dataset_zero_boundary_v2'); out=Path('outputs/v2_zero_boundary_power_review'); source=pd.read_csv(root/'metadata/sample_manifest.csv'); review=pd.read_csv(out/'review_manifest.csv'); assert len(source)==len(review)==53; assert review.groupby('split').size().to_dict()=={'test':5,'train':38,'validation':10}; assert len(list(out.rglob('*.png')))==53; assert all((out/p).is_file() for p in review.image_relative_path); assert all(hashlib.sha256((root/p).read_bytes()).hexdigest()==h for p,h in zip(review.source_relative_path,review.source_sha256)); assert (review.minimum_kw<0).any(); html=(out/'index.html').read_text(encoding='utf-8'); assert html.index('id=\"train\"')<html.index('id=\"validation\"')<html.index('id=\"test\"'); print('53_figures_verified')"
```

Expected: `53_figures_verified`.

- [ ] **Step 4: Confirm protected source trees are unchanged**

```powershell
Get-ChildItem data\processed\operating_dataset_zero_boundary_v2 -Recurse -File | Get-FileHash -Algorithm SHA256 | Export-Csv .codex_tmp\zero_boundary_dataset_after_review.csv -NoTypeInformation -Encoding utf8
Get-ChildItem outputs\v2_segment_power_review -Recurse -File | Get-FileHash -Algorithm SHA256 | Export-Csv .codex_tmp\old_power_review_after.csv -NoTypeInformation -Encoding utf8
python -X utf8 -B -c "import pandas as pd; from pathlib import Path; pairs=[('zero_boundary_dataset_before_review.csv','zero_boundary_dataset_after_review.csv'),('old_power_review_before.csv','old_power_review_after.csv')]; root=Path('.codex_tmp'); [(lambda a,b: (_ for _ in ()).throw(AssertionError(x)) if not a.equals(b) else None)(pd.read_csv(root/x).sort_values('Path').reset_index(drop=True),pd.read_csv(root/y).sort_values('Path').reset_index(drop=True)) for x,y in pairs]; print('protected_sources_unchanged')"
```

Expected: `protected_sources_unchanged`.

- [ ] **Step 5: Visually inspect representative plots**

Open one Train plot with an internal negative interval, one Validation plot, and all five Test plots. Confirm readable labels, visible zero endpoints, visible negative intervals, independent per-segment y-limits, no blank images, and no clipped titles. Open `index.html` and confirm all three split sections render in the frozen order.

- [ ] **Step 6: Run final verification**

```powershell
python -X utf8 -B -m unittest tests.test_v2_zero_boundary_power_review -v
Get-ChildItem tests\test_v2_*.py | ForEach-Object { python -X utf8 -B $_.FullName; if ($LASTEXITCODE -ne 0) { throw "failed: $($_.Name)" } }
python -X utf8 -B -m unittest discover -s tests -p 'test_*.py'
python -X utf8 -B -m compileall -q src tests
git diff --check
git status --short
```

Expected: focused, all v2, and complete suites pass; compilation and diff checks exit zero; only the intended source, test, plan, and generated-review files appear.

- [ ] **Step 7: Commit generated review**

```powershell
git add -- outputs/v2_zero_boundary_power_review
git diff --cached --check
git commit -m "data(v2): add final power review"
```

Do not push unless the user explicitly requests it.
