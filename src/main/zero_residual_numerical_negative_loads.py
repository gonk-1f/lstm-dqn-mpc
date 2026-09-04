"""Zero only the already-audited residual numerical negatives in final 1 s data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = REPO_ROOT / "data" / "processed" / "operating_segments_1s_rebuilt"


def zero_residual_numerical_negatives(root: Path = DATASET_ROOT) -> dict[str, int | float]:
    """Set values in (-1, 0) to zero after a full-dataset precondition scan."""
    root = Path(root)
    paths = sorted((root / "operating_segments_1s").glob("*.csv"))
    if not paths:
        raise ValueError("no final 1 s segment CSV files found")

    frames: list[tuple[Path, pd.DataFrame, pd.Series]] = []
    before_min = np.inf
    before_zero_count = 0
    numerical_zero_clipped_points = 0
    for path in paths:
        frame = pd.read_csv(path)
        if "load_total_kw" not in frame:
            raise ValueError(f"missing load_total_kw: {path}")
        values = pd.to_numeric(frame["load_total_kw"], errors="raise")
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite final 1 s load: {path}")
        if values.le(-1.0).any():
            raise ValueError(f"negative load outside the approved (-1, 0) numerical tolerance: {path}")
        frames.append((path, frame, values))
        before_min = min(before_min, float(values.min()))
        before_zero_count += int(values.eq(0.0).sum())
        numerical_zero_clipped_points += int(values.lt(0.0).sum())

    for path, frame, values in frames:
        if values.lt(0.0).any():
            frame["load_total_kw"] = values.mask(values.lt(0.0), 0.0)
            frame.to_csv(path, index=False, encoding="utf-8-sig")

    qa_path = root / "qa_summary.json"
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    pchip = qa.setdefault("pchip_1s", {})
    pchip["min_kw"] = 0.0 if numerical_zero_clipped_points else before_min
    pchip["negative_count"] = 0
    pchip["zero_count"] = before_zero_count + numerical_zero_clipped_points
    pchip["under_minus_1_kw_count"] = 0
    qa["numerical_zero_clipping"] = {
        "numerical_zero_clipped_points": numerical_zero_clipped_points,
        "before_min_load_kw": before_min,
        "after_min_load_kw": 0.0 if numerical_zero_clipped_points else before_min,
        "before_zero_count": before_zero_count,
        "after_zero_count": before_zero_count + numerical_zero_clipped_points,
        "approved_range_kw": "-1 < load_total_kw < 0",
    }
    negative = qa.setdefault("negative_load", {})
    negative["residual_tolerance_point_count_before_clipping"] = numerical_zero_clipped_points
    negative["point_count"] = 0
    negative["under_minus_1_point_count"] = 0
    negative["remaining_substantive_interval_count"] = 0
    qa_path.write_text(json.dumps(qa, ensure_ascii=False, indent=2, default=float) + "\n", encoding="utf-8")
    return {
        "numerical_zero_clipped_points": numerical_zero_clipped_points,
        "before_min_load_kw": before_min,
        "after_min_load_kw": 0.0 if numerical_zero_clipped_points else before_min,
        "negative_point_count": 0,
    }


if __name__ == "__main__":
    print(json.dumps(zero_residual_numerical_negatives(), ensure_ascii=False, indent=2))
