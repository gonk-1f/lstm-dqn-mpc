"""V6 one-second episodes from the unchanged canonical Train/Validation files."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .independent_dataset import IndependentEpisode


class V6TrainingDataset:
    task_mode = "independent_voyage"

    def __init__(self, root: Path, manifest: pd.DataFrame, metadata: dict) -> None:
        self.root = root
        self._manifest = manifest
        self._metadata = metadata
        self._cache: dict[str, tuple[IndependentEpisode, ...]] = {}

    @classmethod
    def open(cls, root: Path) -> "V6TrainingDataset":
        root = Path(root).resolve()
        metadata = json.loads((root / "metadata/active_split.json").read_text(encoding="utf-8"))
        manifest = pd.read_csv(root / metadata["manifest_relative_path"], encoding="utf-8-sig")
        return cls(root, manifest, metadata)

    @property
    def split_sizes(self) -> tuple[int, int]:
        return tuple(int(self._manifest["split"].eq(split).sum())
                     for split in ("train", "validation"))

    @property
    def split_counts(self) -> dict[str, int]:
        return dict(self._metadata["split_counts"])

    @property
    def opened_test_payloads(self) -> int:
        return 0

    @property
    def task_contract(self) -> dict:
        return {
            "task_mode": self.task_mode,
            "dataset_version": self._metadata["dataset_version"],
            "split_version": self._metadata["split_version"],
            "initial_soc": .6,
            "shore_control": False,
            "terminal_charging": False,
            "model_step_s": 1,
            "execution_grid": "t=k<duration_s",
            "terminal_boundary": "duration_s_node_without_extra_action",
            "load_sampling": "reuse_existing_canonical_1s_left_endpoint",
            "loader_load_policy": "preserve_signed_source_values",
            "historical_checkpoints_valid_for_new_task": False,
        }

    def split_episode_ids(self, split: str) -> tuple[str, ...]:
        if split not in ("train", "validation"):
            raise ValueError("V6 training only loads Train and Validation")
        return tuple(sorted(self._manifest.loc[self._manifest["split"].eq(split), "candidate_id"]))

    def _load_episode(self, row: object) -> IndependentEpisode:
        frame = pd.read_csv(self.root / row.relative_path, encoding="utf-8-sig",
                            float_precision="round_trip")
        stamps = tuple(pd.to_datetime(frame["timestamp"]))
        time = frame["time_s"].to_numpy(dtype=float)
        load = frame["load_total_kw"].to_numpy(dtype=float)
        source_time = frame["source_time_s"].to_numpy(dtype=float)
        source_rows = frame["source_1s_row"].to_numpy(dtype=np.int64)
        return IndependentEpisode(
            sample_id=str(row.candidate_id), split=str(row.split),
            parent_sample_id=str(row.parent_sample_id), timestamp=stamps[:-1],
            time_s=time[:-1], dt_s=np.ones(len(time) - 1, dtype=float),
            load_kw=load[:-1], source_time_s=source_time[:-1], source_1s_row=source_rows[:-1],
            duration_s=float(row.duration_s), terminal_boundary_time_s=float(time[-1]),
            terminal_boundary_load_kw=float(load[-1]), terminal_boundary_timestamp=stamps[-1],
            terminal_boundary_source_time_s=float(source_time[-1]),
            terminal_boundary_source_1s_row=int(source_rows[-1]),
        )

    def load_split(self, split: str) -> tuple[IndependentEpisode, ...]:
        self.split_episode_ids(split)
        if split not in self._cache:
            rows = self._manifest[self._manifest["split"].eq(split)].sort_values("candidate_id")
            self._cache[split] = tuple(self._load_episode(row) for row in rows.itertuples(index=False))
        return self._cache[split]

    def load_train(self) -> tuple[IndependentEpisode, ...]:
        return self.load_split("train")

    def load_validation(self) -> tuple[IndependentEpisode, ...]:
        return self.load_split("validation")
