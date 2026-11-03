"""Explicit Test4 data adapter; training and validation environments stay unchanged."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .validate import DEFAULT_DATA_ROOT  # Reuse the existing src-path bootstrap.
from .env import AggregateDDPGEnv
from v4.independent_dataset import IndependentTrainingDataset, _check_sha256, _contained, _sha256
from v4.v6_dataset import V6TrainingDataset


class AggregateDDPGTestEnv(AggregateDDPGEnv):
    """Reuse all parent physics/actions, with only the frozen Test4 voyages selected."""

    def __init__(self, data_root: Path, test_source_root: Path | None = None):
        data_root = Path(data_root).resolve()
        current = V6TrainingDataset.open(data_root)
        metadata = current._metadata
        source_root = Path(test_source_root or metadata["source_dataset_path"]).resolve()
        # This authenticates frozen split metadata but does not load any payload.
        source = IndependentTrainingDataset.open(source_root)
        source_manifest = _contained(source_root, metadata["source_manifest_relative_path"])
        if (
            metadata["split_counts"] != {"train": 35, "validation": 8, "test": 4}
            or source.split_counts != metadata["split_counts"]
            or source._policy["split_version"] != metadata["split_version"]
            or source._identity_paths[1] != metadata["source_manifest_relative_path"]
        ):
            raise ValueError("Test4 source must match the existing frozen V6 split")
        selected = current._manifest[current._manifest["split"].eq("test")].sort_values("candidate_id")
        rows = source._manifest[source._manifest["split"].eq("test")].sort_values("candidate_id")
        if len(rows) != 4 or list(selected["candidate_id"]) != list(rows["candidate_id"]):
            raise ValueError("V6 and source manifests must identify the same four Test voyages")

        # The V6 loader preserves every original second; the independent training
        # loader's _load_episode would instead select 30-second intervals.
        reader = V6TrainingDataset(source_root, source._manifest, metadata)
        episodes, payloads = [], []
        for target, row in zip(selected.itertuples(index=False), rows.itertuples(index=False)):
            if (
                target.parent_sample_id != row.parent_sample_id
                or int(target.duration_s) != int(row.duration_s)
                or target.source_payload_relative_path != row.relative_path
            ):
                raise ValueError(f"{row.candidate_id}: Test voyage identity differs")
            path = _contained(source_root, row.relative_path)
            _check_sha256(path, row.sha256)
            episode = reader._load_episode(row)
            duration = int(row.duration_s)
            if (
                not np.array_equal(episode.time_s, np.arange(duration))
                or not np.array_equal(episode.dt_s, np.ones(duration))
                or episode.terminal_boundary_time_s != duration
            ):
                raise ValueError(f"{row.candidate_id}: expected complete canonical one-second voyage")
            episodes.append(episode)
            payloads.append({
                "sample_id": episode.sample_id,
                "path": str(path),
                "relative_path": row.relative_path,
                "sha256": row.sha256,
                "duration_s": duration,
                "steps": episode.step_count,
                "terminal_boundary_time_s": episode.terminal_boundary_time_s,
            })

        # Preserve the established constructor, physical plant and spaces exactly.
        # It reads Validation8 once for initialization, but executes no voyage.
        super().__init__(data_root, split="validation")
        self.episodes = tuple(episodes)
        self.split = "test"
        self._validation_cursor = 0
        self.opened_test_payloads = len(episodes)
        self.test_data_identity = {
            "source_root": str(source_root),
            "source_manifest_path": str(source_manifest),
            "source_manifest_sha256": _sha256(source_manifest),
            "source_metadata_sha256": source.identity_hashes(),
            "v6_data_root": str(data_root),
            "v6_active_split_sha256": _sha256(data_root / "metadata/active_split.json"),
            "v6_manifest_sha256": _sha256(_contained(data_root, metadata["manifest_relative_path"])),
            "split_version": metadata["split_version"],
            "model_step_s": 1,
            "test_order": [episode.sample_id for episode in episodes],
            "payloads": payloads,
        }
