"""Run the reproducible Train-only v2 DQN state audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Sequence

from ..analysis.train_state_audit import (
    AuditSupervisorySample,
    align_battery_soc_to_timestamps,
    assemble_audit_supervisory_samples,
    build_formal_episode_feature_rows,
    build_train_feature_rows,
    episode_life_upper_bounds,
    load_train_segments,
    write_audit_artifacts,
)
from ..data.formal_training_dataset import FormalEpisode, FormalTrainingDataset
from ..data.segment_power_source import _load_parent


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_ROOT = (
    PROJECT_ROOT / "data" / "processed" / "operating_dataset_zero_boundary_v2"
)
DEFAULT_RAW_ROOT = Path.home() / "OneDrive" / "Desktop" / "氢舟一号"
DEFAULT_AIS_ROOT = (
    PROJECT_ROOT / "data" / "processed" / "operating_dataset_zero_boundary_v2_ais"
)
DEFAULT_MODE_ROOT = (
    PROJECT_ROOT / "data" / "processed" / "operating_dataset_zero_boundary_v2_modes"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "v2_dqn_state_audit"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "docs" / "v2_dqn_state_audit.md"


StateLoader = Callable[[str], Sequence[AuditSupervisorySample]]
SocLoader = Callable[[FormalEpisode], Sequence[float | None]]


def _raw_state_loader(raw_root: Path) -> StateLoader:
    root = Path(raw_root).resolve()

    def load(parent: str) -> tuple[AuditSupervisorySample, ...]:
        loaded = _load_parent(root, parent)
        return assemble_audit_supervisory_samples(loaded.channels)

    return load


def run_train_state_audit(
    *,
    dataset_root: str | Path,
    raw_root: str | Path,
    output_root: str | Path,
    report_path: str | Path,
    state_loader: StateLoader | None = None,
    ais_root: str | Path | None = None,
    mode_root: str | Path | None = None,
    soc_loader: SocLoader | None = None,
) -> dict[str, object]:
    """Build evidence from the authenticated active Train split only."""

    dataset = Path(dataset_root).resolve()
    raw = Path(raw_root).resolve()
    segments = load_train_segments(dataset)
    if (ais_root is None) != (mode_root is None):
        raise ValueError("AIS and mode roots must be supplied together")
    if ais_root is None:
        if soc_loader is not None:
            raise ValueError("soc_loader requires authenticated AIS/mode inputs")
        loader = state_loader if state_loader is not None else _raw_state_loader(raw)
        rows = build_train_feature_rows(dataset, segments, loader)
        input_manifests: dict[str, Path] = {}
        formal_onboard_row_count = len(rows)
        life_bounds: dict[str, object] = {
            "status": "NOT_EVALUATED_LEGACY_FIXTURE"
        }
    else:
        if state_loader is not None:
            raise ValueError("state_loader is incompatible with formal S8 audit mode")
        ais = Path(ais_root).resolve()
        modes = Path(mode_root).resolve()
        formal = FormalTrainingDataset.open(dataset, ais, modes)
        episodes = formal.load_train()
        expected_ids = {segment.sample_id for segment in segments}
        actual_ids = {episode.sample_id for episode in episodes}
        if actual_ids != expected_ids or len(episodes) != len(segments):
            raise ValueError("formal Train episodes differ from authenticated segments")
        channel_cache = {}

        def default_soc_loader(episode: FormalEpisode) -> Sequence[float | None]:
            if episode.parent not in channel_cache:
                channel_cache[episode.parent] = _load_parent(
                    raw, episode.parent
                ).channels
            timestamps = tuple(value.to_pydatetime() for value in episode.timestamp)
            return align_battery_soc_to_timestamps(
                channel_cache[episode.parent], timestamps
            )

        active_soc_loader = soc_loader or default_soc_loader
        formal_rows = []
        for episode in episodes:
            formal_rows.extend(
                build_formal_episode_feature_rows(
                    episode, active_soc_loader(episode)
                )
            )
        rows = tuple(formal_rows)
        formal_onboard_row_count = sum(
            mode == "onboard"
            for episode in episodes
            for mode in episode.operating_mode
        )
        episode_bounds = {
            episode.sample_id: episode_life_upper_bounds(episode)
            for episode in episodes
        }
        worst_fc_id = max(
            episode_bounds,
            key=lambda key: episode_bounds[key]["fc_raw_life_fraction_upper_bound"],
        )
        worst_battery_id = max(
            episode_bounds,
            key=lambda key: episode_bounds[key]["battery_raw_life_fraction_upper_bound"],
        )
        life_bounds = {
            "status": "VERIFIED_BELOW_EOL",
            "method": "per-episode conservative physical upper bound with reset",
            "max_fc_raw_life_fraction_upper_bound": episode_bounds[worst_fc_id][
                "fc_raw_life_fraction_upper_bound"
            ],
            "max_fc_episode_id": worst_fc_id,
            "max_battery_raw_life_fraction_upper_bound": episode_bounds[
                worst_battery_id
            ]["battery_raw_life_fraction_upper_bound"],
            "max_battery_episode_id": worst_battery_id,
        }
        if (
            life_bounds["max_fc_raw_life_fraction_upper_bound"] >= 1.0
            or life_bounds["max_battery_raw_life_fraction_upper_bound"] >= 1.0
        ):
            raise ValueError(
                "cumulative degradation must enter the state because EOL clipping "
                "is reachable within a reset episode"
            )
        if formal.opened_test_payloads != 0:
            raise AssertionError("state audit opened Test payloads")
        input_manifests = {
            "power": dataset / "metadata" / "sample_manifest.csv",
            "ais": ais / "metadata" / "sample_manifest.csv",
            "modes": modes / "metadata" / "sample_manifest.csv",
        }
    return write_audit_artifacts(
        rows=rows,
        segments=segments,
        output_root=output_root,
        report_path=report_path,
        dataset_root=dataset,
        raw_root=raw,
        power_manifest_path=input_manifests.get("power"),
        ais_manifest_path=input_manifests.get("ais"),
        mode_manifest_path=input_manifests.get("modes"),
        formal_onboard_row_count=formal_onboard_row_count,
        hidden_life_state_upper_bounds=life_bounds,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the Train-only v2 DQN state audit evidence bundle."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--ais-root", type=Path, default=DEFAULT_AIS_ROOT)
    parser.add_argument("--mode-root", type=Path, default=DEFAULT_MODE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    manifest = run_train_state_audit(
        dataset_root=arguments.dataset_root,
        raw_root=arguments.raw_root,
        ais_root=arguments.ais_root,
        mode_root=arguments.mode_root,
        output_root=arguments.output_root,
        report_path=arguments.report_path,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
