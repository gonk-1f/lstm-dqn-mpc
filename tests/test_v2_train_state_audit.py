from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_dataset_fixture(root: Path, *, train_relative_path: str) -> None:
    metadata = root / "metadata"
    train = root / "train"
    metadata.mkdir(parents=True)
    train.mkdir()
    train_path = train / "train_001.csv"
    pd.DataFrame(
        {
            "timestamp": ["2024-01-01T00:00:00+08:00"],
            "time_s": [0.0],
            "load_total_kw": [10.0],
        }
    ).to_csv(train_path, index=False)
    pd.DataFrame(
        [
            {
                "parent": "train_parent",
                "sample_id": "train_001",
                "relative_path": train_relative_path,
                "split": "train",
                "point_count_1s": 1,
                "start_timestamp": "2024-01-01T00:00:00+08:00",
                "end_timestamp": "2024-01-01T00:00:00+08:00",
                "duration_s": 0.0,
                "sha256": _sha256(train_path),
            },
            {
                "parent": "validation_parent",
                "sample_id": "validation_001",
                "relative_path": "validation/validation_001.csv",
                "split": "validation",
                "point_count_1s": 1,
                "start_timestamp": "2024-01-02T00:00:00+08:00",
                "end_timestamp": "2024-01-02T00:00:00+08:00",
                "duration_s": 0.0,
                "sha256": "held-out-file-must-not-be-opened",
            },
            {
                "parent": "test_parent",
                "sample_id": "test_001",
                "relative_path": "test/test_001.csv",
                "split": "test",
                "point_count_1s": 1,
                "start_timestamp": "2024-01-03T00:00:00+08:00",
                "end_timestamp": "2024-01-03T00:00:00+08:00",
                "duration_s": 0.0,
                "sha256": "held-out-file-must-not-be-opened",
            },
        ]
    ).to_csv(metadata / "sample_manifest.csv", index=False)
    (metadata / "qa_summary.json").write_text(
        json.dumps({"dataset_version": "operating_dataset_zero_boundary_v2"}),
        encoding="utf-8",
    )


class TrainStateAuditInputTests(unittest.TestCase):
    def test_load_train_segments_filters_held_out_without_opening_them(self) -> None:
        from v2.analysis.train_state_audit import load_train_segments

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            _write_dataset_fixture(root, train_relative_path="train/train_001.csv")

            segments = load_train_segments(root)

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].parent, "train_parent")
        self.assertEqual(segments[0].sample_id, "train_001")
        self.assertEqual(segments[0].relative_path, "train/train_001.csv")
        self.assertEqual(
            segments[0].dataset_version,
            "operating_dataset_zero_boundary_v2",
        )

    def test_load_train_segments_rejects_non_train_or_escaping_path(self) -> None:
        from v2.analysis.train_state_audit import load_train_segments

        for relative_path in ("validation/train_001.csv", "../train/train_001.csv"):
            with self.subTest(relative_path=relative_path):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary) / "dataset"
                    _write_dataset_fixture(root, train_relative_path=relative_path)
                    with self.assertRaisesRegex(ValueError, "Train path"):
                        load_train_segments(root)

    def test_load_train_segments_rejects_mutated_train_file(self) -> None:
        from v2.analysis.train_state_audit import load_train_segments

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            _write_dataset_fixture(root, train_relative_path="train/train_001.csv")
            (root / "train" / "train_001.csv").write_text(
                "mutated\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_train_segments(root)


class TrainStateAuditCausalFeatureTests(unittest.TestCase):
    @staticmethod
    def _fixture(include_future: bool = False):
        from v2.analysis.train_state_audit import TrainSegment
        from v2.data.supervisory_rules import OperatingMode
        from v2.data.train_supervisory_audit import ParentSupervisoryState

        start = datetime(2024, 1, 1, tzinfo=ZoneInfo("Asia/Shanghai"))
        loads = [100.0, 120.0, 140.0, 160.0, 180.0, 200.0]
        fc = [80.0, 90.0, 100.0, 110.0, 120.0, 130.0]
        soc = [0.600, 0.599, 0.598, 0.597, 0.596, 0.595]
        if include_future:
            loads.append(999.0)
            fc.append(500.0)
            soc.append(0.594)
        states = []
        for index, (load_kw, fc_kw, soc_value) in enumerate(zip(loads, fc, soc)):
            timestamp = start + timedelta(seconds=30 * index)
            previous_fc = 70.0 if index == 0 else fc[index - 1]
            states.append(
                ParentSupervisoryState(
                    parent_id="train_parent",
                    timestamp=timestamp,
                    mode=OperatingMode.SAILING_ISLAND,
                    p_fc_total_kw=fc_kw,
                    p_batt_total_kw=load_kw - fc_kw,
                    p_load_kw=load_kw,
                    soc_system=soc_value,
                    previous_p_fc_total_kw=previous_fc,
                    channels_complete=True,
                    conflicting_duplicate=False,
                    long_gap_contaminated=False,
                    source_provenance=(),
                )
            )
        load_frame = pd.DataFrame(
            {
                "timestamp": [state.timestamp for state in states],
                "time_s": [float(index * 30) for index in range(len(states))],
                "load_total_kw": loads,
            }
        )
        segment = TrainSegment(
            parent="train_parent",
            sample_id="train_001",
            relative_path="train/train_001.csv",
            start_timestamp=start,
            end_timestamp=start + timedelta(seconds=150),
            sha256="synthetic",
            dataset_version="operating_dataset_zero_boundary_v2",
        )
        return segment, tuple(states), load_frame

    def test_builds_six_sample_causal_window_with_frozen_lpf(self) -> None:
        from v2.analysis.train_state_audit import build_causal_feature_rows

        segment, states, load_frame = self._fixture()

        rows = build_causal_feature_rows(segment, states, load_frame)

        self.assertEqual(len(rows), 1)
        row = rows[0]
        alpha = math.exp(-30.0 / 90.0)
        expected_base = 100.0
        for load in (120.0, 140.0, 160.0, 180.0, 200.0):
            expected_base = alpha * expected_base + (1.0 - alpha) * load
        self.assertAlmostEqual(row.base_load_kw, expected_base)
        self.assertAlmostEqual(row.delta_load_kw, 200.0 - expected_base)
        self.assertAlmostEqual(row.delta_fc_kw, 10.0)
        self.assertAlmostEqual(row.recent_delta_soc, -0.005)
        self.assertAlmostEqual(row.recent_load_mean_kw, 150.0)
        self.assertAlmostEqual(row.recent_load_trend_kw_per_s, 20.0 / 30.0)
        self.assertAlmostEqual(row.measured_power_balance_residual_kw, 0.0)
        self.assertEqual(row.history_sample_count, 6)

    def test_future_sample_cannot_change_current_feature_row(self) -> None:
        from v2.analysis.train_state_audit import build_causal_feature_rows

        segment, states, load_frame = self._fixture()
        expected = build_causal_feature_rows(segment, states, load_frame)
        future_segment, future_states, future_load_frame = self._fixture(
            include_future=True
        )

        actual = build_causal_feature_rows(
            future_segment,
            future_states,
            future_load_frame,
        )

        self.assertEqual(actual, expected)

    def test_history_before_segment_boundary_is_not_reused(self) -> None:
        from v2.analysis.train_state_audit import build_causal_feature_rows

        segment, states, load_frame = self._fixture()
        later_segment = type(segment)(
            parent=segment.parent,
            sample_id="train_002",
            relative_path="train/train_002.csv",
            start_timestamp=segment.start_timestamp + timedelta(seconds=30),
            end_timestamp=segment.end_timestamp,
            sha256="synthetic",
            dataset_version=segment.dataset_version,
        )

        rows = build_causal_feature_rows(later_segment, states, load_frame)

        self.assertEqual(rows, ())

    def test_train_row_builder_loads_only_whitelisted_parent_and_segment(self) -> None:
        from v2.analysis.train_state_audit import build_train_feature_rows

        segment, states, load_frame = self._fixture()
        calls: list[str] = []

        def state_loader(parent: str):
            calls.append(parent)
            return states

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / segment.relative_path
            path.parent.mkdir(parents=True)
            load_frame.to_csv(path, index=False)

            rows = build_train_feature_rows(root, (segment,), state_loader)

        self.assertEqual(calls, ["train_parent"])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].sample_id, "train_001")


class TrainStateAuditNumericalTests(unittest.TestCase):
    def _rows(self):
        from v2.analysis.train_state_audit import build_causal_feature_rows

        segment, states, load_frame = TrainStateAuditCausalFeatureTests._fixture()
        base = build_causal_feature_rows(segment, states, load_frame)[0]
        return tuple(
            replace(
                base,
                timestamp=base.timestamp + timedelta(seconds=30 * index),
                soc=base.soc + 0.01 * index,
                fc_power_kw=base.fc_power_kw + 20.0 * index,
                previous_fc_power_kw=base.previous_fc_power_kw + 10.0 * index,
                battery_power_kw=base.battery_power_kw - 5.0 * index,
                load_power_kw=base.load_power_kw + 15.0 * index,
                recent_load_mean_kw=base.recent_load_mean_kw + 12.0 * index,
                recent_load_population_std_kw=(
                    base.recent_load_population_std_kw + 2.0 * index
                ),
                recent_load_trend_kw_per_s=(
                    base.recent_load_trend_kw_per_s + 0.1 * index
                ),
                base_load_kw=base.base_load_kw + 8.0 * index,
                recent_delta_soc=base.recent_delta_soc - 0.001 * index,
                delta_load_kw=base.delta_load_kw + 7.0 * index,
                delta_fc_kw=base.delta_fc_kw + 10.0 * index,
                measured_power_balance_residual_kw=0.1 * index,
            )
            for index in range(5)
        )

    def test_normalization_uses_frozen_physical_scales(self) -> None:
        from v2.analysis.train_state_audit import (
            AUDIT_BATTERY_POWER_SCALE_KW,
            AUDIT_HISTORY_SECONDS,
            AUDIT_POWER_SCALE_KW,
            feature_frame,
            normalize_feature_frame,
        )

        physical = feature_frame(self._rows())
        normalized = normalize_feature_frame(physical)

        self.assertAlmostEqual(
            normalized.iloc[0]["fuel_cell_power_fraction"],
            physical.iloc[0]["fc_power_kw"] / AUDIT_POWER_SCALE_KW,
        )
        self.assertAlmostEqual(
            normalized.iloc[0]["battery_power_fraction"],
            physical.iloc[0]["battery_power_kw"] / AUDIT_BATTERY_POWER_SCALE_KW,
        )
        self.assertAlmostEqual(
            normalized.iloc[0]["recent_load_window_trend_fraction"],
            physical.iloc[0]["recent_load_trend_kw_per_s"]
            * AUDIT_HISTORY_SECONDS
            / AUDIT_POWER_SCALE_KW,
        )
        self.assertAlmostEqual(
            normalized.iloc[0]["load_residual_fraction"],
            physical.iloc[0]["delta_load_kw"] / AUDIT_POWER_SCALE_KW,
        )

    def test_descriptive_statistics_use_population_std_and_fixed_columns(self) -> None:
        from v2.analysis.train_state_audit import descriptive_statistics

        frame = pd.DataFrame({"feature": [1.0, 2.0, 3.0, 4.0]})
        statistics = descriptive_statistics(frame)

        self.assertEqual(
            statistics.columns.tolist(),
            [
                "feature",
                "count",
                "missing_count",
                "min",
                "max",
                "mean",
                "std",
                "p01",
                "p05",
                "p50",
                "p95",
                "p99",
                "near_zero_variance",
            ],
        )
        self.assertAlmostEqual(statistics.iloc[0]["std"], math.sqrt(1.25))
        self.assertFalse(bool(statistics.iloc[0]["near_zero_variance"]))

    def test_correlations_are_symmetric_with_unit_diagonal(self) -> None:
        from v2.analysis.train_state_audit import correlation_matrices

        frame = pd.DataFrame(
            {
                "a": [1.0, 2.0, 4.0, 8.0],
                "b": [2.0, 1.0, 8.0, 3.0],
                "c": [9.0, 3.0, 5.0, 7.0],
            }
        )
        pearson, spearman = correlation_matrices(frame)

        pd.testing.assert_frame_equal(pearson, pearson.T)
        pd.testing.assert_frame_equal(spearman, spearman.T)
        self.assertTrue(np.allclose(np.diag(pearson), 1.0))
        self.assertTrue(np.allclose(np.diag(spearman), 1.0))

    def test_redundancy_and_regime_outputs_are_explicit(self) -> None:
        from v2.analysis.train_state_audit import (
            feature_frame,
            redundancy_summary,
            regime_summary,
        )

        frame = feature_frame(self._rows())
        redundancy = redundancy_summary(frame)
        regimes = regime_summary(frame)

        balance = redundancy.loc[
            redundancy["relationship"].eq("environment_power_balance")
        ].iloc[0]
        self.assertEqual(balance["model_status"], "EXACT_IDENTITY")
        self.assertGreater(float(balance["measured_max_abs_residual_kw"]), 0.0)
        self.assertIn("threshold_status", regimes.columns)
        self.assertTrue(
            set(regimes["regime"]).issuperset(
                {"steady_load", "load_rise", "load_fall", "high_volatility"}
            )
        )

    def test_state_comparison_has_exact_requested_ablations(self) -> None:
        from v2.analysis.train_state_audit import state_comparison

        comparison = state_comparison().set_index("state_id")

        self.assertEqual(comparison.loc["S10", "dimension"], 10)
        self.assertEqual(comparison.loc["S7", "dimension"], 7)
        self.assertEqual(comparison.loc["S6-A", "dimension"], 6)
        self.assertNotIn(
            "fuel_cell_delta_fraction",
            comparison.loc["S6-A", "features"].split("|"),
        )
        self.assertNotIn(
            "recent_load_population_std_fraction",
            comparison.loc["S6-B", "features"].split("|"),
        )
        self.assertEqual(comparison.loc["MINIMUM", "dimension"], 5)


class TrainStateAuditArtifactTests(unittest.TestCase):
    def test_markov_inventory_covers_required_memory(self) -> None:
        from v2.analysis.train_state_audit import markov_inventory

        inventory = markov_inventory().set_index("memory_id")

        required = {
            "causal_lpf_state",
            "previous_executed_fc_power",
            "cumulative_fc_voltage_loss",
            "cumulative_battery_weighted_ah",
            "previous_dqn_action",
            "mpc_warm_start",
            "terminal_recharge_initial_soc",
            "macro_step_position",
        }
        self.assertTrue(required.issubset(inventory.index))
        self.assertFalse(inventory["classification"].eq("").any())
        self.assertFalse(inventory["state_treatment"].eq("").any())
        self.assertFalse(inventory["code_evidence"].eq("").any())

    def test_artifact_writer_emits_reproducible_audit_bundle(self) -> None:
        from v2.analysis.train_state_audit import (
            build_causal_feature_rows,
            write_audit_artifacts,
        )

        segment, states, load_frame = TrainStateAuditCausalFeatureTests._fixture()
        base = build_causal_feature_rows(segment, states, load_frame)[0]
        rows = tuple(
            replace(
                base,
                timestamp=base.timestamp + timedelta(seconds=30 * index),
                soc=base.soc + 0.01 * index,
                fc_power_kw=base.fc_power_kw + 20.0 * index,
                previous_fc_power_kw=base.previous_fc_power_kw + 10.0 * index,
                recent_load_population_std_kw=(
                    base.recent_load_population_std_kw + index
                ),
                recent_load_trend_kw_per_s=(
                    base.recent_load_trend_kw_per_s + 0.1 * index
                ),
                delta_fc_kw=base.delta_fc_kw + 10.0 * index,
            )
            for index in range(5)
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "audit"
            report = root / "report.md"

            manifest = write_audit_artifacts(
                rows=rows,
                segments=(segment,),
                output_root=output,
                report_path=report,
                dataset_root=root / "dataset",
                raw_root=root / "raw",
            )

            expected = {
                "audit_manifest.json",
                "feature_rows.csv",
                "feature_statistics.csv",
                "pearson_correlation.csv",
                "spearman_correlation.csv",
                "power_balance_residuals.csv",
                "redundancy_summary.csv",
                "regime_summary.csv",
                "state_comparison.csv",
                "markov_inventory.csv",
                "candidate_correlation_heatmap.png",
                "feature_distributions.png",
                "regime_discrimination.png",
            }
            self.assertEqual(
                {path.name for path in output.iterdir()},
                expected,
            )
            self.assertEqual(manifest["dataset_version"], segment.dataset_version)
            self.assertEqual(manifest["train_segment_count"], 1)
            self.assertEqual(manifest["eligible_row_count"], len(rows))
            text = report.read_text(encoding="utf-8")
            self.assertIn("Current 10-dimensional candidate", text)
            self.assertIn("S7 conclusion", text)
            self.assertIn("KEEP", text)

    def test_runner_executes_with_injected_train_parent_loader(self) -> None:
        from v2.main.run_train_state_audit import run_train_state_audit

        segment, states, load_frame = TrainStateAuditCausalFeatureTests._fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            train_path = dataset / segment.relative_path
            train_path.parent.mkdir(parents=True)
            load_frame.to_csv(train_path, index=False)
            metadata = dataset / "metadata"
            metadata.mkdir()
            pd.DataFrame(
                [
                    {
                        "parent": segment.parent,
                        "sample_id": segment.sample_id,
                        "relative_path": segment.relative_path,
                        "split": "train",
                        "start_timestamp": segment.start_timestamp.isoformat(),
                        "end_timestamp": segment.end_timestamp.isoformat(),
                        "sha256": _sha256(train_path),
                    },
                    {
                        "parent": "held_out",
                        "sample_id": "test_001",
                        "relative_path": "test/test_001.csv",
                        "split": "test",
                        "start_timestamp": segment.start_timestamp.isoformat(),
                        "end_timestamp": segment.end_timestamp.isoformat(),
                        "sha256": "must-not-open",
                    },
                ]
            ).to_csv(metadata / "sample_manifest.csv", index=False)
            (metadata / "qa_summary.json").write_text(
                json.dumps(
                    {"dataset_version": "operating_dataset_zero_boundary_v2"}
                ),
                encoding="utf-8",
            )
            output = root / "output"
            report = root / "report.md"
            calls: list[str] = []

            def loader(parent: str):
                calls.append(parent)
                return states

            manifest = run_train_state_audit(
                dataset_root=dataset,
                raw_root=root / "raw",
                output_root=output,
                report_path=report,
                state_loader=loader,
            )

            self.assertEqual(calls, [segment.parent])
            self.assertEqual(manifest["held_out_segment_files_opened"], 0)
            self.assertTrue(report.is_file())


if __name__ == "__main__":
    unittest.main()
