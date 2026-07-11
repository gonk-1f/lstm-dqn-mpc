from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "src" / "main"
if str(MAIN) not in sys.path:
    sys.path.insert(0, str(MAIN))

from audit_millisecond_10ms_dataset import audit_dataset, audit_member_rows  # noqa: E402
from build_millisecond_10ms_dataset import (  # noqa: E402
    FULL_OVERVIEW_NAME,
    read_condition_sheets,
)
from tests.test_millisecond_10ms_dataset import (  # noqa: E402
    _condition_rows,
    _inline,
    _write_minimal_multisheet_xlsx,
)


class TestIndependentMemberAudit(unittest.TestCase):
    def setUp(self) -> None:
        source = pd.DataFrame(
            {
                "time_s": [index / 1000 for index in range(11)],
                "load_kw": [float(index) for index in range(11)],
                "fuel_cell_kw": [1.0] * 11,
                "battery_kw": [2.0] * 11,
                "bus_voltage_v": [600.0] * 11,
            }
        )
        self.sources = {("book.xlsx", "sheet"): source}

    def test_exact_tenth_source_row_passes(self) -> None:
        combined = pd.DataFrame(
            {
                "time_ms": [10],
                "time_s": [0.01],
                "load_kw": [10.0],
                "fuel_cell_kw": [1.0],
                "battery_kw": [2.0],
                "bus_voltage_v": [600.0],
                "source_members": ["book.xlsx|sheet|10"],
            }
        )
        result = audit_member_rows(combined, self.sources)
        self.assertEqual(result["checked_members"], 1)
        self.assertEqual(result["mismatch_count"], 0)

    def test_non_tenth_source_row_is_rejected(self) -> None:
        combined = pd.DataFrame(
            {
                "time_ms": [9],
                "time_s": [0.009],
                "load_kw": [9.0],
                "fuel_cell_kw": [1.0],
                "battery_kw": [2.0],
                "bus_voltage_v": [600.0],
                "source_members": ["book.xlsx|sheet|9"],
            }
        )
        with self.assertRaisesRegex(ValueError, "not divisible by 10"):
            audit_member_rows(combined, self.sources)

    def test_tiny_change_to_any_retained_numeric_value_is_rejected_exactly(self) -> None:
        retained_values = {
            "time_s": 0.01,
            "load_kw": 10.0,
            "fuel_cell_kw": 1.0,
            "battery_kw": 2.0,
            "bus_voltage_v": 600.0,
        }
        for column in retained_values:
            with self.subTest(column=column):
                changed_values = retained_values.copy()
                changed_values[column] += 5e-10
                combined = pd.DataFrame(
                    {
                        "time_ms": [10],
                        **{name: [value] for name, value in changed_values.items()},
                        "source_members": ["book.xlsx|sheet|10"],
                    }
                )
                with self.assertRaisesRegex(ValueError, f"row 0 {column}"):
                    audit_member_rows(combined, self.sources)


class TestDatasetAuditIntegration(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.dataset_root = self.root / "processed"
        self.dataset_root.mkdir()
        self.source_root = self.root / "supplied"
        self.copy_root = self.root / "raw"
        self.segment_root = self.dataset_root / "segments"
        self.source_root.mkdir()
        self.copy_root.mkdir()
        self.segment_root.mkdir()

        self.source_frames: dict[str, pd.DataFrame] = {}
        source_records: list[dict[str, object]] = []
        for workbook_index, workbook_name in enumerate(("book_a.xlsx", "book_b.xlsx")):
            original = self.source_root / workbook_name
            copied = self.copy_root / workbook_name
            _write_minimal_multisheet_xlsx(
                original,
                [
                    (
                        FULL_OVERVIEW_NAME,
                        "worksheets/overview.xml",
                        [[_inline("excluded")]],
                    ),
                    (
                        "condition",
                        "worksheets/condition.xml",
                        _condition_rows(0.0, 401, 100.0 + workbook_index * 1000.0),
                    ),
                ],
            )
            copied.write_bytes(original.read_bytes())
            source_hash = self._sha256(original)
            source_records.append(
                {
                    "source_path": str(original.resolve()),
                    "copied_path": str(copied.resolve()),
                    "sha256": source_hash,
                    "bytes": original.stat().st_size,
                }
            )
            self.source_frames[workbook_name] = read_condition_sheets(
                copied, overview_names={FULL_OVERVIEW_NAME}
            )["condition"]

        self.source_manifest_path = self.root / "source_manifest.json"
        self._write_json(
            self.source_manifest_path,
            {
                "dataset_version": "test_millisecond_10ms",
                "policy": "project copies are byte-identical to supplied source workbooks",
                "sources": source_records,
            },
        )

        sequence_specs = (
            ("train_book_a", "train", "book_a.xlsx", 36),
            ("train_book_b", "train", "book_b.xlsx", 37),
            ("validation_book_a", "validation", "book_a.xlsx", 38),
            ("validation_book_b", "validation", "book_b.xlsx", 39),
            ("test_book_a", "test", "book_a.xlsx", 40),
            ("test_book_b", "test", "book_b.xlsx", 41),
        )
        combined_frames: list[pd.DataFrame] = []
        atomic_sequences: list[dict[str, object]] = []
        self.expected_sequence_windows: dict[str, int] = {}
        assignments: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
        split_windows: dict[str, int] = {"train": 0, "validation": 0, "test": 0}
        for sequence_index, (sequence_id, split_name, workbook_name, rows) in enumerate(
            sequence_specs, start=1
        ):
            source = self.source_frames[workbook_name]
            source_indices = np.arange(rows, dtype=np.int64) * 10
            selected = source.iloc[source_indices]
            output = pd.DataFrame(
                {
                    "split": split_name,
                    "sequence_id": sequence_id,
                    "time_s": selected["time_s"].to_numpy(copy=True),
                    "time_ms": np.rint(
                        selected["time_s"].to_numpy(dtype=np.float64) * 1000.0
                    ).astype(np.int64),
                    "load_kw": selected["load_kw"].to_numpy(copy=True),
                    "fuel_cell_kw": selected["fuel_cell_kw"].to_numpy(copy=True),
                    "battery_kw": selected["battery_kw"].to_numpy(copy=True),
                    "bus_voltage_v": selected["bus_voltage_v"].to_numpy(copy=True),
                    "source_workbook": workbook_name,
                    "source_members": [
                        f"{workbook_name}|condition|{source_index}"
                        for source_index in source_indices
                    ],
                }
            )
            segment_path = self.segment_root / f"sequence_{sequence_index:03d}.csv"
            output.to_csv(
                segment_path,
                index=False,
                encoding="utf-8-sig",
                float_format="%.17g",
            )
            windows = max(rows - 30 - 6 + 1, 0)
            self.expected_sequence_windows[sequence_id] = windows
            assignments[split_name].append(sequence_id)
            split_windows[split_name] += windows
            atomic_sequences.append(
                {
                    "sequence_id": sequence_id,
                    "split": split_name,
                    "source_workbook": workbook_name,
                    "source_sheets": ["condition"],
                    "rows": rows,
                    "windows_30_to_6": windows,
                    "csv": str(segment_path.resolve()),
                    "sha256": self._sha256(segment_path),
                }
            )
            combined_frames.append(output)

        combined = pd.concat(combined_frames, ignore_index=True)
        combined.to_csv(
            self.dataset_root / "millisecond_load_10ms.csv",
            index=False,
            encoding="utf-8-sig",
            float_format="%.17g",
        )
        self.split_path = self.root / "millisecond_10ms_split_721.json"
        self._write_json(
            self.split_path,
            {"assignments": assignments, "split_windows": split_windows},
        )
        self.dataset_manifest_path = self.dataset_root / "dataset_manifest.json"
        self._write_json(
            self.dataset_manifest_path,
            {
                "dataset_version": "test_millisecond_10ms",
                "source_manifest": str(self.source_manifest_path.resolve()),
                "atomic_sequences": atomic_sequences,
            },
        )
        self.audit_path = self.dataset_root / "independent_audit.json"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def _write_json(path: Path, payload: object) -> None:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def test_full_audit_uses_csv_manifest_and_attaches_audit(self) -> None:
        result = audit_dataset(
            dataset_root=self.dataset_root,
            split_path=self.split_path,
            audit_path=self.audit_path,
            attach_to_manifest=True,
        )

        self.assertTrue(result["passed"])
        self.assertEqual(result["sequence_windows"], self.expected_sequence_windows)
        self.assertEqual(result["split_windows"], {"train": 3, "validation": 7, "test": 11})
        attached_manifest = json.loads(self.dataset_manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(attached_manifest["independent_audit"], str(self.audit_path.resolve()))
        self.assertEqual(
            attached_manifest["independent_audit_sha256"], self._sha256(self.audit_path)
        )

    def test_per_sequence_window_mismatch_is_rejected_when_split_totals_match(self) -> None:
        manifest = json.loads(self.dataset_manifest_path.read_text(encoding="utf-8"))
        manifest["atomic_sequences"][0]["windows_30_to_6"] += 1
        self._write_json(self.dataset_manifest_path, manifest)

        with self.assertRaisesRegex(
            ValueError, "Window count differs for sequence train_book_a"
        ):
            audit_dataset(
                dataset_root=self.dataset_root,
                split_path=self.split_path,
                audit_path=self.audit_path,
                attach_to_manifest=False,
            )

    def test_assignment_keys_must_be_exact_required_splits(self) -> None:
        baseline = json.loads(self.split_path.read_text(encoding="utf-8"))
        scenarios: list[tuple[str, dict[str, object], str]] = []
        for missing_split in ("train", "validation", "test"):
            missing_manifest = json.loads(json.dumps(baseline))
            del missing_manifest["assignments"][missing_split]
            scenarios.append(
                (
                    f"missing_{missing_split}",
                    missing_manifest,
                    f"missing=['{missing_split}']",
                )
            )
        extra_manifest = json.loads(json.dumps(baseline))
        extra_manifest["assignments"]["holdout"] = []
        scenarios.append(("extra_holdout", extra_manifest, "extra=['holdout']"))

        for scenario_name, split_manifest, expected_message in scenarios:
            with self.subTest(scenario=scenario_name):
                self._write_json(self.split_path, split_manifest)
                try:
                    audit_dataset(
                        dataset_root=self.dataset_root,
                        split_path=self.split_path,
                        audit_path=self.audit_path,
                        attach_to_manifest=False,
                    )
                except ValueError as exc:
                    self.assertIn(expected_message, str(exc))
                except Exception as exc:  # pragma: no cover - asserted as a failure
                    self.fail(f"Expected ValueError, got {type(exc).__name__}: {exc}")
                else:
                    self.fail("Expected split assignment key validation to reject the manifest")


if __name__ == "__main__":
    unittest.main()
