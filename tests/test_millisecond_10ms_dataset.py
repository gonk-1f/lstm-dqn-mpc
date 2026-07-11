from __future__ import annotations

import hashlib
import json
import posixpath
import sys
import tempfile
import unittest
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "src" / "main"
if str(MAIN) not in sys.path:
    sys.path.insert(0, str(MAIN))

from build_millisecond_10ms_dataset import (  # noqa: E402
    DEFAULT_SOURCE_PATHS,
    FULL_OVERVIEW_NAME,
    HEADER_ALIASES,
    REQUIRED_COLUMNS,
    SequenceGroup,
    allocate_exact_split,
    build_atomic_sequences,
    build_dataset,
    build_parser,
    copy_source_with_hash,
    direct_decimate,
    normalize_sheet_frame,
    read_condition_sheets,
    sha256_file,
    union_sequence_pair,
)


Cell = tuple[str, object]
Sheet = tuple[str, str, list[list[Cell]]]


def _cell_ref(col_idx: int, row_idx: int) -> str:
    letters = ""
    col = col_idx
    while col:
        col, rem = divmod(col - 1, 26)
        letters = chr(ord("A") + rem) + letters
    return f"{letters}{row_idx}"


def _shared(value: object) -> Cell:
    return "s", value


def _inline(value: object) -> Cell:
    return "inlineStr", value


def _string(value: object) -> Cell:
    return "str", value


def _number(value: object) -> Cell:
    return "n", value


def _worksheet_xml(rows: list[list[Cell]], shared_index: dict[str, int]) -> str:
    xml_rows: list[str] = []
    for row_idx, row in enumerate(rows, start=1):
        cells: list[str] = []
        for col_idx, (cell_type, value) in enumerate(row, start=1):
            ref = _cell_ref(col_idx, row_idx)
            text = escape(str(value))
            if cell_type == "s":
                cells.append(f'<c r="{ref}" t="s"><v>{shared_index[str(value)]}</v></c>')
            elif cell_type == "inlineStr":
                cells.append(f'<c r="{ref}" t="inlineStr"><is><t>{text}</t></is></c>')
            elif cell_type == "str":
                cells.append(f'<c r="{ref}" t="str"><v>{text}</v></c>')
            elif cell_type == "n":
                cells.append(f'<c r="{ref}" t="n"><v>{text}</v></c>')
            else:
                raise ValueError(f"Unsupported fixture cell type: {cell_type}")
        xml_rows.append(f'<row r="{row_idx}">{"".join(cells)}</row>')
    return (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(xml_rows)}</sheetData>'
        '</worksheet>'
    )


def _target_member(target: str) -> str:
    if target.startswith("/"):
        return posixpath.normpath(target.lstrip("/"))
    return posixpath.normpath(posixpath.join("xl", target))


def _write_minimal_multisheet_xlsx(path: Path, sheets: list[Sheet]) -> None:
    shared_values: list[str] = []
    for _, _, rows in sheets:
        for row in rows:
            for cell_type, value in row:
                text = str(value)
                if cell_type == "s" and text not in shared_values:
                    shared_values.append(text)
    shared_index = {value: idx for idx, value in enumerate(shared_values)}

    workbook_sheets: list[str] = []
    relationships: list[str] = []
    overrides: list[str] = []
    worksheet_parts: list[tuple[str, str]] = []
    for idx, (name, target, rows) in enumerate(sheets, start=1):
        rel_id = f"rId{idx * 7}"
        member = _target_member(target)
        workbook_sheets.append(
            f'<sheet name={quoteattr(name)} sheetId="{idx}" r:id="{rel_id}"/>'
        )
        relationships.append(
            f'<Relationship Id="{rel_id}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target={quoteattr(target)}/>'
        )
        overrides.append(
            f'<Override PartName="/{member}" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )
        worksheet_parts.append((member, _worksheet_xml(rows, shared_index)))

    workbook = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<sheets>{"".join(workbook_sheets)}</sheets></workbook>'
    )
    workbook_rels = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f'{"".join(relationships)}</Relationships>'
    )
    shared_strings = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        f'count="{len(shared_values)}" uniqueCount="{len(shared_values)}">'
        + "".join(f"<si><t>{escape(value)}</t></si>" for value in shared_values)
        + "</sst>"
    )
    content_types = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/sharedStrings.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/>'
        f'{"".join(overrides)}</Types>'
    )
    root_rels = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/></Relationships>'
    )

    with ZipFile(path, "w", ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", root_rels)
        zf.writestr("xl/workbook.xml", workbook)
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        zf.writestr("xl/sharedStrings.xml", shared_strings)
        for member, worksheet in worksheet_parts:
            zf.writestr(member, worksheet)


def _numeric_frame(times: list[float], loads: list[float] | None = None) -> pd.DataFrame:
    count = len(times)
    load_values = loads if loads is not None else [100.0 + index for index in range(count)]
    return pd.DataFrame(
        {
            "time_s": times,
            "load_kw": load_values,
            "fuel_cell_kw": [0.7 * value for value in load_values],
            "battery_kw": [0.3 * value for value in load_values],
            "bus_voltage_v": [540.0 + float(time_s) * 10.0 for time_s in times],
        }
    )


def _decimated_sequence(
    workbook: str,
    sheet: str,
    times: list[float],
    loads: list[float],
) -> pd.DataFrame:
    frame = _numeric_frame(times, loads)
    frame.insert(0, "source_row_index", np.arange(len(frame), dtype=np.int64) * 10)
    frame.insert(0, "source_sheet", sheet)
    frame.insert(0, "source_workbook", workbook)
    return frame


def _group(sequence_id: str, workbook: str, rows: int, mean: float) -> SequenceGroup:
    return SequenceGroup(
        sequence_id=sequence_id,
        source_workbook=workbook,
        rows=rows,
        load_mean=mean,
        load_std=0.0,
        load_q10=mean,
        load_q50=mean,
        load_q90=mean,
    )


def _condition_rows(start_s: float, count: int, load_base: float) -> list[list[Cell]]:
    rows: list[list[Cell]] = [[_inline(column) for column in REQUIRED_COLUMNS]]
    for index in range(count):
        time_s = start_s + index / 1000.0
        load_kw = load_base + index / max(count, 1)
        rows.append(
            [
                _number(time_s),
                _number(load_kw),
                _number(load_kw * 0.7),
                _number(load_kw * 0.3),
                _number(540.0 + time_s / 100.0),
            ]
        )
    return rows


class TestMillisecondWorkbookReader(unittest.TestCase):
    def test_public_schema_constants(self) -> None:
        self.assertEqual(
            REQUIRED_COLUMNS,
            ("time_s", "load_kw", "fuel_cell_kw", "battery_kw", "bus_voltage_v"),
        )
        expected_aliases = {
            "\u65f6\u95f4_s": "time_s",
            "\u8d1f\u8f7d\u529f\u7387_kW": "load_kw",
            "\u71c3\u6599\u7535\u6c60\u529f\u7387_kW": "fuel_cell_kw",
            "\u9502\u7535\u6c60\u529f\u7387_kW": "battery_kw",
            "\u6bcd\u7ebf\u7535\u538b_V": "bus_voltage_v",
            **{name: name for name in REQUIRED_COLUMNS},
        }
        self.assertEqual(HEADER_ALIASES, expected_aliases)
        self.assertEqual(FULL_OVERVIEW_NAME, "\u5168\u7a0b\u603b\u89c8")

    def test_reads_relationship_order_and_excludes_only_named_overview(self) -> None:
        chinese_headers = [
            "\u65f6\u95f4_s",
            "\u8d1f\u8f7d\u529f\u7387_kW",
            "\u71c3\u6599\u7535\u6c60\u529f\u7387_kW",
            "\u9502\u7535\u6c60\u529f\u7387_kW",
            "\u6bcd\u7ebf\u7535\u538b_V",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "conditions.xlsx"
            _write_minimal_multisheet_xlsx(
                path,
                [
                    (
                        "Condition Shared",
                        "worksheets/../worksheets/condition-main.xml",
                        [
                            [_shared(header) for header in chinese_headers],
                            [_number(0), _number(120.5), _number(100), _number(20.5), _number(750)],
                            [_number(0.01), _string("121.5"), _inline("101.0"), _number(20.5), _number(749.8)],
                        ],
                    ),
                    (
                        FULL_OVERVIEW_NAME,
                        "/xl/worksheets/overview-custom.xml",
                        [[_inline("summary")], [_string("excluded")]],
                    ),
                    (
                        "Summary Candidate",
                        "worksheets/condition-secondary.xml",
                        [
                            [_inline(header) for header in REQUIRED_COLUMNS],
                            [_number(1), _number(80), _number(70), _number(10), _number(748)],
                        ],
                    ),
                ],
            )

            result = read_condition_sheets(path, overview_names={FULL_OVERVIEW_NAME})

        self.assertEqual(list(result), ["Condition Shared", "Summary Candidate"])
        first = result["Condition Shared"]
        self.assertEqual(first.shape, (2, 5))
        self.assertEqual(tuple(first.columns), REQUIRED_COLUMNS)
        self.assertEqual(first.iloc[1].to_dict(), {
            "time_s": 0.01,
            "load_kw": 121.5,
            "fuel_cell_kw": 101.0,
            "battery_kw": 20.5,
            "bus_voltage_v": 749.8,
        })
        self.assertTrue(all(is_numeric_dtype(first[column]) for column in REQUIRED_COLUMNS))
        self.assertEqual(result["Summary Candidate"].shape, (1, 5))

    def test_normalize_rejects_missing_required_column(self) -> None:
        frame = pd.DataFrame(
            {
                "time_s": [0.0],
                "load_kw": [100.0],
                "fuel_cell_kw": [80.0],
                "battery_kw": [20.0],
            }
        )
        with self.assertRaisesRegex(
            ValueError,
            r"source\.xlsx.*Condition Missing.*missing required columns.*bus_voltage_v",
        ):
            normalize_sheet_frame(frame, source="source.xlsx", sheet="Condition Missing")

    def test_normalize_rejects_missing_and_nonfinite_values(self) -> None:
        valid = pd.DataFrame({column: [1.0] for column in REQUIRED_COLUMNS})
        for invalid in (None, float("inf"), float("-inf")):
            with self.subTest(invalid=invalid):
                frame = valid.copy()
                frame.loc[0, "load_kw"] = invalid
                with self.assertRaisesRegex(ValueError, r"source\.xlsx.*Condition Invalid.*non-finite"):
                    normalize_sheet_frame(frame, source="source.xlsx", sheet="Condition Invalid")


class TestMillisecondSourceProvenance(unittest.TestCase):
    def test_copy_source_with_hash_preserves_bytes_and_digest(self) -> None:
        payload = (b"0123456789abcdef" * 65537) + b"tail"
        expected_digest = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.xlsx"
            raw_dir = root / "raw"
            source.write_bytes(payload)

            result = copy_source_with_hash(source, raw_dir)
            repeated = copy_source_with_hash(source, raw_dir)
            copied = raw_dir / source.name

            self.assertTrue(raw_dir.is_dir())
            self.assertEqual(Path(result["source_path"]), source)
            self.assertEqual(Path(result["copied_path"]), copied)
            self.assertEqual(result["sha256"], expected_digest)
            self.assertEqual(result["bytes"], len(payload))
            self.assertEqual(copied.read_bytes(), payload)
            self.assertEqual(sha256_file(source), sha256_file(copied))
            self.assertEqual(repeated, result)

    def test_copy_source_rejects_existing_different_destination(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.xlsx"
            raw_dir = root / "raw"
            raw_dir.mkdir()
            source.write_bytes(b"authoritative source")
            destination = raw_dir / source.name
            destination.write_bytes(b"different existing data")

            with self.assertRaisesRegex(FileExistsError, r"source\.xlsx"):
                copy_source_with_hash(source, raw_dir)

            self.assertEqual(destination.read_bytes(), b"different existing data")


class TestDirectDecimation(unittest.TestCase):
    def test_keeps_exact_source_rows_zero_ten_twenty(self) -> None:
        frame = _numeric_frame(
            (np.arange(21, dtype=np.float64) / 1000.0).tolist(),
            (100.0 + np.arange(21, dtype=np.float64)).tolist(),
        )

        result = direct_decimate(
            frame,
            factor=10,
            source_workbook="book_a.xlsx",
            source_sheet="segment_1",
        )

        self.assertEqual(result["source_row_index"].tolist(), [0, 10, 20])
        self.assertEqual(result["load_kw"].tolist(), [100.0, 110.0, 120.0])
        self.assertEqual(result["source_workbook"].unique().tolist(), ["book_a.xlsx"])
        self.assertEqual(result["source_sheet"].unique().tolist(), ["segment_1"])
        np.testing.assert_allclose(np.diff(result["time_s"]), [0.01, 0.01], atol=1e-12)
        for _, output_row in result.iterrows():
            original = frame.iloc[int(output_row["source_row_index"])]
            np.testing.assert_array_equal(
                output_row.loc[list(REQUIRED_COLUMNS)].to_numpy(dtype=float),
                original.loc[list(REQUIRED_COLUMNS)].to_numpy(dtype=float),
            )

    def test_rejects_non_1ms_input(self) -> None:
        frame = _numeric_frame([0.000, 0.001, 0.003])
        with self.assertRaisesRegex(ValueError, "1 ms"):
            direct_decimate(
                frame,
                factor=10,
                source_workbook="book_a.xlsx",
                source_sheet="segment_bad",
            )

    def test_rejects_nonincreasing_time_and_wrong_factor(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            direct_decimate(
                _numeric_frame([0.000, 0.001, 0.001]),
                factor=10,
                source_workbook="book_a.xlsx",
                source_sheet="segment_bad",
            )
        with self.assertRaisesRegex(ValueError, "factor=10"):
            direct_decimate(
                _numeric_frame([0.000, 0.001]),
                factor=5,
                source_workbook="book_a.xlsx",
                source_sheet="segment_bad",
            )


class TestAtomicSequenceUnion(unittest.TestCase):
    def test_union_overlap_deduplicates_equal_rows_and_preserves_members(self) -> None:
        left = _decimated_sequence(
            "book_a.xlsx", "left", [0.00, 0.01, 0.02], [1.0, 2.0, 3.0]
        )
        right = _decimated_sequence(
            "book_a.xlsx", "right", [0.02, 0.03], [3.0, 4.0]
        )

        merged = union_sequence_pair(left, right, sequence_id="book_a__left__right")

        self.assertEqual(merged["time_ms"].tolist(), [0, 10, 20, 30])
        self.assertEqual(merged["sequence_id"].unique().tolist(), ["book_a__left__right"])
        overlap_members = merged.loc[merged["time_ms"] == 20, "source_members"].iloc[0]
        self.assertIn("book_a.xlsx|left|20", overlap_members)
        self.assertIn("book_a.xlsx|right|0", overlap_members)

    def test_union_overlap_rejects_disagreeing_values(self) -> None:
        left = _decimated_sequence("book_a.xlsx", "left", [0.00, 0.01], [1.0, 2.0])
        right = _decimated_sequence("book_a.xlsx", "right", [0.01, 0.02], [9.0, 3.0])
        with self.assertRaisesRegex(ValueError, "overlap disagreement"):
            union_sequence_pair(left, right, sequence_id="bad")

    def test_build_atomic_sequences_merges_only_declared_pair(self) -> None:
        left_key = ("book_a.xlsx", "left")
        right_key = ("book_a.xlsx", "right")
        solo_key = ("book_b.xlsx", "solo")
        decimated = {
            left_key: _decimated_sequence("book_a.xlsx", "left", [0.00, 0.01], [1.0, 2.0]),
            right_key: _decimated_sequence("book_a.xlsx", "right", [0.01, 0.02], [2.0, 3.0]),
            solo_key: _decimated_sequence("book_b.xlsx", "solo", [1.00, 1.01], [4.0, 5.0]),
        }

        result = build_atomic_sequences(
            decimated,
            overlap_pairs=((left_key, right_key),),
        )

        self.assertEqual(len(result), 2)
        self.assertEqual(sum(len(frame) for frame in result.values()), 5)
        self.assertEqual(
            {tuple(frame["source_workbook"].unique()) for frame in result.values()},
            {("book_a.xlsx",), ("book_b.xlsx",)},
        )

    def test_build_atomic_sequences_rejects_undeclared_duplicate_time(self) -> None:
        decimated = {
            ("book_a.xlsx", "left"): _decimated_sequence(
                "book_a.xlsx", "left", [0.00, 0.01], [1.0, 2.0]
            ),
            ("book_a.xlsx", "right"): _decimated_sequence(
                "book_a.xlsx", "right", [0.01, 0.02], [2.0, 3.0]
            ),
        }
        with self.assertRaisesRegex(ValueError, "duplicate.*atomic sequences"):
            build_atomic_sequences(decimated, overlap_pairs=())


class TestExactSplit(unittest.TestCase):
    def test_allocate_exact_split_is_disjoint_and_deterministic(self) -> None:
        groups = [
            _group("a1", "book_a.xlsx", 40, 1.0),
            _group("a2", "book_a.xlsx", 20, 2.0),
            _group("a3", "book_a.xlsx", 10, 3.0),
            _group("b1", "book_b.xlsx", 40, 1.5),
            _group("b2", "book_b.xlsx", 20, 2.5),
            _group("b3", "book_b.xlsx", 10, 3.5),
        ]
        loads = {
            group.sequence_id: np.full(group.rows, group.load_mean, dtype=np.float64)
            for group in groups
        }

        first = allocate_exact_split(
            groups,
            sequence_loads=loads,
            targets={"train": 80, "validation": 40, "test": 20},
            seed=20260710,
        )
        second = allocate_exact_split(
            groups,
            sequence_loads=loads,
            targets={"train": 80, "validation": 40, "test": 20},
            seed=20260710,
        )

        self.assertEqual(first, second)
        self.assertEqual(
            {name: sum(group.rows for group in first[name]) for name in first},
            {"train": 80, "validation": 40, "test": 20},
        )
        ids = {name: {group.sequence_id for group in first[name]} for name in first}
        self.assertFalse(ids["train"] & ids["validation"])
        self.assertFalse(ids["train"] & ids["test"])
        self.assertFalse(ids["validation"] & ids["test"])
        for name in ("train", "validation", "test"):
            self.assertEqual(
                {group.source_workbook for group in first[name]},
                {"book_a.xlsx", "book_b.xlsx"},
            )

    def test_allocate_exact_split_rejects_impossible_targets(self) -> None:
        with self.assertRaisesRegex(ValueError, "no exact valid assignment"):
            allocate_exact_split(
                [
                    _group("a", "book_a.xlsx", 4, 1.0),
                    _group("b", "book_b.xlsx", 3, 2.0),
                ],
                sequence_loads={"a": np.ones(4), "b": np.full(3, 2.0)},
                targets={"train": 4, "validation": 2, "test": 1},
                seed=1,
            )


class TestDatasetArtifacts(unittest.TestCase):
    def test_parser_defaults_are_isolated_from_existing_experiments(self) -> None:
        args = build_parser().parse_args([])
        self.assertIsNone(args.source)
        self.assertEqual(args.raw_root, Path("data/millisecond_1ms"))
        self.assertEqual(args.processed_root, Path("data/millisecond_10ms"))
        self.assertEqual(
            args.split_path,
            Path("outputs/config/millisecond_10ms_split_721.json"),
        )
        self.assertEqual(args.split_seed, 20260710)
        self.assertEqual(len(DEFAULT_SOURCE_PATHS), 2)

    def test_build_dataset_writes_audited_split_and_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources: list[Path] = []
            for workbook_name, load_offset in (("book_a.xlsx", 0.0), ("book_b.xlsx", 10.0)):
                path = root / workbook_name
                _write_minimal_multisheet_xlsx(
                    path,
                    [
                        (
                            FULL_OVERVIEW_NAME,
                            "worksheets/overview.xml",
                            [[_inline("excluded")], [_string("excluded")]],
                        ),
                        (
                            "segment_40",
                            "worksheets/segment-40.xml",
                            _condition_rows(0.0, 400, load_offset + 1.0),
                        ),
                        (
                            "segment_20",
                            "worksheets/segment-20.xml",
                            _condition_rows(10.0, 200, load_offset + 2.0),
                        ),
                        (
                            "segment_10",
                            "worksheets/segment-10.xml",
                            _condition_rows(20.0, 100, load_offset + 3.0),
                        ),
                    ],
                )
                sources.append(path)

            raw_root = root / "millisecond_1ms"
            processed_root = root / "millisecond_10ms"
            split_path = root / "config" / "split.json"
            result = build_dataset(
                source_paths=sources,
                raw_root=raw_root,
                processed_root=processed_root,
                split_path=split_path,
                split_seed=20260710,
                overlap_pairs=(),
                split_targets={"train": 80, "validation": 40, "test": 20},
            )

            self.assertEqual(result["condition_sheets"], 6)
            self.assertEqual(result["rows_1ms"], 1400)
            self.assertEqual(result["rows_10ms_before_overlap_removal"], 140)
            self.assertEqual(result["unique_rows_10ms"], 140)
            self.assertEqual(result["atomic_sequences"], 6)
            self.assertEqual(result["split_rows"], {"train": 80, "validation": 40, "test": 20})

            source_manifest = json.loads(
                (raw_root / "source_manifest.json").read_text(encoding="utf-8")
            )
            dataset_manifest = json.loads(
                (processed_root / "dataset_manifest.json").read_text(encoding="utf-8")
            )
            split_manifest = json.loads(split_path.read_text(encoding="utf-8"))
            combined = pd.read_csv(processed_root / "millisecond_load_10ms.csv")

            self.assertEqual(len(source_manifest["sources"]), 2)
            self.assertTrue(all(len(item["sha256"]) == 64 for item in source_manifest["sources"]))
            self.assertEqual(dataset_manifest["sample_interval_ms"], 10)
            self.assertEqual(dataset_manifest["history_steps"], 30)
            self.assertEqual(dataset_manifest["prediction_steps"], 6)
            self.assertEqual(dataset_manifest["scaler_fit_scope"], "train_rows_only")
            self.assertEqual(dataset_manifest["direct_decimation"]["source_indices"], "0,10,20,...")
            self.assertEqual(len(dataset_manifest["atomic_sequences"]), 6)
            self.assertTrue(
                all(len(item["sha256"]) == 64 for item in dataset_manifest["atomic_sequences"])
            )
            self.assertEqual(split_manifest["split_rows"], result["split_rows"])
            self.assertEqual(
                split_manifest["window_formula"],
                "max(rows - 30 - 6 + 1, 0) per atomic sequence",
            )
            for split_name in ("train", "validation", "test"):
                self.assertEqual(
                    set(split_manifest["source_workbooks_by_split"][split_name]),
                    {"book_a.xlsx", "book_b.xlsx"},
                )
            self.assertFalse(combined.duplicated(["sequence_id", "time_ms"]).any())
            self.assertEqual(
                combined.groupby("split").size().to_dict(),
                {"test": 20, "train": 80, "validation": 40},
            )
            self.assertEqual(len(list((processed_root / "segments").glob("*.csv"))), 6)
            self.assertEqual(len(list((raw_root / "raw").glob("*.xlsx"))), 2)


if __name__ == "__main__":
    unittest.main()
