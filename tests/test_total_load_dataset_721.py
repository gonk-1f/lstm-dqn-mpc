from __future__ import annotations

import sys
import tempfile
import unittest
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "src" / "main"
if str(MAIN) not in sys.path:
    sys.path.insert(0, str(MAIN))

from build_total_load_dataset_721 import build_dataset  # noqa: E402


def _cell_ref(col_idx: int, row_idx: int) -> str:
    letters = ""
    col = col_idx
    while col:
        col, rem = divmod(col - 1, 26)
        letters = chr(ord("A") + rem) + letters
    return f"{letters}{row_idx}"


def _write_minimal_xlsx(path: Path, rows: list[dict[str, object]]) -> None:
    headers = list(rows[0].keys())
    sheet_rows: list[str] = []
    all_rows = [dict(zip(headers, headers))] + rows
    for row_idx, row in enumerate(all_rows, start=1):
        cells = []
        for col_idx, header in enumerate(headers, start=1):
            value = row[header]
            ref = _cell_ref(col_idx, row_idx)
            if isinstance(value, (int, float)):
                cells.append(f'<c r="{ref}" t="n"><v>{value}</v></c>')
            else:
                cells.append(f'<c r="{ref}" t="str"><v>{value}</v></c>')
        sheet_rows.append(f'<row r="{row_idx}">{"".join(cells)}</row>')
    sheet = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(sheet_rows)}</sheetData>'
        '</worksheet>'
    )
    workbook = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets></workbook>'
    )
    rels = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet1.xml"/></Relationships>'
    )
    content_types = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        '</Types>'
    )
    root_rels = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/></Relationships>'
    )
    with ZipFile(path, "w", ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", root_rels)
        zf.writestr("xl/workbook.xml", workbook)
        zf.writestr("xl/_rels/workbook.xml.rels", rels)
        zf.writestr("xl/worksheets/sheet1.xml", sheet)


class TestTotalLoadDataset721(unittest.TestCase):
    def test_build_dataset_preserves_energy_side_total_load_and_splits_by_voyage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "excels"
            output_dir = root / "dataset"
            config_dir = root / "config"
            input_dir.mkdir()
            for idx, day in enumerate([1, 2, 3], start=1):
                _write_minimal_xlsx(
                    input_dir / f"voyage_{idx}.xlsx",
                    [
                        {
                            "timestamp": f"2024-01-0{day} 00:00:00",
                            "fuel_cell_total_kw": 10.0 * idx,
                            "battery_total_kw": 2.0,
                            "total_load_fc_plus_batt_kw": 10.0 * idx + 2.0,
                            "propulsion_inverter_total_kw": 9.0 * idx,
                        },
                        {
                            "timestamp": f"2024-01-0{day} 00:00:30",
                            "fuel_cell_total_kw": 11.0 * idx,
                            "battery_total_kw": -1.0,
                            "total_load_fc_plus_batt_kw": 11.0 * idx - 1.0,
                            "propulsion_inverter_total_kw": 9.0 * idx,
                        },
                    ],
                )

            result = build_dataset(
                input_dir=input_dir,
                output_dir=output_dir,
                config_dir=config_dir,
                expected_count=3,
                split_counts=(1, 1, 1),
            )

            self.assertEqual(result["excel_count"], 3)
            self.assertEqual(result["train_count"], 1)
            self.assertEqual(result["validation_count"], 1)
            self.assertEqual(result["test_count"], 1)
            summary = pd.read_csv(output_dir / "summary_total_load_66.csv")
            self.assertEqual(list(summary["voyage_id"]), ["voyage_001", "voyage_002", "voyage_003"])
            self.assertEqual(int(summary["timestamp_gap_count"].sum()), 0)
            segments = pd.read_csv(output_dir / "total_load_66_segments.csv")
            self.assertIn("load_total_kw", segments.columns)
            self.assertIn("voyage_name", segments.columns)
            self.assertAlmostEqual(float(segments.iloc[0]["load_total_kw"]), 12.0)
            self.assertAlmostEqual(float(segments.iloc[0]["load_gap_kw"]), 3.0)
            self.assertTrue((segments["load_total_kw"] == segments["fuel_cell_total_kw"] + segments["battery_total_kw"]).all())
            split_text = (config_dir / "SPLIT_TOTAL_LOAD_721.txt").read_text(encoding="utf-8")
            self.assertIn("总航段数：3", split_text)
            self.assertIn("目标负荷：total_load_fc_plus_batt_kw", split_text)

    def test_build_dataset_can_merge_ais_speed_without_changing_load_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "excels"
            output_dir = root / "dataset"
            config_dir = root / "config"
            ais_root = root / "ais"
            input_dir.mkdir()
            _write_minimal_xlsx(
                input_dir / "voyage_1.xlsx",
                [
                    {
                        "timestamp": "2024-01-01 00:00:30",
                        "fuel_cell_total_kw": 10.0,
                        "battery_total_kw": 2.0,
                        "total_load_fc_plus_batt_kw": 12.0,
                        "propulsion_inverter_total_kw": 9.0,
                    },
                    {
                        "timestamp": "2024-01-01 00:01:00",
                        "fuel_cell_total_kw": 11.0,
                        "battery_total_kw": -1.0,
                        "total_load_fc_plus_batt_kw": 10.0,
                        "propulsion_inverter_total_kw": 9.0,
                    },
                ],
            )
            speed_dir = ais_root / "voyage_1" / "\u63a8\u8fdb\u7cfb\u7edf"
            speed_dir.mkdir(parents=True)
            pd.DataFrame(
                {
                    "Time": ["2024-01-01 00:00:15", "2024-01-01 00:00:45"],
                    "\u822a\u901f(\u8282)": ["1.5 kn", "2.0 kn"],
                }
            ).to_csv(speed_dir / "AIS_speed.csv", index=False, encoding="utf-8-sig")

            result = build_dataset(
                input_dir=input_dir,
                output_dir=output_dir,
                config_dir=config_dir,
                expected_count=1,
                split_counts=(1, 0, 0),
                include_ais_speed=True,
                ais_root=ais_root,
            )

            self.assertTrue(result["include_ais_speed"])
            segments = pd.read_csv(output_dir / "total_load_66_segments.csv")
            self.assertIn("speed_knots", segments.columns)
            self.assertAlmostEqual(float(segments.iloc[0]["speed_knots"]), 1.5)
            self.assertAlmostEqual(float(segments.iloc[1]["speed_knots"]), 2.0)
            self.assertTrue((segments["load_total_kw"] == segments["fuel_cell_total_kw"] + segments["battery_total_kw"]).all())
            self.assertTrue((config_dir / "voyage_split_total_load_speed_721.json").exists())

    def test_build_dataset_can_build_1s_split_and_preserve_existing_speed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "excels_1s"
            output_dir = root / "dataset_1s"
            config_dir = root / "config"
            input_dir.mkdir()
            for idx, day in enumerate([1, 2, 3], start=1):
                _write_minimal_xlsx(
                    input_dir / f"voyage_{idx}.xlsx",
                    [
                        {
                            "timestamp": f"2024-01-0{day} 00:00:00",
                            "fuel_cell_total_kw": 10.0 * idx,
                            "battery_total_kw": 2.0,
                            "total_load_fc_plus_batt_kw": 10.0 * idx + 2.0,
                            "speed_knots": 1.0 * idx,
                        },
                        {
                            "timestamp": f"2024-01-0{day} 00:00:01",
                            "fuel_cell_total_kw": 11.0 * idx,
                            "battery_total_kw": -1.0,
                            "total_load_fc_plus_batt_kw": 11.0 * idx - 1.0,
                            "speed_knots": 1.0 * idx + 0.5,
                        },
                    ],
                )

            result = build_dataset(
                input_dir=input_dir,
                output_dir=output_dir,
                config_dir=config_dir,
                expected_count=3,
                split_counts=(1, 1, 1),
                sample_interval_seconds=1.0,
                include_existing_speed=True,
                split_json_name="voyage_split_test_sample_1s.json",
                split_txt_name="SPLIT_TEST_SAMPLE_1S.txt",
            )

            self.assertEqual(result["sample_interval_seconds"], 1.0)
            summary = pd.read_csv(output_dir / "summary_total_load_66.csv")
            self.assertEqual(int(summary["timestamp_gap_count"].sum()), 0)
            segments = pd.read_csv(output_dir / "total_load_66_segments.csv")
            self.assertIn("speed_knots", segments.columns)
            self.assertAlmostEqual(float(segments.iloc[1]["speed_knots"]), 1.5)
            split = json.loads((config_dir / "voyage_split_test_sample_1s.json").read_text(encoding="utf-8"))
            self.assertEqual(float(split["sample_interval_seconds"]), 1.0)
            self.assertEqual(split["train"], ["voyage_001"])


if __name__ == "__main__":
    unittest.main()
