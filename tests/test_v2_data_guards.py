from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class V2DataGuardTests(unittest.TestCase):
    def test_only_original_excel_extensions_are_raw_measurement_sources(self) -> None:
        from v2.data.raw_inventory import RawSourcePolicy, UnsupportedRawSourceError

        for name in ("voyage.xlsx", "voyage.xls"):
            RawSourcePolicy.require_original_measurement(Path(name))
        for name in ("legacy.csv", "interpolation.mat", "plot.png"):
            with self.subTest(name=name), self.assertRaises(UnsupportedRawSourceError):
                RawSourcePolicy.require_original_measurement(Path(name))

    def test_candidate_source_classes_include_excel_and_technical_specification(self) -> None:
        from v2.data.raw_inventory import RawSourcePolicy, UnsupportedRawSourceError

        self.assertEqual(
            RawSourcePolicy.require_candidate_source(Path("telemetry.XLSX")),
            "excel_workbook",
        )
        self.assertEqual(
            RawSourcePolicy.require_candidate_source(Path("技术规格书.PDF")),
            "technical_specification",
        )
        for name in ("old.csv", "legacy.mat", "interpolated.png"):
            with self.subTest(name=name), self.assertRaises(UnsupportedRawSourceError):
                RawSourcePolicy.require_candidate_source(Path(name))

    def test_calibration_and_selection_are_train_only(self) -> None:
        from v2.data.raw_inventory import HeldOutDataAccessError, require_train_only

        require_train_only("Train")
        for split in ("Validation", "Test"):
            with self.subTest(split=split), self.assertRaises(HeldOutDataAccessError):
                require_train_only(split)

    def test_inventory_reports_no_workbooks_without_inventing_rows(self) -> None:
        from v2.data.raw_inventory import scan_workbooks

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "derived.csv").write_text("time,power\n0,1\n", encoding="utf-8")
            inventory = scan_workbooks(root)

        self.assertEqual(inventory.workbooks, ())
        self.assertFalse(inventory.raw_measurements_available)

    def test_candidate_workbook_does_not_imply_usable_measurements(self) -> None:
        from v2.data.raw_inventory import scan_workbooks

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            helper = root / "1修改名称.xlsx"
            helper.touch()
            inventory = scan_workbooks(root)

        self.assertEqual(inventory.workbooks, (helper.resolve(),))
        self.assertEqual(len(inventory.records), 1)
        self.assertIsNone(inventory.records[0].sheet)
        self.assertIsNone(inventory.records[0].timestamp)
        self.assertFalse(inventory.records[0].usable)
        self.assertFalse(inventory.raw_measurements_available)

    def test_only_explicit_usable_measurement_records_satisfy_inventory(self) -> None:
        from v2.data.raw_inventory import ExcelInventoryRecord, RawExcelInventory

        workbook = Path("device-export.xlsx")
        record = ExcelInventoryRecord(
            workbook=workbook,
            sheet="Telemetry",
            column="StackPower",
            unit="kW",
            timestamp="Timestamp",
            actual_sampling_interval_seconds=30.0,
            missing_rate=0.0,
            physical_meaning="measured aggregate fuel-cell stack output",
            usable=True,
            reason="original timestamped device export authorized for v2",
        )
        inventory = RawExcelInventory(root=Path("."), records=(record,))

        self.assertTrue(inventory.raw_measurements_available)
        self.assertEqual(inventory.usable_measurements, (record,))

    def test_preflight_reports_each_missing_evidence_item_without_loading(self) -> None:
        from v2.data.raw_inventory import RawExcelInventory
        from v2.preflight import PreflightBlockedError, load_train_payload

        accesses = 0

        def payload_loader() -> object:
            nonlocal accesses
            accesses += 1
            return object()

        inventory = RawExcelInventory(root=Path("."), records=())
        with self.assertRaises(PreflightBlockedError) as caught:
            load_train_payload(
                split="Train",
                inventory=inventory,
                technical_specification_available=False,
                payload_loader=payload_loader,
            )

        self.assertEqual(accesses, 0)
        self.assertEqual(
            {issue.code for issue in caught.exception.report.issues},
            {"missing_usable_raw_measurements", "missing_technical_specification"},
        )
        self.assertNotEqual(str(caught.exception), "BLOCKED")

    def test_split_gate_runs_before_payload_access(self) -> None:
        from v2.data.raw_inventory import (
            ExcelInventoryRecord,
            HeldOutDataAccessError,
            RawExcelInventory,
        )
        from v2.preflight import load_train_payload

        accesses = 0

        def payload_loader() -> str:
            nonlocal accesses
            accesses += 1
            return "payload"

        record = ExcelInventoryRecord(
            workbook=Path("device-export.xlsx"),
            sheet="Telemetry",
            column="Power",
            unit="kW",
            timestamp="Time",
            actual_sampling_interval_seconds=30.0,
            missing_rate=0.0,
            physical_meaning="measured power",
            usable=True,
            reason="authorized original export",
        )
        inventory = RawExcelInventory(root=Path("."), records=(record,))

        for split in ("Validation", "Test"):
            with self.subTest(split=split), self.assertRaises(HeldOutDataAccessError):
                load_train_payload(
                    split=split,
                    inventory=inventory,
                    technical_specification_available=True,
                    payload_loader=payload_loader,
                )
        self.assertEqual(accesses, 0)

    def test_preflight_loads_train_payload_only_after_all_gates_pass(self) -> None:
        from v2.data.raw_inventory import ExcelInventoryRecord, RawExcelInventory
        from v2.preflight import load_train_payload

        record = ExcelInventoryRecord(
            workbook=Path("device-export.xlsx"),
            sheet="Telemetry",
            column="Power",
            unit="kW",
            timestamp="Time",
            actual_sampling_interval_seconds=30.0,
            missing_rate=0.01,
            physical_meaning="measured power",
            usable=True,
            reason="authorized original export",
        )
        inventory = RawExcelInventory(root=Path("."), records=(record,))

        payload = load_train_payload(
            split="tRaIn",
            inventory=inventory,
            technical_specification_available=True,
            payload_loader=lambda: {"rows": 10},
        )

        self.assertEqual(payload, {"rows": 10})

    def test_plant_values_are_research_simulation_configuration(self) -> None:
        from v2.config import PlantConfig

        plant = PlantConfig.research_simulation()

        self.assertEqual(plant.fuel_cell_rated_total_kw, 600.0)
        self.assertEqual(plant.battery_nominal_energy_kwh, 624.0)
        self.assertEqual(plant.source_type, "research_simulation")
        self.assertIn("10.1016/j.oceaneng.2026.125687", plant.source_reference)
        self.assertNotIn("JMSE", plant.source_reference)

    def test_real_vessel_specification_is_separate_from_simulation(self) -> None:
        from v2.config import PlantConfig, RealVesselSpecification

        research = PlantConfig.research_simulation()
        vessel = RealVesselSpecification.from_authoritative_specification()

        self.assertEqual(vessel.fuel_cell_rated_total_kw, 560.0)
        self.assertEqual(vessel.fuel_cell_module_count, 8)
        self.assertEqual(vessel.fuel_cell_module_rated_kw, 70.0)
        self.assertEqual(vessel.battery_nominal_energy_kwh, 1806.0)
        self.assertEqual(vessel.battery_cluster_count, 12)
        self.assertEqual(vessel.battery_rated_output_min_kw, 900.0)
        self.assertEqual(vessel.battery_rated_voltage_v, 537.6)
        self.assertTrue(vessel.operation_ends_with_shore_charging)
        self.assertNotEqual(
            research.fuel_cell_rated_total_kw,
            vessel.fuel_cell_rated_total_kw,
        )


if __name__ == "__main__":
    unittest.main()
