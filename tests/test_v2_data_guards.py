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

    def test_non_excel_record_is_rejected_even_with_complete_original_metadata(
        self,
    ) -> None:
        from v2.data.raw_inventory import (
            ExcelInventoryRecord,
            MeasurementSourceClass,
            UnsupportedRawSourceError,
        )

        with self.assertRaises(UnsupportedRawSourceError):
            ExcelInventoryRecord(
                workbook=Path("device-export.csv"),
                source_class=MeasurementSourceClass.ORIGINAL_MEASUREMENT,
                sheet="Telemetry",
                column="Power",
                unit="kW",
                timestamp="Time",
                actual_sampling_interval_seconds=30.0,
                missing_rate=0.0,
                physical_meaning="measured power",
                usable=True,
                reason="claimed original device export",
            )

    def test_known_excel_derivative_classes_cannot_be_marked_usable(self) -> None:
        from v2.data.raw_inventory import (
            ExcelInventoryRecord,
            MeasurementSourceClass,
        )

        derivative_sources = (
            ("1修改名称.xlsx", MeasurementSourceClass.RENAME_HELPER),
            ("Cleaned_Power_Data.xlsx", MeasurementSourceClass.PROCESSED_AGGREGATE),
            ("interpolated_load.xlsx", MeasurementSourceClass.INTERPOLATED),
            ("ship_training_profile.xlsx", MeasurementSourceClass.GENERATED),
            ("extracted_curves.xlsx", MeasurementSourceClass.DIGITIZED),
        )
        for workbook, source_class in derivative_sources:
            with self.subTest(workbook=workbook), self.assertRaises(ValueError):
                ExcelInventoryRecord(
                    workbook=Path(workbook),
                    source_class=source_class,
                    sheet="Sheet1",
                    column="Power",
                    unit="kW",
                    timestamp="Time",
                    actual_sampling_interval_seconds=30.0,
                    missing_rate=0.0,
                    physical_meaning="derived power",
                    usable=True,
                    reason="known derivative lineage",
                )

    def test_only_explicit_original_measurement_records_satisfy_inventory(
        self,
    ) -> None:
        from v2.data.raw_inventory import (
            ExcelInventoryRecord,
            MeasurementSourceClass,
            RawExcelInventory,
        )

        workbook = Path("device-export.xlsx")
        record = ExcelInventoryRecord(
            workbook=workbook,
            source_class=MeasurementSourceClass.ORIGINAL_MEASUREMENT,
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

    def test_inventory_normalizes_immutable_container_fields(self) -> None:
        from v2.data.raw_inventory import (
            ExcelInventoryRecord,
            MeasurementSourceClass,
            RawExcelInventory,
        )

        record = ExcelInventoryRecord(
            workbook=Path("device-export.xlsx"),
            source_class=MeasurementSourceClass.ORIGINAL_MEASUREMENT,
            sheet="Telemetry",
            column="Power",
            unit="kW",
            timestamp="Time",
            actual_sampling_interval_seconds=30,
            missing_rate=0,
            physical_meaning="measured power",
            usable=True,
            reason="authorized original export",
        )
        inventory = RawExcelInventory(
            root="audit-root",  # type: ignore[arg-type]
            records=[record],  # type: ignore[arg-type]
        )

        self.assertEqual(inventory.root, Path("audit-root"))
        self.assertIs(type(inventory.records), tuple)
        self.assertEqual(inventory.records, (record,))
        self.assertIs(type(record.actual_sampling_interval_seconds), float)
        self.assertIs(type(record.missing_rate), float)

    def test_inventory_rejects_fake_records_before_payload_loader(self) -> None:
        from v2.data.raw_inventory import RawExcelInventory

        accesses = 0

        class FakeRecord:
            usable = True
            workbook = Path("forged.xlsx")

        def payload_loader() -> object:
            nonlocal accesses
            accesses += 1
            return object()

        with self.assertRaises(TypeError):
            inventory = RawExcelInventory(
                root=Path("."),
                records=(FakeRecord(),),  # type: ignore[arg-type]
            )
            from v2.preflight import load_train_payload

            load_train_payload(
                split="Train",
                inventory=inventory,
                technical_specification=None,
                payload_loader=payload_loader,
            )

        self.assertEqual(accesses, 0)

    def test_inventory_rejects_record_subclasses(self) -> None:
        from v2.data.raw_inventory import (
            ExcelInventoryRecord,
            MeasurementSourceClass,
            RawExcelInventory,
        )

        class ExcelInventoryRecordSubclass(ExcelInventoryRecord):
            pass

        record = ExcelInventoryRecordSubclass(
            workbook=Path("device-export.xlsx"),
            source_class=MeasurementSourceClass.ORIGINAL_MEASUREMENT,
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

        with self.assertRaises(TypeError):
            RawExcelInventory(root=Path("."), records=(record,))

    def test_measurement_record_requires_a_strict_boolean_usable_flag(self) -> None:
        from v2.data.raw_inventory import ExcelInventoryRecord

        with self.assertRaises(ValueError):
            ExcelInventoryRecord(
                workbook=Path("device-export.xlsx"),
                source_class="original_measurement",  # type: ignore[arg-type]
                sheet="Telemetry",
                column="Power",
                unit="kW",
                timestamp="Time",
                actual_sampling_interval_seconds=30.0,
                missing_rate=0.0,
                physical_meaning="measured power",
                usable="False",  # type: ignore[arg-type]
                reason="invalid usable flag",
            )

    def test_measurement_record_rejects_invalid_numeric_metadata(self) -> None:
        from v2.data.raw_inventory import ExcelInventoryRecord

        valid = {
            "workbook": Path("device-export.xlsx"),
            "source_class": "original_measurement",
            "sheet": "Telemetry",
            "column": "Power",
            "unit": "kW",
            "timestamp": "Time",
            "actual_sampling_interval_seconds": 30.0,
            "missing_rate": 0.0,
            "physical_meaning": "measured power",
            "usable": True,
            "reason": "authorized original export",
        }
        invalid_overrides = (
            {"actual_sampling_interval_seconds": True},
            {"actual_sampling_interval_seconds": "30"},
            {"actual_sampling_interval_seconds": float("nan")},
            {"actual_sampling_interval_seconds": float("inf")},
            {"actual_sampling_interval_seconds": 0},
            {"missing_rate": False},
            {"missing_rate": "0"},
            {"missing_rate": float("nan")},
            {"missing_rate": float("inf")},
            {"missing_rate": -0.01},
            {"missing_rate": 1.01},
        )
        for override in invalid_overrides:
            with self.subTest(override=override), self.assertRaises(ValueError):
                ExcelInventoryRecord(**(valid | override))

    def test_missing_rate_boundaries_are_valid_and_normalized(self) -> None:
        from v2.data.raw_inventory import ExcelInventoryRecord

        for missing_rate in (0, 1):
            with self.subTest(missing_rate=missing_rate):
                record = ExcelInventoryRecord(
                    workbook=Path("device-export.xlsx"),
                    source_class="original_measurement",  # type: ignore[arg-type]
                    sheet="Telemetry",
                    column="Power",
                    unit="kW",
                    timestamp="Time",
                    actual_sampling_interval_seconds=30,
                    missing_rate=missing_rate,
                    physical_meaning="measured power",
                    usable=True,
                    reason="authorized original export",
                )

                self.assertEqual(record.missing_rate, float(missing_rate))
                self.assertIs(type(record.missing_rate), float)
                self.assertIs(type(record.actual_sampling_interval_seconds), float)

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
                technical_specification=None,
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
            MeasurementSourceClass,
            RawExcelInventory,
        )
        from v2.preflight import (
            TechnicalSpecificationRecord,
            TechnicalSpecificationSourceClass,
            load_train_payload,
        )

        accesses = 0

        def payload_loader() -> str:
            nonlocal accesses
            accesses += 1
            return "payload"

        record = ExcelInventoryRecord(
            workbook=Path("device-export.xlsx"),
            source_class=MeasurementSourceClass.ORIGINAL_MEASUREMENT,
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
        specification = TechnicalSpecificationRecord(
            document=Path("vessel-technical-specification.pdf"),
            source_class=(
                TechnicalSpecificationSourceClass.AUTHORITATIVE_VESSEL_SPECIFICATION
            ),
            sha256="a" * 64,
            page_count=19,
            source_identifier="Three Gorges Hydrogen Boat No. 1",
            source_reference="approved propulsion-system specification V1",
        )

        for split in ("Validation", "Test"):
            with self.subTest(split=split), self.assertRaises(HeldOutDataAccessError):
                load_train_payload(
                    split=split,
                    inventory=inventory,
                    technical_specification=specification,
                    payload_loader=payload_loader,
                )
        self.assertEqual(accesses, 0)

    def test_preflight_loads_train_payload_only_after_all_gates_pass(self) -> None:
        from v2.data.raw_inventory import (
            ExcelInventoryRecord,
            MeasurementSourceClass,
            RawExcelInventory,
        )
        from v2.preflight import (
            TechnicalSpecificationRecord,
            TechnicalSpecificationSourceClass,
            load_train_payload,
        )

        record = ExcelInventoryRecord(
            workbook=Path("device-export.xlsx"),
            source_class=MeasurementSourceClass.ORIGINAL_MEASUREMENT,
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
        specification = TechnicalSpecificationRecord(
            document=Path("vessel-technical-specification.pdf"),
            source_class=(
                TechnicalSpecificationSourceClass.AUTHORITATIVE_VESSEL_SPECIFICATION
            ),
            sha256="C269F9D7E9DEDC23118514FED8EBC0987500948ABF88F8A2129EA0F264315A34",
            page_count=19,
            source_identifier="Three Gorges Hydrogen Boat No. 1",
            source_reference="approved propulsion-system specification V1",
        )

        payload = load_train_payload(
            split="tRaIn",
            inventory=inventory,
            technical_specification=specification,
            payload_loader=lambda: {"rows": 10},
        )

        self.assertEqual(payload, {"rows": 10})

    def test_technical_specification_record_rejects_non_authoritative_evidence(
        self,
    ) -> None:
        from v2.preflight import TechnicalSpecificationRecord

        valid = {
            "document": Path("vessel-specification.pdf"),
            "source_class": "authoritative_vessel_specification",
            "sha256": "b" * 64,
            "page_count": 19,
            "source_identifier": "Three Gorges Hydrogen Boat No. 1",
            "source_reference": "approved propulsion-system specification V1",
        }
        invalid_overrides = (
            {"document": Path("vessel-specification.xlsx")},
            {"source_class": "processed_summary"},
            {"sha256": "not-a-sha256"},
            {"page_count": 0},
            {"page_count": True},
            {"source_identifier": ""},
            {"source_reference": "   "},
        )
        for override in invalid_overrides:
            with self.subTest(override=override), self.assertRaises(ValueError):
                TechnicalSpecificationRecord(**(valid | override))

    def test_boolean_cannot_bypass_technical_specification_gate(self) -> None:
        from v2.data.raw_inventory import RawExcelInventory
        from v2.preflight import assess_data_preflight

        with self.assertRaises(TypeError):
            assess_data_preflight(
                inventory=RawExcelInventory(root=Path("."), records=()),
                technical_specification_available=True,
            )
        with self.assertRaises(TypeError):
            assess_data_preflight(
                inventory=RawExcelInventory(root=Path("."), records=()),
                technical_specification=True,  # type: ignore[arg-type]
            )

    def test_plant_values_are_research_simulation_configuration(self) -> None:
        from v2.config import PlantConfig

        plant = PlantConfig.research_simulation()

        self.assertEqual(plant.fuel_cell_rated_total_kw, 600.0)
        self.assertEqual(plant.battery_nominal_energy_kwh, 624.0)
        self.assertEqual(plant.source_type, "research_simulation")
        self.assertIn("10.1016/j.oceaneng.2026.125687", plant.source_reference)
        self.assertNotIn("JMSE", plant.source_reference)
        self.assertFalse(hasattr(PlantConfig, "project_configuration"))

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
