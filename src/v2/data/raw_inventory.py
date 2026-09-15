from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from pathlib import Path


class UnsupportedRawSourceError(ValueError):
    pass


class HeldOutDataAccessError(PermissionError):
    pass


class MeasurementSourceClass(str, Enum):
    UNAUDITED_CANDIDATE = "unaudited_candidate"
    ORIGINAL_MEASUREMENT = "original_measurement"
    RENAME_HELPER = "rename_helper"
    PROCESSED_AGGREGATE = "processed_aggregate"
    INTERPOLATED = "interpolated"
    GENERATED = "generated"
    DIGITIZED = "digitized"


class RawSourcePolicy:
    ORIGINAL_EXCEL_SUFFIXES = frozenset({".xlsx", ".xls"})
    TECHNICAL_SPECIFICATION_SUFFIXES = frozenset({".pdf"})

    @classmethod
    def require_candidate_source(cls, path: Path) -> str:
        suffix = Path(path).suffix.casefold()
        if suffix in cls.ORIGINAL_EXCEL_SUFFIXES:
            return "excel_workbook"
        if suffix in cls.TECHNICAL_SPECIFICATION_SUFFIXES:
            return "technical_specification"
        raise UnsupportedRawSourceError(
            "v2 candidate sources are user-provided Excel workbooks or technical "
            "specifications; CSV, MAT, interpolations, and image transcriptions "
            "cannot be promoted to formal raw facts"
        )

    @classmethod
    def require_original_measurement(cls, path: Path) -> None:
        if Path(path).suffix.casefold() not in cls.ORIGINAL_EXCEL_SUFFIXES:
            raise UnsupportedRawSourceError(
                "v2 raw measurements must come from an original Excel workbook; "
                "CSV, MAT, interpolations, and image transcriptions are not raw facts"
            )


@dataclass(frozen=True)
class ExcelInventoryRecord:
    """Explicit audit result for one prospective workbook measurement column.

    An accepted file extension establishes only candidacy.  ``usable`` becomes
    true only after the workbook's measurement semantics have been audited and
    recorded in every metadata field below.
    """

    workbook: Path
    source_class: MeasurementSourceClass
    sheet: str | None
    column: str | None
    unit: str | None
    timestamp: str | None
    actual_sampling_interval_seconds: float | None
    missing_rate: float | None
    physical_meaning: str | None
    usable: bool
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "workbook", Path(self.workbook))
        try:
            source_class = MeasurementSourceClass(self.source_class)
        except (TypeError, ValueError) as exc:
            raise ValueError("measurement source_class must be explicit and known") from exc
        object.__setattr__(self, "source_class", source_class)
        RawSourcePolicy.require_original_measurement(self.workbook)
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("inventory record reason must be explicit")
        if not self.usable:
            return
        if source_class is not MeasurementSourceClass.ORIGINAL_MEASUREMENT:
            raise ValueError(
                "usable measurement records must have original_measurement lineage"
            )
        required_text = {
            "sheet": self.sheet,
            "column": self.column,
            "unit": self.unit,
            "timestamp": self.timestamp,
            "physical_meaning": self.physical_meaning,
        }
        missing = [
            name
            for name, value in required_text.items()
            if not isinstance(value, str) or not value.strip()
        ]
        if missing:
            raise ValueError(
                "usable measurement records require " + ", ".join(missing)
            )
        interval = self.actual_sampling_interval_seconds
        if interval is None or not math.isfinite(interval) or interval <= 0:
            raise ValueError(
                "usable measurement records require a finite positive sampling interval"
            )
        missing_rate = self.missing_rate
        if (
            missing_rate is None
            or not math.isfinite(missing_rate)
            or not 0.0 <= missing_rate <= 1.0
        ):
            raise ValueError("usable measurement records require missing_rate in [0, 1]")

    @classmethod
    def unaudited_candidate(cls, workbook: Path) -> "ExcelInventoryRecord":
        return cls(
            workbook=workbook,
            source_class=MeasurementSourceClass.UNAUDITED_CANDIDATE,
            sheet=None,
            column=None,
            unit=None,
            timestamp=None,
            actual_sampling_interval_seconds=None,
            missing_rate=None,
            physical_meaning=None,
            usable=False,
            reason=(
                "candidate workbook only; sheet, columns, units, timestamps, "
                "sampling interval, missingness, and physical meaning are unaudited"
            ),
        )


def require_train_only(split: str) -> None:
    if str(split).strip().casefold() != "train":
        raise HeldOutDataAccessError(
            "method selection and calibration are Train-only; "
            "Validation/Test payload access is forbidden"
        )


@dataclass(frozen=True)
class RawExcelInventory:
    root: Path
    records: tuple[ExcelInventoryRecord, ...]

    @property
    def workbooks(self) -> tuple[Path, ...]:
        return tuple(record.workbook for record in self.records)

    @property
    def usable_measurements(self) -> tuple[ExcelInventoryRecord, ...]:
        return tuple(record for record in self.records if record.usable)

    @property
    def raw_measurements_available(self) -> bool:
        return bool(self.usable_measurements)


def scan_workbooks(root: Path) -> RawExcelInventory:
    resolved = Path(root).resolve()
    if not resolved.is_dir():
        raise NotADirectoryError(resolved)
    workbooks = tuple(
        sorted(
            path.resolve()
            for path in resolved.rglob("*")
            if path.is_file()
            and path.suffix.casefold() in RawSourcePolicy.ORIGINAL_EXCEL_SUFFIXES
        )
    )
    records = tuple(
        ExcelInventoryRecord.unaudited_candidate(workbook) for workbook in workbooks
    )
    return RawExcelInventory(root=resolved, records=records)
