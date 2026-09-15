from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import re
from typing import TypeVar

from .data.raw_inventory import RawExcelInventory, require_train_only


@dataclass(frozen=True)
class PreflightIssue:
    code: str
    message: str


@dataclass(frozen=True)
class PreflightReport:
    issues: tuple[PreflightIssue, ...]

    @property
    def ready(self) -> bool:
        return not self.issues


class PreflightBlockedError(RuntimeError):
    def __init__(self, report: PreflightReport) -> None:
        self.report = report
        detail = "; ".join(f"{issue.code}: {issue.message}" for issue in report.issues)
        super().__init__(f"v2 preflight failed: {detail}")


class TechnicalSpecificationSourceClass(str, Enum):
    AUTHORITATIVE_VESSEL_SPECIFICATION = "authoritative_vessel_specification"
    PROCESSED_SUMMARY = "processed_summary"
    GENERATED = "generated"
    DIGITIZED = "digitized"


@dataclass(frozen=True)
class TechnicalSpecificationRecord:
    document: Path
    source_class: TechnicalSpecificationSourceClass
    sha256: str
    page_count: int
    source_identifier: str
    source_reference: str

    def __post_init__(self) -> None:
        document = Path(self.document)
        object.__setattr__(self, "document", document)
        if document.suffix.casefold() != ".pdf":
            raise ValueError("technical specification evidence must be a PDF")
        try:
            source_class = TechnicalSpecificationSourceClass(self.source_class)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "technical specification source_class must be explicit and known"
            ) from exc
        object.__setattr__(self, "source_class", source_class)
        if (
            source_class
            is not TechnicalSpecificationSourceClass.AUTHORITATIVE_VESSEL_SPECIFICATION
        ):
            raise ValueError("technical specification evidence must be authoritative")
        if not isinstance(self.sha256, str) or re.fullmatch(
            r"[0-9a-fA-F]{64}", self.sha256
        ) is None:
            raise ValueError("technical specification sha256 must contain 64 hex digits")
        if type(self.page_count) is not int or self.page_count <= 0:
            raise ValueError(
                "technical specification page_count must be a positive integer"
            )
        for field_name in ("source_identifier", "source_reference"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"technical specification {field_name} must be explicit"
                )


def assess_data_preflight(
    *,
    inventory: RawExcelInventory,
    technical_specification: TechnicalSpecificationRecord | None,
) -> PreflightReport:
    if technical_specification is not None and not isinstance(
        technical_specification, TechnicalSpecificationRecord
    ):
        raise TypeError(
            "technical_specification must be a TechnicalSpecificationRecord or None"
        )
    issues: list[PreflightIssue] = []
    if not inventory.raw_measurements_available:
        issues.append(
            PreflightIssue(
                code="missing_usable_raw_measurements",
                message=(
                    "no explicitly audited, authorized, timestamped Excel measurement "
                    "record is usable for the mode-aware v2 dataset"
                ),
            )
        )
    if technical_specification is None:
        issues.append(
            PreflightIssue(
                code="missing_technical_specification",
                message="no authoritative vessel technical specification is registered",
            )
        )
    return PreflightReport(issues=tuple(issues))


PayloadT = TypeVar("PayloadT")


def load_train_payload(
    *,
    split: str,
    inventory: RawExcelInventory,
    technical_specification: TechnicalSpecificationRecord | None,
    payload_loader: Callable[[], PayloadT],
) -> PayloadT:
    """Load only after split and provenance gates have passed.

    Keeping ``payload_loader`` lazy makes the no-held-out-access invariant
    observable and testable.
    """

    require_train_only(split)
    report = assess_data_preflight(
        inventory=inventory,
        technical_specification=technical_specification,
    )
    if not report.ready:
        raise PreflightBlockedError(report)
    return payload_loader()
