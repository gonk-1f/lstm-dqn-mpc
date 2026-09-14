from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
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


def assess_data_preflight(
    *,
    inventory: RawExcelInventory,
    technical_specification_available: bool,
) -> PreflightReport:
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
    if not technical_specification_available:
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
    technical_specification_available: bool,
    payload_loader: Callable[[], PayloadT],
) -> PayloadT:
    """Load only after split and provenance gates have passed.

    Keeping ``payload_loader`` lazy makes the no-held-out-access invariant
    observable and testable.
    """

    require_train_only(split)
    report = assess_data_preflight(
        inventory=inventory,
        technical_specification_available=technical_specification_available,
    )
    if not report.ready:
        raise PreflightBlockedError(report)
    return payload_loader()
