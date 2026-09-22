from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import re
from typing import TypeVar

from .analysis.timescale_audit import FORMAL_TIMESCALE_SELECTION_STATUS
from .analysis.objective_scale_audit import FORMAL_OBJECTIVE_SCALE_AUDIT_STATUS
from .data.raw_inventory import RawExcelInventory, require_train_only
from .dqn.action_space import ACTION_CATALOG_STATUS
from .dqn.state import CANDIDATE_STATE_STATUS
from .economics import FORMAL_PRICE_CATALOG, SHORE_TARIFF_SOURCE
from .models.battery_degradation import BATTERY_LIFETIME_NORMALIZATION_STATUS
from .models.battery_energy import BATTERY_EFFICIENCY_CALIBRATION_STATUS
from .models.fuel_cell_degradation import FC_LIFETIME_NORMALIZATION_STATUS
from .models.fuel_cell_efficiency import FC_EFFICIENCY_CALIBRATION_STATUS


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


class CalibrationStatus(str, Enum):
    VERIFIED = "VERIFIED"
    PROVISIONAL = "PROVISIONAL"
    NO_GO = "NO-GO"
    UNRESOLVED = "UNRESOLVED"


@dataclass(frozen=True)
class FormalCalibrationCheck:
    key: str
    status: CalibrationStatus
    evidence: str


@dataclass(frozen=True)
class FormalTrainingPreflight:
    checks: tuple[FormalCalibrationCheck, ...]

    @property
    def ready(self) -> bool:
        return all(check.status is CalibrationStatus.VERIFIED for check in self.checks)

    @property
    def formal_training(self) -> str:
        return "GO" if self.ready else "NO-GO"

    @property
    def issues(self) -> tuple[PreflightIssue, ...]:
        return tuple(
            PreflightIssue(
                code=f"unfrozen_{check.key}",
                message=f"{check.status.value}: {check.evidence}",
            )
            for check in self.checks
            if check.status is not CalibrationStatus.VERIFIED
        )


class FormalTrainingBlockedError(RuntimeError):
    def __init__(self, report: FormalTrainingPreflight) -> None:
        self.report = report
        detail = "; ".join(
            f"{issue.code}: {issue.message}" for issue in report.issues
        )
        super().__init__(f"FORMAL_TRAINING=NO-GO: {detail}")


def assess_formal_training_preflight() -> FormalTrainingPreflight:
    """Return the repository's complete, non-overridable formal-training gate.

    The source text calls this a twelve-item gate but enumerates thirteen
    distinct calibrations.  All thirteen are kept explicit here.  A unit test
    or CLI argument cannot promote an unresolved item to ``VERIFIED``.
    """

    battery_efficiency_verified = (
        BATTERY_EFFICIENCY_CALIBRATION_STATUS == "SOURCE_BACKED"
    )
    checks = (
        FormalCalibrationCheck(
            "eta_fc_curve",
            CalibrationStatus.VERIFIED
            if FC_EFFICIENCY_CALIBRATION_STATUS == "SOURCE_BACKED"
            else CalibrationStatus.NO_GO,
            "docs/v2_fc_efficiency_model.md; FC_Data.xlsx SHA-256 and A2:B12 are frozen",
        ),
        FormalCalibrationCheck(
            "eta_chg",
            CalibrationStatus.VERIFIED
            if battery_efficiency_verified
            else CalibrationStatus.NO_GO,
            "0.95; DOI 10.11930/j.issn.1004-9649.202507065, Table 3",
        ),
        FormalCalibrationCheck(
            "eta_dis",
            CalibrationStatus.VERIFIED
            if battery_efficiency_verified
            else CalibrationStatus.NO_GO,
            "0.95; DOI 10.11930/j.issn.1004-9649.202507065, Table 3",
        ),
        FormalCalibrationCheck(
            "fc_degradation_normalization",
            CalibrationStatus.NO_GO
            if FC_LIFETIME_NORMALIZATION_STATUS == "NO-GO"
            else CalibrationStatus.UNRESOLVED,
            "single-cell initial-voltage normalization and aggregate mapping are unapproved",
        ),
        FormalCalibrationCheck(
            "battery_q_lifetime_normalization",
            CalibrationStatus.NO_GO
            if BATTERY_LIFETIME_NORMALIZATION_STATUS == "NO-GO"
            else CalibrationStatus.UNRESOLVED,
            "authoritative lifetime-throughput Q_lifetime is absent",
        ),
        FormalCalibrationCheck(
            "shore_electricity_price",
            CalibrationStatus.VERIFIED
            if FORMAL_PRICE_CATALOG.shore_cny_per_kwh == 1.10
            and SHORE_TARIFF_SOURCE.classification == "scenario_not_measured"
            else CalibrationStatus.NO_GO,
            "1.10 CNY/kWh peak-tariff scenario; not a measured wharf tariff",
        ),
        FormalCalibrationCheck(
            "ts_mpc",
            CalibrationStatus.PROVISIONAL,
            "30 s nominal baseline; Train clock audit supports cadence, formal selection remains provisional",
        ),
        FormalCalibrationCheck(
            "n_mpc",
            CalibrationStatus.PROVISIONAL
            if FORMAL_TIMESCALE_SELECTION_STATUS == "NO-GO"
            else CalibrationStatus.UNRESOLVED,
            "N=5 baseline completed the scale audit; no formal Train selection",
        ),
        FormalCalibrationCheck(
            "dqn_switch_steps",
            CalibrationStatus.PROVISIONAL
            if FORMAL_TIMESCALE_SELECTION_STATUS == "NO-GO"
            else CalibrationStatus.UNRESOLVED,
            "M=5 provisional baseline; sensitivity is restricted to {5,10} with no formal selection",
        ),
        FormalCalibrationCheck(
            "tau_lpf",
            CalibrationStatus.UNRESOLVED,
            "90 s is audit-only provisional; formal training calibration remains unfrozen",
        ),
        FormalCalibrationCheck(
            "soc_deadband",
            CalibrationStatus.VERIFIED,
            "soft band [0.40,0.60], hard bounds [0.20,0.80], SOC scale 0.60",
        ),
        FormalCalibrationCheck(
            "final_dqn_state",
            CalibrationStatus.NO_GO
            if CANDIDATE_STATE_STATUS == "NO-GO"
            else CalibrationStatus.UNRESOLVED,
            "candidate state has not passed required Train-only audits",
        ),
        FormalCalibrationCheck(
            "final_action_catalog",
            CalibrationStatus.NO_GO
            if ACTION_CATALOG_STATUS == "NO-GO"
            else CalibrationStatus.UNRESOLVED,
            "36 candidates exist, but screened final K/catalog is unset",
        ),
        FormalCalibrationCheck(
            "objective_scale_comparability",
            CalibrationStatus.VERIFIED
            if FORMAL_OBJECTIVE_SCALE_AUDIT_STATUS == "GO"
            else (
                CalibrationStatus.NO_GO
                if FORMAL_OBJECTIVE_SCALE_AUDIT_STATUS == "NO-GO"
                else CalibrationStatus.UNRESOLVED
            ),
            "accepted Train-only audit: active-P95 scale ratio 1.827863 (PASS)",
        ),
    )
    return FormalTrainingPreflight(checks=checks)


def require_formal_training_ready() -> FormalTrainingPreflight:
    report = assess_formal_training_preflight()
    if not report.ready:
        raise FormalTrainingBlockedError(report)
    return report


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


def load_formal_train_payload(
    *,
    split: str,
    inventory: RawExcelInventory,
    technical_specification: TechnicalSpecificationRecord | None,
    payload_loader: Callable[[], PayloadT],
) -> PayloadT:
    """Fail on every unfrozen formal calibration before any payload access."""

    require_formal_training_ready()
    return load_train_payload(
        split=split,
        inventory=inventory,
        technical_specification=technical_specification,
        payload_loader=payload_loader,
    )
