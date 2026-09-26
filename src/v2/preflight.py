from __future__ import annotations

from collections.abc import Callable
import csv
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
import re
from typing import TypeVar

from .analysis.objective_scale_audit import FORMAL_OBJECTIVE_SCALE_AUDIT_STATUS
from .config import (
    DQN_SWITCH_STEPS,
    DQN_SWITCH_STEPS_EVIDENCE_STATUS,
    FORMAL_TIMESCALE_CONFIGURATION_STATUS,
    N_MPC,
    N_MPC_EVIDENCE_STATUS,
    TAU_LPF_EVIDENCE_STATUS,
    TAU_LPF_SECONDS,
)
from .data.raw_inventory import RawExcelInventory, require_train_only
from .dqn.action_space import ACTION_CATALOG_DIGEST, ACTION_CATALOG_STATUS
from .dqn.state import FORMAL_STATE_STATUS
from .failure_policy import FORMAL_FAILURE_POLICY
from .economics import (
    FORMAL_PRICE_CATALOG,
    SHORE_CHARGING_EFFICIENCY_EVIDENCE,
    SHORE_CHARGING_EFFICIENCY_STATUS,
    SHORE_TARIFF_SOURCE,
)
from .models.battery_degradation import (
    BATTERY_LIFETIME_CONFIGURATION_STATUS,
    BATTERY_LIFETIME_EVIDENCE_BASIS,
    BATTERY_LIFETIME_EVIDENCE_STATUS,
    BATTERY_LIFETIME_THROUGHPUT_FACTOR,
)
from .models.battery_energy import BATTERY_EFFICIENCY_CALIBRATION_STATUS
from .models.fuel_cell_degradation import (
    FC_AGGREGATE_POWER_MAPPING_STATUS,
    FC_LIFETIME_EVIDENCE_CLASS,
    FC_LIFETIME_NORMALIZATION_STATUS,
)
from .models.fuel_cell_efficiency import FC_EFFICIENCY_CALIBRATION_STATUS


DEFAULT_MODE_MANIFEST = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "processed"
    / "operating_dataset_zero_boundary_v2_modes"
    / "metadata"
    / "sample_manifest.csv"
)
DEFAULT_DATASET_QA = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "processed"
    / "operating_dataset_zero_boundary_v2"
    / "metadata"
    / "qa_summary.json"
)
DEFAULT_POWER_MANIFEST = DEFAULT_DATASET_QA.parent / "sample_manifest.csv"
DEFAULT_AIS_MANIFEST = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "processed"
    / "operating_dataset_zero_boundary_v2_ais"
    / "metadata"
    / "sample_manifest.csv"
)
DEFAULT_STATE_AUDIT_ROOT = (
    Path(__file__).resolve().parents[2] / "outputs" / "v2_dqn_state_audit"
)
DEFAULT_STATE_AUDIT_MANIFEST = DEFAULT_STATE_AUDIT_ROOT / "audit_manifest.json"
DEFAULT_OBJECTIVE_AUDIT = (
    Path(__file__).resolve().parents[2]
    / "outputs"
    / "v2_objective_scale_audit"
    / "audit_summary.json"
)
EXPECTED_OBJECTIVE_AUDIT_RESULT_DIGEST = (
    "3a7243751169a118ec62079a4f76c9b6391413777ebbc42f24a117bc9a47605e"
)
DEFAULT_FAILURE_POLICY_AUDIT = (
    Path(__file__).resolve().parents[2]
    / "outputs"
    / "v2_failure_penalty_audit"
    / "audit_summary.json"
)
EXPECTED_FAILURE_POLICY_AUDIT_RESULT_DIGEST = (
    "112c25d23474f09f71013386e91571f37894a66579d8899debde807a170c9801"
)
EXPECTED_FAILURE_REFERENCE_MAX_COST_CNY = 20_779.575664249034


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    """A formal-use gate; VERIFIED does not imply measured evidence."""

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


def _shore_mode_evidence() -> tuple[CalibrationStatus, str]:
    if not DEFAULT_MODE_MANIFEST.is_file():
        return CalibrationStatus.NO_GO, "authenticated shore-mode manifest is missing"
    counts = {"train": 0, "validation": 0, "test": 0}
    try:
        with DEFAULT_MODE_MANIFEST.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = tuple(csv.DictReader(handle))
        for row in rows:
            split = row["split"]
            if split not in counts:
                raise ValueError("unknown split")
            counts[split] += int(row["unresolved_row_count"])
    except (KeyError, TypeError, ValueError, OSError) as exc:
        return CalibrationStatus.NO_GO, f"shore-mode manifest is invalid: {exc}"
    evidence = (
        "authenticated AIS/BMS shore classification with FC retained as diagnostic; "
        "unresolved rows: "
        f"Train={counts['train']}, Validation={counts['validation']}, Test={counts['test']}"
    )
    if counts["train"] or counts["validation"]:
        return CalibrationStatus.UNRESOLVED, evidence
    return CalibrationStatus.VERIFIED, evidence


def _formal_state_audit_evidence() -> tuple[CalibrationStatus, str]:
    if not DEFAULT_STATE_AUDIT_MANIFEST.is_file():
        return CalibrationStatus.NO_GO, "formal S8 audit manifest is missing"
    try:
        payload = json.loads(DEFAULT_STATE_AUDIT_MANIFEST.read_text(encoding="utf-8"))
        from .analysis.train_state_audit import ACTIVE_DATASET_VERSION
        from .dqn.state import FORMAL_STATE_SCHEMA_DIGEST, FORMAL_STATE_SCHEMA_VERSION

        if payload["dataset_version"] != ACTIVE_DATASET_VERSION:
            raise ValueError("dataset version differs")
        if (
            payload["formal_state_schema_version"] != FORMAL_STATE_SCHEMA_VERSION
            or payload["formal_state_schema_digest"] != FORMAL_STATE_SCHEMA_DIGEST
        ):
            raise ValueError("formal S8 schema differs")
        if (
            payload["split"] != "train"
            or payload["train_segment_count"] != 30
            or payload["train_parent_count"] != 30
            or payload["held_out_segment_files_opened"] != 0
            or payload["formal_training_started"] is not False
            or payload["action_catalog_modified"] is not False
            or payload["sample_seconds"] != 30.0
            or payload["tau_lpf_seconds"] != 90.0
        ):
            raise ValueError("formal audit boundary fields differ")

        input_paths = {
            "power": DEFAULT_POWER_MANIFEST,
            "ais": DEFAULT_AIS_MANIFEST,
            "modes": DEFAULT_MODE_MANIFEST,
        }
        expected_inputs = {name: _sha256(path) for name, path in input_paths.items()}
        if payload["input_manifest_sha256"] != expected_inputs:
            raise ValueError("input manifest hashes differ")

        with DEFAULT_POWER_MANIFEST.open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            power_rows = tuple(csv.DictReader(handle))
        train_rows = tuple(row for row in power_rows if row["split"] == "train")
        expected_segments = {row["sample_id"]: row["sha256"] for row in train_rows}
        if (
            len(train_rows) != 30
            or set(payload["segment_ids"]) != set(expected_segments)
            or payload["segment_sha256"] != expected_segments
        ):
            raise ValueError("Train segment identity or hashes differ")

        artifact_hashes = payload["artifact_sha256"]
        if not isinstance(artifact_hashes, dict) or not artifact_hashes:
            raise ValueError("audit artifact hashes are missing")
        for name, expected in artifact_hashes.items():
            if type(name) is not str or Path(name).name != name:
                raise ValueError("audit artifact name is unsafe")
            path = DEFAULT_STATE_AUDIT_ROOT / name
            if not path.is_file() or _sha256(path) != expected:
                raise ValueError(f"audit artifact hash differs: {name}")

        coverage = payload["measured_proxy_coverage"]
        onboard = int(coverage["formal_onboard_row_count"])
        proxy = int(coverage["measured_proxy_row_count"])
        if onboard != 18_448 or proxy != int(payload["eligible_row_count"]):
            raise ValueError("state-audit coverage counts differ")
        if not 0 < proxy <= onboard:
            raise ValueError("state-audit measured proxy coverage is invalid")

        bounds = payload["hidden_life_state_upper_bounds"]
        fc_bound = float(bounds["max_fc_raw_life_fraction_upper_bound"])
        battery_bound = float(bounds["max_battery_raw_life_fraction_upper_bound"])
        if (
            bounds["status"] != "VERIFIED_BELOW_EOL"
            or not 0.0 <= fc_bound < 1.0
            or not 0.0 <= battery_bound < 1.0
        ):
            raise ValueError("hidden cumulative-life clipping is not excluded")
    except (KeyError, TypeError, ValueError, OSError) as exc:
        return CalibrationStatus.NO_GO, f"formal S8 audit is invalid: {exc}"
    return (
        CalibrationStatus.VERIFIED,
        f"{payload['formal_state_schema_version']} authenticated against current "
        f"power/AIS/mode manifests; measured proxy={proxy}/{onboard}; "
        f"conservative episode EOL bounds FC={fc_bound:.6f}, battery={battery_bound:.6f}",
    )


def _objective_scale_evidence() -> tuple[CalibrationStatus, str]:
    if FORMAL_OBJECTIVE_SCALE_AUDIT_STATUS != "GO":
        return CalibrationStatus.NO_GO, "objective-scale model gate is not GO"
    if not DEFAULT_OBJECTIVE_AUDIT.is_file():
        return CalibrationStatus.NO_GO, "current objective-scale audit artifact is missing"
    try:
        payload = json.loads(DEFAULT_OBJECTIVE_AUDIT.read_text(encoding="utf-8"))
        source_manifest = DEFAULT_POWER_MANIFEST.parent / "source_files.csv"
        expected_inputs = {
            "sample_manifest.csv": _sha256(DEFAULT_POWER_MANIFEST),
            "source_files.csv": _sha256(source_manifest),
        }
        with DEFAULT_POWER_MANIFEST.open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            rows = tuple(csv.DictReader(handle))
        train_ids = {row["sample_id"] for row in rows if row["split"] == "train"}
        ratio = float(payload["scale_ratio"])
        if (
            payload["dataset_version"] != "operating_dataset_zero_boundary_v2"
            or payload["input_manifest_sha256"] != expected_inputs
            or set(payload["train_segment_ids"]) != train_ids
            or payload["train_parent_count"] != 30
            or payload["readiness"] != "YES"
            or payload["status"] != "GO"
            or payload["representative_case_count"] != 6
            or not 1.0 <= ratio <= 5.0
            or payload["result_digest"] != EXPECTED_OBJECTIVE_AUDIT_RESULT_DIGEST
        ):
            raise ValueError("objective-scale audit identity or accepted result differs")
    except (KeyError, TypeError, ValueError, OSError) as exc:
        return CalibrationStatus.NO_GO, f"objective-scale audit is invalid: {exc}"
    return (
        CalibrationStatus.VERIFIED,
        f"current 30-Train-segment audit authenticated; 216 solves; "
        f"active-P95 scale ratio={ratio:.6f} (PASS)",
    )


def _failure_policy_evidence(
    audit_path: Path = DEFAULT_FAILURE_POLICY_AUDIT,
) -> tuple[CalibrationStatus, str]:
    if not Path(audit_path).is_file():
        return CalibrationStatus.NO_GO, "Train-only terminal-failure audit is missing"
    try:
        payload = json.loads(Path(audit_path).read_text(encoding="utf-8"))
        supplied_digest = payload["result_digest"]
        digest_body = dict(payload)
        del digest_body["result_digest"]
        calculated_digest = hashlib.sha256(
            json.dumps(
                digest_body,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("ascii")
        ).hexdigest()
        expected_inputs = {
            "power": _sha256(DEFAULT_POWER_MANIFEST),
            "ais": _sha256(DEFAULT_AIS_MANIFEST),
            "modes": _sha256(DEFAULT_MODE_MANIFEST),
        }
        with DEFAULT_POWER_MANIFEST.open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            rows = tuple(csv.DictReader(handle))
        expected_ids = [row["sample_id"] for row in rows if row["split"] == "train"]
        episode_results = payload["episode_results"]
        if type(episode_results) is not list or len(episode_results) != 30:
            raise ValueError("episode result count differs")
        result_ids = [row["sample_id"] for row in episode_results]
        if result_ids != expected_ids:
            raise ValueError("episode result identities differ")
        for row in episode_results:
            raw_cost = row["raw_economic_cost_cny"]
            penalty = row["failure_penalty_score"]
            reward = row["learning_reward"]
            if any(type(value) is not float for value in (raw_cost, penalty, reward)):
                raise TypeError("audit score fields must be exact floats")
            if raw_cost < 0.0 or penalty < 0.0 or reward != -raw_cost - penalty:
                raise ValueError("episode score identity differs")
            failed = row["episode_completed"] is False
            if failed != (row["failure_kind"] == FORMAL_FAILURE_POLICY.failure_kind):
                raise ValueError("episode failure identity differs")
            expected_penalty = FORMAL_FAILURE_POLICY.penalty_score if failed else 0.0
            if penalty != expected_penalty:
                raise ValueError("episode penalty differs")
        if (
            payload["schema_version"] != "v2_terminal_failure_audit_v1"
            or payload["dataset_version"] != "operating_dataset_zero_boundary_v2"
            or payload["split"] != "train"
            or payload["input_manifest_sha256"] != expected_inputs
            or payload["train_segment_ids"] != expected_ids
            or payload["train_segment_count"] != 30
            or payload["reference_action_id"] != "w_8_1_1"
            or payload["action_catalog_digest"] != ACTION_CATALOG_DIGEST
            or payload["failure_penalty_score"] != FORMAL_FAILURE_POLICY.penalty_score
            or payload["failure_kind"] != FORMAL_FAILURE_POLICY.failure_kind
            or payload["evidence_status"] != FORMAL_FAILURE_POLICY.evidence_status
            or payload["calibration_id"] != FORMAL_FAILURE_POLICY.calibration_id
            or payload["completed_episode_count"] != 29
            or payload["failed_episode_count"] != 1
            or payload["failed_episode_ids"] != ["zero_boundary_015"]
            or payload["maximum_completed_raw_economic_cost_cny"]
            != EXPECTED_FAILURE_REFERENCE_MAX_COST_CNY
            or payload["maximum_completed_cost_episode_id"] != "zero_boundary_046"
            or payload["test_payloads_opened"] != 0
            or payload["formal_training_started"] is not False
            or supplied_digest != calculated_digest
            or supplied_digest != EXPECTED_FAILURE_POLICY_AUDIT_RESULT_DIGEST
        ):
            raise ValueError("terminal-failure audit identity or accepted result differs")
    except (KeyError, TypeError, ValueError, OSError) as exc:
        return CalibrationStatus.NO_GO, f"terminal-failure audit is invalid: {exc}"
    return (
        CalibrationStatus.VERIFIED,
        "Train-only fixed-action audit authenticated: 29/30 complete, "
        "zero_boundary_015 physical infeasibility, completed maximum "
        f"{EXPECTED_FAILURE_REFERENCE_MAX_COST_CNY:.9f} CNY; "
        f"penalty={FORMAL_FAILURE_POLICY.penalty_score:.1f}; "
        f"evidence={FORMAL_FAILURE_POLICY.evidence_status}; Test payloads opened=0",
    )


def _curated_dataset_release_evidence(
    state_audit_status: CalibrationStatus,
) -> tuple[CalibrationStatus, str]:
    if not DEFAULT_DATASET_QA.is_file():
        return CalibrationStatus.NO_GO, "curated dataset QA summary is missing"
    try:
        payload = json.loads(DEFAULT_DATASET_QA.read_text(encoding="utf-8"))
        status = payload["formal_training_status"]
        blockers = payload["formal_training_blockers"]
        checks = payload["acceptance_checks"]
        split_counts = payload["split_parent_counts"]
        if type(status) is not str or not isinstance(blockers, list) or any(
            type(item) is not str or not item.strip() for item in blockers
        ):
            raise ValueError("release fields have invalid types")
        if not isinstance(checks, dict) or not checks or not all(
            value is True for value in checks.values()
        ):
            raise ValueError("dataset acceptance checks are not all true")
        if split_counts != {"train": 30, "validation": 8, "test": 5}:
            raise ValueError("dataset split counts differ")
        if payload["artifact_hashes"]["metadata/sample_manifest.csv"] != _sha256(
            DEFAULT_POWER_MANIFEST
        ):
            raise ValueError("dataset QA is not bound to the current power manifest")
    except (KeyError, TypeError, ValueError, OSError) as exc:
        return CalibrationStatus.NO_GO, f"curated dataset QA summary is invalid: {exc}"
    evidence = f"dataset QA status={status}; blockers=" + (
        "; ".join(blockers) if blockers else "none"
    )
    discharged = {
        "DQN state audit must be recomputed after dataset exclusion",
        "integrated formal-training preflight must authenticate the curated manifests",
    }
    if status == "NO-GO" and set(blockers) == discharged:
        if state_audit_status is not CalibrationStatus.VERIFIED:
            return CalibrationStatus.NO_GO, evidence
        return (
            CalibrationStatus.VERIFIED,
            "immutable build-time blockers discharged by authenticated current S8 audit "
            "and integrated power/AIS/mode manifest checks",
        )
    if status != "GO" or blockers:
        return CalibrationStatus.NO_GO, evidence
    return CalibrationStatus.VERIFIED, evidence


def assess_formal_training_preflight() -> FormalTrainingPreflight:
    """Return the repository's complete, non-overridable formal-training gate.

    All evidence checks remain explicit. A unit test or CLI argument cannot
    promote an unresolved item to ``VERIFIED``.
    """

    battery_efficiency_verified = (
        BATTERY_EFFICIENCY_CALIBRATION_STATUS == "SOURCE_BACKED"
    )
    shore_mode_status, shore_mode_evidence = _shore_mode_evidence()
    state_audit_status, state_audit_evidence = _formal_state_audit_evidence()
    objective_status, objective_evidence = _objective_scale_evidence()
    failure_policy_status, failure_policy_evidence = _failure_policy_evidence()
    dataset_release_status, dataset_release_evidence = (
        _curated_dataset_release_evidence(state_audit_status)
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
            CalibrationStatus.VERIFIED
            if FC_LIFETIME_NORMALIZATION_STATUS == "VERIFIED"
            else CalibrationStatus.NO_GO,
            (
                "aggregate-equivalent 70,000 microvolt EOL model; "
                f"{FC_LIFETIME_EVIDENCE_CLASS}"
            ),
        ),
        FormalCalibrationCheck(
            "battery_q_lifetime_normalization",
            CalibrationStatus.VERIFIED
            if BATTERY_LIFETIME_CONFIGURATION_STATUS == "FROZEN"
            and BATTERY_LIFETIME_THROUGHPUT_FACTOR == 15_000.0
            else CalibrationStatus.NO_GO,
            (
                f"configuration={BATTERY_LIFETIME_CONFIGURATION_STATUS}; "
                f"evidence={BATTERY_LIFETIME_EVIDENCE_STATUS}; "
                f"{BATTERY_LIFETIME_EVIDENCE_BASIS}; not vessel measured"
            ),
        ),
        FormalCalibrationCheck(
            "shore_charging_efficiency",
            CalibrationStatus.VERIFIED
            if SHORE_CHARGING_EFFICIENCY_STATUS == "VERIFIED"
            else CalibrationStatus.NO_GO,
            f"0.95; {SHORE_CHARGING_EFFICIENCY_EVIDENCE}",
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
            CalibrationStatus.VERIFIED,
            "30 s frozen nominal control interval; Train clock audit supports cadence",
        ),
        FormalCalibrationCheck(
            "n_mpc",
            CalibrationStatus.VERIFIED
            if FORMAL_TIMESCALE_CONFIGURATION_STATUS == "FROZEN_PROJECT_DESIGN"
            and N_MPC == 5
            else CalibrationStatus.NO_GO,
            (
                f"configuration={FORMAL_TIMESCALE_CONFIGURATION_STATUS}; "
                f"evidence={N_MPC_EVIDENCE_STATUS}; N=5 gives a 150 s prediction "
                "horizon and is not claimed as a literature-proven global optimum"
            ),
        ),
        FormalCalibrationCheck(
            "dqn_switch_steps",
            CalibrationStatus.VERIFIED
            if FORMAL_TIMESCALE_CONFIGURATION_STATUS == "FROZEN_PROJECT_DESIGN"
            and DQN_SWITCH_STEPS == 5
            else CalibrationStatus.NO_GO,
            (
                f"configuration={FORMAL_TIMESCALE_CONFIGURATION_STATUS}; "
                f"evidence={DQN_SWITCH_STEPS_EVIDENCE_STATUS}; M=5 holds one DQN "
                "action across five real rolling MPC solves (150 s), independent of N"
            ),
        ),
        FormalCalibrationCheck(
            "tau_lpf",
            CalibrationStatus.VERIFIED
            if FORMAL_TIMESCALE_CONFIGURATION_STATUS == "FROZEN_PROJECT_DESIGN"
            and TAU_LPF_SECONDS == 90.0
            else CalibrationStatus.NO_GO,
            (
                f"configuration={FORMAL_TIMESCALE_CONFIGURATION_STATUS}; "
                f"evidence={TAU_LPF_EVIDENCE_STATUS}; tau=90 s project control "
                "design with LPF/FC-low-frequency literature structure support; "
                "not vessel-measured and not a unique optimum"
            ),
        ),
        FormalCalibrationCheck(
            "soc_deadband",
            CalibrationStatus.VERIFIED,
            "soft band [0.40,0.60], hard bounds [0.20,0.80], SOC scale 0.60",
        ),
        FormalCalibrationCheck(
            "fc_aggregate_power_mapping",
            CalibrationStatus.VERIFIED
            if FC_AGGREGATE_POWER_MAPPING_STATUS == "FROZEN_PROJECT_MODEL"
            else CalibrationStatus.NO_GO,
            (
                "600 kW aggregate to 100 kW literature reference by equal "
                "normalized load; frozen project model, not vessel-measured"
            ),
        ),
        FormalCalibrationCheck(
            "final_dqn_state",
            CalibrationStatus.VERIFIED
            if FORMAL_STATE_STATUS == "FROZEN_PROJECT_BASELINE"
            and state_audit_status is CalibrationStatus.VERIFIED
            else CalibrationStatus.NO_GO,
            state_audit_evidence,
        ),
        FormalCalibrationCheck(
            "final_action_catalog",
            CalibrationStatus.VERIFIED
            if ACTION_CATALOG_STATUS == "FROZEN_PROJECT_BASELINE"
            else CalibrationStatus.NO_GO,
            "complete canonical 36-action catalog frozen for the first baseline",
        ),
        FormalCalibrationCheck(
            "objective_scale_comparability",
            objective_status,
            objective_evidence,
        ),
        FormalCalibrationCheck(
            "terminal_failure_policy",
            failure_policy_status,
            failure_policy_evidence,
        ),
        FormalCalibrationCheck(
            "curated_dataset_release",
            dataset_release_status,
            dataset_release_evidence,
        ),
        FormalCalibrationCheck(
            "shore_mode_sidecar",
            shore_mode_status,
            shore_mode_evidence,
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
