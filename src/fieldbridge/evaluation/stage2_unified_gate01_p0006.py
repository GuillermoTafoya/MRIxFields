"""Strict development-validation import of the sealed Gate01Private P:0006 graph.

This module does not train, select, or recompute a baseline model.  It verifies the
existing Gate 0.1 archive, materializes no prediction arrays, and exposes the frozen
P:0006 endpoints only to development/model assessment after readiness has been sealed.
This evidence cannot support population or generalization claims.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import tarfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import torch

from fieldbridge.data.photometry_factored_bank_dataset import (
    FactoredLatentStats,
    PhotometryFactoredLatentBankIndex,
)
from fieldbridge.data.photometry_factorization import (
    canonical_tensor_sha256,
    sha256_file,
    sha256_json,
    sha256_text,
    write_json_atomic,
)
from fieldbridge.data.vae_splits import load_vae_splits, vae_splits_fingerprint
from fieldbridge.evaluation.stage2_gate01 import (
    GATE01_CONTRACT_VERSION,
    GATE01_INPUT_CONTRACT_VERSION,
    Gate01Case,
    load_gate01_input_manifest,
)
from fieldbridge.evaluation.stage2_gate01_calibration import (
    GATE01_CALIBRATOR_CONTRACT_VERSION,
    PosthocTargetCalibrator,
)
from fieldbridge.evaluation.stage2_gate01_protocol import (
    GATE01_SCIENTIFIC_MODULES,
    GATE01_PROTOCOL_LOCK_CONTRACT_VERSION,
    Gate01ProtocolLock,
)
from fieldbridge.evaluation.stage2_photometry_baseline import PairedEvaluationCase
from fieldbridge.training.stage2_unified import (
    DEFAULT_UNIFIED_WEIGHTS,
    UNIFIED_A100_GATE_CONTRACT,
    UNIFIED_A100_PEAK_ALLOCATED_LIMIT_BYTES,
    UNIFIED_A100_QUALIFICATION_ONLY_CONTRACT,
    UNIFIED_A100_QUALIFICATION_RECEIPT_FILENAME,
    UNIFIED_ANATOMY_MEMORY_CONTRACT,
    UNIFIED_GENERATOR_ACCUMULATION_CONTRACT,
    UNIFIED_HISTORY_CONTRACT,
    UNIFIED_RESUME_CONTRACT,
    UNIFIED_SELECTION_CONTRACT,
    UNIFIED_SELECTION_RECEIPT_CONTRACT,
    UNIFIED_SELECTION_RULE_SHA256,
    UNIFIED_STAGE2_CONTRACT,
    UNIFIED_TERM_GRADIENT_QUALIFICATION_CONTRACT,
    UNIFIED_TRANSLATOR_CHECKPOINT_CONTRACT,
    UNIFIED_VALIDATION_PLAN_CONTRACT,
    UnifiedStage2Config,
    build_unified_validation_plan,
)
from fieldbridge.training.checkpoints import load_checkpoint

GATE01_P0006_EVALUATION_PROTOCOL_V2 = (
    "stage2-unified-gate01-p0006-development-validation-evaluation-only-protocol-v2"
)
GATE01_P0006_EVALUATION_PROTOCOL_V3 = (
    "stage2-unified-gate01-p0006-development-validation-evaluation-only-protocol-v3"
)
GATE01_P0006_EVALUATION_PROTOCOL = (
    "stage2-unified-gate01-p0006-development-validation-evaluation-only-protocol-v4"
)
SUPPORTED_GATE01_P0006_EVALUATION_PROTOCOLS = frozenset(
    {
        GATE01_P0006_EVALUATION_PROTOCOL_V2,
        GATE01_P0006_EVALUATION_PROTOCOL_V3,
        GATE01_P0006_EVALUATION_PROTOCOL,
    }
)
GATE01_ARCHIVE_LAYOUT_MODERN_FLAT_V1 = "gate01-archive-modern-flat-csv-v1"
GATE01_ARCHIVE_LAYOUT_REVIEWED_LEGACY_JSON_V1 = (
    "gate01-archive-reviewed-legacy-parent-json-v1"
)
GATE01_ARCHIVE_LAYOUT_REVIEWED_LEGACY_JSON_V2 = (
    "gate01-archive-reviewed-legacy-path-aware-json-v2"
)
GATE01_ARCHIVE_PREFLIGHT_CONTRACT = "gate01-p0006-metadata-preflight-v2"
STAGE2_COMPLETED_PILOT_REUSE_CONTRACT = "stage2-completed-pilot-reuse-verification-v1"
STAGE2_BANK_TAR_RESTORE_CONTRACT = "stage2-reviewed-bank-tar-restore-v1"
STAGE2_LOCAL_ARCHIVE_COPY_CONTRACT = "stage2-reviewed-bank-local-copy-v1"
STAGE2_LOCAL_DISK_PREFLIGHT_CONTRACT = "stage2-recovery-local-disk-preflight-v1"
STAGE2_BANK_MANIFEST_FILENAME = "photometry_factored_latent_bank_manifest.json"
STAGE2_RECOVERY_DRIVE_LAYOUT_CONTRACT = "stage2-recovery-drive-layout-v1"
STAGE2_RECOVERY_OUTPUT_ROOT_NAME = "UnifiedStage2_1ca2b4a_01"
STAGE2_RECOVERY_V7_ROOT_NAME = "stage2_unified_v7"
STAGE2_RECOVERY_BANK_NAMESPACE_NAME = "bank_8081ce89a0ea"
STAGE2_RECOVERY_TRAINING_NAMESPACE_NAME = "implementation_82633d66e5ea"
STAGE2_RECOVERY_BANK_TAR_NAME = "photometry_factored_latent_bank_v2.tar"
STAGE2_RECOVERY_UNRECEIPTED_BANK_DIR_NAME = "photometry_factored_latent_bank_v2"
STAGE2_RECOVERY_PAIR_FEASIBILITY_NAME = (
    "stage2_retrospective_pair_feasibility_v2.json"
)
STAGE2_RECOVERY_SELECTION_RECEIPT_NAME = (
    "stage2_unified_full_selection_step000000200.json"
)
STAGE2_PROGRESS_MAX_INTERVAL_SECONDS = 30.0
STAGE2_PROGRESS_MAX_INTERVAL_BYTES = 512 * 1024**2
STAGE2_TREE_PROGRESS_MAX_INTERVAL_FILES = 256
STAGE2_LOCAL_DISK_RESERVE_BYTES = 2 * 1024**3
P0006_COUNT_PROGRESS_ENV = "FIELDBRIDGE_STAGE2_P0006_COUNT_PROGRESS"
P0006_COUNT_PROGRESS_VALUE = "count-only-json-v1"
TRAINING_EVIDENCE_COMMIT = "82633d66e5ea47f96b149ea22cc192fcf4526f06"
REVIEWED_GATE01_RESULT_FILE_SHA256 = (
    "454747cd3e4b1376855915244a7c40fe281b758150e86f584fbea96f94d531f5"
)
EXPECTED_STAGE2_SELECTION_RECEIPT_FILE_SHA256 = (
    "c8d73fec48815224fcb87333dfd093c15738cc41dce89c4fb8ccf2cd874ef828"
)
EXPECTED_STAGE2_VALIDATION_PLAN_SHA256 = (
    "3afca2bab6a440529f88e7c8d9a9294fed9ecbf07eea1e308ed0910e2ba16421"
)
EXPECTED_STAGE2_SELECTION_RULE_SHA256 = (
    "fd15be634185a29d5ddedec3f2d7a24527bf5e59a49731f101f62cafcf1b06d6"
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT_RE = re.compile(r"[0-9a-f]{40}")

_GATE01_LEGACY_ROOT_MARKER = "Gate01Private_8012a3f"
_GATE01_LEGACY_SPLIT_STORED_PATH = (
    "/content/drive/MyDrive/MRIxFields2026/split_v3.json"
)
_GATE01_LEGACY_SPLIT_RELATIVE_PATH = "archive/split_v3.json"
_GATE01_LEGACY_SPLIT_SHA256 = (
    "f6a19d7a31c4c3bb73edd92088ea078192e88ee4b276309bad81c548ab7f94d5"
)
_GATE01_LEGACY_SPLIT_SIZE_BYTES = 798444
_GATE01_LEGACY_REQUIRED_RELATIVE_PATHS = frozenset(
    {
        "frozen-scientific-resplit.json",
        _GATE01_LEGACY_SPLIT_RELATIVE_PATH,
        "gate01-target-calibrator.json",
        "gate01-prospective-selection.json",
        "gate01-protocol-spec.json",
        "gate01-protocol-lock.json",
        "gate01-producer-spec.json",
        "producer-state/producer-state.json",
        "producer-output/private-build-plan.json",
        "gate01-private-manifest.json",
        "gate01-private-build-state.json",
        "gate01-results.json",
        "gate01-report.md",
        "gate01-result-contract.json",
    }
)
_GATE01_LEGACY_SUPPLEMENTAL_SPECS = {
    "colab-operational-source-split.json": (
        "972e497e2d29755e928414a4aa51f906951674ec0a950b0e9ac73881fffd0c54",
        799986,
        "canonical_vae_split_with_gate_bank_membership_linkage",
    ),
    "gate01-reviewed-module-sha256-8012a3f.json": (
        "ea5f40b580cbba26766ee60ce243d466ab93d32b1856125c067eace9a7d1ed36",
        7634,
        "reviewed_scientific_module_sha256_map",
    ),
}
_GATE01_REVIEWED_MODULE_COMPARISON_KEYS = frozenset(
    {
        "changed_modules",
        "evaluation_git_commit",
        "evaluation_module_sha256",
        "previous_evaluation_git_commit",
        "previous_evaluation_module_sha256",
    }
)
_GATE01_REVIEWED_CHANGED_MODULES = (
    "src/fieldbridge/models/translators/flow_transport.py",
)
_GATE01_REVIEWED_MODULE_COUNT = 31

_GATE01_REQUIRED_COMMON = frozenset(
    {
        "gate01-private-manifest.json",
        "gate01-private-build-state.json",
        "gate01-results.json",
        "gate01-result-contract.json",
        "gate01-protocol-lock.json",
        "gate01-protocol-spec.json",
        "gate01-target-calibrator.json",
        "gate01-producer-spec.json",
        "gate01-prospective-selection.json",
        "frozen-scientific-resplit.json",
    }
)
_GATE01_REQUIRED_BY_LAYOUT = {
    GATE01_ARCHIVE_LAYOUT_MODERN_FLAT_V1: _GATE01_REQUIRED_COMMON
    | {
        "original-split-v3.json",
        "producer-state.json",
        "private-build-plan.json",
        "gate01-report.md",
    },
    GATE01_ARCHIVE_LAYOUT_REVIEWED_LEGACY_JSON_V2: (
        _GATE01_LEGACY_REQUIRED_RELATIVE_PATHS
    ),
}
P0006_SUBJECT_GROUP = "P:0006"
P0006_IDENTITY_SHA256 = sha256_text(P0006_SUBJECT_GROUP)
P0006_DEVELOPMENT_VALIDATION_DATA_ROLE = (
    "development_validation_P0006_evaluation_only"
)
P0006_EVIDENCE_LIMITATION = (
    "development/model assessment only; cannot support population or generalization claims"
)
P0009_CONFIRMATION_STATUS = "frozen_and_unused_for_possible_later_confirmation"
FORBIDDEN_OTHER_TRAVELLER_HASHES = {
    sha256_text("P:0007"),
    sha256_text("P:0009"),
}


@dataclass(frozen=True, slots=True)
class VerifiedGate01InventoryEntry:
    """One inventory row verified at an exact logical-root-relative path."""

    relative_path: str
    sha256: str
    size_bytes: int
    resolution_rule: str
    stored_path_label_sha256: str

    @property
    def basename(self) -> str:
        return PurePosixPath(self.relative_path).name

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "basename": self.basename,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "resolution_rule": self.resolution_rule,
            "stored_path_label_sha256": self.stored_path_label_sha256,
        }


@dataclass(frozen=True, slots=True)
class VerifiedGate01SupplementalDependency:
    """Separately pinned metadata dependency outside the legacy inventory."""

    relative_path: str
    sha256: str
    size_bytes: int
    semantic_contract: str
    semantic_identity_sha256: str
    semantic_entry_count: int

    @property
    def basename(self) -> str:
        return PurePosixPath(self.relative_path).name

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "basename": self.basename,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "semantic_contract": self.semantic_contract,
            "semantic_identity_sha256": self.semantic_identity_sha256,
            "semantic_entry_count": self.semantic_entry_count,
        }


@dataclass(frozen=True, slots=True)
class ReviewedGate01ModuleComparisonEvidence:
    """Strict current-versus-previous module evidence from the reviewed supplement."""

    evaluation_git_commit: str
    evaluation_module_sha256: tuple[tuple[str, str], ...]
    previous_evaluation_git_commit: str
    previous_evaluation_module_sha256: tuple[tuple[str, str], ...]
    changed_modules: tuple[str, ...]

    @property
    def current_module_hashes(self) -> dict[str, str]:
        return dict(self.evaluation_module_sha256)

    @property
    def previous_module_hashes(self) -> dict[str, str]:
        return dict(self.previous_evaluation_module_sha256)


@dataclass(frozen=True, slots=True)
class Gate01ArchiveLayout:
    """Resolved, fully verified Gate 0.1 logical archive contract."""

    logical_root: Path
    inventory_path: Path
    layout_contract: str
    inventory_format: Literal["csv", "reviewed_legacy_json"]
    entries: tuple[VerifiedGate01InventoryEntry, ...]
    supplemental_dependencies: tuple[VerifiedGate01SupplementalDependency, ...]
    inventory_file_sha256: str
    normalized_inventory_sha256: str

    def path_for(self, relative_path: str) -> Path:
        matches = [
            entry for entry in self.entries if entry.relative_path == relative_path
        ]
        if len(matches) != 1:
            raise ValueError(
                "Gate 0.1 inventory requires exactly one verified relative path "
                f"{relative_path!r}; found {len(matches)}."
            )
        return _verified_relative_child(self.logical_root, relative_path)

    def supplemental_path_for(self, relative_path: str) -> Path:
        matches = [
            dependency
            for dependency in self.supplemental_dependencies
            if dependency.relative_path == relative_path
        ]
        if len(matches) != 1:
            raise ValueError(
                "Gate 0.1 requires exactly one verified supplemental dependency "
                f"{relative_path!r}; found {len(matches)}."
            )
        return _verified_relative_child(self.logical_root, relative_path)

    def file_with_sha256(self, digest: str) -> Path:
        matches = [entry for entry in self.entries if entry.sha256 == digest]
        if len(matches) != 1:
            raise ValueError(
                "Gate 0.1 inventory must identify exactly one file with the expected "
                "result SHA-256."
            )
        return _verified_relative_child(self.logical_root, matches[0].relative_path)

    def provenance(self) -> dict[str, Any]:
        return {
            "logical_root": str(self.logical_root),
            "logical_root_path_identity_sha256": sha256_text(str(self.logical_root)),
            "layout_contract": self.layout_contract,
            "inventory_format": self.inventory_format,
            "inventory_path": str(self.inventory_path),
            "inventory_file_sha256": self.inventory_file_sha256,
            "normalized_inventory_entry_count": len(self.entries),
            "normalized_inventory_sha256": self.normalized_inventory_sha256,
            "verified_entries": [entry.to_dict() for entry in self.entries],
            "supplemental_dependencies": [
                dependency.to_dict()
                for dependency in self.supplemental_dependencies
            ],
            "stored_inventory_paths_trusted_for_file_access": False,
            "file_access_derivation": (
                "verified_exact_root_relative_path_from_layout_resolution_rule"
            ),
        }


@dataclass(frozen=True, slots=True)
class VerifiedStage2BankRestore:
    """Identity of a safely restored immutable Stage-2 bank tar."""

    archive_path: Path
    archive_file_sha256: str
    bank_root: Path
    tree_sha256: str
    file_count: int
    total_bytes: int
    bank_artifact_sha256: str
    restored_from_tar: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": STAGE2_BANK_TAR_RESTORE_CONTRACT,
            "archive_path": str(self.archive_path),
            "archive_file_sha256": self.archive_file_sha256,
            "bank_root": str(self.bank_root),
            "tree_sha256": self.tree_sha256,
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
            "bank_artifact_sha256": self.bank_artifact_sha256,
            "restored_from_tar": self.restored_from_tar,
            "safe_extraction": True,
            "links_or_special_members_accepted": False,
        }


@dataclass(frozen=True, slots=True)
class Stage2LocalDiskCapacityPreflight:
    """Count-only local capacity proof before copying or extracting the bank."""

    archive_bytes: int
    extracted_bytes: int
    reserve_bytes: int
    additional_archive_bytes: int
    additional_extracted_bytes: int
    required_available_bytes: int
    free_bytes: int
    local_archive_already_present: bool
    local_bank_already_present: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": STAGE2_LOCAL_DISK_PREFLIGHT_CONTRACT,
            "archive_bytes": self.archive_bytes,
            "extracted_bytes": self.extracted_bytes,
            "reserve_bytes": self.reserve_bytes,
            "additional_archive_bytes": self.additional_archive_bytes,
            "additional_extracted_bytes": self.additional_extracted_bytes,
            "required_available_bytes": self.required_available_bytes,
            "free_bytes": self.free_bytes,
            "local_archive_already_present": self.local_archive_already_present,
            "local_bank_already_present": self.local_bank_already_present,
            "status": "pass",
        }


@dataclass(frozen=True, slots=True)
class VerifiedStage2LocalArchive:
    """A reviewed Drive tar copied and SHA-verified in one streaming pass."""

    source_archive_path: Path
    local_archive_path: Path
    archive_file_sha256: str
    archive_bytes: int
    local_mtime_ns: int
    copied_from_source: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": STAGE2_LOCAL_ARCHIVE_COPY_CONTRACT,
            "source_archive_path": str(self.source_archive_path),
            "local_archive_path": str(self.local_archive_path),
            "archive_file_sha256": self.archive_file_sha256,
            "archive_bytes": self.archive_bytes,
            "copied_from_source": self.copied_from_source,
            "copy_and_sha256_same_pass": self.copied_from_source,
            "exact_local_resume_verified": not self.copied_from_source,
        }


@dataclass(frozen=True, slots=True)
class Stage2RecoveryDriveLayout:
    """Exact reviewed Drive topology for the completed Stage-2 v7 evidence."""

    drive_root: Path
    output_root: Path
    stage2_v7_root: Path
    bank_namespace: Path
    training_namespace: Path
    bank_archive: Path
    pair_feasibility: Path
    selection_receipt: Path
    ignored_empty_unreceipted_bank_directory: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": STAGE2_RECOVERY_DRIVE_LAYOUT_CONTRACT,
            "drive_root": str(self.drive_root),
            "output_root": str(self.output_root),
            "stage2_v7_root": str(self.stage2_v7_root),
            "bank_namespace": str(self.bank_namespace),
            "training_namespace": str(self.training_namespace),
            "bank_archive": str(self.bank_archive),
            "pair_feasibility": str(self.pair_feasibility),
            "selection_receipt": str(self.selection_receipt),
            "ignored_empty_unreceipted_bank_directory": (
                self.ignored_empty_unreceipted_bank_directory
            ),
        }


def resolve_stage2_recovery_drive_layout(
    drive_root: str | Path,
) -> Stage2RecoveryDriveLayout:
    """Resolve only the exact reviewed output, evidence, tar, and receipt topology."""

    root = _stage2_recovery_directory(Path(drive_root), label="Drive root")
    output_root = _stage2_recovery_direct_directory(
        root, STAGE2_RECOVERY_OUTPUT_ROOT_NAME, label="output root"
    )
    stage2_v7_root = _stage2_recovery_direct_directory(
        output_root, STAGE2_RECOVERY_V7_ROOT_NAME, label="Stage-2 v7 root"
    )
    bank_namespace = _stage2_recovery_direct_directory(
        stage2_v7_root,
        STAGE2_RECOVERY_BANK_NAMESPACE_NAME,
        label="bank namespace",
    )
    training_namespace = _stage2_recovery_direct_directory(
        bank_namespace,
        STAGE2_RECOVERY_TRAINING_NAMESPACE_NAME,
        label="training namespace",
    )
    bank_archive = _stage2_recovery_direct_file(
        output_root, STAGE2_RECOVERY_BANK_TAR_NAME, label="reviewed bank tar"
    )
    pair_feasibility = _stage2_recovery_direct_file(
        output_root,
        STAGE2_RECOVERY_PAIR_FEASIBILITY_NAME,
        label="pair-feasibility receipt",
    )
    pilot_root = _stage2_recovery_direct_directory(
        training_namespace,
        "unified_full_objective_pilot_200",
        label="step-200 pilot root",
    )
    attempts_root = _stage2_recovery_direct_directory(
        pilot_root, "scientific_attempts", label="scientific attempts root"
    )
    attempt_root = _stage2_recovery_direct_directory(
        attempts_root, "attempt-0001", label="completed scientific attempt"
    )
    checkpoint_root = _stage2_recovery_direct_directory(
        attempt_root, "checkpoints", label="completed checkpoint root"
    )
    selection_receipt = _stage2_recovery_direct_file(
        checkpoint_root,
        STAGE2_RECOVERY_SELECTION_RECEIPT_NAME,
        label="step-200 selection receipt",
    )

    unreceipted = output_root / STAGE2_RECOVERY_UNRECEIPTED_BANK_DIR_NAME
    ignored_empty = False
    if unreceipted.exists() or unreceipted.is_symlink():
        empty_root = _stage2_recovery_directory(
            unreceipted, label="unreceipted bank directory"
        )
        if any(empty_root.iterdir()):
            raise ValueError(
                "The unreceipted bank directory is not empty; refusing archive ambiguity."
            )
        ignored_empty = True
    return Stage2RecoveryDriveLayout(
        drive_root=root,
        output_root=output_root,
        stage2_v7_root=stage2_v7_root,
        bank_namespace=bank_namespace,
        training_namespace=training_namespace,
        bank_archive=bank_archive,
        pair_feasibility=pair_feasibility,
        selection_receipt=selection_receipt,
        ignored_empty_unreceipted_bank_directory=ignored_empty,
    )


def _stage2_recovery_directory(path: Path, *, label: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"Stage-2 recovery {label} may not be a symlink.")
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Stage-2 recovery {label} is missing: {path}") from exc
    if not resolved.is_dir():
        raise ValueError(f"Stage-2 recovery {label} is not a directory: {path}")
    return resolved


def _stage2_recovery_direct_directory(
    parent: Path, name: str, *, label: str
) -> Path:
    candidate = _stage2_recovery_directory(parent / name, label=label)
    if candidate.parent != parent:
        raise ValueError(f"Stage-2 recovery {label} is not the expected direct child.")
    return candidate


def _stage2_recovery_direct_file(parent: Path, name: str, *, label: str) -> Path:
    raw = parent / name
    if raw.is_symlink():
        raise ValueError(f"Stage-2 recovery {label} may not be a symlink.")
    try:
        candidate = raw.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Stage-2 recovery {label} is missing: {raw}") from exc
    if candidate.parent != parent or not candidate.is_file():
        raise ValueError(
            f"Stage-2 recovery {label} is not the expected direct regular file."
        )
    return candidate


def preflight_stage2_local_disk_capacity(
    source_archive_path: str | Path,
    scratch_root: str | Path,
    *,
    local_archive_path: str | Path,
    local_bank_root: str | Path,
    expected_extracted_bytes: int,
    reserve_bytes: int = STAGE2_LOCAL_DISK_RESERVE_BYTES,
) -> Stage2LocalDiskCapacityPreflight:
    """Fail before large I/O unless local scratch can hold the tar and bank."""

    if type(expected_extracted_bytes) is not int or expected_extracted_bytes <= 0:
        raise ValueError("Expected extracted byte count must be a positive integer.")
    if type(reserve_bytes) is not int or reserve_bytes < 0:
        raise ValueError("Local disk reserve must be a nonnegative integer.")
    source = _validated_stage2_archive_file(
        source_archive_path, label="reviewed Drive bank archive"
    )
    archive_bytes = source.stat().st_size
    if archive_bytes <= 0:
        raise ValueError("Reviewed Stage-2 bank archive is empty.")

    local_archive = Path(local_archive_path)
    archive_present = local_archive.exists() or local_archive.is_symlink()
    if archive_present:
        verified_local = _validated_stage2_archive_file(
            local_archive, label="existing local bank archive"
        )
        if verified_local.stat().st_size != archive_bytes:
            raise ValueError("Existing local bank archive byte count differs from Drive.")

    local_bank = Path(local_bank_root)
    bank_present = local_bank.exists() or local_bank.is_symlink()
    if bank_present and (local_bank.is_symlink() or not local_bank.is_dir()):
        raise ValueError("Existing local bank root must be a nonsymlink directory.")

    probe = Path(scratch_root)
    while not probe.exists():
        if probe.parent == probe:
            raise FileNotFoundError("No existing parent is available for local scratch.")
        probe = probe.parent
    if probe.is_symlink():
        raise ValueError("Local scratch capacity probe may not be a symlink.")
    usage = shutil.disk_usage(probe.resolve(strict=True))
    additional_archive_bytes = 0 if archive_present else archive_bytes
    additional_extracted_bytes = 0 if bank_present else expected_extracted_bytes
    required = additional_archive_bytes + additional_extracted_bytes + reserve_bytes
    if usage.free < required:
        raise OSError(
            "Insufficient local disk for reviewed Stage-2 recovery: "
            f"required_available_bytes={required}, free_bytes={usage.free}."
        )
    return Stage2LocalDiskCapacityPreflight(
        archive_bytes=archive_bytes,
        extracted_bytes=expected_extracted_bytes,
        reserve_bytes=reserve_bytes,
        additional_archive_bytes=additional_archive_bytes,
        additional_extracted_bytes=additional_extracted_bytes,
        required_available_bytes=required,
        free_bytes=usage.free,
        local_archive_already_present=archive_present,
        local_bank_already_present=bank_present,
    )


def copy_verified_stage2_bank_tar_to_local(
    source_archive_path: str | Path,
    local_archive_path: str | Path,
    *,
    expected_archive_sha256: str,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    progress_interval_seconds: float = STAGE2_PROGRESS_MAX_INTERVAL_SECONDS,
    progress_interval_bytes: int = STAGE2_PROGRESS_MAX_INTERVAL_BYTES,
) -> VerifiedStage2LocalArchive:
    """Copy Drive tar to local scratch while hashing the single read stream."""

    if _SHA256_RE.fullmatch(expected_archive_sha256) is None:
        raise ValueError("Expected Stage-2 archive identity must be lowercase SHA-256.")
    _validate_progress_intervals(
        seconds=progress_interval_seconds,
        byte_interval=progress_interval_bytes,
    )
    source = _validated_stage2_archive_file(
        source_archive_path, label="reviewed Drive bank archive"
    )
    source_stat = source.stat()
    if source_stat.st_size <= 0:
        raise ValueError("Reviewed Stage-2 bank archive is empty.")

    raw_destination = Path(local_archive_path)
    if raw_destination.is_symlink():
        raise ValueError("Local Stage-2 bank archive may not be a symlink.")
    raw_destination.parent.mkdir(parents=True, exist_ok=True)
    destination_parent = raw_destination.parent.resolve(strict=True)
    destination = destination_parent / raw_destination.name
    if source == destination:
        raise ValueError("Drive source and local Stage-2 archive must differ.")

    if destination.exists():
        existing = _validated_stage2_archive_file(
            destination, label="existing local bank archive"
        )
        if existing.stat().st_size != source_stat.st_size:
            raise ValueError("Existing local bank archive byte count differs from Drive.")
        digest, copied_bytes = _stream_stage2_archive(
            existing,
            destination_handle=None,
            mode="verify_existing",
            progress_callback=progress_callback,
            progress_interval_seconds=progress_interval_seconds,
            progress_interval_bytes=progress_interval_bytes,
        )
        if digest != expected_archive_sha256:
            raise ValueError("Existing local Stage-2 bank tar SHA-256 mismatch.")
        final_stat = existing.stat()
        return VerifiedStage2LocalArchive(
            source_archive_path=source,
            local_archive_path=existing,
            archive_file_sha256=digest,
            archive_bytes=copied_bytes,
            local_mtime_ns=final_stat.st_mtime_ns,
            copied_from_source=False,
        )

    attempt = 1
    while True:
        staging = destination_parent / (
            f"{destination.name}.copy-attempt-{attempt:04d}.partial"
        )
        if not staging.exists() and not staging.is_symlink():
            break
        attempt += 1
    with staging.open("xb") as destination_handle:
        digest, copied_bytes = _stream_stage2_archive(
            source,
            destination_handle=destination_handle,
            mode="copy",
            progress_callback=progress_callback,
            progress_interval_seconds=progress_interval_seconds,
            progress_interval_bytes=progress_interval_bytes,
        )
        destination_handle.flush()
        os.fsync(destination_handle.fileno())
    source_after = source.stat()
    if (
        source_after.st_size != source_stat.st_size
        or source_after.st_mtime_ns != source_stat.st_mtime_ns
        or copied_bytes != source_stat.st_size
    ):
        raise ValueError("Reviewed Drive bank archive changed during local copy.")
    if digest != expected_archive_sha256:
        raise ValueError("Reviewed Stage-2 bank tar SHA-256 mismatch during local copy.")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Concurrent Stage-2 archive copy created the final path.")
    try:
        os.link(staging, destination)
    except FileExistsError as exc:
        raise FileExistsError(
            "Concurrent Stage-2 archive copy created the final path."
        ) from exc
    staging.unlink()
    final_stat = destination.stat()
    return VerifiedStage2LocalArchive(
        source_archive_path=source,
        local_archive_path=destination,
        archive_file_sha256=digest,
        archive_bytes=copied_bytes,
        local_mtime_ns=final_stat.st_mtime_ns,
        copied_from_source=True,
    )


def _validated_stage2_archive_file(path: str | Path, *, label: str) -> Path:
    raw = Path(path)
    if raw.is_symlink():
        raise ValueError(f"{label} may not be a symlink.")
    try:
        resolved = raw.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{label} is missing: {raw}") from exc
    if not resolved.is_file() or resolved.suffix != ".tar":
        raise ValueError(f"{label} must be a regular .tar file.")
    return resolved


def _validate_progress_intervals(
    *,
    seconds: float,
    byte_interval: int | None = None,
    file_interval: int | None = None,
) -> None:
    if (
        isinstance(seconds, bool)
        or not isinstance(seconds, (int, float))
        or not 0 < float(seconds) <= STAGE2_PROGRESS_MAX_INTERVAL_SECONDS
    ):
        raise ValueError("Progress seconds must be greater than zero and at most 30.")
    if byte_interval is not None and (
        type(byte_interval) is not int
        or not 0 < byte_interval <= STAGE2_PROGRESS_MAX_INTERVAL_BYTES
    ):
        raise ValueError("Progress byte interval must be at most 512 MiB.")
    if file_interval is not None and (
        type(file_interval) is not int
        or not 0 < file_interval <= STAGE2_TREE_PROGRESS_MAX_INTERVAL_FILES
    ):
        raise ValueError("Tree progress file interval must be at most 256 files.")


def _stream_stage2_archive(
    source: Path,
    *,
    destination_handle: Any | None,
    mode: Literal["copy", "verify_existing"],
    progress_callback: Callable[[dict[str, Any]], None] | None,
    progress_interval_seconds: float,
    progress_interval_bytes: int,
) -> tuple[str, int]:
    total_bytes = source.stat().st_size
    processed = 0
    digest = hashlib.sha256()
    start = time.monotonic()
    last_time = start
    last_bytes = 0
    _emit_progress(
        progress_callback,
        {
            "stage": "bank_archive_local_copy",
            "status": "start",
            "mode": mode,
            "bytes_processed": 0,
            "total_bytes": total_bytes,
        },
    )
    with source.open("rb") as source_handle:
        while True:
            chunk = source_handle.read(8 * 1024**2)
            if not chunk:
                break
            if destination_handle is not None:
                destination_handle.write(chunk)
            digest.update(chunk)
            processed += len(chunk)
            now = time.monotonic()
            if (
                now - last_time >= progress_interval_seconds
                or processed - last_bytes >= progress_interval_bytes
            ):
                _emit_progress(
                    progress_callback,
                    {
                        "stage": "bank_archive_local_copy",
                        "status": "periodic",
                        "mode": mode,
                        "bytes_processed": processed,
                        "total_bytes": total_bytes,
                    },
                )
                last_time = now
                last_bytes = processed
    _emit_progress(
        progress_callback,
        {
            "stage": "bank_archive_local_copy",
            "status": "end",
            "mode": mode,
            "bytes_processed": processed,
            "total_bytes": total_bytes,
        },
    )
    return digest.hexdigest(), processed


def _emit_progress(
    callback: Callable[[dict[str, Any]], None] | None,
    payload: dict[str, Any],
) -> None:
    if callback is not None:
        callback(dict(payload))


def stage2_bank_tree_identity(
    root: str | Path,
    *,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    progress_interval_seconds: float = STAGE2_PROGRESS_MAX_INTERVAL_SECONDS,
    progress_interval_files: int = STAGE2_TREE_PROGRESS_MAX_INTERVAL_FILES,
) -> dict[str, Any]:
    """Hash the complete bank tree using the reviewed v7 identity contract."""

    _validate_progress_intervals(
        seconds=progress_interval_seconds,
        file_interval=progress_interval_files,
    )
    raw_root = Path(root)
    if raw_root.is_symlink():
        raise ValueError("Stage-2 bank root may not be a symlink.")
    resolved_root = raw_root.resolve()
    if not resolved_root.is_dir():
        raise FileNotFoundError(f"Stage-2 bank root is missing: {resolved_root}")
    file_paths: list[Path] = []
    for path in sorted(resolved_root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Stage-2 bank tree contains a symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"Stage-2 bank tree contains a special entry: {path}")
        file_paths.append(path)
    if not file_paths:
        raise ValueError("Stage-2 bank tree is empty.")

    rows: list[dict[str, Any]] = []
    processed_bytes = 0
    start = time.monotonic()
    last_time = start
    last_files = 0
    _emit_progress(
        progress_callback,
        {
            "stage": "bank_tree_verification",
            "status": "start",
            "files_processed": 0,
            "total_files": len(file_paths),
            "bytes_processed": 0,
        },
    )
    for file_count, path in enumerate(file_paths, start=1):
        resolved = path.resolve(strict=True)
        try:
            relative = resolved.relative_to(resolved_root)
        except ValueError as exc:
            raise ValueError("Stage-2 bank file escapes the restored root.") from exc
        size = resolved.stat().st_size
        rows.append(
            {
                "path": relative.as_posix(),
                "bytes": size,
                "sha256": sha256_file(resolved),
            }
        )
        processed_bytes += size
        now = time.monotonic()
        if (
            now - last_time >= progress_interval_seconds
            or file_count - last_files >= progress_interval_files
        ):
            _emit_progress(
                progress_callback,
                {
                    "stage": "bank_tree_verification",
                    "status": "periodic",
                    "files_processed": file_count,
                    "total_files": len(file_paths),
                    "bytes_processed": processed_bytes,
                },
            )
            last_time = now
            last_files = file_count
    _emit_progress(
        progress_callback,
        {
            "stage": "bank_tree_verification",
            "status": "end",
            "files_processed": len(rows),
            "total_files": len(file_paths),
            "bytes_processed": processed_bytes,
        },
    )
    return {
        "file_count": len(rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "tree_sha256": sha256_json(rows),
    }


def restore_verified_stage2_bank_tar(
    archive_path: str | Path | VerifiedStage2LocalArchive,
    destination: str | Path,
    *,
    expected_archive_sha256: str,
    expected_tree_sha256: str,
    expected_bank_artifact_sha256: str,
    expected_file_count: int,
    expected_total_bytes: int,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    progress_interval_seconds: float = STAGE2_PROGRESS_MAX_INTERVAL_SECONDS,
    progress_interval_bytes: int = STAGE2_PROGRESS_MAX_INTERVAL_BYTES,
    tree_progress_interval_files: int = STAGE2_TREE_PROGRESS_MAX_INTERVAL_FILES,
) -> VerifiedStage2BankRestore:
    """Safely restore the exact reviewed bank tar into an immutable local root."""

    for label, digest in {
        "archive": expected_archive_sha256,
        "tree": expected_tree_sha256,
        "bank artifact": expected_bank_artifact_sha256,
    }.items():
        if _SHA256_RE.fullmatch(digest) is None:
            raise ValueError(f"Expected Stage-2 {label} identity must be lowercase SHA-256.")
    if type(expected_file_count) is not int or expected_file_count <= 0:
        raise ValueError("Expected Stage-2 bank file count must be a positive integer.")
    if type(expected_total_bytes) is not int or expected_total_bytes <= 0:
        raise ValueError("Expected Stage-2 bank byte count must be a positive integer.")
    _validate_progress_intervals(
        seconds=progress_interval_seconds,
        byte_interval=progress_interval_bytes,
        file_interval=tree_progress_interval_files,
    )

    if isinstance(archive_path, VerifiedStage2LocalArchive):
        verified_archive = archive_path
        archive = _validated_stage2_archive_file(
            verified_archive.local_archive_path,
            label="verified local Stage-2 bank archive",
        )
        current_stat = archive.stat()
        if (
            verified_archive.archive_file_sha256 != expected_archive_sha256
            or verified_archive.archive_bytes != current_stat.st_size
            or verified_archive.local_mtime_ns != current_stat.st_mtime_ns
        ):
            raise ValueError("Verified local Stage-2 bank archive receipt changed.")
    else:
        archive = _validated_stage2_archive_file(
            archive_path, label="reviewed Stage-2 bank archive"
        )
        if sha256_file(archive) != expected_archive_sha256:
            raise ValueError("Reviewed Stage-2 bank tar SHA-256 mismatch.")

    raw_bank_root = Path(destination)
    if raw_bank_root.is_symlink():
        raise ValueError("Local Stage-2 bank root may not be a symlink.")
    bank_root = raw_bank_root.resolve()
    if bank_root.exists():
        identity = _verify_restored_stage2_bank(
            bank_root,
            expected_tree_sha256=expected_tree_sha256,
            expected_bank_artifact_sha256=expected_bank_artifact_sha256,
            expected_file_count=expected_file_count,
            expected_total_bytes=expected_total_bytes,
            progress_callback=progress_callback,
            progress_interval_seconds=progress_interval_seconds,
            progress_interval_files=tree_progress_interval_files,
        )
        return VerifiedStage2BankRestore(
            archive_path=archive,
            archive_file_sha256=expected_archive_sha256,
            bank_root=bank_root,
            tree_sha256=identity["tree_sha256"],
            file_count=identity["file_count"],
            total_bytes=identity["total_bytes"],
            bank_artifact_sha256=expected_bank_artifact_sha256,
            restored_from_tar=False,
        )

    bank_root.parent.mkdir(parents=True, exist_ok=True)
    attempt = 1
    while True:
        staging = bank_root.parent / (
            f"{bank_root.name}.restore-attempt-{attempt:04d}.partial"
        )
        if not staging.exists() and not staging.is_symlink():
            staging.mkdir()
            break
        attempt += 1
    _safe_extract_reviewed_stage2_bank_tar(
        archive,
        staging,
        progress_callback=progress_callback,
        progress_interval_seconds=progress_interval_seconds,
        progress_interval_bytes=progress_interval_bytes,
    )
    extracted_root = _resolve_extracted_stage2_bank_root(staging)
    identity = _verify_restored_stage2_bank(
        extracted_root,
        expected_tree_sha256=expected_tree_sha256,
        expected_bank_artifact_sha256=expected_bank_artifact_sha256,
        expected_file_count=expected_file_count,
        expected_total_bytes=expected_total_bytes,
        progress_callback=progress_callback,
        progress_interval_seconds=progress_interval_seconds,
        progress_interval_files=tree_progress_interval_files,
    )
    if bank_root.exists():
        raise FileExistsError(
            "Concurrent Stage-2 bank restore created the final local destination."
        )
    os.rename(extracted_root, bank_root)
    if staging.exists():
        staging.rmdir()
    return VerifiedStage2BankRestore(
        archive_path=archive,
        archive_file_sha256=expected_archive_sha256,
        bank_root=bank_root,
        tree_sha256=identity["tree_sha256"],
        file_count=identity["file_count"],
        total_bytes=identity["total_bytes"],
        bank_artifact_sha256=expected_bank_artifact_sha256,
        restored_from_tar=True,
    )


def _safe_extract_reviewed_stage2_bank_tar(
    archive: Path,
    staging: Path,
    *,
    progress_callback: Callable[[dict[str, Any]], None] | None,
    progress_interval_seconds: float,
    progress_interval_bytes: int,
) -> None:
    seen: set[str] = set()
    with tarfile.open(archive, mode="r:") as bundle:
        members = bundle.getmembers()
        if not members:
            raise ValueError("Reviewed Stage-2 bank tar is empty.")
        planned: list[tuple[tarfile.TarInfo, PurePosixPath]] = []
        for member in members:
            relative = _safe_stage2_tar_member_path(member)
            if relative is None:
                continue
            folded = relative.as_posix().casefold()
            if folded in seen:
                raise ValueError("Reviewed Stage-2 bank tar has duplicate member paths.")
            seen.add(folded)
            if not member.isdir() and (not member.isreg() or member.size < 0):
                raise ValueError(
                    "Reviewed Stage-2 bank tar contains a link or special member."
                )
            planned.append((member, relative))

        total_files = sum(1 for member, _ in planned if member.isreg())
        total_bytes = sum(member.size for member, _ in planned if member.isreg())
        if total_files == 0:
            raise ValueError("Reviewed Stage-2 bank tar contains no regular files.")
        processed_files = 0
        processed_bytes = 0
        start = time.monotonic()
        last_time = start
        last_bytes = 0
        last_files = 0
        _emit_progress(
            progress_callback,
            {
                "stage": "bank_archive_extraction",
                "status": "start",
                "files_processed": 0,
                "total_files": total_files,
                "bytes_processed": 0,
                "total_bytes": total_bytes,
            },
        )
        for member, relative in planned:
            target = (staging / Path(*relative.parts)).resolve()
            try:
                target.relative_to(staging.resolve())
            except ValueError as exc:
                raise ValueError("Reviewed Stage-2 bank tar member escapes staging.") from exc
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise ValueError("Reviewed Stage-2 bank tar would overwrite a member.")
            source = bundle.extractfile(member)
            if source is None:
                raise ValueError("Reviewed Stage-2 bank tar regular member is unreadable.")
            member_bytes = 0
            with source, target.open("xb") as destination_handle:
                while True:
                    chunk = source.read(8 * 1024**2)
                    if not chunk:
                        break
                    destination_handle.write(chunk)
                    member_bytes += len(chunk)
                    processed_bytes += len(chunk)
                    now = time.monotonic()
                    if (
                        now - last_time >= progress_interval_seconds
                        or processed_bytes - last_bytes >= progress_interval_bytes
                    ):
                        _emit_progress(
                            progress_callback,
                            {
                                "stage": "bank_archive_extraction",
                                "status": "periodic",
                                "files_processed": processed_files,
                                "total_files": total_files,
                                "bytes_processed": processed_bytes,
                                "total_bytes": total_bytes,
                            },
                        )
                        last_time = now
                        last_bytes = processed_bytes
                        last_files = processed_files
            if member_bytes != member.size or target.stat().st_size != member.size:
                raise ValueError("Reviewed Stage-2 bank tar member size changed on extract.")
            processed_files += 1
            now = time.monotonic()
            if (
                now - last_time >= progress_interval_seconds
                or processed_bytes - last_bytes >= progress_interval_bytes
                or processed_files - last_files
                >= STAGE2_TREE_PROGRESS_MAX_INTERVAL_FILES
            ):
                _emit_progress(
                    progress_callback,
                    {
                        "stage": "bank_archive_extraction",
                        "status": "periodic",
                        "files_processed": processed_files,
                        "total_files": total_files,
                        "bytes_processed": processed_bytes,
                        "total_bytes": total_bytes,
                    },
                )
                last_time = now
                last_bytes = processed_bytes
                last_files = processed_files
        _emit_progress(
            progress_callback,
            {
                "stage": "bank_archive_extraction",
                "status": "end",
                "files_processed": processed_files,
                "total_files": total_files,
                "bytes_processed": processed_bytes,
                "total_bytes": total_bytes,
            },
        )


def _safe_stage2_tar_member_path(member: tarfile.TarInfo) -> PurePosixPath | None:
    name = member.name
    if (
        not isinstance(name, str)
        or not name
        or "\x00" in name
        or chr(92) in name
        or name.startswith("/")
        or any(ord(character) < 32 for character in name)
    ):
        raise ValueError("Reviewed Stage-2 bank tar contains a malformed member path.")
    while name.startswith("./"):
        name = name[2:]
    if member.isdir():
        name = name.rstrip("/")
    if name in {"", "."}:
        if member.isdir():
            return None
        raise ValueError("Reviewed Stage-2 bank tar has an empty file basename.")
    if "//" in name:
        raise ValueError("Reviewed Stage-2 bank tar contains an ambiguous member path.")
    parts = name.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("Reviewed Stage-2 bank tar contains path traversal.")
    if re.fullmatch(r"[A-Za-z]:", parts[0]) or any(":" in part for part in parts):
        raise ValueError("Reviewed Stage-2 bank tar contains a drive-qualified path.")
    return PurePosixPath(*parts)


def _resolve_extracted_stage2_bank_root(staging: Path) -> Path:
    direct_manifest = staging / STAGE2_BANK_MANIFEST_FILENAME
    if direct_manifest.is_file():
        return staging
    children = list(staging.iterdir())
    directories = [child for child in children if child.is_dir() and not child.is_symlink()]
    if len(children) == 1 and len(directories) == 1:
        nested = directories[0]
        if (nested / STAGE2_BANK_MANIFEST_FILENAME).is_file():
            return nested
    raise ValueError(
        "Reviewed Stage-2 bank tar has an ambiguous root or lacks the bank manifest."
    )


def _verify_restored_stage2_bank(
    root: Path,
    *,
    expected_tree_sha256: str,
    expected_bank_artifact_sha256: str,
    expected_file_count: int,
    expected_total_bytes: int,
    progress_callback: Callable[[dict[str, Any]], None] | None,
    progress_interval_seconds: float,
    progress_interval_files: int,
) -> dict[str, Any]:
    identity = stage2_bank_tree_identity(
        root,
        progress_callback=progress_callback,
        progress_interval_seconds=progress_interval_seconds,
        progress_interval_files=progress_interval_files,
    )
    if identity != {
        "file_count": expected_file_count,
        "total_bytes": expected_total_bytes,
        "tree_sha256": expected_tree_sha256,
    }:
        raise ValueError(
            "Restored Stage-2 bank tree count, byte total, or SHA-256 mismatch."
        )
    manifest = _read_json(root / STAGE2_BANK_MANIFEST_FILENAME)
    if manifest.get("artifact_sha256") != expected_bank_artifact_sha256:
        raise ValueError("Restored Stage-2 bank artifact SHA-256 mismatch.")
    return identity


@dataclass(frozen=True, slots=True)
class _Gate01MetadataPreflight:
    layout: Gate01ArchiveLayout
    result_path: Path
    result: dict[str, Any]
    verified_result_file_sha256: str
    manifest_path: Path
    lock_path: Path
    calibrator_path: Path
    lock: Any
    calibrator: Any
    gate_manifest: Any
    gate_metadata: dict[str, Any]


def resolve_gate01_p0006_archive_layout(
    archive_root: str | Path,
) -> Gate01ArchiveLayout:
    """Resolve only the modern flat CSV or reviewed parent-root legacy JSON layout."""

    supplied_root = Path(archive_root)
    if supplied_root.is_symlink():
        raise ValueError("Gate 0.1 logical archive root may not be a symlink.")
    root = supplied_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Gate 0.1 archive root does not exist: {root}")
    modern = root / "sha256-inventory.csv"
    legacy = root / "archive" / "sha256-inventory.json"
    matches: list[tuple[str, Literal["csv", "reviewed_legacy_json"], Path]] = []
    if modern.exists():
        matches.append((GATE01_ARCHIVE_LAYOUT_MODERN_FLAT_V1, "csv", modern))
    if legacy.exists():
        matches.append(
            (
                GATE01_ARCHIVE_LAYOUT_REVIEWED_LEGACY_JSON_V2,
                "reviewed_legacy_json",
                legacy,
            )
        )
    if len(matches) != 1:
        if not matches:
            raise FileNotFoundError(
                "Gate 0.1 archive matches neither supported layout: expected exactly "
                "root/sha256-inventory.csv or root/archive/sha256-inventory.json."
            )
        raise ValueError(
            "Gate 0.1 archive layout is ambiguous because modern CSV and reviewed "
            "legacy JSON inventories are both present."
        )
    layout_contract, inventory_format, inventory_path = matches[0]
    _assert_contained_regular_file(root, inventory_path, label="inventory")
    if inventory_format == "csv":
        entries = _verify_csv_archive_inventory(root, inventory_path)
        supplemental_dependencies: tuple[
            VerifiedGate01SupplementalDependency, ...
        ] = ()
    else:
        entries = _verify_legacy_json_archive_inventory(root, inventory_path)
        supplemental_dependencies = _verify_legacy_supplemental_dependencies(root)
    required = _GATE01_REQUIRED_BY_LAYOUT[layout_contract]
    observed = {entry.relative_path for entry in entries}
    missing = sorted(required - observed)
    unexpected = (
        sorted(observed - required)
        if layout_contract == GATE01_ARCHIVE_LAYOUT_REVIEWED_LEGACY_JSON_V2
        else []
    )
    if missing or unexpected:
        details = []
        if missing:
            details.append("missing=" + ", ".join(missing))
        if unexpected:
            details.append("unexpected=" + ", ".join(unexpected))
        raise ValueError(
            "Gate 0.1 inventory lacks required scientific dependencies: "
            + "; ".join(details)
        )
    normalized = [entry.to_dict() for entry in entries]
    return Gate01ArchiveLayout(
        logical_root=root,
        inventory_path=inventory_path.resolve(),
        layout_contract=layout_contract,
        inventory_format=inventory_format,
        entries=entries,
        supplemental_dependencies=supplemental_dependencies,
        inventory_file_sha256=sha256_file(inventory_path),
        normalized_inventory_sha256=sha256_json(normalized),
    )


def preflight_gate01_p0006_archive(
    archive_root: str | Path,
    *,
    expected_gate01_result_sha256: str,
) -> dict[str, Any]:
    """Validate the complete Gate 0.1 metadata graph without loading private arrays."""

    metadata = _preflight_gate01_p0006_archive(
        archive_root,
        expected_gate01_result_sha256=expected_gate01_result_sha256,
    )
    layout = metadata.layout
    receipt: dict[str, Any] = {
        "contract_version": GATE01_ARCHIVE_PREFLIGHT_CONTRACT,
        "status": "pass",
        "logical_archive_root": str(layout.logical_root),
        "logical_archive_root_identity_sha256": sha256_text(str(layout.logical_root)),
        "archive_layout_contract": layout.layout_contract,
        "inventory_format": layout.inventory_format,
        "inventory_path": str(layout.inventory_path),
        "inventory_file_sha256": layout.inventory_file_sha256,
        "normalized_inventory_entry_count": len(layout.entries),
        "normalized_inventory_sha256": layout.normalized_inventory_sha256,
        "supplemental_dependencies": [
            dependency.to_dict()
            for dependency in layout.supplemental_dependencies
        ],
        "expected_gate01_result_sha256": expected_gate01_result_sha256,
        "expected_gate01_result_match": True,
        "verified_gate01_result_file_sha256": (
            metadata.verified_result_file_sha256
        ),
        "stored_absolute_inventory_paths_trusted_for_file_access": False,
        "private_array_payloads_opened": 0,
        "required_scientific_contracts_unique": True,
        "logical_root_containment_verified": True,
    }
    receipt["preflight_sha256"] = sha256_json(receipt)
    return receipt


def _load_verified_gate01_result_snapshot(
    result_path: Path,
    result_entry: VerifiedGate01InventoryEntry,
    *,
    expected_gate01_result_sha256: str,
) -> tuple[dict[str, Any], str]:
    snapshot = result_path.read_bytes()
    snapshot_sha256 = hashlib.sha256(snapshot).hexdigest()
    if len(snapshot) != result_entry.size_bytes:
        raise ValueError(
            "Gate 0.1 result byte snapshot size differs from the reviewed "
            "inventory row."
        )
    if result_entry.sha256 != expected_gate01_result_sha256:
        raise ValueError(
            "Expected Gate 0.1 result SHA-256 does not match the exact reviewed "
            "result path."
        )
    if expected_gate01_result_sha256 != REVIEWED_GATE01_RESULT_FILE_SHA256:
        raise ValueError(
            "Expected Gate 0.1 result SHA-256 differs from the pinned authentic "
            "result file identity."
        )
    if snapshot_sha256 != result_entry.sha256:
        raise ValueError(
            "Gate 0.1 result byte snapshot SHA-256 differs from the reviewed "
            "inventory row."
        )
    try:
        decoded = snapshot.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("Gate 0.1 result byte snapshot is not UTF-8 JSON.") from exc
    payload = _json_loads_without_duplicate_keys(
        decoded,
        label="Gate 0.1 result byte snapshot",
    )
    if not isinstance(payload, Mapping):
        raise ValueError("Gate 0.1 result byte snapshot must be a JSON object.")
    result = dict(payload)
    if result.get("contract_version") != GATE01_CONTRACT_VERSION:
        raise ValueError("Expected Gate 0.1 result has an incompatible contract.")
    return result, snapshot_sha256


def _preflight_gate01_p0006_archive(
    archive_root: str | Path,
    *,
    expected_gate01_result_sha256: str,
) -> _Gate01MetadataPreflight:
    if _SHA256_RE.fullmatch(expected_gate01_result_sha256) is None:
        raise ValueError("Expected Gate 0.1 result identity must be lowercase SHA-256.")
    layout = resolve_gate01_p0006_archive_layout(archive_root)
    root = layout.logical_root
    result_entry = next(
        (
            entry
            for entry in layout.entries
            if entry.relative_path == "gate01-results.json"
        ),
        None,
    )
    if result_entry is None:
        raise ValueError(
            "Gate 0.1 inventory lacks the exact reviewed result path."
        )
    result_path = layout.path_for("gate01-results.json")
    if result_path.name != "gate01-results.json":
        raise ValueError("Expected Gate 0.1 result SHA resolved to the wrong dependency.")
    result, verified_result_file_sha256 = _load_verified_gate01_result_snapshot(
        result_path,
        result_entry,
        expected_gate01_result_sha256=expected_gate01_result_sha256,
    )
    verified_payloads = {"gate01-results.json": result}

    manifest_path = _single_inventory_json_contract(
        layout,
        GATE01_INPUT_CONTRACT_VERSION,
        verified_payloads=verified_payloads,
    )
    lock_path = _single_inventory_json_contract(
        layout,
        GATE01_PROTOCOL_LOCK_CONTRACT_VERSION,
        verified_payloads=verified_payloads,
    )
    calibrator_path = _single_inventory_json_contract(
        layout,
        GATE01_CALIBRATOR_CONTRACT_VERSION,
        verified_payloads=verified_payloads,
    )
    expected_names = {
        manifest_path.name: "gate01-private-manifest.json",
        lock_path.name: "gate01-protocol-lock.json",
        calibrator_path.name: "gate01-target-calibrator.json",
    }
    if any(observed != expected for observed, expected in expected_names.items()):
        raise ValueError("Gate 0.1 required contract resolved to an unexpected basename.")
    for relative_path in _GATE01_REQUIRED_BY_LAYOUT[layout.layout_contract]:
        candidate = layout.path_for(relative_path)
        if candidate.suffix == ".json":
            if relative_path not in verified_payloads:
                _read_json(candidate)

    lock = Gate01ProtocolLock.load(lock_path)
    if layout.layout_contract == GATE01_ARCHIVE_LAYOUT_REVIEWED_LEGACY_JSON_V2:
        _verify_supplemental_linkage(layout, lock)
    calibrator = PosthocTargetCalibrator.load(
        calibrator_path,
        expected_split_fingerprint=lock.split_fingerprint,
        expected_template_sha256=lock.calibrator_template_sha256,
        expected_artifact_sha256=lock.calibrator_artifact_sha256,
    )
    gate_manifest, gate_metadata = load_gate01_input_manifest(
        manifest_path, protocol_lock=lock, calibrator=calibrator
    )
    _assert_gate_arrays_inside_archive(gate_manifest, root)
    _assert_p0006_metadata(gate_metadata, lock)
    return _Gate01MetadataPreflight(
        layout=layout,
        result_path=result_path,
        result=result,
        verified_result_file_sha256=verified_result_file_sha256,
        manifest_path=manifest_path,
        lock_path=lock_path,
        calibrator_path=calibrator_path,
        lock=lock,
        calibrator=calibrator,
        gate_manifest=gate_manifest,
        gate_metadata=gate_metadata,
    )


def _assert_p0006_metadata(gate_metadata: Mapping[str, Any], lock: Any) -> None:
    if gate_metadata.get("execution_mode") != "scientific":
        raise ValueError("P:0006 import requires the sealed scientific Gate 0.1 graph.")
    evidence = gate_metadata.get("evidence_scope")
    if not isinstance(evidence, Mapping) or (
        evidence.get("private_data_run") is not True
        or evidence.get("evidence_kind") != "private"
        or evidence.get("traveller_identity_sha256") != P0006_IDENTITY_SHA256
    ):
        raise ValueError("Gate 0.1 archive is not the reviewed P:0006 private graph.")
    if lock.traveller_identity_sha256 != P0006_IDENTITY_SHA256:
        raise ValueError("Gate 0.1 protocol lock is not pinned to P:0006.")
    if lock.traveller_identity_sha256 in FORBIDDEN_OTHER_TRAVELLER_HASHES:
        raise ValueError("P:0007/P:0009 are forbidden from the P:0006 protocol.")
    producer = gate_metadata.get("producer_receipt")
    sealed = producer.get("producer_receipt") if isinstance(producer, Mapping) else None
    if not isinstance(sealed, Mapping) or (
        sealed.get("decode_strategy") != "full"
        or sealed.get("path_used") != ["full"]
        or sealed.get("direction_count") != 60
        or sealed.get("wrong_target_reference_count") != 180
    ):
        raise ValueError("P:0006 protocol lacks the frozen full-volume decode proof.")


def _verify_csv_archive_inventory(
    root: Path, path: Path
) -> tuple[VerifiedGate01InventoryEntry, ...]:
    rows: list[VerifiedGate01InventoryEntry] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            lowered = {
                str(key).lower(): "" if value is None else str(value)
                for key, value in raw.items()
            }
            digest = lowered.get("hash", "").lower()
            source_path = lowered.get("path", "")
            if _SHA256_RE.fullmatch(digest) is None or not source_path:
                raise ValueError("Gate 0.1 archive inventory row is malformed.")
            basename = _inventory_basename(source_path)
            candidate = _verified_direct_child(root, basename)
            size = candidate.stat().st_size
            declared_size = next(
                (
                    lowered[key]
                    for key in ("size_bytes", "length", "bytes")
                    if lowered.get(key, "") != ""
                ),
                None,
            )
            if declared_size is not None:
                if not declared_size.isdecimal() or int(declared_size) != size:
                    raise ValueError(
                        f"Gate 0.1 archived file size mismatch: {basename}"
                    )
            if sha256_file(candidate) != digest:
                raise ValueError(
                    f"Gate 0.1 archived file is missing or changed: {basename}"
                )
            rows.append(
                VerifiedGate01InventoryEntry(
                    relative_path=basename,
                    sha256=digest,
                    size_bytes=size,
                    resolution_rule="modern_flat_direct_logical_root_child",
                    stored_path_label_sha256=sha256_text(source_path),
                )
            )
    return _normalize_inventory_rows(rows)


def _verify_legacy_json_archive_inventory(
    root: Path, path: Path
) -> tuple[VerifiedGate01InventoryEntry, ...]:
    payload = _json_loads_without_duplicate_keys(
        path.read_text(encoding="utf-8-sig"),
        label="reviewed legacy Gate 0.1 inventory",
    )
    if not isinstance(payload, list):
        raise ValueError("Reviewed legacy Gate 0.1 inventory must be a top-level JSON list.")
    if len(payload) != len(_GATE01_LEGACY_REQUIRED_RELATIVE_PATHS):
        raise ValueError(
            "Reviewed legacy Gate 0.1 inventory must contain exactly 14 rows."
        )
    rows: list[VerifiedGate01InventoryEntry] = []
    seen_records: set[tuple[str, str, int]] = set()
    for index, raw in enumerate(payload):
        if not isinstance(raw, Mapping) or set(raw) != {"path", "sha256", "size_bytes"}:
            raise ValueError(f"Reviewed legacy inventory row {index} is malformed.")
        stored_path = raw["path"]
        digest = raw["sha256"]
        size = raw["size_bytes"]
        if not isinstance(stored_path, str) or not isinstance(digest, str):
            raise ValueError(f"Reviewed legacy inventory row {index} is malformed.")
        if _SHA256_RE.fullmatch(digest) is None:
            raise ValueError(f"Reviewed legacy inventory row {index} has malformed SHA-256.")
        if type(size) is not int or size < 0:
            raise ValueError(f"Reviewed legacy inventory row {index} has invalid size_bytes.")
        record = (stored_path, digest, size)
        if record in seen_records:
            raise ValueError("Reviewed legacy inventory contains a duplicate entry.")
        seen_records.add(record)
        relative_path, resolution_rule, normalized_label = _legacy_relative_path(
            stored_path,
            digest=digest,
            size_bytes=size,
        )
        candidate = _verified_relative_child(root, relative_path)
        if candidate.stat().st_size != size:
            raise ValueError(
                f"Gate 0.1 archived file size mismatch: {relative_path}"
            )
        if sha256_file(candidate) != digest:
            raise ValueError(
                f"Gate 0.1 archived file hash mismatch: {relative_path}"
            )
        rows.append(
            VerifiedGate01InventoryEntry(
                relative_path=relative_path,
                sha256=digest,
                size_bytes=size,
                resolution_rule=resolution_rule,
                stored_path_label_sha256=sha256_text(normalized_label),
            )
        )
    return _normalize_inventory_rows(rows)


def _normalize_inventory_rows(
    rows: list[VerifiedGate01InventoryEntry],
) -> tuple[VerifiedGate01InventoryEntry, ...]:
    if not rows:
        raise ValueError("Gate 0.1 archive inventory is empty.")
    folded = [entry.relative_path.casefold() for entry in rows]
    if len(set(folded)) != len(folded):
        raise ValueError(
            "Gate 0.1 archive inventory has duplicate or case-colliding relative paths."
        )
    return tuple(sorted(rows, key=lambda item: item.relative_path))


def _legacy_relative_path(
    stored_path: str,
    *,
    digest: str,
    size_bytes: int,
) -> tuple[str, str, str]:
    normalized = _normalize_legacy_path_label(stored_path)
    if normalized == _GATE01_LEGACY_SPLIT_STORED_PATH:
        if (
            PurePosixPath(normalized).name != "split_v3.json"
            or digest != _GATE01_LEGACY_SPLIT_SHA256
            or size_bytes != _GATE01_LEGACY_SPLIT_SIZE_BYTES
        ):
            raise ValueError(
                "Reviewed legacy split relocation identity is not exactly pinned."
            )
        return (
            _GATE01_LEGACY_SPLIT_RELATIVE_PATH,
            "reviewed_pinned_external_split_relocation",
            normalized,
        )

    parts = PurePosixPath(normalized).parts
    marker_indexes = [
        index for index, part in enumerate(parts) if part == _GATE01_LEGACY_ROOT_MARKER
    ]
    if len(marker_indexes) != 1:
        raise ValueError(
            "Reviewed legacy stored path must contain exactly one bundle-root marker."
        )
    suffix = parts[marker_indexes[0] + 1 :]
    relative_path = _validated_relative_path_parts(suffix)
    return relative_path, "anchored_suffix_after_reviewed_bundle_root", normalized


def _normalize_legacy_path_label(stored_path: str) -> str:
    if (
        not isinstance(stored_path, str)
        or not stored_path
        or stored_path != stored_path.strip()
        or not stored_path.startswith("/")
        or chr(92) in stored_path
        or "//" in stored_path
        or stored_path.endswith("/")
        or any(ord(character) < 32 or ord(character) == 127 for character in stored_path)
    ):
        raise ValueError("Reviewed legacy inventory contains a malformed stored path label.")
    parts = PurePosixPath(stored_path).parts
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("Reviewed legacy inventory stored path contains dot traversal.")
    normalized = "/" + "/".join(parts[1:])
    if normalized != stored_path:
        raise ValueError("Reviewed legacy inventory stored path is ambiguously normalized.")
    return normalized


def _validated_relative_path_parts(parts: tuple[str, ...]) -> str:
    if not parts:
        raise ValueError("Reviewed legacy inventory has an empty anchored suffix.")
    if any(
        not part
        or part in {".", ".."}
        or "/" in part
        or chr(92) in part
        or any(ord(character) < 32 or ord(character) == 127 for character in part)
        for part in parts
    ):
        raise ValueError("Reviewed legacy inventory has a malformed anchored suffix.")
    if re.fullmatch(r"[A-Za-z]:", parts[0]) is not None:
        raise ValueError("Reviewed legacy inventory suffix may not be drive-qualified.")
    relative_path = PurePosixPath(*parts)
    if relative_path.is_absolute():
        raise ValueError("Reviewed legacy inventory suffix must be relative.")
    return relative_path.as_posix()


def _verify_legacy_supplemental_dependencies(
    root: Path,
) -> tuple[VerifiedGate01SupplementalDependency, ...]:
    verified: list[VerifiedGate01SupplementalDependency] = []
    for relative_path, (digest, size_bytes, semantic_contract) in sorted(
        _GATE01_LEGACY_SUPPLEMENTAL_SPECS.items()
    ):
        candidate = _verified_direct_child(root, relative_path)
        if candidate.stat().st_size != size_bytes:
            raise ValueError(
                f"Gate 0.1 supplemental dependency size mismatch: {relative_path}"
            )
        if sha256_file(candidate) != digest:
            raise ValueError(
                f"Gate 0.1 supplemental dependency hash mismatch: {relative_path}"
            )
        if relative_path == "colab-operational-source-split.json":
            operational_payload = _json_loads_without_duplicate_keys(
                candidate.read_text(encoding="utf-8-sig"),
                label="Gate 0.1 operational source split",
            )
            if not isinstance(operational_payload, Mapping):
                raise ValueError(
                    "Gate 0.1 operational source split must be a JSON object."
                )
            split = load_vae_splits(candidate)
            semantic_identity_sha256 = vae_splits_fingerprint(split)
            semantic_entry_count = sum(
                len(split.records_for(name))
                for name in ("train", "validation", "test")
            )
        else:
            module_evidence = _load_reviewed_module_comparison_evidence(candidate)
            semantic_identity_sha256 = sha256_json(
                module_evidence.current_module_hashes
            )
            semantic_entry_count = len(module_evidence.evaluation_module_sha256)
        verified.append(
            VerifiedGate01SupplementalDependency(
                relative_path=relative_path,
                sha256=digest,
                size_bytes=size_bytes,
                semantic_contract=semantic_contract,
                semantic_identity_sha256=semantic_identity_sha256,
                semantic_entry_count=semantic_entry_count,
            )
        )
    return tuple(verified)


def _canonical_scientific_module_path(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} is not a canonical module path.")
    if (
        chr(92) in value
        or value.startswith("/")
        or "//" in value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError(f"{label} is not a canonical module path.")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
        or re.match(r"[A-Za-z]:", value) is not None
    ):
        raise ValueError(f"{label} is not a canonical module path.")
    return value


def _validated_reviewed_module_map(
    value: Any, *, label: str
) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a module-keyed JSON object.")
    module_hashes: dict[str, str] = {}
    folded_modules: set[str] = set()
    for module, digest in value.items():
        canonical_module = _canonical_scientific_module_path(module, label=label)
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            raise ValueError(f"{label} contains a malformed SHA-256 digest.")
        folded = canonical_module.casefold()
        if folded in folded_modules:
            raise ValueError(f"{label} contains case-fold-colliding module keys.")
        folded_modules.add(folded)
        module_hashes[canonical_module] = digest
    if (
        len(module_hashes) != _GATE01_REVIEWED_MODULE_COUNT
        or set(module_hashes) != set(GATE01_SCIENTIFIC_MODULES)
    ):
        raise ValueError(f"{label} has missing or unexpected scientific modules.")
    return tuple(sorted(module_hashes.items()))


def _load_reviewed_module_comparison_evidence(
    path: Path,
) -> ReviewedGate01ModuleComparisonEvidence:
    payload = _json_loads_without_duplicate_keys(
        path.read_text(encoding="utf-8-sig"),
        label="reviewed Gate 0.1 module comparison evidence",
    )
    if not isinstance(payload, Mapping):
        raise ValueError(
            "Reviewed Gate 0.1 module comparison evidence must be a JSON object."
        )
    if set(payload) != _GATE01_REVIEWED_MODULE_COMPARISON_KEYS:
        raise ValueError(
            "Reviewed Gate 0.1 module comparison evidence has an incorrect "
            "top-level key set."
        )
    evaluation_git_commit = payload["evaluation_git_commit"]
    previous_evaluation_git_commit = payload["previous_evaluation_git_commit"]
    if (
        not isinstance(evaluation_git_commit, str)
        or _GIT_COMMIT_RE.fullmatch(evaluation_git_commit) is None
        or not isinstance(previous_evaluation_git_commit, str)
        or _GIT_COMMIT_RE.fullmatch(previous_evaluation_git_commit) is None
    ):
        raise ValueError(
            "Reviewed Gate 0.1 module comparison evidence has a malformed "
            "Git commit identity."
        )
    if evaluation_git_commit == previous_evaluation_git_commit:
        raise ValueError(
            "Reviewed Gate 0.1 current and previous evaluation commits must "
            "be distinct."
        )
    current = _validated_reviewed_module_map(
        payload["evaluation_module_sha256"],
        label="Reviewed Gate 0.1 current module map",
    )
    previous = _validated_reviewed_module_map(
        payload["previous_evaluation_module_sha256"],
        label="Reviewed Gate 0.1 previous module map",
    )
    changed_payload = payload["changed_modules"]
    if not isinstance(changed_payload, list):
        raise ValueError(
            "Reviewed Gate 0.1 changed_modules must be a JSON list."
        )
    changed_modules = tuple(
        _canonical_scientific_module_path(
            module, label="Reviewed Gate 0.1 changed_modules entry"
        )
        for module in changed_payload
    )
    if len(set(changed_modules)) != len(changed_modules):
        raise ValueError("Reviewed Gate 0.1 changed_modules contains duplicates.")
    if any(module not in GATE01_SCIENTIFIC_MODULES for module in changed_modules):
        raise ValueError(
            "Reviewed Gate 0.1 changed_modules contains an unexpected "
            "scientific module."
        )
    current_map = dict(current)
    previous_map = dict(previous)
    computed_changes = tuple(
        sorted(
            module
            for module in GATE01_SCIENTIFIC_MODULES
            if current_map[module] != previous_map[module]
        )
    )
    if changed_modules != computed_changes:
        raise ValueError(
            "Reviewed Gate 0.1 changed_modules differs from the computed "
            "module-map changes."
        )
    if changed_modules != _GATE01_REVIEWED_CHANGED_MODULES:
        raise ValueError(
            "Reviewed Gate 0.1 module comparison evidence has an unexpected "
            "reviewed change set."
        )
    return ReviewedGate01ModuleComparisonEvidence(
        evaluation_git_commit=evaluation_git_commit,
        evaluation_module_sha256=current,
        previous_evaluation_git_commit=previous_evaluation_git_commit,
        previous_evaluation_module_sha256=previous,
        changed_modules=changed_modules,
    )


def _verify_supplemental_linkage(layout: Gate01ArchiveLayout, lock: Any) -> None:
    split_path = layout.supplemental_path_for(
        "colab-operational-source-split.json"
    )
    operational_split = load_vae_splits(split_path)
    if (
        vae_splits_fingerprint(operational_split)
        != lock.bank_source_split_fingerprint
    ):
        raise ValueError(
            "Gate 0.1 operational source split is not linked to the frozen bank split."
        )
    reviewed_module_path = layout.supplemental_path_for(
        "gate01-reviewed-module-sha256-8012a3f.json"
    )
    module_evidence = _load_reviewed_module_comparison_evidence(reviewed_module_path)
    if module_evidence.evaluation_git_commit != lock.evaluation_git_commit:
        raise ValueError(
            "Gate 0.1 reviewed current evaluation commit differs from the protocol lock."
        )
    if module_evidence.current_module_hashes != dict(
        lock.evaluation_module_sha256
    ):
        raise ValueError(
            "Gate 0.1 reviewed current module identities differ from the protocol lock."
        )
    if (
        module_evidence.previous_evaluation_git_commit == lock.evaluation_git_commit
        or module_evidence.previous_module_hashes
        == dict(lock.evaluation_module_sha256)
    ):
        raise ValueError(
            "Gate 0.1 previous module provenance cannot authorize current evaluation."
        )


def _json_loads_without_duplicate_keys(text: str, *, label: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains a duplicate JSON key: {key!r}.")
            result[key] = value
        return result

    try:
        return json.loads(text, object_pairs_hook=reject_duplicates)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is malformed JSON.") from exc


def _inventory_basename(stored_path: str) -> str:
    if stored_path != stored_path.strip() or not stored_path or "\x00" in stored_path:
        raise ValueError("Gate 0.1 inventory contains a malformed stored path label.")
    normalized = stored_path.replace(chr(92), "/")
    if normalized.endswith("/") or "//" in normalized:
        raise ValueError("Gate 0.1 inventory contains a malformed stored path label.")
    path = PurePosixPath(normalized)
    if any(part in {".", ".."} for part in path.parts):
        raise ValueError("Gate 0.1 inventory stored path contains dot traversal.")
    basename = path.name
    if basename in {"", ".", ".."} or "/" in basename or chr(92) in basename:
        raise ValueError("Gate 0.1 inventory stored path has an empty basename.")
    return basename


def _assert_contained_regular_file(root: Path, path: Path, *, label: str) -> Path:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Gate 0.1 {label} escapes the logical archive root.") from exc
    current = root
    for component in relative.parts:
        current = current / component
        if current.is_symlink():
            raise ValueError(
                f"Gate 0.1 {label} may not use a symlinked file or parent directory."
            )
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Gate 0.1 {label} is missing: {path}") from exc
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Gate 0.1 {label} escapes the logical archive root.") from exc
    if not resolved.is_file():
        raise ValueError(f"Gate 0.1 {label} is not a regular file: {path}")
    return resolved


def _verified_direct_child(root: Path, basename: str) -> Path:
    candidate = root / basename
    if candidate.parent != root:
        raise ValueError("Gate 0.1 inventory basename is not a direct-root child.")
    return _assert_contained_regular_file(root, candidate, label=f"artifact {basename!r}")


def _verified_relative_child(root: Path, relative_path: str) -> Path:
    parts = PurePosixPath(relative_path).parts
    normalized = _validated_relative_path_parts(parts)
    candidate = root.joinpath(*PurePosixPath(normalized).parts)
    return _assert_contained_regular_file(
        root,
        candidate,
        label=f"artifact {normalized!r}",
    )


def _single_inventory_json_contract(
    layout: Gate01ArchiveLayout,
    contract: str,
    *,
    verified_payloads: Mapping[str, Mapping[str, Any]] | None = None,
) -> Path:
    candidates: list[Path] = []
    trusted_payloads = verified_payloads or {}
    for entry in layout.entries:
        if not entry.basename.endswith(".json"):
            continue
        path = layout.path_for(entry.relative_path)
        try:
            payload = trusted_payloads.get(entry.relative_path)
            if payload is None:
                payload = _read_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if payload.get("contract_version") == contract:
            candidates.append(path)
    if len(candidates) != 1:
        raise ValueError(
            f"Gate 0.1 archive requires exactly one {contract!r} JSON; found {len(candidates)}."
        )
    return candidates[0]


def _resolve_p0006_count_progress_callback(
    callback: Callable[[dict[str, Any]], None] | None,
) -> Callable[[dict[str, Any]], None] | None:
    if callback is not None:
        return callback
    if os.environ.get(P0006_COUNT_PROGRESS_ENV) != P0006_COUNT_PROGRESS_VALUE:
        return None

    def emit(payload: dict[str, Any]) -> None:
        print(
            json.dumps({"p0006_import_progress": payload}, sort_keys=True),
            flush=True,
        )

    return emit


def import_gate01_p0006_evaluation_protocol(
    archive_root: str | Path,
    *,
    expected_gate01_result_sha256: str,
    bank_dir: str | Path,
    validation_plan_path: str | Path,
    output_path: str | Path,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Seal Gate01Private_8012a3f for P:0006 development/model assessment only."""

    progress = _resolve_p0006_count_progress_callback(progress_callback)
    _emit_progress(
        progress,
        {
            "stage": "p0006_import",
            "status": "start",
            "verified_inventory_entry_count": 0,
            "case_count": 0,
        },
    )
    metadata = _preflight_gate01_p0006_archive(
        archive_root,
        expected_gate01_result_sha256=expected_gate01_result_sha256,
    )
    layout = metadata.layout
    root = layout.logical_root
    result_path = metadata.result_path
    verified_result_file_sha256 = metadata.verified_result_file_sha256
    manifest_path = metadata.manifest_path
    lock_path = metadata.lock_path
    calibrator_path = metadata.calibrator_path
    lock = metadata.lock
    calibrator = metadata.calibrator
    gate_manifest = metadata.gate_manifest
    gate_metadata = metadata.gate_metadata
    _emit_progress(
        progress,
        {
            "stage": "p0006_import",
            "status": "periodic",
            "verified_inventory_entry_count": len(layout.entries),
            "case_count": 0,
        },
    )

    train_index = PhotometryFactoredLatentBankIndex(bank_dir, "train")
    validation_index = PhotometryFactoredLatentBankIndex(bank_dir, "validation")
    if any(
        record.sidecar.get("cohort") != "R"
        for record in (*train_index.records, *validation_index.records)
    ):
        raise ValueError("P:0006 evaluation import found a non-R factored-bank record.")
    validation_plan = _load_self_hashed(
        validation_plan_path, "validation_plan_sha256"
    )
    if validation_plan.get("contract_version") != UNIFIED_VALIDATION_PLAN_CONTRACT:
        raise ValueError("P:0006 import rejects an old unified validation plan.")
    rebuilt = build_unified_validation_plan(
        validation_index, validation_seed=int(validation_plan["validation_seed"])
    )
    if rebuilt != validation_plan:
        raise ValueError("Frozen unpaired validation plan differs from its R/validation bank.")
    if any(
        str(entry.get(role, "")).startswith("P:")
        for entry in validation_plan["entries"]
        for role in (
            "source_subject_group_identity",
            "target_subject_group_identity",
        )
    ):
        raise ValueError("A P endpoint entered the frozen unpaired validation plan.")
    _emit_progress(
        progress,
        {
            "stage": "p0006_import",
            "status": "periodic",
            "train_record_count": len(train_index.records),
            "validation_record_count": len(validation_index.records),
            "case_count": 0,
        },
    )

    cases = list(gate_manifest)
    case_receipts = []
    for case_count, case in enumerate(cases, start=1):
        case_receipts.append(_case_receipt(case, calibrator))
        if case_count % 10 == 0 or case_count == len(cases):
            _emit_progress(
                progress,
                {
                    "stage": "p0006_import",
                    "status": "periodic",
                    "case_count": case_count,
                    "expected_case_count": len(cases),
                },
            )
    directed_cells = {
        (
            case.source_domain.contrast.value,
            float(case.source_domain.field_strength_t),
            float(case.target_domain.field_strength_t),
        )
        for case in cases
    }
    if len(cases) != 60 or len(directed_cells) != 60:
        raise ValueError("P:0006 protocol requires exactly 60 directed field cases.")
    acquisition_nodes = {
        (case.source_domain.label, case.array_sha256["source_image"])
        for case in cases
    } | {
        (case.target_domain.label, case.array_sha256["target"])
        for case in cases
    }
    if len(acquisition_nodes) != 15:
        raise ValueError("P:0006 protocol requires exactly 15 acquisition nodes.")
    producer = gate_metadata.get("producer_receipt")
    sealed_producer = producer.get("producer_receipt") if isinstance(producer, Mapping) else None
    if not isinstance(sealed_producer, Mapping) or (
        sealed_producer.get("decode_strategy") != "full"
        or sealed_producer.get("path_used") != ["full"]
        or sealed_producer.get("direction_count") != 60
        or sealed_producer.get("wrong_target_reference_count") != 180
    ):
        raise ValueError("P:0006 protocol lacks the frozen full-volume decode proof.")

    body: dict[str, Any] = {
        "contract_version": GATE01_P0006_EVALUATION_PROTOCOL,
        "data_role": P0006_DEVELOPMENT_VALIDATION_DATA_ROLE,
        "evidence_interpretation": P0006_EVIDENCE_LIMITATION,
        "population_or_generalization_claims_authorized": False,
        "archive_identity": {
            "logical_root": str(root),
            "root_path_identity_sha256": sha256_text(str(root)),
            "layout_contract": layout.layout_contract,
            "inventory_format": layout.inventory_format,
            "inventory_file_sha256": layout.inventory_file_sha256,
            "verified_inventory_entry_count": len(layout.entries),
            "normalized_inventory_sha256": layout.normalized_inventory_sha256,
            "supplemental_dependency_count": len(
                layout.supplemental_dependencies
            ),
            "supplemental_dependencies_sha256": sha256_json(
                [
                    dependency.to_dict()
                    for dependency in layout.supplemental_dependencies
                ]
            ),
            "stored_inventory_paths_trusted_for_file_access": False,
        },
        "archive_inventory": {
            "path": str(layout.inventory_path),
            "file_sha256": layout.inventory_file_sha256,
            "layout_contract": layout.layout_contract,
            "format": layout.inventory_format,
            "normalized_entry_count": len(layout.entries),
            "normalized_inventory_sha256": layout.normalized_inventory_sha256,
            "verified_entries": [entry.to_dict() for entry in layout.entries],
            "stored_absolute_paths_trusted_for_file_access": False,
            "file_access_derivation": (
                "verified_exact_root_relative_path_from_layout_resolution_rule"
            ),
        },
        "supplemental_dependencies": {
            "included_in_archive_inventory": False,
            "normalized_dependency_count": len(layout.supplemental_dependencies),
            "normalized_dependencies_sha256": sha256_json(
                [
                    dependency.to_dict()
                    for dependency in layout.supplemental_dependencies
                ]
            ),
            "verified_dependencies": [
                dependency.to_dict()
                for dependency in layout.supplemental_dependencies
            ],
        },
        "gate01_result": {
            "path": str(result_path),
            "file_sha256": verified_result_file_sha256,
            "internal_self_hash_defined": False,
        },
        "gate01_manifest": {
            "path": str(manifest_path),
            "file_sha256": sha256_file(manifest_path),
            "selection_fingerprint_sha256": gate_metadata[
                "selection_fingerprint_sha256"
            ],
            "scientific_hash_graph": gate_metadata["scientific_hash_graph"],
        },
        "protocol_lock": {
            "path": str(lock_path),
            "file_sha256": sha256_file(lock_path),
            "artifact_sha256": lock.artifact_sha256,
        },
        "calibrator": {
            "path": str(calibrator_path),
            "file_sha256": sha256_file(calibrator_path),
            "artifact_sha256": calibrator.artifact_sha256,
        },
        "subject_group_identity": P0006_SUBJECT_GROUP,
        "traveller_identity_sha256": P0006_IDENTITY_SHA256,
        "acquisition_count": 15,
        "directed_pair_count": 60,
        "wrong_target_reference_count": 180,
        "full_volume_decode_proof": {
            "decode_strategy": "full",
            "path_used": ["full"],
            "producer_receipt_sha256": sha256_json(sealed_producer),
        },
        "factored_bank": {
            "train_artifact_sha256": train_index.artifact_sha256,
            "validation_artifact_sha256": validation_index.artifact_sha256,
            "train_record_count": len(train_index.records),
            "validation_record_count": len(validation_index.records),
            "accepted_cohorts": ["R"],
            "P_record_count": 0,
        },
        "frozen_unpaired_validation": {
            "validation_plan_sha256": validation_plan["validation_plan_sha256"],
            "contract_version": validation_plan["contract_version"],
            "P_endpoint_count": 0,
        },
        "case_receipts": case_receipts,
        "private_arrays_validated": True,
        "baseline_predictions_recomputed": False,
        "calibrated_identity_derivation": (
            "frozen_Gate01_calibrator_applied_to_existing_raw_identity_and_source_support"
        ),
        "training_or_model_selection_use": False,
        "forbidden_travellers": ["P:0007", "P:0009"],
        "P0009_confirmation_status": P0009_CONFIRMATION_STATUS,
        "P0009_executed": False,
    }
    body["protocol_sha256"] = sha256_json(body)
    write_json_atomic(output_path, body, refuse_existing=True)
    _emit_progress(
        progress,
        {
            "stage": "p0006_import",
            "status": "end",
            "verified_inventory_entry_count": len(layout.entries),
            "train_record_count": len(train_index.records),
            "validation_record_count": len(validation_index.records),
            "case_count": len(cases),
            "acquisition_node_count": len(acquisition_nodes),
        },
    )
    return body


def load_gate01_p0006_evaluation_protocol(
    path: str | Path,
    *,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[
    dict[str, Any],
    tuple[PairedEvaluationCase, ...],
    dict[str, dict[str, torch.Tensor]],
]:
    """Revalidate the sealed P:0006 graph and stream its full-volume arrays."""

    progress = _resolve_p0006_count_progress_callback(progress_callback)
    _emit_progress(
        progress,
        {"stage": "p0006_load", "status": "start", "case_count": 0},
    )
    protocol = _load_self_hashed(path, "protocol_sha256")
    if protocol.get("contract_version") not in SUPPORTED_GATE01_P0006_EVALUATION_PROTOCOLS:
        raise ValueError("Unsupported P:0006 evaluation-only protocol contract.")
    if protocol["contract_version"] == GATE01_P0006_EVALUATION_PROTOCOL:
        archive_root = _reverify_v4_archive_provenance(protocol)
    elif protocol["contract_version"] == GATE01_P0006_EVALUATION_PROTOCOL_V3:
        archive_root = _reverify_v3_archive_provenance(protocol)
    else:
        archive_root = None
    if (
        protocol.get("subject_group_identity") != P0006_SUBJECT_GROUP
        or protocol.get("traveller_identity_sha256") != P0006_IDENTITY_SHA256
        or protocol.get("data_role") != P0006_DEVELOPMENT_VALIDATION_DATA_ROLE
        or protocol.get("evidence_interpretation") != P0006_EVIDENCE_LIMITATION
        or protocol.get("population_or_generalization_claims_authorized") is not False
        or protocol.get("training_or_model_selection_use") is not False
        or protocol.get("private_arrays_validated") is not True
        or protocol.get("P0009_confirmation_status") != P0009_CONFIRMATION_STATUS
        or protocol.get("P0009_executed") is not False
    ):
        raise ValueError("P:0006 evaluation-only role or identity changed.")
    manifest_path = _verified_protocol_file(protocol, "gate01_manifest")
    _verified_protocol_file(protocol, "archive_inventory")
    lock_path = _verified_protocol_file(protocol, "protocol_lock")
    calibrator_path = _verified_protocol_file(protocol, "calibrator")
    _verified_protocol_file(protocol, "gate01_result")
    lock = Gate01ProtocolLock.load(lock_path)
    if protocol["contract_version"] == GATE01_P0006_EVALUATION_PROTOCOL:
        if archive_root is None:
            raise ValueError("P:0006 v4 archive root was not reverified.")
        reverified_layout = resolve_gate01_p0006_archive_layout(archive_root)
        if (
            reverified_layout.layout_contract
            == GATE01_ARCHIVE_LAYOUT_REVIEWED_LEGACY_JSON_V2
        ):
            _verify_supplemental_linkage(reverified_layout, lock)
    calibrator = PosthocTargetCalibrator.load(
        calibrator_path,
        expected_split_fingerprint=lock.split_fingerprint,
        expected_template_sha256=lock.calibrator_template_sha256,
        expected_artifact_sha256=lock.calibrator_artifact_sha256,
    )
    gate_manifest, metadata = load_gate01_input_manifest(
        manifest_path, protocol_lock=lock, calibrator=calibrator
    )
    _emit_progress(
        progress,
        {
            "stage": "p0006_load",
            "status": "periodic",
            "verified_inventory_entry_count": len(
                protocol.get("archive_inventory", {}).get("verified_entries", [])
            ),
            "case_count": 0,
        },
    )
    _assert_gate_arrays_inside_archive(
        gate_manifest,
        archive_root if archive_root is not None else manifest_path.resolve().parent,
    )
    if metadata["evidence_scope"]["traveller_identity_sha256"] != P0006_IDENTITY_SHA256:
        raise ValueError("P:0006 evaluation manifest identity changed.")
    expected_receipts = {
        str(item["case_identity"]): item for item in protocol["case_receipts"]
    }
    cases: list[PairedEvaluationCase] = []
    baselines: dict[str, dict[str, torch.Tensor]] = {}
    for case_count, gate_case in enumerate(gate_manifest, start=1):
        receipt = _case_receipt(gate_case, calibrator)
        expected = expected_receipts.pop(gate_case.case_id, None)
        if expected != receipt:
            raise ValueError("P:0006 case content or calibration identity changed.")
        if gate_case.source_image is None:
            raise ValueError("P:0006 source acquisition was not loaded.")
        cases.append(
            PairedEvaluationCase(
                case_identity=gate_case.case_id,
                source=_as_channel_volume(gate_case.source_image),
                target=_as_channel_volume(gate_case.target),
                source_domain=gate_case.source_domain,
                target_domain=gate_case.target_domain,
                subject_group_identity=P0006_SUBJECT_GROUP,
                source_provenance={
                    "case_id": gate_case.case_id,
                    "subject_group_identity": P0006_SUBJECT_GROUP,
                    "cohort": "P",
                    "data_role": protocol["data_role"],
                    "loaded_array_sha256": gate_case.array_sha256["source_image"],
                },
                target_provenance={
                    "case_id": gate_case.case_id,
                    "subject_group_identity": P0006_SUBJECT_GROUP,
                    "cohort": "P",
                    "data_role": protocol["data_role"],
                    "loaded_array_sha256": gate_case.array_sha256["target"],
                },
                stage1_reconstruction=_as_channel_volume(
                    gate_case.stage1_reconstruction_ceiling
                ),
                stage1_provenance={
                    "loaded_array_sha256": gate_case.array_sha256[
                        "stage1_reconstruction_ceiling"
                    ],
                    "path_used": "full",
                },
            )
        )
        baselines[gate_case.case_id] = {
            "gate01_calibrated_identity": _as_channel_volume(
                calibrator.apply(
                    gate_case.raw_identity,
                    gate_case.target_domain,
                    support_mask=gate_case.support_mask,
                )
            ),
            "original_sb_v2": _as_channel_volume(gate_case.raw_sb_v2),
        }
        if case_count % 10 == 0 or case_count == 60:
            _emit_progress(
                progress,
                {
                    "stage": "p0006_load",
                    "status": "periodic",
                    "case_count": case_count,
                    "expected_case_count": 60,
                },
            )
    if expected_receipts or len(cases) != 60:
        raise ValueError("P:0006 evaluation case inventory is incomplete or changed.")
    _emit_progress(
        progress,
        {
            "stage": "p0006_load",
            "status": "end",
            "case_count": len(cases),
            "expected_case_count": 60,
        },
    )
    return protocol, tuple(cases), baselines


def _reverify_v4_archive_provenance(protocol: Mapping[str, Any]) -> Path:
    identity = protocol.get("archive_identity")
    inventory = protocol.get("archive_inventory")
    supplemental = protocol.get("supplemental_dependencies")
    if (
        not isinstance(identity, Mapping)
        or not isinstance(inventory, Mapping)
        or not isinstance(supplemental, Mapping)
    ):
        raise ValueError("P:0006 v4 protocol lacks archive-layout provenance.")
    root_text = identity.get("logical_root")
    if not isinstance(root_text, str) or not root_text:
        raise ValueError("P:0006 v4 protocol lacks the logical archive root.")
    root = Path(root_text).resolve()
    layout = resolve_gate01_p0006_archive_layout(root)
    entries = [entry.to_dict() for entry in layout.entries]
    supplemental_entries = [
        dependency.to_dict() for dependency in layout.supplemental_dependencies
    ]
    supplemental_sha256 = sha256_json(supplemental_entries)
    if (
        identity.get("root_path_identity_sha256") != sha256_text(str(root))
        or identity.get("layout_contract") != layout.layout_contract
        or identity.get("inventory_format") != layout.inventory_format
        or identity.get("inventory_file_sha256") != layout.inventory_file_sha256
        or identity.get("verified_inventory_entry_count") != len(layout.entries)
        or identity.get("normalized_inventory_sha256")
        != layout.normalized_inventory_sha256
        or identity.get("supplemental_dependency_count")
        != len(layout.supplemental_dependencies)
        or identity.get("supplemental_dependencies_sha256")
        != supplemental_sha256
        or identity.get("stored_inventory_paths_trusted_for_file_access") is not False
        or Path(str(inventory.get("path", ""))).resolve() != layout.inventory_path
        or inventory.get("file_sha256") != layout.inventory_file_sha256
        or inventory.get("layout_contract") != layout.layout_contract
        or inventory.get("format") != layout.inventory_format
        or inventory.get("normalized_entry_count") != len(layout.entries)
        or inventory.get("normalized_inventory_sha256")
        != layout.normalized_inventory_sha256
        or inventory.get("verified_entries") != entries
        or inventory.get("stored_absolute_paths_trusted_for_file_access") is not False
        or inventory.get("file_access_derivation")
        != "verified_exact_root_relative_path_from_layout_resolution_rule"
        or supplemental.get("included_in_archive_inventory") is not False
        or supplemental.get("normalized_dependency_count")
        != len(layout.supplemental_dependencies)
        or supplemental.get("normalized_dependencies_sha256")
        != supplemental_sha256
        or supplemental.get("verified_dependencies") != supplemental_entries
    ):
        raise ValueError("P:0006 v4 archive-layout provenance changed or is incomplete.")
    expected_dependencies = {
        "gate01_result": "gate01-results.json",
        "gate01_manifest": "gate01-private-manifest.json",
        "protocol_lock": "gate01-protocol-lock.json",
        "calibrator": "gate01-target-calibrator.json",
    }
    entry_map = {entry.relative_path: entry.sha256 for entry in layout.entries}
    gate01_result = protocol.get("gate01_result")
    if (
        not isinstance(gate01_result, Mapping)
        or set(gate01_result)
        != {"path", "file_sha256", "internal_self_hash_defined"}
        or gate01_result.get("file_sha256")
        != REVIEWED_GATE01_RESULT_FILE_SHA256
        or gate01_result.get("internal_self_hash_defined") is not False
    ):
        raise ValueError("P:0006 v4 Gate 0.1 result provenance is malformed.")
    for key, relative_path in expected_dependencies.items():
        spec = protocol.get(key)
        if not isinstance(spec, Mapping):
            raise ValueError(f"P:0006 v4 protocol lacks {key} provenance.")
        dependency = Path(str(spec.get("path", ""))).resolve()
        if (
            dependency != layout.path_for(relative_path)
            or entry_map.get(relative_path) != spec.get("file_sha256")
        ):
            raise ValueError(
                f"P:0006 v4 dependency is not the exact inventoried path: {key}."
            )
    return root


def _reverify_v3_archive_provenance(protocol: Mapping[str, Any]) -> Path:
    """Safely retain the already-published direct-child v3 receipt semantics."""

    identity = protocol.get("archive_identity")
    inventory = protocol.get("archive_inventory")
    if not isinstance(identity, Mapping) or not isinstance(inventory, Mapping):
        raise ValueError("P:0006 v3 protocol lacks archive-layout provenance.")
    root_text = identity.get("logical_root")
    if not isinstance(root_text, str) or not root_text:
        raise ValueError("P:0006 v3 protocol lacks the logical archive root.")
    root = Path(root_text).resolve()
    if not root.is_dir():
        raise FileNotFoundError("P:0006 v3 logical archive root is missing.")
    layout_contract = identity.get("layout_contract")
    inventory_format = identity.get("inventory_format")
    if layout_contract == GATE01_ARCHIVE_LAYOUT_MODERN_FLAT_V1:
        expected_inventory_path = root / "sha256-inventory.csv"
        expected_format = "csv"
    elif layout_contract == GATE01_ARCHIVE_LAYOUT_REVIEWED_LEGACY_JSON_V1:
        expected_inventory_path = root / "archive" / "sha256-inventory.json"
        expected_format = "reviewed_legacy_json"
    else:
        raise ValueError("P:0006 v3 archive layout contract is unsupported.")
    if inventory_format != expected_format:
        raise ValueError("P:0006 v3 archive inventory format changed.")
    inventory_path = _assert_contained_regular_file(
        root, expected_inventory_path, label="v3 archive inventory"
    )
    raw_entries = inventory.get("verified_entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("P:0006 v3 archive inventory entries are missing.")
    entries: list[dict[str, Any]] = []
    folded: set[str] = set()
    for index, raw in enumerate(raw_entries):
        if not isinstance(raw, Mapping) or set(raw) != {
            "basename",
            "sha256",
            "size_bytes",
        }:
            raise ValueError(f"P:0006 v3 archive entry {index} is malformed.")
        basename = raw["basename"]
        digest = raw["sha256"]
        size_bytes = raw["size_bytes"]
        if (
            not isinstance(basename, str)
            or _inventory_basename(basename) != basename
            or basename.casefold() in folded
            or not isinstance(digest, str)
            or _SHA256_RE.fullmatch(digest) is None
            or type(size_bytes) is not int
            or size_bytes < 0
        ):
            raise ValueError(f"P:0006 v3 archive entry {index} is malformed.")
        folded.add(basename.casefold())
        dependency = _verified_direct_child(root, basename)
        if dependency.stat().st_size != size_bytes or sha256_file(dependency) != digest:
            raise ValueError(f"P:0006 v3 archive entry changed: {basename}.")
        entries.append(
            {"basename": basename, "sha256": digest, "size_bytes": size_bytes}
        )
    entries.sort(key=lambda item: str(item["basename"]))
    normalized_sha256 = sha256_json(entries)
    if (
        identity.get("root_path_identity_sha256") != sha256_text(str(root))
        or identity.get("inventory_file_sha256") != sha256_file(inventory_path)
        or identity.get("verified_inventory_entry_count") != len(entries)
        or identity.get("normalized_inventory_sha256") != normalized_sha256
        or identity.get("stored_inventory_paths_trusted_for_file_access") is not False
        or Path(str(inventory.get("path", ""))).resolve() != inventory_path
        or inventory.get("file_sha256") != sha256_file(inventory_path)
        or inventory.get("layout_contract") != layout_contract
        or inventory.get("format") != expected_format
        or inventory.get("normalized_entry_count") != len(entries)
        or inventory.get("normalized_inventory_sha256") != normalized_sha256
        or raw_entries != entries
        or inventory.get("stored_absolute_paths_trusted_for_file_access") is not False
        or inventory.get("file_access_derivation")
        != "verified_basename_as_direct_logical_root_child"
    ):
        raise ValueError("P:0006 v3 archive-layout provenance changed or is incomplete.")
    entry_map = {str(entry["basename"]): str(entry["sha256"]) for entry in entries}
    for key in ("gate01_result", "gate01_manifest", "protocol_lock", "calibrator"):
        spec = protocol.get(key)
        if not isinstance(spec, Mapping):
            raise ValueError(f"P:0006 v3 protocol lacks {key} provenance.")
        dependency = Path(str(spec.get("path", ""))).resolve()
        if (
            dependency.parent != root
            or entry_map.get(dependency.name) != spec.get("file_sha256")
            or dependency != _verified_direct_child(root, dependency.name)
        ):
            raise ValueError(
                f"P:0006 v3 dependency is not the inventoried root child: {key}."
            )
    return root


def verify_completed_stage2_pilot_evidence(
    training_namespace_root: str | Path,
    *,
    bank_dir: str | Path,
    expected_selection_receipt_path: str | Path | None = None,
    training_evidence_commit: str = TRAINING_EVIDENCE_COMMIT,
    expected_selection_receipt_file_sha256: str = (
        EXPECTED_STAGE2_SELECTION_RECEIPT_FILE_SHA256
    ),
    expected_validation_plan_sha256: str = EXPECTED_STAGE2_VALIDATION_PLAN_SHA256,
    expected_selection_rule_sha256: str = EXPECTED_STAGE2_SELECTION_RULE_SHA256,
) -> dict[str, Any]:
    """Verify the completed v7 qualification and pilots without invoking training."""

    if (
        not isinstance(training_evidence_commit, str)
        or _GIT_COMMIT_RE.fullmatch(training_evidence_commit) is None
    ):
        raise ValueError(
            "Expected training evidence commit identity must be a lowercase Git commit "
            "identity."
        )
    for label, digest in {
        "selection receipt file": expected_selection_receipt_file_sha256,
        "validation plan": expected_validation_plan_sha256,
        "selection rule": expected_selection_rule_sha256,
    }.items():
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            raise ValueError(f"Expected {label} identity must be lowercase SHA-256.")
    if expected_selection_rule_sha256 != UNIFIED_SELECTION_RULE_SHA256:
        raise ValueError("Pinned selection-rule SHA differs from the v7 implementation.")
    root = Path(training_namespace_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Completed training namespace is missing: {root}")
    expected_namespace = f"implementation_{training_evidence_commit[:12]}"
    if root.name != expected_namespace:
        raise ValueError(
            "Completed pilot must remain in its original read-only implementation namespace."
        )

    train_index = PhotometryFactoredLatentBankIndex(bank_dir, "train")
    validation_index = PhotometryFactoredLatentBankIndex(bank_dir, "validation")
    stats = FactoredLatentStats.from_bank(bank_dir)
    if any(
        record.sidecar.get("cohort") != "R"
        for record in (*train_index.records, *validation_index.records)
    ):
        raise ValueError("Completed pilot bank contains a non-R record.")

    full_run_dir = root / "unified_full_objective_pilot_200"
    if expected_selection_receipt_path is None:
        full_receipt = _unique_file(
            full_run_dir,
            "scientific_attempts/attempt-*/checkpoints/"
            "stage2_unified_full_selection_step000000200.json",
            label="step-200 selection receipt",
        )
    else:
        full_receipt = _assert_path_inside(
            full_run_dir,
            Path(expected_selection_receipt_path),
            label="step-200 selection receipt",
        )
        if full_receipt.name != "stage2_unified_full_selection_step000000200.json":
            raise ValueError("Unexpected step-200 selection receipt filename.")
    if sha256_file(full_receipt) != expected_selection_receipt_file_sha256:
        raise ValueError("Step-200 selection-receipt file SHA-256 mismatch.")

    full = _verify_completed_pilot_attempt(
        full_run_dir,
        steps=200,
        selection_receipt_path=full_receipt,
        train_index=train_index,
        validation_index=validation_index,
        stats=stats,
        training_evidence_commit=training_evidence_commit,
        expected_validation_plan_sha256=expected_validation_plan_sha256,
        expected_selection_rule_sha256=expected_selection_rule_sha256,
    )
    short_run_dir = root / "unified_full_objective_pilot_20"
    short_receipt = _unique_file(
        short_run_dir,
        "scientific_attempts/attempt-*/checkpoints/"
        "stage2_unified_full_selection_step000000020.json",
        label="step-20 selection receipt",
    )
    short = _verify_completed_pilot_attempt(
        short_run_dir,
        steps=20,
        selection_receipt_path=short_receipt,
        train_index=train_index,
        validation_index=validation_index,
        stats=stats,
        training_evidence_commit=training_evidence_commit,
        expected_validation_plan_sha256=expected_validation_plan_sha256,
        expected_selection_rule_sha256=expected_selection_rule_sha256,
    )
    qualification = _verify_completed_a100_qualification(
        root / "a100_full_objective_gate_1",
        train_index=train_index,
        validation_index=validation_index,
        stats=stats,
        training_evidence_commit=training_evidence_commit,
        expected_validation_plan_sha256=expected_validation_plan_sha256,
        expected_selection_rule_sha256=expected_selection_rule_sha256,
    )

    decoder_hashes = {
        qualification["frozen_decoder_state_sha256"],
        short["frozen_decoder_state_sha256"],
        full["frozen_decoder_state_sha256"],
    }
    decoder_evidence = {
        qualification["decoder_activation_checkpoint_sha256"],
        short["decoder_activation_checkpoint_sha256"],
        full["decoder_activation_checkpoint_sha256"],
    }
    if len(decoder_hashes) != 1 or len(decoder_evidence) != 1:
        raise ValueError("Qualification and pilots used different frozen decoder identities.")
    if {
        short["bank_artifact_sha256"],
        full["bank_artifact_sha256"],
        train_index.artifact_sha256,
    } != {train_index.artifact_sha256}:
        raise ValueError("Completed pilots do not share the restored bank identity.")

    receipt: dict[str, Any] = {
        "contract_version": STAGE2_COMPLETED_PILOT_REUSE_CONTRACT,
        "status": "pass",
        "training_reused": True,
        "training_invoked": False,
        "training_evidence_commit": training_evidence_commit,
        "training_namespace": str(root),
        "training_namespace_read_only": True,
        "selection_receipt": str(full_receipt),
        "selection_receipt_file_sha256": sha256_file(full_receipt),
        "latest_completed_step": 200,
        "checkpoint_file_sha256": full["latest_checkpoint_file_sha256"],
        "run_fingerprint": full["run_fingerprint"],
        "resolved_config_sha256": full["resolved_config_sha256"],
        "bank_artifact_sha256": train_index.artifact_sha256,
        "validation_bank_artifact_sha256": validation_index.artifact_sha256,
        "latent_statistics_sha256": stats.artifact_sha256,
        "validation_plan_sha256": expected_validation_plan_sha256,
        "selection_rule_sha256": expected_selection_rule_sha256,
        "complete_r_validation_inventory": True,
        "paired_targets_used": False,
        "full_objective_pilot_status": "pass",
        "a100_memory_gate_status": "pass",
        "anatomy_qualification_status": "pass",
        "decoder_state_unchanged": True,
        "six_required_objective_terms_enabled_and_qualified": True,
        "qualification": qualification,
        "pilot_20": short,
        "pilot_200": full,
    }
    receipt["reuse_verification_sha256"] = sha256_json(receipt)
    return receipt


def _verify_completed_pilot_attempt(
    run_dir: Path,
    *,
    steps: int,
    selection_receipt_path: Path,
    train_index: PhotometryFactoredLatentBankIndex,
    validation_index: PhotometryFactoredLatentBankIndex,
    stats: FactoredLatentStats,
    training_evidence_commit: str,
    expected_validation_plan_sha256: str,
    expected_selection_rule_sha256: str,
) -> dict[str, Any]:
    receipt = _load_self_hashed(selection_receipt_path, "selection_sha256")
    if (
        receipt.get("receipt_contract_version") != UNIFIED_SELECTION_RECEIPT_CONTRACT
        or receipt.get("contract_version") != UNIFIED_SELECTION_CONTRACT
        or receipt.get("variant") != "full"
        or receipt.get("receipt_step") != steps
        or receipt.get("terminal_step") != steps
        or receipt.get("latest_step") != steps
        or receipt.get("run_complete") is not True
        or receipt.get("complete_r_validation_inventory") is not True
        or receipt.get("paired_targets_used") is not False
        or receipt.get("validation_plan_sha256") != expected_validation_plan_sha256
        or receipt.get("selection_rule_sha256") != expected_selection_rule_sha256
    ):
        raise ValueError(f"Step-{steps} selection receipt is incomplete or incompatible.")
    checkpoint_dir = selection_receipt_path.resolve().parent
    validation_plan_path = checkpoint_dir / "stage2_unified_validation_plan_v2.json"
    plan = _verify_completed_validation_plan(
        validation_plan_path,
        validation_index=validation_index,
        expected_validation_plan_sha256=expected_validation_plan_sha256,
    )
    resolved_config, config_dict = _load_completed_stage2_config(run_dir)
    if (
        config_dict.get("steps") != steps
        or config_dict.get("pilot_steps") != steps
        or config_dict.get("variant") != "full"
        or config_dict.get("loss_weights") != DEFAULT_UNIFIED_WEIGHTS
    ):
        raise ValueError(f"Step-{steps} resolved pilot configuration changed.")

    hashes = receipt.get("checkpoint_hashes")
    if not isinstance(hashes, Mapping) or not {"latest", "best", "final"} <= set(hashes):
        raise ValueError(f"Step-{steps} selection receipt lacks checkpoint identities.")
    checkpoint_specs: dict[str, tuple[Path, str]] = {}
    for role in ("latest", "best", "final"):
        raw = hashes.get(role)
        if not isinstance(raw, Mapping):
            raise ValueError(f"Step-{steps} checkpoint {role} identity is malformed.")
        checkpoint_path = _assert_path_inside(
            checkpoint_dir,
            Path(str(raw.get("path", ""))),
            label=f"step-{steps} {role} checkpoint",
        )
        digest = raw.get("file_sha256")
        if not isinstance(digest, str) or sha256_file(checkpoint_path) != digest:
            raise ValueError(f"Step-{steps} checkpoint {role} file SHA-256 mismatch.")
        checkpoint_specs[role] = (checkpoint_path, digest)
    if checkpoint_specs["latest"] != checkpoint_specs["final"]:
        raise ValueError(f"Step-{steps} final/latest checkpoint identities disagree.")
    if str(receipt.get("latest_checkpoint")) != str(hashes["latest"]["path"]) or str(
        receipt.get("best_checkpoint")
    ) != str(hashes["best"]["path"]):
        raise ValueError(f"Step-{steps} selection checkpoint pointers disagree.")

    summaries: dict[Path, dict[str, Any]] = {}
    unique_checkpoints = {
        checkpoint_path for checkpoint_path, _digest in checkpoint_specs.values()
    }
    for checkpoint_path in sorted(unique_checkpoints):
        state = load_checkpoint(checkpoint_path, map_location="cpu")
        meta = state.get("_meta")
        if (
            state.get("contract_version") != UNIFIED_RESUME_CONTRACT
            or not isinstance(meta, Mapping)
            or meta.get("git_commit") != training_evidence_commit
            or meta.get("config") != config_dict
            or state.get("validation_plan_sha256") != expected_validation_plan_sha256
            or state.get("selection_rule_sha256") != expected_selection_rule_sha256
        ):
            raise ValueError(f"Step-{steps} checkpoint metadata or identity changed.")
        run_fingerprint = state.get("run_fingerprint")
        if not isinstance(run_fingerprint, str) or _SHA256_RE.fullmatch(run_fingerprint) is None:
            raise ValueError(f"Step-{steps} checkpoint run fingerprint is invalid.")
        summaries[checkpoint_path] = {
            "training_cursor": state.get("training_cursor"),
            "run_fingerprint": run_fingerprint,
            "pilot_report": state.get("pilot_report"),
            "selection": state.get("validation_selection"),
            "history_prefix_bytes": state.get("history_prefix_bytes"),
            "history_prefix_sha256": state.get("history_prefix_sha256"),
        }
        del state
    run_fingerprints = {summary["run_fingerprint"] for summary in summaries.values()}
    if len(run_fingerprints) != 1:
        raise ValueError(f"Step-{steps} selected checkpoints belong to different runs.")
    run_fingerprint = next(iter(run_fingerprints))
    latest_summary = summaries[checkpoint_specs["latest"][0]]
    best_summary = summaries[checkpoint_specs["best"][0]]
    if latest_summary["training_cursor"] != steps or best_summary[
        "training_cursor"
    ] != receipt.get("best_step"):
        raise ValueError(f"Step-{steps} checkpoint cursors disagree with selection.")
    selection = latest_summary["selection"]
    if not isinstance(selection, Mapping) or any(
        selection.get(key) != receipt.get(key)
        for key in (
            "contract_version",
            "validation_plan_sha256",
            "selection_rule_sha256",
            "paired_targets_used",
            "complete_r_validation_inventory",
            "latest_step",
            "latest_checkpoint",
            "best_step",
            "best_checkpoint",
            "best_score",
        )
    ):
        raise ValueError(f"Step-{steps} checkpoint/selection receipt contents disagree.")

    pilot = latest_summary["pilot_report"]
    _assert_completed_pilot_report(pilot, steps=steps)
    assert isinstance(pilot, Mapping)
    anatomy = pilot["one_step_anatomy_memory_qualification"]
    decoder_checkpoint_evidence = anatomy["checkpoint_evidence"]
    rebuilt_fingerprint = _rebuild_stage2_run_fingerprint(
        config_dict,
        train_index=train_index,
        validation_index=validation_index,
        stats=stats,
        validation_plan=plan,
        frozen_decoder_state_sha256=anatomy["decoder_state_sha256_before"],
        decoder_checkpoint_evidence=decoder_checkpoint_evidence,
        git_commit=training_evidence_commit,
        execution_mode="train_with_frozen_validation_and_selection",
        accumulation=pilot["generator_gradient_accumulation"],
    )
    if rebuilt_fingerprint != run_fingerprint:
        raise ValueError(f"Step-{steps} run fingerprint/config/bank identities disagree.")

    history_path = selection_receipt_path.parent.parent / "history.jsonl"
    history = _verify_completed_history(
        history_path,
        steps=steps,
        run_fingerprint=run_fingerprint,
        validation_plan_sha256=expected_validation_plan_sha256,
        pilot=pilot,
    )
    history_bytes = history_path.read_bytes()
    prefix_bytes = latest_summary["history_prefix_bytes"]
    if (
        type(prefix_bytes) is not int
        or prefix_bytes < 0
        or len(history_bytes) < prefix_bytes
        or hashlib.sha256(history_bytes[:prefix_bytes]).hexdigest()
        != latest_summary["history_prefix_sha256"]
    ):
        raise ValueError(f"Step-{steps} checkpoint-bound history prefix is incomplete.")
    return {
        "status": "pass",
        "steps": steps,
        "selection_receipt": str(selection_receipt_path),
        "selection_receipt_file_sha256": sha256_file(selection_receipt_path),
        "latest_checkpoint": str(checkpoint_specs["latest"][0]),
        "latest_checkpoint_file_sha256": checkpoint_specs["latest"][1],
        "best_checkpoint": str(checkpoint_specs["best"][0]),
        "best_checkpoint_file_sha256": checkpoint_specs["best"][1],
        "history_jsonl": str(history_path),
        "history_file_sha256": sha256_file(history_path),
        "history_training_row_count": history["training_row_count"],
        "run_fingerprint": run_fingerprint,
        "resolved_config_file_sha256": sha256_file(resolved_config),
        "resolved_config_sha256": sha256_json(config_dict),
        "bank_artifact_sha256": train_index.artifact_sha256,
        "validation_bank_artifact_sha256": validation_index.artifact_sha256,
        "validation_plan_sha256": expected_validation_plan_sha256,
        "selection_rule_sha256": expected_selection_rule_sha256,
        "frozen_decoder_state_sha256": anatomy["decoder_state_sha256_before"],
        "decoder_activation_checkpoint_sha256": sha256_json(
            decoder_checkpoint_evidence
        ),
        "all_six_objectives_enabled_and_qualified": True,
        "complete_r_validation_inventory": True,
        "paired_targets_used": False,
    }


def _verify_completed_a100_qualification(
    run_dir: Path,
    *,
    train_index: PhotometryFactoredLatentBankIndex,
    validation_index: PhotometryFactoredLatentBankIndex,
    stats: FactoredLatentStats,
    training_evidence_commit: str,
    expected_validation_plan_sha256: str,
    expected_selection_rule_sha256: str,
) -> dict[str, Any]:
    receipt_path = _unique_file(
        run_dir,
        "scientific_attempts/attempt-*/checkpoints/"
        + UNIFIED_A100_QUALIFICATION_RECEIPT_FILENAME,
        label="A100 qualification receipt",
    )
    receipt = _load_self_hashed(receipt_path, "receipt_sha256")
    gate = receipt.get("gate")
    anatomy = receipt.get("anatomy_memory_qualification")
    if (
        receipt.get("contract_version") != UNIFIED_A100_QUALIFICATION_ONLY_CONTRACT
        or receipt.get("status") != "pass"
        or receipt.get("step") != 1
        or receipt.get("complete_validation_executed") is not False
        or receipt.get("checkpoint_written") is not False
        or receipt.get("generator_optimizer_updates") != 1
        or receipt.get("validation_plan_sha256") != expected_validation_plan_sha256
        or not isinstance(gate, Mapping)
        or not isinstance(anatomy, Mapping)
    ):
        raise ValueError("A100 qualification-only receipt is incomplete or changed.")
    _assert_a100_gate(gate)
    _assert_anatomy_qualification(anatomy)
    validation_plan = _verify_completed_validation_plan(
        receipt_path.parent / "stage2_unified_validation_plan_v2.json",
        validation_index=validation_index,
        expected_validation_plan_sha256=expected_validation_plan_sha256,
    )
    resolved_config, config_dict = _load_completed_stage2_config(run_dir)
    if config_dict.get("steps") != 1 or config_dict.get("pilot_steps") != 0:
        raise ValueError("A100 qualification resolved configuration changed.")
    expected_fingerprint = _rebuild_stage2_run_fingerprint(
        config_dict,
        train_index=train_index,
        validation_index=validation_index,
        stats=stats,
        validation_plan=validation_plan,
        frozen_decoder_state_sha256=receipt["frozen_decoder_state_sha256"],
        decoder_checkpoint_evidence=anatomy["checkpoint_evidence"],
        git_commit=training_evidence_commit,
        execution_mode=UNIFIED_A100_QUALIFICATION_ONLY_CONTRACT,
        accumulation=None,
    )
    if receipt.get("run_fingerprint") != expected_fingerprint:
        raise ValueError("A100 qualification run fingerprint/config/bank identities disagree.")
    if (
        receipt.get("selection_rule_sha256", expected_selection_rule_sha256)
        != expected_selection_rule_sha256
        or receipt.get("decoder_activation_checkpoint_sha256")
        != sha256_json(anatomy["checkpoint_evidence"])
        or receipt.get("frozen_decoder_state_sha256")
        != anatomy["decoder_state_sha256_before"]
    ):
        raise ValueError("A100 qualification decoder/selection provenance changed.")
    return {
        "status": "pass",
        "receipt": str(receipt_path),
        "receipt_file_sha256": sha256_file(receipt_path),
        "run_fingerprint": expected_fingerprint,
        "resolved_config_file_sha256": sha256_file(resolved_config),
        "frozen_decoder_state_sha256": receipt["frozen_decoder_state_sha256"],
        "decoder_activation_checkpoint_sha256": receipt[
            "decoder_activation_checkpoint_sha256"
        ],
        "a100_memory_gate_status": "pass",
        "anatomy_qualification_status": "pass",
    }


def _verify_completed_validation_plan(
    path: Path,
    *,
    validation_index: PhotometryFactoredLatentBankIndex,
    expected_validation_plan_sha256: str,
) -> dict[str, Any]:
    plan = _load_self_hashed(path, "validation_plan_sha256")
    entries = plan.get("entries")
    if (
        plan.get("contract_version") != UNIFIED_VALIDATION_PLAN_CONTRACT
        or plan.get("validation_plan_sha256") != expected_validation_plan_sha256
        or plan.get("scope") != "complete_R_validation_inventory"
        or plan.get("required_directed_domain_cell_count") != 60
        or plan.get("bank_artifact_sha256") != validation_index.artifact_sha256
        or not isinstance(entries, list)
        or any(not isinstance(entry, Mapping) for entry in entries)
        or any(
            str(entry.get(role, "")).startswith("P:")
            for entry in entries
            if isinstance(entry, Mapping)
            for role in (
                "source_subject_group_identity",
                "target_subject_group_identity",
            )
        )
    ):
        raise ValueError("Completed pilot validation plan is incomplete or changed.")
    rebuilt = build_unified_validation_plan(
        validation_index, validation_seed=int(plan["validation_seed"])
    )
    if rebuilt != plan:
        raise ValueError("Completed pilot validation plan differs from the restored bank.")
    return plan


def _load_completed_stage2_config(run_dir: Path) -> tuple[Path, dict[str, Any]]:
    path = _assert_path_inside(
        run_dir, run_dir / "resolved_config.json", label="resolved config"
    )
    payload = _read_json(path)
    config = UnifiedStage2Config.from_mapping(payload).to_dict()
    return path, config


def _rebuild_stage2_run_fingerprint(
    config: Mapping[str, Any],
    *,
    train_index: PhotometryFactoredLatentBankIndex,
    validation_index: PhotometryFactoredLatentBankIndex,
    stats: FactoredLatentStats,
    validation_plan: Mapping[str, Any],
    frozen_decoder_state_sha256: str,
    decoder_checkpoint_evidence: Mapping[str, Any],
    git_commit: str,
    execution_mode: str,
    accumulation: Mapping[str, Any] | None,
) -> str:
    if _SHA256_RE.fullmatch(frozen_decoder_state_sha256) is None:
        raise ValueError("Frozen decoder state identity is invalid.")
    if accumulation is None:
        accumulation_identity = _expected_accumulation_identity()
    else:
        accumulation_identity = dict(accumulation)
        accumulation_identity.pop("joint_six_term_gradient_probe", None)
        if accumulation_identity != _expected_accumulation_identity():
            raise ValueError("Completed pilot accumulation contract changed.")
    run_identity = {
        "contract_version": UNIFIED_STAGE2_CONTRACT,
        "config": dict(config),
        "bank_artifact_sha256": train_index.artifact_sha256,
        "validation_bank_artifact_sha256": validation_index.artifact_sha256,
        "validation_inventory_sha256": sha256_json(
            [item.resume_key for item in validation_index.records]
        ),
        "validation_plan_sha256": validation_plan["validation_plan_sha256"],
        "selection_rule_sha256": UNIFIED_SELECTION_RULE_SHA256,
        "latent_statistics_sha256": stats.artifact_sha256,
        "bank_vae_provenance": dict(train_index.manifest.get("vae", {})),
        "frozen_decoder_state_sha256": frozen_decoder_state_sha256,
        "decoder_activation_checkpoint": dict(decoder_checkpoint_evidence),
        "decoder_activation_checkpoint_sha256": sha256_json(
            decoder_checkpoint_evidence
        ),
        "generator_gradient_accumulation": accumulation_identity,
        "execution_mode": execution_mode,
        "git_commit": git_commit,
    }
    return sha256_json(run_identity)


def _expected_accumulation_identity() -> dict[str, Any]:
    return {
        "contract_version": UNIFIED_GENERATOR_ACCUMULATION_CONTRACT,
        "term_order": list(DEFAULT_UNIFIED_WEIGHTS),
        "graph_construction": "one_term_at_a_time",
        "forward_backward_interleaved": True,
        "retain_graph": False,
        "graph_release": "before_next_term",
        "gradient_measurement": "inline_during_term_backward",
        "gradient_measurement_contract": UNIFIED_TERM_GRADIENT_QUALIFICATION_CONTRACT,
        "gradient_measurement_scope": "pilot_steps_only",
        "long_run_hook_measurement": "disabled_after_pilot",
        "saved_tensor_policy": "save_on_cpu",
        "translator_checkpoint_contract": UNIFIED_TRANSLATOR_CHECKPOINT_CONTRACT,
        "translator_checkpoint_use_reentrant": False,
        "translator_checkpoint_preserve_rng_state": True,
        "frozen_step_plan_replayed_per_term": True,
        "generator_optimizer_updates_per_step": 1,
    }


def _assert_completed_pilot_report(pilot: Any, *, steps: int) -> None:
    if not isinstance(pilot, Mapping):
        raise ValueError(f"Step-{steps} checkpoint lacks a pilot report.")
    term_gradients = pilot.get("term_gradient_norms")
    raw_terms = pilot.get("raw_term_means")
    weighted_terms = pilot.get("weighted_term_means")
    if (
        pilot.get("status") != "pass"
        or pilot.get("failures") != []
        or pilot.get("steps") != steps
        or pilot.get("full_objective") is not True
        or not isinstance(term_gradients, Mapping)
        or not isinstance(raw_terms, Mapping)
        or not isinstance(weighted_terms, Mapping)
        or set(term_gradients) != set(DEFAULT_UNIFIED_WEIGHTS)
        or set(raw_terms) != set(DEFAULT_UNIFIED_WEIGHTS)
        or set(weighted_terms) != set(DEFAULT_UNIFIED_WEIGHTS)
    ):
        raise ValueError(f"Step-{steps} full-objective pilot did not pass all six terms.")
    for term in DEFAULT_UNIFIED_WEIGHTS:
        item = term_gradients[term]
        if (
            not isinstance(item, Mapping)
            or item.get("enabled") is not True
            or not all(
                math.isfinite(float(item.get(key, float("nan"))))
                for key in ("mean", "minimum", "maximum")
            )
            or float(item["maximum"]) <= 0
        ):
            raise ValueError(f"Step-{steps} objective {term!r} was not gradient-qualified.")
    if pilot.get("decoder_activation_checkpoint") != {
        "contract_version": "klvae3d-full-volume-fine-grained-activation-checkpoint-v1",
        "mode": "fine_grained_full_volume_v1",
        "outer_full_decoder_checkpoint": False,
    }:
        raise ValueError(f"Step-{steps} decoder checkpoint mode changed.")
    accumulation = dict(pilot.get("generator_gradient_accumulation", {}))
    if accumulation.pop("joint_six_term_gradient_probe", None) is not False or (
        accumulation != _expected_accumulation_identity()
    ):
        raise ValueError(f"Step-{steps} six-term accumulation contract changed.")
    _assert_a100_gate(pilot.get("one_step_a100_memory_gate"))
    _assert_anatomy_qualification(pilot.get("one_step_anatomy_memory_qualification"))


def _assert_a100_gate(gate: Any) -> None:
    if not isinstance(gate, Mapping) or (
        gate.get("contract_version") != UNIFIED_A100_GATE_CONTRACT
        or gate.get("status") != "pass"
        or gate.get("gpu_identity_matches") is not True
        or gate.get("within_allocated_limit") is not True
        or gate.get("peak_allocated_limit_bytes")
        != UNIFIED_A100_PEAK_ALLOCATED_LIMIT_BYTES
        or type(gate.get("peak_allocated_bytes")) is not int
        or not 0
        < gate["peak_allocated_bytes"]
        <= UNIFIED_A100_PEAK_ALLOCATED_LIMIT_BYTES
        or gate.get("full_objective") is not True
    ):
        raise ValueError("Completed evidence lacks a passing one-step A100 memory gate.")


def _assert_anatomy_qualification(anatomy: Any) -> None:
    if not isinstance(anatomy, Mapping):
        raise ValueError("Completed evidence lacks anatomy qualification.")
    checkpoint = anatomy.get("checkpoint_evidence")
    before = anatomy.get("decoder_state_sha256_before")
    if (
        anatomy.get("contract_version") != UNIFIED_ANATOMY_MEMORY_CONTRACT
        or anatomy.get("status") != "pass"
        or anatomy.get("decoder_state_unchanged") is not True
        or anatomy.get("decoder_state_sha256_after") != before
        or _SHA256_RE.fullmatch(str(before)) is None
        or anatomy.get("source_decode_checkpointed") is not False
        or anatomy.get("spatial_crop_or_tile") is not False
        or anatomy.get("allocator_fallback") is not False
        or not isinstance(checkpoint, Mapping)
        or checkpoint.get("mode") != "fine_grained_full_volume_v1"
        or checkpoint.get("full_volume") is not True
        or checkpoint.get("group_norm_scope") != "complete_spatial_volume"
        or checkpoint.get("upsample_regions") != ["up1", "up2"]
        or not checkpoint.get("residual_branch_regions")
        or checkpoint.get("outer_full_decoder_checkpoint") is not False
        or anatomy.get("checkpoint_evidence_sha256") != sha256_json(checkpoint)
    ):
        raise ValueError(
            "Completed evidence lacks a passing unchanged-decoder qualification."
        )


def _verify_completed_history(
    path: Path,
    *,
    steps: int,
    run_fingerprint: str,
    validation_plan_sha256: str,
    pilot: Mapping[str, Any],
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Completed pilot history is missing: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line:
            continue
        raw = json.loads(line)
        if not isinstance(raw, Mapping):
            raise ValueError(f"Completed pilot history row {line_number} is malformed.")
        row = dict(raw)
        if (
            row.get("contract_version") != UNIFIED_HISTORY_CONTRACT
            or row.get("run_fingerprint") != run_fingerprint
        ):
            raise ValueError("Completed pilot history contract/run fingerprint changed.")
        rows.append(row)
    if not rows or any(row.get("event") == "oom_hard_stop" for row in rows):
        raise ValueError("Completed pilot history is empty or contains a hard stop.")
    training_rows = [row for row in rows if "event" not in row]
    if [row.get("step") for row in training_rows] != list(range(1, steps + 1)):
        raise ValueError(
            f"Completed pilot history is not exactly complete through step {steps}."
        )
    for row in training_rows:
        for term in DEFAULT_UNIFIED_WEIGHTS:
            keys = (f"raw/{term}", f"weighted/{term}", f"gradient/term_{term}")
            if any(key not in row or not math.isfinite(float(row[key])) for key in keys):
                raise ValueError(f"Completed pilot history lacks finite evidence for {term}.")
    pilots = [row for row in rows if row.get("event") == "full_objective_pilot"]
    validations = [
        row
        for row in rows
        if row.get("event") == "unpaired_validation" and row.get("step") == steps
    ]
    if (
        len(pilots) != 1
        or pilots[0].get("step") != steps
        or pilots[0].get("pilot") != dict(pilot)
        or pilots[0].get("validation_plan_sha256") != validation_plan_sha256
        or len(validations) != 1
        or not isinstance(validations[0].get("validation"), Mapping)
        or validations[0]["validation"].get("complete_inventory_used") is not True
        or validations[0]["validation"].get("validation_plan_sha256")
        != validation_plan_sha256
    ):
        raise ValueError(
            "Completed pilot history lacks its final pilot/validation evidence."
        )
    return {
        "training_row_count": len(training_rows),
        "event_row_count": len(rows) - len(training_rows),
    }


def _unique_file(root: Path, pattern: str, *, label: str) -> Path:
    matches = sorted(path.resolve() for path in root.glob(pattern) if path.is_file())
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {label}; found {len(matches)}.")
    return _assert_path_inside(root, matches[0], label=label)


def _assert_path_inside(root: Path, path: Path, *, label: str) -> Path:
    resolved_root = root.resolve()
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing {label}: {path}") from exc
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes its immutable run namespace.") from exc
    if path.is_symlink() or not resolved.is_file():
        raise ValueError(f"{label} is not a regular immutable file.")
    return resolved


def _case_receipt(
    case: Gate01Case, calibrator: PosthocTargetCalibrator
) -> dict[str, Any]:
    if (
        case.traveller_identity_sha256 != P0006_IDENTITY_SHA256
        or case.traveller_identity_sha256 in FORBIDDEN_OTHER_TRAVELLER_HASHES
        or case.source_image is None
    ):
        raise ValueError("Gate 0.1 case is not exclusively the reviewed P:0006 identity.")
    calibrated = calibrator.apply(
        case.raw_identity, case.target_domain, support_mask=case.support_mask
    )
    required = {
        "source_image",
        "source_support_mask",
        "target",
        "raw_identity",
        "raw_sb_v2",
        "stage1_reconstruction_ceiling",
    }
    if not required.issubset(case.array_sha256):
        raise ValueError("P:0006 case lacks required source/target/baseline identities.")
    wrong = {
        label: case.array_sha256[f"wrong_target_sb_v2[{label}]"]
        for label in sorted(case.wrong_target_sb_v2)
    }
    if len(wrong) != 3:
        raise ValueError("Each P:0006 edge requires all three wrong-target references.")
    return {
        "case_identity": case.case_id,
        "source_domain": case.source_domain.to_dict(),
        "target_domain": case.target_domain.to_dict(),
        "source_image_sha256": case.array_sha256["source_image"],
        "source_support_sha256": case.array_sha256["source_support_mask"],
        "target_sha256": case.array_sha256["target"],
        "raw_identity_sha256": case.array_sha256["raw_identity"],
        "calibrated_identity_sha256": canonical_tensor_sha256(calibrated),
        "original_sb_v2_sha256": case.array_sha256["raw_sb_v2"],
        "stage1_reconstruction_ceiling_sha256": case.array_sha256[
            "stage1_reconstruction_ceiling"
        ],
        "wrong_target_sb_v2_sha256": wrong,
    }


def _assert_gate_arrays_inside_archive(manifest: Any, archive_root: Path) -> None:
    manifest_root = getattr(manifest, "root", None)
    case_specs = getattr(manifest, "case_specs", None)
    if not isinstance(manifest_root, Path) or not isinstance(case_specs, tuple):
        raise ValueError("Gate 0.1 importer requires the metadata-only manifest contract.")
    root = archive_root.resolve()
    for case in case_specs:
        references = [
            case[key]
            for key in (
                "source_image",
                "source_support_mask",
                "target",
                "raw_identity",
                "raw_sb_v2",
                "stage1_reconstruction_ceiling",
            )
        ]
        references.extend(dict(case.get("wrong_target_sb_v2", {})).values())
        for reference in references:
            if not isinstance(reference, Mapping):
                raise ValueError("Gate 0.1 archive contains a malformed array reference.")
            raw_path = Path(str(reference.get("path", "")))
            path = (
                raw_path.resolve()
                if raw_path.is_absolute()
                else (manifest_root / raw_path).resolve()
            )
            try:
                path.relative_to(root)
            except ValueError as exc:
                raise ValueError(
                    "Gate 0.1 manifest references an array outside the reviewed archive."
                ) from exc
            if not path.is_file():
                raise FileNotFoundError(f"Gate 0.1 archived array is missing: {path}")


def _as_channel_volume(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 3:
        return value.unsqueeze(0)
    if value.ndim == 4 and value.shape[0] == 1:
        return value
    raise ValueError(
        "P:0006 evaluation arrays must be full 3-D volumes with one explicit or "
        "implicit channel."
    )

def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected a JSON object: {path}")
    return dict(payload)


def _load_self_hashed(path: str | Path, hash_key: str) -> dict[str, Any]:
    payload = _read_json(Path(path))
    body = dict(payload)
    stored = body.pop(hash_key, None)
    if stored != sha256_json(body):
        raise ValueError(f"Self-hash mismatch for {Path(path).name}.")
    return payload


def _verified_protocol_file(protocol: Mapping[str, Any], key: str) -> Path:
    spec = protocol.get(key)
    if not isinstance(spec, Mapping):
        raise ValueError(f"P:0006 protocol lacks {key} identity.")
    path = Path(str(spec.get("path", ""))).resolve()
    if not path.is_file() or sha256_file(path) != spec.get("file_sha256"):
        raise ValueError(f"P:0006 protocol dependency changed: {key}.")
    return path


__all__ = [
    "EXPECTED_STAGE2_SELECTION_RECEIPT_FILE_SHA256",
    "EXPECTED_STAGE2_SELECTION_RULE_SHA256",
    "EXPECTED_STAGE2_VALIDATION_PLAN_SHA256",
    "GATE01_ARCHIVE_LAYOUT_MODERN_FLAT_V1",
    "GATE01_ARCHIVE_LAYOUT_REVIEWED_LEGACY_JSON_V1",
    "GATE01_ARCHIVE_LAYOUT_REVIEWED_LEGACY_JSON_V2",
    "GATE01_ARCHIVE_PREFLIGHT_CONTRACT",
    "GATE01_P0006_EVALUATION_PROTOCOL",
    "GATE01_P0006_EVALUATION_PROTOCOL_V2",
    "GATE01_P0006_EVALUATION_PROTOCOL_V3",
    "P0006_DEVELOPMENT_VALIDATION_DATA_ROLE",
    "P0006_EVIDENCE_LIMITATION",
    "P0006_IDENTITY_SHA256",
    "P0006_SUBJECT_GROUP",
    "P0006_COUNT_PROGRESS_ENV",
    "P0006_COUNT_PROGRESS_VALUE",
    "P0009_CONFIRMATION_STATUS",
    "REVIEWED_GATE01_RESULT_FILE_SHA256",
    "STAGE2_BANK_MANIFEST_FILENAME",
    "STAGE2_BANK_TAR_RESTORE_CONTRACT",
    "STAGE2_COMPLETED_PILOT_REUSE_CONTRACT",
    "STAGE2_LOCAL_ARCHIVE_COPY_CONTRACT",
    "STAGE2_LOCAL_DISK_PREFLIGHT_CONTRACT",
    "STAGE2_LOCAL_DISK_RESERVE_BYTES",
    "STAGE2_RECOVERY_DRIVE_LAYOUT_CONTRACT",
    "STAGE2_RECOVERY_OUTPUT_ROOT_NAME",
    "Stage2LocalDiskCapacityPreflight",
    "Stage2RecoveryDriveLayout",
    "SUPPORTED_GATE01_P0006_EVALUATION_PROTOCOLS",
    "TRAINING_EVIDENCE_COMMIT",
    "VerifiedStage2LocalArchive",
    "copy_verified_stage2_bank_tar_to_local",
    "import_gate01_p0006_evaluation_protocol",
    "load_gate01_p0006_evaluation_protocol",
    "preflight_stage2_local_disk_capacity",
    "preflight_gate01_p0006_archive",
    "resolve_stage2_recovery_drive_layout",
    "restore_verified_stage2_bank_tar",
    "resolve_gate01_p0006_archive_layout",
    "stage2_bank_tree_identity",
    "verify_completed_stage2_pilot_evidence",
]
