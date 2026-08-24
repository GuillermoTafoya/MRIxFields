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
import re
from collections.abc import Mapping
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
GATE01_P0006_EVALUATION_PROTOCOL = (
    "stage2-unified-gate01-p0006-development-validation-evaluation-only-protocol-v3"
)
SUPPORTED_GATE01_P0006_EVALUATION_PROTOCOLS = frozenset(
    {GATE01_P0006_EVALUATION_PROTOCOL_V2, GATE01_P0006_EVALUATION_PROTOCOL}
)
GATE01_ARCHIVE_LAYOUT_MODERN_FLAT_V1 = "gate01-archive-modern-flat-csv-v1"
GATE01_ARCHIVE_LAYOUT_REVIEWED_LEGACY_JSON_V1 = (
    "gate01-archive-reviewed-legacy-parent-json-v1"
)
GATE01_ARCHIVE_PREFLIGHT_CONTRACT = "gate01-p0006-metadata-preflight-v1"
STAGE2_COMPLETED_PILOT_REUSE_CONTRACT = "stage2-completed-pilot-reuse-verification-v1"
TRAINING_EVIDENCE_COMMIT = "82633d66e5ea47f96b149ea22cc192fcf4526f06"
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
    GATE01_ARCHIVE_LAYOUT_REVIEWED_LEGACY_JSON_V1: _GATE01_REQUIRED_COMMON
    | {
        "colab-operational-source-split.json",
        "reviewed-module-hashes.json",
    },
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
    """One inventory row verified against a direct logical-root child."""

    basename: str
    sha256: str
    size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "basename": self.basename,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class Gate01ArchiveLayout:
    """Resolved, fully verified Gate 0.1 logical archive contract."""

    logical_root: Path
    inventory_path: Path
    layout_contract: str
    inventory_format: Literal["csv", "reviewed_legacy_json"]
    entries: tuple[VerifiedGate01InventoryEntry, ...]
    inventory_file_sha256: str
    normalized_inventory_sha256: str

    def path_for(self, basename: str) -> Path:
        matches = [entry for entry in self.entries if entry.basename == basename]
        if len(matches) != 1:
            raise ValueError(
                f"Gate 0.1 inventory requires exactly one {basename!r}; found {len(matches)}."
            )
        return _verified_direct_child(self.logical_root, basename)

    def file_with_sha256(self, digest: str) -> Path:
        matches = [entry for entry in self.entries if entry.sha256 == digest]
        if len(matches) != 1:
            raise ValueError(
                "Gate 0.1 inventory must identify exactly one file with the expected "
                "result SHA-256."
            )
        return _verified_direct_child(self.logical_root, matches[0].basename)

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
            "stored_inventory_paths_trusted_for_file_access": False,
            "file_access_derivation": "verified_basename_as_direct_logical_root_child",
        }


@dataclass(frozen=True, slots=True)
class _Gate01MetadataPreflight:
    layout: Gate01ArchiveLayout
    result_path: Path
    result: dict[str, Any]
    stored_result_hash: str
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

    root = Path(archive_root).resolve()
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
                GATE01_ARCHIVE_LAYOUT_REVIEWED_LEGACY_JSON_V1,
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
    else:
        entries = _verify_legacy_json_archive_inventory(root, inventory_path)
    required = _GATE01_REQUIRED_BY_LAYOUT[layout_contract]
    observed = {entry.basename for entry in entries}
    missing = sorted(required - observed)
    if missing:
        raise ValueError(
            "Gate 0.1 inventory lacks required scientific dependencies: "
            + ", ".join(missing)
        )
    normalized = [entry.to_dict() for entry in entries]
    return Gate01ArchiveLayout(
        logical_root=root,
        inventory_path=inventory_path.resolve(),
        layout_contract=layout_contract,
        inventory_format=inventory_format,
        entries=entries,
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
        "expected_gate01_result_sha256": expected_gate01_result_sha256,
        "expected_gate01_result_match": True,
        "stored_absolute_inventory_paths_trusted_for_file_access": False,
        "private_array_payloads_opened": 0,
        "required_scientific_contracts_unique": True,
        "logical_root_containment_verified": True,
    }
    receipt["preflight_sha256"] = sha256_json(receipt)
    return receipt


def _preflight_gate01_p0006_archive(
    archive_root: str | Path,
    *,
    expected_gate01_result_sha256: str,
) -> _Gate01MetadataPreflight:
    if _SHA256_RE.fullmatch(expected_gate01_result_sha256) is None:
        raise ValueError("Expected Gate 0.1 result identity must be lowercase SHA-256.")
    layout = resolve_gate01_p0006_archive_layout(archive_root)
    root = layout.logical_root
    result_path = layout.file_with_sha256(expected_gate01_result_sha256)
    if result_path.name != "gate01-results.json":
        raise ValueError("Expected Gate 0.1 result SHA resolved to the wrong dependency.")
    result = _read_json(result_path)
    if result.get("contract_version") != GATE01_CONTRACT_VERSION:
        raise ValueError("Expected Gate 0.1 result has an incompatible contract.")
    result_body = dict(result)
    stored_result_hash = result_body.pop("result_sha256", None)
    if stored_result_hash != sha256_json(result_body):
        raise ValueError("Gate 0.1 result self-hash mismatch.")

    manifest_path = _single_inventory_json_contract(
        layout, GATE01_INPUT_CONTRACT_VERSION
    )
    lock_path = _single_inventory_json_contract(
        layout, GATE01_PROTOCOL_LOCK_CONTRACT_VERSION
    )
    calibrator_path = _single_inventory_json_contract(
        layout, GATE01_CALIBRATOR_CONTRACT_VERSION
    )
    expected_names = {
        manifest_path.name: "gate01-private-manifest.json",
        lock_path.name: "gate01-protocol-lock.json",
        calibrator_path.name: "gate01-target-calibrator.json",
    }
    if any(observed != expected for observed, expected in expected_names.items()):
        raise ValueError("Gate 0.1 required contract resolved to an unexpected basename.")
    for basename in _GATE01_REQUIRED_BY_LAYOUT[layout.layout_contract]:
        candidate = layout.path_for(basename)
        if candidate.suffix == ".json":
            _read_json(candidate)

    lock = Gate01ProtocolLock.load(lock_path)
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
        stored_result_hash=str(stored_result_hash),
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
            rows.append(VerifiedGate01InventoryEntry(basename, digest, size))
    return _normalize_inventory_rows(rows)


def _verify_legacy_json_archive_inventory(
    root: Path, path: Path
) -> tuple[VerifiedGate01InventoryEntry, ...]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list):
        raise ValueError("Reviewed legacy Gate 0.1 inventory must be a top-level JSON list.")
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
        basename = _inventory_basename(stored_path)
        candidate = _verified_direct_child(root, basename)
        if candidate.stat().st_size != size:
            raise ValueError(f"Gate 0.1 archived file size mismatch: {basename}")
        if sha256_file(candidate) != digest:
            raise ValueError(f"Gate 0.1 archived file hash mismatch: {basename}")
        rows.append(VerifiedGate01InventoryEntry(basename, digest, size))
    return _normalize_inventory_rows(rows)


def _normalize_inventory_rows(
    rows: list[VerifiedGate01InventoryEntry],
) -> tuple[VerifiedGate01InventoryEntry, ...]:
    if not rows:
        raise ValueError("Gate 0.1 archive inventory is empty.")
    folded = [entry.basename.casefold() for entry in rows]
    if len(set(folded)) != len(folded):
        raise ValueError("Gate 0.1 archive inventory has duplicate basenames.")
    return tuple(sorted(rows, key=lambda item: item.basename))


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
    if path.is_symlink():
        raise ValueError(f"Gate 0.1 {label} may not be a symlink.")
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


def _single_inventory_json_contract(
    layout: Gate01ArchiveLayout, contract: str
) -> Path:
    candidates: list[Path] = []
    for entry in layout.entries:
        if not entry.basename.endswith(".json"):
            continue
        path = _verified_direct_child(layout.logical_root, entry.basename)
        try:
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


def import_gate01_p0006_evaluation_protocol(
    archive_root: str | Path,
    *,
    expected_gate01_result_sha256: str,
    bank_dir: str | Path,
    validation_plan_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Seal Gate01Private_8012a3f for P:0006 development/model assessment only."""

    metadata = _preflight_gate01_p0006_archive(
        archive_root,
        expected_gate01_result_sha256=expected_gate01_result_sha256,
    )
    layout = metadata.layout
    root = layout.logical_root
    result_path = metadata.result_path
    stored_result_hash = metadata.stored_result_hash
    manifest_path = metadata.manifest_path
    lock_path = metadata.lock_path
    calibrator_path = metadata.calibrator_path
    lock = metadata.lock
    calibrator = metadata.calibrator
    gate_manifest = metadata.gate_manifest
    gate_metadata = metadata.gate_metadata

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

    cases = list(gate_manifest)
    case_receipts = [_case_receipt(case, calibrator) for case in cases]
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
            "file_access_derivation": "verified_basename_as_direct_logical_root_child",
        },
        "gate01_result": {
            "path": str(result_path),
            "file_sha256": expected_gate01_result_sha256,
            "result_sha256": stored_result_hash,
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
    return body


def load_gate01_p0006_evaluation_protocol(
    path: str | Path,
) -> tuple[
    dict[str, Any],
    tuple[PairedEvaluationCase, ...],
    dict[str, dict[str, torch.Tensor]],
]:
    """Revalidate the sealed P:0006 graph and stream its full-volume arrays."""

    protocol = _load_self_hashed(path, "protocol_sha256")
    if protocol.get("contract_version") not in SUPPORTED_GATE01_P0006_EVALUATION_PROTOCOLS:
        raise ValueError("Unsupported P:0006 evaluation-only protocol contract.")
    archive_root = (
        _reverify_v3_archive_provenance(protocol)
        if protocol["contract_version"] == GATE01_P0006_EVALUATION_PROTOCOL
        else None
    )
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
    calibrator = PosthocTargetCalibrator.load(
        calibrator_path,
        expected_split_fingerprint=lock.split_fingerprint,
        expected_template_sha256=lock.calibrator_template_sha256,
        expected_artifact_sha256=lock.calibrator_artifact_sha256,
    )
    gate_manifest, metadata = load_gate01_input_manifest(
        manifest_path, protocol_lock=lock, calibrator=calibrator
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
    for gate_case in gate_manifest:
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
    if expected_receipts or len(cases) != 60:
        raise ValueError("P:0006 evaluation case inventory is incomplete or changed.")
    return protocol, tuple(cases), baselines


def _reverify_v3_archive_provenance(protocol: Mapping[str, Any]) -> Path:
    identity = protocol.get("archive_identity")
    inventory = protocol.get("archive_inventory")
    if not isinstance(identity, Mapping) or not isinstance(inventory, Mapping):
        raise ValueError("P:0006 v3 protocol lacks archive-layout provenance.")
    root_text = identity.get("logical_root")
    if not isinstance(root_text, str) or not root_text:
        raise ValueError("P:0006 v3 protocol lacks the logical archive root.")
    root = Path(root_text).resolve()
    layout = resolve_gate01_p0006_archive_layout(root)
    entries = [entry.to_dict() for entry in layout.entries]
    if (
        identity.get("root_path_identity_sha256") != sha256_text(str(root))
        or identity.get("layout_contract") != layout.layout_contract
        or identity.get("inventory_format") != layout.inventory_format
        or identity.get("inventory_file_sha256") != layout.inventory_file_sha256
        or identity.get("verified_inventory_entry_count") != len(layout.entries)
        or identity.get("normalized_inventory_sha256")
        != layout.normalized_inventory_sha256
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
        != "verified_basename_as_direct_logical_root_child"
    ):
        raise ValueError("P:0006 v3 archive-layout provenance changed or is incomplete.")
    entry_map = {entry.basename: entry.sha256 for entry in layout.entries}
    for key in ("gate01_result", "gate01_manifest", "protocol_lock", "calibrator"):
        spec = protocol.get(key)
        if not isinstance(spec, Mapping):
            raise ValueError(f"P:0006 v3 protocol lacks {key} provenance.")
        dependency = Path(str(spec.get("path", ""))).resolve()
        try:
            dependency.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"P:0006 v3 dependency escapes logical root: {key}.") from exc
        if dependency.parent != root or entry_map.get(dependency.name) != spec.get(
            "file_sha256"
        ):
            raise ValueError(f"P:0006 v3 dependency is not the inventoried root child: {key}.")
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

    for label, digest in {
        "training evidence commit": training_evidence_commit,
        "selection receipt file": expected_selection_receipt_file_sha256,
        "validation plan": expected_validation_plan_sha256,
        "selection rule": expected_selection_rule_sha256,
    }.items():
        if _SHA256_RE.fullmatch(digest) is None:
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
    "GATE01_ARCHIVE_PREFLIGHT_CONTRACT",
    "GATE01_P0006_EVALUATION_PROTOCOL",
    "GATE01_P0006_EVALUATION_PROTOCOL_V2",
    "P0006_DEVELOPMENT_VALIDATION_DATA_ROLE",
    "P0006_EVIDENCE_LIMITATION",
    "P0006_IDENTITY_SHA256",
    "P0006_SUBJECT_GROUP",
    "P0009_CONFIRMATION_STATUS",
    "STAGE2_COMPLETED_PILOT_REUSE_CONTRACT",
    "SUPPORTED_GATE01_P0006_EVALUATION_PROTOCOLS",
    "TRAINING_EVIDENCE_COMMIT",
    "import_gate01_p0006_evaluation_protocol",
    "load_gate01_p0006_evaluation_protocol",
    "preflight_gate01_p0006_archive",
    "resolve_gate01_p0006_archive_layout",
    "verify_completed_stage2_pilot_evidence",
]
