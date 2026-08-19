"""Strict development-validation import of the sealed Gate01Private P:0006 graph.

This module does not train, select, or recompute a baseline model.  It verifies the
existing Gate 0.1 archive, materializes no prediction arrays, and exposes the frozen
P:0006 endpoints only to development/model assessment after readiness has been sealed.
This evidence cannot support population or generalization claims.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from fieldbridge.data.photometry_factored_bank_dataset import (
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
    UNIFIED_VALIDATION_PLAN_CONTRACT,
    build_unified_validation_plan,
)

GATE01_P0006_EVALUATION_PROTOCOL = (
    "stage2-unified-gate01-p0006-development-validation-evaluation-only-protocol-v2"
)
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


def import_gate01_p0006_evaluation_protocol(
    archive_root: str | Path,
    *,
    expected_gate01_result_sha256: str,
    bank_dir: str | Path,
    validation_plan_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Seal Gate01Private_8012a3f for P:0006 development/model assessment only."""

    root = Path(archive_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Gate 0.1 archive root does not exist: {root}")
    if len(expected_gate01_result_sha256) != 64:
        raise ValueError("Expected Gate 0.1 result identity must be SHA-256.")
    inventory_path = root / "sha256-inventory.csv"
    inventory = _verify_archive_inventory(root, inventory_path)
    result_path = _single_file_with_sha(root, expected_gate01_result_sha256)
    result = _read_json(result_path)
    if result.get("contract_version") != GATE01_CONTRACT_VERSION:
        raise ValueError("Expected Gate 0.1 result has an incompatible contract.")
    result_body = dict(result)
    stored_result_hash = result_body.pop("result_sha256", None)
    if stored_result_hash != sha256_json(result_body):
        raise ValueError("Gate 0.1 result self-hash mismatch.")

    manifest_path = _single_json_contract(root, GATE01_INPUT_CONTRACT_VERSION)
    lock_path = _single_json_contract(root, GATE01_PROTOCOL_LOCK_CONTRACT_VERSION)
    calibrator_path = _single_json_contract(root, GATE01_CALIBRATOR_CONTRACT_VERSION)
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
    if gate_metadata["execution_mode"] != "scientific":
        raise ValueError("P:0006 import requires the sealed scientific Gate 0.1 graph.")
    evidence = gate_metadata["evidence_scope"]
    if (
        evidence.get("private_data_run") is not True
        or evidence.get("evidence_kind") != "private"
        or evidence.get("traveller_identity_sha256") != P0006_IDENTITY_SHA256
    ):
        raise ValueError("Gate 0.1 archive is not the reviewed P:0006 private graph.")
    if lock.traveller_identity_sha256 != P0006_IDENTITY_SHA256:
        raise ValueError("Gate 0.1 protocol lock is not pinned to P:0006.")
    if lock.traveller_identity_sha256 in FORBIDDEN_OTHER_TRAVELLER_HASHES:
        raise ValueError("P:0007/P:0009 are forbidden from the P:0006 protocol.")

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
            "root_path_identity_sha256": sha256_text(str(root)),
            "inventory_file_sha256": sha256_file(inventory_path),
            "verified_inventory_entry_count": len(inventory),
        },
        "archive_inventory": {
            "path": str(inventory_path),
            "file_sha256": sha256_file(inventory_path),
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
    if protocol.get("contract_version") != GATE01_P0006_EVALUATION_PROTOCOL:
        raise ValueError("Unsupported P:0006 evaluation-only protocol contract.")
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


def _verify_archive_inventory(root: Path, path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError("Gate 0.1 archive lacks sha256-inventory.csv.")
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            lowered = {str(key).lower(): str(value) for key, value in raw.items()}
            digest = lowered.get("hash", "").lower()
            source_path = lowered.get("path", "")
            if len(digest) != 64 or not source_path:
                raise ValueError("Gate 0.1 archive inventory row is malformed.")
            candidate = root / Path(source_path.replace("\\", "/")).name
            if not candidate.is_file() or sha256_file(candidate) != digest:
                raise ValueError(
                    f"Gate 0.1 archived file is missing or changed: {candidate.name}"
                )
            rows.append({"name": candidate.name, "sha256": digest})
    if not rows or len({row["name"] for row in rows}) != len(rows):
        raise ValueError("Gate 0.1 archive inventory is empty or has duplicate basenames.")
    return sorted(rows, key=lambda item: item["name"])


def _single_json_contract(root: Path, contract: str) -> Path:
    candidates = []
    for path in root.glob("*.json"):
        try:
            payload = _read_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if payload.get("contract_version") == contract:
            candidates.append(path.resolve())
    if len(candidates) != 1:
        raise ValueError(
            f"Gate 0.1 archive requires exactly one {contract!r} JSON; found {len(candidates)}."
        )
    return candidates[0]


def _single_file_with_sha(root: Path, digest: str) -> Path:
    matches = [path.resolve() for path in root.iterdir() if path.is_file() and sha256_file(path) == digest]
    if len(matches) != 1:
        raise ValueError(
            "Gate 0.1 archive must contain exactly one file with the expected result SHA-256."
        )
    return matches[0]


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
    "GATE01_P0006_EVALUATION_PROTOCOL",
    "P0006_DEVELOPMENT_VALIDATION_DATA_ROLE",
    "P0006_EVIDENCE_LIMITATION",
    "P0006_IDENTITY_SHA256",
    "P0006_SUBJECT_GROUP",
    "P0009_CONFIRMATION_STATUS",
    "import_gate01_p0006_evaluation_protocol",
    "load_gate01_p0006_evaluation_protocol",
]
