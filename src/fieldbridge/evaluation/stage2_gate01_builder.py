"""Deterministic, resumable builder for external private Gate 0.1 manifests."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fieldbridge.data.domains import Domain
from fieldbridge.evaluation.mrixfields2026_official import load_official_nifti
from fieldbridge.evaluation.stage2_gate01 import (
    GATE01_INPUT_CONTRACT_VERSION,
    GATE01_VERIFIED_PRODUCER_RECEIPT_VERSION,
    canonical_loaded_array_sha256,
    load_gate01_input_manifest,
    validate_gate01_verified_producer_receipt,
)
from fieldbridge.evaluation.stage2_gate01_calibration import (
    FULL_LATENT_BANK_BUILD_COMMIT,
    FULL_LATENT_BANK_SOURCE_SPLIT_FILE_SHA256,
    FULL_LATENT_BANK_SOURCE_SPLIT_FINGERPRINT,
    RESPLIT_FINGERPRINT,
    SB_V2_CONFIG_SHA256,
    SB_V2_CHECKPOINT_SHA256,
    STAGE1_RUN_C_CONFIG_SHA256,
    STAGE1_RUN_C_CHECKPOINT_SHA256,
    PosthocTargetCalibrator,
)
from fieldbridge.evaluation.stage2_gate01_protocol import Gate01ProtocolLock

GATE01_PRIVATE_BUILD_PLAN_VERSION = "stage2-gate01-private-build-plan-v3"
GATE01_PRIVATE_BUILD_STATE_VERSION = "stage2-gate01-private-build-state-v3"
GATE01_PRIVATE_PRODUCER_SPEC_VERSION = "stage2-gate01-private-producer-spec-v4"
GATE01_PRIVATE_PRODUCER_STATE_VERSION = "stage2-gate01-private-producer-state-v3"
GATE01_PRIVATE_PRODUCER_RECEIPT_VERSION = "stage2-gate01-private-producer-receipt-v2"

_PLAN_KEYS = {
    "contract_version",
    "execution_mode",
    "selection_fingerprint_sha256",
    "evidence_scope",
    "split_fingerprint",
    "split_provenance",
    "artifact_provenance",
    "source_support_contract",
    "producer_receipt",
    "cases",
}
_CASE_KEYS = {
    "case_id",
    "traveller_identity_sha256",
    "source_domain",
    "target_domain",
    "source_image",
    "source_support_mask",
    "target",
    "raw_identity",
    "raw_sb_v2",
    "stage1_reconstruction_ceiling",
    "wrong_target_sb_v2",
}
_INPUT_REFERENCE_KEYS = {"path", "expected_sha256"}
_PRODUCER_SPEC_KEYS = {
    "contract_version",
    "selection_artifact_sha256",
    "selection_fingerprint_sha256",
    "traveller_identity_sha256",
    "split_provenance",
    "selected_source_acquisitions",
    "selected_payload_identity_set_sha256",
    "selected_payload_count",
    "latent_bank",
    "stage1_config_sha256",
    "stage1_checkpoint_sha256",
    "sb_v2_config_sha256",
    "sb_v2_checkpoint_sha256",
    "sampler",
    "decode",
    "deterministic_seed",
    "protocol_lock_artifact_sha256",
    "artifact_sha256",
}
_PRODUCER_RECEIPT_KEYS = {
    "contract_version",
    "producer_spec_artifact_sha256",
    "protocol_lock_artifact_sha256",
    "selection_artifact_sha256",
    "selection_fingerprint_sha256",
    "split_provenance",
    "selected_source_acquisitions_sha256",
    "selected_payload_identity_set_sha256",
    "selected_payload_count",
    "latent_bank",
    "stage1_config_sha256",
    "stage1_checkpoint_sha256",
    "sb_v2_config_sha256",
    "sb_v2_checkpoint_sha256",
    "sampler_specification_sha256",
    "decode_specification_sha256",
    "decode_strategy",
    "path_used",
    "acquisition_count",
    "stage1_inference_count",
    "direction_count",
    "sb_v2_inference_count",
    "wrong_target_reference_count",
}


def sealed_gate01_producer_receipt(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Derive the path-free producer receipt independently from a sealed spec."""

    bank = spec.get("latent_bank")
    if not isinstance(bank, Mapping):
        raise ValueError("Gate 0.1 producer latent-bank identity is malformed.")
    receipt = {
        "contract_version": GATE01_PRIVATE_PRODUCER_RECEIPT_VERSION,
        "producer_spec_artifact_sha256": spec.get("artifact_sha256"),
        "protocol_lock_artifact_sha256": spec.get(
            "protocol_lock_artifact_sha256"
        ),
        "selection_artifact_sha256": spec.get("selection_artifact_sha256"),
        "selection_fingerprint_sha256": spec.get(
            "selection_fingerprint_sha256"
        ),
        "split_provenance": spec.get("split_provenance"),
        "selected_source_acquisitions_sha256": _sha256_json(
            spec.get("selected_source_acquisitions")
        ),
        "selected_payload_identity_set_sha256": spec.get(
            "selected_payload_identity_set_sha256"
        ),
        "selected_payload_count": spec.get("selected_payload_count"),
        "latent_bank": dict(bank),
        "stage1_config_sha256": spec.get("stage1_config_sha256"),
        "stage1_checkpoint_sha256": spec.get("stage1_checkpoint_sha256"),
        "sb_v2_config_sha256": spec.get("sb_v2_config_sha256"),
        "sb_v2_checkpoint_sha256": spec.get("sb_v2_checkpoint_sha256"),
        "sampler_specification_sha256": _sha256_json(spec.get("sampler")),
        "decode_specification_sha256": _sha256_json(spec.get("decode")),
        "decode_strategy": "full",
        "path_used": ["full"],
        "acquisition_count": 15,
        "stage1_inference_count": 15,
        "direction_count": 60,
        "sb_v2_inference_count": 60,
        "wrong_target_reference_count": 180,
    }
    _assert_exact_keys(receipt, _PRODUCER_RECEIPT_KEYS, "Gate 0.1 producer receipt")
    return receipt


def _validate_producer_spec(
    raw: bytes, protocol_lock: Gate01ProtocolLock
) -> dict[str, Any]:
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Gate 0.1 producer specification root must be a mapping.")
    _assert_exact_keys(payload, _PRODUCER_SPEC_KEYS, "Gate 0.1 producer specification")
    if payload["contract_version"] != GATE01_PRIVATE_PRODUCER_SPEC_VERSION:
        raise ValueError("Gate 0.1 producer-spec contract is incompatible.")
    body = {key: payload[key] for key in _PRODUCER_SPEC_KEYS - {"artifact_sha256"}}
    if payload["artifact_sha256"] != _sha256_json(body):
        raise ValueError("Gate 0.1 producer-spec artifact hash mismatch.")
    expected = {
        "stage1_config_sha256": STAGE1_RUN_C_CONFIG_SHA256,
        "stage1_checkpoint_sha256": STAGE1_RUN_C_CHECKPOINT_SHA256,
        "sb_v2_config_sha256": SB_V2_CONFIG_SHA256,
        "sb_v2_checkpoint_sha256": SB_V2_CHECKPOINT_SHA256,
        "protocol_lock_artifact_sha256": protocol_lock.artifact_sha256,
        "selected_payload_count": 15,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("Gate 0.1 producer specification has stale frozen identities.")
    _validate_split_provenance(payload.get("split_provenance"), protocol_lock)
    sources = payload["selected_source_acquisitions"]
    if not isinstance(sources, Mapping) or len(sources) != 15:
        raise ValueError("Gate 0.1 producer spec must seal exactly 15 source arrays.")
    for source in sources.values():
        if (
            not isinstance(source, Mapping)
            or set(source) != {"canonical_loaded_array_sha256", "shape"}
            or not _is_sha256(str(source["canonical_loaded_array_sha256"]))
            or not isinstance(source["shape"], list)
            or len(source["shape"]) != 5
            or source["shape"][:2] != [1, 1]
            or any(not isinstance(value, int) or value <= 0 for value in source["shape"])
        ):
            raise ValueError("Gate 0.1 producer spec has a malformed source-array identity.")
    if not _is_sha256(str(payload["selected_payload_identity_set_sha256"])):
        raise ValueError("Gate 0.1 selected-payload identity set is invalid.")
    decode = payload["decode"]
    if (
        not isinstance(decode, Mapping)
        or set(decode) != {"strategy", "block_size", "halo", "precision"}
        or decode.get("strategy") != "full"
        or decode.get("precision") not in {"float32", "bfloat16"}
        or not isinstance(decode.get("block_size"), list)
        or len(decode["block_size"]) != 3
        or any(not isinstance(value, int) or value <= 0 for value in decode["block_size"])
        or not isinstance(decode.get("halo"), list)
        or len(decode["halo"]) != 3
        or any(not isinstance(value, int) or value < 0 for value in decode["halo"])
    ):
        raise ValueError("Gate 0.1 producer specification is not full-decode sealed.")
    sampler = payload["sampler"]
    if (
        not isinstance(sampler, Mapping)
        or set(sampler) != {"solver", "n_steps"}
        or sampler["solver"] not in {"euler", "heun"}
        or not isinstance(sampler["n_steps"], int)
        or sampler["n_steps"] < 1
    ):
        raise ValueError("Gate 0.1 producer sampler specification is invalid.")
    bank = payload["latent_bank"]
    if (
        not isinstance(bank, Mapping)
        or set(bank)
        != {
            "artifact_sha256",
            "manifest_sha256",
            "stats_sha256",
            "record_count",
            "build_git_commit",
            "vae_checkpoint_sha256",
            "encode_provenance",
        }
        or not all(
            _is_sha256(str(bank[key]))
            for key in (
                "artifact_sha256",
                "manifest_sha256",
                "stats_sha256",
                "vae_checkpoint_sha256",
            )
        )
        or not isinstance(bank["record_count"], int)
        or bank["record_count"] < 15
        or bank.get("build_git_commit") != FULL_LATENT_BANK_BUILD_COMMIT
        or bank.get("vae_checkpoint_sha256") != STAGE1_RUN_C_CHECKPOINT_SHA256
        or bank.get("encode_provenance")
        != {"strategy": "full", "path_used": ["full"]}
    ):
        raise ValueError("Gate 0.1 producer specification has a stale full latent bank.")
    return dict(payload)


def _validate_split_provenance(
    value: Any, protocol_lock: Gate01ProtocolLock
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"evaluation", "bank_storage"}:
        raise ValueError("Gate 0.1 split provenance must contain both split roles.")
    expected = protocol_lock.split_provenance
    if dict(value) != expected:
        raise ValueError("Gate 0.1 split provenance disagrees with the protocol lock.")
    evaluation = value["evaluation"]
    bank = value["bank_storage"]
    if (
        not isinstance(evaluation, Mapping)
        or not isinstance(bank, Mapping)
    ):
        raise ValueError("Gate 0.1 split provenance is stale or malformed.")
    if (
        set(evaluation) != {"role", "file_sha256", "membership_fingerprint"}
        or set(bank) != {"role", "file_sha256", "membership_fingerprint"}
        or evaluation["membership_fingerprint"] != RESPLIT_FINGERPRINT
        or bank["file_sha256"] != FULL_LATENT_BANK_SOURCE_SPLIT_FILE_SHA256
        or bank["membership_fingerprint"]
        != FULL_LATENT_BANK_SOURCE_SPLIT_FINGERPRINT
    ):
        raise ValueError("Gate 0.1 split provenance is stale or malformed.")
    return {str(key): dict(item) for key, item in value.items()}


def _validate_producer_state(
    raw: bytes,
    *,
    plan: Mapping[str, Any],
    plan_sha256: str,
    producer_spec: Mapping[str, Any],
    protocol_lock: Gate01ProtocolLock,
) -> dict[str, Any]:
    state = json.loads(raw.decode("utf-8"))
    if not isinstance(state, Mapping):
        raise ValueError("Gate 0.1 producer state root must be a mapping.")
    required = {
        "contract_version",
        "producer_spec_artifact_sha256",
        "protocol_lock_artifact_sha256",
        "producer_provenance",
        "split_provenance",
        "status",
        "completed",
        "pending",
        "counts",
        "build_plan_sha256",
        "acquisition_count",
        "direction_count",
        "wrong_target_reference_count",
        "producer_receipt",
    }
    _assert_exact_keys(state, required, "Gate 0.1 producer state")
    expected_receipt = sealed_gate01_producer_receipt(producer_spec)
    if (
        state["contract_version"] != GATE01_PRIVATE_PRODUCER_STATE_VERSION
        or state["status"] != "complete"
        or state["pending"] != {}
        or state["build_plan_sha256"] != plan_sha256
        or state["producer_spec_artifact_sha256"]
        != producer_spec["artifact_sha256"]
        or state["protocol_lock_artifact_sha256"] != protocol_lock.artifact_sha256
        or state["split_provenance"] != protocol_lock.split_provenance
        or state["producer_provenance"]
        != {"decode_strategy": "full", "path_used": ["full"]}
        or state["counts"]
        != {"stage1_inference_count": 15, "sb_v2_inference_count": 60}
        or state["acquisition_count"] != 15
        or state["direction_count"] != 60
        or state["wrong_target_reference_count"] != 180
        or state["producer_receipt"] != expected_receipt
        or plan.get("producer_receipt") != expected_receipt
    ):
        raise ValueError("Gate 0.1 producer state/plan/spec provenance is incompatible.")
    completed = state["completed"]
    if not isinstance(completed, Mapping) or len(completed) != 105:
        raise ValueError("Gate 0.1 producer state must contain exactly 105 completed arrays.")
    return dict(state)


def _validate_plan_against_producer(
    plan: Mapping[str, Any],
    *,
    producer_spec: Mapping[str, Any],
    producer_state: Mapping[str, Any],
) -> None:
    if (
        plan.get("selection_fingerprint_sha256")
        != producer_spec["selection_fingerprint_sha256"]
        or plan.get("split_fingerprint")
        != producer_spec["split_provenance"]["evaluation"]["membership_fingerprint"]
        or plan.get("split_provenance") != producer_spec["split_provenance"]
    ):
        raise ValueError("Gate 0.1 producer spec and plan selection/split differ.")
    expected_completed: set[tuple[str, str, bool]] = set()
    source_hashes: dict[str, str] = {}
    cases = plan.get("cases")
    if not isinstance(cases, list) or len(cases) != 60:
        raise ValueError("Gate 0.1 producer plan must contain exactly 60 directions.")
    for case in cases:
        if not isinstance(case, Mapping):
            raise ValueError("Gate 0.1 producer plan case is malformed.")
        source_label = Domain.from_dict(dict(case["source_domain"])).label
        source_hashes[source_label] = str(case["source_image"]["expected_sha256"])
        for role in (
            "source_image",
            "source_support_mask",
            "target",
            "raw_identity",
            "raw_sb_v2",
            "stage1_reconstruction_ceiling",
        ):
            reference = case[role]
            expected_completed.add(
                (
                    str(reference["path"]),
                    str(reference["expected_sha256"]),
                    role == "source_support_mask",
                )
            )
        for reference in dict(case.get("wrong_target_sb_v2", {})).values():
            expected_completed.add(
                (str(reference["path"]), str(reference["expected_sha256"]), False)
            )
    if len(expected_completed) != 105:
        raise ValueError("Gate 0.1 producer plan does not contain the exact 15/15/15/60 arrays.")
    completed_entries: set[tuple[str, str, bool]] = set()
    for value in producer_state["completed"].values():
        if not isinstance(value, Mapping) or set(value) != {
            "relative_path",
            "canonical_loaded_array_sha256",
            "mask",
        }:
            raise ValueError("Gate 0.1 producer completed-array receipt is malformed.")
        completed_entries.add(
            (
                str(value["relative_path"]),
                str(value["canonical_loaded_array_sha256"]),
                value["mask"] is True,
            )
        )
    if completed_entries != expected_completed:
        raise ValueError("Gate 0.1 producer state and build-plan array identities differ.")
    spec_sources = producer_spec["selected_source_acquisitions"]
    if set(source_hashes) != set(spec_sources) or any(
        source_hashes[label] != spec_sources[label]["canonical_loaded_array_sha256"]
        for label in source_hashes
    ):
        raise ValueError("Gate 0.1 producer source-array identities do not agree.")


def _verified_producer_receipt(
    *,
    plan_raw: bytes,
    producer_spec_raw: bytes,
    producer_state_raw: bytes,
    producer_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "contract_version": GATE01_VERIFIED_PRODUCER_RECEIPT_VERSION,
        "producer_spec_contract_version": GATE01_PRIVATE_PRODUCER_SPEC_VERSION,
        "producer_state_contract_version": GATE01_PRIVATE_PRODUCER_STATE_VERSION,
        "producer_spec_file_sha256": hashlib.sha256(producer_spec_raw).hexdigest(),
        "producer_spec_artifact_sha256": producer_receipt[
            "producer_spec_artifact_sha256"
        ],
        "producer_state_file_sha256": hashlib.sha256(producer_state_raw).hexdigest(),
        "build_plan_sha256": hashlib.sha256(plan_raw).hexdigest(),
        "producer_receipt": dict(producer_receipt),
    }


def write_gate01_protocol_lock(
    spec_path: str | Path,
    out_path: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> Gate01ProtocolLock:
    """Write a hash-sealed external protocol lock from an independent specification."""

    source = assert_gate01_external_path(spec_path, repo_root=repo_root)
    output = assert_gate01_external_path(out_path, repo_root=repo_root)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Gate 0.1 protocol-lock specification root must be a mapping.")
    lock = Gate01ProtocolLock.from_spec(payload)
    lock.save(output)
    return lock


def build_gate01_private_manifest(
    plan_path: str | Path,
    producer_spec_path: str | Path,
    producer_state_path: str | Path,
    protocol_lock_path: str | Path,
    calibrator_path: str | Path,
    out_path: str | Path,
    state_path: str | Path,
    *,
    resume: bool,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Verify sealed producer evidence and atomically assemble the external v4 manifest.

    The builder never trains or predicts.  Its only volume operations are canonical loads
    and hashes of paths already declared in the external build plan.
    """

    plan_source = assert_gate01_external_path(plan_path, repo_root=repo_root)
    producer_spec_source = assert_gate01_external_path(
        producer_spec_path, repo_root=repo_root
    )
    producer_state_source = assert_gate01_external_path(
        producer_state_path, repo_root=repo_root
    )
    protocol_lock_source = assert_gate01_external_path(
        protocol_lock_path, repo_root=repo_root
    )
    calibrator_source = assert_gate01_external_path(
        calibrator_path, repo_root=repo_root
    )
    output = assert_gate01_external_path(out_path, repo_root=repo_root)
    state_output = assert_gate01_external_path(state_path, repo_root=repo_root)
    plan_raw = plan_source.read_bytes()
    plan = json.loads(plan_raw.decode("utf-8"))
    if not isinstance(plan, Mapping):
        raise ValueError("Gate 0.1 private build plan root must be a mapping.")
    _assert_exact_keys(plan, _PLAN_KEYS, "Gate 0.1 private build plan")
    if plan["contract_version"] != GATE01_PRIVATE_BUILD_PLAN_VERSION:
        raise ValueError("Gate 0.1 private build-plan contract is incompatible.")
    if plan["execution_mode"] != "scientific":
        raise ValueError("Gate 0.1 private builder accepts scientific plans only.")
    evidence = plan["evidence_scope"]
    if not isinstance(evidence, Mapping) or evidence.get("private_data_run") is not True:
        raise ValueError("Gate 0.1 private build plan requires private_data_run=true.")
    if evidence.get("evidence_kind") != "private":
        raise ValueError("Gate 0.1 private build plan requires evidence_kind=private.")

    protocol_lock = Gate01ProtocolLock.load(protocol_lock_source)
    producer_spec_raw = producer_spec_source.read_bytes()
    producer_spec = _validate_producer_spec(producer_spec_raw, protocol_lock)
    producer_state_raw = producer_state_source.read_bytes()
    producer_state = _validate_producer_state(
        producer_state_raw,
        plan=plan,
        plan_sha256=hashlib.sha256(plan_raw).hexdigest(),
        producer_spec=producer_spec,
        protocol_lock=protocol_lock,
    )
    verified_producer_receipt = _verified_producer_receipt(
        plan_raw=plan_raw,
        producer_spec_raw=producer_spec_raw,
        producer_state_raw=producer_state_raw,
        producer_receipt=producer_state["producer_receipt"],
    )
    validate_gate01_verified_producer_receipt(verified_producer_receipt)
    _validate_plan_against_producer(
        plan, producer_spec=producer_spec, producer_state=producer_state
    )
    calibrator = PosthocTargetCalibrator.load(
        calibrator_source,
        expected_split_fingerprint=protocol_lock.split_fingerprint,
        expected_template_sha256=protocol_lock.calibrator_template_sha256,
        expected_artifact_sha256=protocol_lock.calibrator_artifact_sha256,
    )
    protocol_lock.assert_calibrator(calibrator)

    plan_sha256 = hashlib.sha256(plan_raw).hexdigest()
    state = _initial_or_resumed_state(
        state_output,
        resume=resume,
        plan_sha256=plan_sha256,
        producer_spec_file_sha256=verified_producer_receipt[
            "producer_spec_file_sha256"
        ],
        producer_state_file_sha256=verified_producer_receipt[
            "producer_state_file_sha256"
        ],
        protocol_lock=protocol_lock,
        calibrator=calibrator,
        split_provenance=producer_spec["split_provenance"],
    )
    cases = plan["cases"]
    if not isinstance(cases, list):
        raise ValueError("Gate 0.1 private build plan cases must be a list.")
    built_cases: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        if not isinstance(case, Mapping):
            raise ValueError(f"Gate 0.1 private build case {index} must be a mapping.")
        _assert_exact_keys(
            case,
            _CASE_KEYS,
            f"Gate 0.1 private build case {index}",
            optional={"wrong_target_sb_v2"},
        )
        cache_key = str(index)
        cached = state["completed_cases"].get(cache_key)
        if cached is not None and _cached_case_is_current(cached):
            built = dict(cached["manifest_case"])
        else:
            built, file_stamps = _verify_and_build_case(
                case, root=plan_source.parent, output_root=output.parent, index=index
            )
            state["completed_cases"][cache_key] = {
                "manifest_case": built,
                "file_stamps": file_stamps,
            }
            _write_json_atomic(state_output, state)
        built_cases.append(built)

    manifest = {
        "contract_version": GATE01_INPUT_CONTRACT_VERSION,
        "execution_mode": plan["execution_mode"],
        "selection_fingerprint_sha256": plan["selection_fingerprint_sha256"],
        "evidence_scope": dict(plan["evidence_scope"]),
        "split_fingerprint": plan["split_fingerprint"],
        "split_provenance": dict(plan["split_provenance"]),
        "artifact_provenance": dict(plan["artifact_provenance"]),
        "source_support_contract": dict(plan["source_support_contract"]),
        "producer_receipt": verified_producer_receipt,
        "cases": built_cases,
    }
    # Validate the complete metadata/hash graph before publishing the final manifest.
    # The candidate lives beside the output so all relative volume paths resolve with
    # exactly the same semantics.  A failed build never replaces a prior valid output.
    candidate = output.with_name(f".{output.name}.validation.tmp")
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    try:
        load_gate01_input_manifest(
            candidate, protocol_lock=protocol_lock, calibrator=calibrator
        )
        candidate.replace(output)
    except BaseException:
        candidate.unlink(missing_ok=True)
        raise
    manifest_sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
    state.update(
        {
            "status": "complete",
            "manifest_sha256": manifest_sha256,
            "case_count": len(built_cases),
        }
    )
    _write_json_atomic(state_output, state)
    return {
        "contract_version": GATE01_PRIVATE_BUILD_STATE_VERSION,
        "status": "complete",
        "manifest_sha256": manifest_sha256,
        "case_count": len(built_cases),
        "protocol_lock_artifact_sha256": protocol_lock.artifact_sha256,
        "calibrator_artifact_sha256": calibrator.artifact_sha256,
        "split_provenance": dict(plan["split_provenance"]),
        "producer_receipt": verified_producer_receipt,
    }


def _initial_or_resumed_state(
    path: Path,
    *,
    resume: bool,
    plan_sha256: str,
    producer_spec_file_sha256: str,
    producer_state_file_sha256: str,
    protocol_lock: Gate01ProtocolLock,
    calibrator: PosthocTargetCalibrator,
    split_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    expected = {
        "contract_version": GATE01_PRIVATE_BUILD_STATE_VERSION,
        "plan_sha256": plan_sha256,
        "producer_spec_file_sha256": producer_spec_file_sha256,
        "producer_state_file_sha256": producer_state_file_sha256,
        "protocol_lock_artifact_sha256": protocol_lock.artifact_sha256,
        "calibrator_artifact_sha256": calibrator.artifact_sha256,
        "split_provenance": dict(split_provenance),
    }
    if path.exists():
        if not resume:
            raise FileExistsError(
                "Gate 0.1 build state exists; pass resume=True or choose a new state path."
            )
        state = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(state, Mapping):
            raise ValueError("Gate 0.1 build state root must be a mapping.")
        if any(state.get(key) != value for key, value in expected.items()):
            raise ValueError("Gate 0.1 build state is stale or belongs to different inputs.")
        completed = state.get("completed_cases")
        if not isinstance(completed, Mapping):
            raise ValueError("Gate 0.1 build state completed_cases is invalid.")
        return {**dict(state), "completed_cases": dict(completed), "status": "building"}
    state = {**expected, "status": "building", "completed_cases": {}}
    _write_json_atomic(path, state)
    return state


def _verify_and_build_case(
    case: Mapping[str, Any],
    *,
    root: Path,
    output_root: Path,
    index: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    result = {
        key: case[key]
        for key in (
            "case_id",
            "traveller_identity_sha256",
            "source_domain",
            "target_domain",
        )
    }
    stamps: list[dict[str, Any]] = []
    for key in (
        "source_image",
        "source_support_mask",
        "target",
        "raw_identity",
        "raw_sb_v2",
        "stage1_reconstruction_ceiling",
    ):
        result[key], stamp = _verify_reference(
            case[key],
            root=root,
            output_root=output_root,
            role=f"case {index}.{key}",
            mask=key == "source_support_mask",
        )
        stamps.append(stamp)
    wrong_result: dict[str, Any] = {}
    for label, reference in sorted(dict(case.get("wrong_target_sb_v2", {})).items()):
        wrong_result[label], stamp = _verify_reference(
            reference,
            root=root,
            output_root=output_root,
            role=f"case {index}.wrong_target_sb_v2[{label}]",
            mask=False,
        )
        stamps.append(stamp)
    if wrong_result:
        result["wrong_target_sb_v2"] = wrong_result
    return result, stamps


def _verify_reference(
    value: Any,
    *,
    root: Path,
    output_root: Path,
    role: str,
    mask: bool,
) -> tuple[dict[str, str], dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{role} must be a path/expected_sha256 mapping.")
    _assert_exact_keys(value, _INPUT_REFERENCE_KEYS, role)
    expected = str(value["expected_sha256"])
    path_value = str(value["path"])
    path = Path(path_value)
    path = path.resolve() if path.is_absolute() else (root / path).resolve()
    actual = _canonical_path_sha256(path, mask=mask)
    if actual != expected:
        raise ValueError(f"Gate 0.1 frozen input hash mismatch for {role}.")
    stat = path.stat()
    relative = os.path.relpath(path, output_root)
    return (
        {"path": relative, "sha256": actual},
        {
            "path": str(path),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "mask": mask,
            "canonical_loaded_array_sha256": actual,
        },
    )


def _canonical_path_sha256(path: Path, *, mask: bool) -> str:
    if mask:
        if path.suffix.lower() != ".npy":
            raise ValueError("Gate 0.1 support masks must use .npy.")
        raw = np.load(path, allow_pickle=False)
        if raw.dtype != np.bool_ and not np.isin(raw, (0, 1)).all():
            raise ValueError("Gate 0.1 support mask contains non-boolean values.")
        tensor = torch.from_numpy(np.asarray(raw, dtype=np.bool_))
    elif path.name.lower().endswith((".nii", ".nii.gz")):
        array, _ = load_official_nifti(path)
        tensor = torch.from_numpy(np.asarray(array, dtype=np.float32))
    elif path.suffix.lower() == ".npy":
        tensor = torch.from_numpy(
            np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
        )
    else:
        raise ValueError("Gate 0.1 builder supports NIfTI and .npy arrays only.")
    if tensor.ndim < 3 or not bool(torch.isfinite(tensor).all()):
        raise ValueError("Gate 0.1 builder input must be a finite full volume.")
    return canonical_loaded_array_sha256(tensor)


def _cached_case_is_current(cached: Any) -> bool:
    if not isinstance(cached, Mapping):
        return False
    stamps = cached.get("file_stamps")
    if not isinstance(stamps, list) or not stamps:
        return False
    for item in stamps:
        if not isinstance(item, Mapping):
            return False
        path = Path(str(item.get("path", "")))
        if not path.is_file():
            return False
        stat = path.stat()
        if stat.st_size != item.get("size") or stat.st_mtime_ns != item.get("mtime_ns"):
            return False
        expected = str(item.get("canonical_loaded_array_sha256", ""))
        try:
            actual = _canonical_path_sha256(path, mask=item.get("mask") is True)
        except (OSError, TypeError, ValueError):
            return False
        if actual != expected:
            return False
    return isinstance(cached.get("manifest_case"), Mapping)


def assert_gate01_external_path(
    path: str | Path, *, repo_root: str | Path | None = None
) -> Path:
    path = Path(path).resolve()
    root = (
        Path(repo_root).resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[3]
    )
    try:
        path.relative_to(root)
    except ValueError:
        return path
    raise ValueError(
        "Gate 0.1 private protocol inputs and outputs must remain outside the Git repository."
    )


def _assert_exact_keys(
    payload: Mapping[str, Any],
    expected: set[str],
    name: str,
    *,
    optional: set[str] | None = None,
) -> None:
    optional = optional or set()
    missing = sorted((expected - optional) - set(payload))
    unexpected = sorted(set(payload) - expected)
    if missing or unexpected:
        raise ValueError(
            f"{name} schema mismatch: missing={missing}, unexpected={unexpected}."
        )


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


__all__ = [
    "GATE01_PRIVATE_BUILD_PLAN_VERSION",
    "GATE01_PRIVATE_BUILD_STATE_VERSION",
    "GATE01_PRIVATE_PRODUCER_RECEIPT_VERSION",
    "GATE01_PRIVATE_PRODUCER_SPEC_VERSION",
    "assert_gate01_external_path",
    "build_gate01_private_manifest",
    "sealed_gate01_producer_receipt",
    "write_gate01_protocol_lock",
]
