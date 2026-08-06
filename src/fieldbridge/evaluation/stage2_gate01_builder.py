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

from fieldbridge.evaluation.mrixfields2026_official import load_official_nifti
from fieldbridge.evaluation.stage2_gate01 import (
    GATE01_INPUT_CONTRACT_VERSION,
    canonical_loaded_array_sha256,
    load_gate01_input_manifest,
)
from fieldbridge.evaluation.stage2_gate01_calibration import PosthocTargetCalibrator
from fieldbridge.evaluation.stage2_gate01_protocol import Gate01ProtocolLock

GATE01_PRIVATE_BUILD_PLAN_VERSION = "stage2-gate01-private-build-plan-v1"
GATE01_PRIVATE_BUILD_STATE_VERSION = "stage2-gate01-private-build-state-v1"

_PLAN_KEYS = {
    "contract_version",
    "execution_mode",
    "selection_fingerprint_sha256",
    "evidence_scope",
    "split_fingerprint",
    "artifact_provenance",
    "source_support_contract",
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
    protocol_lock_path: str | Path,
    calibrator_path: str | Path,
    out_path: str | Path,
    state_path: str | Path,
    *,
    resume: bool,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Verify frozen arrays and atomically assemble the external v3 manifest.

    The builder never trains or predicts.  Its only volume operations are canonical loads
    and hashes of paths already declared in the external build plan.
    """

    plan_source = assert_gate01_external_path(plan_path, repo_root=repo_root)
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
        protocol_lock=protocol_lock,
        calibrator=calibrator,
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
        "artifact_provenance": dict(plan["artifact_provenance"]),
        "source_support_contract": dict(plan["source_support_contract"]),
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
    }


def _initial_or_resumed_state(
    path: Path,
    *,
    resume: bool,
    plan_sha256: str,
    protocol_lock: Gate01ProtocolLock,
    calibrator: PosthocTargetCalibrator,
) -> dict[str, Any]:
    expected = {
        "contract_version": GATE01_PRIVATE_BUILD_STATE_VERSION,
        "plan_sha256": plan_sha256,
        "protocol_lock_artifact_sha256": protocol_lock.artifact_sha256,
        "calibrator_artifact_sha256": calibrator.artifact_sha256,
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


__all__ = [
    "GATE01_PRIVATE_BUILD_PLAN_VERSION",
    "GATE01_PRIVATE_BUILD_STATE_VERSION",
    "assert_gate01_external_path",
    "build_gate01_private_manifest",
    "write_gate01_protocol_lock",
]
