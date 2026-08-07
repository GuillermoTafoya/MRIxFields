"""External-only, deterministic producer for the private Gate 0.1 volume graph.

The producer turns one frozen 15-acquisition traveller selection into the complete
15 reconstruction / 60 directed-prediction / 180 sibling wrong-target graph.  It is
deliberately separate from the manifest verifier and never writes inside the Git tree.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

from fieldbridge.config import load_yaml_config
from fieldbridge.data.contracts import VolumeRecord
from fieldbridge.data.domains import CONTRASTS, FIELD_STRENGTHS_T, Contrast, Domain
from fieldbridge.data.latent_bank import (
    LATENT_BANK_CONTRACT_VERSION,
    decode_latent,
    downsample_factor,
    load_volume,
)
from fieldbridge.data.latent_bank_dataset import LatentStats
from fieldbridge.data.vae_splits import load_vae_splits, vae_splits_fingerprint
from fieldbridge.evaluation.stage2_gate01 import (
    canonical_loaded_array_sha256,
    gate01_code_provenance,
    gate01_selection_fingerprint,
    validate_gate01_scientific_hash_graph,
)
from fieldbridge.evaluation.stage2_gate01_builder import (
    GATE01_PRIVATE_BUILD_PLAN_VERSION,
    assert_gate01_external_path,
)
from fieldbridge.evaluation.stage2_gate01_calibration import (
    FULL_LATENT_BANK_BUILD_COMMIT,
    GATE01_SUPPORT_THRESHOLD,
    RESPLIT_FINGERPRINT,
    SB_V2_CONFIG_SHA256,
    SB_V2_CHECKPOINT_SHA256,
    STAGE1_RUN_C_CONFIG_SHA256,
    STAGE1_RUN_C_CHECKPOINT_SHA256,
)
from fieldbridge.evaluation.stage2_gate01_protocol import Gate01ProtocolLock
from fieldbridge.evaluation.stage2_transport_eval import (
    DecodeSpec,
    TransportSamplerConfig,
    sample_transport,
)
from fieldbridge.models.factory import build_decoder, build_translator
from fieldbridge.training.checkpoints import load_checkpoint, resolve_git_commit

GATE01_PROSPECTIVE_SELECTION_VERSION = "stage2-gate01-prospective-selection-v1"
GATE01_PRIVATE_PRODUCER_SPEC_VERSION = "stage2-gate01-private-producer-spec-v2"
GATE01_PRIVATE_PRODUCER_STATE_VERSION = "stage2-gate01-private-producer-state-v2"
GATE01_PRODUCER_DECODE_STRATEGY = "full"
GATE01_VAE_DOWNSAMPLE_FACTOR = 4

_SELECTION_KEYS = {
    "contract_version",
    "traveller_identity_sha256",
    "split_name",
    "acquisition_case_identity_sha256",
    "selection_fingerprint_sha256",
    "artifact_sha256",
}
_SPEC_KEYS = {
    "contract_version",
    "selection_artifact_sha256",
    "selection_fingerprint_sha256",
    "traveller_identity_sha256",
    "split_file_sha256",
    "split_fingerprint",
    "selected_source_acquisitions",
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


class Gate01InferenceBackend(Protocol):
    """Injected boundary used by synthetic tests and the real frozen models alike."""

    def source(self, record: VolumeRecord) -> torch.Tensor: ...

    def reconstruct(self, record: VolumeRecord, latent_path: Path) -> torch.Tensor: ...

    def translate(
        self, record: VolumeRecord, target_domain: Domain, latent_path: Path
    ) -> torch.Tensor: ...

    @property
    def decode_paths_used(self) -> Sequence[str]: ...


@dataclass(frozen=True, slots=True)
class ResolvedGate01Selection:
    payload: Mapping[str, Any]
    records: Mapping[str, VolumeRecord]


def prepare_gate01_prospective_selection(
    split_path: str | Path,
    traveller_subject_id: str,
    out_path: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Freeze 15 acquisition identities without persisting the private subject or IDs."""

    split_source = assert_gate01_external_path(split_path, repo_root=repo_root)
    output = assert_gate01_external_path(out_path, repo_root=repo_root)
    splits = load_vae_splits(split_source)
    if vae_splits_fingerprint(splits) != RESPLIT_FINGERPRINT:
        raise ValueError("Gate 0.1 selection requires the frozen resplit fingerprint.")
    matches: list[tuple[str, VolumeRecord]] = []
    for split_name in ("train", "validation", "test"):
        for record in splits.records_for(split_name):
            prefix = str(record.metadata.get("prefix", str(record.case_id).split("_", 1)[0]))
            if prefix == "P" and str(record.subject_id) == str(traveller_subject_id):
                matches.append((split_name, record))
    if len(matches) != 15:
        raise ValueError("Gate 0.1 prospective selection must resolve exactly 15 acquisitions.")
    split_names = {name for name, _ in matches}
    if len(split_names) != 1:
        raise ValueError("Gate 0.1 traveller acquisitions must belong to one split.")
    traveller_hash = _sha256_text(f"P:{traveller_subject_id}")
    records = {record.domain.label: record for _, record in matches}
    payload = _selection_payload(records, traveller_hash, next(iter(split_names)))
    _write_json_atomic(output, payload)
    return payload


def load_gate01_prospective_selection(
    path: str | Path,
    split_path: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> ResolvedGate01Selection:
    selection_source = assert_gate01_external_path(path, repo_root=repo_root)
    split_source = assert_gate01_external_path(split_path, repo_root=repo_root)
    payload = json.loads(selection_source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Gate 0.1 prospective selection root must be a mapping.")
    _assert_exact_keys(payload, _SELECTION_KEYS, "Gate 0.1 prospective selection")
    if payload["contract_version"] != GATE01_PROSPECTIVE_SELECTION_VERSION:
        raise ValueError("Gate 0.1 prospective-selection contract is incompatible.")
    expected_artifact = _sha256_json(
        {key: payload[key] for key in _SELECTION_KEYS - {"artifact_sha256"}}
    )
    if payload["artifact_sha256"] != expected_artifact:
        raise ValueError("Gate 0.1 prospective-selection artifact hash mismatch.")
    splits = load_vae_splits(split_source)
    if vae_splits_fingerprint(splits) != RESPLIT_FINGERPRINT:
        raise ValueError("Gate 0.1 selection requires the frozen resplit fingerprint.")
    split_name = str(payload["split_name"])
    if split_name not in {"train", "validation", "test"}:
        raise ValueError("Gate 0.1 prospective selection names an unknown split.")
    identities = payload["acquisition_case_identity_sha256"]
    if not isinstance(identities, Mapping) or set(identities) != _all_domain_labels():
        raise ValueError("Gate 0.1 selection must identify all 15 domains exactly once.")
    by_hash: dict[str, VolumeRecord] = {}
    for record in splits.records_for(split_name):
        identity = _sha256_text(str(record.case_id))
        if identity in by_hash:
            raise ValueError("Gate 0.1 split has duplicate canonical case identities.")
        by_hash[identity] = record
    records: dict[str, VolumeRecord] = {}
    subjects: set[str] = set()
    for label, identity in identities.items():
        record = by_hash.get(str(identity))
        if record is None or record.domain.label != label:
            raise ValueError("Gate 0.1 selection is stale or has a domain/case mismatch.")
        prefix = str(record.metadata.get("prefix", str(record.case_id).split("_", 1)[0]))
        if prefix != "P":
            raise ValueError("Gate 0.1 selection must contain prospective acquisitions only.")
        source_value = Path(record.image_path)
        source_path = assert_gate01_external_path(
            source_value if source_value.is_absolute() else split_source.parent / source_value,
            repo_root=repo_root,
        )
        if not source_path.is_file():
            raise ValueError("Gate 0.1 selected acquisition source is missing.")
        record = replace(record, image_path=source_path)
        subjects.add(str(record.subject_id))
        records[str(label)] = record
    if len(subjects) != 1:
        raise ValueError("Gate 0.1 selection mixes travellers.")
    traveller_hash = _sha256_text(f"P:{next(iter(subjects))}")
    if traveller_hash != payload["traveller_identity_sha256"]:
        raise ValueError("Gate 0.1 selection traveller identity mismatch.")
    regenerated = _selection_payload(records, traveller_hash, split_name)
    if regenerated != dict(payload):
        raise ValueError("Gate 0.1 selection fingerprint or identities are stale.")
    return ResolvedGate01Selection(payload=dict(payload), records=records)


def write_gate01_producer_spec(
    *,
    selection_path: str | Path,
    split_path: str | Path,
    bank_dir: str | Path,
    stage1_config_path: str | Path,
    stage1_checkpoint_path: str | Path,
    sb_v2_config_path: str | Path,
    sb_v2_checkpoint_path: str | Path,
    protocol_lock_path: str | Path,
    sampler: TransportSamplerConfig,
    decode: DecodeSpec,
    out_path: str | Path,
    deterministic_seed: int = 0,
    repo_root: str | Path | None = None,
    file_sha256: Callable[[Path], str] | None = None,
) -> dict[str, Any]:
    """Hash-seal every producer input and inference parameter into an external spec."""

    hash_file = file_sha256 or _sha256_file
    _validate_sampler_decode(
        {"solver": sampler.solver, "n_steps": int(sampler.n_steps)},
        {
            "strategy": GATE01_PRODUCER_DECODE_STRATEGY,
            "block_size": [int(value) for value in decode.block_size],
            "halo": [int(value) for value in decode.halo],
            "precision": decode.precision,
        },
    )
    paths = _external_inputs(
        selection_path,
        split_path,
        bank_dir,
        stage1_config_path,
        stage1_checkpoint_path,
        sb_v2_config_path,
        sb_v2_checkpoint_path,
        protocol_lock_path,
        repo_root=repo_root,
    )
    output = assert_gate01_external_path(out_path, repo_root=repo_root)
    resolved = load_gate01_prospective_selection(
        paths["selection"], paths["split"], repo_root=repo_root
    )
    lock = Gate01ProtocolLock.load(paths["protocol_lock"])
    if lock.split_fingerprint != RESPLIT_FINGERPRINT:
        raise ValueError("Gate 0.1 producer protocol lock has the wrong split fingerprint.")
    if lock.traveller_identity_sha256 != resolved.payload["traveller_identity_sha256"]:
        raise ValueError(
            "Gate 0.1 protocol and prospective selection identify different travellers."
        )
    if (
        lock.selection_fingerprint_sha256
        != resolved.payload["selection_fingerprint_sha256"]
    ):
        raise ValueError("Gate 0.1 protocol and prospective selection fingerprints differ.")
    stage1_sha = hash_file(paths["stage1_checkpoint"])
    sb_sha = hash_file(paths["sb_v2_checkpoint"])
    if stage1_sha != STAGE1_RUN_C_CHECKPOINT_SHA256 or sb_sha != SB_V2_CHECKPOINT_SHA256:
        raise ValueError("Gate 0.1 producer checkpoint identity is stale or incompatible.")
    stage1_config_sha = hash_file(paths["stage1_config"])
    sb_config_sha = hash_file(paths["sb_v2_config"])
    if stage1_config_sha != STAGE1_RUN_C_CONFIG_SHA256:
        raise ValueError("Gate 0.1 producer Stage-1 configuration identity is stale.")
    if sb_config_sha != SB_V2_CONFIG_SHA256:
        raise ValueError("Gate 0.1 producer SB-v2 configuration identity is stale.")
    bank_identity = _latent_bank_identity(paths["bank"], hash_file)
    if bank_identity["build_git_commit"] != FULL_LATENT_BANK_BUILD_COMMIT:
        raise ValueError("Gate 0.1 producer latent-bank build commit is stale.")
    if bank_identity["vae_checkpoint_sha256"] != STAGE1_RUN_C_CHECKPOINT_SHA256:
        raise ValueError("Gate 0.1 producer latent bank used the wrong Stage-1 checkpoint.")
    source_identities = _selected_source_identities(resolved.records)
    _selected_latent_paths(
        paths["bank"],
        resolved.records,
        split_name=str(resolved.payload["split_name"]),
        source_identities=source_identities,
    )
    payload: dict[str, Any] = {
        "contract_version": GATE01_PRIVATE_PRODUCER_SPEC_VERSION,
        "selection_artifact_sha256": resolved.payload["artifact_sha256"],
        "selection_fingerprint_sha256": resolved.payload["selection_fingerprint_sha256"],
        "traveller_identity_sha256": resolved.payload["traveller_identity_sha256"],
        "split_file_sha256": hash_file(paths["split"]),
        "split_fingerprint": RESPLIT_FINGERPRINT,
        "selected_source_acquisitions": source_identities,
        "latent_bank": bank_identity,
        "stage1_config_sha256": stage1_config_sha,
        "stage1_checkpoint_sha256": stage1_sha,
        "sb_v2_config_sha256": sb_config_sha,
        "sb_v2_checkpoint_sha256": sb_sha,
        "sampler": {"solver": sampler.solver, "n_steps": int(sampler.n_steps)},
        "decode": {
            "strategy": GATE01_PRODUCER_DECODE_STRATEGY,
            "block_size": [int(value) for value in decode.block_size],
            "halo": [int(value) for value in decode.halo],
            "precision": decode.precision,
        },
        "deterministic_seed": int(deterministic_seed),
        "protocol_lock_artifact_sha256": lock.artifact_sha256,
    }
    payload["artifact_sha256"] = _sha256_json(payload)
    _write_json_atomic(output, payload)
    return payload


def produce_gate01_private_artifacts(
    *,
    spec_path: str | Path,
    selection_path: str | Path,
    split_path: str | Path,
    bank_dir: str | Path,
    stage1_config_path: str | Path,
    stage1_checkpoint_path: str | Path,
    sb_v2_config_path: str | Path,
    sb_v2_checkpoint_path: str | Path,
    protocol_lock_path: str | Path,
    output_dir: str | Path,
    state_dir: str | Path,
    device: str = "cuda",
    resume: bool,
    repo_root: str | Path | None = None,
    backend: Gate01InferenceBackend | None = None,
    file_sha256: Callable[[Path], str] | None = None,
    code_provenance: Mapping[str, Any] | None = None,
    code_commit: str | None = None,
    progress_observer: Callable[[str, int], None] | None = None,
) -> dict[str, Any]:
    """Produce and verify the complete private graph, resuming only hash-clean work."""

    hash_file = file_sha256 or _sha256_file
    spec_source = assert_gate01_external_path(spec_path, repo_root=repo_root)
    output = assert_gate01_external_path(output_dir, repo_root=repo_root)
    state_root = assert_gate01_external_path(state_dir, repo_root=repo_root)
    if output == state_root or _is_within(output, state_root) or _is_within(state_root, output):
        raise ValueError("Gate 0.1 output and state directories must be separate.")
    paths = _external_inputs(
        selection_path,
        split_path,
        bank_dir,
        stage1_config_path,
        stage1_checkpoint_path,
        sb_v2_config_path,
        sb_v2_checkpoint_path,
        protocol_lock_path,
        repo_root=repo_root,
    )
    spec = _load_producer_spec(spec_source)
    lock = Gate01ProtocolLock.load(paths["protocol_lock"])
    provenance = dict(code_provenance or gate01_code_provenance())
    commit = str(code_commit or resolve_git_commit())
    lock.assert_evaluation_code_provenance(code_commit=commit, code_provenance=provenance)
    resolved = load_gate01_prospective_selection(
        paths["selection"], paths["split"], repo_root=repo_root
    )
    observed = _observed_spec_inputs(paths, resolved, lock, spec, hash_file)
    for key, value in observed.items():
        if spec.get(key) != value:
            raise ValueError(f"Gate 0.1 producer input {key} is stale or incompatible.")
    output.mkdir(parents=True, exist_ok=True)
    state_root.mkdir(parents=True, exist_ok=True)
    state_path = state_root / "producer-state.json"
    state = _load_or_initialize_state(state_path, spec, resume=resume)
    _reject_unexpected_files(
        output, state, allow_plan=(output / "private-build-plan.json").exists()
    )
    latent_paths = _selected_latent_paths(
        paths["bank"],
        resolved.records,
        split_name=str(resolved.payload["split_name"]),
        source_identities=spec["selected_source_acquisitions"],
    )
    actual_backend = backend or _RealGate01Backend.create(
        bank_dir=paths["bank"],
        stage1_config=paths["stage1_config"],
        stage1_checkpoint=paths["stage1_checkpoint"],
        sb_config=paths["sb_v2_config"],
        sb_checkpoint=paths["sb_v2_checkpoint"],
        sampler=_sampler_from_spec(spec),
        decode=_decode_from_spec(spec),
        device=device,
    )
    operation = 0

    def materialize(
        key: str,
        relative: str,
        factory: Callable[[], torch.Tensor],
        *,
        mask: bool = False,
        inference_counter: str | None = None,
        expected_shape: tuple[int, ...] | None = None,
        expected_sha256: str | None = None,
    ) -> dict[str, str]:
        nonlocal operation
        destination = output / relative
        cached = state["completed"].get(key)
        if cached is not None:
            _verify_completed(destination, cached, mask=mask)
        else:
            pending_expected = {
                "relative_path": relative,
                "mask": mask,
                "inference_counter": inference_counter,
                "decode_strategy": (
                    GATE01_PRODUCER_DECODE_STRATEGY
                    if inference_counter is not None
                    else None
                ),
                "expected_shape": list(expected_shape) if expected_shape is not None else None,
            }
            pending = state["pending"].get(key)
            if pending is not None and pending != pending_expected:
                raise ValueError("Gate 0.1 producer pending state is stale or incompatible.")
            if pending is None:
                if destination.exists():
                    raise ValueError("Gate 0.1 producer found an unexpected untracked output file.")
                state["pending"][key] = pending_expected
                _write_json_atomic(state_path, state)
            adopting_pending_output = destination.exists()
            if adopting_pending_output:
                # The array was atomically published before an interrupted state update.
                # A valid pending journal lets resume adopt it without duplicate inference.
                tensor = _load_output_tensor(destination, mask=mask)
            else:
                tensor = factory().detach().cpu()
                if tensor.ndim < 3 or not bool(torch.isfinite(tensor).all()):
                    raise ValueError(
                        "Gate 0.1 producer emitted a non-finite or non-volume array."
                    )
                if mask:
                    tensor = tensor.to(torch.bool)
            if expected_shape is not None and tuple(tensor.shape) != expected_shape:
                raise ValueError(
                    "Gate 0.1 producer output shape does not match its source acquisition."
                )
            identity = canonical_loaded_array_sha256(tensor)
            if expected_sha256 is not None and identity != expected_sha256:
                raise ValueError(
                    "Gate 0.1 producer loaded source acquisition differs from its sealed identity."
                )
            if inference_counter is not None:
                paths_used = set(state["producer_provenance"]["path_used"])
                backend_paths = set(actual_backend.decode_paths_used)
                if (
                    not adopting_pending_output
                    and backend_paths != {GATE01_PRODUCER_DECODE_STRATEGY}
                ) or backend_paths - {GATE01_PRODUCER_DECODE_STRATEGY}:
                    raise ValueError("Gate 0.1 producer used a non-full decode path.")
                paths_used.add(GATE01_PRODUCER_DECODE_STRATEGY)
                state["producer_provenance"]["path_used"] = sorted(paths_used)
            if not adopting_pending_output:
                _write_array_atomic(destination, tensor, mask=mask)
            state["completed"][key] = {
                "relative_path": relative,
                "canonical_loaded_array_sha256": identity,
                "mask": mask,
            }
            state["pending"].pop(key)
            state["counts"] = _completed_inference_counts(state["completed"])
            _write_json_atomic(state_path, state)
            operation += 1
            if progress_observer is not None:
                progress_observer(key, operation)
        identity = str(state["completed"][key]["canonical_loaded_array_sha256"])
        return {"path": relative, "expected_sha256": identity}

    with _deterministic_runtime(int(spec["deterministic_seed"])):
        acquisition: dict[str, dict[str, str]] = {}
        support: dict[str, dict[str, str]] = {}
        stage1: dict[str, dict[str, str]] = {}
        shapes: dict[str, tuple[int, ...]] = {}
        for label in sorted(resolved.records):
            record = resolved.records[label]
            stem = _safe_domain_stem(record.domain)
            acquisition[label] = materialize(
                f"acquisition:{label}",
                f"arrays/acquisition-{stem}.npy",
                lambda record=record: actual_backend.source(record),
                expected_shape=tuple(
                    int(value)
                    for value in spec["selected_source_acquisitions"][label]["shape"]
                ),
                expected_sha256=str(
                    spec["selected_source_acquisitions"][label][
                        "canonical_loaded_array_sha256"
                    ]
                ),
            )
            source_tensor = _load_output_tensor(output / acquisition[label]["path"], mask=False)
            shapes[label] = tuple(source_tensor.shape)
            support[label] = materialize(
                f"support:{label}",
                f"arrays/support-{stem}.npy",
                lambda source_tensor=source_tensor: (
                    source_tensor.abs() > GATE01_SUPPORT_THRESHOLD
                ),
                mask=True,
                expected_shape=shapes[label],
            )
            stage1[label] = materialize(
                f"stage1:{label}",
                f"arrays/stage1-{stem}.npy",
                lambda record=record, label=label: actual_backend.reconstruct(
                    record, latent_paths[label]
                ),
                inference_counter="stage1_inference_count",
                expected_shape=shapes[label],
            )
        predictions: dict[tuple[str, str], dict[str, str]] = {}
        for contrast in CONTRASTS:
            for source_field in FIELD_STRENGTHS_T:
                source_label = Domain(source_field, contrast).label
                record = resolved.records[source_label]
                for target_field in FIELD_STRENGTHS_T:
                    if source_field == target_field:
                        continue
                    target_domain = Domain(target_field, contrast)
                    target_label = target_domain.label
                    key = (source_label, target_label)
                    predictions[key] = materialize(
                        f"sb:{source_label}->{target_label}",
                        (
                            f"arrays/sb-{_safe_domain_stem(record.domain)}-to-"
                            f"{_safe_domain_stem(target_domain)}.npy"
                        ),
                        lambda record=record, target_domain=target_domain, source_label=(
                            source_label
                        ): actual_backend.translate(
                            record, target_domain, latent_paths[source_label]
                        ),
                        inference_counter="sb_v2_inference_count",
                        expected_shape=shapes[source_label],
                    )

    cases: list[dict[str, Any]] = []
    descriptors: list[dict[str, Any]] = []
    for contrast in CONTRASTS:
        for source_field in FIELD_STRENGTHS_T:
            source_domain = Domain(source_field, contrast)
            source_label = source_domain.label
            for target_field in FIELD_STRENGTHS_T:
                if source_field == target_field:
                    continue
                target_domain = Domain(target_field, contrast)
                target_label = target_domain.label
                case_id = _direction_case_id(source_label, target_label)
                descriptors.append(
                    {
                        "case_identity_sha256": _sha256_text(case_id),
                        "traveller_identity_sha256": spec["traveller_identity_sha256"],
                        "contrast": contrast.value,
                        "source_field_t": float(source_field),
                        "target_field_t": float(target_field),
                    }
                )
                wrong = {
                    f"{wrong_field:g}T": predictions[
                        (source_label, Domain(wrong_field, contrast).label)
                    ]
                    for wrong_field in FIELD_STRENGTHS_T
                    if wrong_field not in {source_field, target_field}
                }
                cases.append(
                    {
                        "case_id": case_id,
                        "traveller_identity_sha256": spec["traveller_identity_sha256"],
                        "source_domain": source_domain.to_dict(),
                        "target_domain": target_domain.to_dict(),
                        "source_image": acquisition[source_label],
                        "source_support_mask": support[source_label],
                        "target": acquisition[target_label],
                        "raw_identity": stage1[source_label],
                        "raw_sb_v2": predictions[(source_label, target_label)],
                        "stage1_reconstruction_ceiling": stage1[target_label],
                        "wrong_target_sb_v2": wrong,
                    }
                )
    if gate01_selection_fingerprint(descriptors) != spec["selection_fingerprint_sha256"]:
        raise ValueError("Gate 0.1 produced graph does not match the frozen selection fingerprint.")
    validate_gate01_scientific_hash_graph(
        [
            {
                "contrast": Domain.from_dict(case["target_domain"]).contrast,
                "source_field_t": Domain.from_dict(case["source_domain"]).field_strength_t,
                "target_field_t": Domain.from_dict(case["target_domain"]).field_strength_t,
                "arrays": {
                    **{
                        role: reference["expected_sha256"]
                        for role, reference in case.items()
                        if role
                        in {
                            "source_image",
                            "source_support_mask",
                            "target",
                            "raw_identity",
                            "raw_sb_v2",
                            "stage1_reconstruction_ceiling",
                        }
                    },
                    **{
                        f"wrong_target_sb_v2[{label}]": reference["expected_sha256"]
                        for label, reference in case["wrong_target_sb_v2"].items()
                    },
                },
            }
            for case in cases
        ]
    )
    if state["counts"] != {"stage1_inference_count": 15, "sb_v2_inference_count": 60}:
        raise ValueError("Gate 0.1 producer inference counts are incomplete or duplicated.")
    if state["producer_provenance"] != {
        "decode_strategy": GATE01_PRODUCER_DECODE_STRATEGY,
        "path_used": [GATE01_PRODUCER_DECODE_STRATEGY],
    }:
        raise ValueError("Gate 0.1 producer did not prove exclusively full-volume decoding.")
    plan = {
        "contract_version": GATE01_PRIVATE_BUILD_PLAN_VERSION,
        "execution_mode": "scientific",
        "selection_fingerprint_sha256": spec["selection_fingerprint_sha256"],
        "evidence_scope": {
            "role": "private Gate 0.1 frozen traveller",
            "evidence_kind": "private",
            "private_data_run": True,
            "traveller_identity_sha256": spec["traveller_identity_sha256"],
        },
        "split_fingerprint": RESPLIT_FINGERPRINT,
        "artifact_provenance": dict(lock.artifact_provenance),
        "source_support_contract": {
            "derivation": "abs(source_image)>threshold",
            "threshold": GATE01_SUPPORT_THRESHOLD,
        },
        "cases": cases,
    }
    plan_path = output / "private-build-plan.json"
    if plan_path.exists():
        current = json.loads(plan_path.read_text(encoding="utf-8"))
        if current != plan:
            raise ValueError("Gate 0.1 existing build plan differs from resumed output.")
    else:
        _write_json_atomic(plan_path, plan)
    state.update(
        {
            "status": "complete",
            "build_plan_sha256": _sha256_file(plan_path),
            "acquisition_count": 15,
            "direction_count": 60,
            "wrong_target_reference_count": 180,
            "producer_provenance": dict(state["producer_provenance"]),
        }
    )
    _write_json_atomic(state_path, state)
    _reject_unexpected_files(output, state, allow_plan=True)
    return {
        "contract_version": GATE01_PRIVATE_PRODUCER_STATE_VERSION,
        "status": "complete",
        "build_plan": str(plan_path),
        "build_plan_sha256": state["build_plan_sha256"],
        "acquisition_count": 15,
        "stage1_inference_count": 15,
        "direction_count": 60,
        "sb_v2_inference_count": 60,
        "wrong_target_reference_count": 180,
        "protocol_lock_artifact_sha256": lock.artifact_sha256,
        "producer_spec_artifact_sha256": spec["artifact_sha256"],
        "producer_provenance": dict(state["producer_provenance"]),
    }


@dataclass(slots=True)
class _RealGate01Backend:
    decoder: Any
    translator: Any
    stats: LatentStats
    sampler: TransportSamplerConfig
    decode: DecodeSpec
    device: torch.device
    factor: int
    _path_used: set[str] = field(default_factory=set)

    @property
    def decode_paths_used(self) -> tuple[str, ...]:
        return tuple(sorted(self._path_used))

    @classmethod
    def create(
        cls,
        *,
        bank_dir: Path,
        stage1_config: Path,
        stage1_checkpoint: Path,
        sb_config: Path,
        sb_checkpoint: Path,
        sampler: TransportSamplerConfig,
        decode: DecodeSpec,
        device: str,
    ) -> "_RealGate01Backend":
        runtime_device = torch.device(device)
        if runtime_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Gate 0.1 producer requested CUDA but it is unavailable.")
        stage1_model = _model_section(load_yaml_config(stage1_config))
        decoder = build_decoder("kl_vae", **_kl_decoder_kwargs(stage1_model))
        decoder.load_state_dict(load_checkpoint(stage1_checkpoint)["decoder"], strict=True)
        decoder.requires_grad_(False).to(runtime_device).eval()
        sb_model = _model_section(load_yaml_config(sb_config))
        translator_name = str(sb_model.pop("name", "flow_matching_latent"))
        translator = build_translator(translator_name, **sb_model)
        translator.load_state_dict(load_checkpoint(sb_checkpoint)["translator"], strict=True)
        translator.requires_grad_(False).to(runtime_device).eval()
        return cls(
            decoder=decoder,
            translator=translator,
            stats=LatentStats.from_json(bank_dir / "latent_stats.json"),
            sampler=sampler,
            decode=decode,
            device=runtime_device,
            factor=downsample_factor(decoder),
        )

    def source(self, record: VolumeRecord) -> torch.Tensor:
        return load_volume(record).to(torch.float32)

    @torch.inference_mode()
    def reconstruct(self, record: VolumeRecord, latent_path: Path) -> torch.Tensor:
        latent = _load_latent(latent_path, self.device)
        return self._decode(latent, record.domain).cpu()

    @torch.inference_mode()
    def translate(
        self, record: VolumeRecord, target_domain: Domain, latent_path: Path
    ) -> torch.Tensor:
        latent = _load_latent(latent_path, self.device)
        normalized = self.stats.normalize(latent)
        translated = sample_transport(
            self.translator, normalized, record.domain, target_domain, self.sampler
        )
        return self._decode(self.stats.denormalize(translated), target_domain).cpu()

    def _decode(self, latent: torch.Tensor, domain: Domain) -> torch.Tensor:
        image, path_used = decode_latent(
            self.decoder,
            latent,
            domain,
            factor=self.factor,
            strategy=GATE01_PRODUCER_DECODE_STRATEGY,
            block_size=self.decode.block_size,
            halo=self.decode.halo,
            precision=self.decode.precision,
        )
        if path_used != GATE01_PRODUCER_DECODE_STRATEGY:
            raise RuntimeError("Gate 0.1 producer requires an exact full-volume decode.")
        self._path_used.add(path_used)
        return image


def _selection_payload(
    records: Mapping[str, VolumeRecord], traveller_hash: str, split_name: str
) -> dict[str, Any]:
    if set(records) != _all_domain_labels():
        raise ValueError("Gate 0.1 prospective selection must cover all 15 domains.")
    identities = {label: _sha256_text(str(records[label].case_id)) for label in sorted(records)}
    descriptors = []
    for contrast in CONTRASTS:
        for source in FIELD_STRENGTHS_T:
            for target in FIELD_STRENGTHS_T:
                if source != target:
                    source_label = Domain(source, contrast).label
                    target_label = Domain(target, contrast).label
                    descriptors.append(
                        {
                            "case_identity_sha256": _sha256_text(
                                _direction_case_id(source_label, target_label)
                            ),
                            "traveller_identity_sha256": traveller_hash,
                            "contrast": contrast.value,
                            "source_field_t": float(source),
                            "target_field_t": float(target),
                        }
                    )
    payload: dict[str, Any] = {
        "contract_version": GATE01_PROSPECTIVE_SELECTION_VERSION,
        "traveller_identity_sha256": traveller_hash,
        "split_name": split_name,
        "acquisition_case_identity_sha256": identities,
        "selection_fingerprint_sha256": gate01_selection_fingerprint(descriptors),
    }
    payload["artifact_sha256"] = _sha256_json(payload)
    return payload


def _observed_spec_inputs(
    paths: Mapping[str, Path],
    resolved: ResolvedGate01Selection,
    lock: Gate01ProtocolLock,
    spec: Mapping[str, Any],
    hash_file: Callable[[Path], str],
) -> dict[str, Any]:
    return {
        "selection_artifact_sha256": resolved.payload["artifact_sha256"],
        "selection_fingerprint_sha256": resolved.payload["selection_fingerprint_sha256"],
        "traveller_identity_sha256": resolved.payload["traveller_identity_sha256"],
        "split_file_sha256": hash_file(paths["split"]),
        "split_fingerprint": RESPLIT_FINGERPRINT,
        "selected_source_acquisitions": _selected_source_identities(resolved.records),
        "latent_bank": _latent_bank_identity(paths["bank"], hash_file),
        "stage1_config_sha256": hash_file(paths["stage1_config"]),
        "stage1_checkpoint_sha256": hash_file(paths["stage1_checkpoint"]),
        "sb_v2_config_sha256": hash_file(paths["sb_v2_config"]),
        "sb_v2_checkpoint_sha256": hash_file(paths["sb_v2_checkpoint"]),
        "protocol_lock_artifact_sha256": lock.artifact_sha256,
        "sampler": spec["sampler"],
        "decode": spec["decode"],
        "deterministic_seed": spec["deterministic_seed"],
    }


def _external_inputs(*values: str | Path, repo_root: str | Path | None) -> dict[str, Path]:
    names = (
        "selection",
        "split",
        "bank",
        "stage1_config",
        "stage1_checkpoint",
        "sb_v2_config",
        "sb_v2_checkpoint",
        "protocol_lock",
    )
    return {
        name: assert_gate01_external_path(value, repo_root=repo_root)
        for name, value in zip(names, values, strict=True)
    }


def _latent_bank_identity(bank_dir: Path, hash_file: Callable[[Path], str]) -> dict[str, Any]:
    manifest_path = bank_dir / "latent_bank_manifest.json"
    stats_path = bank_dir / "latent_stats.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping) or not isinstance(manifest.get("records"), list):
        raise ValueError("Gate 0.1 latent-bank manifest is invalid.")
    if manifest.get("contract_version") != LATENT_BANK_CONTRACT_VERSION:
        raise ValueError("Gate 0.1 latent-bank contract is incompatible.")
    config = manifest.get("config")
    if not isinstance(config, Mapping) or config.get("strategy") != "full":
        raise ValueError("Gate 0.1 latent bank does not prove full-volume encoding.")
    if manifest.get("strategy_used") != ["full"]:
        raise ValueError("Gate 0.1 latent bank contains a non-full encoding path.")
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    if not isinstance(stats, Mapping):
        raise ValueError("Gate 0.1 latent-bank statistics are invalid.")
    if (
        stats.get("vae_checkpoint_sha256") != manifest.get("vae_checkpoint_sha256")
        or stats.get("git_commit") != manifest.get("git_commit")
    ):
        raise ValueError("Gate 0.1 latent-bank statistics provenance is incompatible.")
    entries: list[dict[str, str]] = []
    seen: set[str] = set()
    for entry in manifest["records"]:
        relative = str(entry["path"])
        path = (bank_dir / relative).resolve()
        if not _is_within(path, bank_dir.resolve()) or relative in seen:
            raise ValueError("Gate 0.1 latent bank contains duplicate or escaping paths.")
        seen.add(relative)
        entries.append({"path": relative, "sha256": hash_file(path)})
    content = {
        "manifest_sha256": hash_file(manifest_path),
        "stats_sha256": hash_file(stats_path),
        "record_file_sha256": sorted(entries, key=lambda item: item["path"]),
    }
    return {
        "artifact_sha256": _sha256_json(content),
        "manifest_sha256": content["manifest_sha256"],
        "stats_sha256": content["stats_sha256"],
        "record_count": len(entries),
        "build_git_commit": str(manifest.get("git_commit")),
        "vae_checkpoint_sha256": str(manifest.get("vae_checkpoint_sha256")),
        "encode_provenance": {"strategy": "full", "path_used": ["full"]},
    }


def _selected_latent_paths(
    bank_dir: Path,
    records: Mapping[str, VolumeRecord],
    *,
    split_name: str,
    source_identities: Mapping[str, Any],
) -> dict[str, Path]:
    manifest = json.loads((bank_dir / "latent_bank_manifest.json").read_text(encoding="utf-8"))
    by_case: dict[str, tuple[Path, Mapping[str, Any]]] = {}
    for entry in manifest["records"]:
        if not isinstance(entry, Mapping):
            raise ValueError("Gate 0.1 latent-bank manifest record is invalid.")
        case_id = str(entry["case_id"])
        if case_id in by_case:
            raise ValueError("Gate 0.1 latent bank contains duplicate case IDs.")
        path = (bank_dir / str(entry["path"])).resolve()
        if not _is_within(path, bank_dir.resolve()):
            raise ValueError("Gate 0.1 latent-bank path escapes its root.")
        by_case[case_id] = (path, entry)
    result: dict[str, Path] = {}
    for label, record in records.items():
        found = by_case.get(str(record.case_id))
        if found is None or not found[0].is_file():
            raise ValueError("Gate 0.1 selected acquisition is missing from the latent bank.")
        path, entry = found
        payload = torch.load(path, map_location="cpu")
        _validate_selected_latent_payload(
            payload,
            entry=entry,
            record=record,
            split_name=split_name,
            source_identity=source_identities[label],
        )
        result[label] = path
    return result


def _selected_source_identities(
    records: Mapping[str, VolumeRecord],
) -> dict[str, dict[str, Any]]:
    """Stream the 15 selected acquisitions and retain only canonical identities."""

    if set(records) != _all_domain_labels():
        raise ValueError("Gate 0.1 source identities require exactly all 15 domains.")
    identities: dict[str, dict[str, Any]] = {}
    for label in sorted(records):
        tensor = load_volume(records[label]).detach().to(torch.float32).cpu()
        if tensor.ndim != 5 or tuple(tensor.shape[:2]) != (1, 1):
            raise ValueError("Gate 0.1 selected source is not a canonical full volume.")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError("Gate 0.1 selected source contains non-finite data.")
        identities[label] = {
            "canonical_loaded_array_sha256": canonical_loaded_array_sha256(tensor),
            "shape": [int(value) for value in tensor.shape],
        }
        del tensor
    return identities


def _validate_selected_latent_payload(
    payload: Any,
    *,
    entry: Mapping[str, Any],
    record: VolumeRecord,
    split_name: str,
    source_identity: Any,
) -> None:
    if not isinstance(payload, Mapping) or not isinstance(source_identity, Mapping):
        raise ValueError("Gate 0.1 selected latent payload is malformed.")
    required = {
        "contract_version",
        "case_id",
        "split",
        "domain",
        "latent",
        "latent_shape",
        "source_shape",
        "downsample_factor",
        "encode_strategy",
        "vae_checkpoint_sha256",
        "git_commit",
    }
    if not required.issubset(payload):
        raise ValueError("Gate 0.1 selected latent payload lacks required provenance.")
    if payload["contract_version"] != LATENT_BANK_CONTRACT_VERSION:
        raise ValueError("Gate 0.1 selected latent payload contract is incompatible.")
    if str(payload["case_id"]) != str(record.case_id) or str(entry.get("case_id")) != str(
        record.case_id
    ):
        raise ValueError("Gate 0.1 selected latent case identity is incompatible.")
    if payload["split"] != split_name or entry.get("split") != split_name:
        raise ValueError("Gate 0.1 selected latent split identity is incompatible.")
    try:
        payload_domain = Domain.from_dict(payload["domain"])
        entry_domain = Domain.from_dict(entry["domain"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Gate 0.1 selected latent domain provenance is malformed.") from error
    if payload_domain != record.domain or entry_domain != record.domain:
        raise ValueError("Gate 0.1 selected latent domain identity is incompatible.")
    if int(payload["downsample_factor"]) != GATE01_VAE_DOWNSAMPLE_FACTOR:
        raise ValueError("Gate 0.1 selected latent has the wrong VAE downsample factor.")
    if payload["encode_strategy"] != "full":
        raise ValueError("Gate 0.1 selected latent was not encoded as a full volume.")
    if payload["vae_checkpoint_sha256"] != STAGE1_RUN_C_CHECKPOINT_SHA256:
        raise ValueError("Gate 0.1 selected latent used the wrong Stage-1 checkpoint.")
    if payload["git_commit"] != FULL_LATENT_BANK_BUILD_COMMIT:
        raise ValueError("Gate 0.1 selected latent used the wrong bank-build commit.")
    latent = payload["latent"]
    if not isinstance(latent, torch.Tensor) or latent.ndim != 4:
        raise ValueError("Gate 0.1 selected latent tensor is malformed.")
    if not bool(torch.isfinite(latent).all()):
        raise ValueError("Gate 0.1 selected latent tensor contains non-finite data.")
    latent_shape = [int(value) for value in latent.shape]
    source_shape = [int(value) for value in source_identity.get("shape", ())]
    if len(source_shape) != 5 or source_shape[:2] != [1, 1]:
        raise ValueError("Gate 0.1 sealed selected-source shape is malformed.")
    stored_source_shape = [int(value) for value in payload["source_shape"]]
    if stored_source_shape != source_shape[1:]:
        raise ValueError("Gate 0.1 selected latent source shape is incompatible.")
    expected_spatial = [value // GATE01_VAE_DOWNSAMPLE_FACTOR for value in source_shape[-3:]]
    if any(
        value % GATE01_VAE_DOWNSAMPLE_FACTOR != 0 for value in source_shape[-3:]
    ) or latent_shape[1:] != expected_spatial:
        raise ValueError("Gate 0.1 selected latent does not represent the full source extent.")
    if [int(value) for value in payload["latent_shape"]] != latent_shape:
        raise ValueError("Gate 0.1 selected latent shape metadata is stale.")
    if [int(value) for value in entry.get("latent_shape", ())] != latent_shape:
        raise ValueError("Gate 0.1 latent manifest/payload shape mismatch.")
    if [int(value) for value in entry.get("source_shape", ())] != stored_source_shape:
        raise ValueError("Gate 0.1 latent manifest/source shape mismatch.")


def _load_producer_spec(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Gate 0.1 producer specification root must be a mapping.")
    _assert_exact_keys(payload, _SPEC_KEYS, "Gate 0.1 producer specification")
    if payload["contract_version"] != GATE01_PRIVATE_PRODUCER_SPEC_VERSION:
        raise ValueError("Gate 0.1 producer-spec contract is incompatible.")
    body = {key: payload[key] for key in _SPEC_KEYS - {"artifact_sha256"}}
    if payload["artifact_sha256"] != _sha256_json(body):
        raise ValueError("Gate 0.1 producer-spec artifact hash mismatch.")
    _validate_sampler_decode(payload["sampler"], payload["decode"])
    if payload["stage1_config_sha256"] != STAGE1_RUN_C_CONFIG_SHA256:
        raise ValueError("Gate 0.1 producer specification has the wrong Stage-1 config hash.")
    if payload["sb_v2_config_sha256"] != SB_V2_CONFIG_SHA256:
        raise ValueError("Gate 0.1 producer specification has the wrong SB-v2 config hash.")
    _validate_source_identity_contract(payload["selected_source_acquisitions"])
    if int(payload["deterministic_seed"]) < 0:
        raise ValueError("Gate 0.1 producer deterministic seed must be non-negative.")
    return dict(payload)


def _validate_sampler_decode(sampler: Any, decode: Any) -> None:
    if not isinstance(sampler, Mapping) or set(sampler) != {"solver", "n_steps"}:
        raise ValueError("Gate 0.1 producer sampler specification is invalid.")
    if sampler["solver"] not in {"euler", "heun"} or int(sampler["n_steps"]) < 1:
        raise ValueError("Gate 0.1 producer solver/step-count specification is invalid.")
    if not isinstance(decode, Mapping) or set(decode) != {
        "strategy",
        "block_size",
        "halo",
        "precision",
    }:
        raise ValueError("Gate 0.1 producer decode specification is invalid.")
    block = decode["block_size"]
    halo = decode["halo"]
    if (
        not isinstance(block, Sequence)
        or isinstance(block, (str, bytes))
        or len(block) != 3
        or any(int(value) <= 0 for value in block)
        or not isinstance(halo, Sequence)
        or isinstance(halo, (str, bytes))
        or len(halo) != 3
        or any(int(value) < 0 for value in halo)
        or decode["precision"] not in {"float32", "bfloat16"}
        or decode["strategy"] != GATE01_PRODUCER_DECODE_STRATEGY
    ):
        raise ValueError("Gate 0.1 producer full-volume decode specification is invalid.")


def _validate_source_identity_contract(value: Any) -> None:
    if not isinstance(value, Mapping) or set(value) != _all_domain_labels():
        raise ValueError("Gate 0.1 producer must seal exactly 15 selected source identities.")
    for identity in value.values():
        if not isinstance(identity, Mapping) or set(identity) != {
            "canonical_loaded_array_sha256",
            "shape",
        }:
            raise ValueError("Gate 0.1 producer selected-source identity is malformed.")
        digest = identity["canonical_loaded_array_sha256"]
        shape = identity["shape"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not isinstance(shape, list)
            or len(shape) != 5
            or shape[:2] != [1, 1]
            or any(not isinstance(item, int) or item <= 0 for item in shape)
        ):
            raise ValueError("Gate 0.1 producer selected-source identity is malformed.")


def _load_or_initialize_state(
    path: Path, spec: Mapping[str, Any], *, resume: bool
) -> dict[str, Any]:
    expected = {
        "contract_version": GATE01_PRIVATE_PRODUCER_STATE_VERSION,
        "producer_spec_artifact_sha256": spec["artifact_sha256"],
        "protocol_lock_artifact_sha256": spec["protocol_lock_artifact_sha256"],
        "producer_provenance": {
            "decode_strategy": GATE01_PRODUCER_DECODE_STRATEGY,
            "path_used": [],
        },
    }
    if path.exists():
        if not resume:
            raise FileExistsError(
                "Gate 0.1 producer state exists; pass --resume to continue it."
            )
        state = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(state, Mapping) or any(
            state.get(key) != value
            for key, value in expected.items()
            if key != "producer_provenance"
        ):
            raise ValueError("Gate 0.1 producer state is stale or belongs to different inputs.")
        if (
            not isinstance(state.get("completed"), Mapping)
            or not isinstance(state.get("pending"), Mapping)
            or not isinstance(state.get("counts"), Mapping)
            or not isinstance(state.get("producer_provenance"), Mapping)
        ):
            raise ValueError("Gate 0.1 producer state is malformed.")
        producer_provenance = state["producer_provenance"]
        if (
            producer_provenance.get("decode_strategy")
            != GATE01_PRODUCER_DECODE_STRATEGY
            or producer_provenance.get("path_used") not in ([], ["full"])
            or (
                sum(int(value) for value in state["counts"].values()) > 0
                and producer_provenance.get("path_used") != ["full"]
            )
        ):
            raise ValueError("Gate 0.1 producer state decode provenance is incompatible.")
        return {
            **dict(state),
            "completed": dict(state["completed"]),
            "pending": dict(state["pending"]),
            "counts": dict(state["counts"]),
            "producer_provenance": dict(producer_provenance),
            "status": "building",
        }
    state = {
        **expected,
        "status": "building",
        "completed": {},
        "pending": {},
        "counts": {"stage1_inference_count": 0, "sb_v2_inference_count": 0},
    }
    _write_json_atomic(path, state)
    return state


def _verify_completed(path: Path, cached: Any, *, mask: bool) -> None:
    if not isinstance(cached, Mapping) or cached.get("mask") is not mask:
        raise ValueError("Gate 0.1 producer state has an incompatible completed entry.")
    if not path.is_file():
        raise ValueError("Gate 0.1 producer completed output is missing.")
    tensor = _load_output_tensor(path, mask=mask)
    if canonical_loaded_array_sha256(tensor) != cached.get(
        "canonical_loaded_array_sha256"
    ):
        raise ValueError("Gate 0.1 producer detected a mutated completed array.")


def _reject_unexpected_files(
    output: Path, state: Mapping[str, Any], *, allow_plan: bool = False
) -> None:
    expected = {
        str((output / str(item["relative_path"])).resolve())
        for item in state.get("completed", {}).values()
        if isinstance(item, Mapping)
    }
    expected.update(
        str((output / str(item["relative_path"])).resolve())
        for item in state.get("pending", {}).values()
        if isinstance(item, Mapping)
    )
    if allow_plan:
        expected.add(str((output / "private-build-plan.json").resolve()))
    for path in output.rglob("*"):
        if path.is_symlink():
            raise ValueError("Gate 0.1 producer rejects symlinks in its output tree.")
        if path.is_file() and str(path.resolve()) not in expected:
            raise ValueError("Gate 0.1 producer found an unexpected path in its output tree.")


def _write_array_atomic(path: Path, tensor: torch.Tensor, *, mask: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    array = tensor.numpy().astype(np.bool_ if mask else np.float32, copy=False)
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _completed_inference_counts(completed: Mapping[str, Any]) -> dict[str, int]:
    return {
        "stage1_inference_count": sum(key.startswith("stage1:") for key in completed),
        "sb_v2_inference_count": sum(key.startswith("sb:") for key in completed),
    }


def _load_output_tensor(path: Path, *, mask: bool) -> torch.Tensor:
    raw = np.load(path, allow_pickle=False)
    if mask:
        if raw.dtype != np.bool_:
            raise ValueError("Gate 0.1 support mask output is not canonical boolean data.")
        return torch.from_numpy(np.asarray(raw, dtype=np.bool_))
    tensor = torch.from_numpy(np.asarray(raw, dtype=np.float32))
    if tensor.ndim < 3 or not bool(torch.isfinite(tensor).all()):
        raise ValueError("Gate 0.1 producer output is not a finite full volume.")
    return tensor


def _sampler_from_spec(spec: Mapping[str, Any]) -> TransportSamplerConfig:
    value = spec["sampler"]
    return TransportSamplerConfig(
        solver=str(value["solver"]),  # type: ignore[arg-type]
        n_steps=int(value["n_steps"]),
    )


def _decode_from_spec(spec: Mapping[str, Any]) -> DecodeSpec:
    value = spec["decode"]
    return DecodeSpec(
        block_size=tuple(int(item) for item in value["block_size"]),
        halo=tuple(int(item) for item in value["halo"]),
        precision=str(value["precision"]),  # type: ignore[arg-type]
    )


def _load_latent(path: Path, device: torch.device) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping) or "latent" not in payload:
        raise ValueError("Gate 0.1 latent payload is malformed.")
    return payload["latent"].to(torch.float32).unsqueeze(0).to(device)


def _model_section(config: Mapping[str, Any]) -> dict[str, Any]:
    value = config.get("model", {})
    if not isinstance(value, Mapping):
        raise ValueError("Gate 0.1 model config must contain a model mapping.")
    return dict(value)


def _kl_decoder_kwargs(model: Mapping[str, Any]) -> dict[str, Any]:
    keys = {
        "base_channels",
        "latent_channels",
        "spatial_dims",
        "activation",
        "use_norm",
        "num_res_blocks",
        "out_channels",
        "output_activation",
        "domain_conditioning_dim",
    }
    return {key: value for key, value in model.items() if key in keys}


@contextmanager
def _deterministic_runtime(seed: int):
    previous = torch.are_deterministic_algorithms_enabled()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous)


def _all_domain_labels() -> set[str]:
    return {Domain(field, contrast).label for contrast in CONTRASTS for field in FIELD_STRENGTHS_T}


def _direction_case_id(source_label: str, target_label: str) -> str:
    return f"gate01-{_sha256_text(source_label + '->' + target_label)[:24]}"


def _safe_domain_stem(domain: Domain) -> str:
    contrast = Contrast.parse(domain.contrast).value.replace("-", "_")
    return f"{contrast}-{domain.field_strength_t:g}T"


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _assert_exact_keys(payload: Mapping[str, Any], expected: set[str], name: str) -> None:
    missing = sorted(expected - set(payload))
    unexpected = sorted(set(payload) - expected)
    if missing or unexpected:
        raise ValueError(f"{name} schema mismatch: missing={missing}, unexpected={unexpected}.")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(path)
    return path


__all__ = [
    "GATE01_PRIVATE_PRODUCER_SPEC_VERSION",
    "GATE01_PRIVATE_PRODUCER_STATE_VERSION",
    "GATE01_PROSPECTIVE_SELECTION_VERSION",
    "Gate01InferenceBackend",
    "ResolvedGate01Selection",
    "load_gate01_prospective_selection",
    "prepare_gate01_prospective_selection",
    "produce_gate01_private_artifacts",
    "write_gate01_producer_spec",
]
