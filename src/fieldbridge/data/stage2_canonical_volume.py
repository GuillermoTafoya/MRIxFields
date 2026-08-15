"""Retrospective-only canonical-volume artifacts for Stage-2.

This module is the persisted Variant-A.5 boundary described by the reviewed
photometry runbook.  It materializes exactly ``N_d(x_d)`` and the source-derived
support; it never accepts a paired endpoint, target tensor, or prospective record.
"""

from __future__ import annotations

import json
import os
import subprocess
import uuid
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from fieldbridge.data.contracts import VolumeRecord
from fieldbridge.data.domains import CONTRASTS, FIELD_STRENGTHS_T, Domain
from fieldbridge.data.photometry_factorization import (
    PHOTOMETRY_FACTORIZATION_CONTRACT_VERSION,
    PHOTOMETRY_SUPPORT_POLICY,
    FrozenPhotometryArtifact,
    canonical_tensor_sha256,
    classify_variant_a_cohort,
    reject_target_or_prediction_derived_fields,
    sha256_file,
    sha256_json,
    sha256_text,
)
from fieldbridge.evaluation.stage2_photometry_baseline import (
    VARIANT_A_QUALIFICATION_CONTRACT_VERSION,
)

CANONICAL_VOLUME_CONTRACT_VERSION = "stage2-canonical-volume-v1"
CANONICAL_ARTIFACT_CONFIG_VERSION = "stage2-canonical-artifacts-config-v1"
CANONICAL_VOLUME_MANIFEST = "canonical_volume_manifest.json"
CANONICAL_VOLUME_SEMANTICS = "frozen-N_d-source-canonical-volume-v1"
CANONICAL_VOLUME_ELIGIBILITY = "cohort=R;split-in(train,validation)"
CANONICAL_VOLUME_SPLITS = ("train", "validation")
CANONICAL_VOLUME_SOURCE_MODULES = (
    "src/fieldbridge/data/stage2_canonical_volume.py",
    "src/fieldbridge/data/photometry_factored_latent_bank.py",
    "src/fieldbridge/data/photometry_factorization.py",
    "src/fieldbridge/data/vae_splits.py",
    "src/fieldbridge/cli.py",
)
PROSPECTIVE_EXCLUSION_REASON = "prospective-cohort-rejected-before-source-array-load"

VolumeLoader = Callable[[VolumeRecord], torch.Tensor]
FileHasher = Callable[[str | Path], str]
Progress = Callable[[str, int, int, str], None]


@dataclass(frozen=True, slots=True)
class CanonicalVolumeBuildConfig:
    out_dir: Path
    splits: tuple[str, ...] = CANONICAL_VOLUME_SPLITS
    store_dtype: str = "float32"

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, Any], *, out_dir: str | Path
    ) -> "CanonicalVolumeBuildConfig":
        validated = validate_canonical_artifact_config(data)
        section = validated["canonical_volume"]
        return cls(
            out_dir=Path(out_dir),
            splits=tuple(str(value) for value in validated["eligibility"]["splits"]),
            store_dtype=str(section["store_dtype"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "out_dir": str(self.out_dir),
            "splits": list(self.splits),
            "store_dtype": self.store_dtype,
        }


@dataclass(frozen=True, slots=True)
class _EligibleRecord:
    record: VolumeRecord
    split: str
    subject_identity: str
    subject_group_identity: str
    source_path_identity_sha256: str
    source_file_sha256: str


def validate_canonical_artifact_config(data: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the reviewed engineering-only artifact configuration."""

    config = _json_safe_mapping(data)
    if config.get("contract") != CANONICAL_ARTIFACT_CONFIG_VERSION:
        raise ValueError("Stage-2 canonical-artifact config contract is incompatible.")
    if config.get("scope") != "retrospective_only_canonical_artifacts":
        raise ValueError("Stage-2 canonical-artifact config scope is incompatible.")
    eligibility = config.get("eligibility")
    if not isinstance(eligibility, Mapping):
        raise ValueError("Canonical-artifact config requires eligibility settings.")
    if eligibility.get("cohort") != "R":
        raise ValueError("Canonical artifacts are retrospective R-only.")
    splits = eligibility.get("splits")
    if splits != list(CANONICAL_VOLUME_SPLITS):
        raise ValueError("Canonical artifact v1 permits exactly train and validation roles.")
    if eligibility.get("prospective_records") != "exclude_before_array_load":
        raise ValueError("Canonical artifacts must exclude every P identity before array load.")

    canonical = config.get("canonical_volume")
    if not isinstance(canonical, Mapping):
        raise ValueError("Canonical-artifact config requires canonical_volume settings.")
    expected_canonical = {
        "contract": CANONICAL_VOLUME_CONTRACT_VERSION,
        "semantics": CANONICAL_VOLUME_SEMANTICS,
        "store_dtype": "float32",
        "support": PHOTOMETRY_SUPPORT_POLICY,
        "source": "FrozenPhotometryArtifact.normalize_source",
    }
    if dict(canonical) != expected_canonical:
        raise ValueError("Canonical-volume v1 settings differ from the reviewed contract.")

    latent = config.get("latent_bank")
    if not isinstance(latent, Mapping):
        raise ValueError("Canonical-artifact config requires latent_bank settings.")
    required_latent = {
        "contract",
        "encoder_statistic",
        "strategy",
        "store_dtype",
        "precision",
        "block_size",
        "halo",
        "support_downsample",
    }
    if set(latent) != required_latent:
        raise ValueError("Photometry-factored latent-bank settings are incomplete.")
    if latent.get("contract") != "photometry-factored-latent-bank-v1":
        raise ValueError("Photometry-factored latent-bank contract is incompatible.")
    if latent.get("encoder_statistic") != "posterior_mean":
        raise ValueError("The factored bank must use the frozen VAE posterior mean.")
    if latent.get("strategy") not in {"full", "tiled"}:
        raise ValueError("Factored-bank encoding strategy must be deterministic full or tiled.")
    if latent.get("store_dtype") not in {"float16", "float32"}:
        raise ValueError("Factored-bank store dtype must be float16 or float32.")
    if latent.get("precision") not in {"float32", "bfloat16"}:
        raise ValueError("Factored-bank precision must be float32 or bfloat16.")
    for name in ("block_size", "halo"):
        values = latent.get(name)
        if (
            not isinstance(values, list)
            or len(values) != 3
            or any(not isinstance(value, int) or value < 0 for value in values)
            or (name == "block_size" and any(value == 0 for value in values))
        ):
            raise ValueError(f"Factored-bank {name} must contain three valid integers.")
    if latent.get("support_downsample") != "all-source-voxels-supported-per-latent-cell-v1":
        raise ValueError("Factored-bank support downsampling must remain conservative.")

    statistics = config.get("statistics")
    if statistics != {
        "contract": "photometry-factored-latent-statistics-v1",
        "computed_over": {"cohort": "R", "split": "train"},
    }:
        raise ValueError("Canonical latent statistics must be computed over R/train only.")
    descriptor = config.get("structural_descriptor")
    if not isinstance(descriptor, Mapping):
        raise ValueError("Canonical-artifact config requires structural_descriptor settings.")
    expected_descriptor = {
        "contract": "photometry-factored-structural-descriptor-v1",
        "computed_over": {"cohort": "R", "split": "train"},
        "input": "canonical_standardized_latent_only",
        "pool_output_sizes": [1, 2, 4],
        "pooling": "support-normalized-adaptive-average-3d-v1",
        "gradients": "absolute-first-forward-difference-xyz-v1",
        "dtype": "float32",
        "paired_endpoint_or_target_input": "forbidden",
    }
    if dict(descriptor) != expected_descriptor:
        raise ValueError("Structural-descriptor v1 settings differ from the reviewed contract.")
    reject_target_or_prediction_derived_fields(config)
    return config


def capture_canonical_artifact_code_provenance(
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    root = (
        Path(repo_root).resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[3]
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "git_head": head,
        "checkout_clean": not bool(status.strip()),
        "module_sha256": {
            relative: sha256_file(root / relative)
            for relative in CANONICAL_VOLUME_SOURCE_MODULES
        },
    }


def validate_variant_a_qualification(
    payload: Mapping[str, Any],
    *,
    artifact: FrozenPhotometryArtifact,
    source_split_file_sha256: str,
    source_membership_fingerprint: str,
    source_recovery_fingerprint: str,
    vae_config_sha256: str | None = None,
    vae_checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    """Require a hash-valid, passing Variant-A authorization artifact."""

    result = _json_safe_mapping(payload)
    if result.get("contract_version") != VARIANT_A_QUALIFICATION_CONTRACT_VERSION:
        raise ValueError("Canonical artifacts require the reviewed Variant-A qualification.")
    stored_hash = str(result.get("result_sha256", ""))
    unhashed = dict(result)
    unhashed.pop("result_sha256", None)
    if stored_hash != sha256_json(unhashed):
        raise ValueError("Variant-A qualification result hash mismatch.")
    if result.get("artifact_sha256") != artifact.artifact_sha256:
        raise ValueError("Variant-A qualification photometry-artifact hash mismatch.")
    if result.get("resolved_config_sha256") != artifact.provenance["resolved_config_sha256"]:
        raise ValueError("Variant-A qualification photometry-config hash mismatch.")
    source_split = result.get("source_split")
    expected_split = {
        "file_sha256": source_split_file_sha256,
        "membership_fingerprint": source_membership_fingerprint,
        "recovery_fingerprint": source_recovery_fingerprint,
    }
    if source_split != expected_split:
        raise ValueError("Variant-A qualification source-split identity mismatch.")
    failures = result.get("failure_classification")
    if failures != [] or result.get("canonical_latent_bank_authorized") is not True:
        raise ValueError("Variant-A qualification did not authorize a canonical latent bank.")
    vae = result.get("vae_provenance")
    if not isinstance(vae, Mapping):
        raise ValueError("Variant-A qualification is missing frozen-VAE provenance.")
    _require_sha256(str(vae.get("config_file_sha256", "")), "VAE config")
    _require_sha256(str(vae.get("checkpoint_sha256", "")), "VAE checkpoint")
    if vae_config_sha256 is not None and vae.get("config_file_sha256") != vae_config_sha256:
        raise ValueError("Frozen-VAE config hash differs from Variant-A qualification.")
    if (
        vae_checkpoint_sha256 is not None
        and vae.get("checkpoint_sha256") != vae_checkpoint_sha256
    ):
        raise ValueError("Frozen-VAE checkpoint hash differs from Variant-A qualification.")
    return result


def load_variant_a_qualification(path: str | Path, **kwargs: Any) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not load Variant-A qualification {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Variant-A qualification root must be a JSON object.")
    return validate_variant_a_qualification(payload, **kwargs)


def build_canonical_volume_artifact(
    *,
    artifact: FrozenPhotometryArtifact,
    qualification: Mapping[str, Any],
    records_by_split: Mapping[str, Sequence[VolumeRecord]],
    config: CanonicalVolumeBuildConfig,
    resolved_config: Mapping[str, Any],
    source_split_file_sha256: str,
    source_membership_fingerprint: str,
    source_recovery_fingerprint: str,
    photometry_artifact_file_sha256: str,
    qualification_file_sha256: str,
    code_provenance: Mapping[str, Any],
    volume_loader: VolumeLoader,
    source_file_hasher: FileHasher = sha256_file,
    resume: bool = False,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Materialize deterministic ``N_d(x_d)`` records without any target input."""

    resolved = validate_canonical_artifact_config(resolved_config)
    expected_config = CanonicalVolumeBuildConfig.from_mapping(resolved, out_dir=config.out_dir)
    if config != expected_config:
        raise ValueError("Canonical-volume runtime config differs from the resolved config.")
    _require_sha256(source_split_file_sha256, "source split file")
    _require_sha256(photometry_artifact_file_sha256, "photometry artifact file")
    _require_sha256(qualification_file_sha256, "qualification file")
    if source_membership_fingerprint != artifact.split_fingerprint:
        raise ValueError("Canonical-volume membership fingerprint differs from photometry.")
    if source_recovery_fingerprint != artifact.recovery_fingerprint:
        raise ValueError("Canonical-volume recovery fingerprint differs from photometry.")
    if source_split_file_sha256 != artifact.provenance["source_split_file_sha256"]:
        raise ValueError("Canonical-volume split-file hash differs from photometry.")
    qualified = validate_variant_a_qualification(
        qualification,
        artifact=artifact,
        source_split_file_sha256=source_split_file_sha256,
        source_membership_fingerprint=source_membership_fingerprint,
        source_recovery_fingerprint=source_recovery_fingerprint,
    )
    _require_sha256(str(qualified["result_sha256"]), "qualification result")
    _validate_code_provenance(code_provenance)

    eligible, excluded = preflight_retrospective_records(
        records_by_split,
        splits=config.splits,
        source_file_hasher=source_file_hasher,
    )
    _require_all_domains(eligible, config.splits)
    config_sha = sha256_json(resolved)
    run_identity = {
        "contract_version": CANONICAL_VOLUME_CONTRACT_VERSION,
        "source_split_file_sha256": source_split_file_sha256,
        "source_membership_fingerprint": source_membership_fingerprint,
        "source_recovery_fingerprint": source_recovery_fingerprint,
        "photometry_artifact_sha256": artifact.artifact_sha256,
        "photometry_artifact_file_sha256": photometry_artifact_file_sha256,
        "photometry_resolved_config_sha256": artifact.provenance[
            "resolved_config_sha256"
        ],
        "qualification_result_sha256": qualified["result_sha256"],
        "qualification_file_sha256": qualification_file_sha256,
        "resolved_config_sha256": config_sha,
        "code_provenance_sha256": sha256_json(code_provenance),
        "records": [
            _eligible_source_identity(item)
            for item in sorted(eligible, key=lambda value: (value.split, value.record.case_id))
        ],
        "excluded_prospective_records": excluded,
    }
    run_fingerprint = sha256_json(run_identity)
    out_dir = config.out_dir
    manifest_path = out_dir / CANONICAL_VOLUME_MANIFEST
    if manifest_path.exists() and not resume:
        raise FileExistsError(f"Refusing to overwrite canonical artifact: {manifest_path}")

    records: list[dict[str, Any]] = []
    total = len(eligible)
    for index, item in enumerate(
        sorted(eligible, key=lambda value: (value.split, value.record.case_id)), start=1
    ):
        destination = _canonical_record_path(out_dir, item.split, item.record.case_id)
        resume_key = sha256_json(
            {
                "run_fingerprint": run_fingerprint,
                "record": _eligible_source_identity(item),
            }
        )
        if destination.exists():
            if not resume:
                raise FileExistsError(f"Refusing to overwrite canonical record: {destination}")
            payload = _load_canonical_record(destination, expected_resume_key=resume_key)
        else:
            source = _validated_source_volume(
                volume_loader(item.record), item.record.case_id
            )
            source_loaded_sha = canonical_tensor_sha256(source)
            canonical = artifact.normalize_source(source, item.record.domain)
            values = canonical.values.detach().cpu().to(torch.float32).contiguous()
            support = canonical.support_mask.detach().cpu().to(torch.bool).contiguous()
            metadata = {
                "contract_version": CANONICAL_VOLUME_CONTRACT_VERSION,
                "record_identity": item.record.case_id,
                "record_identity_sha256": sha256_text(item.record.case_id),
                "subject_identity": item.subject_identity,
                "subject_group_identity": item.subject_group_identity,
                "cohort": "R",
                "split": item.split,
                "domain": item.record.domain.to_dict(),
                "source_path_identity_sha256": item.source_path_identity_sha256,
                "source_file_sha256": item.source_file_sha256,
                "source_loaded_array_sha256": source_loaded_sha,
                "source_shape": list(source.shape),
                "source_dtype": _dtype_name(source.dtype),
                "canonical_tensor_sha256": canonical_tensor_sha256(values),
                "canonical_shape": list(values.shape),
                "canonical_dtype": _dtype_name(values.dtype),
                "support_tensor_sha256": storage_tensor_sha256(support),
                "support_shape": list(support.shape),
                "support_dtype": _dtype_name(support.dtype),
                "support_nonzero_count": int(support.sum()),
                "photometry_artifact_sha256": artifact.artifact_sha256,
                "photometry_resolved_config_sha256": artifact.provenance[
                    "resolved_config_sha256"
                ],
                "source_split_file_sha256": source_split_file_sha256,
                "source_membership_fingerprint": source_membership_fingerprint,
                "source_recovery_fingerprint": source_recovery_fingerprint,
                "code_provenance_sha256": sha256_json(code_provenance),
                "normalization_path": "FrozenPhotometryArtifact.normalize_source",
                "support_policy": PHOTOMETRY_SUPPORT_POLICY,
                "resume_key": resume_key,
            }
            metadata["payload_sha256"] = sha256_json(metadata)
            payload = {
                **metadata,
                "canonical_tensor": values,
                "source_support": support,
            }
            atomic_torch_save_no_clobber(destination, payload)
            payload = _load_canonical_record(destination, expected_resume_key=resume_key)
        record_entry = _canonical_manifest_entry(payload, destination, out_dir)
        records.append(record_entry)
        if progress is not None:
            progress("canonical_volume", index, total, item.record.case_id)

    records.sort(key=lambda item: (item["split"], item["record_identity"]))
    counts = _domain_counts(records, config.splits)
    manifest: dict[str, Any] = {
        "contract_version": CANONICAL_VOLUME_CONTRACT_VERSION,
        "semantics": CANONICAL_VOLUME_SEMANTICS,
        "eligibility_rule": CANONICAL_VOLUME_ELIGIBILITY,
        "permitted_splits": list(config.splits),
        "resolved_config": resolved,
        "resolved_config_sha256": config_sha,
        "source_split": {
            "file_sha256": source_split_file_sha256,
            "membership_fingerprint": source_membership_fingerprint,
            "recovery_fingerprint": source_recovery_fingerprint,
        },
        "photometry": {
            "contract_version": PHOTOMETRY_FACTORIZATION_CONTRACT_VERSION,
            "artifact_sha256": artifact.artifact_sha256,
            "artifact_file_sha256": photometry_artifact_file_sha256,
            "resolved_config_sha256": artifact.provenance["resolved_config_sha256"],
        },
        "qualification": {
            "contract_version": qualified["contract_version"],
            "result_sha256": qualified["result_sha256"],
            "file_sha256": qualification_file_sha256,
            "canonical_latent_bank_authorized": True,
            "vae_provenance": dict(qualified["vae_provenance"]),
        },
        "code_provenance": dict(code_provenance),
        "run_fingerprint": run_fingerprint,
        "eligibility_proof": {
            "accepted_cohort": "R",
            "accepted_record_count": len(records),
            "prospective_accepted_count": 0,
            "prospective_excluded_count": len(excluded),
            "classification_completed_before_source_array_load": True,
        },
        "excluded_prospective_records": excluded,
        "excluded_prospective_records_sha256": sha256_json(excluded),
        "domain_counts": counts,
        "record_count": len(records),
        "records": records,
        "records_sha256": sha256_json(records),
    }
    manifest["artifact_sha256"] = sha256_json(manifest)
    write_json_resume_exact(manifest_path, manifest, resume=resume)
    return load_canonical_volume_manifest(out_dir)


def preflight_retrospective_records(
    records_by_split: Mapping[str, Sequence[VolumeRecord]],
    *,
    splits: Sequence[str] = CANONICAL_VOLUME_SPLITS,
    source_file_hasher: FileHasher = sha256_file,
) -> tuple[list[_EligibleRecord], list[dict[str, Any]]]:
    """Classify every identity before hashing or loading any source array."""

    requested = tuple(str(value) for value in splits)
    if not requested or any(value not in CANONICAL_VOLUME_SPLITS for value in requested):
        raise ValueError("Canonical volumes permit only train and validation split roles.")
    classified: list[tuple[VolumeRecord, str, str, str, str]] = []
    excluded: list[dict[str, Any]] = []
    seen: set[str] = set()
    for split in requested:
        for record in records_by_split.get(split, ()):
            if record.case_id in seen:
                raise ValueError("Canonical-volume input contains a duplicate record identity.")
            seen.add(record.case_id)
            if record.split is not None and str(record.split) != split:
                raise ValueError("Canonical-volume record split conflicts with its frozen role.")
            prefix = _record_prefix(record)
            identity = classify_variant_a_cohort(
                case_identity=record.case_id,
                metadata_prefix=prefix,
                supplied_cohort=prefix,
                subject_identity=record.subject_id,
                allowed_cohorts=("R", "P"),
            )
            path_identity_hash = sha256_text(str(record.image_path))
            if identity.cohort == "P":
                excluded.append(
                    {
                        "record_identity": identity.case_identity,
                        "record_identity_sha256": sha256_text(identity.case_identity),
                        "subject_identity": identity.subject_identity,
                        "subject_group_identity": identity.subject_group_identity,
                        "cohort": "P",
                        "split": split,
                        "source_path_identity_sha256": path_identity_hash,
                        "reason": PROSPECTIVE_EXCLUSION_REASON,
                    }
                )
                continue
            classified.append(
                (
                    record,
                    split,
                    identity.subject_identity,
                    identity.subject_group_identity,
                    path_identity_hash,
                )
            )

    eligible: list[_EligibleRecord] = []
    for record, split, subject, group, path_hash in classified:
        source_hash = str(source_file_hasher(record.image_path))
        _require_sha256(source_hash, "source file")
        eligible.append(
            _EligibleRecord(
                record=record,
                split=split,
                subject_identity=subject,
                subject_group_identity=group,
                source_path_identity_sha256=path_hash,
                source_file_sha256=source_hash,
            )
        )
    excluded.sort(key=lambda item: (item["split"], item["record_identity"]))
    return eligible, excluded


def load_canonical_volume_manifest(
    root: str | Path,
    *,
    expected_artifact_sha256: str | None = None,
) -> dict[str, Any]:
    path = Path(root) / CANONICAL_VOLUME_MANIFEST
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not load canonical-volume manifest {path}: {exc}") from exc
    if not isinstance(manifest, Mapping):
        raise ValueError("Canonical-volume manifest root must be a JSON object.")
    payload = _json_safe_mapping(manifest)
    if payload.get("contract_version") != CANONICAL_VOLUME_CONTRACT_VERSION:
        raise ValueError("Canonical-volume contract mismatch.")
    stored_hash = str(payload.get("artifact_sha256", ""))
    unhashed = dict(payload)
    unhashed.pop("artifact_sha256", None)
    if stored_hash != sha256_json(unhashed):
        raise ValueError("Canonical-volume manifest content hash mismatch.")
    if expected_artifact_sha256 is not None and stored_hash != expected_artifact_sha256:
        raise ValueError("Canonical-volume artifact identity mismatch.")
    if payload.get("semantics") != CANONICAL_VOLUME_SEMANTICS:
        raise ValueError("Canonical-volume semantics mismatch.")
    if payload.get("permitted_splits") != list(CANONICAL_VOLUME_SPLITS):
        raise ValueError("Canonical-volume split roles are incompatible.")
    if sha256_json(payload.get("resolved_config")) != payload.get("resolved_config_sha256"):
        raise ValueError("Canonical-volume resolved-config hash mismatch.")
    validate_canonical_artifact_config(payload["resolved_config"])
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("Canonical-volume manifest has no records.")
    if sha256_json(records) != payload.get("records_sha256"):
        raise ValueError("Canonical-volume record-list hash mismatch.")
    identities: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("Canonical-volume manifest contains a malformed record.")
        identity = classify_variant_a_cohort(
            case_identity=str(record.get("record_identity", "")),
            metadata_prefix="R",
            supplied_cohort=str(record.get("cohort", "")),
            subject_identity=record.get("subject_identity"),
            allowed_cohorts=("R",),
        )
        if record.get("subject_group_identity") != identity.subject_group_identity:
            raise ValueError("Canonical-volume subject-group identity mismatch.")
        if record.get("split") not in CANONICAL_VOLUME_SPLITS:
            raise ValueError("Canonical-volume record has a forbidden split role.")
        if identity.case_identity in identities:
            raise ValueError("Canonical-volume manifest has duplicate record identities.")
        identities.add(identity.case_identity)
        _require_sha256(str(record.get("source_file_sha256", "")), "source file")
        _require_sha256(
            str(record.get("canonical_tensor_sha256", "")), "canonical tensor"
        )
        _require_sha256(str(record.get("payload_file_sha256", "")), "record file")
        _safe_relative_path(Path(root), str(record.get("path", "")))
    if payload.get("record_count") != len(records):
        raise ValueError("Canonical-volume record count mismatch.")
    if payload.get("eligibility_proof", {}).get("prospective_accepted_count") != 0:
        raise ValueError("Canonical-volume manifest accepted prospective data.")
    validate_canonical_artifact_code_provenance(payload.get("code_provenance", {}))
    excluded = payload.get("excluded_prospective_records")
    if not isinstance(excluded, list):
        raise ValueError("Canonical-volume prospective exclusions are malformed.")
    if sha256_json(excluded) != payload.get("excluded_prospective_records_sha256"):
        raise ValueError("Canonical-volume prospective-exclusion hash mismatch.")
    if any(item.get("cohort") != "P" for item in excluded):
        raise ValueError("Canonical-volume exclusion evidence contains a non-P record.")
    return payload


def load_canonical_volume_record(
    root: str | Path, entry: Mapping[str, Any]
) -> dict[str, Any]:
    """Load one manifest-bound canonical record and verify its file/content identity."""

    path = _safe_relative_path(Path(root), str(entry.get("path", "")))
    if sha256_file(path) != entry.get("payload_file_sha256"):
        raise ValueError("Canonical-volume record file hash mismatch.")
    payload = _load_canonical_record(
        path, expected_resume_key=str(entry.get("resume_key", ""))
    )
    for key in (
        "record_identity",
        "subject_identity",
        "subject_group_identity",
        "cohort",
        "split",
        "domain",
        "source_file_sha256",
        "source_loaded_array_sha256",
        "canonical_tensor_sha256",
        "support_tensor_sha256",
        "payload_sha256",
    ):
        if payload.get(key) != entry.get(key):
            raise ValueError(f"Canonical-volume record manifest mismatch for {key!r}.")
    return payload


def validate_canonical_artifact_code_provenance(
    provenance: Mapping[str, Any],
) -> None:
    """Public validation boundary shared by both canonical artifact builders."""

    _validate_code_provenance(provenance)


def audit_canonical_volume_artifact(
    *,
    root: str | Path,
    artifact: FrozenPhotometryArtifact,
    qualification: Mapping[str, Any],
    records_by_split: Mapping[str, Sequence[VolumeRecord]],
    source_split_file_sha256: str,
    source_membership_fingerprint: str,
    source_recovery_fingerprint: str,
    photometry_artifact_file_sha256: str,
    qualification_file_sha256: str,
    resolved_config: Mapping[str, Any] | None = None,
    volume_loader: VolumeLoader,
    source_file_hasher: FileHasher = sha256_file,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Hash-audit and recompute every saved ``N_d(x_d)`` tensor and support."""

    manifest = load_canonical_volume_manifest(root)
    if resolved_config is not None:
        resolved = validate_canonical_artifact_config(resolved_config)
        if manifest["resolved_config_sha256"] != sha256_json(resolved):
            raise ValueError("Canonical-volume audit config hash mismatch.")
    validate_variant_a_qualification(
        qualification,
        artifact=artifact,
        source_split_file_sha256=source_split_file_sha256,
        source_membership_fingerprint=source_membership_fingerprint,
        source_recovery_fingerprint=source_recovery_fingerprint,
    )
    expected_inputs = {
        "photometry_artifact_sha256": artifact.artifact_sha256,
        "photometry_artifact_file_sha256": photometry_artifact_file_sha256,
        "qualification_result_sha256": qualification["result_sha256"],
        "qualification_file_sha256": qualification_file_sha256,
    }
    if manifest["photometry"]["artifact_sha256"] != expected_inputs[
        "photometry_artifact_sha256"
    ] or manifest["photometry"]["artifact_file_sha256"] != expected_inputs[
        "photometry_artifact_file_sha256"
    ]:
        raise ValueError("Canonical-volume photometry provenance mismatch.")
    if manifest["qualification"]["result_sha256"] != expected_inputs[
        "qualification_result_sha256"
    ] or manifest["qualification"]["file_sha256"] != expected_inputs[
        "qualification_file_sha256"
    ]:
        raise ValueError("Canonical-volume qualification provenance mismatch.")
    if manifest["source_split"] != {
        "file_sha256": source_split_file_sha256,
        "membership_fingerprint": source_membership_fingerprint,
        "recovery_fingerprint": source_recovery_fingerprint,
    }:
        raise ValueError("Canonical-volume source-split provenance mismatch.")

    eligible, excluded = preflight_retrospective_records(
        records_by_split,
        splits=manifest["permitted_splits"],
        source_file_hasher=source_file_hasher,
    )
    if excluded != manifest["excluded_prospective_records"]:
        raise ValueError("Canonical-volume prospective exclusion evidence changed.")
    by_identity = {item.record.case_id: item for item in eligible}
    if set(by_identity) != {item["record_identity"] for item in manifest["records"]}:
        raise ValueError("Canonical-volume source record membership changed.")
    total = len(manifest["records"])
    for index, entry in enumerate(manifest["records"], start=1):
        item = by_identity[entry["record_identity"]]
        if item.source_file_sha256 != entry["source_file_sha256"]:
            raise ValueError("Canonical-volume source-content hash changed.")
        path = _safe_relative_path(Path(root), entry["path"])
        if sha256_file(path) != entry["payload_file_sha256"]:
            raise ValueError("Canonical-volume record file hash mismatch.")
        payload = _load_canonical_record(path, expected_resume_key=entry["resume_key"])
        source = _validated_source_volume(volume_loader(item.record), item.record.case_id)
        canonical = artifact.normalize_source(source, item.record.domain)
        values = canonical.values.detach().cpu().to(torch.float32).contiguous()
        support = canonical.support_mask.detach().cpu().to(torch.bool).contiguous()
        if canonical_tensor_sha256(source) != payload["source_loaded_array_sha256"]:
            raise ValueError("Canonical-volume loaded source tensor identity changed.")
        if canonical_tensor_sha256(values) != payload["canonical_tensor_sha256"]:
            raise ValueError("Canonical-volume saved tensor differs from N_d(source).")
        if storage_tensor_sha256(support) != payload["support_tensor_sha256"]:
            raise ValueError("Canonical-volume saved support differs from source support.")
        if progress is not None:
            progress("audit_canonical_volume", index, total, item.record.case_id)
    return {
        "contract_version": CANONICAL_VOLUME_CONTRACT_VERSION,
        "artifact_sha256": manifest["artifact_sha256"],
        "record_count": total,
        "all_records_verified": True,
        "all_records_retrospective": True,
        "source_content_verified": True,
    }


def storage_tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash tensor shape, actual dtype, and contiguous stored bytes."""

    work = tensor.detach().cpu().contiguous()
    header = json.dumps(
        {"dtype": _dtype_name(work.dtype), "shape": list(work.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    raw = work.view(torch.uint8).numpy().tobytes(order="C")
    import hashlib

    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(raw)
    return digest.hexdigest()


def atomic_torch_save_no_clobber(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {target}")
    temporary = target.with_name(
        f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        torch.save(dict(payload), temporary)
        with temporary.open("r+b") as handle:
            os.fsync(handle.fileno())
        _publish_no_clobber(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


def write_json_resume_exact(
    path: str | Path, payload: Mapping[str, Any], *, resume: bool
) -> Path:
    target = Path(path)
    normalized = _json_safe_mapping(payload)
    if target.exists():
        if not resume:
            raise FileExistsError(f"Refusing to overwrite existing artifact: {target}")
        try:
            current = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not resume JSON artifact {target}: {exc}") from exc
        if current != normalized:
            raise ValueError(f"Existing artifact is incompatible with exact resume: {target}")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(
        f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        encoded = json.dumps(normalized, indent=2, sort_keys=True, allow_nan=False) + "\n"
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        _publish_no_clobber(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


def _publish_no_clobber(temporary: Path, target: Path) -> None:
    try:
        os.link(temporary, target)
    except FileExistsError as exc:
        raise FileExistsError(f"Refusing to overwrite existing artifact: {target}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _load_canonical_record(
    path: Path, *, expected_resume_key: str | None = None
) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(f"Could not load canonical-volume record {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Canonical-volume record payload must be a mapping.")
    result = dict(payload)
    if result.get("contract_version") != CANONICAL_VOLUME_CONTRACT_VERSION:
        raise ValueError("Canonical-volume record contract mismatch.")
    if expected_resume_key is not None and result.get("resume_key") != expected_resume_key:
        raise ValueError("Canonical-volume record is incompatible with exact resume.")
    values = result.get("canonical_tensor")
    support = result.get("source_support")
    if not isinstance(values, torch.Tensor) or not isinstance(support, torch.Tensor):
        raise ValueError("Canonical-volume record tensors are missing.")
    if values.dtype != torch.float32 or support.dtype != torch.bool:
        raise ValueError("Canonical-volume tensor dtype contract mismatch.")
    if tuple(values.shape) != tuple(support.shape):
        raise ValueError("Canonical-volume tensor/support shape mismatch.")
    if canonical_tensor_sha256(values) != result.get("canonical_tensor_sha256"):
        raise ValueError("Canonical-volume record tensor hash mismatch.")
    if storage_tensor_sha256(support) != result.get("support_tensor_sha256"):
        raise ValueError("Canonical-volume record support hash mismatch.")
    if result.get("canonical_shape") != list(values.shape) or result.get(
        "support_shape"
    ) != list(support.shape):
        raise ValueError("Canonical-volume record shape identity mismatch.")
    if result.get("canonical_dtype") != "float32" or result.get("support_dtype") != "bool":
        raise ValueError("Canonical-volume record dtype identity mismatch.")
    metadata = {key: value for key, value in result.items() if key not in {
        "canonical_tensor", "source_support", "payload_sha256"
    }}
    if result.get("payload_sha256") != sha256_json(metadata):
        raise ValueError("Canonical-volume record metadata hash mismatch.")
    return result


def _canonical_manifest_entry(
    payload: Mapping[str, Any], path: Path, root: Path
) -> dict[str, Any]:
    keys = (
        "record_identity",
        "record_identity_sha256",
        "subject_identity",
        "subject_group_identity",
        "cohort",
        "split",
        "domain",
        "source_path_identity_sha256",
        "source_file_sha256",
        "source_loaded_array_sha256",
        "source_shape",
        "source_dtype",
        "canonical_tensor_sha256",
        "canonical_shape",
        "canonical_dtype",
        "support_tensor_sha256",
        "support_shape",
        "support_dtype",
        "support_nonzero_count",
        "code_provenance_sha256",
        "normalization_path",
        "support_policy",
        "resume_key",
        "payload_sha256",
    )
    return {
        **{key: payload[key] for key in keys},
        "path": path.relative_to(root).as_posix(),
        "payload_file_sha256": sha256_file(path),
    }


def _eligible_source_identity(item: _EligibleRecord) -> dict[str, Any]:
    return {
        "record_identity": item.record.case_id,
        "record_identity_sha256": sha256_text(item.record.case_id),
        "subject_identity": item.subject_identity,
        "subject_group_identity": item.subject_group_identity,
        "cohort": "R",
        "split": item.split,
        "domain": item.record.domain.to_dict(),
        "source_path_identity_sha256": item.source_path_identity_sha256,
        "source_file_sha256": item.source_file_sha256,
    }


def _canonical_record_path(root: Path, split: str, case_id: str) -> Path:
    return root / "records" / split / f"{sha256_text(case_id)}.pt"


def _record_prefix(record: VolumeRecord) -> str:
    prefix = record.metadata.get("prefix") if isinstance(record.metadata, Mapping) else None
    if prefix is None or not str(prefix).strip():
        raise ValueError(f"Canonical-volume record {record.case_id!r} has no R/P prefix.")
    return str(prefix).strip().upper()


def _validated_source_volume(volume: torch.Tensor, identity: str) -> torch.Tensor:
    if not isinstance(volume, torch.Tensor):
        raise TypeError(f"Canonical-volume source {identity!r} must be a tensor.")
    if volume.ndim != 5 or tuple(volume.shape[:2]) != (1, 1):
        raise ValueError(
            f"Canonical-volume source must be (1,1,X,Y,Z), got {tuple(volume.shape)}."
        )
    if volume.dtype != torch.float32 or not bool(torch.isfinite(volume).all()):
        raise ValueError("Canonical-volume source must be finite float32.")
    if float(volume.min()) < 0.0 or float(volume.max()) > 1.0:
        raise ValueError("Canonical-volume source violates the official [0,1] range.")
    return volume


def _require_all_domains(records: Sequence[_EligibleRecord], splits: Sequence[str]) -> None:
    expected = {
        Domain(field, contrast).label
        for contrast in CONTRASTS
        for field in FIELD_STRENGTHS_T
    }
    for split in splits:
        actual = {item.record.domain.label for item in records if item.split == split}
        if actual != expected:
            raise ValueError(
                f"Canonical-volume split {split!r} must contain all 15 domains; "
                f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}."
            )


def _domain_counts(
    records: Sequence[Mapping[str, Any]], splits: Sequence[str]
) -> dict[str, dict[str, int]]:
    return {
        split: dict(
            sorted(
                Counter(
                    Domain.from_dict(dict(item["domain"])).label
                    for item in records
                    if item["split"] == split
                ).items()
            )
        )
        for split in splits
    }


def _safe_relative_path(root: Path, relative: str) -> Path:
    if not relative:
        raise ValueError("Artifact record path is empty.")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("Artifact record path escapes its root.") from exc
    return path


def _validate_code_provenance(provenance: Mapping[str, Any]) -> None:
    if not isinstance(provenance, Mapping):
        raise ValueError("Canonical-artifact code provenance must be a mapping.")
    if provenance.get("checkout_clean") is not True:
        raise ValueError("Canonical-artifact production requires a clean checkout.")
    commit = str(provenance.get("git_head", ""))
    if not commit:
        raise ValueError("Canonical-artifact code provenance has no Git commit.")
    hashes = provenance.get("module_sha256")
    if not isinstance(hashes, Mapping) or set(hashes) != set(
        CANONICAL_VOLUME_SOURCE_MODULES
    ):
        raise ValueError("Canonical-artifact source-module hashes are incomplete.")
    for value in hashes.values():
        _require_sha256(str(value), "source module")


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _require_sha256(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"Canonical-artifact {name} SHA-256 is invalid.")


def _json_safe_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    encoded = json.dumps(dict(value), sort_keys=True, allow_nan=False)
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):
        raise TypeError("Canonical-artifact payload must be a mapping.")
    return decoded


__all__ = [
    "CANONICAL_ARTIFACT_CONFIG_VERSION",
    "CANONICAL_VOLUME_CONTRACT_VERSION",
    "CANONICAL_VOLUME_MANIFEST",
    "CANONICAL_VOLUME_SOURCE_MODULES",
    "CANONICAL_VOLUME_SPLITS",
    "CanonicalVolumeBuildConfig",
    "atomic_torch_save_no_clobber",
    "audit_canonical_volume_artifact",
    "build_canonical_volume_artifact",
    "capture_canonical_artifact_code_provenance",
    "load_canonical_volume_manifest",
    "load_canonical_volume_record",
    "load_variant_a_qualification",
    "preflight_retrospective_records",
    "storage_tensor_sha256",
    "validate_canonical_artifact_config",
    "validate_canonical_artifact_code_provenance",
    "validate_variant_a_qualification",
    "write_json_resume_exact",
]
