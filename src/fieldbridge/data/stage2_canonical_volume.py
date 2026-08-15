"""Shared contracts for streamed Stage-2 canonical-latent artifacts.

``stage2-canonical-volume-v1`` is an in-memory, hash-sealed computation boundary.  The
primary artifact path deliberately does not persist the full float32 ``N_d(x_d)`` tensor or
its full-resolution Boolean support.  This module owns the fail-closed cohort preflight,
computational provenance, and atomic no-clobber publication primitives used by the streamed
bank builder.
"""

from __future__ import annotations

import errno
import json
import os
import platform
import shutil
import subprocess
import sys
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fieldbridge.data.contracts import VolumeRecord
from fieldbridge.data.domains import CONTRASTS, FIELD_STRENGTHS_T, Domain
from fieldbridge.data.photometry_factorization import (
    PHOTOMETRY_SUPPORT_POLICY,
    FrozenPhotometryArtifact,
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
CANONICAL_STREAM_SEMANTICS = "ephemeral-N_d-hash-then-immediate-full-encode-v1"
CANONICAL_VOLUME_SPLITS = ("train", "validation")
PROSPECTIVE_EXCLUSION_REASON = "prospective-cohort-rejected-before-source-array-load"
COMPUTATIONAL_PROVENANCE_VERSION = "stage2-computational-provenance-v1"
DEPENDENCY_MAP_VERSION = "stage2-reviewed-dependency-map-v1"
FILESYSTEM_PREFLIGHT_VERSION = "atomic-hardlink-no-clobber-preflight-v1"
STORAGE_PREFLIGHT_VERSION = "stage2-streamed-storage-preflight-v1"

# This map is intentionally explicit.  Every checked-in production module involved in
# classification/loading, N_d, VAE construction/encoding, support, derivation, or config and
# checkpoint loading is part of the artifact identity.
REVIEWED_DEPENDENCY_MAP: dict[str, tuple[str, ...]] = {
    "src/fieldbridge/cli.py": (
        "configuration-loading",
        "checkpoint-loading",
        "vae-construction",
        "command-routing",
    ),
    "src/fieldbridge/config/__init__.py": ("configuration-loading",),
    "src/fieldbridge/data/contracts.py": ("record-contract",),
    "src/fieldbridge/data/domains.py": ("domain-contract",),
    "src/fieldbridge/data/sources.py": ("source-loading", "source-shape-preflight"),
    "src/fieldbridge/data/transforms.py": ("source-range-validation",),
    "src/fieldbridge/data/vae_splits.py": (
        "split-loading",
        "split-classification",
        "split-fingerprints",
    ),
    "src/fieldbridge/data/photometry_factorization.py": (
        "N_d",
        "source-support-generation",
        "photometry-artifact-loading",
    ),
    "src/fieldbridge/data/stage2_canonical_volume.py": (
        "cohort-preflight",
        "computational-provenance",
        "atomic-publication",
        "storage-preflight",
    ),
    "src/fieldbridge/data/photometry_factored_latent_bank.py": (
        "streamed-build-audit",
        "support-propagation-packing",
        "masked-statistics",
        "structural-descriptors",
    ),
    "src/fieldbridge/data/latent_bank.py": ("encode_latent", "full-encoding"),
    "src/fieldbridge/models/autoencoders/base.py": ("vae-encoder-contract",),
    "src/fieldbridge/models/autoencoders/kl_vae.py": (
        "vae-construction",
        "encoder-forward-arithmetic",
    ),
    "src/fieldbridge/models/diffusion/field_conditioner.py": (
        "kl-vae-module-dependency",
    ),
    "src/fieldbridge/models/film.py": ("kl-vae-module-dependency",),
    "src/fieldbridge/training/checkpoints.py": ("checkpoint-loading",),
    "src/fieldbridge/training/train_loop.py": ("frozen-module-validation",),
    "src/fieldbridge/evaluation/stage2_photometry_baseline.py": (
        "variant-a-qualification-contract",
    ),
}

VolumeLoader = Callable[[VolumeRecord], torch.Tensor]
SourceShapeResolver = Callable[[VolumeRecord], Sequence[int]]
FileHasher = Callable[[str | Path], str]
LinkPath = str | bytes | os.PathLike[str] | os.PathLike[bytes]
Linker = Callable[[LinkPath, LinkPath], None]


@dataclass(frozen=True, slots=True)
class EligibleRecord:
    record: VolumeRecord
    split: str
    subject_identity: str
    subject_group_identity: str
    source_path_identity_sha256: str
    source_file_sha256: str


class AtomicPublicationUnavailable(RuntimeError):
    """Raised when the target filesystem cannot guarantee atomic no-clobber publication."""


def validate_canonical_artifact_config(data: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the exact streamed, retrospective-only v1 configuration."""

    config = _json_safe_mapping(data)
    if config.get("contract") != CANONICAL_ARTIFACT_CONFIG_VERSION:
        raise ValueError("Stage-2 canonical-artifact config contract is incompatible.")
    if config.get("scope") != "retrospective_only_canonical_artifacts":
        raise ValueError("Stage-2 canonical-artifact config scope is incompatible.")
    if config.get("eligibility") != {
        "cohort": "R",
        "splits": list(CANONICAL_VOLUME_SPLITS),
        "prospective_records": "exclude_before_array_load",
    }:
        raise ValueError("Canonical artifacts require exact R-only train/validation eligibility.")
    if config.get("canonical_stream") != {
        "contract": CANONICAL_VOLUME_CONTRACT_VERSION,
        "semantics": CANONICAL_STREAM_SEMANTICS,
        "persistence": "hashes-and-provenance-only",
        "support": PHOTOMETRY_SUPPORT_POLICY,
        "source": "FrozenPhotometryArtifact.normalize_source",
        "debug_full_volume_artifact": "disabled",
    }:
        raise ValueError("Canonical stream settings differ from the reviewed v1 contract.")

    latent = config.get("latent_bank")
    if not isinstance(latent, Mapping) or set(latent) != {
        "contract",
        "encoder_statistic",
        "strategy",
        "path_used_required",
        "store_dtype",
        "precision",
        "support_propagation",
        "support_packing",
    }:
        raise ValueError("Photometry-factored latent-bank settings are incomplete.")
    if latent.get("contract") != "photometry-factored-latent-bank-v1":
        raise ValueError("Photometry-factored latent-bank contract is incompatible.")
    if latent.get("encoder_statistic") != "posterior_mean":
        raise ValueError("The factored bank must use the frozen VAE posterior mean.")
    if latent.get("strategy") != "full" or latent.get("path_used_required") != "full":
        raise ValueError("Photometry-factored latent-bank v1 requires exact full encoding.")
    if latent.get("store_dtype") not in {"float16", "float32"}:
        raise ValueError("Factored-bank store dtype must be float16 or float32.")
    if latent.get("precision") != "float32":
        raise ValueError("Primary factored-bank v1 requires float32 encoder arithmetic.")
    if latent.get("support_propagation") != "frozen-encoder-dependency-propagation-v1":
        raise ValueError("Factored-bank support propagation is incompatible.")
    if latent.get("support_packing") != "numpy-packbits-c-order-little-bit-v1":
        raise ValueError("Factored-bank support packing is incompatible.")

    if config.get("statistics") != {
        "contract": "photometry-factored-latent-statistics-v1",
        "computed_over": {"cohort": "R", "split": "train", "cells": "supported_only"},
        "accumulation": "channelwise-masked-welford-float64-v1",
    }:
        raise ValueError("Canonical latent statistics must be masked R/train Welford stats.")
    if config.get("structural_descriptor") != {
        "contract": "photometry-factored-structural-descriptor-v1",
        "computed_over": {"cohort": "R", "split": "train"},
        "input": "canonical_standardized_supported_latent_only",
        "pool_output_sizes": [1, 2, 4],
        "pooling": "support-normalized-adaptive-average-3d-v1",
        "gradients": "absolute-first-forward-difference-xyz-v1",
        "dtype": "float32",
        "paired_endpoint_or_target_input": "forbidden",
        "coupling_authorized": False,
        "qualification_required": (
            "photometry-factored-structural-descriptor-qualification-v1"
        ),
    }:
        raise ValueError("Structural-descriptor v1 settings differ from the reviewed contract.")
    if config.get("publication") != {
        "atomic": "same-directory-hardlink-publication-v1",
        "no_clobber": True,
        "filesystem_preflight": FILESYSTEM_PREFLIGHT_VERSION,
        "unsupported_filesystem_action": "stop-and-stage-on-local-scratch",
    }:
        raise ValueError("Artifact publication settings are not fail-closed.")
    storage = config.get("storage_preflight")
    if storage != {
        "contract": STORAGE_PREFLIGHT_VERSION,
        "shape_source": "header-or-injected-resolver-without-array-load-v1",
        "per_record_sidecar_estimate_bytes": 16384,
        "fixed_artifact_overhead_bytes": 1048576,
    }:
        raise ValueError("Streamed storage-preflight settings are incompatible.")
    reject_target_or_prediction_derived_fields(config)
    return config


def capture_canonical_artifact_code_provenance(
    repo_root: str | Path | None = None,
    *,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Capture the reviewed dependency map plus language/framework/runtime identities."""

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
    dependency_map = {
        key: list(value) for key, value in sorted(REVIEWED_DEPENDENCY_MAP.items())
    }
    runtime = _runtime_provenance(torch.device(device))
    payload: dict[str, Any] = {
        "contract_version": COMPUTATIONAL_PROVENANCE_VERSION,
        "git_head": head,
        "checkout_clean": not bool(status.strip()),
        "dependency_map_version": DEPENDENCY_MAP_VERSION,
        "dependency_map": dependency_map,
        "dependency_map_sha256": sha256_json(
            {"version": DEPENDENCY_MAP_VERSION, "modules": dependency_map}
        ),
        "module_sha256": {
            relative: sha256_file(root / relative) for relative in dependency_map
        },
        "runtime": runtime,
        "runtime_sha256": sha256_json(runtime),
    }
    payload["provenance_sha256"] = sha256_json(payload)
    return payload


def validate_canonical_artifact_code_provenance(provenance: Mapping[str, Any]) -> None:
    if not isinstance(provenance, Mapping):
        raise ValueError("Canonical-artifact computational provenance must be a mapping.")
    value = _json_safe_mapping(provenance)
    if value.get("contract_version") != COMPUTATIONAL_PROVENANCE_VERSION:
        raise ValueError("Computational provenance contract is incompatible.")
    if value.get("checkout_clean") is not True or not str(value.get("git_head", "")):
        raise ValueError("Canonical-artifact production requires a clean identified checkout.")
    expected_map = {
        key: list(item) for key, item in sorted(REVIEWED_DEPENDENCY_MAP.items())
    }
    if value.get("dependency_map_version") != DEPENDENCY_MAP_VERSION:
        raise ValueError("Computational dependency-map version is incompatible.")
    if value.get("dependency_map") != expected_map:
        raise ValueError("Computational dependency map is incomplete or changed.")
    expected_map_hash = sha256_json(
        {"version": DEPENDENCY_MAP_VERSION, "modules": expected_map}
    )
    if value.get("dependency_map_sha256") != expected_map_hash:
        raise ValueError("Computational dependency-map hash mismatch.")
    hashes = value.get("module_sha256")
    if not isinstance(hashes, Mapping) or set(hashes) != set(expected_map):
        raise ValueError("Computational source-module identities are incomplete.")
    for module_hash in hashes.values():
        _require_sha256(str(module_hash), "source module")
    runtime = value.get("runtime")
    if not isinstance(runtime, Mapping) or value.get("runtime_sha256") != sha256_json(runtime):
        raise ValueError("Computational runtime provenance hash mismatch.")
    _validate_runtime(runtime)
    stored = str(value.get("provenance_sha256", ""))
    unhashed = dict(value)
    unhashed.pop("provenance_sha256", None)
    if stored != sha256_json(unhashed):
        raise ValueError("Computational provenance content hash mismatch.")


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
    """Require a hash-valid passing Variant-A authorization artifact."""

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
    expected_split = {
        "file_sha256": source_split_file_sha256,
        "membership_fingerprint": source_membership_fingerprint,
        "recovery_fingerprint": source_recovery_fingerprint,
    }
    if result.get("source_split") != expected_split:
        raise ValueError("Variant-A qualification source-split identity mismatch.")
    if result.get("failure_classification") != [] or result.get(
        "canonical_latent_bank_authorized"
    ) is not True:
        raise ValueError("Variant-A qualification did not authorize a canonical latent bank.")
    vae = result.get("vae_provenance")
    if not isinstance(vae, Mapping):
        raise ValueError("Variant-A qualification is missing frozen-VAE provenance.")
    _require_sha256(str(vae.get("config_file_sha256", "")), "VAE config")
    _require_sha256(str(vae.get("checkpoint_sha256", "")), "VAE checkpoint")
    if vae.get("encoder_statistic") != "posterior_mean":
        raise ValueError("Variant-A qualification did not seal the posterior mean.")
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


def preflight_retrospective_records(
    records_by_split: Mapping[str, Sequence[VolumeRecord]],
    *,
    splits: Sequence[str] = CANONICAL_VOLUME_SPLITS,
    source_file_hasher: FileHasher = sha256_file,
) -> tuple[list[EligibleRecord], list[dict[str, Any]]]:
    """Classify every identity before hashing or loading any source array."""

    requested = tuple(str(value) for value in splits)
    if requested != CANONICAL_VOLUME_SPLITS:
        raise ValueError("Canonical artifacts permit exactly train and validation roles.")
    classified: list[tuple[VolumeRecord, str, str, str, str]] = []
    excluded: list[dict[str, Any]] = []
    seen: set[str] = set()
    for split in requested:
        for record in records_by_split.get(split, ()):
            if record.case_id in seen:
                raise ValueError("Canonical-artifact input contains a duplicate record identity.")
            seen.add(record.case_id)
            if record.split is not None and str(record.split) != split:
                raise ValueError("Canonical-artifact record split conflicts with its frozen role.")
            prefix = _record_prefix(record)
            identity = classify_variant_a_cohort(
                case_identity=record.case_id,
                metadata_prefix=prefix,
                supplied_cohort=prefix,
                subject_identity=record.subject_id,
                allowed_cohorts=("R", "P"),
            )
            path_hash = sha256_text(str(record.image_path))
            if identity.cohort == "P":
                excluded.append(
                    {
                        "record_identity": identity.case_identity,
                        "record_identity_sha256": sha256_text(identity.case_identity),
                        "subject_identity": identity.subject_identity,
                        "subject_group_identity": identity.subject_group_identity,
                        "cohort": "P",
                        "split": split,
                        "source_path_identity_sha256": path_hash,
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
                    path_hash,
                )
            )

    eligible: list[EligibleRecord] = []
    for record, split, subject, group, path_hash in classified:
        source_hash = str(source_file_hasher(record.image_path))
        _require_sha256(source_hash, "source file")
        eligible.append(
            EligibleRecord(
                record=record,
                split=split,
                subject_identity=subject,
                subject_group_identity=group,
                source_path_identity_sha256=path_hash,
                source_file_sha256=source_hash,
            )
        )
    eligible.sort(key=lambda item: (item.split, item.record.case_id))
    excluded.sort(key=lambda item: (item["split"], item["record_identity"]))
    require_all_domains(eligible)
    return eligible, excluded


def resolve_source_shapes(
    records: Sequence[EligibleRecord], resolver: SourceShapeResolver
) -> dict[str, tuple[int, int, int, int, int]]:
    """Resolve source shapes without loading voxel arrays."""

    result: dict[str, tuple[int, int, int, int, int]] = {}
    for item in records:
        raw = tuple(int(value) for value in resolver(item.record))
        shape = (1, 1, *raw) if len(raw) == 3 else raw
        if len(shape) != 5 or shape[:2] != (1, 1) or any(value <= 0 for value in shape):
            raise ValueError(
                f"Source-shape preflight requires (1,1,X,Y,Z), got {shape} for "
                f"{item.record.case_id!r}."
            )
        result[item.record.case_id] = shape  # type: ignore[assignment]
    return result


def build_storage_preflight_report(
    *,
    records: Sequence[EligibleRecord],
    source_shapes: Mapping[str, Sequence[int]],
    downsample_factor: int,
    latent_channels: int,
    store_dtype: str,
    descriptor_pool_sizes: Sequence[int],
    resolved_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Estimate avoided full-volume persistence and required streamed/output storage."""

    if downsample_factor <= 0 or latent_channels <= 0:
        raise ValueError("Storage preflight requires positive VAE factor and channels.")
    item_bytes = {"float16": 2, "float32": 4}.get(store_dtype)
    if item_bytes is None:
        raise ValueError("Storage preflight received an unsupported latent dtype.")
    config = validate_canonical_artifact_config(resolved_config)
    storage_config = config["storage_preflight"]
    total_voxels = 0
    train_count = 0
    latent_bytes = 0
    packed_support_bytes = 0
    largest_record_output = 0
    peak_working = 0
    shapes: list[dict[str, Any]] = []
    for item in records:
        shape = tuple(int(value) for value in source_shapes[item.record.case_id])
        if len(shape) != 5 or shape[:2] != (1, 1):
            raise ValueError("Storage-preflight source shape is incompatible.")
        spatial = shape[-3:]
        if any(value % downsample_factor != 0 for value in spatial):
            raise ValueError(
                f"Source shape {spatial} is not divisible by frozen-VAE factor "
                f"{downsample_factor}."
            )
        voxels = int(np.prod(spatial, dtype=np.int64))
        latent_cells = int(
            np.prod(tuple(value // downsample_factor for value in spatial), dtype=np.int64)
        )
        record_latent_bytes = latent_channels * latent_cells * item_bytes
        record_support_bytes = (latent_cells + 7) // 8
        sidecar_bytes = int(storage_config["per_record_sidecar_estimate_bytes"])
        record_output = record_latent_bytes + record_support_bytes + sidecar_bytes
        total_voxels += voxels
        train_count += int(item.split == "train")
        latent_bytes += record_latent_bytes
        packed_support_bytes += record_support_bytes
        largest_record_output = max(largest_record_output, record_output)
        # Source float32 + canonical float32 + full Boolean support + full float32 latent.
        peak_working = max(
            peak_working,
            voxels * 9 + latent_channels * latent_cells * 4,
        )
        shapes.append(
            {
                "record_identity_sha256": sha256_text(item.record.case_id),
                "split": item.split,
                "source_shape": list(shape),
                "source_voxels": voxels,
                "predicted_latent_shape": [
                    latent_channels,
                    *(value // downsample_factor for value in spatial),
                ],
            }
        )
    descriptor_values = (
        latent_channels * sum(int(size) ** 3 for size in descriptor_pool_sizes) * 4
    )
    descriptor_bytes = train_count * descriptor_values * 4
    sidecar_total = len(records) * int(storage_config["per_record_sidecar_estimate_bytes"])
    fixed_overhead = int(storage_config["fixed_artifact_overhead_bytes"])
    output_bytes = (
        latent_bytes
        + packed_support_bytes
        + sidecar_total
        + descriptor_bytes
        + fixed_overhead
    )
    temporary_bytes = max(largest_record_output, descriptor_values * 4, fixed_overhead)
    report: dict[str, Any] = {
        "contract_version": STORAGE_PREFLIGHT_VERSION,
        "record_count": len(records),
        "train_record_count": train_count,
        "source_voxels": total_voxels,
        "full_canonical_float32_bytes_avoided": total_voxels * 4,
        "full_boolean_support_bytes_avoided": total_voxels,
        "full_volume_bytes_avoided": total_voxels * 5,
        "predicted_latent_bytes": latent_bytes,
        "predicted_packed_support_bytes": packed_support_bytes,
        "predicted_descriptor_bytes": descriptor_bytes,
        "predicted_output_storage_bytes": output_bytes,
        "required_temporary_storage_bytes": temporary_bytes,
        "required_free_storage_bytes": output_bytes + temporary_bytes,
        "peak_streamed_working_set_bytes": peak_working,
        "canonical_persistence": "none-primary-streamed-path",
        "source_shapes_sha256": sha256_json(shapes),
        "source_shapes": shapes,
    }
    report["report_sha256"] = sha256_json(report)
    return report


def preflight_atomic_no_clobber_filesystem(
    root: str | Path,
    *,
    required_free_bytes: int,
    linker: Linker = os.link,
) -> dict[str, Any]:
    """Prove hard-link no-clobber semantics and capacity before source processing."""

    output = Path(root)
    output.mkdir(parents=True, exist_ok=True)
    token = f"{os.getpid()}.{uuid.uuid4().hex}"
    temporary = output / f".fieldbridge-atomic-probe.{token}.tmp"
    destination = output / f".fieldbridge-atomic-probe.{token}.published"
    try:
        with temporary.open("xb") as handle:
            handle.write(b"fieldbridge-atomic-probe-v1")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            linker(temporary, destination)
        except OSError as exc:
            raise _atomic_unavailable(output, exc) from exc
        if destination.read_bytes() != b"fieldbridge-atomic-probe-v1":
            raise AtomicPublicationUnavailable(
                "Atomic publication probe did not preserve bytes; use local scratch staging "
                "followed by hash-verified archival."
            )
        try:
            linker(temporary, destination)
        except FileExistsError:
            pass
        except OSError as exc:
            raise _atomic_unavailable(output, exc) from exc
        else:
            raise AtomicPublicationUnavailable(
                "Target filesystem did not enforce no-clobber link publication; use local "
                "scratch staging followed by hash-verified archival."
            )
        free = int(shutil.disk_usage(output).free)
        required = int(required_free_bytes)
        if required < 0:
            raise ValueError("Required free storage must be non-negative.")
        if free < required:
            raise OSError(
                f"Insufficient output storage: required={required} bytes, free={free} bytes."
            )
        report = {
            "contract_version": FILESYSTEM_PREFLIGHT_VERSION,
            "atomic_no_clobber": True,
            "hardlink_publication": True,
            "overwrite_fallback": "forbidden",
            "available_bytes": free,
            "required_free_bytes": required,
            "operator_action_on_failure": (
                "use-local-scratch-staging-then-hash-verified-archival"
            ),
        }
        report["report_sha256"] = sha256_json(report)
        return report
    finally:
        temporary.unlink(missing_ok=True)
        destination.unlink(missing_ok=True)


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


def atomic_torch_save_no_clobber(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    linker: Linker = os.link,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {target}")
    temporary = target.with_name(f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(dict(payload), temporary)
        with temporary.open("r+b") as handle:
            os.fsync(handle.fileno())
        _publish_no_clobber(temporary, target, linker=linker)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


def write_json_resume_exact(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    resume: bool,
    linker: Linker = os.link,
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
    temporary = target.with_name(f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        encoded = json.dumps(normalized, indent=2, sort_keys=True, allow_nan=False) + "\n"
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        _publish_no_clobber(temporary, target, linker=linker)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


def eligible_source_identity(
    item: EligibleRecord, source_shape: Sequence[int]
) -> dict[str, Any]:
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
        "source_shape": [int(value) for value in source_shape],
    }


def require_all_domains(records: Sequence[EligibleRecord]) -> None:
    expected = {
        Domain(field, contrast).label
        for contrast in CONTRASTS
        for field in FIELD_STRENGTHS_T
    }
    for split in CANONICAL_VOLUME_SPLITS:
        actual = {item.record.domain.label for item in records if item.split == split}
        if actual != expected:
            raise ValueError(
                f"Canonical-artifact split {split!r} must contain all 15 domains; "
                f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}."
            )


def validated_source_volume(volume: torch.Tensor, identity: str) -> torch.Tensor:
    if not isinstance(volume, torch.Tensor):
        raise TypeError(f"Canonical source {identity!r} must be a tensor.")
    if volume.ndim != 5 or tuple(volume.shape[:2]) != (1, 1):
        raise ValueError(f"Canonical source must be (1,1,X,Y,Z), got {tuple(volume.shape)}.")
    if volume.dtype != torch.float32 or not bool(torch.isfinite(volume).all()):
        raise ValueError("Canonical source must be finite float32.")
    if float(volume.min()) < 0.0 or float(volume.max()) > 1.0:
        raise ValueError("Canonical source violates the official [0,1] range.")
    return volume.detach().cpu().contiguous()


def safe_relative_path(root: str | Path, relative: str) -> Path:
    base = Path(root).resolve()
    path = (base / relative).resolve()
    try:
        path.relative_to(base)
    except ValueError as exc:
        raise ValueError("Artifact record path escapes its artifact root.") from exc
    return path


def require_sha256(value: str, name: str) -> None:
    _require_sha256(value, name)


def json_safe_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return _json_safe_mapping(value)


def dtype_name(dtype: torch.dtype) -> str:
    return _dtype_name(dtype)


def _runtime_provenance(device: torch.device) -> dict[str, Any]:
    index = device.index
    if device.type == "cuda" and index is None and torch.cuda.is_available():
        index = torch.cuda.current_device()
    device_name: str | None = None
    capability: list[int] | None = None
    if device.type == "cuda" and torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(index)
        capability = list(torch.cuda.get_device_capability(index))
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "torch_version": str(torch.__version__),
        "numpy_version": str(np.__version__),
        "cuda_compiled_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cudnn_version": torch.backends.cudnn.version(),
        "device": {
            "type": device.type,
            "index": index,
            "name": device_name,
            "capability": capability,
        },
    }


def _validate_runtime(runtime: Mapping[str, Any]) -> None:
    required = {
        "python_version",
        "python_implementation",
        "platform",
        "torch_version",
        "numpy_version",
        "cuda_compiled_version",
        "cuda_available",
        "cudnn_version",
        "device",
    }
    if set(runtime) != required or not isinstance(runtime.get("device"), Mapping):
        raise ValueError("Computational runtime provenance is incomplete.")
    if set(runtime["device"]) != {"type", "index", "name", "capability"}:
        raise ValueError("Computational device provenance is incomplete.")
    if runtime["device"].get("type") not in {"cpu", "cuda"}:
        raise ValueError("Computational device provenance is incompatible.")


def _record_prefix(record: VolumeRecord) -> str:
    prefix = record.metadata.get("prefix") if isinstance(record.metadata, Mapping) else None
    if prefix is None or not str(prefix).strip():
        raise ValueError(f"Canonical-artifact record {record.case_id!r} has no R/P prefix.")
    return str(prefix).strip().upper()


def _publish_no_clobber(temporary: Path, target: Path, *, linker: Linker = os.link) -> None:
    try:
        linker(temporary, target)
    except FileExistsError as exc:
        raise FileExistsError(f"Refusing to overwrite existing artifact: {target}") from exc
    except OSError as exc:
        raise _atomic_unavailable(target.parent, exc) from exc
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_unavailable(root: Path, error: OSError) -> AtomicPublicationUnavailable:
    reason = errno.errorcode.get(error.errno or 0, str(error.errno))
    return AtomicPublicationUnavailable(
        f"Atomic no-clobber hard-link publication is unavailable at {root} ({reason}). "
        "Stop here; use local scratch staging followed by hash-verified archival. "
        "No overwrite-prone fallback is permitted."
    )


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
    "AtomicPublicationUnavailable",
    "CANONICAL_ARTIFACT_CONFIG_VERSION",
    "CANONICAL_STREAM_SEMANTICS",
    "CANONICAL_VOLUME_CONTRACT_VERSION",
    "CANONICAL_VOLUME_SPLITS",
    "COMPUTATIONAL_PROVENANCE_VERSION",
    "DEPENDENCY_MAP_VERSION",
    "EligibleRecord",
    "FILESYSTEM_PREFLIGHT_VERSION",
    "REVIEWED_DEPENDENCY_MAP",
    "STORAGE_PREFLIGHT_VERSION",
    "atomic_torch_save_no_clobber",
    "build_storage_preflight_report",
    "capture_canonical_artifact_code_provenance",
    "dtype_name",
    "eligible_source_identity",
    "json_safe_mapping",
    "load_variant_a_qualification",
    "preflight_atomic_no_clobber_filesystem",
    "preflight_retrospective_records",
    "require_sha256",
    "resolve_source_shapes",
    "safe_relative_path",
    "storage_tensor_sha256",
    "validate_canonical_artifact_code_provenance",
    "validate_canonical_artifact_config",
    "validate_variant_a_qualification",
    "validated_source_volume",
    "write_json_resume_exact",
]
