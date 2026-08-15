"""Photometry-factored posterior-mean latent bank and structural descriptors.

The bank is a new contract.  It does not mutate, alias, or relabel ``latent-bank-v1``.
Every latent is produced from a verified ``stage2-canonical-volume-v1`` record, so the
only encoding path is ``z_d = E(N_d(x_d))`` with the frozen VAE posterior mean.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from fieldbridge.data.domains import CONTRASTS, FIELD_STRENGTHS_T, Domain
from fieldbridge.data.latent_bank import downsample_factor, encode_latent
from fieldbridge.data.photometry_factorization import (
    FrozenPhotometryArtifact,
    classify_variant_a_cohort,
    sha256_file,
    sha256_json,
    sha256_text,
)
from fieldbridge.data.stage2_canonical_volume import (
    CANONICAL_VOLUME_CONTRACT_VERSION,
    CANONICAL_VOLUME_SPLITS,
    atomic_torch_save_no_clobber,
    load_canonical_volume_manifest,
    load_canonical_volume_record,
    storage_tensor_sha256,
    validate_canonical_artifact_code_provenance,
    validate_canonical_artifact_config,
    validate_variant_a_qualification,
    write_json_resume_exact,
)
from fieldbridge.training.train_loop import assert_frozen

PHOTOMETRY_FACTORED_LATENT_BANK_VERSION = "photometry-factored-latent-bank-v1"
PHOTOMETRY_FACTORED_LATENT_STATS_VERSION = "photometry-factored-latent-statistics-v1"
PHOTOMETRY_FACTORED_DESCRIPTOR_VERSION = (
    "photometry-factored-structural-descriptor-v1"
)
FACTORED_LATENT_BANK_MANIFEST = "photometry_factored_latent_bank_manifest.json"
FACTORED_LATENT_STATS_FILE = "latent_stats.json"
STRUCTURAL_DESCRIPTOR_MANIFEST = "structural_descriptor_manifest.json"
SUPPORT_DOWNSAMPLE_RULE = "all-source-voxels-supported-per-latent-cell-v1"
SUPPORT_PACKING_RULE = "numpy-packbits-c-order-little-bit-v1"
DESCRIPTOR_INPUT = "canonical_standardized_latent_only"
DESCRIPTOR_POOLING = "support-normalized-adaptive-average-3d-v1"
DESCRIPTOR_GRADIENTS = "absolute-first-forward-difference-xyz-v1"

Progress = Callable[[str, int, int, str], None]


@dataclass(frozen=True, slots=True)
class PhotometryFactoredLatentBankConfig:
    out_dir: Path
    strategy: str = "tiled"
    store_dtype: str = "float16"
    precision: str = "float32"
    block_size: tuple[int, int, int] = (128, 128, 128)
    halo: tuple[int, int, int] = (32, 32, 32)
    splits: tuple[str, ...] = CANONICAL_VOLUME_SPLITS
    descriptor_pool_sizes: tuple[int, ...] = (1, 2, 4)

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, Any], *, out_dir: str | Path
    ) -> "PhotometryFactoredLatentBankConfig":
        resolved = validate_canonical_artifact_config(data)
        latent = resolved["latent_bank"]
        descriptor = resolved["structural_descriptor"]
        return cls(
            out_dir=Path(out_dir),
            strategy=str(latent["strategy"]),
            store_dtype=str(latent["store_dtype"]),
            precision=str(latent["precision"]),
            block_size=tuple(int(value) for value in latent["block_size"]),
            halo=tuple(int(value) for value in latent["halo"]),
            splits=tuple(str(value) for value in resolved["eligibility"]["splits"]),
            descriptor_pool_sizes=tuple(
                int(value) for value in descriptor["pool_output_sizes"]
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "out_dir": str(self.out_dir),
            "strategy": self.strategy,
            "store_dtype": self.store_dtype,
            "precision": self.precision,
            "block_size": list(self.block_size),
            "halo": list(self.halo),
            "splits": list(self.splits),
            "descriptor_pool_sizes": list(self.descriptor_pool_sizes),
        }


class _ChannelStats:
    def __init__(self, channels: int) -> None:
        self.channels = int(channels)
        self.total = torch.zeros(self.channels, dtype=torch.float64)
        self.total_squares = torch.zeros(self.channels, dtype=torch.float64)
        self.count = 0

    def update(self, latent: torch.Tensor) -> None:
        work = latent.detach().cpu().to(torch.float64)
        if work.ndim != 4 or work.shape[0] != self.channels:
            raise ValueError("Factored latent statistics require one (C,X,Y,Z) tensor.")
        flat = work.reshape(self.channels, -1)
        self.total += flat.sum(dim=1)
        self.total_squares += flat.square().sum(dim=1)
        self.count += int(flat.shape[1])

    def compute(self) -> dict[str, Any]:
        if self.count <= 0:
            raise ValueError("No R/train canonical latents were available for statistics.")
        mean = self.total / self.count
        variance = (self.total_squares / self.count) - mean.square()
        std = variance.clamp_min(0.0).sqrt()
        if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(std).all()):
            raise ValueError("Canonical latent statistics are non-finite.")
        return {
            "per_channel_mean": [float(value) for value in mean],
            "per_channel_std": [float(value) for value in std],
            "global_mean": float(mean.mean()),
            "global_std": float(std.mean()),
            "voxels_per_channel": int(self.count),
            "channels": int(self.channels),
        }


def build_photometry_factored_latent_bank(
    *,
    encoder: Any,
    artifact: FrozenPhotometryArtifact,
    qualification: Mapping[str, Any],
    canonical_dir: str | Path,
    config: PhotometryFactoredLatentBankConfig,
    resolved_config: Mapping[str, Any],
    photometry_artifact_file_sha256: str,
    qualification_file_sha256: str,
    vae_config_sha256: str,
    vae_checkpoint_sha256: str,
    code_provenance: Mapping[str, Any],
    device: torch.device,
    resume: bool = False,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Build raw canonical posterior means, train-only stats, and train descriptors."""

    resolved = validate_canonical_artifact_config(resolved_config)
    expected_config = PhotometryFactoredLatentBankConfig.from_mapping(
        resolved, out_dir=config.out_dir
    )
    if config != expected_config:
        raise ValueError("Factored-bank runtime config differs from the resolved config.")
    _require_sha256(photometry_artifact_file_sha256, "photometry artifact file")
    _require_sha256(qualification_file_sha256, "qualification file")
    _require_sha256(vae_config_sha256, "VAE config")
    _require_sha256(vae_checkpoint_sha256, "VAE checkpoint")
    validate_canonical_artifact_code_provenance(code_provenance)
    assert_frozen(encoder)
    encoder = encoder.to(device).eval()

    canonical_root = Path(canonical_dir)
    canonical_manifest = load_canonical_volume_manifest(canonical_root)
    _preflight_canonical_manifest(canonical_manifest)
    if canonical_manifest["resolved_config_sha256"] != sha256_json(resolved):
        raise ValueError("Canonical volume and factored bank resolved configs differ.")
    if canonical_manifest["photometry"]["artifact_sha256"] != artifact.artifact_sha256:
        raise ValueError("Canonical-volume photometry artifact differs from bank input.")
    if canonical_manifest["photometry"]["artifact_file_sha256"] != (
        photometry_artifact_file_sha256
    ):
        raise ValueError("Canonical-volume photometry file hash differs from bank input.")
    source_split = canonical_manifest["source_split"]
    qualified = validate_variant_a_qualification(
        qualification,
        artifact=artifact,
        source_split_file_sha256=source_split["file_sha256"],
        source_membership_fingerprint=source_split["membership_fingerprint"],
        source_recovery_fingerprint=source_split["recovery_fingerprint"],
        vae_config_sha256=vae_config_sha256,
        vae_checkpoint_sha256=vae_checkpoint_sha256,
    )
    if canonical_manifest["qualification"]["result_sha256"] != qualified[
        "result_sha256"
    ]:
        raise ValueError("Canonical-volume qualification differs from bank input.")
    if canonical_manifest["qualification"]["file_sha256"] != qualification_file_sha256:
        raise ValueError("Canonical-volume qualification file hash differs from bank input.")

    factor = downsample_factor(encoder)
    for value in (*config.block_size, *config.halo):
        if value % factor != 0:
            raise ValueError(
                "Factored-bank block size and halo must be multiples of the VAE "
                f"downsample factor {factor}; got {value}."
            )
    canonical_manifest_file_sha = sha256_file(
        canonical_root / "canonical_volume_manifest.json"
    )
    config_sha = sha256_json(resolved)
    canonical_record_identities = [
        _canonical_record_identity(entry) for entry in canonical_manifest["records"]
    ]
    run_identity = {
        "contract_version": PHOTOMETRY_FACTORED_LATENT_BANK_VERSION,
        "canonical_artifact_sha256": canonical_manifest["artifact_sha256"],
        "canonical_manifest_file_sha256": canonical_manifest_file_sha,
        "photometry_artifact_sha256": artifact.artifact_sha256,
        "photometry_artifact_file_sha256": photometry_artifact_file_sha256,
        "photometry_resolved_config_sha256": artifact.provenance[
            "resolved_config_sha256"
        ],
        "qualification_result_sha256": qualified["result_sha256"],
        "qualification_file_sha256": qualification_file_sha256,
        "vae_config_sha256": vae_config_sha256,
        "vae_checkpoint_sha256": vae_checkpoint_sha256,
        "resolved_config_sha256": config_sha,
        "code_provenance_sha256": sha256_json(code_provenance),
        "records": canonical_record_identities,
    }
    run_fingerprint = sha256_json(run_identity)
    out_dir = config.out_dir
    manifest_path = out_dir / FACTORED_LATENT_BANK_MANIFEST
    if manifest_path.exists() and not resume:
        raise FileExistsError(f"Refusing to overwrite factored bank: {manifest_path}")

    dtype = {"float16": torch.float16, "float32": torch.float32}[config.store_dtype]
    records: list[dict[str, Any]] = []
    total = len(canonical_manifest["records"])
    for index, canonical_entry in enumerate(canonical_manifest["records"], start=1):
        record_identity = str(canonical_entry["record_identity"])
        destination = _latent_record_path(
            out_dir, str(canonical_entry["split"]), record_identity
        )
        resume_key = sha256_json(
            {
                "run_fingerprint": run_fingerprint,
                "record": _canonical_record_identity(canonical_entry),
            }
        )
        if destination.exists():
            if not resume:
                raise FileExistsError(f"Refusing to overwrite factored latent: {destination}")
            payload = _load_latent_record(destination, expected_resume_key=resume_key)
        else:
            canonical_payload = load_canonical_volume_record(
                canonical_root, canonical_entry
            )
            canonical = canonical_payload["canonical_tensor"].to(device)
            domain = Domain.from_dict(dict(canonical_entry["domain"]))
            latent, strategy_used = encode_latent(
                encoder,
                canonical,
                domain,
                strategy=config.strategy,
                block_size=config.block_size,
                halo=config.halo,
                precision=config.precision,
            )
            if strategy_used != config.strategy:
                raise ValueError("Factored-bank encoding path changed unexpectedly.")
            stored = latent[0].detach().cpu().to(dtype).contiguous()
            conservative_support = downsample_source_support(
                canonical_payload["source_support"], factor=factor
            )
            if tuple(conservative_support.shape) != tuple(stored.shape[1:]):
                raise ValueError(
                    "Downsampled source support does not match latent spatial shape."
                )
            packed = pack_support_mask(conservative_support)
            metadata = {
                "contract_version": PHOTOMETRY_FACTORED_LATENT_BANK_VERSION,
                "record_identity": record_identity,
                "record_identity_sha256": canonical_entry["record_identity_sha256"],
                "subject_identity": canonical_entry["subject_identity"],
                "subject_group_identity": canonical_entry["subject_group_identity"],
                "cohort": "R",
                "split": canonical_entry["split"],
                "domain": canonical_entry["domain"],
                "source_path_identity_sha256": canonical_entry[
                    "source_path_identity_sha256"
                ],
                "source_file_sha256": canonical_entry["source_file_sha256"],
                "source_loaded_array_sha256": canonical_entry[
                    "source_loaded_array_sha256"
                ],
                "canonical_tensor_sha256": canonical_entry["canonical_tensor_sha256"],
                "canonical_record_payload_sha256": canonical_entry["payload_sha256"],
                "canonical_record_file_sha256": canonical_entry["payload_file_sha256"],
                "latent_tensor_sha256": storage_tensor_sha256(stored),
                "latent_shape": list(stored.shape),
                "latent_dtype": _dtype_name(stored.dtype),
                "source_shape": canonical_entry["source_shape"],
                "source_dtype": canonical_entry["source_dtype"],
                "downsample_factor": factor,
                "encoding_path": {
                    "photometry": "FrozenPhotometryArtifact.normalize_source",
                    "vae": "frozen_encode_dist_posterior_mean",
                    "strategy": strategy_used,
                    "precision": config.precision,
                },
                "support_downsample_rule": SUPPORT_DOWNSAMPLE_RULE,
                "support_packing_rule": SUPPORT_PACKING_RULE,
                "support_shape": list(conservative_support.shape),
                "support_dtype": "bool",
                "support_nonzero_count": int(conservative_support.sum()),
                "packed_support_shape": list(packed.shape),
                "packed_support_dtype": "uint8",
                "packed_support_sha256": storage_tensor_sha256(packed),
                "photometry_artifact_sha256": artifact.artifact_sha256,
                "photometry_resolved_config_sha256": artifact.provenance[
                    "resolved_config_sha256"
                ],
                "vae_config_sha256": vae_config_sha256,
                "vae_checkpoint_sha256": vae_checkpoint_sha256,
                "source_membership_fingerprint": source_split[
                    "membership_fingerprint"
                ],
                "source_recovery_fingerprint": source_split["recovery_fingerprint"],
                "code_provenance_sha256": sha256_json(code_provenance),
                "resume_key": resume_key,
            }
            metadata["source_content_fingerprint"] = _source_content_fingerprint(
                metadata
            )
            metadata["payload_sha256"] = sha256_json(metadata)
            payload = {**metadata, "latent": stored, "packed_source_support": packed}
            atomic_torch_save_no_clobber(destination, payload)
            payload = _load_latent_record(destination, expected_resume_key=resume_key)
        records.append(_latent_manifest_entry(payload, destination, out_dir))
        if progress is not None:
            progress("factored_latent", index, total, record_identity)

    records.sort(key=lambda item: (item["split"], item["record_identity"]))
    _require_factored_bank_roles(records)
    statistics = _compute_statistics_payload(
        records=records,
        root=out_dir,
        run_fingerprint=run_fingerprint,
        resolved_config_sha256=config_sha,
        source_split=source_split,
        artifact=artifact,
        vae_config_sha256=vae_config_sha256,
        vae_checkpoint_sha256=vae_checkpoint_sha256,
        code_provenance=code_provenance,
    )
    stats_path = out_dir / FACTORED_LATENT_STATS_FILE
    write_json_resume_exact(stats_path, statistics, resume=resume)

    descriptor_manifest = _build_descriptors(
        records=records,
        root=out_dir,
        statistics=statistics,
        config=config,
        resolved_config_sha256=config_sha,
        run_fingerprint=run_fingerprint,
        code_provenance=code_provenance,
        resume=resume,
        progress=progress,
    )
    descriptor_path = out_dir / STRUCTURAL_DESCRIPTOR_MANIFEST
    write_json_resume_exact(descriptor_path, descriptor_manifest, resume=resume)

    domain_counts = _domain_counts(records, config.splits)
    manifest: dict[str, Any] = {
        "contract_version": PHOTOMETRY_FACTORED_LATENT_BANK_VERSION,
        "semantics": "z_d=E(N_d(x_d));frozen-vae-posterior-mean-v1",
        "legacy_contract_mutation": "latent-bank-v1-unchanged",
        "eligibility_rule": "cohort=R;split-in(train,validation)",
        "permitted_splits": list(config.splits),
        "resolved_config": resolved,
        "resolved_config_sha256": config_sha,
        "run_fingerprint": run_fingerprint,
        "source_split": dict(source_split),
        "canonical_volume": {
            "contract_version": CANONICAL_VOLUME_CONTRACT_VERSION,
            "artifact_sha256": canonical_manifest["artifact_sha256"],
            "manifest_file_sha256": canonical_manifest_file_sha,
        },
        "photometry": {
            "artifact_sha256": artifact.artifact_sha256,
            "artifact_file_sha256": photometry_artifact_file_sha256,
            "resolved_config_sha256": artifact.provenance["resolved_config_sha256"],
        },
        "qualification": {
            "result_sha256": qualified["result_sha256"],
            "file_sha256": qualification_file_sha256,
            "canonical_latent_bank_authorized": True,
        },
        "vae": {
            "config_sha256": vae_config_sha256,
            "checkpoint_sha256": vae_checkpoint_sha256,
            "encoder_statistic": "posterior_mean",
            "frozen": True,
        },
        "code_provenance": dict(code_provenance),
        "eligibility_proof": {
            "all_cohort_R": all(item["cohort"] == "R" for item in records),
            "prospective_accepted_count": 0,
            "statistics_role": "R/train only",
            "descriptor_role": "R/train only",
        },
        "domain_counts": domain_counts,
        "record_count": len(records),
        "records": records,
        "records_sha256": sha256_json(records),
        "latent_statistics": {
            "contract_version": PHOTOMETRY_FACTORED_LATENT_STATS_VERSION,
            "artifact_sha256": statistics["artifact_sha256"],
            "file_sha256": sha256_file(stats_path),
            "computed_over": statistics["computed_over"],
        },
        "structural_descriptors": {
            "contract_version": PHOTOMETRY_FACTORED_DESCRIPTOR_VERSION,
            "artifact_sha256": descriptor_manifest["artifact_sha256"],
            "file_sha256": sha256_file(descriptor_path),
            "computed_over": descriptor_manifest["computed_over"],
            "record_count": descriptor_manifest["record_count"],
        },
    }
    manifest["artifact_sha256"] = sha256_json(manifest)
    write_json_resume_exact(manifest_path, manifest, resume=resume)
    return load_photometry_factored_latent_bank_manifest(out_dir)


def downsample_source_support(mask: torch.Tensor, *, factor: int) -> torch.Tensor:
    """Conservatively keep a latent cell only when its full source block is supported."""

    if factor <= 0:
        raise ValueError("Support downsample factor must be positive.")
    work = mask.detach().cpu().to(torch.bool)
    if work.ndim != 5 or tuple(work.shape[:2]) != (1, 1):
        raise ValueError("Source support must have shape (1,1,X,Y,Z).")
    if any(int(size) % factor != 0 for size in work.shape[-3:]):
        raise ValueError("Source support extent must be divisible by the VAE factor.")
    unsupported = (~work).to(torch.float32)
    any_unsupported = F.max_pool3d(unsupported, kernel_size=factor, stride=factor)
    conservative = any_unsupported == 0
    return conservative[0, 0].contiguous()


def pack_support_mask(mask: torch.Tensor) -> torch.Tensor:
    work = mask.detach().cpu().to(torch.bool).contiguous().numpy().reshape(-1)
    packed = np.packbits(work, bitorder="little")
    return torch.from_numpy(np.asarray(packed, dtype=np.uint8).copy())


def unpack_support_mask(packed: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
    if packed.dtype != torch.uint8 or packed.ndim != 1:
        raise ValueError("Packed source support must be a 1-D uint8 tensor.")
    normalized_shape = tuple(int(value) for value in shape)
    if not normalized_shape or any(value <= 0 for value in normalized_shape):
        raise ValueError("Packed source-support shape must be positive.")
    count = math.prod(normalized_shape)
    expected_bytes = (count + 7) // 8
    if packed.numel() != expected_bytes:
        raise ValueError("Packed source-support byte count does not match its shape.")
    values = np.unpackbits(
        packed.detach().cpu().numpy(), count=count, bitorder="little"
    ).astype(np.bool_)
    return torch.from_numpy(values.reshape(normalized_shape).copy())


def structural_descriptor(
    standardized_latent: torch.Tensor,
    support: torch.Tensor,
    *,
    pool_output_sizes: Sequence[int] = (1, 2, 4),
) -> torch.Tensor:
    """Fixed support-normalized multiscale and spatial-gradient descriptor."""

    latent = standardized_latent.detach().cpu().to(torch.float32)
    mask = support.detach().cpu().to(torch.bool)
    if latent.ndim != 4 or mask.ndim != 3 or tuple(latent.shape[1:]) != tuple(mask.shape):
        raise ValueError("Descriptor requires latent (C,X,Y,Z) and matching support.")
    sizes = tuple(int(value) for value in pool_output_sizes)
    if not sizes or any(value <= 0 for value in sizes):
        raise ValueError("Descriptor pool output sizes must be positive.")
    image = latent.unsqueeze(0)
    mask_image = mask.to(torch.float32).unsqueeze(0).unsqueeze(0)
    features: list[tuple[torch.Tensor, torch.Tensor]] = [(image, mask_image)]
    for axis in range(3):
        dimension = axis + 2
        difference = image.diff(dim=dimension).abs()
        left = mask_image.narrow(dimension, 0, mask_image.shape[dimension] - 1)
        right = mask_image.narrow(dimension, 1, mask_image.shape[dimension] - 1)
        valid = left * right
        padding = [0, 0, 0, 0, 0, 0]
        padding[2 * (2 - axis) + 1] = 1
        features.append((F.pad(difference, padding), F.pad(valid, padding)))
    pieces: list[torch.Tensor] = []
    for values, valid in features:
        for size in sizes:
            numerator = F.adaptive_avg_pool3d(values * valid, size)
            denominator = F.adaptive_avg_pool3d(valid, size)
            pooled = numerator / denominator.clamp_min(1e-12)
            pooled = pooled * (denominator > 0)
            pieces.append(pooled.reshape(-1))
    descriptor = torch.cat(pieces).to(torch.float32).contiguous()
    if not bool(torch.isfinite(descriptor).all()):
        raise ValueError("Structural descriptor contains non-finite values.")
    return descriptor


def load_photometry_factored_latent_bank_manifest(
    root: str | Path,
    *,
    expected_artifact_sha256: str | None = None,
) -> dict[str, Any]:
    path = Path(root) / FACTORED_LATENT_BANK_MANIFEST
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not load factored-bank manifest {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Factored-bank manifest root must be a JSON object.")
    manifest = _json_safe_mapping(payload)
    if manifest.get("contract_version") != PHOTOMETRY_FACTORED_LATENT_BANK_VERSION:
        raise ValueError("Photometry-factored latent-bank contract mismatch.")
    stored_hash = str(manifest.get("artifact_sha256", ""))
    unhashed = dict(manifest)
    unhashed.pop("artifact_sha256", None)
    if stored_hash != sha256_json(unhashed):
        raise ValueError("Factored-bank manifest content hash mismatch.")
    if expected_artifact_sha256 is not None and stored_hash != expected_artifact_sha256:
        raise ValueError("Factored-bank artifact identity mismatch.")
    if manifest.get("legacy_contract_mutation") != "latent-bank-v1-unchanged":
        raise ValueError("Factored-bank manifest does not preserve latent-bank-v1.")
    if manifest.get("permitted_splits") != list(CANONICAL_VOLUME_SPLITS):
        raise ValueError("Factored-bank split roles are incompatible.")
    if sha256_json(manifest.get("resolved_config")) != manifest.get(
        "resolved_config_sha256"
    ):
        raise ValueError("Factored-bank resolved-config hash mismatch.")
    validate_canonical_artifact_config(manifest["resolved_config"])
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("Factored-bank manifest has no records.")
    if sha256_json(records) != manifest.get("records_sha256"):
        raise ValueError("Factored-bank record-list hash mismatch.")
    _require_factored_bank_roles(records)
    if manifest.get("record_count") != len(records):
        raise ValueError("Factored-bank record count mismatch.")
    for entry in records:
        _safe_relative_path(Path(root), str(entry.get("path", "")))
        _require_sha256(str(entry.get("payload_file_sha256", "")), "latent file")
    if manifest.get("eligibility_proof", {}).get("prospective_accepted_count") != 0:
        raise ValueError("Factored bank accepted prospective data.")
    validate_canonical_artifact_code_provenance(manifest.get("code_provenance", {}))
    statistics = manifest.get("latent_statistics")
    descriptors = manifest.get("structural_descriptors")
    if not isinstance(statistics, Mapping) or not isinstance(descriptors, Mapping):
        raise ValueError("Factored-bank derived-artifact provenance is incomplete.")
    if statistics.get("contract_version") != PHOTOMETRY_FACTORED_LATENT_STATS_VERSION:
        raise ValueError("Factored-bank statistics contract mismatch.")
    if descriptors.get("contract_version") != PHOTOMETRY_FACTORED_DESCRIPTOR_VERSION:
        raise ValueError("Factored-bank descriptor contract mismatch.")
    statistics_path = Path(root) / FACTORED_LATENT_STATS_FILE
    descriptor_path = Path(root) / STRUCTURAL_DESCRIPTOR_MANIFEST
    if sha256_file(statistics_path) != statistics.get("file_sha256"):
        raise ValueError("Factored-bank statistics file hash mismatch.")
    if sha256_file(descriptor_path) != descriptors.get("file_sha256"):
        raise ValueError("Factored-bank descriptor file hash mismatch.")
    statistics_payload = _load_json(statistics_path)
    descriptor_payload = _load_json(descriptor_path)
    if statistics_payload["artifact_sha256"] != statistics.get("artifact_sha256"):
        raise ValueError("Factored-bank statistics artifact identity mismatch.")
    if descriptor_payload["artifact_sha256"] != descriptors.get("artifact_sha256"):
        raise ValueError("Factored-bank descriptor artifact identity mismatch.")
    return manifest


def audit_photometry_factored_latent_bank(
    *,
    root: str | Path,
    canonical_dir: str | Path,
    encoder: Any,
    artifact: FrozenPhotometryArtifact,
    qualification: Mapping[str, Any],
    resolved_config: Mapping[str, Any],
    photometry_artifact_file_sha256: str,
    qualification_file_sha256: str,
    vae_config_sha256: str,
    vae_checkpoint_sha256: str,
    device: torch.device,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Re-encode all canonical tensors and recompute stats/descriptors without writes."""

    bank_root = Path(root)
    canonical_root = Path(canonical_dir)
    manifest = load_photometry_factored_latent_bank_manifest(bank_root)
    canonical_manifest = load_canonical_volume_manifest(canonical_root)
    _preflight_canonical_manifest(canonical_manifest)
    resolved = validate_canonical_artifact_config(resolved_config)
    config = PhotometryFactoredLatentBankConfig.from_mapping(resolved, out_dir=bank_root)
    if manifest["resolved_config_sha256"] != sha256_json(resolved):
        raise ValueError("Factored-bank audit config hash mismatch.")
    if manifest["canonical_volume"]["artifact_sha256"] != canonical_manifest[
        "artifact_sha256"
    ]:
        raise ValueError("Factored-bank canonical artifact hash mismatch.")
    if manifest["canonical_volume"]["manifest_file_sha256"] != sha256_file(
        canonical_root / "canonical_volume_manifest.json"
    ):
        raise ValueError("Factored-bank canonical manifest file hash mismatch.")
    if manifest["photometry"] != {
        "artifact_sha256": artifact.artifact_sha256,
        "artifact_file_sha256": photometry_artifact_file_sha256,
        "resolved_config_sha256": artifact.provenance["resolved_config_sha256"],
    }:
        raise ValueError("Factored-bank photometry provenance mismatch.")
    source_split = canonical_manifest["source_split"]
    qualified = validate_variant_a_qualification(
        qualification,
        artifact=artifact,
        source_split_file_sha256=source_split["file_sha256"],
        source_membership_fingerprint=source_split["membership_fingerprint"],
        source_recovery_fingerprint=source_split["recovery_fingerprint"],
        vae_config_sha256=vae_config_sha256,
        vae_checkpoint_sha256=vae_checkpoint_sha256,
    )
    if manifest["qualification"] != {
        "result_sha256": qualified["result_sha256"],
        "file_sha256": qualification_file_sha256,
        "canonical_latent_bank_authorized": True,
    }:
        raise ValueError("Factored-bank qualification provenance mismatch.")
    if manifest["vae"] != {
        "config_sha256": vae_config_sha256,
        "checkpoint_sha256": vae_checkpoint_sha256,
        "encoder_statistic": "posterior_mean",
        "frozen": True,
    }:
        raise ValueError("Factored-bank VAE provenance mismatch.")
    assert_frozen(encoder)
    encoder = encoder.to(device).eval()
    factor = downsample_factor(encoder)
    canonical_by_identity = {
        item["record_identity"]: item for item in canonical_manifest["records"]
    }
    if set(canonical_by_identity) != {
        item["record_identity"] for item in manifest["records"]
    }:
        raise ValueError("Factored-bank and canonical record membership differ.")
    total = len(manifest["records"])
    for index, entry in enumerate(manifest["records"], start=1):
        path = _safe_relative_path(bank_root, entry["path"])
        if sha256_file(path) != entry["payload_file_sha256"]:
            raise ValueError("Factored latent file hash mismatch.")
        payload = _load_latent_record(path, expected_resume_key=entry["resume_key"])
        canonical_entry = canonical_by_identity[entry["record_identity"]]
        canonical_payload = load_canonical_volume_record(canonical_root, canonical_entry)
        domain = Domain.from_dict(dict(entry["domain"]))
        encoded, used = encode_latent(
            encoder,
            canonical_payload["canonical_tensor"].to(device),
            domain,
            strategy=config.strategy,
            block_size=config.block_size,
            halo=config.halo,
            precision=config.precision,
        )
        stored = encoded[0].detach().cpu().to(payload["latent"].dtype).contiguous()
        if used != payload["encoding_path"]["strategy"]:
            raise ValueError("Factored latent encoding-path identity changed.")
        if storage_tensor_sha256(stored) != payload["latent_tensor_sha256"]:
            raise ValueError("Factored latent differs from E(N_d(source)).")
        support = downsample_source_support(
            canonical_payload["source_support"], factor=factor
        )
        packed = pack_support_mask(support)
        if storage_tensor_sha256(packed) != payload["packed_support_sha256"]:
            raise ValueError("Factored latent packed support changed.")
        if progress is not None:
            progress("audit_factored_latent", index, total, entry["record_identity"])

    expected_stats = _compute_statistics_payload(
        records=manifest["records"],
        root=bank_root,
        run_fingerprint=manifest["run_fingerprint"],
        resolved_config_sha256=manifest["resolved_config_sha256"],
        source_split=manifest["source_split"],
        artifact=artifact,
        vae_config_sha256=vae_config_sha256,
        vae_checkpoint_sha256=vae_checkpoint_sha256,
        code_provenance=manifest["code_provenance"],
    )
    actual_stats = _load_json(bank_root / FACTORED_LATENT_STATS_FILE)
    if actual_stats != expected_stats:
        raise ValueError("Factored-bank train-only statistics do not recompute exactly.")
    if manifest["latent_statistics"] != {
        "contract_version": PHOTOMETRY_FACTORED_LATENT_STATS_VERSION,
        "artifact_sha256": actual_stats["artifact_sha256"],
        "file_sha256": sha256_file(bank_root / FACTORED_LATENT_STATS_FILE),
        "computed_over": actual_stats["computed_over"],
    }:
        raise ValueError("Factored-bank statistics file provenance mismatch.")
    expected_descriptor = _descriptor_manifest_from_existing(
        records=manifest["records"],
        root=bank_root,
        statistics=actual_stats,
        config=config,
        resolved_config_sha256=manifest["resolved_config_sha256"],
        run_fingerprint=manifest["run_fingerprint"],
        code_provenance=manifest["code_provenance"],
    )
    actual_descriptor = _load_json(bank_root / STRUCTURAL_DESCRIPTOR_MANIFEST)
    if actual_descriptor != expected_descriptor:
        raise ValueError("Structural-descriptor artifact does not recompute exactly.")
    if manifest["structural_descriptors"] != {
        "contract_version": PHOTOMETRY_FACTORED_DESCRIPTOR_VERSION,
        "artifact_sha256": actual_descriptor["artifact_sha256"],
        "file_sha256": sha256_file(bank_root / STRUCTURAL_DESCRIPTOR_MANIFEST),
        "computed_over": actual_descriptor["computed_over"],
        "record_count": actual_descriptor["record_count"],
    }:
        raise ValueError("Structural-descriptor file provenance mismatch.")
    return {
        "contract_version": PHOTOMETRY_FACTORED_LATENT_BANK_VERSION,
        "artifact_sha256": manifest["artifact_sha256"],
        "record_count": total,
        "train_statistics_verified": True,
        "structural_descriptors_verified": True,
        "packed_support_verified": True,
        "all_records_retrospective": True,
    }


def _compute_statistics_payload(
    *,
    records: Sequence[Mapping[str, Any]],
    root: Path,
    run_fingerprint: str,
    resolved_config_sha256: str,
    source_split: Mapping[str, Any],
    artifact: FrozenPhotometryArtifact,
    vae_config_sha256: str,
    vae_checkpoint_sha256: str,
    code_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    train = [item for item in records if item["split"] == "train"]
    if not train:
        raise ValueError("Factored-bank statistics require R/train records.")
    first = _load_latent_record(_safe_relative_path(root, train[0]["path"]))
    stats = _ChannelStats(int(first["latent"].shape[0]))
    inputs: list[dict[str, Any]] = []
    for entry in train:
        payload = _load_latent_record(
            _safe_relative_path(root, entry["path"]),
            expected_resume_key=entry["resume_key"],
        )
        if payload["cohort"] != "R" or payload["split"] != "train":
            raise ValueError("Canonical latent statistics saw a non-R/train record.")
        stats.update(payload["latent"])
        inputs.append(
            {
                "record_identity": payload["record_identity"],
                "subject_group_identity": payload["subject_group_identity"],
                "latent_tensor_sha256": payload["latent_tensor_sha256"],
                "source_content_fingerprint": payload["source_content_fingerprint"],
            }
        )
    inputs.sort(key=lambda item: item["record_identity"])
    payload: dict[str, Any] = {
        "contract_version": PHOTOMETRY_FACTORED_LATENT_STATS_VERSION,
        "computed_over": {"cohort": "R", "split": "train"},
        "excluded_roles": ["R/validation", "all P identities"],
        "record_count": len(inputs),
        "records": inputs,
        "records_sha256": sha256_json(inputs),
        "bank_run_fingerprint": run_fingerprint,
        "resolved_config_sha256": resolved_config_sha256,
        "source_split": dict(source_split),
        "photometry_artifact_sha256": artifact.artifact_sha256,
        "photometry_resolved_config_sha256": artifact.provenance[
            "resolved_config_sha256"
        ],
        "vae_config_sha256": vae_config_sha256,
        "vae_checkpoint_sha256": vae_checkpoint_sha256,
        "code_provenance_sha256": sha256_json(code_provenance),
        **stats.compute(),
    }
    payload["artifact_sha256"] = sha256_json(payload)
    return payload


def _build_descriptors(
    *,
    records: Sequence[Mapping[str, Any]],
    root: Path,
    statistics: Mapping[str, Any],
    config: PhotometryFactoredLatentBankConfig,
    resolved_config_sha256: str,
    run_fingerprint: str,
    code_provenance: Mapping[str, Any],
    resume: bool,
    progress: Progress | None,
) -> dict[str, Any]:
    train = [item for item in records if item["split"] == "train"]
    mean = torch.tensor(statistics["per_channel_mean"], dtype=torch.float32)
    std = torch.tensor(statistics["per_channel_std"], dtype=torch.float32).clamp_min(
        1e-6
    )
    entries: list[dict[str, Any]] = []
    for index, entry in enumerate(train, start=1):
        latent_payload = _load_latent_record(
            _safe_relative_path(root, entry["path"]),
            expected_resume_key=entry["resume_key"],
        )
        standardized = (
            latent_payload["latent"].to(torch.float32)
            - mean.reshape(-1, 1, 1, 1)
        ) / std.reshape(-1, 1, 1, 1)
        support = unpack_support_mask(
            latent_payload["packed_source_support"], latent_payload["support_shape"]
        )
        descriptor = structural_descriptor(
            standardized,
            support,
            pool_output_sizes=config.descriptor_pool_sizes,
        )
        descriptor_path = _descriptor_record_path(
            root, latent_payload["record_identity"]
        )
        descriptor_config = _descriptor_config(config)
        resume_key = sha256_json(
            {
                "bank_run_fingerprint": run_fingerprint,
                "record_identity": latent_payload["record_identity"],
                "source_content_fingerprint": latent_payload[
                    "source_content_fingerprint"
                ],
                "latent_statistics_sha256": statistics["artifact_sha256"],
                "descriptor_config_sha256": sha256_json(descriptor_config),
            }
        )
        metadata = {
            "contract_version": PHOTOMETRY_FACTORED_DESCRIPTOR_VERSION,
            "record_identity": latent_payload["record_identity"],
            "record_identity_sha256": latent_payload["record_identity_sha256"],
            "subject_identity": latent_payload["subject_identity"],
            "subject_group_identity": latent_payload["subject_group_identity"],
            "cohort": "R",
            "split": "train",
            "domain": latent_payload["domain"],
            "input": DESCRIPTOR_INPUT,
            "standardized_latent_sha256": storage_tensor_sha256(standardized),
            "source_content_fingerprint": latent_payload[
                "source_content_fingerprint"
            ],
            "latent_statistics_sha256": statistics["artifact_sha256"],
            "bank_run_fingerprint": run_fingerprint,
            "resolved_config_sha256": resolved_config_sha256,
            "code_provenance_sha256": sha256_json(code_provenance),
            "descriptor_config": descriptor_config,
            "descriptor_config_sha256": sha256_json(descriptor_config),
            "descriptor_shape": list(descriptor.shape),
            "descriptor_dtype": "float32",
            "descriptor_tensor_sha256": storage_tensor_sha256(descriptor),
            "paired_endpoint_or_target_input": "none",
            "resume_key": resume_key,
        }
        metadata["payload_sha256"] = sha256_json(metadata)
        if descriptor_path.exists():
            if not resume:
                raise FileExistsError(
                    f"Refusing to overwrite structural descriptor: {descriptor_path}"
                )
            loaded = _load_descriptor_record(
                descriptor_path, expected_resume_key=resume_key
            )
            if loaded["descriptor_tensor_sha256"] != metadata[
                "descriptor_tensor_sha256"
            ]:
                raise ValueError("Structural descriptor changed during exact resume.")
        else:
            atomic_torch_save_no_clobber(
                descriptor_path, {**metadata, "descriptor": descriptor}
            )
            loaded = _load_descriptor_record(
                descriptor_path, expected_resume_key=resume_key
            )
        entries.append(_descriptor_manifest_entry(loaded, descriptor_path, root))
        if progress is not None:
            progress(
                "structural_descriptor",
                index,
                len(train),
                latent_payload["record_identity"],
            )
    return _make_descriptor_manifest(
        entries=entries,
        statistics=statistics,
        config=config,
        resolved_config_sha256=resolved_config_sha256,
        run_fingerprint=run_fingerprint,
        code_provenance=code_provenance,
    )


def _descriptor_manifest_from_existing(
    *,
    records: Sequence[Mapping[str, Any]],
    root: Path,
    statistics: Mapping[str, Any],
    config: PhotometryFactoredLatentBankConfig,
    resolved_config_sha256: str,
    run_fingerprint: str,
    code_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    mean = torch.tensor(statistics["per_channel_mean"], dtype=torch.float32)
    std = torch.tensor(statistics["per_channel_std"], dtype=torch.float32).clamp_min(
        1e-6
    )
    for entry in records:
        if entry["split"] != "train":
            continue
        latent = _load_latent_record(
            _safe_relative_path(root, entry["path"]),
            expected_resume_key=entry["resume_key"],
        )
        standardized = (latent["latent"].to(torch.float32) - mean[:, None, None, None]) / std[
            :, None, None, None
        ]
        support = unpack_support_mask(
            latent["packed_source_support"], latent["support_shape"]
        )
        recomputed = structural_descriptor(
            standardized, support, pool_output_sizes=config.descriptor_pool_sizes
        )
        path = _descriptor_record_path(root, latent["record_identity"])
        loaded = _load_descriptor_record(path)
        if storage_tensor_sha256(recomputed) != loaded["descriptor_tensor_sha256"]:
            raise ValueError("Structural descriptor tensor does not recompute exactly.")
        entries.append(_descriptor_manifest_entry(loaded, path, root))
    return _make_descriptor_manifest(
        entries=entries,
        statistics=statistics,
        config=config,
        resolved_config_sha256=resolved_config_sha256,
        run_fingerprint=run_fingerprint,
        code_provenance=code_provenance,
    )


def _make_descriptor_manifest(
    *,
    entries: Sequence[Mapping[str, Any]],
    statistics: Mapping[str, Any],
    config: PhotometryFactoredLatentBankConfig,
    resolved_config_sha256: str,
    run_fingerprint: str,
    code_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    records = sorted(
        (_json_safe_mapping(item) for item in entries),
        key=lambda item: item["record_identity"],
    )
    descriptor_config = _descriptor_config(config)
    manifest: dict[str, Any] = {
        "contract_version": PHOTOMETRY_FACTORED_DESCRIPTOR_VERSION,
        "computed_over": {"cohort": "R", "split": "train"},
        "input": DESCRIPTOR_INPUT,
        "paired_endpoint_or_target_input": "forbidden",
        "descriptor_config": descriptor_config,
        "descriptor_config_sha256": sha256_json(descriptor_config),
        "latent_statistics_sha256": statistics["artifact_sha256"],
        "bank_run_fingerprint": run_fingerprint,
        "resolved_config_sha256": resolved_config_sha256,
        "code_provenance_sha256": sha256_json(code_provenance),
        "record_count": len(records),
        "records": records,
        "records_sha256": sha256_json(records),
    }
    manifest["artifact_sha256"] = sha256_json(manifest)
    return manifest


def _descriptor_config(config: PhotometryFactoredLatentBankConfig) -> dict[str, Any]:
    return {
        "input": DESCRIPTOR_INPUT,
        "pool_output_sizes": list(config.descriptor_pool_sizes),
        "pooling": DESCRIPTOR_POOLING,
        "gradients": DESCRIPTOR_GRADIENTS,
        "feature_order": ["latent", "gradient_x", "gradient_y", "gradient_z"],
        "scale_order": list(config.descriptor_pool_sizes),
        "support": "conservatively-downsampled-source-support",
        "dtype": "float32",
    }


def _load_latent_record(
    path: Path, *, expected_resume_key: str | None = None
) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(f"Could not load factored latent {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Factored latent payload must be a mapping.")
    result = dict(payload)
    if result.get("contract_version") != PHOTOMETRY_FACTORED_LATENT_BANK_VERSION:
        raise ValueError("Factored latent record contract mismatch.")
    if expected_resume_key is not None and result.get("resume_key") != expected_resume_key:
        raise ValueError("Factored latent record is incompatible with exact resume.")
    identity = classify_variant_a_cohort(
        case_identity=str(result.get("record_identity", "")),
        metadata_prefix="R",
        supplied_cohort=str(result.get("cohort", "")),
        subject_identity=result.get("subject_identity"),
        allowed_cohorts=("R",),
    )
    if result.get("subject_group_identity") != identity.subject_group_identity:
        raise ValueError("Factored latent subject-group identity mismatch.")
    if result.get("split") not in CANONICAL_VOLUME_SPLITS:
        raise ValueError("Factored latent record has a forbidden split role.")
    latent = result.get("latent")
    packed = result.get("packed_source_support")
    if not isinstance(latent, torch.Tensor) or not isinstance(packed, torch.Tensor):
        raise ValueError("Factored latent record tensors are missing.")
    if latent.ndim != 4 or latent.dtype not in {torch.float16, torch.float32}:
        raise ValueError("Factored latent tensor shape/dtype is incompatible.")
    if storage_tensor_sha256(latent) != result.get("latent_tensor_sha256"):
        raise ValueError("Factored latent tensor hash mismatch.")
    if storage_tensor_sha256(packed) != result.get("packed_support_sha256"):
        raise ValueError("Factored latent packed-support hash mismatch.")
    support = unpack_support_mask(packed, result.get("support_shape", ()))
    if tuple(support.shape) != tuple(latent.shape[1:]):
        raise ValueError("Factored latent support shape mismatch.")
    if int(support.sum()) != int(result.get("support_nonzero_count", -1)):
        raise ValueError("Factored latent support count mismatch.")
    metadata = {
        key: value
        for key, value in result.items()
        if key not in {"latent", "packed_source_support", "payload_sha256"}
    }
    if result.get("payload_sha256") != sha256_json(metadata):
        raise ValueError("Factored latent metadata hash mismatch.")
    if result.get("source_content_fingerprint") != _source_content_fingerprint(result):
        raise ValueError("Factored latent source-content fingerprint mismatch.")
    return result


def _load_descriptor_record(
    path: Path, *, expected_resume_key: str | None = None
) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(f"Could not load structural descriptor {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Structural descriptor payload must be a mapping.")
    result = dict(payload)
    if result.get("contract_version") != PHOTOMETRY_FACTORED_DESCRIPTOR_VERSION:
        raise ValueError("Structural descriptor contract mismatch.")
    if expected_resume_key is not None and result.get("resume_key") != expected_resume_key:
        raise ValueError("Structural descriptor is incompatible with exact resume.")
    if result.get("cohort") != "R" or result.get("split") != "train":
        raise ValueError("Structural descriptors are R/train-only.")
    if result.get("paired_endpoint_or_target_input") != "none":
        raise ValueError("Structural descriptor contains endpoint/target input.")
    descriptor = result.get("descriptor")
    if not isinstance(descriptor, torch.Tensor) or descriptor.dtype != torch.float32:
        raise ValueError("Structural descriptor tensor is missing or has the wrong dtype.")
    if storage_tensor_sha256(descriptor) != result.get("descriptor_tensor_sha256"):
        raise ValueError("Structural descriptor tensor hash mismatch.")
    metadata = {
        key: value
        for key, value in result.items()
        if key not in {"descriptor", "payload_sha256"}
    }
    if result.get("payload_sha256") != sha256_json(metadata):
        raise ValueError("Structural descriptor metadata hash mismatch.")
    return result


def _latent_manifest_entry(
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
        "canonical_tensor_sha256",
        "canonical_record_payload_sha256",
        "canonical_record_file_sha256",
        "source_content_fingerprint",
        "latent_tensor_sha256",
        "latent_shape",
        "latent_dtype",
        "source_shape",
        "source_dtype",
        "downsample_factor",
        "encoding_path",
        "support_downsample_rule",
        "support_packing_rule",
        "support_shape",
        "support_dtype",
        "support_nonzero_count",
        "packed_support_shape",
        "packed_support_dtype",
        "packed_support_sha256",
        "code_provenance_sha256",
        "resume_key",
        "payload_sha256",
    )
    return {
        **{key: payload[key] for key in keys},
        "path": path.relative_to(root).as_posix(),
        "payload_file_sha256": sha256_file(path),
    }


def _descriptor_manifest_entry(
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
        "input",
        "standardized_latent_sha256",
        "source_content_fingerprint",
        "latent_statistics_sha256",
        "bank_run_fingerprint",
        "resolved_config_sha256",
        "code_provenance_sha256",
        "descriptor_config_sha256",
        "descriptor_shape",
        "descriptor_dtype",
        "descriptor_tensor_sha256",
        "paired_endpoint_or_target_input",
        "resume_key",
        "payload_sha256",
    )
    return {
        **{key: payload[key] for key in keys},
        "path": path.relative_to(root).as_posix(),
        "payload_file_sha256": sha256_file(path),
    }


def _source_content_fingerprint(payload: Mapping[str, Any]) -> str:
    return sha256_json(
        {
            "record_identity_sha256": payload["record_identity_sha256"],
            "source_path_identity_sha256": payload["source_path_identity_sha256"],
            "source_file_sha256": payload["source_file_sha256"],
            "source_loaded_array_sha256": payload["source_loaded_array_sha256"],
            "canonical_tensor_sha256": payload["canonical_tensor_sha256"],
            "canonical_record_payload_sha256": payload[
                "canonical_record_payload_sha256"
            ],
            "latent_tensor_sha256": payload["latent_tensor_sha256"],
            "packed_support_sha256": payload["packed_support_sha256"],
        }
    )


def _canonical_record_identity(entry: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: entry[key]
        for key in (
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
            "canonical_tensor_sha256",
            "support_tensor_sha256",
            "payload_sha256",
            "payload_file_sha256",
        )
    }


def _preflight_canonical_manifest(manifest: Mapping[str, Any]) -> None:
    """Reject cohort/role leakage before any canonical tensor is loaded."""

    expected = {
        Domain(field, contrast).label
        for contrast in CONTRASTS
        for field in FIELD_STRENGTHS_T
    }
    for split in CANONICAL_VOLUME_SPLITS:
        actual: set[str] = set()
        for entry in manifest["records"]:
            if entry["split"] != split:
                continue
            identity = classify_variant_a_cohort(
                case_identity=entry["record_identity"],
                metadata_prefix="R",
                supplied_cohort=entry["cohort"],
                subject_identity=entry["subject_identity"],
                allowed_cohorts=("R",),
            )
            if entry["subject_group_identity"] != identity.subject_group_identity:
                raise ValueError("Canonical record subject grouping is inconsistent.")
            actual.add(Domain.from_dict(dict(entry["domain"])).label)
        if actual != expected:
            raise ValueError(f"Canonical split {split!r} does not cover all 15 domains.")


def _require_factored_bank_roles(records: Sequence[Mapping[str, Any]]) -> None:
    identities: set[str] = set()
    for entry in records:
        identity = classify_variant_a_cohort(
            case_identity=str(entry.get("record_identity", "")),
            metadata_prefix="R",
            supplied_cohort=str(entry.get("cohort", "")),
            subject_identity=entry.get("subject_identity"),
            allowed_cohorts=("R",),
        )
        if entry.get("subject_group_identity") != identity.subject_group_identity:
            raise ValueError("Factored-bank subject-group identity mismatch.")
        if entry.get("split") not in CANONICAL_VOLUME_SPLITS:
            raise ValueError("Factored-bank record has a forbidden split role.")
        if identity.case_identity in identities:
            raise ValueError("Factored-bank record identity is duplicated.")
        identities.add(identity.case_identity)
    expected = {
        Domain(field, contrast).label
        for contrast in CONTRASTS
        for field in FIELD_STRENGTHS_T
    }
    for split in CANONICAL_VOLUME_SPLITS:
        actual = {
            Domain.from_dict(dict(item["domain"])).label
            for item in records
            if item["split"] == split
        }
        if actual != expected:
            raise ValueError(f"Factored-bank split {split!r} does not cover all 15 domains.")


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


def _latent_record_path(root: Path, split: str, identity: str) -> Path:
    return root / "latents" / split / f"{sha256_text(identity)}.pt"


def _descriptor_record_path(root: Path, identity: str) -> Path:
    return root / "descriptors" / "train" / f"{sha256_text(identity)}.pt"


def _safe_relative_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("Factored-bank record path escapes its artifact root.") from exc
    return path


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not load JSON artifact {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON artifact {path} must contain an object.")
    result = _json_safe_mapping(payload)
    stored = str(result.get("artifact_sha256", ""))
    unhashed = dict(result)
    unhashed.pop("artifact_sha256", None)
    if stored != sha256_json(unhashed):
        raise ValueError(f"JSON artifact {path} content hash mismatch.")
    return result


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _require_sha256(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"Factored-bank {name} SHA-256 is invalid.")


def _json_safe_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    decoded = json.loads(json.dumps(dict(value), sort_keys=True, allow_nan=False))
    if not isinstance(decoded, dict):
        raise TypeError("Factored-bank payload must be a mapping.")
    return decoded


__all__ = [
    "FACTORED_LATENT_BANK_MANIFEST",
    "FACTORED_LATENT_STATS_FILE",
    "PHOTOMETRY_FACTORED_DESCRIPTOR_VERSION",
    "PHOTOMETRY_FACTORED_LATENT_BANK_VERSION",
    "PHOTOMETRY_FACTORED_LATENT_STATS_VERSION",
    "STRUCTURAL_DESCRIPTOR_MANIFEST",
    "PhotometryFactoredLatentBankConfig",
    "audit_photometry_factored_latent_bank",
    "build_photometry_factored_latent_bank",
    "downsample_source_support",
    "load_photometry_factored_latent_bank_manifest",
    "pack_support_mask",
    "structural_descriptor",
    "unpack_support_mask",
]
