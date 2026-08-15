"""Streamed retrospective photometry-factored latent bank.

The primary v1 path computes ``source -> N_d(source) -> E(N_d(source))`` one record at a
time.  It persists only the posterior-mean latent, packed encoder-conservative latent
support, record sidecar/provenance, masked train statistics, and structural descriptors.
Full canonical tensors and full-resolution support masks are never published.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from fieldbridge.data.contracts import VolumeRecord
from fieldbridge.data.domains import CONTRASTS, FIELD_STRENGTHS_T, Domain
from fieldbridge.data.latent_bank import downsample_factor, encode_latent
from fieldbridge.data.photometry_factorization import (
    PHOTOMETRY_FACTORIZATION_CONTRACT_VERSION,
    PHOTOMETRY_SUPPORT_POLICY,
    FrozenPhotometryArtifact,
    canonical_tensor_sha256,
    classify_variant_a_cohort,
    sha256_file,
    sha256_json,
    sha256_text,
)
from fieldbridge.data.stage2_canonical_volume import (
    CANONICAL_STREAM_SEMANTICS,
    CANONICAL_VOLUME_CONTRACT_VERSION,
    CANONICAL_VOLUME_SPLITS,
    EligibleRecord,
    FileHasher,
    Linker,
    SourceShapeResolver,
    VolumeLoader,
    atomic_torch_save_no_clobber,
    build_storage_preflight_report,
    dtype_name,
    eligible_source_identity,
    json_safe_mapping,
    preflight_atomic_no_clobber_filesystem,
    preflight_retrospective_records,
    require_sha256,
    resolve_source_shapes,
    safe_relative_path,
    storage_tensor_sha256,
    validate_canonical_artifact_code_provenance,
    validate_canonical_artifact_config,
    validate_variant_a_qualification,
    validated_source_volume,
    write_json_resume_exact,
)
from fieldbridge.training.train_loop import assert_frozen

PHOTOMETRY_FACTORED_LATENT_BANK_VERSION = "photometry-factored-latent-bank-v1"
PHOTOMETRY_FACTORED_LATENT_STATS_VERSION = "photometry-factored-latent-statistics-v1"
PHOTOMETRY_FACTORED_DESCRIPTOR_VERSION = "photometry-factored-structural-descriptor-v1"
PHOTOMETRY_FACTORED_DESCRIPTOR_QUALIFICATION_VERSION = (
    "photometry-factored-structural-descriptor-qualification-v1"
)
FACTORED_LATENT_BANK_MANIFEST = "photometry_factored_latent_bank_manifest.json"
FACTORED_LATENT_STATS_FILE = "latent_stats.json"
STRUCTURAL_DESCRIPTOR_MANIFEST = "structural_descriptor_manifest.json"
SUPPORT_PROPAGATION_RULE = "frozen-encoder-dependency-propagation-v1"
SUPPORT_RULE_EVIDENCE_VERSION = "klvae-encoder-support-evidence-v1"
SUPPORT_PACKING_RULE = "numpy-packbits-c-order-little-bit-v1"
MASKED_WELFORD_RULE = "channelwise-masked-welford-float64-v1"
MIN_SUPPORTED_CHANNEL_VARIANCE = 1.0e-12
DESCRIPTOR_INPUT = "canonical_standardized_supported_latent_only"
DESCRIPTOR_POOLING = "support-normalized-adaptive-average-3d-v1"
DESCRIPTOR_GRADIENTS = "absolute-first-forward-difference-xyz-v1"

Progress = Callable[[str, int, int, str], None]


@dataclass(frozen=True, slots=True)
class PhotometryFactoredLatentBankConfig:
    out_dir: Path
    strategy: str = "full"
    store_dtype: str = "float16"
    precision: str = "float32"
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
            "splits": list(self.splits),
            "descriptor_pool_sizes": list(self.descriptor_pool_sizes),
        }


@dataclass(frozen=True, slots=True)
class _PreparedRun:
    resolved: dict[str, Any]
    qualified: dict[str, Any]
    eligible: tuple[EligibleRecord, ...]
    excluded: tuple[dict[str, Any], ...]
    source_shapes: dict[str, tuple[int, int, int, int, int]]
    support_rule: dict[str, Any]
    storage_report: dict[str, Any]
    filesystem_report: dict[str, Any] | None
    source_split: dict[str, str]
    run_fingerprint: str
    config_sha256: str


class MaskedChannelWelford:
    """Stable float64 channel statistics over supported spatial cells only."""

    def __init__(self, channels: int) -> None:
        if int(channels) <= 0:
            raise ValueError("Masked Welford statistics require positive channels.")
        self.channels = int(channels)
        self.count = torch.zeros(self.channels, dtype=torch.int64)
        self.mean = torch.zeros(self.channels, dtype=torch.float64)
        self.m2 = torch.zeros(self.channels, dtype=torch.float64)

    def update(self, latent: torch.Tensor, support: torch.Tensor) -> None:
        work = latent.detach().cpu().to(torch.float64)
        mask = support.detach().cpu().to(torch.bool)
        if work.ndim != 4 or int(work.shape[0]) != self.channels:
            raise ValueError("Masked latent statistics require one (C,X,Y,Z) tensor.")
        if tuple(mask.shape) != tuple(work.shape[1:]):
            raise ValueError("Masked latent statistics support shape mismatch.")
        selected_count = int(mask.sum())
        if selected_count <= 0:
            raise ValueError("Factored latent support is empty.")
        for channel in range(self.channels):
            values = work[channel][mask]
            if not bool(torch.isfinite(values).all()):
                raise ValueError("Supported canonical latent values are non-finite.")
            batch_count = int(values.numel())
            batch_mean = values.mean()
            batch_m2 = (values - batch_mean).square().sum()
            old_count = int(self.count[channel])
            combined = old_count + batch_count
            delta = batch_mean - self.mean[channel]
            self.mean[channel] += delta * (batch_count / combined)
            self.m2[channel] += batch_m2 + delta.square() * (
                old_count * batch_count / combined
            )
            self.count[channel] = combined

    def compute(self) -> dict[str, Any]:
        if bool((self.count < 2).any()):
            raise ValueError("Masked latent statistics require at least two values per channel.")
        variance = self.m2 / self.count.to(torch.float64)
        if not bool(torch.isfinite(self.mean).all()) or not bool(torch.isfinite(variance).all()):
            raise ValueError("Masked canonical latent statistics are non-finite.")
        if bool((variance <= MIN_SUPPORTED_CHANNEL_VARIANCE).any()):
            raise ValueError("Masked canonical latent channel variance is degenerate.")
        std = variance.sqrt()
        counts = [int(value) for value in self.count]
        return {
            "algorithm": MASKED_WELFORD_RULE,
            "minimum_channel_variance": MIN_SUPPORTED_CHANNEL_VARIANCE,
            "per_channel_mean": [float(value) for value in self.mean],
            "per_channel_std": [float(value) for value in std],
            "per_channel_supported_count": counts,
            "total_supported_value_count": int(sum(counts)),
            "supported_spatial_cell_count": int(counts[0]),
            "channels": self.channels,
        }


def derive_encoder_support_rule(encoder: Any) -> dict[str, Any]:
    """Inspect the complete frozen KLVAE encoder graph and seal dependency evidence."""

    required = ("stem", "res1", "down1", "res2", "down2", "res3", "to_dist")
    if encoder.__class__.__name__ != "KLVAEEncoder" or any(
        not hasattr(encoder, name) for name in required
    ):
        raise ValueError(
            "Support propagation v1 is qualified only for the frozen KLVAEEncoder graph."
        )
    if int(getattr(encoder, "spatial_dims", 0)) != 3:
        raise ValueError("Support propagation v1 requires the frozen 3-D KLVAE encoder.")
    factor = downsample_factor(encoder)
    operations: list[dict[str, Any]] = []
    receptive_field = [1, 1, 1]
    stride = [1, 1, 1]
    offset = [0.0, 0.0, 0.0]
    group_norm_count = 0

    def add_conv(name: str, module: nn.Module) -> None:
        nonlocal receptive_field, stride, offset
        spec = _conv_spec(module, name)
        before_stride = list(stride)
        for axis in range(3):
            offset[axis] += (
                ((spec["kernel_size"][axis] - 1) * spec["dilation"][axis]) / 2
                - spec["padding"][axis]
            ) * before_stride[axis]
            receptive_field[axis] += (
                (spec["kernel_size"][axis] - 1)
                * spec["dilation"][axis]
                * before_stride[axis]
            )
            stride[axis] *= spec["stride"][axis]
        operations.append({"name": name, "operator": "conv3d", **spec})

    def add_norm(name: str, module: nn.Module) -> None:
        nonlocal group_norm_count
        if isinstance(module, nn.Identity):
            operations.append({"name": name, "operator": "identity-normalization"})
        elif isinstance(module, nn.GroupNorm):
            group_norm_count += 1
            operations.append(
                {
                    "name": name,
                    "operator": "groupnorm",
                    "num_groups": int(module.num_groups),
                    "spatial_dependency": "global-per-sample-channel-group",
                }
            )
        else:
            raise ValueError(f"Unsupported encoder normalization in support graph: {name}.")

    add_conv("stem", encoder.stem)
    for stack_name in ("res1",):
        for index, block in enumerate(getattr(encoder, stack_name)):
            add_norm(f"{stack_name}.{index}.norm1", block.norm1)
            add_conv(f"{stack_name}.{index}.conv1", block.conv1)
            add_norm(f"{stack_name}.{index}.norm2", block.norm2)
            add_conv(f"{stack_name}.{index}.conv2", block.conv2)
            _validate_skip(block.skip, f"{stack_name}.{index}.skip")
    add_conv("down1", encoder.down1)
    for stack_name in ("res2",):
        for index, block in enumerate(getattr(encoder, stack_name)):
            add_norm(f"{stack_name}.{index}.norm1", block.norm1)
            add_conv(f"{stack_name}.{index}.conv1", block.conv1)
            add_norm(f"{stack_name}.{index}.norm2", block.norm2)
            add_conv(f"{stack_name}.{index}.conv2", block.conv2)
            _validate_skip(block.skip, f"{stack_name}.{index}.skip")
    add_conv("down2", encoder.down2)
    for stack_name in ("res3",):
        for index, block in enumerate(getattr(encoder, stack_name)):
            add_norm(f"{stack_name}.{index}.norm1", block.norm1)
            add_conv(f"{stack_name}.{index}.conv1", block.conv1)
            add_norm(f"{stack_name}.{index}.norm2", block.norm2)
            add_conv(f"{stack_name}.{index}.conv2", block.conv2)
            _validate_skip(block.skip, f"{stack_name}.{index}.skip")
    add_conv("to_dist", encoder.to_dist)
    if stride != [factor, factor, factor]:
        raise ValueError("Encoder graph stride differs from its sealed downsample factor.")
    if any(abs(value) > 1e-12 for value in offset):
        raise ValueError("Encoder graph alignment is not the reviewed zero-offset alignment.")
    graph = {
        "evidence_version": SUPPORT_RULE_EVIDENCE_VERSION,
        "encoder_class": f"{encoder.__class__.__module__}.{encoder.__class__.__qualname__}",
        "spatial_dims": 3,
        "num_res_blocks": int(getattr(encoder, "num_res_blocks")),
        "operations": operations,
    }
    rule: dict[str, Any] = {
        "rule": SUPPORT_PROPAGATION_RULE,
        "graph": graph,
        "graph_sha256": sha256_json(graph),
        "convolutional_receptive_field_size": receptive_field,
        "convolutional_receptive_field_radius": [
            (value - 1) // 2 for value in receptive_field
        ],
        "output_stride": stride,
        "alignment": {
            "latent_index_to_source_index": "source_index=latent_index*output_stride",
            "source_index_offset": [0, 0, 0],
            "outside_volume_padding": "constant-zero-not-a-source-dependency",
        },
        "groupnorm_count": group_norm_count,
        "complete_spatial_dependency": (
            "full-input-spatial-extent" if group_norm_count else "convolutional-receptive-field"
        ),
        "supported_cell_definition": (
            "no-unsupported-source-voxel-in-complete-encoder-dependency-set"
        ),
    }
    rule["rule_sha256"] = sha256_json(rule)
    return rule


def propagate_encoder_support(
    source_support: torch.Tensor,
    encoder: Any,
    *,
    expected_rule_sha256: str | None = None,
) -> torch.Tensor:
    """Propagate Boolean validity through the actual frozen encoder graph."""

    rule = derive_encoder_support_rule(encoder)
    if expected_rule_sha256 is not None and rule["rule_sha256"] != expected_rule_sha256:
        raise ValueError("Frozen encoder support-rule identity changed.")
    support = source_support.detach().cpu().to(torch.bool)
    if support.ndim == 3:
        support = support[None, None]
    if support.ndim != 5 or tuple(support.shape[:2]) != (1, 1):
        raise ValueError("Encoder support propagation requires (1,1,X,Y,Z) support.")
    support = _conv_support(support, encoder.stem)
    support = _res_stack_support(support, encoder.res1)
    support = _conv_support(support, encoder.down1)
    support = _res_stack_support(support, encoder.res2)
    support = _conv_support(support, encoder.down2)
    support = _res_stack_support(support, encoder.res3)
    support = _conv_support(support, encoder.to_dist)
    return support[0, 0].contiguous()


def receptive_field_source_bounds(
    rule: Mapping[str, Any],
    latent_index: Sequence[int],
    source_shape: Sequence[int],
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Return inclusive source-index dependency bounds for regression/audit evidence."""

    index = tuple(int(value) for value in latent_index)
    shape = tuple(int(value) for value in source_shape)[-3:]
    if len(index) != 3 or len(shape) != 3:
        raise ValueError("Receptive-field bounds require three spatial dimensions.")
    if rule.get("complete_spatial_dependency") == "full-input-spatial-extent":
        return tuple((0, value - 1) for value in shape)  # type: ignore[return-value]
    stride = tuple(int(value) for value in rule["output_stride"])
    radius = tuple(int(value) for value in rule["convolutional_receptive_field_radius"])
    return tuple(
        (max(0, index[axis] * stride[axis] - radius[axis]),
         min(shape[axis] - 1, index[axis] * stride[axis] + radius[axis]))
        for axis in range(3)
    )  # type: ignore[return-value]


def pack_support_mask(mask: torch.Tensor) -> torch.Tensor:
    values = mask.detach().cpu().to(torch.bool).contiguous().numpy().reshape(-1, order="C")
    packed = np.packbits(values, bitorder="little")
    return torch.from_numpy(np.ascontiguousarray(packed)).to(torch.uint8)


def unpack_support_mask(packed: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
    dimensions = tuple(int(value) for value in shape)
    if len(dimensions) != 3 or any(value <= 0 for value in dimensions):
        raise ValueError("Packed support requires a positive three-dimensional shape.")
    values = packed.detach().cpu().to(torch.uint8).contiguous().numpy()
    count = math.prod(dimensions)
    unpacked = np.unpackbits(values, bitorder="little", count=count)
    return torch.from_numpy(unpacked.reshape(dimensions, order="C").astype(np.bool_))


def standardize_supported_latent(
    latent: torch.Tensor,
    support: torch.Tensor,
    mean: Sequence[float],
    std: Sequence[float],
) -> torch.Tensor:
    work = latent.detach().cpu().to(torch.float32)
    mask = support.detach().cpu().to(torch.bool)
    if work.ndim != 4 or tuple(mask.shape) != tuple(work.shape[1:]):
        raise ValueError("Supported latent standardization shape mismatch.")
    if not bool(mask.any()):
        raise ValueError("Supported latent standardization received an empty mask.")
    channel_mean = torch.tensor(mean, dtype=torch.float32)
    channel_std = torch.tensor(std, dtype=torch.float32)
    if channel_mean.shape != (work.shape[0],) or channel_std.shape != channel_mean.shape:
        raise ValueError("Latent-statistics channel count mismatch.")
    if not bool(torch.isfinite(channel_mean).all()) or not bool(torch.isfinite(channel_std).all()):
        raise ValueError("Latent standardization statistics are non-finite.")
    if bool((channel_std <= 0).any()):
        raise ValueError("Latent standardization statistics are degenerate.")
    standardized = (work - channel_mean[:, None, None, None]) / channel_std[
        :, None, None, None
    ]
    # Unsupported values are erased before hashing or descriptor arithmetic.
    return torch.where(mask[None], standardized, torch.zeros_like(standardized)).contiguous()


def structural_descriptor(
    standardized_latent: torch.Tensor,
    support: torch.Tensor,
    *,
    pool_output_sizes: Sequence[int] = (1, 2, 4),
) -> torch.Tensor:
    """Fixed masked multiscale intensity/gradient descriptor."""

    latent = standardized_latent.detach().cpu().to(torch.float32)
    mask = support.detach().cpu().to(torch.bool)
    if latent.ndim != 4 or tuple(mask.shape) != tuple(latent.shape[1:]):
        raise ValueError("Structural descriptor latent/support shape mismatch.")
    if not bool(mask.any()):
        raise ValueError("Structural descriptor support is empty.")
    if not bool(torch.isfinite(latent[:, mask]).all()):
        raise ValueError("Structural descriptor supported values are non-finite.")
    masked = torch.where(mask[None], latent, torch.zeros_like(latent))
    features: list[tuple[torch.Tensor, torch.Tensor]] = [(masked, mask[None])]
    for axis in range(3):
        dim = axis + 1
        before = latent.narrow(dim, 0, latent.shape[dim] - 1)
        after = latent.narrow(dim, 1, latent.shape[dim] - 1)
        mask_before = mask.narrow(axis, 0, mask.shape[axis] - 1)
        mask_after = mask.narrow(axis, 1, mask.shape[axis] - 1)
        valid = (mask_before & mask_after)[None]
        difference = torch.where(valid, (after - before).abs(), torch.zeros_like(after))
        padding = [0, 0, 0, 0, 0, 0]
        padding[2 * (2 - axis) + 1] = 1
        features.append((F.pad(difference, padding), F.pad(valid, padding)))
    pooled: list[torch.Tensor] = []
    for values, valid in features:
        valid_values = valid.to(torch.float32)
        for size in pool_output_sizes:
            output_size = (int(size),) * 3
            numerator = F.adaptive_avg_pool3d(
                (values * valid_values).unsqueeze(0), output_size
            )[0]
            denominator = F.adaptive_avg_pool3d(valid_values.unsqueeze(0), output_size)[0]
            pooled.append(
                torch.where(
                    denominator > 0,
                    numerator / denominator.clamp_min(1e-12),
                    torch.zeros_like(numerator),
                ).reshape(-1)
            )
    descriptor = torch.cat(pooled).to(torch.float32).contiguous()
    if not bool(torch.isfinite(descriptor).all()):
        raise ValueError("Structural descriptor produced non-finite values.")
    return descriptor


def preflight_photometry_factored_latent_bank(
    *,
    encoder: Any,
    artifact: FrozenPhotometryArtifact,
    qualification: Mapping[str, Any],
    records_by_split: Mapping[str, Sequence[VolumeRecord]],
    config: PhotometryFactoredLatentBankConfig,
    resolved_config: Mapping[str, Any],
    source_split_file_sha256: str,
    source_membership_fingerprint: str,
    source_recovery_fingerprint: str,
    photometry_artifact_file_sha256: str,
    qualification_file_sha256: str,
    vae_config_sha256: str,
    vae_checkpoint_sha256: str,
    code_provenance: Mapping[str, Any],
    source_shape_resolver: SourceShapeResolver,
    device: torch.device,
    source_file_hasher: FileHasher = sha256_file,
    publication_linker: Linker = os.link,
) -> dict[str, Any]:
    prepared = _prepare_run(
        encoder=encoder,
        artifact=artifact,
        qualification=qualification,
        records_by_split=records_by_split,
        config=config,
        resolved_config=resolved_config,
        source_split_file_sha256=source_split_file_sha256,
        source_membership_fingerprint=source_membership_fingerprint,
        source_recovery_fingerprint=source_recovery_fingerprint,
        photometry_artifact_file_sha256=photometry_artifact_file_sha256,
        qualification_file_sha256=qualification_file_sha256,
        vae_config_sha256=vae_config_sha256,
        vae_checkpoint_sha256=vae_checkpoint_sha256,
        code_provenance=code_provenance,
        source_shape_resolver=source_shape_resolver,
        device=device,
        source_file_hasher=source_file_hasher,
        publication_linker=publication_linker,
        check_publication=True,
    )
    return {
        "contract_version": "photometry-factored-latent-bank-preflight-v1",
        "run_fingerprint": prepared.run_fingerprint,
        "eligibility": {
            "accepted_record_count": len(prepared.eligible),
            "prospective_accepted_count": 0,
            "prospective_excluded_count": len(prepared.excluded),
            "classification_completed_before_source_array_load": True,
        },
        "support_rule": prepared.support_rule,
        "storage": prepared.storage_report,
        "filesystem": prepared.filesystem_report,
    }


def build_photometry_factored_latent_bank(
    *,
    encoder: Any,
    artifact: FrozenPhotometryArtifact,
    qualification: Mapping[str, Any],
    records_by_split: Mapping[str, Sequence[VolumeRecord]],
    config: PhotometryFactoredLatentBankConfig,
    resolved_config: Mapping[str, Any],
    source_split_file_sha256: str,
    source_membership_fingerprint: str,
    source_recovery_fingerprint: str,
    photometry_artifact_file_sha256: str,
    qualification_file_sha256: str,
    vae_config_sha256: str,
    vae_checkpoint_sha256: str,
    code_provenance: Mapping[str, Any],
    volume_loader: VolumeLoader,
    source_shape_resolver: SourceShapeResolver,
    device: torch.device,
    source_file_hasher: FileHasher = sha256_file,
    publication_linker: Linker = os.link,
    resume: bool = False,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Stream N_d(x) directly into the frozen VAE posterior-mean bank."""

    prepared = _prepare_run(
        encoder=encoder,
        artifact=artifact,
        qualification=qualification,
        records_by_split=records_by_split,
        config=config,
        resolved_config=resolved_config,
        source_split_file_sha256=source_split_file_sha256,
        source_membership_fingerprint=source_membership_fingerprint,
        source_recovery_fingerprint=source_recovery_fingerprint,
        photometry_artifact_file_sha256=photometry_artifact_file_sha256,
        qualification_file_sha256=qualification_file_sha256,
        vae_config_sha256=vae_config_sha256,
        vae_checkpoint_sha256=vae_checkpoint_sha256,
        code_provenance=code_provenance,
        source_shape_resolver=source_shape_resolver,
        device=device,
        source_file_hasher=source_file_hasher,
        publication_linker=publication_linker,
        check_publication=True,
    )
    out_dir = config.out_dir
    manifest_path = out_dir / FACTORED_LATENT_BANK_MANIFEST
    if manifest_path.exists():
        if not resume:
            raise FileExistsError(f"Refusing to overwrite factored bank: {manifest_path}")
        manifest = load_photometry_factored_latent_bank_manifest(out_dir)
        _validate_manifest_against_run(manifest, prepared, code_provenance)
        _verify_complete_published_artifact(out_dir, manifest)
        return manifest

    assert_frozen(encoder)
    encoder = encoder.to(device).eval()
    store_dtype = {"float16": torch.float16, "float32": torch.float32}[
        config.store_dtype
    ]
    records: list[dict[str, Any]] = []
    total = len(prepared.eligible)
    for index, item in enumerate(prepared.eligible, start=1):
        identity = item.record.case_id
        source_identity = eligible_source_identity(item, prepared.source_shapes[identity])
        resume_key = sha256_json(
            {"run_fingerprint": prepared.run_fingerprint, "record": source_identity}
        )
        destination = _latent_record_path(out_dir, item.split, identity)
        if destination.exists():
            if not resume:
                raise FileExistsError(f"Refusing to overwrite factored latent: {destination}")
            payload = _load_latent_record(destination, expected_resume_key=resume_key)
        else:
            source = validated_source_volume(volume_loader(item.record), identity)
            if tuple(source.shape) != prepared.source_shapes[identity]:
                raise ValueError("Loaded source shape differs from the storage preflight.")
            source_loaded_sha = canonical_tensor_sha256(source)
            canonical_context = artifact.normalize_source(source, item.record.domain)
            canonical = canonical_context.values.detach().cpu().to(torch.float32).contiguous()
            source_support = (
                canonical_context.support_mask.detach().cpu().to(torch.bool).contiguous()
            )
            canonical_sha = canonical_tensor_sha256(canonical)
            source_support_sha = storage_tensor_sha256(source_support)
            latent, path_used = encode_latent(
                encoder,
                canonical.to(device),
                item.record.domain,
                strategy="full",
                block_size=(1, 1, 1),
                halo=(0, 0, 0),
                precision=config.precision,
            )
            if path_used != "full":
                raise ValueError("Factored-bank v1 requires actual path_used=full.")
            stored = latent[0].detach().cpu().to(store_dtype).contiguous()
            if not bool(torch.isfinite(stored).all()):
                raise ValueError("Posterior-mean latent contains non-finite values.")
            latent_support = propagate_encoder_support(
                source_support,
                encoder,
                expected_rule_sha256=prepared.support_rule["rule_sha256"],
            )
            if tuple(latent_support.shape) != tuple(stored.shape[1:]):
                raise ValueError("Encoder-propagated support does not match latent shape.")
            support_count = int(latent_support.sum())
            if support_count <= 0:
                raise ValueError(
                    "Encoder-conservative latent support is empty. The frozen encoder graph "
                    "cannot support masked statistics for this record."
                )
            packed = pack_support_mask(latent_support)
            sidecar: dict[str, Any] = {
                "contract_version": PHOTOMETRY_FACTORED_LATENT_BANK_VERSION,
                **source_identity,
                "source_loaded_array_sha256": source_loaded_sha,
                "source_dtype": dtype_name(source.dtype),
                "canonical_contract_version": CANONICAL_VOLUME_CONTRACT_VERSION,
                "canonical_stream_semantics": CANONICAL_STREAM_SEMANTICS,
                "canonical_persisted": False,
                "canonical_tensor_sha256": canonical_sha,
                "canonical_shape": list(canonical.shape),
                "canonical_dtype": "float32",
                "source_support_policy": PHOTOMETRY_SUPPORT_POLICY,
                "source_support_tensor_sha256": source_support_sha,
                "source_support_shape": list(source_support.shape),
                "source_support_nonzero_count": int(source_support.sum()),
                "latent_tensor_sha256": storage_tensor_sha256(stored),
                "latent_shape": list(stored.shape),
                "latent_dtype": dtype_name(stored.dtype),
                "encoding_path": {
                    "photometry": "FrozenPhotometryArtifact.normalize_source",
                    "vae": "frozen_encode_dist_posterior_mean",
                    "strategy_requested": "full",
                    "path_used": path_used,
                    "oom_fallback": "forbidden-hard-stop",
                    "precision": config.precision,
                },
                "downsample_factor": downsample_factor(encoder),
                "support_propagation_rule": SUPPORT_PROPAGATION_RULE,
                "support_rule_sha256": prepared.support_rule["rule_sha256"],
                "receptive_field_evidence": {
                    "graph_sha256": prepared.support_rule["graph_sha256"],
                    "convolutional_receptive_field_size": prepared.support_rule[
                        "convolutional_receptive_field_size"
                    ],
                    "output_stride": prepared.support_rule["output_stride"],
                    "alignment": prepared.support_rule["alignment"],
                    "complete_spatial_dependency": prepared.support_rule[
                        "complete_spatial_dependency"
                    ],
                },
                "latent_support_mask_sha256": storage_tensor_sha256(latent_support),
                "support_shape": list(latent_support.shape),
                "support_dtype": "bool",
                "support_nonzero_count": support_count,
                "support_packing_rule": SUPPORT_PACKING_RULE,
                "packed_support_shape": list(packed.shape),
                "packed_support_dtype": "uint8",
                "packed_support_sha256": storage_tensor_sha256(packed),
                "photometry_artifact_sha256": artifact.artifact_sha256,
                "photometry_artifact_file_sha256": photometry_artifact_file_sha256,
                "photometry_resolved_config_sha256": artifact.provenance[
                    "resolved_config_sha256"
                ],
                "qualification_result_sha256": prepared.qualified["result_sha256"],
                "qualification_file_sha256": qualification_file_sha256,
                "vae_config_sha256": vae_config_sha256,
                "vae_checkpoint_sha256": vae_checkpoint_sha256,
                "source_split_file_sha256": source_split_file_sha256,
                "source_membership_fingerprint": source_membership_fingerprint,
                "source_recovery_fingerprint": source_recovery_fingerprint,
                "computational_provenance_sha256": code_provenance[
                    "provenance_sha256"
                ],
                "dependency_map_sha256": code_provenance["dependency_map_sha256"],
                "runtime_sha256": code_provenance["runtime_sha256"],
                "resume_key": resume_key,
            }
            sidecar["source_content_fingerprint"] = _source_content_fingerprint(sidecar)
            sidecar_sha = sha256_json(sidecar)
            atomic_torch_save_no_clobber(
                destination,
                {
                    "contract_version": PHOTOMETRY_FACTORED_LATENT_BANK_VERSION,
                    "sidecar": sidecar,
                    "sidecar_sha256": sidecar_sha,
                    "latent": stored,
                    "packed_latent_support": packed,
                },
                linker=publication_linker,
            )
            payload = _load_latent_record(destination, expected_resume_key=resume_key)
            del source, canonical, source_support, latent, stored, latent_support, packed
        records.append(_latent_manifest_entry(payload, destination, out_dir))
        if progress is not None:
            progress("streamed_factored_latent", index, total, identity)

    records.sort(key=lambda entry: (_entry_sidecar(entry)["split"], _entry_identity(entry)))
    _require_factored_bank_roles(records)
    statistics = _compute_statistics_payload(
        records=records,
        root=out_dir,
        prepared=prepared,
        artifact=artifact,
        vae_config_sha256=vae_config_sha256,
        vae_checkpoint_sha256=vae_checkpoint_sha256,
        code_provenance=code_provenance,
    )
    stats_path = out_dir / FACTORED_LATENT_STATS_FILE
    write_json_resume_exact(
        stats_path, statistics, resume=resume, linker=publication_linker
    )
    descriptor_manifest = _build_descriptors(
        records=records,
        root=out_dir,
        statistics=statistics,
        config=config,
        prepared=prepared,
        code_provenance=code_provenance,
        resume=resume,
        publication_linker=publication_linker,
        progress=progress,
    )
    descriptor_manifest_path = out_dir / STRUCTURAL_DESCRIPTOR_MANIFEST
    write_json_resume_exact(
        descriptor_manifest_path,
        descriptor_manifest,
        resume=resume,
        linker=publication_linker,
    )
    manifest: dict[str, Any] = {
        "contract_version": PHOTOMETRY_FACTORED_LATENT_BANK_VERSION,
        "legacy_contract_mutation": "latent-bank-v1-unchanged",
        "scope": "retrospective-R-only",
        "canonical_stream": {
            "contract_version": CANONICAL_VOLUME_CONTRACT_VERSION,
            "semantics": CANONICAL_STREAM_SEMANTICS,
            "full_canonical_tensor_persisted": False,
            "full_source_support_persisted": False,
        },
        "encoding": {
            "encoder_statistic": "posterior_mean",
            "strategy_requested": "full",
            "path_used_required": "full",
            "oom_fallback": "forbidden-hard-stop",
            "precision": config.precision,
            "store_dtype": config.store_dtype,
        },
        "resolved_config": prepared.resolved,
        "resolved_config_sha256": prepared.config_sha256,
        "source_split": prepared.source_split,
        "photometry": {
            "contract_version": PHOTOMETRY_FACTORIZATION_CONTRACT_VERSION,
            "artifact_sha256": artifact.artifact_sha256,
            "artifact_file_sha256": photometry_artifact_file_sha256,
            "resolved_config_sha256": artifact.provenance["resolved_config_sha256"],
        },
        "qualification": {
            "result_sha256": prepared.qualified["result_sha256"],
            "file_sha256": qualification_file_sha256,
            "canonical_latent_bank_authorized": True,
        },
        "vae": {
            "config_sha256": vae_config_sha256,
            "checkpoint_sha256": vae_checkpoint_sha256,
        },
        "computational_provenance": json_safe_mapping(code_provenance),
        "support_rule": prepared.support_rule,
        "storage_preflight": prepared.storage_report,
        "filesystem_preflight": prepared.filesystem_report,
        "run_fingerprint": prepared.run_fingerprint,
        "eligibility_proof": {
            "accepted_cohort": "R",
            "accepted_record_count": len(records),
            "prospective_accepted_count": 0,
            "prospective_excluded_count": len(prepared.excluded),
            "classification_completed_before_source_array_load": True,
        },
        "excluded_prospective_records": list(prepared.excluded),
        "excluded_prospective_records_sha256": sha256_json(list(prepared.excluded)),
        "domain_counts": _domain_counts(records),
        "record_count": len(records),
        "records": records,
        "records_sha256": sha256_json(records),
        "latent_statistics": {
            "path": FACTORED_LATENT_STATS_FILE,
            "file_sha256": sha256_file(stats_path),
            "artifact_sha256": statistics["artifact_sha256"],
        },
        "structural_descriptors": {
            "path": STRUCTURAL_DESCRIPTOR_MANIFEST,
            "file_sha256": sha256_file(descriptor_manifest_path),
            "artifact_sha256": descriptor_manifest["artifact_sha256"],
            "coupling_authorized": False,
            "qualification_required": (
                PHOTOMETRY_FACTORED_DESCRIPTOR_QUALIFICATION_VERSION
            ),
        },
    }
    manifest["artifact_sha256"] = sha256_json(manifest)
    write_json_resume_exact(
        manifest_path, manifest, resume=resume, linker=publication_linker
    )
    return load_photometry_factored_latent_bank_manifest(out_dir)


def load_photometry_factored_latent_bank_manifest(
    root: str | Path, *, expected_artifact_sha256: str | None = None
) -> dict[str, Any]:
    manifest = _load_json(Path(root) / FACTORED_LATENT_BANK_MANIFEST)
    if manifest.get("contract_version") != PHOTOMETRY_FACTORED_LATENT_BANK_VERSION:
        raise ValueError("Photometry-factored latent-bank contract mismatch.")
    if (
        expected_artifact_sha256 is not None
        and manifest["artifact_sha256"] != expected_artifact_sha256
    ):
        raise ValueError("Photometry-factored latent-bank artifact identity mismatch.")
    if manifest.get("legacy_contract_mutation") != "latent-bank-v1-unchanged":
        raise ValueError("Legacy latent-bank-v1 compatibility boundary changed.")
    canonical = manifest.get("canonical_stream")
    if canonical != {
        "contract_version": CANONICAL_VOLUME_CONTRACT_VERSION,
        "semantics": CANONICAL_STREAM_SEMANTICS,
        "full_canonical_tensor_persisted": False,
        "full_source_support_persisted": False,
    }:
        raise ValueError("Factored bank persisted a forbidden full canonical artifact.")
    encoding = manifest.get("encoding", {})
    if encoding.get("strategy_requested") != "full" or encoding.get(
        "path_used_required"
    ) != "full" or encoding.get("oom_fallback") != "forbidden-hard-stop":
        raise ValueError("Factored bank encoding path is not sealed to full-only.")
    if manifest.get("resolved_config_sha256") != sha256_json(manifest.get("resolved_config")):
        raise ValueError("Factored-bank resolved-config hash mismatch.")
    validate_canonical_artifact_config(manifest["resolved_config"])
    validate_canonical_artifact_code_provenance(manifest["computational_provenance"])
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("Factored-bank manifest has no records.")
    if manifest.get("record_count") != len(records) or manifest.get(
        "records_sha256"
    ) != sha256_json(records):
        raise ValueError("Factored-bank record-list identity mismatch.")
    _require_factored_bank_roles(records)
    if manifest.get("eligibility_proof", {}).get("prospective_accepted_count") != 0:
        raise ValueError("Factored-bank manifest accepted prospective data.")
    excluded = manifest.get("excluded_prospective_records")
    if not isinstance(excluded, list) or sha256_json(excluded) != manifest.get(
        "excluded_prospective_records_sha256"
    ):
        raise ValueError("Factored-bank prospective-exclusion evidence mismatch.")
    descriptors = manifest.get("structural_descriptors", {})
    if descriptors.get("coupling_authorized") is not False or descriptors.get(
        "qualification_required"
    ) != PHOTOMETRY_FACTORED_DESCRIPTOR_QUALIFICATION_VERSION:
        raise ValueError("Structural-descriptor qualification boundary is missing.")
    return manifest


def audit_photometry_factored_latent_bank(
    *,
    root: str | Path,
    encoder: Any,
    artifact: FrozenPhotometryArtifact,
    qualification: Mapping[str, Any],
    records_by_split: Mapping[str, Sequence[VolumeRecord]],
    resolved_config: Mapping[str, Any],
    source_split_file_sha256: str,
    source_membership_fingerprint: str,
    source_recovery_fingerprint: str,
    photometry_artifact_file_sha256: str,
    qualification_file_sha256: str,
    vae_config_sha256: str,
    vae_checkpoint_sha256: str,
    code_provenance: Mapping[str, Any],
    volume_loader: VolumeLoader,
    source_shape_resolver: SourceShapeResolver,
    device: torch.device,
    source_file_hasher: FileHasher = sha256_file,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Recompute source -> N_d -> full E and every derived artifact without writing."""

    root_path = Path(root)
    config = PhotometryFactoredLatentBankConfig.from_mapping(
        resolved_config, out_dir=root_path
    )
    prepared = _prepare_run(
        encoder=encoder,
        artifact=artifact,
        qualification=qualification,
        records_by_split=records_by_split,
        config=config,
        resolved_config=resolved_config,
        source_split_file_sha256=source_split_file_sha256,
        source_membership_fingerprint=source_membership_fingerprint,
        source_recovery_fingerprint=source_recovery_fingerprint,
        photometry_artifact_file_sha256=photometry_artifact_file_sha256,
        qualification_file_sha256=qualification_file_sha256,
        vae_config_sha256=vae_config_sha256,
        vae_checkpoint_sha256=vae_checkpoint_sha256,
        code_provenance=code_provenance,
        source_shape_resolver=source_shape_resolver,
        device=device,
        source_file_hasher=source_file_hasher,
        publication_linker=os.link,
        check_publication=False,
    )
    manifest = load_photometry_factored_latent_bank_manifest(root_path)
    _validate_manifest_against_run(manifest, prepared, code_provenance)
    assert_frozen(encoder)
    encoder = encoder.to(device).eval()
    by_identity = {item.record.case_id: item for item in prepared.eligible}
    recomputed: list[tuple[Mapping[str, Any], torch.Tensor, torch.Tensor]] = []
    store_dtype = {"float16": torch.float16, "float32": torch.float32}[
        config.store_dtype
    ]
    for index, entry in enumerate(manifest["records"], start=1):
        sidecar = _entry_sidecar(entry)
        identity = str(sidecar["record_identity"])
        item = by_identity[identity]
        path = safe_relative_path(root_path, str(entry["path"]))
        if sha256_file(path) != entry["payload_file_sha256"]:
            raise ValueError("Factored latent record file hash mismatch.")
        loaded = _load_latent_record(path, expected_resume_key=sidecar["resume_key"])
        source = validated_source_volume(volume_loader(item.record), identity)
        if tuple(source.shape) != prepared.source_shapes[identity]:
            raise ValueError("Audit source shape differs from the sealed preflight.")
        canonical_context = artifact.normalize_source(source, item.record.domain)
        canonical = canonical_context.values.detach().cpu().to(torch.float32).contiguous()
        source_support = canonical_context.support_mask.detach().cpu().to(torch.bool).contiguous()
        if canonical_tensor_sha256(source) != sidecar["source_loaded_array_sha256"]:
            raise ValueError("Audit loaded source tensor identity changed.")
        if canonical_tensor_sha256(canonical) != sidecar["canonical_tensor_sha256"]:
            raise ValueError("Audit N_d(source) tensor identity changed.")
        if storage_tensor_sha256(source_support) != sidecar[
            "source_support_tensor_sha256"
        ]:
            raise ValueError("Audit source support identity changed.")
        latent, path_used = encode_latent(
            encoder,
            canonical.to(device),
            item.record.domain,
            strategy="full",
            block_size=(1, 1, 1),
            halo=(0, 0, 0),
            precision=config.precision,
        )
        if path_used != "full":
            raise ValueError("Audit VAE encoding did not use the required full path.")
        stored = latent[0].detach().cpu().to(store_dtype).contiguous()
        support = propagate_encoder_support(
            source_support,
            encoder,
            expected_rule_sha256=prepared.support_rule["rule_sha256"],
        )
        if storage_tensor_sha256(stored) != sidecar["latent_tensor_sha256"]:
            raise ValueError("Audit posterior-mean latent identity changed.")
        if storage_tensor_sha256(support) != sidecar["latent_support_mask_sha256"]:
            raise ValueError("Audit encoder-propagated support identity changed.")
        if storage_tensor_sha256(pack_support_mask(support)) != sidecar[
            "packed_support_sha256"
        ]:
            raise ValueError("Audit packed latent support identity changed.")
        if not torch.equal(stored, loaded["latent"]):
            raise ValueError("Stored posterior-mean latent differs from recomputation.")
        recomputed.append((sidecar, stored, support))
        if progress is not None:
            progress("audit_streamed_factored_latent", index, len(manifest["records"]), identity)

    statistics = _statistics_from_samples(
        samples=[item for item in recomputed if item[0]["split"] == "train"],
        prepared=prepared,
        artifact=artifact,
        vae_config_sha256=vae_config_sha256,
        vae_checkpoint_sha256=vae_checkpoint_sha256,
        code_provenance=code_provenance,
    )
    stats_path = root_path / FACTORED_LATENT_STATS_FILE
    stored_statistics = _load_json(stats_path)
    if statistics != stored_statistics or sha256_file(stats_path) != manifest[
        "latent_statistics"
    ]["file_sha256"]:
        raise ValueError("Masked train-only latent statistics do not recompute exactly.")
    descriptor_manifest = _load_json(root_path / STRUCTURAL_DESCRIPTOR_MANIFEST)
    _audit_descriptors_from_samples(
        samples=[item for item in recomputed if item[0]["split"] == "train"],
        root=root_path,
        statistics=statistics,
        descriptor_manifest=descriptor_manifest,
        config=config,
        prepared=prepared,
        code_provenance=code_provenance,
    )
    return {
        "contract_version": PHOTOMETRY_FACTORED_LATENT_BANK_VERSION,
        "artifact_sha256": manifest["artifact_sha256"],
        "record_count": len(recomputed),
        "all_records_verified": True,
        "source_to_N_d_to_E_recomputed": True,
        "encoding_path_used": "full",
        "masked_train_statistics_verified": True,
        "structural_descriptors_verified": True,
        "coupling_authorized": False,
        "qualification_required": PHOTOMETRY_FACTORED_DESCRIPTOR_QUALIFICATION_VERSION,
    }


def _prepare_run(
    *,
    encoder: Any,
    artifact: FrozenPhotometryArtifact,
    qualification: Mapping[str, Any],
    records_by_split: Mapping[str, Sequence[VolumeRecord]],
    config: PhotometryFactoredLatentBankConfig,
    resolved_config: Mapping[str, Any],
    source_split_file_sha256: str,
    source_membership_fingerprint: str,
    source_recovery_fingerprint: str,
    photometry_artifact_file_sha256: str,
    qualification_file_sha256: str,
    vae_config_sha256: str,
    vae_checkpoint_sha256: str,
    code_provenance: Mapping[str, Any],
    source_shape_resolver: SourceShapeResolver,
    device: torch.device,
    source_file_hasher: FileHasher,
    publication_linker: Linker,
    check_publication: bool,
) -> _PreparedRun:
    resolved = validate_canonical_artifact_config(resolved_config)
    expected_config = PhotometryFactoredLatentBankConfig.from_mapping(
        resolved, out_dir=config.out_dir
    )
    if config != expected_config:
        raise ValueError("Factored-bank runtime config differs from the resolved config.")
    if config.strategy != "full":
        raise ValueError("Factored-bank v1 strategy must remain full.")
    for name, value in (
        ("source split file", source_split_file_sha256),
        ("photometry artifact file", photometry_artifact_file_sha256),
        ("qualification file", qualification_file_sha256),
        ("VAE config", vae_config_sha256),
        ("VAE checkpoint", vae_checkpoint_sha256),
    ):
        require_sha256(value, name)
    if source_split_file_sha256 != artifact.provenance["source_split_file_sha256"]:
        raise ValueError("Factored-bank split-file identity differs from photometry.")
    if source_membership_fingerprint != artifact.split_fingerprint:
        raise ValueError("Factored-bank membership fingerprint differs from photometry.")
    if source_recovery_fingerprint != artifact.recovery_fingerprint:
        raise ValueError("Factored-bank recovery fingerprint differs from photometry.")
    qualified = validate_variant_a_qualification(
        qualification,
        artifact=artifact,
        source_split_file_sha256=source_split_file_sha256,
        source_membership_fingerprint=source_membership_fingerprint,
        source_recovery_fingerprint=source_recovery_fingerprint,
        vae_config_sha256=vae_config_sha256,
        vae_checkpoint_sha256=vae_checkpoint_sha256,
    )
    validate_canonical_artifact_code_provenance(code_provenance)
    if code_provenance["runtime"]["device"]["type"] != device.type:
        raise ValueError("Computational provenance device differs from the runtime device.")
    assert_frozen(encoder)
    eligible, excluded = preflight_retrospective_records(
        records_by_split,
        source_file_hasher=source_file_hasher,
    )
    source_shapes = resolve_source_shapes(eligible, source_shape_resolver)
    support_rule = derive_encoder_support_rule(encoder)
    factor = downsample_factor(encoder)
    latent_channels = int(getattr(encoder, "latent_channels", 0))
    storage_report = build_storage_preflight_report(
        records=eligible,
        source_shapes=source_shapes,
        downsample_factor=factor,
        latent_channels=latent_channels,
        store_dtype=config.store_dtype,
        descriptor_pool_sizes=config.descriptor_pool_sizes,
        resolved_config=resolved,
    )
    filesystem_report = (
        preflight_atomic_no_clobber_filesystem(
            config.out_dir,
            required_free_bytes=storage_report["required_free_storage_bytes"],
            linker=publication_linker,
        )
        if check_publication
        else None
    )
    source_split = {
        "file_sha256": source_split_file_sha256,
        "membership_fingerprint": source_membership_fingerprint,
        "recovery_fingerprint": source_recovery_fingerprint,
    }
    config_sha = sha256_json(resolved)
    run_identity = {
        "contract_version": PHOTOMETRY_FACTORED_LATENT_BANK_VERSION,
        "canonical_stream_contract": CANONICAL_VOLUME_CONTRACT_VERSION,
        "canonical_stream_semantics": CANONICAL_STREAM_SEMANTICS,
        "source_split": source_split,
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
        "computational_provenance_sha256": code_provenance["provenance_sha256"],
        "dependency_map_sha256": code_provenance["dependency_map_sha256"],
        "runtime_sha256": code_provenance["runtime_sha256"],
        "support_rule_sha256": support_rule["rule_sha256"],
        "storage_report_sha256": storage_report["report_sha256"],
        "encoding": {
            "strategy": "full",
            "path_used_required": "full",
            "precision": config.precision,
            "store_dtype": config.store_dtype,
        },
        "records": [
            eligible_source_identity(item, source_shapes[item.record.case_id])
            for item in eligible
        ],
        "excluded_prospective_records": excluded,
    }
    return _PreparedRun(
        resolved=resolved,
        qualified=qualified,
        eligible=tuple(eligible),
        excluded=tuple(excluded),
        source_shapes=source_shapes,
        support_rule=support_rule,
        storage_report=storage_report,
        filesystem_report=filesystem_report,
        source_split=source_split,
        run_fingerprint=sha256_json(run_identity),
        config_sha256=config_sha,
    )


def _compute_statistics_payload(
    *,
    records: Sequence[Mapping[str, Any]],
    root: Path,
    prepared: _PreparedRun,
    artifact: FrozenPhotometryArtifact,
    vae_config_sha256: str,
    vae_checkpoint_sha256: str,
    code_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    samples: list[tuple[Mapping[str, Any], torch.Tensor, torch.Tensor]] = []
    for entry in records:
        sidecar = _entry_sidecar(entry)
        if sidecar["split"] != "train":
            continue
        payload = _load_latent_record(
            safe_relative_path(root, str(entry["path"])),
            expected_resume_key=sidecar["resume_key"],
        )
        support = unpack_support_mask(
            payload["packed_latent_support"], sidecar["support_shape"]
        )
        samples.append((sidecar, payload["latent"], support))
    return _statistics_from_samples(
        samples=samples,
        prepared=prepared,
        artifact=artifact,
        vae_config_sha256=vae_config_sha256,
        vae_checkpoint_sha256=vae_checkpoint_sha256,
        code_provenance=code_provenance,
    )


def _statistics_from_samples(
    *,
    samples: Sequence[tuple[Mapping[str, Any], torch.Tensor, torch.Tensor]],
    prepared: _PreparedRun,
    artifact: FrozenPhotometryArtifact,
    vae_config_sha256: str,
    vae_checkpoint_sha256: str,
    code_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    if not samples:
        raise ValueError("No R/train canonical latents were available for statistics.")
    accumulator = MaskedChannelWelford(int(samples[0][1].shape[0]))
    records: list[dict[str, Any]] = []
    for sidecar, latent, support in samples:
        if sidecar["cohort"] != "R" or sidecar["split"] != "train":
            raise ValueError("Canonical latent statistics are restricted to R/train.")
        accumulator.update(latent, support)
        records.append(
            {
                "record_identity": sidecar["record_identity"],
                "record_identity_sha256": sidecar["record_identity_sha256"],
                "source_content_fingerprint": sidecar["source_content_fingerprint"],
                "latent_tensor_sha256": sidecar["latent_tensor_sha256"],
                "latent_support_mask_sha256": sidecar["latent_support_mask_sha256"],
                "supported_cell_count": sidecar["support_nonzero_count"],
            }
        )
    records.sort(key=lambda item: item["record_identity"])
    payload: dict[str, Any] = {
        "contract_version": PHOTOMETRY_FACTORED_LATENT_STATS_VERSION,
        "computed_over": {"cohort": "R", "split": "train", "cells": "supported_only"},
        **accumulator.compute(),
        "record_count": len(records),
        "records": records,
        "records_sha256": sha256_json(records),
        "bank_run_fingerprint": prepared.run_fingerprint,
        "source_split": prepared.source_split,
        "photometry_artifact_sha256": artifact.artifact_sha256,
        "vae_config_sha256": vae_config_sha256,
        "vae_checkpoint_sha256": vae_checkpoint_sha256,
        "resolved_config_sha256": prepared.config_sha256,
        "computational_provenance_sha256": code_provenance["provenance_sha256"],
        "dependency_map_sha256": code_provenance["dependency_map_sha256"],
        "runtime_sha256": code_provenance["runtime_sha256"],
    }
    payload["artifact_sha256"] = sha256_json(payload)
    return payload


def _build_descriptors(
    *,
    records: Sequence[Mapping[str, Any]],
    root: Path,
    statistics: Mapping[str, Any],
    config: PhotometryFactoredLatentBankConfig,
    prepared: _PreparedRun,
    code_provenance: Mapping[str, Any],
    resume: bool,
    publication_linker: Linker,
    progress: Progress | None,
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    train_records = [entry for entry in records if _entry_sidecar(entry)["split"] == "train"]
    descriptor_config = _descriptor_config(config)
    for index, entry in enumerate(train_records, start=1):
        latent_payload = _load_latent_record(
            safe_relative_path(root, str(entry["path"])),
            expected_resume_key=_entry_sidecar(entry)["resume_key"],
        )
        sidecar = latent_payload["sidecar"]
        support = unpack_support_mask(
            latent_payload["packed_latent_support"], sidecar["support_shape"]
        )
        standardized = standardize_supported_latent(
            latent_payload["latent"],
            support,
            statistics["per_channel_mean"],
            statistics["per_channel_std"],
        )
        descriptor = structural_descriptor(
            standardized, support, pool_output_sizes=config.descriptor_pool_sizes
        )
        resume_key = sha256_json(
            {
                "bank_run_fingerprint": prepared.run_fingerprint,
                "record_identity": sidecar["record_identity"],
                "source_content_fingerprint": sidecar["source_content_fingerprint"],
                "latent_statistics_sha256": statistics["artifact_sha256"],
                "descriptor_config_sha256": sha256_json(descriptor_config),
            }
        )
        destination = _descriptor_record_path(root, sidecar["record_identity"])
        if destination.exists():
            if not resume:
                raise FileExistsError(f"Refusing to overwrite descriptor: {destination}")
            payload = _load_descriptor_record(destination, expected_resume_key=resume_key)
        else:
            descriptor_sidecar = {
                "contract_version": PHOTOMETRY_FACTORED_DESCRIPTOR_VERSION,
                "record_identity": sidecar["record_identity"],
                "record_identity_sha256": sidecar["record_identity_sha256"],
                "subject_identity": sidecar["subject_identity"],
                "subject_group_identity": sidecar["subject_group_identity"],
                "cohort": "R",
                "split": "train",
                "domain": sidecar["domain"],
                "input": DESCRIPTOR_INPUT,
                "standardized_supported_latent_sha256": storage_tensor_sha256(standardized),
                "source_content_fingerprint": sidecar["source_content_fingerprint"],
                "latent_support_mask_sha256": sidecar["latent_support_mask_sha256"],
                "latent_statistics_sha256": statistics["artifact_sha256"],
                "bank_run_fingerprint": prepared.run_fingerprint,
                "resolved_config_sha256": prepared.config_sha256,
                "computational_provenance_sha256": code_provenance["provenance_sha256"],
                "dependency_map_sha256": code_provenance["dependency_map_sha256"],
                "runtime_sha256": code_provenance["runtime_sha256"],
                "descriptor_config_sha256": sha256_json(descriptor_config),
                "descriptor_shape": list(descriptor.shape),
                "descriptor_dtype": "float32",
                "descriptor_tensor_sha256": storage_tensor_sha256(descriptor),
                "paired_endpoint_or_target_input": "none",
                "coupling_authorized": False,
                "qualification_required": (
                    PHOTOMETRY_FACTORED_DESCRIPTOR_QUALIFICATION_VERSION
                ),
                "learned_disentanglement_claim": "forbidden",
                "resume_key": resume_key,
            }
            descriptor_sidecar_sha = sha256_json(descriptor_sidecar)
            atomic_torch_save_no_clobber(
                destination,
                {
                    "contract_version": PHOTOMETRY_FACTORED_DESCRIPTOR_VERSION,
                    "sidecar": descriptor_sidecar,
                    "sidecar_sha256": descriptor_sidecar_sha,
                    "descriptor": descriptor,
                },
                linker=publication_linker,
            )
            payload = _load_descriptor_record(destination, expected_resume_key=resume_key)
        entries.append(_descriptor_manifest_entry(payload, destination, root))
        if progress is not None:
            progress(
                "structural_descriptor",
                index,
                len(train_records),
                sidecar["record_identity"],
            )
    return _make_descriptor_manifest(
        entries=entries,
        statistics=statistics,
        config=config,
        prepared=prepared,
        code_provenance=code_provenance,
    )


def _make_descriptor_manifest(
    *,
    entries: Sequence[Mapping[str, Any]],
    statistics: Mapping[str, Any],
    config: PhotometryFactoredLatentBankConfig,
    prepared: _PreparedRun,
    code_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    records = sorted((json_safe_mapping(item) for item in entries), key=_entry_identity)
    descriptor_config = _descriptor_config(config)
    manifest: dict[str, Any] = {
        "contract_version": PHOTOMETRY_FACTORED_DESCRIPTOR_VERSION,
        "computed_over": {"cohort": "R", "split": "train"},
        "input": DESCRIPTOR_INPUT,
        "paired_endpoint_or_target_input": "forbidden",
        "coupling_authorized": False,
        "qualification_required": PHOTOMETRY_FACTORED_DESCRIPTOR_QUALIFICATION_VERSION,
        "learned_disentanglement_claim": "forbidden",
        "qualification_requirements": {
            "retrospective_split": "validation",
            "subject_group_exclusion": "required",
            "retained_instance_or_anatomical_signal": "must-demonstrate",
            "reduced_field_predictability": "must-demonstrate",
        },
        "descriptor_config": descriptor_config,
        "descriptor_config_sha256": sha256_json(descriptor_config),
        "latent_statistics_sha256": statistics["artifact_sha256"],
        "bank_run_fingerprint": prepared.run_fingerprint,
        "resolved_config_sha256": prepared.config_sha256,
        "computational_provenance_sha256": code_provenance["provenance_sha256"],
        "record_count": len(records),
        "records": records,
        "records_sha256": sha256_json(records),
    }
    manifest["artifact_sha256"] = sha256_json(manifest)
    return manifest


def _audit_descriptors_from_samples(
    *,
    samples: Sequence[tuple[Mapping[str, Any], torch.Tensor, torch.Tensor]],
    root: Path,
    statistics: Mapping[str, Any],
    descriptor_manifest: Mapping[str, Any],
    config: PhotometryFactoredLatentBankConfig,
    prepared: _PreparedRun,
    code_provenance: Mapping[str, Any],
) -> None:
    expected_entries: list[dict[str, Any]] = []
    by_identity = {_entry_identity(entry): entry for entry in descriptor_manifest["records"]}
    for sidecar, latent, support in samples:
        standardized = standardize_supported_latent(
            latent,
            support,
            statistics["per_channel_mean"],
            statistics["per_channel_std"],
        )
        descriptor = structural_descriptor(
            standardized, support, pool_output_sizes=config.descriptor_pool_sizes
        )
        entry = by_identity[sidecar["record_identity"]]
        path = safe_relative_path(root, entry["path"])
        if sha256_file(path) != entry["payload_file_sha256"]:
            raise ValueError("Structural descriptor file hash mismatch.")
        payload = _load_descriptor_record(path)
        descriptor_sidecar = payload["sidecar"]
        if storage_tensor_sha256(standardized) != descriptor_sidecar[
            "standardized_supported_latent_sha256"
        ]:
            raise ValueError("Standardized supported latent identity changed.")
        if storage_tensor_sha256(descriptor) != descriptor_sidecar[
            "descriptor_tensor_sha256"
        ] or not torch.equal(descriptor, payload["descriptor"]):
            raise ValueError("Structural descriptor does not recompute exactly.")
        expected_entries.append(_descriptor_manifest_entry(payload, path, root))
    expected = _make_descriptor_manifest(
        entries=expected_entries,
        statistics=statistics,
        config=config,
        prepared=prepared,
        code_provenance=code_provenance,
    )
    if expected != descriptor_manifest:
        raise ValueError("Structural descriptor manifest does not recompute exactly.")


def _descriptor_config(config: PhotometryFactoredLatentBankConfig) -> dict[str, Any]:
    return {
        "input": DESCRIPTOR_INPUT,
        "pool_output_sizes": list(config.descriptor_pool_sizes),
        "pooling": DESCRIPTOR_POOLING,
        "gradients": DESCRIPTOR_GRADIENTS,
        "feature_order": ["latent", "gradient_x", "gradient_y", "gradient_z"],
        "scale_order": list(config.descriptor_pool_sizes),
        "support": "actual-frozen-encoder-dependency-propagation",
        "unsupported_standardized_values": "forced-exact-zero-before-hash-and-features",
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
    sidecar = result.get("sidecar")
    if not isinstance(sidecar, Mapping) or result.get("sidecar_sha256") != sha256_json(sidecar):
        raise ValueError("Factored latent sidecar hash mismatch.")
    sidecar = json_safe_mapping(sidecar)
    result["sidecar"] = sidecar
    if expected_resume_key is not None and sidecar.get("resume_key") != expected_resume_key:
        raise ValueError("Factored latent record is incompatible with exact resume.")
    identity = classify_variant_a_cohort(
        case_identity=str(sidecar.get("record_identity", "")),
        metadata_prefix="R",
        supplied_cohort=str(sidecar.get("cohort", "")),
        subject_identity=sidecar.get("subject_identity"),
        allowed_cohorts=("R",),
    )
    if sidecar.get("subject_group_identity") != identity.subject_group_identity:
        raise ValueError("Factored latent subject-group identity mismatch.")
    if sidecar.get("split") not in CANONICAL_VOLUME_SPLITS:
        raise ValueError("Factored latent record has a forbidden split role.")
    if sidecar.get("canonical_persisted") is not False:
        raise ValueError("Factored latent sidecar claims a persisted canonical tensor.")
    if sidecar.get("encoding_path", {}).get("path_used") != "full":
        raise ValueError("Factored latent record did not use full encoding.")
    latent = result.get("latent")
    packed = result.get("packed_latent_support")
    if not isinstance(latent, torch.Tensor) or not isinstance(packed, torch.Tensor):
        raise ValueError("Factored latent record tensors are missing.")
    if latent.ndim != 4 or latent.dtype not in {torch.float16, torch.float32}:
        raise ValueError("Factored latent tensor shape/dtype is incompatible.")
    if not bool(torch.isfinite(latent).all()):
        raise ValueError("Factored latent tensor contains non-finite values.")
    if storage_tensor_sha256(latent) != sidecar.get("latent_tensor_sha256"):
        raise ValueError("Factored latent tensor hash mismatch.")
    if packed.dtype != torch.uint8 or storage_tensor_sha256(packed) != sidecar.get(
        "packed_support_sha256"
    ):
        raise ValueError("Factored latent packed-support hash mismatch.")
    support = unpack_support_mask(packed, sidecar.get("support_shape", ()))
    if tuple(support.shape) != tuple(latent.shape[1:]):
        raise ValueError("Factored latent support shape mismatch.")
    if int(support.sum()) <= 0 or int(support.sum()) != int(
        sidecar.get("support_nonzero_count", -1)
    ):
        raise ValueError("Factored latent support count is empty or inconsistent.")
    if storage_tensor_sha256(support) != sidecar.get("latent_support_mask_sha256"):
        raise ValueError("Factored latent support-mask hash mismatch.")
    if sidecar.get("source_content_fingerprint") != _source_content_fingerprint(sidecar):
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
    sidecar = result.get("sidecar")
    if not isinstance(sidecar, Mapping) or result.get("sidecar_sha256") != sha256_json(sidecar):
        raise ValueError("Structural descriptor sidecar hash mismatch.")
    sidecar = json_safe_mapping(sidecar)
    result["sidecar"] = sidecar
    if expected_resume_key is not None and sidecar.get("resume_key") != expected_resume_key:
        raise ValueError("Structural descriptor is incompatible with exact resume.")
    if sidecar.get("cohort") != "R" or sidecar.get("split") != "train":
        raise ValueError("Structural descriptors are R/train-only.")
    if sidecar.get("coupling_authorized") is not False or sidecar.get(
        "qualification_required"
    ) != PHOTOMETRY_FACTORED_DESCRIPTOR_QUALIFICATION_VERSION:
        raise ValueError("Structural descriptor coupling boundary is missing.")
    descriptor = result.get("descriptor")
    if not isinstance(descriptor, torch.Tensor) or descriptor.dtype != torch.float32:
        raise ValueError("Structural descriptor tensor is missing or has the wrong dtype.")
    if not bool(torch.isfinite(descriptor).all()) or storage_tensor_sha256(
        descriptor
    ) != sidecar.get("descriptor_tensor_sha256"):
        raise ValueError("Structural descriptor tensor hash mismatch.")
    return result


def _latent_manifest_entry(payload: Mapping[str, Any], path: Path, root: Path) -> dict[str, Any]:
    return {
        "sidecar": json_safe_mapping(payload["sidecar"]),
        "sidecar_sha256": payload["sidecar_sha256"],
        "path": path.relative_to(root).as_posix(),
        "payload_file_sha256": sha256_file(path),
    }


def _descriptor_manifest_entry(
    payload: Mapping[str, Any], path: Path, root: Path
) -> dict[str, Any]:
    return {
        "sidecar": json_safe_mapping(payload["sidecar"]),
        "sidecar_sha256": payload["sidecar_sha256"],
        "path": path.relative_to(root).as_posix(),
        "payload_file_sha256": sha256_file(path),
    }


def _source_content_fingerprint(sidecar: Mapping[str, Any]) -> str:
    return sha256_json(
        {
            key: sidecar[key]
            for key in (
                "record_identity_sha256",
                "source_path_identity_sha256",
                "source_file_sha256",
                "source_loaded_array_sha256",
                "canonical_tensor_sha256",
                "source_support_tensor_sha256",
                "latent_tensor_sha256",
                "latent_support_mask_sha256",
                "packed_support_sha256",
            )
        }
    )


def _validate_manifest_against_run(
    manifest: Mapping[str, Any],
    prepared: _PreparedRun,
    code_provenance: Mapping[str, Any],
) -> None:
    if manifest.get("run_fingerprint") != prepared.run_fingerprint:
        raise ValueError("Existing factored bank is incompatible with exact resume/audit.")
    if manifest.get("resolved_config_sha256") != prepared.config_sha256:
        raise ValueError("Factored-bank resolved configuration changed.")
    if manifest.get("source_split") != prepared.source_split:
        raise ValueError("Factored-bank source-split identity changed.")
    if manifest.get("support_rule", {}).get("rule_sha256") != prepared.support_rule[
        "rule_sha256"
    ]:
        raise ValueError("Factored-bank encoder support-rule identity changed.")
    if manifest.get("storage_preflight", {}).get("report_sha256") != prepared.storage_report[
        "report_sha256"
    ]:
        raise ValueError("Factored-bank storage/source-shape preflight changed.")
    if manifest.get("computational_provenance") != json_safe_mapping(code_provenance):
        raise ValueError("Factored-bank computational provenance changed.")


def _verify_complete_published_artifact(root: Path, manifest: Mapping[str, Any]) -> None:
    for entry in manifest["records"]:
        path = safe_relative_path(root, entry["path"])
        if sha256_file(path) != entry["payload_file_sha256"]:
            raise ValueError("Published factored latent file identity changed.")
        _load_latent_record(path, expected_resume_key=_entry_sidecar(entry)["resume_key"])
    stats_path = safe_relative_path(root, manifest["latent_statistics"]["path"])
    if sha256_file(stats_path) != manifest["latent_statistics"]["file_sha256"]:
        raise ValueError("Published latent-statistics file identity changed.")
    _load_json(stats_path)
    descriptor_path = safe_relative_path(root, manifest["structural_descriptors"]["path"])
    if sha256_file(descriptor_path) != manifest["structural_descriptors"]["file_sha256"]:
        raise ValueError("Published descriptor-manifest file identity changed.")
    descriptor_manifest = _load_json(descriptor_path)
    for entry in descriptor_manifest["records"]:
        path = safe_relative_path(root, entry["path"])
        if sha256_file(path) != entry["payload_file_sha256"]:
            raise ValueError("Published structural-descriptor file identity changed.")
        _load_descriptor_record(path, expected_resume_key=_entry_sidecar(entry)["resume_key"])


def _require_factored_bank_roles(records: Sequence[Mapping[str, Any]]) -> None:
    identities: set[str] = set()
    expected = {
        Domain(field, contrast).label
        for contrast in CONTRASTS
        for field in FIELD_STRENGTHS_T
    }
    domains: dict[str, set[str]] = {split: set() for split in CANONICAL_VOLUME_SPLITS}
    for entry in records:
        sidecar = _entry_sidecar(entry)
        identity = classify_variant_a_cohort(
            case_identity=str(sidecar.get("record_identity", "")),
            metadata_prefix="R",
            supplied_cohort=str(sidecar.get("cohort", "")),
            subject_identity=sidecar.get("subject_identity"),
            allowed_cohorts=("R",),
        )
        if sidecar.get("subject_group_identity") != identity.subject_group_identity:
            raise ValueError("Factored-bank subject-group identity mismatch.")
        split = str(sidecar.get("split", ""))
        if split not in CANONICAL_VOLUME_SPLITS:
            raise ValueError("Factored-bank record has a forbidden split role.")
        if identity.case_identity in identities:
            raise ValueError("Factored-bank record identity is duplicated.")
        identities.add(identity.case_identity)
        domains[split].add(Domain.from_dict(dict(sidecar["domain"])).label)
        if sidecar.get("encoding_path", {}).get("path_used") != "full":
            raise ValueError("Factored-bank record did not use full encoding.")
    for split in CANONICAL_VOLUME_SPLITS:
        if domains[split] != expected:
            raise ValueError(f"Factored-bank split {split!r} does not cover all 15 domains.")


def _domain_counts(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    return {
        split: dict(
            sorted(
                Counter(
                    Domain.from_dict(dict(_entry_sidecar(entry)["domain"])).label
                    for entry in records
                    if _entry_sidecar(entry)["split"] == split
                ).items()
            )
        )
        for split in CANONICAL_VOLUME_SPLITS
    }


def _conv_support(support: torch.Tensor, module: nn.Module) -> torch.Tensor:
    spec = _conv_spec(module, "runtime")
    unsupported = (~support).to(torch.float32)
    propagated = F.max_pool3d(
        unsupported,
        kernel_size=tuple(spec["kernel_size"]),
        stride=tuple(spec["stride"]),
        padding=tuple(spec["padding"]),
        dilation=tuple(spec["dilation"]),
    )
    return propagated == 0


def _res_stack_support(support: torch.Tensor, stack: nn.Module) -> torch.Tensor:
    result = support
    for block in stack:
        main = _norm_support(result, block.norm1)
        main = _conv_support(main, block.conv1)
        main = _norm_support(main, block.norm2)
        main = _conv_support(main, block.conv2)
        skip = result if isinstance(block.skip, nn.Identity) else _conv_support(result, block.skip)
        result = main & skip
    return result


def _norm_support(support: torch.Tensor, module: nn.Module) -> torch.Tensor:
    if isinstance(module, nn.Identity):
        return support
    if isinstance(module, nn.GroupNorm):
        return torch.full_like(support, bool(support.all()))
    raise ValueError("Unsupported normalization in frozen encoder support propagation.")


def _conv_spec(module: nn.Module, name: str) -> dict[str, list[int]]:
    if not isinstance(module, nn.Conv3d):
        raise ValueError(f"Support graph expected Conv3d at {name}, got {type(module).__name__}.")
    return {
        "kernel_size": [int(value) for value in module.kernel_size],
        "stride": [int(value) for value in module.stride],
        "padding": [int(value) for value in module.padding],
        "dilation": [int(value) for value in module.dilation],
    }


def _validate_skip(module: nn.Module, name: str) -> None:
    if not isinstance(module, (nn.Identity, nn.Conv3d)):
        raise ValueError(f"Unsupported residual skip in encoder support graph: {name}.")
    if isinstance(module, nn.Conv3d):
        spec = _conv_spec(module, name)
        if spec != {
            "kernel_size": [1, 1, 1],
            "stride": [1, 1, 1],
            "padding": [0, 0, 0],
            "dilation": [1, 1, 1],
        }:
            raise ValueError("Encoder residual skip is outside the reviewed support graph.")


def _entry_sidecar(entry: Mapping[str, Any]) -> Mapping[str, Any]:
    sidecar = entry.get("sidecar")
    if not isinstance(sidecar, Mapping) or entry.get("sidecar_sha256") != sha256_json(sidecar):
        raise ValueError("Artifact manifest sidecar identity mismatch.")
    return sidecar


def _entry_identity(entry: Mapping[str, Any]) -> str:
    return str(_entry_sidecar(entry)["record_identity"])


def _latent_record_path(root: Path, split: str, identity: str) -> Path:
    return root / "latents" / split / f"{sha256_text(identity)}.pt"


def _descriptor_record_path(root: Path, identity: str) -> Path:
    return root / "descriptors" / "train" / f"{sha256_text(identity)}.pt"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not load JSON artifact {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON artifact {path} must contain an object.")
    result = json_safe_mapping(payload)
    stored = str(result.get("artifact_sha256", ""))
    unhashed = dict(result)
    unhashed.pop("artifact_sha256", None)
    if stored != sha256_json(unhashed):
        raise ValueError(f"JSON artifact {path} content hash mismatch.")
    return result


__all__ = [
    "DESCRIPTOR_INPUT",
    "FACTORED_LATENT_BANK_MANIFEST",
    "FACTORED_LATENT_STATS_FILE",
    "MASKED_WELFORD_RULE",
    "MIN_SUPPORTED_CHANNEL_VARIANCE",
    "MaskedChannelWelford",
    "PHOTOMETRY_FACTORED_DESCRIPTOR_QUALIFICATION_VERSION",
    "PHOTOMETRY_FACTORED_DESCRIPTOR_VERSION",
    "PHOTOMETRY_FACTORED_LATENT_BANK_VERSION",
    "PHOTOMETRY_FACTORED_LATENT_STATS_VERSION",
    "PhotometryFactoredLatentBankConfig",
    "STRUCTURAL_DESCRIPTOR_MANIFEST",
    "SUPPORT_PACKING_RULE",
    "SUPPORT_PROPAGATION_RULE",
    "audit_photometry_factored_latent_bank",
    "build_photometry_factored_latent_bank",
    "derive_encoder_support_rule",
    "load_photometry_factored_latent_bank_manifest",
    "pack_support_mask",
    "preflight_photometry_factored_latent_bank",
    "propagate_encoder_support",
    "receptive_field_source_bounds",
    "standardize_supported_latent",
    "structural_descriptor",
    "unpack_support_mask",
]
