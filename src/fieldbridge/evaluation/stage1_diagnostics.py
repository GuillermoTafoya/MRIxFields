"""Diagnostic-only contracts for an existing Stage-1 KL-VAE reconstruction run."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch
from torch import nn

from fieldbridge.data.contracts import RawBatch, VolumeRecord
from fieldbridge.data.manifests import Manifest
from fieldbridge.data.masks import clean_brain_mask
from fieldbridge.data.sources import nifti_image_loader
from fieldbridge.data.transforms import normalize_percentile_clip_to_unit_range
from fieldbridge.evaluation.metrics import mae, masked_mae, masked_mse, mse, nrmse, ssim3d
from fieldbridge.evaluation.stage1_report import _tiled_starts, sliding_window_reconstruct
from fieldbridge.models.autoencoders.kl_vae import KLVAEDecoder, KLVAEEncoder
from fieldbridge.training.checkpoints import load_checkpoint
from fieldbridge.training.losses import build_lpips_net, kl_divergence

_DATA_RANGE = 2.0
_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
_BANK_META_NAME = "bank_meta.json"
_BANK_INDEX_NAME = "bank_index.jsonl"
_LEGACY_LOSS_WEIGHTS = {"ssim": 1.0, "nrmse": 1.0, "lpips": 1.0, "kl": 1e-4}

DiagnosticLogger = Callable[[str], None]
ImageLoader = Callable[[Path, VolumeRecord], torch.Tensor]


@dataclass(frozen=True, slots=True)
class Stage1DiagnosticSpec:
    """Predeclared inference-only choices for the reconstruction diagnosis."""

    fixed_patch_index: int = 13
    fixed_volume_index: int = 0
    sampled_latent_seed: int = 13
    overlap_sweep: tuple[float, ...] = (0.25, 0.5, 0.75)
    reference_overlap: float = 0.5
    background_threshold_minus_one_one: float = -0.95
    mask_kernel_size: int = 3
    mask_closing_iterations: int = 1
    histogram_bins: int = 20
    lpips_num_slices: int = 8
    identity_tolerance: float = 1e-5
    direct_tiled_tolerance: float = 1e-5
    collapse_std_ratio_threshold: float = 0.5
    overlap_nrmse_span_threshold: float = 0.01
    seam_ratio_span_threshold: float = 0.1

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "Stage1DiagnosticSpec":
        section = config.get("diagnostic", config)
        if not isinstance(section, Mapping):
            raise ValueError("Diagnostic config must contain a mapping.")
        defaults = cls()
        spec = cls(
            fixed_patch_index=int(section.get("fixed_patch_index", defaults.fixed_patch_index)),
            fixed_volume_index=int(section.get("fixed_volume_index", defaults.fixed_volume_index)),
            sampled_latent_seed=int(
                section.get("sampled_latent_seed", defaults.sampled_latent_seed)
            ),
            overlap_sweep=tuple(
                float(value) for value in section.get("overlap_sweep", defaults.overlap_sweep)
            ),
            reference_overlap=float(
                section.get("reference_overlap", defaults.reference_overlap)
            ),
            background_threshold_minus_one_one=float(
                section.get(
                    "background_threshold_minus_one_one",
                    defaults.background_threshold_minus_one_one,
                )
            ),
            mask_kernel_size=int(section.get("mask_kernel_size", defaults.mask_kernel_size)),
            mask_closing_iterations=int(
                section.get("mask_closing_iterations", defaults.mask_closing_iterations)
            ),
            histogram_bins=int(section.get("histogram_bins", defaults.histogram_bins)),
            lpips_num_slices=int(
                section.get("lpips_num_slices", defaults.lpips_num_slices)
            ),
            identity_tolerance=float(
                section.get("identity_tolerance", defaults.identity_tolerance)
            ),
            direct_tiled_tolerance=float(
                section.get("direct_tiled_tolerance", defaults.direct_tiled_tolerance)
            ),
            collapse_std_ratio_threshold=float(
                section.get(
                    "collapse_std_ratio_threshold", defaults.collapse_std_ratio_threshold
                )
            ),
            overlap_nrmse_span_threshold=float(
                section.get(
                    "overlap_nrmse_span_threshold", defaults.overlap_nrmse_span_threshold
                )
            ),
            seam_ratio_span_threshold=float(
                section.get("seam_ratio_span_threshold", defaults.seam_ratio_span_threshold)
            ),
        )
        spec.validate()
        return spec

    def validate(self) -> None:
        if self.fixed_patch_index < 0 or self.fixed_volume_index < 0:
            raise ValueError("Fixed patch and volume indices must be non-negative.")
        if self.overlap_sweep != (0.25, 0.5, 0.75):
            raise ValueError("Stage-1 diagnostic overlap sweep must remain (0.25, 0.5, 0.75).")
        if self.reference_overlap not in self.overlap_sweep:
            raise ValueError("reference_overlap must be one of the declared overlap values.")
        validate_minus_one_one_background_threshold(
            self.background_threshold_minus_one_one
        )
        if self.mask_kernel_size <= 0 or self.mask_kernel_size % 2 == 0:
            raise ValueError("mask_kernel_size must be a positive odd integer.")
        if self.mask_closing_iterations < 1:
            raise ValueError("mask_closing_iterations must be positive.")
        if self.histogram_bins < 2:
            raise ValueError("histogram_bins must be at least 2.")
        if self.lpips_num_slices < 0:
            raise ValueError("lpips_num_slices cannot be negative.")
        if not 0.0 < self.collapse_std_ratio_threshold < 1.0:
            raise ValueError("collapse_std_ratio_threshold must be in (0, 1).")
        if self.overlap_nrmse_span_threshold <= 0.0:
            raise ValueError("overlap_nrmse_span_threshold must be positive.")
        if self.seam_ratio_span_threshold <= 0.0:
            raise ValueError("seam_ratio_span_threshold must be positive.")


@dataclass(frozen=True, slots=True)
class _PatchBankEntry:
    volume_index: int
    case_id: str
    shard: str
    domain: Mapping[str, Any]
    num_patches: int


@dataclass(frozen=True, slots=True)
class _PatchBankState:
    patch_size: tuple[int, int, int]
    patches_per_volume: int
    seed: int
    entries: tuple[_PatchBankEntry, ...]
    fingerprint_sha256: str


class _LegacySignedRangeDecoder(KLVAEDecoder):
    """Restore the unconditional Tanh head used by diagnostic-v1 checkpoints."""

    def decode(self, z: torch.Tensor, domain: Any) -> torch.Tensor:
        return torch.tanh(super().decode(z, domain))


def validate_minus_one_one_background_threshold(threshold: float) -> float:
    """Validate an anatomy/background threshold for tensors stored in ``[-1, 1]``."""

    value = float(threshold)
    if not -1.0 < value < 0.0:
        raise ValueError(
            "The Stage-1 background threshold must be strictly between -1 and 0 in "
            f"the configured [-1,1] model range; got {threshold}."
        )
    return value


def minus_one_one_foreground_mask(
    target: torch.Tensor,
    *,
    threshold: float = -0.95,
    kernel_size: int = 3,
    iterations: int = 1,
) -> torch.Tensor:
    """Build a 3D-capable support mask without treating model-space zero as background."""

    validate_minus_one_one_background_threshold(threshold)
    _validate_minus_one_one_tensor(target, "target")
    return clean_brain_mask(
        target,
        threshold=threshold,
        kernel_size=kernel_size,
        iterations=iterations,
    )


@torch.inference_mode()
def identity_tiler_contract(
    *,
    overlaps: Sequence[float] = (0.25, 0.5, 0.75),
    tolerance: float = 1e-5,
) -> dict[str, Any]:
    """Prove that Hann blending itself is an identity partition of unity."""

    class _IdentityEncoder:
        def encode_dist(
            self, x: torch.Tensor, domain: object
        ) -> tuple[torch.Tensor, torch.Tensor]:
            del domain
            return x, torch.zeros_like(x)

    class _IdentityDecoder:
        def decode(self, z: torch.Tensor, domain: object) -> torch.Tensor:
            del domain
            return z

    values = torch.linspace(-1.0, 1.0, 1 * 1 * 17 * 19 * 21).reshape(1, 1, 17, 19, 21)
    by_overlap: dict[str, Any] = {}
    passed = True
    for overlap in overlaps:
        reconstructed = sliding_window_reconstruct(
            _IdentityEncoder(),  # type: ignore[arg-type]
            _IdentityDecoder(),  # type: ignore[arg-type]
            values,
            patch_size=(8, 8, 8),
            domain=None,
            overlap=float(overlap),
            clamp_output=False,
        )
        difference = torch.abs(reconstructed - values)
        maximum = float(difference.max())
        mean_value = float(difference.mean())
        overlap_passed = maximum <= tolerance
        passed = passed and overlap_passed
        by_overlap[_overlap_key(overlap)] = {
            "max_abs_error": maximum,
            "mean_abs_error": mean_value,
            "passed": overlap_passed,
        }
    return {"passed": passed, "tolerance": tolerance, "by_overlap": by_overlap}


def seam_gradient_metric(
    volume: torch.Tensor,
    *,
    patch_size: Sequence[int],
    overlap: float,
) -> dict[str, float | int | None]:
    """Measure gradient excess at the declared sliding-window tile faces."""

    if volume.ndim != 5:
        raise ValueError("seam_gradient_metric expects a (B,C,X,Y,Z) tensor.")
    patch = tuple(int(value) for value in patch_size)
    if len(patch) != 3:
        raise ValueError("patch_size must have three spatial dimensions.")
    boundary_sum = 0.0
    boundary_count = 0
    reference_sum = 0.0
    reference_count = 0
    for spatial_offset, (dimension, patch_dim) in enumerate(
        zip(volume.shape[-3:], patch), start=2
    ):
        stride = max(1, round(patch_dim * (1.0 - overlap)))
        starts = _tiled_starts(int(dimension), patch_dim, stride)
        boundary_indices = {
            boundary - 1
            for start in starts
            for boundary in (start, start + patch_dim)
            if 0 < boundary < int(dimension)
        }
        gradient = volume.diff(dim=spatial_offset).abs()
        if boundary_indices:
            index = torch.tensor(sorted(boundary_indices), device=volume.device)
            boundary_values = gradient.index_select(spatial_offset, index)
            boundary_sum += float(boundary_values.sum())
            boundary_count += boundary_values.numel()
        reference_sum += float(gradient.sum())
        reference_count += gradient.numel()
    overall_mean = reference_sum / max(reference_count, 1)
    boundary_mean = boundary_sum / boundary_count if boundary_count else None
    ratio = None
    excess = None
    if boundary_mean is not None:
        ratio = boundary_mean / max(overall_mean, torch.finfo(torch.float32).eps)
        excess = boundary_mean - overall_mean
    return {
        "boundary_gradient_mean_abs": boundary_mean,
        "overall_gradient_mean_abs": overall_mean,
        "boundary_to_overall_ratio": ratio,
        "boundary_gradient_excess": excess,
        "boundary_value_count": boundary_count,
    }


@torch.inference_mode()
def run_stage1_reconstruction_diagnostics(
    *,
    checkpoint_path: str | Path,
    patch_bank_dir: str | Path,
    manifest: Manifest,
    resolved_config: Mapping[str, Any],
    diagnostic_spec: Stage1DiagnosticSpec | Mapping[str, Any],
    checkpoint_sweep_paths: Sequence[str | Path] = (),
    image_loader: ImageLoader = nifti_image_loader,
    device: torch.device | None = None,
    logger: DiagnosticLogger | None = None,
) -> dict[str, Any]:
    """Diagnose one existing Stage-1 run without training or checkpoint selection."""

    spec = (
        diagnostic_spec
        if isinstance(diagnostic_spec, Stage1DiagnosticSpec)
        else Stage1DiagnosticSpec.from_mapping(diagnostic_spec)
    )
    spec.validate()
    log = logger or (lambda message: None)
    selected_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_state, checkpoint_report = _load_checkpoint_contract(checkpoint_path)
    _validate_checkpoint_config(checkpoint_report, resolved_config)
    encoder, decoder = _models_from_config(resolved_config)
    encoder.load_state_dict(checkpoint_state["encoder"], strict=True)
    decoder.load_state_dict(checkpoint_state["decoder"], strict=True)
    encoder = encoder.to(selected_device).eval()
    decoder = decoder.to(selected_device).eval()

    bank = _load_patch_bank_state(patch_bank_dir)
    bank_report = _validate_patch_bank(
        bank,
        bank_dir=Path(patch_bank_dir),
        manifest=manifest,
        resolved_config=resolved_config,
    )
    fixed_patch = _load_patch(
        Path(patch_bank_dir), bank, global_index=spec.fixed_patch_index
    )
    patch_image = fixed_patch.image.unsqueeze(0).to(selected_device)
    patch_mask = minus_one_one_foreground_mask(
        patch_image,
        threshold=spec.background_threshold_minus_one_one,
        kernel_size=spec.mask_kernel_size,
        iterations=spec.mask_closing_iterations,
    )
    lpips_net, lpips_status = _optional_lpips_net(selected_device, spec.lpips_num_slices)

    log("stage1 diagnostic: fixed patch direct mean/sample reconstruction")
    patch_report, mean_reconstruction = _diagnose_fixed_patch(
        encoder,
        decoder,
        fixed_patch,
        patch_image,
        patch_mask,
        spec,
        resolved_config,
        lpips_net=lpips_net,
        lpips_status=lpips_status,
    )
    tiled_patch = sliding_window_reconstruct(
        encoder,
        decoder,
        patch_image,
        patch_size=bank.patch_size,
        domain=fixed_patch.source_domain,
        overlap=spec.reference_overlap,
        clamp_output=False,
    )
    direct_tiled_difference = torch.abs(tiled_patch - mean_reconstruction)
    direct_tiled_report = {
        "reference_overlap": spec.reference_overlap,
        "mean_abs_difference": float(direct_tiled_difference.mean()),
        "max_abs_difference": float(direct_tiled_difference.max()),
        "tolerance": spec.direct_tiled_tolerance,
        "passed": float(direct_tiled_difference.max()) <= spec.direct_tiled_tolerance,
    }
    del tiled_patch, mean_reconstruction

    if spec.fixed_volume_index >= len(manifest.records):
        raise IndexError(
            f"fixed_volume_index {spec.fixed_volume_index} exceeds manifest size "
            f"{len(manifest.records)}."
        )
    volume_record = manifest.records[spec.fixed_volume_index]
    log("stage1 diagnostic: loading fixed full volume")
    raw_volume = image_loader(volume_record.image_path, volume_record)
    normalized_volume = normalize_percentile_clip_to_unit_range(raw_volume).unsqueeze(0)
    _validate_minus_one_one_tensor(normalized_volume, "normalized full volume")
    volume_image = normalized_volume.to(selected_device)
    volume_mask = minus_one_one_foreground_mask(
        volume_image,
        threshold=spec.background_threshold_minus_one_one,
        kernel_size=spec.mask_kernel_size,
        iterations=spec.mask_closing_iterations,
    )

    overlap_reports: dict[str, Any] = {}
    for overlap in spec.overlap_sweep:
        log(f"stage1 diagnostic: full-volume overlap={overlap:.2f}")
        reconstruction = sliding_window_reconstruct(
            encoder,
            decoder,
            volume_image,
            patch_size=bank.patch_size,
            domain=volume_record.domain,
            overlap=overlap,
            clamp_output=False,
        )
        overlap_reports[_overlap_key(overlap)] = {
            "overlap": overlap,
            "official_full_volume_metrics": _official_metrics(
                reconstruction,
                volume_image,
                lpips_net=lpips_net,
                lpips_status=lpips_status,
                lpips_num_slices=spec.lpips_num_slices,
            ),
            "masked_diagnostics": _masked_errors(reconstruction, volume_image, volume_mask),
            "seam_metric": seam_gradient_metric(
                reconstruction,
                patch_size=bank.patch_size,
                overlap=overlap,
            ),
            "reconstruction_distribution": tensor_distribution_summary(
                reconstruction, bins=spec.histogram_bins
            ),
        }
        del reconstruction
        if selected_device.type == "cuda":
            torch.cuda.empty_cache()

    checkpoint_sweep = _run_checkpoint_sweep(
        checkpoint_sweep_paths,
        patch_image=patch_image,
        patch_mask=patch_mask,
        patch_domain=fixed_patch.source_domain,
        resolved_config=resolved_config,
        spec=spec,
        lpips_net=lpips_net,
        lpips_status=lpips_status,
        device=selected_device,
    )
    identity_report = identity_tiler_contract(
        overlaps=spec.overlap_sweep,
        tolerance=spec.identity_tolerance,
    )
    coverage = _manifest_coverage(manifest)
    provenance_fingerprint = _canonical_sha256(
        {
            "checkpoint_git_commit": checkpoint_report["git_commit"],
            "checkpoint_step": checkpoint_report["step"],
            "checkpoint_config_sha256": checkpoint_report["config_sha256"],
            "patch_bank_sha256": bank.fingerprint_sha256,
            "manifest_sha256": _manifest_fingerprint(manifest),
            "resolved_config_sha256": _canonical_sha256(_json_safe(resolved_config)),
        }
    )

    report: dict[str, Any] = {
        "diagnostic_contract_version": 1,
        "evidence_scope": "stage1_reconstruction_engineering_diagnostic",
        "held_out": False,
        "confirmatory": False,
        "training_performed": False,
        "stage2_started": False,
        "checkpoint": checkpoint_report,
        "resolved_config": _resolved_config_report(resolved_config),
        "patch_bank": bank_report,
        "manifest": {
            "audit_ok": True,
            "volumes": len(manifest.records),
            "coverage_by_field_contrast": coverage,
            "fingerprint_sha256": _manifest_fingerprint(manifest),
        },
        "provenance_fingerprint_sha256": provenance_fingerprint,
        "mask_contract": {
            "model_range": "minus_one_one",
            "background_threshold": spec.background_threshold_minus_one_one,
            "foreground_rule": "target_greater_than_threshold_then_3d_binary_closing",
            "kernel_size": spec.mask_kernel_size,
            "iterations": spec.mask_closing_iterations,
        },
        "identity_tiler_contract": identity_report,
        "fixed_patch": patch_report,
        "direct_vs_tiled_fixed_patch": direct_tiled_report,
        "fixed_full_volume": {
            "selection": {
                "policy": "fixed_manifest_index",
                "index": spec.fixed_volume_index,
                "field_contrast": volume_record.domain.label,
                "split_name": volume_record.split,
            },
            "complete_volume": True,
            "target_distribution": tensor_distribution_summary(
                volume_image, bins=spec.histogram_bins
            ),
            "overlap_sweep": overlap_reports,
        },
        "checkpoint_step_sweep": {
            "selection_policy": "all_user_supplied_steps_reported_chronologically",
            "best_checkpoint_selected": False,
            "results": checkpoint_sweep,
        },
        "limitations": [
            "same manifest was used for training and diagnostic evaluation",
            "fixed development samples are not held out",
            "diagnostic overlap sweep is not checkpoint or hyperparameter selection",
            "legacy unmasked project proxies are not published Task-3 metric parity",
        ],
    }
    report["recommendation"] = _result_dependent_recommendation(report, spec)
    _assert_sanitized_report(report)
    return report


def tensor_distribution_summary(tensor: torch.Tensor, *, bins: int = 20) -> dict[str, Any]:
    """Return bounded numeric range and histogram summaries without retaining voxels."""

    _validate_minus_one_one_tensor(tensor, "distribution tensor")
    detached = tensor.detach().float()
    histogram = torch.histc(detached, bins=bins, min=-1.0, max=1.0)
    total = histogram.sum().clamp_min(1.0)
    edges = torch.linspace(-1.0, 1.0, bins + 1)
    return {
        "min": float(detached.min()),
        "max": float(detached.max()),
        "mean": float(detached.mean()),
        "std": float(detached.std(unbiased=False)),
        "histogram": {
            "bin_edges": [float(value) for value in edges],
            "fractions": [float(value) for value in histogram / total],
        },
    }


def _diagnose_fixed_patch(
    encoder: KLVAEEncoder,
    decoder: KLVAEDecoder,
    patch: RawBatch,
    image: torch.Tensor,
    mask: torch.Tensor,
    spec: Stage1DiagnosticSpec,
    resolved_config: Mapping[str, Any],
    *,
    lpips_net: nn.Module | None,
    lpips_status: str,
) -> tuple[dict[str, Any], torch.Tensor]:
    with torch.no_grad():
        mean, logvar = encoder.encode_dist(image, patch.source_domain)
        generator = torch.Generator(device=image.device).manual_seed(spec.sampled_latent_seed)
        epsilon = torch.randn(
            mean.shape,
            generator=generator,
            device=mean.device,
            dtype=mean.dtype,
        )
        sampled = mean + epsilon * torch.exp(0.5 * logvar)
        reconstructed_mean = decoder.decode(mean, patch.source_domain)
        reconstructed_sample = decoder.decode(sampled, patch.source_domain)
        kl_value = float(kl_divergence(mean, logvar))

    mean_metrics = _official_metrics(
        reconstructed_mean,
        image,
        lpips_net=lpips_net,
        lpips_status=lpips_status,
        lpips_num_slices=spec.lpips_num_slices,
    )
    sampled_metrics = _official_metrics(
        reconstructed_sample,
        image,
        lpips_net=lpips_net,
        lpips_status=lpips_status,
        lpips_num_slices=spec.lpips_num_slices,
    )
    loss_components = {
        "nrmse": sampled_metrics["nrmse"],
        "ssim3d": sampled_metrics["ssim3d"],
        "ssim_loss": 1.0 - float(sampled_metrics["ssim3d"]),
        "lpips": sampled_metrics["lpips"],
        "kl": kl_value,
        "weights": dict(_legacy_training_config(resolved_config)["loss_weights"]),
    }
    report = {
        "selection": {
            "policy": "fixed_global_patch_index",
            "index": spec.fixed_patch_index,
            "field_contrast": _domain_label(patch.source_domain),
        },
        "foreground_occupancy": float(mask.mean()),
        "foreground_voxels": int(mask.sum()),
        "total_voxels": mask.numel(),
        "target_distribution": tensor_distribution_summary(image, bins=spec.histogram_bins),
        "latent_distribution": {
            "mean_min": float(mean.min()),
            "mean_max": float(mean.max()),
            "mean_mean": float(mean.mean()),
            "logvar_min": float(logvar.min()),
            "logvar_max": float(logvar.max()),
            "logvar_mean": float(logvar.mean()),
        },
        "reconstruction_from_latent_mean": {
            "official_metrics": mean_metrics,
            "masked_diagnostics": _masked_errors(reconstructed_mean, image, mask),
            "distribution": tensor_distribution_summary(
                reconstructed_mean, bins=spec.histogram_bins
            ),
        },
        "reconstruction_from_sampled_latent": {
            "sample_seed": spec.sampled_latent_seed,
            "official_metrics": sampled_metrics,
            "masked_diagnostics": _masked_errors(reconstructed_sample, image, mask),
            "distribution": tensor_distribution_summary(
                reconstructed_sample, bins=spec.histogram_bins
            ),
        },
        "training_loss_components_on_sampled_reconstruction": loss_components,
    }
    return report, reconstructed_mean


def _official_metrics(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    *,
    lpips_net: nn.Module | None,
    lpips_status: str,
    lpips_num_slices: int,
) -> dict[str, Any]:
    """Legacy diagnostic-v1 project proxies.

    The function and serialized ``official_*`` keys predate publication of the
    MRIxFields2026 evaluator and remain for diagnostic-v1 schema compatibility. These
    range-normalized RMSE, Torch SSIM3D, and VGG slice-LPIPS values are not official
    Task-3 metric implementations.
    """

    lpips_payload: dict[str, Any]
    if lpips_net is None or lpips_num_slices <= 0:
        lpips_payload = {"status": lpips_status, "value": None}
    else:
        lpips_payload = {
            "status": "computed",
            "value": float(
                _legacy_signed_lpips_3d(
                    reconstruction,
                    target,
                    net=lpips_net,
                    num_slices=lpips_num_slices,
                )
            ),
        }
    return {
        "metric_contract": "stage1-diagnostic-v1-project-proxies",
        "data_range": _DATA_RANGE,
        "nrmse": float(nrmse(reconstruction, target, data_range=_DATA_RANGE)),
        "ssim3d": float(ssim3d(reconstruction, target, data_range=_DATA_RANGE)),
        "lpips": lpips_payload,
        "mae": float(mae(reconstruction, target)),
        "mse": float(mse(reconstruction, target)),
    }


def _legacy_signed_lpips_3d(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    net: nn.Module,
    num_slices: int,
    axis: int = 2,
) -> torch.Tensor:
    """Evaluate diagnostic-v1 LPIPS on tensors that are already in ``[-1, 1]``."""

    if prediction.ndim != 5 or prediction.shape != target.shape:
        raise ValueError("Diagnostic-v1 LPIPS expects same-shaped 5D tensors.")
    depth = int(prediction.shape[axis])
    count = min(num_slices, depth)
    indices = (
        torch.linspace(0, depth - 1, count, device=prediction.device)
        .round()
        .long()
        .unique()
    )
    prediction_2d = prediction.index_select(axis, indices).movedim(axis, 1).flatten(0, 1)
    target_2d = target.index_select(axis, indices).movedim(axis, 1).flatten(0, 1)
    return net(
        _diagnostic_three_channel(prediction_2d),
        _diagnostic_three_channel(target_2d),
    ).mean()


def _diagnostic_three_channel(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.shape[1] == 3:
        return tensor
    if tensor.shape[1] == 1:
        return tensor.repeat(1, 3, 1, 1)
    raise ValueError(f"Diagnostic-v1 LPIPS expects 1 or 3 channels, got {tensor.shape[1]}.")


def _masked_errors(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    foreground_mask: torch.Tensor,
) -> dict[str, Any]:
    outside = 1.0 - foreground_mask
    foreground_voxels = int(foreground_mask.sum())
    outside_voxels = int(outside.sum())
    return {
        "foreground_voxels": foreground_voxels,
        "outside_voxels": outside_voxels,
        "foreground_mae": _optional_masked_metric(
            masked_mae, reconstruction, target, foreground_mask, foreground_voxels
        ),
        "foreground_mse": _optional_masked_metric(
            masked_mse, reconstruction, target, foreground_mask, foreground_voxels
        ),
        "outside_mae": _optional_masked_metric(
            masked_mae, reconstruction, target, outside, outside_voxels
        ),
        "outside_mse": _optional_masked_metric(
            masked_mse, reconstruction, target, outside, outside_voxels
        ),
        "outside_target_mean": _optional_region_mean(target, outside, outside_voxels),
        "outside_reconstruction_mean": _optional_region_mean(
            reconstruction, outside, outside_voxels
        ),
    }


def _optional_masked_metric(
    metric: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    selected: int,
) -> float | None:
    if selected == 0:
        return None
    return float(metric(reconstruction, target, mask))


def _optional_region_mean(
    tensor: torch.Tensor, mask: torch.Tensor, selected: int
) -> float | None:
    if selected == 0:
        return None
    return float((tensor * mask).sum() / mask.sum())


def _load_checkpoint_contract(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    state = load_checkpoint(path, map_location="cpu")
    required = ("encoder", "decoder", "optimizer", "step", "_meta")
    missing = [key for key in required if key not in state]
    if missing:
        raise ValueError(f"Stage-1 checkpoint is missing required keys: {missing}.")
    metadata = state["_meta"]
    if not isinstance(metadata, Mapping):
        raise ValueError("Stage-1 checkpoint _meta must be a mapping.")
    git_commit = str(metadata.get("git_commit", ""))
    if _SHA_PATTERN.fullmatch(git_commit) is None:
        raise ValueError("Stage-1 checkpoint does not record an exact 40-character git SHA.")
    recorded_config = metadata.get("config")
    if not isinstance(recorded_config, Mapping):
        raise ValueError("Stage-1 checkpoint does not record its training config.")
    step = int(state["step"])
    if step <= 0:
        raise ValueError(f"Stage-1 checkpoint step must be positive, got {step}.")
    early_stop = state.get("early_stop")
    if early_stop is not None and not isinstance(early_stop, Mapping):
        raise ValueError("Stage-1 checkpoint early_stop state must be a mapping when present.")
    checkpoint_version = state.get("checkpoint_version", metadata.get("checkpoint_version"))
    report = {
        "git_commit": git_commit,
        "step": step,
        "seed": metadata.get("seed"),
        "checkpoint_version": checkpoint_version,
        "checkpoint_schema": (
            "stage1_vae_versioned" if checkpoint_version is not None else "stage1_vae_unversioned"
        ),
        "config": _sanitize_config(recorded_config),
        "config_sha256": _canonical_sha256(_json_safe(recorded_config)),
        "early_stop": None if early_stop is None else _json_safe(early_stop),
    }
    return state, report


def _validate_checkpoint_config(
    checkpoint_report: Mapping[str, Any], resolved_config: Mapping[str, Any]
) -> None:
    recorded = dict(checkpoint_report["config"])
    expected = _legacy_training_config(resolved_config)
    mismatches = [
        key
        for key in sorted(set(recorded) | set(expected))
        if recorded.get(key) != expected.get(key)
    ]
    if mismatches:
        raise ValueError(
            "Resolved config is incompatible with checkpoint-recorded training config "
            f"for keys: {sorted(mismatches)}."
        )


def _models_from_config(
    config: Mapping[str, Any],
) -> tuple[KLVAEEncoder, KLVAEDecoder]:
    model = config.get("model")
    if not isinstance(model, Mapping) or str(model.get("name")) != "kl_vae":
        raise ValueError("Resolved config must declare model.name=kl_vae.")
    shared_keys = (
        "base_channels",
        "latent_channels",
        "spatial_dims",
        "activation",
        "use_norm",
        "num_res_blocks",
    )
    shared = {key: model[key] for key in shared_keys if key in model}
    encoder_kwargs = dict(shared)
    decoder_kwargs = dict(shared)
    if "in_channels" in model:
        encoder_kwargs["in_channels"] = model["in_channels"]
    if "out_channels" in model:
        decoder_kwargs["out_channels"] = model["out_channels"]
    return KLVAEEncoder(**encoder_kwargs), _LegacySignedRangeDecoder(**decoder_kwargs)


def _load_patch_bank_state(bank_dir: str | Path) -> _PatchBankState:
    root = Path(bank_dir)
    meta = json.loads((root / _BANK_META_NAME).read_text(encoding="utf-8"))
    patch_size = tuple(int(value) for value in meta["patch_size"])
    if len(patch_size) != 3:
        raise ValueError("Stage-1 patch bank must contain 3D patches.")
    patches_per_volume = int(meta["patches_per_volume"])
    seed = int(meta["seed"])
    entries: list[_PatchBankEntry] = []
    for line_number, line in enumerate(
        (root / _BANK_INDEX_NAME).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        raw = json.loads(line)
        shard = str(raw["shard"])
        shard_path = PurePosixPath(shard)
        if shard_path.is_absolute() or ".." in shard_path.parts:
            raise ValueError(f"Patch-bank index line {line_number} has unsafe shard path.")
        entry = _PatchBankEntry(
            volume_index=int(raw["vol_index"]),
            case_id=str(raw["case_id"]),
            shard=shard,
            domain=dict(raw["domain"]),
            num_patches=int(raw["num_patches"]),
        )
        if entry.num_patches != patches_per_volume:
            raise ValueError(
                f"Patch-bank entry {line_number} has {entry.num_patches} patches, "
                f"expected {patches_per_volume}."
            )
        entries.append(entry)
    if not entries:
        raise ValueError("Patch-bank index is empty.")
    if len({entry.volume_index for entry in entries}) != len(entries):
        raise ValueError("Patch-bank index contains duplicate volume indices.")
    canonical = {
        "meta": {
            "patch_size": patch_size,
            "patches_per_volume": patches_per_volume,
            "seed": seed,
        },
        "entries": [
            {
                "volume_index": entry.volume_index,
                "case_id": entry.case_id,
                "shard": entry.shard,
                "domain": entry.domain,
                "num_patches": entry.num_patches,
            }
            for entry in entries
        ],
    }
    return _PatchBankState(
        patch_size=patch_size,  # type: ignore[arg-type]
        patches_per_volume=patches_per_volume,
        seed=seed,
        entries=tuple(entries),
        fingerprint_sha256=_canonical_sha256(canonical),
    )


def _validate_patch_bank(
    bank: _PatchBankState,
    *,
    bank_dir: Path,
    manifest: Manifest,
    resolved_config: Mapping[str, Any],
) -> dict[str, Any]:
    data = resolved_config.get("data")
    model = resolved_config.get("model")
    if not isinstance(data, Mapping) or not isinstance(model, Mapping):
        raise ValueError("Resolved config requires data and model mappings.")
    configured_patch = tuple(int(value) for value in data.get("patch_size", ()))
    configured_ppv = int(data.get("patches_per_volume", -1))
    configured_seed = int(resolved_config.get("seed", -1))
    checks = {
        "patch_size_matches": configured_patch == bank.patch_size,
        "patches_per_volume_matches": configured_ppv == bank.patches_per_volume,
        "seed_matches": configured_seed == bank.seed,
        "spatial_dims_is_3": int(model.get("spatial_dims", -1)) == 3,
        "volume_count_matches_manifest": len(bank.entries) == len(manifest.records),
    }
    if not all(checks.values()):
        failed = sorted(key for key, passed in checks.items() if not passed)
        raise ValueError(f"Patch-bank/config/manifest compatibility failed: {failed}.")

    ordered_entries = sorted(bank.entries, key=lambda entry: entry.volume_index)
    if [entry.volume_index for entry in ordered_entries] != list(range(len(ordered_entries))):
        raise ValueError("Patch-bank volume indices are not contiguous from zero.")
    domains_match = all(
        _entry_domain_label(entry) == record.domain.label
        for entry, record in zip(ordered_entries, manifest.records)
    )
    exact_sample_ids = all(
        entry.case_id == record.case_id
        for entry, record in zip(ordered_entries, manifest.records)
    )
    legacy_subject_ids = all(
        record.subject_id is not None and entry.case_id == record.subject_id
        for entry, record in zip(ordered_entries, manifest.records)
    )
    if not domains_match:
        raise ValueError("Patch-bank domain order does not match the adapted manifest.")
    if exact_sample_ids:
        identity_alignment = "official_sample_id"
    elif legacy_subject_ids:
        identity_alignment = "legacy_subject_id_order_only"
    else:
        raise ValueError(
            "Patch-bank identities do not align with official sample or subject order."
        )

    for entry in ordered_entries:
        if not (bank_dir / entry.shard).is_file():
            raise FileNotFoundError("Patch-bank index references a missing shard.")
    case_counts = Counter(entry.case_id for entry in ordered_entries)
    bank_coverage = Counter(_entry_domain_label(entry) for entry in ordered_entries)
    return {
        "compatibility_ok": True,
        "checks": checks,
        "patch_size": list(bank.patch_size),
        "patches_per_volume": bank.patches_per_volume,
        "seed": bank.seed,
        "volumes": len(bank.entries),
        "patches": len(bank.entries) * bank.patches_per_volume,
        "identity_alignment": identity_alignment,
        "case_ids_unique": all(count == 1 for count in case_counts.values()),
        "duplicate_case_id_values": sum(count > 1 for count in case_counts.values()),
        "coverage_by_field_contrast": dict(sorted(bank_coverage.items())),
        "fingerprint_sha256": bank.fingerprint_sha256,
    }


def _load_patch(bank_dir: Path, bank: _PatchBankState, *, global_index: int) -> RawBatch:
    total = len(bank.entries) * bank.patches_per_volume
    if not 0 <= global_index < total:
        raise IndexError(f"fixed_patch_index {global_index} is outside [0, {total}).")
    volume_index, local_index = divmod(global_index, bank.patches_per_volume)
    entry = sorted(bank.entries, key=lambda value: value.volume_index)[volume_index]
    shard = np.load(bank_dir / entry.shard, mmap_mode="r")
    expected_shape = (bank.patches_per_volume, 1, *bank.patch_size)
    if tuple(int(value) for value in shard.shape) != expected_shape:
        raise ValueError(
            f"Fixed patch shard shape {tuple(shard.shape)} does not match {expected_shape}."
        )
    patch = torch.from_numpy(np.array(shard[local_index], copy=True)).float()
    if not torch.isfinite(patch).all():
        raise ValueError("Fixed patch contains non-finite values.")
    _validate_minus_one_one_tensor(patch, "fixed patch")
    domain_mapping = dict(entry.domain)
    from fieldbridge.data.domains import Domain

    domain = Domain.from_dict(domain_mapping)
    return RawBatch(
        image=patch,
        source_domain=domain,
        target_domain=domain,
        metadata={"global_patch_index": global_index, "volume_index": volume_index},
    )


def _run_checkpoint_sweep(
    paths: Sequence[str | Path],
    *,
    patch_image: torch.Tensor,
    patch_mask: torch.Tensor,
    patch_domain: Any,
    resolved_config: Mapping[str, Any],
    spec: Stage1DiagnosticSpec,
    lpips_net: nn.Module | None,
    lpips_status: str,
    device: torch.device,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for path in paths:
        state, checkpoint = _load_checkpoint_contract(path)
        _validate_checkpoint_config(checkpoint, resolved_config)
        encoder, decoder = _models_from_config(resolved_config)
        encoder.load_state_dict(state["encoder"], strict=True)
        decoder.load_state_dict(state["decoder"], strict=True)
        encoder = encoder.to(device).eval()
        decoder = decoder.to(device).eval()
        with torch.no_grad():
            mean, logvar = encoder.encode_dist(patch_image, patch_domain)
            reconstruction = decoder.decode(mean, patch_domain)
        results.append(
            {
                "step": checkpoint["step"],
                "git_commit": checkpoint["git_commit"],
                "checkpoint_version": checkpoint["checkpoint_version"],
                "latent_mean_reconstruction": {
                    "official_metrics": _official_metrics(
                        reconstruction,
                        patch_image,
                        lpips_net=lpips_net,
                        lpips_status=lpips_status,
                        lpips_num_slices=spec.lpips_num_slices,
                    ),
                    "masked_diagnostics": _masked_errors(
                        reconstruction, patch_image, patch_mask
                    ),
                    "kl": float(kl_divergence(mean, logvar)),
                },
            }
        )
    return sorted(results, key=lambda item: int(item["step"]))


def _optional_lpips_net(
    device: torch.device, num_slices: int
) -> tuple[nn.Module | None, str]:
    if num_slices <= 0:
        return None, "skipped_disabled"
    try:
        return build_lpips_net(device), "computed"
    except ImportError:
        return None, "skipped_optional_dependency_unavailable"


def _manifest_coverage(manifest: Manifest) -> dict[str, Any]:
    coverage: dict[str, Counter[str]] = {}
    for record in manifest.records:
        key = record.domain.label
        split_counts = coverage.setdefault(key, Counter())
        split_counts[record.split or "unspecified"] += 1
    return {
        key: {
            "volumes": sum(split_counts.values()),
            "splits": dict(sorted(split_counts.items())),
        }
        for key, split_counts in sorted(coverage.items())
    }


def _manifest_fingerprint(manifest: Manifest) -> str:
    return _canonical_sha256(
        [
            {
                "case_id": record.case_id,
                "subject_id": record.subject_id,
                "split": record.split,
                "image_path": str(record.image_path),
                "domain": record.domain.to_dict(),
            }
            for record in manifest.records
        ]
    )


def _resolved_config_report(config: Mapping[str, Any]) -> dict[str, Any]:
    data = config.get("data", {})
    model = config.get("model", {})
    if not isinstance(data, Mapping) or not isinstance(model, Mapping):
        raise ValueError("Resolved config requires data and model mappings.")
    return {
        "fingerprint_sha256": _canonical_sha256(_json_safe(config)),
        "seed": int(config.get("seed", -1)),
        "data": {
            "patch_size": list(data.get("patch_size", ())),
            "patches_per_volume": int(data.get("patches_per_volume", -1)),
            "normalization": "percentile_clip_to_minus_one_one",
        },
        "model": _json_safe(model),
        "training": _legacy_training_config(config),
    }


def _legacy_training_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Project a full run YAML onto the checkpoint-recorded diagnostic-v1 contract."""

    raw_training = config.get("training", {})
    if not isinstance(raw_training, Mapping):
        raise ValueError("Resolved config training section must be a mapping.")
    training = dict(raw_training)
    weights = dict(_LEGACY_LOSS_WEIGHTS)
    raw_weights = training.get("loss_weights", config.get("loss_weights", {}))
    if not isinstance(raw_weights, Mapping):
        raise ValueError("Resolved config loss_weights must be a mapping.")
    weights.update({str(key): float(value) for key, value in raw_weights.items()})
    return {
        "steps": int(training.get("steps", config.get("steps", 2))),
        "batch_size": int(training.get("batch_size", config.get("batch_size", 2))),
        "seed": int(config.get("seed", training.get("seed", 13))),
        "lr": float(training.get("lr", config.get("lr", 1e-4))),
        "device": training.get("device", config.get("device", "auto")),
        "precision": training.get("precision", config.get("precision", "fp32")),
        "loss_weights": weights,
        "ssim_window_size": int(
            training.get("ssim_window_size", config.get("ssim_window_size", 7))
        ),
        "lpips_num_slices": int(
            training.get("lpips_num_slices", config.get("lpips_num_slices", 8))
        ),
        "grad_clip_norm": float(
            training.get("grad_clip_norm", config.get("grad_clip_norm", 1.0))
        ),
        "steps_per_epoch": int(
            training.get("steps_per_epoch", config.get("steps_per_epoch", 0))
        ),
        "early_stopping": bool(
            training.get("early_stopping", config.get("early_stopping", False))
        ),
        "early_stopping_patience": int(
            training.get(
                "early_stopping_patience", config.get("early_stopping_patience", 5)
            )
        ),
        "early_stopping_min_delta": float(
            training.get(
                "early_stopping_min_delta", config.get("early_stopping_min_delta", 0.005)
            )
        ),
        "early_stopping_ema_decay": float(
            training.get(
                "early_stopping_ema_decay", config.get("early_stopping_ema_decay", 0.98)
            )
        ),
        "checkpoint_at_end": bool(
            training.get("checkpoint_at_end", config.get("checkpoint_at_end", False))
        ),
        "log_every_steps": int(
            training.get("log_every_steps", config.get("log_every_steps", 0))
        ),
    }


def _result_dependent_recommendation(
    report: Mapping[str, Any], spec: Stage1DiagnosticSpec
) -> dict[str, Any]:
    identity_passed = bool(report["identity_tiler_contract"]["passed"])
    direct_tiled_passed = bool(report["direct_vs_tiled_fixed_patch"]["passed"])
    patch = report["fixed_patch"]
    target_std = float(patch["target_distribution"]["std"])
    reconstruction_std = float(
        patch["reconstruction_from_latent_mean"]["distribution"]["std"]
    )
    std_ratio = reconstruction_std / max(target_std, torch.finfo(torch.float32).eps)
    overlap_reports = report["fixed_full_volume"]["overlap_sweep"]
    nrmse_values = [
        float(value["official_full_volume_metrics"]["nrmse"])
        for value in overlap_reports.values()
    ]
    seam_ratios = [
        value["seam_metric"]["boundary_to_overall_ratio"]
        for value in overlap_reports.values()
        if value["seam_metric"]["boundary_to_overall_ratio"] is not None
    ]
    overlap_nrmse_span = max(nrmse_values) - min(nrmse_values)
    seam_ratio_span = max(seam_ratios) - min(seam_ratios) if seam_ratios else 0.0

    if not identity_passed or not direct_tiled_passed:
        focus = "tiler_contract_or_model_path"
        action = "Fix the reconstruction tiler before interpreting model quality."
    elif std_ratio < spec.collapse_std_ratio_threshold:
        focus = "checkpoint_or_patch_reconstruction_collapse"
        action = (
            "Investigate normalization, checkpoint state, and patch-level reconstruction "
            "before declaring an inference-only tiling cause."
        )
    elif (
        overlap_nrmse_span > spec.overlap_nrmse_span_threshold
        or seam_ratio_span > spec.seam_ratio_span_threshold
    ):
        focus = "overlap_sensitive_full_volume_inference"
        action = (
            "Treat overlap sensitivity as an inference engineering issue; predeclare a "
            "held-out comparison before locking an overlap."
        )
    else:
        focus = "no_single_failure_isolated"
        action = (
            "Do not start another training experiment; review the full diagnostic and "
            "establish held-out reconstruction evidence first."
        )
    return {
        "status": "NO_NEXT_TRAINING_EXPERIMENT",
        "focus": focus,
        "action": action,
        "diagnostic_values": {
            "latent_mean_reconstruction_std_ratio": std_ratio,
            "overlap_nrmse_span": overlap_nrmse_span,
            "seam_ratio_span": seam_ratio_span,
        },
        "predeclared_engineering_thresholds": {
            "collapse_std_ratio": spec.collapse_std_ratio_threshold,
            "overlap_nrmse_span": spec.overlap_nrmse_span_threshold,
            "seam_ratio_span": spec.seam_ratio_span_threshold,
        },
        "heuristic_only": True,
    }


def _assert_sanitized_report(report: Mapping[str, Any]) -> None:
    text = json.dumps(_json_safe(report), sort_keys=True).lower()
    forbidden = (
        '"subject_id":',
        '"sample_id":',
        '"case_id":',
        '"image_path":',
        '"raw_uri":',
        ".nii",
        "/content/drive",
        "\\users\\",
    )
    if any(value in text for value in forbidden):
        raise RuntimeError("Stage-1 diagnostic report contains a private identity or path.")


def _validate_minus_one_one_tensor(tensor: torch.Tensor, name: str) -> None:
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains non-finite values.")
    minimum = float(tensor.min())
    maximum = float(tensor.max())
    tolerance = 1e-4
    if minimum < -1.0 - tolerance or maximum > 1.0 + tolerance:
        raise ValueError(
            f"{name} must be in [-1,1], observed [{minimum:.6f}, {maximum:.6f}]."
        )


def _entry_domain_label(entry: _PatchBankEntry) -> str:
    from fieldbridge.data.domains import Domain

    return Domain.from_dict(dict(entry.domain)).label


def _domain_label(domain: Any) -> str:
    if isinstance(domain, Sequence) and not isinstance(domain, (str, bytes)) and domain:
        domain = domain[0]
    return str(getattr(domain, "label", domain))


def _overlap_key(overlap: float) -> str:
    return f"{float(overlap):.2f}"


def _sanitize_config(value: Mapping[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, (str, Path)) and _path_like_key(str(key)):
            sanitized[str(key)] = "<external>"
        elif isinstance(item, Mapping):
            sanitized[str(key)] = _sanitize_config(item)
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            sanitized[str(key)] = [
                _sanitize_config(element) if isinstance(element, Mapping) else _json_safe(element)
                for element in item
            ]
        else:
            sanitized[str(key)] = _json_safe(item)
    return sanitized


def _path_like_key(key: str) -> bool:
    normalized = key.lower()
    return any(
        token in normalized
        for token in ("path", "root", "manifest", "directory", "_dir", "resume_from")
    )


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        _json_safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
