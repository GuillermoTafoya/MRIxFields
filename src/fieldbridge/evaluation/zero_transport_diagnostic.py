"""Full-volume retrospective reconstruction controls for the August masked latent path."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch

from fieldbridge.data.domains import Domain
from fieldbridge.data.latent_bank import encode_latent
from fieldbridge.data.photometry_factored_bank_dataset import FactoredLatentStats
from fieldbridge.data.photometry_factored_latent_bank import (
    propagate_encoder_local_valid_core_support,
)
from fieldbridge.data.photometry_factorization import (
    FrozenPhotometryArtifact,
    canonical_tensor_sha256,
)
from fieldbridge.data.stage2_canonical_volume import storage_tensor_sha256, validated_source_volume
from fieldbridge.evaluation.retrospective_bypass import finish, image_regions, summarize_difference
from fieldbridge.evaluation.stage2_step200_inference_audit import (
    _require_linear_decoder_output_activation,
)
from fieldbridge.models.autoencoders.base import BaseDecoder, BaseEncoder


@contextmanager
def _ieee_encoder_convolutions(device: torch.device) -> Iterator[None]:
    """Disable the CUDA convolution TF32 shortcut just for the Run C encoder."""
    if device.type != "cuda":
        yield
        return
    cudnn = torch.backends.cudnn
    conv = getattr(cudnn, "conv", None)
    if conv is not None and hasattr(conv, "fp32_precision"):
        owner, key, value = conv, "fp32_precision", "ieee"
    else:  # PyTorch releases before the per-operator precision controls.
        owner, key, value = cudnn, "allow_tf32", False
    previous = getattr(owner, key)
    try:
        setattr(owner, key, value)
        yield
    finally:
        setattr(owner, key, previous)


def _verify_sidecar(
    sidecar: Mapping[str, Any],
    source: torch.Tensor,
    canonical: torch.Tensor,
    foreground: torch.Tensor,
    core: torch.Tensor,
) -> None:
    observed = {
        "source_shape": list(source.shape),
        "source_loaded_array_sha256": canonical_tensor_sha256(source),
        "canonical_shape": list(canonical.shape),
        "canonical_tensor_sha256": canonical_tensor_sha256(canonical),
        "source_support_shape": list(foreground.shape),
        "source_support_tensor_sha256": storage_tensor_sha256(foreground),
        "local_valid_core_support_shape": list(core.shape),
        "local_valid_core_support_mask_sha256": storage_tensor_sha256(core),
    }
    for key, value in observed.items():
        if sidecar.get(key) != value:
            raise ValueError(f"Full-volume diagnostic mismatch with frozen bank sidecar: {key}.")


@torch.inference_mode()
def run_zero_transport_scan(
    source: torch.Tensor,
    domain: Domain,
    *,
    artifact: FrozenPhotometryArtifact,
    encoder: BaseEncoder,
    decoder: BaseDecoder,
    stats: FactoredLatentStats,
    support_rule_sha256: str,
    device: str | torch.device,
    expected_sidecar: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare native (A), August round-trip (B), and outside-core restoration (C)."""
    device = torch.device(device)
    _require_linear_decoder_output_activation(decoder)
    if source.ndim == 4:
        source = source.unsqueeze(0)
    source = validated_source_volume(source, domain.label)
    foreground = source != 0
    if not bool(foreground.any()):
        raise ValueError("Raw source foreground is empty.")
    canonical = artifact.normalize_source(source, domain)
    if not torch.equal(canonical.support_mask, foreground):
        raise ValueError("Canonicalization changed the original source foreground.")
    encoder.eval()
    decoder.eval()
    core = propagate_encoder_local_valid_core_support(
        foreground, encoder, expected_rule_sha256=support_rule_sha256,
    )
    if expected_sidecar is not None:
        _verify_sidecar(expected_sidecar, source, canonical.values, foreground, core)
    with _ieee_encoder_convolutions(device), torch.autocast(device.type, enabled=False):
        native, _ = encode_latent(
            encoder,
            canonical.values.to(device),
            domain,
            strategy="full",
            block_size=(1, 1, 1),
            halo=(0, 0, 0),
            precision="float32",
        )
    if tuple(core.shape) != tuple(native.shape[2:]):
        raise ValueError("Propagated support and native posterior mean shapes differ.")
    latent_mask = core[None, None].to(device)
    masked = stats.denormalize(stats.normalize(native, latent_mask))
    restored = torch.where(latent_mask, masked, native)

    core_array = core.cpu().numpy()
    latent_masks = {
        "all": np.ones_like(core_array), "core": core_array, "outside_core": ~core_array,
    }
    native_latent = native[0].cpu().numpy()
    totals: dict[str, list[float]] = {}
    for name, latent in (("masked", masked), ("restored", restored)):
        summarize_difference(
            totals, name + "_latent_vs_native", latent[0].cpu().numpy(),
            native_latent, latent_masks,
        )

    # Projection partitions image scores only. All decoder arms see the full latent grid.
    regions = image_regions(core_array, source.shape[2:], foreground[0, 0].numpy())
    foreground_regions = {key: mask for key, mask in regions.items() if key != "background"}
    measured = canonical.values[0].numpy()

    def decode_raw(latent: torch.Tensor) -> np.ndarray:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            decoded = decoder.decode(latent, [domain])
        if decoded.shape != source.shape:
            raise ValueError("Full-volume decoder output and source shapes differ.")
        return decoded[0].to(device="cpu", dtype=torch.float32).numpy()

    native_image = decode_raw(native)
    summarize_difference(
        totals, "native_image_vs_measured", native_image, measured, foreground_regions,
    )
    for name, latent in (("masked", masked), ("restored", restored)):
        image = decode_raw(latent)
        summarize_difference(
            totals, name + "_image_vs_measured", image, measured, foreground_regions,
        )
        summarize_difference(totals, name + "_image_vs_native", image, native_image, regions)
        if name == "restored":
            summarize_difference(
                totals, "restored_image_vs_native", image, native_image,
                {"full_volume": np.ones_like(regions["foreground"])},
            )
        del image
    metrics = finish(totals)
    strata = {
        "foreground": "foreground",
        "projected_core_foreground": "foreground_core",
        "remaining_foreground": "foreground_rim",
    }
    image_strata = {}
    for label, region in strata.items():
        native_metric = metrics[f"native_image_vs_measured/{region}"]
        a = native_metric["rmse"]
        b = metrics[f"masked_image_vs_measured/{region}"]["rmse"]
        c = metrics[f"restored_image_vs_measured/{region}"]["rmse"]
        difference = metrics[f"restored_image_vs_native/{region}"]
        image_strata[label] = {
            "count": native_metric["count"],
            "a_rmse": a,
            "b_rmse": b,
            "c_rmse": c,
            "b_minus_a_rmse": b - a if a is not None else None,
            "c_minus_a_rmse": c - a if a is not None else None,
            "c_vs_a_rmse": difference["rmse"],
            "c_vs_a_max_abs": difference["max_abs"],
        }
    return {
        "domain": domain.label,
        "image_shape": list(source.shape),
        "latent_shape": list(native.shape),
        "encoder_precision": "float32",
        "encoder_cudnn_tf32": False if device.type == "cuda" else None,
        "decoder_autocast_dtype": "bfloat16",
        "primary_b_minus_a_rmse": image_strata["foreground"]["b_minus_a_rmse"],
        "image_strata": image_strata,
        "latent_c_vs_a": {
            name: metrics[f"restored_latent_vs_native/{name}"] for name in latent_masks
        },
        "full_image_c_vs_a": metrics["restored_image_vs_native/full_volume"],
        "metrics": metrics,
    }
