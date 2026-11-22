"""Decoder input policies for photometry-factored Stage 2 evaluation."""

from __future__ import annotations

import torch

from fieldbridge.data.photometry_factored_bank_dataset import FactoredLatentStats

DECODER_CONTEXT_CONTRACT = "stage2-decoder-context-v1"
HISTORICAL_DENORMALIZE = "historical"
SOURCE_NATIVE_EXTERIOR = "native-exterior"
DECODER_CONTEXT_POLICIES = (HISTORICAL_DENORMALIZE, SOURCE_NATIVE_EXTERIOR)


def decoder_input_latent(
    endpoint: torch.Tensor,
    *,
    stats: FactoredLatentStats,
    native_source: torch.Tensor | None = None,
    valid_core: torch.Tensor | None = None,
    policy: str = HISTORICAL_DENORMALIZE,
) -> torch.Tensor:
    """Assemble the raw decoder input while leaving the normalized endpoint untouched."""
    if policy not in DECODER_CONTEXT_POLICIES:
        raise ValueError(f"Unknown decoder context {policy!r}.")
    if not endpoint.is_floating_point():
        raise ValueError("Expected a floating-point normalized endpoint.")
    if endpoint.ndim != 5 or endpoint.shape[1] != stats.mean.numel():
        raise ValueError("Expected a batched [N,C,D,H,W] endpoint matching the latent statistics.")
    learned = stats.denormalize(endpoint)
    if policy == HISTORICAL_DENORMALIZE:
        return learned
    if native_source is None or native_source.shape != learned.shape or valid_core is None:
        raise ValueError("Native exterior requires the original source latent and valid core.")
    if not native_source.is_floating_point() or native_source.dtype != learned.dtype:
        raise ValueError("Expected a floating-point native source latent in decoder precision.")
    if native_source.device != learned.device or valid_core.device != learned.device:
        raise ValueError("Native source, valid core and endpoint must share a device.")
    if valid_core.dtype != torch.bool or valid_core.shape != learned[:, :1].shape:
        raise ValueError("Expected a boolean [N,1,D,H,W] source valid core.")
    if not bool(torch.isfinite(learned).all()) or not bool(torch.isfinite(native_source).all()):
        raise ValueError("Non-finite transport endpoint or source posterior in decoder context.")
    return torch.where(valid_core, learned, native_source)
