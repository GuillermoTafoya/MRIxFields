"""Batch movement helpers."""

from __future__ import annotations

import torch

from clbfield.data.contracts import LatentBatch, RawBatch


def move_raw_batch(batch: RawBatch, device: torch.device | str) -> RawBatch:
    return RawBatch(
        image=batch.image.to(device),
        source_domain=batch.source_domain,
        target_domain=batch.target_domain,
        metadata=batch.metadata,
    )


def move_latent_batch(batch: LatentBatch, device: torch.device | str) -> LatentBatch:
    return LatentBatch(
        latent=batch.latent.to(device),
        source_domain=batch.source_domain,
        target_domain=batch.target_domain,
        metadata=batch.metadata,
    )

