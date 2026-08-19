"""Support-aware shared 15-domain critics for unified Stage-2."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from fieldbridge.data.domains import CONTRASTS, FIELD_STRENGTHS_T, Domain

DOMAIN_COUNT = len(CONTRASTS) * len(FIELD_STRENGTHS_T)


def domain_labels(
    domains: Domain | Sequence[Domain], batch_size: int, device: torch.device
) -> torch.Tensor:
    values = [domains] * batch_size if isinstance(domains, Domain) else list(domains)
    if len(values) != batch_size:
        raise ValueError(f"Expected {batch_size} domains, got {len(values)}.")
    return torch.tensor(
        [
            item.contrast_index * len(FIELD_STRENGTHS_T)
            + FIELD_STRENGTHS_T.index(item.field_strength_t)
            for item in values
        ],
        dtype=torch.long,
        device=device,
    )


def supported_critic_input(tensor: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    """Mask invalid cells and append support so the critic cannot mistake padding for data."""

    if tensor.ndim != 5 or support.ndim != 5 or support.shape[0] != tensor.shape[0]:
        raise ValueError("Critic tensors/support must be aligned 5-D batches.")
    if support.shape[1] != 1 or support.shape[2:] != tensor.shape[2:]:
        raise ValueError("Critic support must have shape (B,1,D,H,W).")
    if not bool(support.any(dim=(1, 2, 3, 4)).all()):
        raise ValueError("Critic support is empty.")
    mask = support.to(device=tensor.device, dtype=tensor.dtype)
    return torch.cat([tensor * mask, mask], dim=1)


class DomainProjectionDiscriminator(nn.Module):
    """Projection realism critic plus requested-domain classifier."""

    def __init__(self, in_channels: int, channels: Sequence[int] = (32, 64, 128)) -> None:
        super().__init__()
        if in_channels < 1 or not channels or any(int(value) < 1 for value in channels):
            raise ValueError("Critic channel counts must be positive.")
        blocks: list[nn.Module] = []
        current = int(in_channels)
        for output in map(int, channels):
            blocks.extend(
                (nn.Conv3d(current, output, 3, stride=2, padding=1), nn.LeakyReLU(0.2))
            )
            current = output
        self.features = nn.Sequential(*blocks)
        self.realism = nn.Linear(current, 1)
        self.domain = nn.Linear(current, DOMAIN_COUNT)
        self.projection = nn.Embedding(DOMAIN_COUNT, current)

    def forward(
        self, tensor: torch.Tensor, domains: Domain | Sequence[Domain]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tensor.ndim != 5:
            raise ValueError(f"Critic expects (B,C,D,H,W), got {tuple(tensor.shape)}.")
        if max(tensor.shape[2:]) > 64:
            tensor = F.adaptive_avg_pool3d(tensor, 64)
        labels = domain_labels(domains, tensor.shape[0], tensor.device)
        features = self.features(tensor).mean(dim=(2, 3, 4))
        score = self.realism(features).squeeze(1)
        score = score + (features * self.projection(labels)).sum(1)
        return score, self.domain(features)


__all__ = [
    "DOMAIN_COUNT",
    "DomainProjectionDiscriminator",
    "domain_labels",
    "supported_critic_input",
]
