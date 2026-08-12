"""Small projection discriminator for Stage-2 latent and decoded-image ablations."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from fieldbridge.data.domains import CONTRASTS, FIELD_STRENGTHS_T, Domain


class DomainProjectionDiscriminator(nn.Module):
    """Judge realism and predict the acquisition domain of a 3-D tensor."""

    def __init__(self, in_channels: int, channels: Sequence[int] = (32, 64, 128)) -> None:
        super().__init__()
        if in_channels < 1 or not channels or any(int(value) < 1 for value in channels):
            raise ValueError("in_channels and every discriminator channel must be positive.")
        blocks: list[nn.Module] = []
        current = int(in_channels)
        for output in map(int, channels):
            blocks.extend((nn.Conv3d(current, output, 3, stride=2, padding=1), nn.LeakyReLU(0.2)))
            current = output
        self.features = nn.Sequential(*blocks)
        self.realism = nn.Linear(current, 1)
        self.domain = nn.Linear(current, len(CONTRASTS) * len(FIELD_STRENGTHS_T))
        self.projection = nn.Embedding(len(CONTRASTS) * len(FIELD_STRENGTHS_T), current)

    def forward(
        self, tensor: torch.Tensor, domains: Domain | Sequence[Domain]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tensor.ndim != 5:
            raise ValueError(f"discriminator expects (B,C,D,H,W), got {tuple(tensor.shape)}.")
        if max(tensor.shape[2:]) > 64:
            tensor = F.adaptive_avg_pool3d(tensor, 64)
        labels = domain_labels(domains, tensor.shape[0], tensor.device)
        features = self.features(tensor).mean(dim=(2, 3, 4))
        score = self.realism(features).squeeze(1) + (features * self.projection(labels)).sum(1)
        return score, self.domain(features)


def domain_labels(
    domains: Domain | Sequence[Domain], batch_size: int, device: torch.device
) -> torch.Tensor:
    values = [domains] * batch_size if isinstance(domains, Domain) else list(domains)
    if len(values) != batch_size:
        raise ValueError(f"Expected {batch_size} domains, got {len(values)}.")
    labels = [
        domain.contrast_index * len(FIELD_STRENGTHS_T)
        + FIELD_STRENGTHS_T.index(domain.field_strength_t)
        for domain in values
    ]
    return torch.tensor(labels, dtype=torch.long, device=device)


__all__ = ["DomainProjectionDiscriminator", "domain_labels"]
