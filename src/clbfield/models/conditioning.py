"""Domain conditioning modules."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from clbfield.data.domains import CONTRASTS, Domain

DomainBatch = Domain | Sequence[Domain]


def normalize_domain_batch(domains: DomainBatch, *, batch_size: int | None = None) -> list[Domain]:
    if isinstance(domains, Domain):
        count = 1 if batch_size is None else batch_size
        return [domains for _ in range(count)]
    result = list(domains)
    if batch_size is not None and len(result) != batch_size:
        raise ValueError(f"Expected {batch_size} domains, received {len(result)}.")
    return result


class DomainConditioner(nn.Module):
    """Convert field and contrast metadata into conditioning vectors."""

    def __init__(
        self,
        *,
        conditioning_dim: int = 32,
        contrast_embedding_dim: int = 8,
        field_embedding_dim: int = 8,
    ) -> None:
        super().__init__()
        self.conditioning_dim = int(conditioning_dim)
        self.contrast_embedding = nn.Embedding(len(CONTRASTS), contrast_embedding_dim)
        self.field_projection = nn.Sequential(
            nn.Linear(1, field_embedding_dim),
            nn.SiLU(),
        )
        self.output_projection = nn.Sequential(
            nn.Linear(contrast_embedding_dim + field_embedding_dim, conditioning_dim),
            nn.SiLU(),
            nn.Linear(conditioning_dim, conditioning_dim),
        )

    def forward(
        self,
        domains: DomainBatch,
        *,
        batch_size: int | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        domain_list = normalize_domain_batch(domains, batch_size=batch_size)
        if device is None:
            device = next(self.parameters()).device
        fields = torch.stack([domain.field_encoding(device=device) for domain in domain_list], dim=0)
        contrast_ids = torch.tensor(
            [domain.contrast_index for domain in domain_list],
            dtype=torch.long,
            device=device,
        )
        contrast_features = self.contrast_embedding(contrast_ids)
        field_features = self.field_projection(fields)
        return self.output_projection(torch.cat([field_features, contrast_features], dim=-1))

