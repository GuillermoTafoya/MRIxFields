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
    """Convert a (source, target) domain pair into a single conditioning vector.

    The `log(f_target / f_source)` term only makes sense for a pair, not a lone
    `Domain`, so this conditioner takes source and target domains together rather
    than encoding either side in isolation.
    """

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
            nn.Linear(2, field_embedding_dim),
            nn.SiLU(),
        )
        combined_dim = 2 * field_embedding_dim + 1 + 2 * contrast_embedding_dim
        self.output_projection = nn.Sequential(
            nn.Linear(combined_dim, conditioning_dim),
            nn.SiLU(),
            nn.Linear(conditioning_dim, conditioning_dim),
        )

    def forward(
        self,
        source_domains: DomainBatch,
        target_domains: DomainBatch,
        *,
        batch_size: int | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        source_list = normalize_domain_batch(source_domains, batch_size=batch_size)
        target_list = normalize_domain_batch(target_domains, batch_size=len(source_list))
        if device is None:
            device = next(self.parameters()).device

        source_fields = torch.stack([domain.field_encoding(device=device) for domain in source_list], dim=0)
        target_fields = torch.stack([domain.field_encoding(device=device) for domain in target_list], dim=0)
        log_field_ratio = (target_fields[:, 0] - source_fields[:, 0]).unsqueeze(-1)

        source_contrast_ids = torch.tensor(
            [domain.contrast_index for domain in source_list], dtype=torch.long, device=device
        )
        target_contrast_ids = torch.tensor(
            [domain.contrast_index for domain in target_list], dtype=torch.long, device=device
        )

        combined = torch.cat(
            [
                self.field_projection(source_fields),
                self.field_projection(target_fields),
                log_field_ratio,
                self.contrast_embedding(source_contrast_ids),
                self.contrast_embedding(target_contrast_ids),
            ],
            dim=-1,
        )
        return self.output_projection(combined)

