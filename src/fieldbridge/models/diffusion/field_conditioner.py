"""Single-domain field-strength conditioner for the Etapa 1 diffuser.

Unlike `DomainConditioner` (models/conditioning.py, pair-based: source AND target
domain, built for Etapa 2's source->target translation conditioning), this conditions
on ONE volume's own field strength (and contrast, since `Domain.conditioning_vector()`
bundles both) — there is no source->target pair inside Etapa 1's diffuser, only a
single domain per sample.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from fieldbridge.data.domains import Domain
from fieldbridge.models.conditioning import normalize_domain_batch

DomainBatch = Domain | Sequence[Domain]

# Domain.conditioning_vector() = cat([field_encoding() (2-dim), contrast_encoding() (3-dim)]).
_DOMAIN_CONDITIONING_DIM = 5


class FieldStrengthConditioner(nn.Module):
    """Project a single Domain's `conditioning_vector()` into a conditioning embedding."""

    def __init__(self, *, conditioning_dim: int = 32, hidden_dim: int = 32) -> None:
        super().__init__()
        self.conditioning_dim = int(conditioning_dim)
        self.projection = nn.Sequential(
            nn.Linear(_DOMAIN_CONDITIONING_DIM, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.conditioning_dim),
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
        vectors = torch.stack([domain.conditioning_vector(device=device) for domain in domain_list], dim=0)
        return self.projection(vectors)
