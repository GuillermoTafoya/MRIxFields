"""Identity latent translator for smoke tests."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from fieldbridge.data.domains import Domain
from fieldbridge.models.translators.base import BaseTranslator

DomainBatch = Domain | Sequence[Domain]


class IdentityTranslator(BaseTranslator):
    """Pass-through translator with an optional scalar for optimization smoke tests."""

    def __init__(self, *, learnable_scale: bool = False, initial_scale: float = 1.0) -> None:
        super().__init__()
        if learnable_scale:
            self.scale = nn.Parameter(torch.tensor(float(initial_scale)))
        else:
            self.register_buffer("scale", torch.tensor(float(initial_scale)), persistent=False)

    def forward(
        self,
        z: torch.Tensor,
        source_domain: DomainBatch,
        target_domain: DomainBatch,
        t: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        del source_domain, target_domain, t
        return z * self.scale.to(device=z.device, dtype=z.dtype)

