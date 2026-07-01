"""Abstract latent translator."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch
from torch import nn

from fieldbridge.data.domains import Domain

DomainBatch = Domain | Sequence[Domain]


class BaseTranslator(nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        z: torch.Tensor,
        source_domain: DomainBatch,
        target_domain: DomainBatch,
        t: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        """Translate latent tensor from source domain to target domain."""

