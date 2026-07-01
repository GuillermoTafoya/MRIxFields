"""Abstract autoencoder components."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch
from torch import nn

from fieldbridge.data.domains import Domain

DomainBatch = Domain | Sequence[Domain]


class BaseEncoder(nn.Module, ABC):
    @abstractmethod
    def encode(self, x: torch.Tensor, domain: DomainBatch) -> torch.Tensor:
        """Encode an image tensor into a latent tensor."""

    def forward(self, x: torch.Tensor, domain: DomainBatch) -> torch.Tensor:
        return self.encode(x, domain)


class BaseDecoder(nn.Module, ABC):
    @abstractmethod
    def decode(self, z: torch.Tensor, domain: DomainBatch) -> torch.Tensor:
        """Decode a latent tensor into an image tensor."""

    def forward(self, z: torch.Tensor, domain: DomainBatch) -> torch.Tensor:
        return self.decode(z, domain)

