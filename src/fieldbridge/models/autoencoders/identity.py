"""Identity autoencoder components for interface and smoke tests."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from fieldbridge.data.domains import Domain
from fieldbridge.models.autoencoders.base import BaseDecoder, BaseEncoder

DomainBatch = Domain | Sequence[Domain]


class IdentityEncoder(BaseEncoder):
    def encode(self, x: torch.Tensor, domain: DomainBatch) -> torch.Tensor:
        del domain
        return x


class IdentityDecoder(BaseDecoder):
    def decode(self, z: torch.Tensor, domain: DomainBatch) -> torch.Tensor:
        del domain
        return z

