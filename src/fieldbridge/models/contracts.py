"""Model interface contracts."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import torch

from fieldbridge.data.domains import Domain

DomainBatch = Domain | Sequence[Domain]


class Encoder(Protocol):
    def encode(self, x: torch.Tensor, domain: DomainBatch) -> torch.Tensor:
        """Encode image tensor into latent tensor."""


class Decoder(Protocol):
    def decode(self, z: torch.Tensor, domain: DomainBatch) -> torch.Tensor:
        """Decode latent tensor into image tensor."""


class Translator(Protocol):
    def forward(
        self,
        z: torch.Tensor,
        source_domain: DomainBatch,
        target_domain: DomainBatch,
        t: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        """Translate latent tensor from source domain to target domain."""

