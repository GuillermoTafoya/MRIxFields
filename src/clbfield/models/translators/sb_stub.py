"""Stub for future Schrödinger bridge translator."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from clbfield.data.domains import Domain
from clbfield.models.translators.base import BaseTranslator

DomainBatch = Domain | Sequence[Domain]


class SchrodingerBridgeTranslatorStub(BaseTranslator):
    def forward(
        self,
        z: torch.Tensor,
        source_domain: DomainBatch,
        target_domain: DomainBatch,
        t: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        del z, source_domain, target_domain, t
        raise NotImplementedError("Schrödinger bridge translator is a research stub.")

