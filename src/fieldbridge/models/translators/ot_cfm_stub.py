"""Stub for future optimal-transport conditional flow matching translator."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from fieldbridge.data.domains import Domain
from fieldbridge.models.translators.base import BaseTranslator

DomainBatch = Domain | Sequence[Domain]


class OTCFMTranslatorStub(BaseTranslator):
    def forward(
        self,
        z: torch.Tensor,
        source_domain: DomainBatch,
        target_domain: DomainBatch,
        t: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        del z, source_domain, target_domain, t
        raise NotImplementedError("OT-CFM translator is a research stub.")

