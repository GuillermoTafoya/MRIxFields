"""Typed data contracts shared across sources, datasets, and training."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from fieldbridge.data.domains import Domain, domain_from_any

DomainBatch = Domain | Sequence[Domain]
Metadata = Mapping[str, Any] | Sequence[Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class VolumeRecord:
    """Single image volume reference plus acquisition domain metadata."""

    case_id: str
    image_path: Path | str
    domain: Domain | dict[str, Any]
    subject_id: str | None = None
    split: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "image_path", Path(self.image_path))
        object.__setattr__(self, "domain", domain_from_any(self.domain))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "image_path": str(self.image_path),
            "domain": self.domain.to_dict(),
            "subject_id": self.subject_id,
            "split": self.split,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class RawBatch:
    """Batch of image tensors with source and target domains."""

    image: torch.Tensor
    source_domain: DomainBatch
    target_domain: DomainBatch
    metadata: Metadata = field(default_factory=dict)


@dataclass(slots=True)
class LatentBatch:
    """Batch of latent tensors with source and target domains."""

    latent: torch.Tensor
    source_domain: DomainBatch
    target_domain: DomainBatch
    metadata: Metadata = field(default_factory=dict)

