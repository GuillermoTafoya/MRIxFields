"""Pseudo-pair construction for synthetic low-field pretraining."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from fieldbridge.data.degradation import compose_degradation, degradation_strength
from fieldbridge.data.domains import Domain

DomainBatch = Domain | Sequence[Domain]


def make_pseudo_pair(
    x_high: torch.Tensor,
    high_domain: DomainBatch,
    low_domain: DomainBatch,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create ``(x_low_synthetic, x_high_target)`` from high-field tensors."""

    if x_high.ndim not in (4, 5):
        raise ValueError(f"x_high must be a 4D or 5D tensor, got shape {tuple(x_high.shape)}.")
    batch_size = int(x_high.shape[0])
    high_domains = _normalize_domain_batch(high_domain, batch_size=batch_size, name="high_domain")
    low_domains = _normalize_domain_batch(low_domain, batch_size=batch_size, name="low_domain")
    strengths = [
        degradation_strength(high.field_strength_t, low.field_strength_t)
        for high, low in zip(high_domains, low_domains)
    ]
    if len(set(round(strength, 6) for strength in strengths)) == 1:
        x_low = compose_degradation(x_high, strengths[0], generator=generator)
    else:
        degraded = [
            compose_degradation(x_high[index : index + 1], strength, generator=generator)
            for index, strength in enumerate(strengths)
        ]
        x_low = torch.cat(degraded, dim=0)
    return x_low, x_high.clone()


def _normalize_domain_batch(domains: DomainBatch, *, batch_size: int, name: str) -> list[Domain]:
    if isinstance(domains, Domain):
        return [domains for _ in range(batch_size)]
    if isinstance(domains, (str, bytes)):
        raise TypeError(f"{name} must be a Domain or sequence of Domain objects.")
    result = list(domains)
    if len(result) != batch_size:
        raise ValueError(f"{name} must have length {batch_size}; got {len(result)}.")
    for index, domain in enumerate(result):
        if not isinstance(domain, Domain):
            raise TypeError(f"{name}[{index}] must be a Domain, got {type(domain).__name__}.")
    return result
