"""Domain conditioning modules."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from fieldbridge.data.domains import CONTRASTS, Domain

DomainBatch = Domain | Sequence[Domain]


def normalize_domain_batch(
    domains: DomainBatch,
    *,
    batch_size: int | None = None,
    name: str = "domains",
) -> list[Domain]:
    """Return a concrete list of domains, broadcasting a scalar domain if needed."""

    if batch_size is not None and batch_size <= 0:
        raise ValueError(f"{name} batch_size must be positive, got {batch_size}.")
    if isinstance(domains, Domain):
        count = 1 if batch_size is None else batch_size
        return [domains for _ in range(count)]
    if isinstance(domains, (str, bytes)):
        raise TypeError(f"{name} must be a Domain or a sequence of Domain objects.")
    result = list(domains)
    if batch_size is not None and len(result) != batch_size:
        raise ValueError(
            f"{name} sequence length must equal batch_size={batch_size}; got {len(result)}."
        )
    for index, domain in enumerate(result):
        if not isinstance(domain, Domain):
            raise TypeError(
                f"{name}[{index}] must be a Domain object, got {type(domain).__name__}."
            )
    return result


class DomainEmbedding(nn.Module):
    """Convert a (source, target) domain pair into a single conditioning vector.

    The `log(f_target / f_source)` term only makes sense for a pair, not a lone
    `Domain`, so this module takes source and target domains together rather than
    encoding either side in isolation. Scalar domains broadcast to the requested
    batch size; domain sequences must match the batch size.
    """

    def __init__(
        self,
        *,
        cond_dim: int | None = None,
        conditioning_dim: int | None = None,
        contrast_embedding_dim: int = 8,
        field_embedding_dim: int = 8,
    ) -> None:
        super().__init__()
        if cond_dim is None:
            cond_dim = 32 if conditioning_dim is None else int(conditioning_dim)
        elif conditioning_dim is not None and int(conditioning_dim) != int(cond_dim):
            raise ValueError(
                "cond_dim and conditioning_dim refer to the same value; received "
                f"{cond_dim} and {conditioning_dim}."
            )
        self.cond_dim = _positive_int(int(cond_dim), "cond_dim")
        self.conditioning_dim = self.cond_dim
        contrast_dim = _positive_int(contrast_embedding_dim, "contrast_embedding_dim")
        field_dim = _positive_int(field_embedding_dim, "field_embedding_dim")
        self.contrast_embedding = nn.Embedding(len(CONTRASTS), contrast_dim)
        self.field_projection = nn.Sequential(
            nn.Linear(2, field_dim),
            nn.SiLU(),
            nn.Linear(field_dim, field_dim),
            nn.SiLU(),
        )
        combined_dim = 2 * field_dim + 1 + 2 * contrast_dim
        self.output_projection = nn.Sequential(
            nn.Linear(combined_dim, self.cond_dim),
            nn.SiLU(),
            nn.Linear(self.cond_dim, self.cond_dim),
        )

    def forward(
        self,
        source_domains: DomainBatch,
        target_domains: DomainBatch,
        *,
        batch_size: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        source_list, target_list = _resolve_domain_pair(
            source_domains,
            target_domains,
            batch_size=batch_size,
        )
        parameter = next(self.parameters())
        if device is None:
            device = parameter.device
        compute_dtype = parameter.dtype

        source_fields = torch.stack(
            [domain.field_encoding(dtype=compute_dtype, device=device) for domain in source_list],
            dim=0,
        )
        target_fields = torch.stack(
            [domain.field_encoding(dtype=compute_dtype, device=device) for domain in target_list],
            dim=0,
        )
        log_field_ratio = (target_fields[:, 0] - source_fields[:, 0]).unsqueeze(-1)

        source_contrast_ids = torch.tensor(
            [domain.contrast_index for domain in source_list], dtype=torch.long, device=device
        )
        target_contrast_ids = torch.tensor(
            [domain.contrast_index for domain in target_list], dtype=torch.long, device=device
        )

        combined = torch.cat(
            [
                self.field_projection(source_fields),
                self.field_projection(target_fields),
                log_field_ratio,
                self.contrast_embedding(source_contrast_ids),
                self.contrast_embedding(target_contrast_ids),
            ],
            dim=-1,
        )
        conditioning = self.output_projection(combined)
        if dtype is not None:
            conditioning = conditioning.to(dtype=dtype)
        return conditioning


class DomainConditioner(DomainEmbedding):
    """Backward-compatible name for :class:`DomainEmbedding`."""


def _resolve_domain_pair(
    source_domains: DomainBatch,
    target_domains: DomainBatch,
    *,
    batch_size: int | None,
) -> tuple[list[Domain], list[Domain]]:
    if batch_size is None:
        source_is_scalar = isinstance(source_domains, Domain)
        target_is_scalar = isinstance(target_domains, Domain)
        if source_is_scalar and target_is_scalar:
            inferred_batch_size = 1
        elif source_is_scalar:
            target_list = normalize_domain_batch(target_domains, name="target_domain")
            inferred_batch_size = len(target_list)
        elif target_is_scalar:
            source_list = normalize_domain_batch(source_domains, name="source_domain")
            inferred_batch_size = len(source_list)
        else:
            source_list = normalize_domain_batch(source_domains, name="source_domain")
            target_list = normalize_domain_batch(target_domains, name="target_domain")
            if len(source_list) != len(target_list):
                raise ValueError(
                    "source_domain and target_domain sequences must have the same length; "
                    f"got {len(source_list)} and {len(target_list)}."
                )
            inferred_batch_size = len(source_list)
        if inferred_batch_size <= 0:
            raise ValueError("Domain batches must contain at least one domain.")
        batch_size = inferred_batch_size

    source_list = normalize_domain_batch(
        source_domains,
        batch_size=batch_size,
        name="source_domain",
    )
    target_list = normalize_domain_batch(
        target_domains,
        batch_size=batch_size,
        name="target_domain",
    )
    return source_list, target_list


def _positive_int(value: int, name: str) -> int:
    integer = int(value)
    if integer <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")
    return integer

