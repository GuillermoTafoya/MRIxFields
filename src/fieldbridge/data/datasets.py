"""Dataset implementations that are independent of storage backends."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch.utils.data import DataLoader, Dataset

from fieldbridge.data.contracts import RawBatch, VolumeRecord
from fieldbridge.data.domains import CONTRASTS, FIELD_STRENGTHS_T, Domain

ImageLoader = Callable[[Path, VolumeRecord], torch.Tensor]
TargetDomainSelector = Callable[[VolumeRecord], Domain]
ImageTransform = Callable[[torch.Tensor], torch.Tensor]
PairSampling = Literal["cycle", "random_any_to_any"]

ALL_DOMAINS: tuple[Domain, ...] = tuple(Domain(field, contrast) for field in FIELD_STRENGTHS_T for contrast in CONTRASTS)


def random_any_to_any_selector(
    domains: Sequence[Domain] = ALL_DOMAINS, *, seed: int, allow_identity: bool = True
) -> TargetDomainSelector:
    """Build a deterministic any-to-any `TargetDomainSelector` for `ManifestVolumeDataset`.

    Picks the target via a hash of (seed, case_id) rather than global RNG state, so the
    same seed always yields the same target for a given record regardless of call order.
    """

    domain_pool = tuple(domains)
    if not domain_pool:
        raise ValueError("domains must be non-empty.")

    def _select(record: VolumeRecord) -> Domain:
        candidates = domain_pool
        if not allow_identity:
            candidates = tuple(d for d in domain_pool if d != record.domain)
            if not candidates:
                raise ValueError(f"No non-identity target domain available for {record.domain.label}.")
        digest = hashlib.sha256(f"{seed}:{record.case_id}".encode("utf-8")).digest()
        index = int.from_bytes(digest[:8], "big") % len(candidates)
        return candidates[index]

    return _select


@dataclass(frozen=True, slots=True)
class SyntheticDatasetConfig:
    num_samples: int = 4
    volume_shape: tuple[int, int, int, int] = (1, 8, 8, 8)
    seed: int = 13


class ManifestVolumeDataset(Dataset[RawBatch]):
    """Dataset over VolumeRecords using an injected image loader."""

    def __init__(
        self,
        records: Sequence[VolumeRecord],
        *,
        image_loader: ImageLoader,
        target_domain_selector: TargetDomainSelector | None = None,
        transform: ImageTransform | None = None,
    ) -> None:
        self.records = tuple(records)
        self.image_loader = image_loader
        self.target_domain_selector = target_domain_selector or (lambda record: record.domain)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> RawBatch:
        record = self.records[index]
        image = self.image_loader(record.image_path, record)
        if self.transform is not None:
            image = self.transform(image)
        return RawBatch(
            image=image,
            source_domain=record.domain,
            target_domain=self.target_domain_selector(record),
            metadata={"case_id": record.case_id, **dict(record.metadata)},
        )


class SyntheticVolumeDataset(Dataset[RawBatch]):
    """Synthetic 3D tensor dataset for CPU smoke tests."""

    def __init__(
        self,
        *,
        num_samples: int = 4,
        volume_shape: Sequence[int] = (1, 8, 8, 8),
        seed: int = 13,
        source_domains: Sequence[Domain] | None = None,
        target_domains: Sequence[Domain] | None = None,
        pair_sampling: PairSampling = "cycle",
    ) -> None:
        if len(volume_shape) != 4:
            raise ValueError("volume_shape must be [channels, depth, height, width].")
        self.num_samples = int(num_samples)
        self.volume_shape = tuple(int(dim) for dim in volume_shape)
        self.seed = int(seed)
        self.source_domains = tuple(source_domains or (Domain(3.0, "T1w"),))
        self.target_domains = tuple(target_domains or (Domain(1.5, "T2w"),))
        self.pair_sampling: PairSampling = pair_sampling

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> RawBatch:
        generator = torch.Generator().manual_seed(self.seed + int(index))
        image = torch.randn(self.volume_shape, generator=generator)
        if self.pair_sampling == "random_any_to_any":
            source_domain = ALL_DOMAINS[int(torch.randint(0, len(ALL_DOMAINS), (1,), generator=generator).item())]
            target_domain = ALL_DOMAINS[int(torch.randint(0, len(ALL_DOMAINS), (1,), generator=generator).item())]
        else:
            source_domain = self.source_domains[index % len(self.source_domains)]
            target_domain = self.target_domains[index % len(self.target_domains)]
        return RawBatch(
            image=image,
            source_domain=source_domain,
            target_domain=target_domain,
            metadata={"case_id": f"synthetic-{index:04d}", "synthetic": True},
        )


def collate_raw_batches(items: Sequence[RawBatch]) -> RawBatch:
    if not items:
        raise ValueError("Cannot collate an empty batch.")
    return RawBatch(
        image=torch.stack([item.image for item in items], dim=0),
        source_domain=[item.source_domain for item in items],  # type: ignore[list-item]
        target_domain=[item.target_domain for item in items],  # type: ignore[list-item]
        metadata=[dict(item.metadata) for item in items],  # type: ignore[arg-type]
    )


def make_synthetic_loader(
    *,
    batch_size: int = 2,
    shuffle: bool = False,
    **dataset_kwargs: Any,
) -> DataLoader[RawBatch]:
    dataset = SyntheticVolumeDataset(**dataset_kwargs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_raw_batches)

