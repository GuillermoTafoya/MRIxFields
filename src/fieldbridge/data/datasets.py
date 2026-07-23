"""Dataset implementations that are independent of storage backends."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info

from fieldbridge.data.contracts import RawBatch, VolumeRecord
from fieldbridge.data.domains import CONTRASTS, FIELD_STRENGTHS_T, Domain
from fieldbridge.data.masks import threshold_mask
from fieldbridge.data.sampling import exposure_report, joint_domain_subject_balanced_indices
from fieldbridge.data.transforms import StratifiedCropConfig, random_crop, stratified_crop

ImageLoader = Callable[[Path, VolumeRecord], torch.Tensor]
TargetDomainSelector = Callable[[VolumeRecord], Domain]
ImageTransform = Callable[[torch.Tensor], torch.Tensor]
PairSampling = Literal["cycle", "random_any_to_any"]

ALL_DOMAINS: tuple[Domain, ...] = tuple(Domain(field, contrast) for field in FIELD_STRENGTHS_T for contrast in CONTRASTS)


def _identity_target_domain(record: VolumeRecord) -> Domain:
    """Default `TargetDomainSelector`: target == source domain.

    A module-level function, not a lambda: DataLoader worker `spawn` (the default on
    Windows) pickles the dataset, and a lambda attribute makes it unpicklable, so
    `num_workers>0` crashes at worker start. See StreamingPatchDataset.__init__.
    """
    return record.domain


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
        self.target_domain_selector = target_domain_selector or _identity_target_domain
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


class StreamingPatchDataset(IterableDataset[RawBatch]):
    """Streams random patches from a manifest that is far larger than RAM.

    The manifest volumes are full-resolution NIfTI (~231 MB decoded each), but stage-1/
    stage-2 train on small random patches (e.g. 64^3 ~ 1 MB). The plain
    `ManifestVolumeDataset` re-reads and re-decodes a whole volume for *every* patch and
    discards >99% of it, so on a network-backed store (Drive-FUSE) the GPU starves on I/O.
    Caching whole volumes in RAM is not an option here (a manifest of ~1-2k volumes does
    not fit), so this streams instead:

    * shuffles the volume order once per pass (seeded, reproducible),
    * reads each volume from disk exactly once per pass, and
    * yields `patches_per_volume` distinct random crops from it before dropping it.

    Only one volume is resident at a time (minimal RAM), and each disk read is amortized
    across `patches_per_volume` patches — the same total reads as an ideal cache-once
    scheme, without holding the dataset in memory. Intended for `num_workers=0` and
    DataLoader `shuffle=False` (this dataset owns the shuffling); with `num_workers>0` the
    volume list is split disjointly across workers so no volume is read twice per pass.

    `volume_transform` (e.g. percentile-clip normalization) runs once per volume, before
    cropping — patch-invariant preprocessing must not be recomputed per patch.
    """

    def __init__(
        self,
        records: Sequence[VolumeRecord],
        *,
        image_loader: ImageLoader,
        patch_size: Iterable[int] | None,
        patches_per_volume: int = 1,
        volume_transform: ImageTransform | None = None,
        target_domain_selector: TargetDomainSelector | None = None,
        seed: int = 0,
        crop_config: StratifiedCropConfig | None = None,
        foreground_threshold: float = 0.0,
        sampling_weights: Sequence[float] | None = None,
        joint_domain_balance: bool = False,
    ) -> None:
        if patches_per_volume < 1:
            raise ValueError("patches_per_volume must be >= 1.")
        self.records = tuple(records)
        if not self.records:
            raise ValueError("StreamingPatchDataset requires at least one record.")
        # Optional per-record sampling weights (aligned to `records`). None => uniform
        # `randperm` volume order, each volume read exactly once per pass (the default). When
        # given, the volume order is a weighted draw WITH REPLACEMENT (see __iter__) — used for
        # field-strength balancing (see data.sampling.field_balanced_weights).
        if sampling_weights is not None:
            if len(sampling_weights) != len(self.records):
                raise ValueError(
                    f"sampling_weights has {len(sampling_weights)} entries but there are "
                    f"{len(self.records)} records."
                )
            self.sampling_weights = torch.as_tensor(sampling_weights, dtype=torch.float64)
            if self.sampling_weights.numel() and float(self.sampling_weights.min()) < 0:
                raise ValueError("sampling_weights must be non-negative.")
        else:
            self.sampling_weights = None
        if joint_domain_balance and sampling_weights is not None:
            raise ValueError("joint_domain_balance and sampling_weights are mutually exclusive.")
        self.joint_domain_balance = bool(joint_domain_balance)
        self.image_loader = image_loader
        # None => no cropping (yield the whole volume); real configs always set a patch.
        self.patch_size = tuple(int(p) for p in patch_size) if patch_size is not None else None
        self.patches_per_volume = int(patches_per_volume)
        self.volume_transform = volume_transform
        self.target_domain_selector = target_domain_selector or _identity_target_domain
        self.seed = int(seed)
        # None => uniform random cropping (the original behavior). A config switches on
        # foreground/border/air stratified sampling; see transforms.StratifiedCropConfig
        # for why uniform crops waste most of the compute on air at this volume size.
        self.crop_config = crop_config
        self.foreground_threshold = float(foreground_threshold)
        self._pass = 0
        self.last_exposure_report: dict[str, Any] | None = None

    def state_dict(self) -> dict[str, int]:
        return {"pass": int(self._pass)}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self._pass = int(state.get("pass", 0))

    def expected_exposure_report(self, *, pass_index: int | None = None) -> dict[str, Any]:
        pass_value = self._pass if pass_index is None else int(pass_index)
        if self.joint_domain_balance:
            indices = joint_domain_subject_balanced_indices(
                self.records, seed=self.seed, pass_index=pass_value
            )
        else:
            indices = list(range(len(self.records)))
        patch_indices = [
            index for index in indices for _ in range(self.patches_per_volume)
        ]
        report = exposure_report(self.records, patch_indices)
        report.pop("by_volume", None)
        return report

    def __iter__(self) -> "Iterator[RawBatch]":
        records = self.records
        weights = self.sampling_weights
        worker = get_worker_info()
        # Reshuffle each pass so re-iterating (the training loop restarts the iterator on
        # StopIteration) does not replay the same volume order.
        pass_index = self._pass
        generator = torch.Generator().manual_seed(self.seed + pass_index)
        self._pass += 1
        if self.joint_domain_balance:
            order = joint_domain_subject_balanced_indices(
                records, seed=self.seed, pass_index=pass_index
            )
        elif weights is not None:
            # Field-balanced order: a weighted draw WITH REPLACEMENT of `len(records)` volumes.
            # Same number of reads (and steps) per pass as the uniform path, so steps_per_epoch
            # is unchanged — only the field composition shifts. TRADEOFF: with replacement, a
            # volume may be drawn 0..k times per pass, so the "every volume read exactly once per
            # pass" guarantee is relaxed in exchange for up-sampling under-represented fields.
            # Seeded generator keeps it reproducible.
            order = torch.multinomial(
                weights, num_samples=len(records), replacement=True, generator=generator
            ).tolist()
        else:
            order = torch.randperm(len(records), generator=generator).tolist()
        self.last_exposure_report = exposure_report(records, order)
        if worker is not None:
            # Build one global deterministic schedule first, then shard schedule positions.
            # Sharding records first can remove domains from a worker and destroys exact balance.
            order = order[worker.id :: worker.num_workers]
        for record_index in order:
            record = records[record_index]
            volume = self.image_loader(record.image_path, record)
            if self.volume_transform is not None:
                volume = self.volume_transform(volume)
            target_domain = self.target_domain_selector(record)
            metadata = {
                "case_id": record.case_id,
                "subject_id": record.subject_id,
                **dict(record.metadata),
            }
            # Computed once per volume, not per patch: thresholding a full 364^3 volume is
            # far from free and the mask is patch-invariant.
            mask = (
                threshold_mask(volume.unsqueeze(0), threshold=self.foreground_threshold).squeeze(0)
                if self.crop_config is not None and self.patch_size is not None
                else None
            )
            for _ in range(self.patches_per_volume):
                if self.patch_size is None:
                    patch = volume
                elif mask is not None:
                    patch = stratified_crop(
                        volume,
                        patch_size=self.patch_size,
                        mask=mask,
                        config=self.crop_config,
                        generator=generator,
                    )
                else:
                    patch = random_crop(volume, patch_size=self.patch_size, generator=generator)
                yield RawBatch(
                    image=patch,
                    source_domain=record.domain,
                    target_domain=target_domain,
                    metadata=dict(metadata),
                )
            del volume  # drop before reading the next: only one volume resident at a time.


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

