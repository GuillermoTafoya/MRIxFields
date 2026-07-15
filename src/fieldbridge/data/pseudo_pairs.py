"""Lazy pseudo-pair slice dataset for deterministic conditional U-Net training."""

from __future__ import annotations

import hashlib
from collections import Counter, OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from fieldbridge.data.contracts import VolumeRecord
from fieldbridge.data.degradation import compose_degradation, degradation_strength
from fieldbridge.data.domains import Domain
from fieldbridge.data.masks import clean_brain_mask
from fieldbridge.data.preprocessing import (
    SliceGeometry,
    SlicePreprocessingSpec,
    preprocess_volume_slice,
    selected_slice_indices,
    to_model_range,
)

ImageLoader = Callable[[Path, VolumeRecord], torch.Tensor]
PseudoPairMode = Literal["train", "validation", "test"]


@dataclass(frozen=True, slots=True)
class PseudoPairSliceSample:
    x_low: torch.Tensor
    x_high: torch.Tensor
    mask: torch.Tensor
    source_domain: Domain
    target_domain: Domain
    record_id: str
    subject_id: str
    volume_path: str
    slice_index: int
    degradation_seed: int
    degradation_strength: float
    geometry: SliceGeometry


@dataclass(frozen=True, slots=True)
class PseudoPairSliceBatch:
    x_low: torch.Tensor
    x_high: torch.Tensor
    mask: torch.Tensor
    source_domain: list[Domain]
    target_domain: list[Domain]
    record_id: list[str]
    subject_id: list[str]
    volume_path: list[str]
    slice_index: torch.Tensor
    degradation_seed: list[int]
    degradation_strength: torch.Tensor
    geometry: tuple[SliceGeometry, ...]


class PseudoPairSliceDataset(Dataset[PseudoPairSliceSample]):
    """Pseudo-pair slices from high-field target volumes with lazy volume loading."""

    def __init__(
        self,
        records: Sequence[VolumeRecord],
        *,
        image_loader: ImageLoader,
        source_field: float,
        sequence: str,
        preprocessing: SlicePreprocessingSpec | None = None,
        mode: PseudoPairMode = "train",
        seed: int = 0,
        cache_size: int = 2,
        mask_threshold: float | None = None,
    ) -> None:
        if mode not in ("train", "validation", "test"):
            raise ValueError("mode must be 'train', 'validation', or 'test'.")
        self.records = tuple(records)
        for record in self.records:
            if record.subject_id is None or not str(record.subject_id).strip():
                raise ValueError(
                    f"Pseudo-pair record {record.case_id!r} requires a non-empty subject_id."
                )
        self.image_loader = image_loader
        self.source_domain = Domain(source_field, sequence)
        self.preprocessing = preprocessing or SlicePreprocessingSpec()
        self.mode = mode
        self.seed = int(seed)
        self.cache_size = max(0, int(cache_size))
        self.mask_threshold = mask_threshold
        self.slice_indices = selected_slice_indices(self.preprocessing)
        self.samples = tuple(
            (record_index, slice_index)
            for record_index in range(len(self.records))
            for slice_index in self.slice_indices
        )
        self._cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._access_counter = 0

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> PseudoPairSliceSample:
        record_index, slice_index = self.samples[int(index)]
        record = self.records[record_index]
        volume = self._get_volume(record)
        high_01, geometry = preprocess_volume_slice(
            volume,
            slice_index,
            self.preprocessing,
            apply_model_range=False,
        )
        target_domain = record.domain
        strength = degradation_strength(target_domain, self.source_domain)
        degradation_seed = self._degradation_seed(int(index), record, slice_index)
        generator = torch.Generator().manual_seed(degradation_seed)
        low_01 = compose_degradation(high_01.unsqueeze(0), strength, generator=generator).squeeze(0)
        low_01 = low_01.clamp(0.0, 1.0)
        mask = _slice_mask(high_01, geometry, threshold=self.mask_threshold)
        return PseudoPairSliceSample(
            x_low=to_model_range(low_01, self.preprocessing.model_range),
            x_high=to_model_range(high_01, self.preprocessing.model_range),
            mask=mask.to(dtype=torch.float32),
            source_domain=self.source_domain,
            target_domain=target_domain,
            record_id=record.case_id,
            subject_id=str(record.subject_id),
            volume_path=str(record.image_path),
            slice_index=slice_index,
            degradation_seed=degradation_seed,
            degradation_strength=strength,
            geometry=geometry,
        )

    def field_for_index(self, index: int) -> float:
        record_index, _ = self.samples[int(index)]
        return float(self.records[record_index].domain.field_strength_t)

    def clear_cache(self) -> None:
        self._cache.clear()

    def _degradation_seed(self, index: int, record: VolumeRecord, slice_index: int) -> int:
        if self.mode == "train":
            self._access_counter += 1
            salt = self._access_counter
        else:
            salt = 0
        return _stable_seed(self.seed, self.mode, index, record.case_id, slice_index, salt)

    def _get_volume(self, record: VolumeRecord) -> torch.Tensor:
        key = f"{record.case_id}:{record.image_path}"
        if key in self._cache:
            volume = self._cache.pop(key)
            self._cache[key] = volume
            return volume
        volume = self.image_loader(record.image_path, record)
        _validate_loaded_volume(volume, record)
        if self.cache_size > 0:
            self._cache[key] = volume
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return volume


def collate_pseudo_pair_slices(items: Sequence[PseudoPairSliceSample]) -> PseudoPairSliceBatch:
    if not items:
        raise ValueError("Cannot collate an empty pseudo-pair batch.")
    return PseudoPairSliceBatch(
        x_low=torch.stack([item.x_low for item in items], dim=0),
        x_high=torch.stack([item.x_high for item in items], dim=0),
        mask=torch.stack([item.mask for item in items], dim=0),
        source_domain=[item.source_domain for item in items],
        target_domain=[item.target_domain for item in items],
        record_id=[item.record_id for item in items],
        subject_id=[item.subject_id for item in items],
        volume_path=[item.volume_path for item in items],
        slice_index=torch.tensor([item.slice_index for item in items], dtype=torch.long),
        degradation_seed=[item.degradation_seed for item in items],
        degradation_strength=torch.tensor(
            [item.degradation_strength for item in items],
            dtype=torch.float32,
        ),
        geometry=tuple(item.geometry for item in items),
    )


def make_field_balanced_sampler(
    dataset: PseudoPairSliceDataset,
    *,
    seed: int,
    num_samples: int | None = None,
) -> WeightedRandomSampler:
    """Return inverse-frequency sampling weights by target field."""

    if len(dataset) == 0:
        raise ValueError("Cannot build a sampler for an empty dataset.")
    counts = Counter(dataset.field_for_index(index) for index in range(len(dataset)))
    weights = torch.tensor(
        [1.0 / counts[dataset.field_for_index(index)] for index in range(len(dataset))],
        dtype=torch.double,
    )
    generator = torch.Generator().manual_seed(int(seed))
    return WeightedRandomSampler(
        weights,
        num_samples=int(num_samples) if num_samples is not None else len(dataset),
        replacement=True,
        generator=generator,
    )


def _validate_loaded_volume(volume: torch.Tensor, record: VolumeRecord) -> None:
    if volume.ndim != 4:
        raise ValueError(
            f"Loader for record {record.case_id} must return raw NIfTI order (C,X,Y,Z), "
            f"got {tuple(volume.shape)}."
        )
    if not torch.isfinite(volume).all():
        raise ValueError(f"Loader returned non-finite values for record {record.case_id}.")


def _slice_mask(
    image_01: torch.Tensor,
    geometry: SliceGeometry,
    *,
    threshold: float | None,
) -> torch.Tensor:
    geometry_mask = torch.zeros_like(image_01, dtype=torch.float32)
    geometry_mask[
        :,
        geometry.pad_top : geometry.output_height - geometry.pad_bottom,
        geometry.pad_left : geometry.output_width - geometry.pad_right,
    ] = 1.0
    if threshold is None:
        return geometry_mask
    intensity_mask = clean_brain_mask(
        image_01.unsqueeze(0),
        threshold=threshold,
        kernel_size=3,
        iterations=1,
    ).squeeze(0)
    return intensity_mask * geometry_mask


def _stable_seed(*parts: object) -> int:
    text = ":".join(str(part) for part in parts)
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2**31 - 1)
