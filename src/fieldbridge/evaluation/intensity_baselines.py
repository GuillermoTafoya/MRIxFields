"""Frozen Gate-0 image-space intensity baselines.

This module preserves the branch-only Gate-0 v1 API and arithmetic.  In particular,
``ImageIntensityBaseline.apply`` computes the empirical CDF of the supplied image and
projects it onto a target-domain template learned from training images.  It is therefore
a post-hoc target-CDF projection (operation B in the Stage-2 plan), even though Gate 0
only applied it to the identity reconstruction.

Gate 0.1 deliberately does not change this API: its stricter exact-zero-background and
provenance behavior lives in :mod:`fieldbridge.evaluation.stage2_gate01`.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from fieldbridge.data.domains import Domain

IntensityMode = Literal["robust_affine", "histogram"]
INTENSITY_MODES: tuple[IntensityMode, ...] = ("robust_affine", "histogram")
INTENSITY_BASELINE_CONTRACT_VERSION = "image-intensity-baseline-v1"


@dataclass(frozen=True, slots=True)
class DomainIntensityReference:
    """One domain's foreground quantiles on a fixed probability grid."""

    quantiles: torch.Tensor
    volumes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "quantiles": [float(value) for value in self.quantiles],
            "volumes": int(self.volumes),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DomainIntensityReference":
        return cls(
            quantiles=torch.tensor(
                [float(value) for value in data["quantiles"]], dtype=torch.float32
            ),
            volumes=int(data["volumes"]),
        )


def _foreground_values(image: torch.Tensor, mask_threshold: float) -> torch.Tensor:
    flat = image.reshape(-1).to(torch.float32)
    selected = flat[flat > mask_threshold]
    return selected if selected.numel() else flat


def _quantiles_of(
    image: torch.Tensor,
    probabilities: torch.Tensor,
    mask_threshold: float,
) -> torch.Tensor:
    values = _foreground_values(image, mask_threshold)
    limit = 4_000_000
    if values.numel() > limit:
        step = values.numel() // limit + 1
        values = values[::step]
    return torch.quantile(values, probabilities.to(values.device))


@dataclass(frozen=True, slots=True)
class ImageIntensityBaseline:
    """Legacy Gate-0 per-domain intensity references and induced monotone map."""

    references: Mapping[str, DomainIntensityReference]
    probabilities: torch.Tensor
    mode: IntensityMode
    mask_threshold: float = 0.0
    low_probability: float = 0.01
    high_probability: float = 0.99
    provenance: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.references:
            raise ValueError("ImageIntensityBaseline needs at least one domain reference.")
        if self.mode not in INTENSITY_MODES:
            raise ValueError(f"mode must be one of {INTENSITY_MODES}; got {self.mode!r}.")
        if self.provenance is None:
            object.__setattr__(self, "provenance", {})

    def _reference_for(self, domain: Domain) -> DomainIntensityReference:
        found = self.references.get(domain.label)
        if found is None:
            raise KeyError(
                f"No intensity reference for domain {domain.label!r}. Fitted domains: "
                f"{sorted(self.references)}."
            )
        return found

    def apply(self, image: torch.Tensor, target: Domain) -> torch.Tensor:
        """Apply the frozen Gate-0 v1 mapping; arithmetic is intentionally unchanged."""

        reference = self._reference_for(target)
        probabilities = self.probabilities.to(image.device)
        source_quantiles = _quantiles_of(image, probabilities, self.mask_threshold)
        target_quantiles = reference.quantiles.to(image.device, image.dtype)

        if self.mode == "robust_affine":
            low_index = int(
                torch.argmin((probabilities - self.low_probability).abs())
            )
            high_index = int(
                torch.argmin((probabilities - self.high_probability).abs())
            )
            source_low = source_quantiles[low_index]
            source_high = source_quantiles[high_index]
            target_low = target_quantiles[low_index]
            target_high = target_quantiles[high_index]
            span = source_high - source_low
            if float(span.abs()) < 1e-8:
                return image
            scale = (target_high - target_low) / span
            return (image - source_low) * scale + target_low

        flat = image.reshape(-1).to(torch.float32)
        mapped = _interpolate_monotone(flat, source_quantiles, target_quantiles)
        return mapped.reshape(image.shape).to(image.dtype)

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": INTENSITY_BASELINE_CONTRACT_VERSION,
            "mode": self.mode,
            "probabilities": [float(value) for value in self.probabilities],
            "mask_threshold": float(self.mask_threshold),
            "low_probability": float(self.low_probability),
            "high_probability": float(self.high_probability),
            "provenance": dict(self.provenance or {}),
            "references": {
                key: value.to_dict() for key, value in sorted(self.references.items())
            },
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        mode: IntensityMode | None = None,
    ) -> "ImageIntensityBaseline":
        version = str(data.get("contract_version", ""))
        if version != INTENSITY_BASELINE_CONTRACT_VERSION:
            raise ValueError(
                f"Intensity baseline contract mismatch: file has {version!r}, code "
                f"expects {INTENSITY_BASELINE_CONTRACT_VERSION!r}."
            )
        return cls(
            references={
                key: DomainIntensityReference.from_dict(value)
                for key, value in data["references"].items()
            },
            probabilities=torch.tensor(
                [float(value) for value in data["probabilities"]],
                dtype=torch.float32,
            ),
            mode=mode or data["mode"],
            mask_threshold=float(data.get("mask_threshold", 0.0)),
            low_probability=float(data.get("low_probability", 0.01)),
            high_probability=float(data.get("high_probability", 0.99)),
            provenance=dict(data.get("provenance", {})),
        )

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )
        temporary.replace(target)
        return target

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        mode: IntensityMode | None = None,
    ) -> "ImageIntensityBaseline":
        return cls.from_dict(
            json.loads(Path(path).read_text(encoding="utf-8")), mode=mode
        )


def _interpolate_monotone(
    values: torch.Tensor,
    source_grid: torch.Tensor,
    target_grid: torch.Tensor,
) -> torch.Tensor:
    """Piecewise-linear interpolation through ``source_grid -> target_grid``."""

    source_grid = source_grid.to(values.dtype)
    target_grid = target_grid.to(values.dtype)
    indices = torch.searchsorted(
        source_grid.contiguous(), values.contiguous()
    ).clamp(1, source_grid.numel() - 1)
    left = source_grid[indices - 1]
    right = source_grid[indices]
    left_target = target_grid[indices - 1]
    right_target = target_grid[indices]
    span = (right - left).clamp_min(1e-8)
    weight = ((values - left) / span).clamp(0.0, 1.0)
    return left_target + weight * (right_target - left_target)


def fit_image_intensity_baselines(
    volumes: Iterable[tuple[torch.Tensor, Domain]],
    *,
    num_quantiles: int = 256,
    mask_threshold: float = 0.0,
    low_probability: float = 0.01,
    high_probability: float = 0.99,
    provenance: Mapping[str, Any] | None = None,
    log: bool = False,
) -> dict[IntensityMode, ImageIntensityBaseline]:
    """Fit both legacy Gate-0 baselines in one pass over training volumes."""

    probabilities = torch.linspace(0.0, 1.0, num_quantiles, dtype=torch.float32)
    accumulated: dict[str, list[torch.Tensor]] = {}
    for volume, domain in volumes:
        quantiles = _quantiles_of(volume, probabilities, mask_threshold).cpu()
        accumulated.setdefault(domain.label, []).append(quantiles)
        if log:
            print(f"intensity_baseline fit domain={domain.label}", flush=True)

    references = {
        label: DomainIntensityReference(
            quantiles=torch.stack(items).mean(dim=0), volumes=len(items)
        )
        for label, items in accumulated.items()
    }
    if not references:
        raise ValueError("fit_image_intensity_baselines received no reference volumes.")
    base_provenance = dict(provenance or {})
    base_provenance["reference_volumes"] = sum(
        reference.volumes for reference in references.values()
    )
    return {
        mode: ImageIntensityBaseline(
            references=references,
            probabilities=probabilities,
            mode=mode,
            mask_threshold=mask_threshold,
            low_probability=low_probability,
            high_probability=high_probability,
            provenance=base_provenance,
        )
        for mode in INTENSITY_MODES
    }


def reference_volume_records(
    records: Sequence[Any],
    *,
    split_of_case: Mapping[str, str],
    training_splits: Sequence[str] = ("train",),
    per_domain: int = 8,
) -> list[Any]:
    """Preserve Gate-0 v1's first-N-per-domain training record selection."""

    chosen: dict[str, list[Any]] = {}
    allowed = set(training_splits)
    for record in records:
        if split_of_case.get(str(record.case_id)) not in allowed:
            continue
        bucket = chosen.setdefault(record.domain.label, [])
        if len(bucket) < per_domain:
            bucket.append(record)
    return [record for _, bucket in sorted(chosen.items()) for record in bucket]


__all__ = [
    "INTENSITY_BASELINE_CONTRACT_VERSION",
    "INTENSITY_MODES",
    "DomainIntensityReference",
    "ImageIntensityBaseline",
    "fit_image_intensity_baselines",
    "reference_volume_records",
]
