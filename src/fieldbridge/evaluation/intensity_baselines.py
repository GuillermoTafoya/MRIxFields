"""Image-space intensity baselines: the broad form of the shortcut hypothesis.

The latent affine baseline (``models/translators/affine_baseline.py``) tests one narrow
mechanism. Its closed form is exact only under a 1-D Gaussian marginal approximation per
channel, while the real latent channels are spatial, non-Gaussian and decoded through a
nonlinearity. That asymmetry matters for how its result may be read:

- latent affine ~= SB  -> strong evidence of an intensity shortcut.
- latent affine != SB  -> proves nothing about SB having learned field physics. It rules out
  a per-channel latent affine and nothing more.

These baselines close that gap by testing the shortcut where it is actually visible: on the
decoded image. Both are applied to the IDENTITY reconstruction ``decode(z_source)``, so they
cost no extra decode — they are post-hoc intensity transforms of a volume the gate already
has, and they are exactly what "the model only fixed the brightness" would predict.

- ``robust_affine``: match the source reconstruction's [p_low, p_high] window to the target
  domain's, inside a foreground mask. The image-space analogue of the latent affine.
- ``histogram``: full monotone CDF match onto the target domain's foreground quantiles. This
  is the strongest purely photometric map available — it reproduces the target domain's entire
  intensity distribution while leaving spatial structure untouched. Anything a model gains
  over this is, by construction, not photometric.

References are fitted from TRAINING subjects only.
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
    """One domain's foreground intensity distribution, as quantiles on a fixed probability grid."""

    quantiles: torch.Tensor  # (K,)
    volumes: int

    def to_dict(self) -> dict[str, Any]:
        return {"quantiles": [float(v) for v in self.quantiles], "volumes": int(self.volumes)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DomainIntensityReference":
        return cls(
            quantiles=torch.tensor([float(v) for v in data["quantiles"]], dtype=torch.float32),
            volumes=int(data["volumes"]),
        )


def _foreground_values(image: torch.Tensor, mask_threshold: float) -> torch.Tensor:
    flat = image.reshape(-1).to(torch.float32)
    selected = flat[flat > mask_threshold]
    return selected if selected.numel() else flat


def _quantiles_of(
    image: torch.Tensor, probabilities: torch.Tensor, mask_threshold: float
) -> torch.Tensor:
    values = _foreground_values(image, mask_threshold)
    # torch.quantile caps its input size; a sorted subsample is exact enough for a percentile
    # grid and keeps a 364^3 volume affordable.
    limit = 4_000_000
    if values.numel() > limit:
        step = values.numel() // limit + 1
        values = values[::step]
    return torch.quantile(values, probabilities.to(values.device))


@dataclass(frozen=True, slots=True)
class ImageIntensityBaseline:
    """Per-domain intensity references plus the monotone map they induce."""

    references: Mapping[str, DomainIntensityReference]
    probabilities: torch.Tensor
    mode: IntensityMode
    mask_threshold: float = 0.0
    low_probability: float = 0.01
    high_probability: float = 0.99
    provenance: Mapping[str, Any] = None  # type: ignore[assignment]

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
        """Map ``image`` onto the target domain's intensity distribution. Shape is preserved."""

        reference = self._reference_for(target)
        probabilities = self.probabilities.to(image.device)
        source_quantiles = _quantiles_of(image, probabilities, self.mask_threshold)
        target_quantiles = reference.quantiles.to(image.device, image.dtype)

        if self.mode == "robust_affine":
            low_index = int(torch.argmin((probabilities - self.low_probability).abs()))
            high_index = int(torch.argmin((probabilities - self.high_probability).abs()))
            source_low, source_high = source_quantiles[low_index], source_quantiles[high_index]
            target_low, target_high = target_quantiles[low_index], target_quantiles[high_index]
            span = source_high - source_low
            if float(span.abs()) < 1e-8:
                return image
            scale = (target_high - target_low) / span
            return (image - source_low) * scale + target_low

        # histogram: piecewise-linear CDF match through the shared probability grid.
        flat = image.reshape(-1).to(torch.float32)
        mapped = _interpolate_monotone(flat, source_quantiles, target_quantiles)
        return mapped.reshape(image.shape).to(image.dtype)

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": INTENSITY_BASELINE_CONTRACT_VERSION,
            "mode": self.mode,
            "probabilities": [float(p) for p in self.probabilities],
            "mask_threshold": float(self.mask_threshold),
            "low_probability": float(self.low_probability),
            "high_probability": float(self.high_probability),
            "provenance": dict(self.provenance),
            "references": {k: v.to_dict() for k, v in sorted(self.references.items())},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, mode: IntensityMode | None = None) -> "ImageIntensityBaseline":
        version = str(data.get("contract_version", ""))
        if version != INTENSITY_BASELINE_CONTRACT_VERSION:
            raise ValueError(
                f"Intensity baseline contract mismatch: file has {version!r}, code expects "
                f"{INTENSITY_BASELINE_CONTRACT_VERSION!r}."
            )
        return cls(
            references={
                k: DomainIntensityReference.from_dict(v) for k, v in data["references"].items()
            },
            probabilities=torch.tensor(
                [float(p) for p in data["probabilities"]], dtype=torch.float32
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
        temporary.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(target)
        return target

    @classmethod
    def load(cls, path: str | Path, *, mode: IntensityMode | None = None) -> "ImageIntensityBaseline":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")), mode=mode)


def _interpolate_monotone(
    values: torch.Tensor, source_grid: torch.Tensor, target_grid: torch.Tensor
) -> torch.Tensor:
    """Piecewise-linear interpolation of ``values`` through the map source_grid -> target_grid."""

    source_grid = source_grid.to(values.dtype)
    target_grid = target_grid.to(values.dtype)
    indices = torch.searchsorted(source_grid.contiguous(), values.contiguous()).clamp(
        1, source_grid.numel() - 1
    )
    left, right = source_grid[indices - 1], source_grid[indices]
    left_target, right_target = target_grid[indices - 1], target_grid[indices]
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
    """Fit both intensity baselines in one pass over the reference volumes.

    ``volumes`` yields ``(volume, domain)`` for TRAINING subjects only. Per-domain quantiles are
    averaged across the reference volumes of that domain.
    """

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
    base_provenance["reference_volumes"] = sum(r.volumes for r in references.values())
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
    """Pick up to ``per_domain`` TRAINING-split records per domain to fit the references on.

    Fitting on training subjects only is the point: an intensity reference built from the
    evaluated traveller would hand the baseline the answer.
    """

    chosen: dict[str, list[Any]] = {}
    for record in records:
        if split_of_case.get(str(record.case_id)) not in set(training_splits):
            continue
        label = record.domain.label
        bucket = chosen.setdefault(label, [])
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
