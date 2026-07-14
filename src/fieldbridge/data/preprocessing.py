"""Official-aligned slice preprocessing for pseudo-pair training."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import torch
from torch.nn import functional as F

NormalizationMode = Literal["official_01", "none"]
ModelRange = Literal["zero_one", "minus_one_one"]
ResizeMode = Literal["native", "fit_pad"]


@dataclass(frozen=True, slots=True)
class SlicePreprocessingSpec:
    slice_start: int = 72
    slice_end: int = 292
    slices_per_volume: int | None = 32
    normalization: NormalizationMode = "official_01"
    model_range: ModelRange = "minus_one_one"
    resize_mode: ResizeMode = "fit_pad"
    output_height: int | None = 256
    output_width: int | None = 320

    @classmethod
    def from_mapping(cls, data: object) -> "SlicePreprocessingSpec":
        if not isinstance(data, dict):
            return cls()
        defaults = cls()
        return cls(
            slice_start=int(data.get("slice_start", defaults.slice_start)),
            slice_end=int(data.get("slice_end", defaults.slice_end)),
            slices_per_volume=None
            if data.get("slices_per_volume", defaults.slices_per_volume) is None
            else int(data.get("slices_per_volume", defaults.slices_per_volume)),
            normalization=data.get("normalization", defaults.normalization),
            model_range=data.get("model_range", defaults.model_range),
            resize_mode=data.get("resize_mode", defaults.resize_mode),
            output_height=None
            if data.get("output_height", defaults.output_height) is None
            else int(data.get("output_height", defaults.output_height)),
            output_width=None
            if data.get("output_width", defaults.output_width) is None
            else int(data.get("output_width", defaults.output_width)),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "slice_start": self.slice_start,
            "slice_end": self.slice_end,
            "slices_per_volume": self.slices_per_volume,
            "normalization": self.normalization,
            "model_range": self.model_range,
            "resize_mode": self.resize_mode,
            "output_height": self.output_height,
            "output_width": self.output_width,
        }


@dataclass(frozen=True, slots=True)
class SliceGeometry:
    """Geometry metadata sufficient to remove padding and resize back for display."""

    slice_index: int
    original_height: int
    original_width: int
    resized_height: int
    resized_width: int
    output_height: int
    output_width: int
    pad_top: int = 0
    pad_bottom: int = 0
    pad_left: int = 0
    pad_right: int = 0
    resize_mode: ResizeMode = "native"
    model_range: ModelRange = "minus_one_one"
    normalization: NormalizationMode = "official_01"

    def to_dict(self) -> dict[str, object]:
        return {
            "slice_index": self.slice_index,
            "original_height": self.original_height,
            "original_width": self.original_width,
            "resized_height": self.resized_height,
            "resized_width": self.resized_width,
            "output_height": self.output_height,
            "output_width": self.output_width,
            "pad_top": self.pad_top,
            "pad_bottom": self.pad_bottom,
            "pad_left": self.pad_left,
            "pad_right": self.pad_right,
            "resize_mode": self.resize_mode,
            "model_range": self.model_range,
            "normalization": self.normalization,
        }


@dataclass(frozen=True, slots=True)
class PreprocessedSliceBatch:
    image: torch.Tensor
    geometry: tuple[SliceGeometry, ...]


def selected_slice_indices(
    spec: SlicePreprocessingSpec,
    *,
    depth: int | None = None,
) -> tuple[int, ...]:
    """Select approximately uniformly spaced indices in ``[slice_start, slice_end)``."""

    _validate_spec(spec)
    if depth is not None and spec.slice_end > int(depth):
        raise ValueError(
            f"Slice range [{spec.slice_start}, {spec.slice_end}) exceeds volume depth {depth}."
        )
    total = spec.slice_end - spec.slice_start
    count = total if spec.slices_per_volume is None else min(int(spec.slices_per_volume), total)
    if count <= 0:
        raise ValueError("slices_per_volume must select at least one slice.")
    if count == total:
        return tuple(range(spec.slice_start, spec.slice_end))
    if count == 1:
        return (spec.slice_start + (total - 1) // 2,)
    step = (total - 1) / float(count - 1)
    indices = [spec.slice_start + int(round(i * step)) for i in range(count)]
    return tuple(_dedupe_uniform(indices, range(spec.slice_start, spec.slice_end), count))


def preprocess_volume_slices(volume: torch.Tensor, spec: SlicePreprocessingSpec) -> PreprocessedSliceBatch:
    """Extract and preprocess axial slices from a ``(C, D, H, W)`` volume."""

    _validate_volume(volume)
    indices = selected_slice_indices(spec, depth=int(volume.shape[1]))
    images: list[torch.Tensor] = []
    geometry: list[SliceGeometry] = []
    for slice_index in indices:
        image, meta = preprocess_volume_slice(volume, slice_index, spec)
        images.append(image)
        geometry.append(meta)
    return PreprocessedSliceBatch(image=torch.stack(images, dim=0), geometry=tuple(geometry))


def preprocess_volume_slice(
    volume: torch.Tensor,
    slice_index: int,
    spec: SlicePreprocessingSpec,
    *,
    apply_model_range: bool = True,
) -> tuple[torch.Tensor, SliceGeometry]:
    """Preprocess one axial slice from a ``(C, D, H, W)`` volume."""

    _validate_volume(volume)
    _validate_spec(spec)
    depth = int(volume.shape[1])
    index = int(slice_index)
    if index < spec.slice_start or index >= spec.slice_end:
        raise ValueError(f"slice_index {index} is outside [{spec.slice_start}, {spec.slice_end}).")
    if index >= depth:
        raise ValueError(f"slice_index {index} exceeds volume depth {depth}.")
    image = volume[:, index, :, :].detach().clone().to(dtype=torch.float32)
    _validate_intensity(image, spec, context=f"slice {index}")
    image, geometry = _resize_slice(image, spec, slice_index=index)
    if apply_model_range:
        image = to_model_range(image, spec.model_range)
    return image, geometry


def to_model_range(image: torch.Tensor, model_range: ModelRange) -> torch.Tensor:
    """Map official ``[0, 1]`` values to the configured model range."""

    if model_range == "zero_one":
        return image
    if model_range == "minus_one_one":
        return image * 2.0 - 1.0
    raise ValueError(f"Unsupported model_range {model_range!r}.")


def from_model_range(image: torch.Tensor, model_range: ModelRange) -> torch.Tensor:
    """Map model tensors back to official ``[0, 1]`` intensity units."""

    if model_range == "zero_one":
        return image
    if model_range == "minus_one_one":
        return (image + 1.0) * 0.5
    raise ValueError(f"Unsupported model_range {model_range!r}.")


def reverse_preprocessed_slice(image: torch.Tensor, geometry: SliceGeometry) -> torch.Tensor:
    """Remove fit/pad geometry and resize a slice back to its original shape."""

    if image.ndim != 3:
        raise ValueError(f"reverse_preprocessed_slice expects (C,H,W), got {tuple(image.shape)}.")
    cropped = image[
        :,
        geometry.pad_top : geometry.output_height - geometry.pad_bottom,
        geometry.pad_left : geometry.output_width - geometry.pad_right,
    ]
    if (geometry.resized_height, geometry.resized_width) != (
        geometry.original_height,
        geometry.original_width,
    ):
        cropped = F.interpolate(
            cropped.unsqueeze(0),
            size=(geometry.original_height, geometry.original_width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    return cropped


def _resize_slice(image: torch.Tensor, spec: SlicePreprocessingSpec, *, slice_index: int) -> tuple[torch.Tensor, SliceGeometry]:
    channels, height, width = (int(dim) for dim in image.shape)
    del channels
    if spec.resize_mode == "native":
        return image, SliceGeometry(
            slice_index=slice_index,
            original_height=height,
            original_width=width,
            resized_height=height,
            resized_width=width,
            output_height=height,
            output_width=width,
            resize_mode=spec.resize_mode,
            model_range=spec.model_range,
            normalization=spec.normalization,
        )
    if spec.resize_mode != "fit_pad":
        raise ValueError(f"resize_mode must be 'native' or 'fit_pad', got {spec.resize_mode!r}.")
    if spec.output_height is None or spec.output_width is None:
        raise ValueError("fit_pad requires output_height and output_width.")
    output_height = int(spec.output_height)
    output_width = int(spec.output_width)
    if output_height <= 0 or output_width <= 0:
        raise ValueError("output_height and output_width must be positive.")
    scale = min(output_height / float(height), output_width / float(width))
    resized_height = max(1, min(output_height, int(round(height * scale))))
    resized_width = max(1, min(output_width, int(round(width * scale))))
    resized = F.interpolate(
        image.unsqueeze(0),
        size=(resized_height, resized_width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    pad_top = (output_height - resized_height) // 2
    pad_bottom = output_height - resized_height - pad_top
    pad_left = (output_width - resized_width) // 2
    pad_right = output_width - resized_width - pad_left
    padded = F.pad(resized, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)
    return padded, SliceGeometry(
        slice_index=slice_index,
        original_height=height,
        original_width=width,
        resized_height=resized_height,
        resized_width=resized_width,
        output_height=output_height,
        output_width=output_width,
        pad_top=pad_top,
        pad_bottom=pad_bottom,
        pad_left=pad_left,
        pad_right=pad_right,
        resize_mode=spec.resize_mode,
        model_range=spec.model_range,
        normalization=spec.normalization,
    )


def _validate_spec(spec: SlicePreprocessingSpec) -> None:
    if spec.slice_start < 0:
        raise ValueError("slice_start must be non-negative.")
    if spec.slice_end <= spec.slice_start:
        raise ValueError("slice_end must be greater than slice_start.")
    if spec.slices_per_volume is not None and spec.slices_per_volume <= 0:
        raise ValueError("slices_per_volume must be positive when provided.")
    if spec.normalization not in ("official_01", "none"):
        raise ValueError(f"Unsupported normalization {spec.normalization!r}.")
    if spec.model_range not in ("zero_one", "minus_one_one"):
        raise ValueError(f"Unsupported model_range {spec.model_range!r}.")
    if spec.resize_mode not in ("native", "fit_pad"):
        raise ValueError(f"Unsupported resize_mode {spec.resize_mode!r}.")


def _validate_volume(volume: torch.Tensor) -> None:
    if volume.ndim != 4:
        raise ValueError(f"Expected volume tensor shaped (C,D,H,W), got {tuple(volume.shape)}.")
    if any(int(dim) <= 0 for dim in volume.shape):
        raise ValueError(f"Volume dimensions must be positive, got {tuple(volume.shape)}.")
    if not torch.isfinite(volume).all():
        raise ValueError("Volume contains non-finite values.")


def _validate_intensity(image: torch.Tensor, spec: SlicePreprocessingSpec, *, context: str) -> None:
    if not torch.isfinite(image).all():
        raise ValueError(f"{context} contains non-finite values.")
    if spec.normalization != "official_01":
        return
    minimum = float(image.min().item())
    maximum = float(image.max().item())
    hard_tolerance = 1e-4
    if minimum < -hard_tolerance or maximum > 1.0 + hard_tolerance:
        raise ValueError(
            f"{context} uses normalization='official_01' but intensities are outside "
            f"the released [0, 1] range: min={minimum:.6g}, max={maximum:.6g}."
        )
    if minimum < 0.0 or maximum > 1.0:
        warnings.warn(
            f"{context} has minor numeric drift outside [0, 1]: min={minimum:.6g}, max={maximum:.6g}.",
            RuntimeWarning,
            stacklevel=2,
        )


def _dedupe_uniform(indices: list[int], candidates: range, count: int) -> list[int]:
    ordered = sorted(dict.fromkeys(indices))
    if len(ordered) == count:
        return ordered
    for candidate in candidates:
        if candidate not in ordered:
            ordered.append(candidate)
            ordered.sort()
            if len(ordered) == count:
                return ordered
    return ordered
