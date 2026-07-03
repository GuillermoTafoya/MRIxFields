"""Small tensor transforms for synthetic and injected-loader datasets."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import torch

TensorTransform = Callable[[torch.Tensor], torch.Tensor]


def identity_transform(image: torch.Tensor) -> torch.Tensor:
    return image


def normalize_zero_mean_unit_variance(image: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return (image - image.mean()) / (image.std().clamp_min(eps))


_QUANTILE_ELEMENT_LIMIT = 2**24  # torch.quantile's practical size limit on some backends


def normalize_percentile_clip_to_unit_range(
    image: torch.Tensor,
    *,
    lower_percentile: float = 0.5,
    upper_percentile: float = 99.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Clip to [lower_percentile, upper_percentile] then affine-map to [-1, 1].

    Percentile clip rather than raw min-max: MRI intensity has a long right tail (a
    handful of very bright voxels — vessels, motion/susceptibility artifacts) that would
    otherwise dominate the scale and compress the diagnostically-relevant anatomy into a
    narrow sub-range. The output range must be exactly [-1, 1] to match the decoder's
    Tanh() output and the un-normalized `lpips_loss` net (see training/losses.py).
    """

    flat = image.reshape(-1)
    if flat.numel() > _QUANTILE_ELEMENT_LIMIT:
        flat = flat[:: flat.numel() // _QUANTILE_ELEMENT_LIMIT + 1]
    quantiles = torch.quantile(
        flat.float(),
        torch.tensor([lower_percentile / 100.0, upper_percentile / 100.0], dtype=flat.dtype),
    )
    lo, hi = quantiles[0], quantiles[1]
    if (hi - lo) <= eps:
        raise ValueError(
            f"normalize_percentile_clip_to_unit_range: degenerate image, "
            f"percentile range [{lo.item()}, {hi.item()}] is not wide enough to normalize."
        )
    clamped = image.clamp(min=lo, max=hi)
    return 2.0 * (clamped - lo) / (hi - lo) - 1.0


def random_crop(image: torch.Tensor, *, patch_size: Iterable[int]) -> torch.Tensor:
    """Random crop of `patch_size` over the trailing spatial dims.

    Needed for full-resolution 3D volumes (e.g. 364x436x364): decoding back toward full
    resolution with enough latent channels for the diffuser allocates tens of GB per
    intermediate activation, OOMing on essentially any GPU. Patch-based training (random
    crops, not the full volume) is the standard fix for 3D medical imaging at this
    scale — this is not optional once spatial_dims=3 is combined with a real volume's
    resolution, regardless of how much compute budget is available.
    """

    patch = tuple(int(p) for p in patch_size)
    spatial_shape = tuple(image.shape[-len(patch) :])
    starts = []
    for size, size_patch in zip(spatial_shape, patch):
        if size_patch > size:
            raise ValueError(f"patch_size {patch} exceeds image spatial shape {spatial_shape}.")
        starts.append(0 if size_patch == size else int(torch.randint(0, size - size_patch + 1, (1,)).item()))
    slices = tuple(slice(start, start + size_patch) for start, size_patch in zip(starts, patch))
    return image[(..., *slices)]


def compose(transforms: Iterable[TensorTransform]) -> TensorTransform:
    ordered = tuple(transforms)

    def _apply(image: torch.Tensor) -> torch.Tensor:
        value = image
        for transform in ordered:
            value = transform(value)
        return value

    return _apply

