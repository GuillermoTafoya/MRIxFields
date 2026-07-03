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


def compose(transforms: Iterable[TensorTransform]) -> TensorTransform:
    ordered = tuple(transforms)

    def _apply(image: torch.Tensor) -> torch.Tensor:
        value = image
        for transform in ordered:
            value = transform(value)
        return value

    return _apply

