"""Small tensor transforms for synthetic and injected-loader datasets."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import torch

TensorTransform = Callable[[torch.Tensor], torch.Tensor]


def identity_transform(image: torch.Tensor) -> torch.Tensor:
    return image


def normalize_zero_mean_unit_variance(image: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return (image - image.mean()) / (image.std().clamp_min(eps))


def compose(transforms: Iterable[TensorTransform]) -> TensorTransform:
    ordered = tuple(transforms)

    def _apply(image: torch.Tensor) -> torch.Tensor:
        value = image
        for transform in ordered:
            value = transform(value)
        return value

    return _apply

