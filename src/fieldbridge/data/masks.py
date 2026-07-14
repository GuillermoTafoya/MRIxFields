"""Lightweight brain-mask cleanup helpers."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def threshold_mask(x: torch.Tensor, threshold: float | None = None) -> torch.Tensor:
    """Return a binary mask from an image tensor, preserving shape."""

    _validate_mask_tensor(x)
    value = 0.0 if threshold is None else float(threshold)
    return (x > value).to(dtype=x.dtype)


def binary_dilation(
    mask: torch.Tensor,
    *,
    kernel_size: int = 3,
    iterations: int = 1,
) -> torch.Tensor:
    """Binary dilation implemented with max-pooling."""

    result = _as_binary(mask)
    kernel = _validate_kernel_size(kernel_size)
    for _ in range(_validate_iterations(iterations)):
        result = _max_pool(result, kernel)
    return _as_binary(result)


def binary_closing(
    mask: torch.Tensor,
    *,
    kernel_size: int = 3,
    iterations: int = 1,
) -> torch.Tensor:
    """Approximate hole filling via dilation followed by erosion."""

    result = _as_binary(mask)
    kernel = _validate_kernel_size(kernel_size)
    count = _validate_iterations(iterations)
    for _ in range(count):
        result = _max_pool(result, kernel)
    for _ in range(count):
        result = 1.0 - _max_pool(1.0 - result, kernel)
    return _as_binary(result)


def fill_holes_2d(mask: torch.Tensor, *, kernel_size: int = 3, iterations: int = 1) -> torch.Tensor:
    """Fill small 2D holes using binary closing."""

    if mask.ndim != 4:
        raise ValueError(f"fill_holes_2d expects (B, C, H, W), got {tuple(mask.shape)}.")
    return binary_closing(mask, kernel_size=kernel_size, iterations=iterations)


def clean_brain_mask(
    x: torch.Tensor,
    *,
    threshold: float | None = None,
    kernel_size: int = 3,
    iterations: int = 1,
) -> torch.Tensor:
    """Threshold and close a small 2D/3D brain mask approximation."""

    mask = threshold_mask(x, threshold=threshold)
    return binary_closing(mask, kernel_size=kernel_size, iterations=iterations)


def _validate_mask_tensor(x: torch.Tensor) -> None:
    if x.ndim not in (4, 5):
        raise ValueError(f"Expected a 4D or 5D tensor, got shape {tuple(x.shape)}.")


def _as_binary(mask: torch.Tensor) -> torch.Tensor:
    _validate_mask_tensor(mask)
    return (mask > 0.5).to(dtype=mask.dtype, device=mask.device)


def _validate_kernel_size(kernel_size: int) -> int:
    kernel = int(kernel_size)
    if kernel <= 0 or kernel % 2 == 0:
        raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}.")
    return kernel


def _validate_iterations(iterations: int) -> int:
    count = int(iterations)
    if count < 1:
        raise ValueError(f"iterations must be positive, got {iterations}.")
    return count


def _max_pool(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
    padding = kernel_size // 2
    if mask.ndim == 4:
        return F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=padding)
    return F.max_pool3d(mask, kernel_size=kernel_size, stride=1, padding=padding)
