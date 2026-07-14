"""Synthetic field-degradation utilities for pseudo-pair pretraining.

These transforms are deterministic when supplied a ``torch.Generator``. They are
not a physical MRI simulator; they provide low-field-like corruption for a
deterministic conditional U-Net pseudo-pair baseline.
"""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from fieldbridge.data.domains import FIELD_MAX_T, FIELD_STRENGTHS_T, Domain


def degradation_strength(high_field: float | Domain, low_field: float | Domain) -> float:
    """Map a high-to-low field pair to ``[0, 1]`` synthetic degradation strength."""

    high = _field_strength_t(high_field)
    low = _field_strength_t(low_field)
    if high <= low:
        return 0.0
    max_ratio = math.log(FIELD_MAX_T / min(FIELD_STRENGTHS_T))
    return float(max(0.0, min(1.0, math.log(high / low) / max_ratio)))


def downsample_then_upsample(x: torch.Tensor, scale_factor: float) -> torch.Tensor:
    """Downsample spatial dimensions and upsample back to the original shape."""

    spatial_dims = _validate_image_tensor(x)
    scale = float(scale_factor)
    if scale <= 0.0:
        raise ValueError(f"scale_factor must be positive, got {scale_factor}.")
    if scale >= 1.0:
        return x.clone()

    spatial_shape = tuple(int(dim) for dim in x.shape[-spatial_dims:])
    low_shape = tuple(max(1, int(round(dim * scale))) for dim in spatial_shape)
    mode = "bilinear" if spatial_dims == 2 else "trilinear"
    low = F.interpolate(x, size=low_shape, mode=mode, align_corners=False)
    return F.interpolate(low, size=spatial_shape, mode=mode, align_corners=False)


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Apply a small separable Gaussian blur to 2D or 3D tensors."""

    spatial_dims = _validate_image_tensor(x)
    sigma_value = float(sigma)
    if sigma_value <= 0.0:
        return x.clone()
    radius = max(1, int(math.ceil(3.0 * sigma_value)))
    kernel = _gaussian_kernel1d(radius, sigma_value, dtype=x.dtype, device=x.device)
    output = x
    for dim in range(spatial_dims):
        output = _apply_separable_kernel(output, kernel, spatial_dims=spatial_dims, dim=dim)
    return output


def additive_gaussian_noise(
    x: torch.Tensor,
    std: float,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Add Gaussian noise with a standard deviation in tensor intensity units."""

    _validate_image_tensor(x)
    std_value = float(std)
    if std_value <= 0.0:
        return x.clone()
    noise = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)
    return x + std_value * noise


def multiplicative_smooth_bias_field(
    x: torch.Tensor,
    strength: float,
    *,
    generator: torch.Generator | None = None,
    control_points: int = 4,
) -> torch.Tensor:
    """Approximate coil/intensity bias with a smooth multiplicative random field."""

    spatial_dims = _validate_image_tensor(x)
    amount = float(strength)
    if amount <= 0.0:
        return x.clone()
    low_shape = tuple(max(2, min(int(control_points), int(dim))) for dim in x.shape[-spatial_dims:])
    field_shape = (int(x.shape[0]), 1, *low_shape)
    field = torch.randn(field_shape, dtype=x.dtype, device=x.device, generator=generator)
    mode = "bilinear" if spatial_dims == 2 else "trilinear"
    field = F.interpolate(field, size=x.shape[-spatial_dims:], mode=mode, align_corners=False)
    reduce_dims = tuple(range(2, field.ndim))
    field = field - field.mean(dim=reduce_dims, keepdim=True)
    max_abs = field.abs().amax(dim=reduce_dims, keepdim=True).clamp_min(1e-6)
    bias = (1.0 + amount * field / max_abs).clamp_min(0.05)
    return x * bias


def intensity_compression(x: torch.Tensor, amount: float) -> torch.Tensor:
    """Apply a gamma-like dynamic-range compression around zero."""

    _validate_image_tensor(x)
    value = float(amount)
    if value <= 0.0:
        return x.clone()
    gamma = 1.0 + value
    return torch.sign(x) * torch.abs(x).pow(gamma)


def low_pass_filter(x: torch.Tensor, cutoff_fraction: float) -> torch.Tensor:
    """Apply a simple FFT low-pass mask over spatial dimensions."""

    spatial_dims = _validate_image_tensor(x)
    cutoff = float(cutoff_fraction)
    if cutoff <= 0.0 or cutoff > 1.0:
        raise ValueError(f"cutoff_fraction must be in (0, 1], got {cutoff_fraction}.")
    if cutoff >= 1.0:
        return x.clone()

    spatial_shape = tuple(int(dim) for dim in x.shape[-spatial_dims:])
    grids = torch.meshgrid(
        *[
            torch.fft.fftfreq(dim, dtype=x.dtype, device=x.device) / 0.5
            for dim in spatial_shape
        ],
        indexing="ij",
    )
    radius = torch.zeros(spatial_shape, dtype=x.dtype, device=x.device)
    for grid in grids:
        radius = radius + grid.pow(2)
    radius = torch.sqrt(radius / spatial_dims)
    mask = (radius <= cutoff).to(dtype=x.dtype).reshape((1, 1, *spatial_shape))
    spectrum = torch.fft.fftn(x, dim=tuple(range(x.ndim - spatial_dims, x.ndim)))
    filtered = torch.fft.ifftn(spectrum * mask, dim=tuple(range(x.ndim - spatial_dims, x.ndim)))
    return filtered.real.to(dtype=x.dtype)


def compose_degradation(
    x: torch.Tensor,
    strength: float,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Compose blur, resolution loss, bias, noise, and intensity compression."""

    _validate_image_tensor(x)
    value = float(max(0.0, min(1.0, strength)))
    if value == 0.0:
        return x.clone()

    output = downsample_then_upsample(x, scale_factor=max(0.25, 1.0 - 0.65 * value))
    output = gaussian_blur(output, sigma=0.15 + 1.25 * value)
    output = low_pass_filter(output, cutoff_fraction=max(0.30, 1.0 - 0.55 * value))
    output = multiplicative_smooth_bias_field(output, 0.18 * value, generator=generator)
    scale = x.detach().float().std().clamp_min(1e-6).to(dtype=x.dtype, device=x.device)
    noise_std = float(0.06 * value * scale.item())
    output = additive_gaussian_noise(output, noise_std, generator=generator)
    return intensity_compression(output, amount=0.35 * value)


def _validate_image_tensor(x: torch.Tensor) -> int:
    if x.ndim not in (4, 5):
        raise ValueError(f"Expected a 4D or 5D image tensor, got shape {tuple(x.shape)}.")
    if int(x.shape[1]) <= 0:
        raise ValueError("Image tensor must contain at least one channel.")
    return x.ndim - 2


def _field_strength_t(value: float | Domain) -> float:
    return value.field_strength_t if isinstance(value, Domain) else float(value)


def _gaussian_kernel1d(
    radius: int,
    sigma: float,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    coords = torch.arange(-radius, radius + 1, dtype=dtype, device=device)
    kernel = torch.exp(-0.5 * (coords / sigma).pow(2))
    return kernel / kernel.sum().clamp_min(torch.finfo(dtype).eps)


def _apply_separable_kernel(
    x: torch.Tensor,
    kernel: torch.Tensor,
    *,
    spatial_dims: int,
    dim: int,
) -> torch.Tensor:
    channels = int(x.shape[1])
    radius = int(kernel.numel() // 2)
    if spatial_dims == 2:
        shape = (channels, 1, 1, 1)
        weight = kernel.reshape(1, 1, -1, 1) if dim == 0 else kernel.reshape(1, 1, 1, -1)
        padding = (radius, 0) if dim == 0 else (0, radius)
        return F.conv2d(x, weight.repeat(shape), padding=padding, groups=channels)

    shape = (channels, 1, 1, 1, 1)
    if dim == 0:
        weight = kernel.reshape(1, 1, -1, 1, 1)
        padding = (radius, 0, 0)
    elif dim == 1:
        weight = kernel.reshape(1, 1, 1, -1, 1)
        padding = (0, radius, 0)
    else:
        weight = kernel.reshape(1, 1, 1, 1, -1)
        padding = (0, 0, radius)
    return F.conv3d(x, weight.repeat(shape), padding=padding, groups=channels)
