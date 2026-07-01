"""Convolutional autoencoder implementations for MRI tensors.

The default path uses 3D convolutions for full MRI volumes shaped
``(batch, channels, depth, height, width)``. Set ``spatial_dims=2`` for
slice-wise tensors shaped ``(batch, channels, height, width)``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
from torch import nn

from fieldbridge.data.domains import Domain
from fieldbridge.models.autoencoders.base import BaseDecoder, BaseEncoder

DomainBatch = Domain | Sequence[Domain]
SpatialDims = Literal[2, 3]


class CNNEncoder(BaseEncoder):
    """Small strided-convolution encoder for 2D slices or 3D volumes."""

    def __init__(
        self,
        *,
        in_channels: int = 1,
        latent_channels: int = 32,
        hidden_channels: Sequence[int] = (16, 32),
        spatial_dims: SpatialDims = 3,
        activation: str = "silu",
        use_norm: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = _positive_int(in_channels, "in_channels")
        self.spatial_dims = _validate_spatial_dims(spatial_dims)
        channels = _normalize_channels(hidden_channels)
        latent = _positive_int(latent_channels, "latent_channels")
        self.downsample_factor = 2 ** len(channels)

        conv = _conv_nd(self.spatial_dims)
        layers: list[nn.Module] = []
        current_channels = self.in_channels
        for next_channels in channels:
            layers.append(conv(current_channels, next_channels, kernel_size=3, stride=2, padding=1))
            if use_norm:
                layers.append(_norm(next_channels))
            layers.append(_activation(activation))
            current_channels = next_channels
        layers.append(conv(current_channels, latent, kernel_size=1))
        self.net = nn.Sequential(*layers)

    def encode(self, x: torch.Tensor, domain: DomainBatch) -> torch.Tensor:
        del domain
        _validate_input_tensor(
            x,
            spatial_dims=self.spatial_dims,
            channels=self.in_channels,
            downsample_factor=self.downsample_factor,
        )
        return self.net(x)


class CNNDecoder(BaseDecoder):
    """Mirror decoder for :class:`CNNEncoder` latent tensors."""

    def __init__(
        self,
        *,
        out_channels: int = 1,
        latent_channels: int = 32,
        hidden_channels: Sequence[int] = (16, 32),
        spatial_dims: SpatialDims = 3,
        activation: str = "silu",
        final_activation: str | None = None,
        use_norm: bool = True,
    ) -> None:
        super().__init__()
        out = _positive_int(out_channels, "out_channels")
        latent = _positive_int(latent_channels, "latent_channels")
        self.spatial_dims = _validate_spatial_dims(spatial_dims)
        channels = _normalize_channels(hidden_channels)

        conv = _conv_nd(self.spatial_dims)
        deconv = _conv_transpose_nd(self.spatial_dims)
        layers: list[nn.Module] = []
        current_channels = latent
        for next_channels in reversed(channels):
            layers.append(
                deconv(current_channels, next_channels, kernel_size=4, stride=2, padding=1)
            )
            if use_norm:
                layers.append(_norm(next_channels))
            layers.append(_activation(activation))
            current_channels = next_channels
        layers.append(conv(current_channels, out, kernel_size=1))
        if final_activation is not None:
            layers.append(_activation(final_activation))
        self.net = nn.Sequential(*layers)

    def decode(self, z: torch.Tensor, domain: DomainBatch) -> torch.Tensor:
        del domain
        expected_dims = self.spatial_dims + 2
        if z.ndim != expected_dims:
            raise ValueError(f"Expected {expected_dims}D latent tensor, got shape {tuple(z.shape)}.")
        return self.net(z)


def _validate_input_tensor(
    x: torch.Tensor,
    *,
    spatial_dims: int,
    channels: int,
    downsample_factor: int,
) -> None:
    expected_dims = spatial_dims + 2
    if x.ndim != expected_dims:
        raise ValueError(f"Expected {expected_dims}D input tensor, got shape {tuple(x.shape)}.")
    if int(x.shape[1]) != channels:
        raise ValueError(f"Expected {channels} input channels, got {int(x.shape[1])}.")
    if downsample_factor <= 1:
        return
    spatial_shape = tuple(int(dim) for dim in x.shape[-spatial_dims:])
    bad_dims = [dim for dim in spatial_shape if dim % downsample_factor != 0]
    if bad_dims:
        raise ValueError(
            "CNN autoencoder inputs must have spatial dimensions divisible by "
            f"{downsample_factor}; got {spatial_shape}."
        )


def _normalize_channels(channels: Sequence[int]) -> tuple[int, ...]:
    normalized = tuple(_positive_int(value, "hidden_channels") for value in channels)
    return normalized


def _validate_spatial_dims(spatial_dims: int) -> SpatialDims:
    if spatial_dims not in (2, 3):
        raise ValueError(f"spatial_dims must be 2 or 3, got {spatial_dims}.")
    return spatial_dims  # type: ignore[return-value]


def _positive_int(value: int, name: str) -> int:
    integer = int(value)
    if integer <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")
    return integer


def _conv_nd(spatial_dims: int) -> type[nn.Conv2d] | type[nn.Conv3d]:
    return nn.Conv3d if spatial_dims == 3 else nn.Conv2d


def _conv_transpose_nd(spatial_dims: int) -> type[nn.ConvTranspose2d] | type[nn.ConvTranspose3d]:
    return nn.ConvTranspose3d if spatial_dims == 3 else nn.ConvTranspose2d


def _norm(channels: int) -> nn.GroupNorm:
    groups = min(8, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


def _activation(name: str) -> nn.Module:
    normalized = name.lower().replace("-", "_")
    if normalized == "relu":
        return nn.ReLU(inplace=True)
    if normalized == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.2, inplace=True)
    if normalized == "gelu":
        return nn.GELU()
    if normalized == "silu":
        return nn.SiLU(inplace=True)
    if normalized == "sigmoid":
        return nn.Sigmoid()
    if normalized == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation {name!r}.")
