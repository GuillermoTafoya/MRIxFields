"""KL-regularized VAE encoder/decoder for Etapa 1.

Supports 2D slices (spatial_dims=2, the original default) and full 3D volumes
(spatial_dims=3) — 3D support added so real training can match the CNN autoencoder's
already-proven 3D-volume path, since the real manifest's NIfTI files are full volumes
and no slice-extraction step exists in this pipeline. This is a deliberate, confirmed
reversal of fase-b-vae.md's original "2D estricto, never 3D conv" rule for this
component (compute cost was explicitly waved off).

Blind to (field, contrast) — conditioning on field strength does NOT happen here. The
conditional diffuser (models/diffusion/) is where field-strength conditioning is
injected, operating on this encoder's latent output. See docs/plans/fase-b-vae.md and
the Etapa 1 v2 plan (VAE + conditional latent diffuser) for the full design.
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

_DOWNSAMPLE_FACTOR = 4  # 2 stride-2 blocks


class KLVAEEncoder(BaseEncoder):
    """Strided-conv encoder producing a (mean, logvar) latent distribution."""

    def __init__(
        self,
        *,
        in_channels: int = 1,
        base_channels: int = 32,
        latent_channels: int = 128,
        spatial_dims: SpatialDims = 2,
        activation: str = "silu",
        use_norm: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = _positive_int(in_channels, "in_channels")
        self.latent_channels = _positive_int(latent_channels, "latent_channels")
        self.spatial_dims = _validate_spatial_dims(spatial_dims)
        base = _positive_int(base_channels, "base_channels")
        self.downsample_factor = _DOWNSAMPLE_FACTOR

        conv = _conv_nd(self.spatial_dims)
        self.stem = conv(self.in_channels, base, kernel_size=3, stride=1, padding=1)
        self.down1 = _down_block(base, base * 2, spatial_dims=self.spatial_dims, activation=activation, use_norm=use_norm)
        self.down2 = _down_block(
            base * 2, base * 4, spatial_dims=self.spatial_dims, activation=activation, use_norm=use_norm
        )
        self.to_dist = conv(base * 4, 2 * self.latent_channels, kernel_size=1)

    def encode_dist(
        self, x: torch.Tensor, domain: DomainBatch | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del domain
        _validate_input_tensor(
            x, spatial_dims=self.spatial_dims, channels=self.in_channels, downsample_factor=self.downsample_factor
        )
        h = self.stem(x)
        h = self.down1(h)
        h = self.down2(h)
        mean, logvar = self.to_dist(h).chunk(2, dim=1)
        return mean, logvar

    def encode(self, x: torch.Tensor, domain: DomainBatch) -> torch.Tensor:
        mean, logvar = self.encode_dist(x, domain)
        eps = torch.randn_like(mean)
        return mean + eps * torch.exp(0.5 * logvar)


class KLVAEDecoder(BaseDecoder):
    """Mirror decoder for `KLVAEEncoder` latents. Ends in an unconditional `Tanh()`."""

    def __init__(
        self,
        *,
        out_channels: int = 1,
        base_channels: int = 32,
        latent_channels: int = 128,
        spatial_dims: SpatialDims = 2,
        activation: str = "silu",
        use_norm: bool = True,
    ) -> None:
        super().__init__()
        out = _positive_int(out_channels, "out_channels")
        self.latent_channels = _positive_int(latent_channels, "latent_channels")
        self.spatial_dims = _validate_spatial_dims(spatial_dims)
        base = _positive_int(base_channels, "base_channels")

        conv = _conv_nd(self.spatial_dims)
        self.from_latent = conv(self.latent_channels, base * 4, kernel_size=1)
        self.up1 = _up_block(base * 4, base * 2, spatial_dims=self.spatial_dims, activation=activation, use_norm=use_norm)
        self.up2 = _up_block(base * 2, base, spatial_dims=self.spatial_dims, activation=activation, use_norm=use_norm)
        self.to_image = conv(base, out, kernel_size=1)

    def decode(self, z: torch.Tensor, domain: DomainBatch) -> torch.Tensor:
        del domain
        expected_dims = self.spatial_dims + 2
        if z.ndim != expected_dims:
            raise ValueError(f"Expected a {expected_dims}D latent tensor, got shape {tuple(z.shape)}.")
        if int(z.shape[1]) != self.latent_channels:
            raise ValueError(f"Expected {self.latent_channels} latent channels, got {int(z.shape[1])}.")
        h = self.from_latent(z)
        h = self.up1(h)
        h = self.up2(h)
        out = self.to_image(h)
        # Unconditional — do not make this an optional `final_activation` parameter like
        # CNNDecoder's. The [-1, 1] normalization contract (data/transforms.py's
        # normalize_percentile_clip_to_unit_range) and training/losses.py's lpips_loss
        # (which assumes un-normalized [-1, 1] inputs, no `normalize=True` flag) both
        # depend on the decoder output being bounded to [-1, 1]. Making this optional
        # would silently reintroduce the original diagnostic's rango-descalibrado bug.
        return torch.tanh(out)


def _down_block(
    in_channels: int, out_channels: int, *, spatial_dims: SpatialDims, activation: str, use_norm: bool
) -> nn.Sequential:
    conv = _conv_nd(spatial_dims)
    layers: list[nn.Module] = [conv(in_channels, out_channels, kernel_size=3, stride=2, padding=1)]
    if use_norm:
        layers.append(_norm(out_channels))
    layers.append(_activation(activation))
    return nn.Sequential(*layers)


def _up_block(
    in_channels: int, out_channels: int, *, spatial_dims: SpatialDims, activation: str, use_norm: bool
) -> nn.Sequential:
    conv = _conv_nd(spatial_dims)
    layers: list[nn.Module] = [
        nn.Upsample(scale_factor=2, mode="nearest"),
        conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
    ]
    if use_norm:
        layers.append(_norm(out_channels))
    layers.append(_activation(activation))
    return nn.Sequential(*layers)


def _validate_input_tensor(x: torch.Tensor, *, spatial_dims: SpatialDims, channels: int, downsample_factor: int) -> None:
    expected_dims = spatial_dims + 2
    if x.ndim != expected_dims:
        raise ValueError(f"Expected a {expected_dims}D input tensor, got shape {tuple(x.shape)}.")
    if int(x.shape[1]) != channels:
        raise ValueError(f"Expected {channels} input channels, got {int(x.shape[1])}.")
    spatial_shape = tuple(int(dim) for dim in x.shape[-spatial_dims:])
    bad_dims = [dim for dim in spatial_shape if dim % downsample_factor != 0]
    if bad_dims:
        raise ValueError(
            f"KLVAEEncoder inputs must have spatial dimensions divisible by "
            f"{downsample_factor}; got {spatial_shape}."
        )


def _validate_spatial_dims(spatial_dims: int) -> SpatialDims:
    if spatial_dims not in (2, 3):
        raise ValueError(f"spatial_dims must be 2 or 3, got {spatial_dims}.")
    return spatial_dims  # type: ignore[return-value]


def _conv_nd(spatial_dims: SpatialDims) -> type[nn.Conv2d] | type[nn.Conv3d]:
    return nn.Conv3d if spatial_dims == 3 else nn.Conv2d


def _positive_int(value: int, name: str) -> int:
    integer = int(value)
    if integer <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")
    return integer


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
