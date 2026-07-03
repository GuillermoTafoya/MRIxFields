"""Conditional CNN image translator baseline.

This is a compact, CPU-friendly image-to-image translator for interface and
same-domain reconstruction tests. It is not a diffusion model, Schrodinger
bridge, adversarial model, or VAE.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from fieldbridge.data.domains import Domain
from fieldbridge.models.conditioning import DomainEmbedding
from fieldbridge.models.film import FiLMGroupNorm, FiLMLayer
from fieldbridge.models.translators.base import BaseTranslator

DomainBatch = Domain | Sequence[Domain]
SpatialDims = Literal[2, 3]
UpsampleMode = Literal["transpose", "interpolate"]


class ConditionalCNNFieldTranslator(BaseTranslator):
    """Domain-conditioned CNN baseline for full tensor translation.

    Forward computes ``x_hat = G(x, source_domain, target_domain)`` for 2D slice
    tensors ``(B, C, H, W)`` or 3D volume tensors ``(B, C, D, H, W)``.
    """

    def __init__(
        self,
        *,
        in_channels: int = 1,
        out_channels: int = 1,
        hidden_channels: Sequence[int] = (32, 64, 128),
        latent_channels: int = 128,
        cond_dim: int = 128,
        spatial_dims: SpatialDims = 2,
        activation: str = "silu",
        use_norm: bool = True,
        upsample_mode: UpsampleMode = "interpolate",
        final_activation: str | None = None,
    ) -> None:
        super().__init__()
        self.in_channels = _positive_int(in_channels, "in_channels")
        self.out_channels = _positive_int(out_channels, "out_channels")
        self.latent_channels = _positive_int(latent_channels, "latent_channels")
        self.cond_dim = _positive_int(cond_dim, "cond_dim")
        self.spatial_dims = _validate_spatial_dims(spatial_dims)
        self.upsample_mode = _validate_upsample_mode(upsample_mode)
        channels = _normalize_channels(hidden_channels)
        self.downsample_factor = 2 ** len(channels)

        self.domain_embedding = DomainEmbedding(cond_dim=self.cond_dim)

        encoder_blocks: list[nn.Module] = []
        current_channels = self.in_channels
        for next_channels in channels:
            encoder_blocks.append(
                _ConditionalConvBlock(
                    current_channels,
                    next_channels,
                    cond_dim=self.cond_dim,
                    spatial_dims=self.spatial_dims,
                    activation=activation,
                    stride=2,
                    use_norm=use_norm,
                )
            )
            current_channels = next_channels
        self.encoder_blocks = nn.ModuleList(encoder_blocks)
        self.latent_block = _ConditionalConvBlock(
            current_channels,
            self.latent_channels,
            cond_dim=self.cond_dim,
            spatial_dims=self.spatial_dims,
            activation=activation,
            stride=1,
            use_norm=use_norm,
        )

        decoder_blocks: list[nn.Module] = []
        current_channels = self.latent_channels
        for next_channels in reversed(channels):
            decoder_blocks.append(
                _ConditionalUpsampleBlock(
                    current_channels,
                    next_channels,
                    cond_dim=self.cond_dim,
                    spatial_dims=self.spatial_dims,
                    activation=activation,
                    upsample_mode=self.upsample_mode,
                    use_norm=use_norm,
                )
            )
            current_channels = next_channels
        self.decoder_blocks = nn.ModuleList(decoder_blocks)
        self.output_projection = _conv_nd(self.spatial_dims)(
            current_channels,
            self.out_channels,
            kernel_size=1,
        )
        self.final_activation = _activation(final_activation) if final_activation else None

    def encode(self, x: torch.Tensor, source_domain: DomainBatch) -> torch.Tensor:
        """Encode an input tensor with source-domain conditioning."""

        _validate_input_tensor(
            x,
            spatial_dims=self.spatial_dims,
            channels=self.in_channels,
            downsample_factor=self.downsample_factor,
        )
        conditioning = self.domain_embedding(
            source_domain,
            source_domain,
            batch_size=int(x.shape[0]),
            device=x.device,
            dtype=x.dtype,
        )
        return self._encode_with_conditioning(x, conditioning)

    def decode(
        self,
        z: torch.Tensor,
        source_domain: DomainBatch,
        target_domain: DomainBatch,
    ) -> torch.Tensor:
        """Decode a latent tensor with source-target conditioning."""

        expected_dims = self.spatial_dims + 2
        if z.ndim != expected_dims:
            raise ValueError(
                f"Expected {expected_dims}D latent tensor, got shape {tuple(z.shape)}."
            )
        conditioning = self.domain_embedding(
            source_domain,
            target_domain,
            batch_size=int(z.shape[0]),
            device=z.device,
            dtype=z.dtype,
        )
        return self._decode_with_conditioning(z, conditioning)

    def forward(
        self,
        x: torch.Tensor,
        source_domain: DomainBatch,
        target_domain: DomainBatch,
        t: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        del t
        _validate_input_tensor(
            x,
            spatial_dims=self.spatial_dims,
            channels=self.in_channels,
            downsample_factor=self.downsample_factor,
        )
        spatial_shape = tuple(int(dim) for dim in x.shape[-self.spatial_dims :])
        batch_size = int(x.shape[0])
        source_conditioning = self.domain_embedding(
            source_domain,
            source_domain,
            batch_size=batch_size,
            device=x.device,
            dtype=x.dtype,
        )
        pair_conditioning = self.domain_embedding(
            source_domain,
            target_domain,
            batch_size=batch_size,
            device=x.device,
            dtype=x.dtype,
        )
        z = self._encode_with_conditioning(x, source_conditioning)
        output = self._decode_with_conditioning(z, pair_conditioning)
        output_shape = tuple(int(dim) for dim in output.shape[-self.spatial_dims :])
        if output_shape != spatial_shape:
            raise RuntimeError(
                "ConditionalCNNFieldTranslator output spatial shape changed unexpectedly; "
                f"expected {spatial_shape}, got {output_shape}."
            )
        return output

    def _encode_with_conditioning(
        self,
        x: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> torch.Tensor:
        for block in self.encoder_blocks:
            x = block(x, conditioning)
        return self.latent_block(x, conditioning)

    def _decode_with_conditioning(
        self,
        z: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> torch.Tensor:
        for block in self.decoder_blocks:
            z = block(z, conditioning)
        output = self.output_projection(z)
        if self.final_activation is not None:
            output = self.final_activation(output)
        return output


class _ConditionalConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        cond_dim: int,
        spatial_dims: SpatialDims,
        activation: str,
        stride: int,
        use_norm: bool,
    ) -> None:
        super().__init__()
        conv = _conv_nd(spatial_dims)
        self.conv = conv(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.modulation = _modulation(cond_dim, out_channels, use_norm=use_norm)
        self.activation = _activation(activation)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        return self.activation(self.modulation(self.conv(x), conditioning))


class _ConditionalUpsampleBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        cond_dim: int,
        spatial_dims: SpatialDims,
        activation: str,
        upsample_mode: UpsampleMode,
        use_norm: bool,
    ) -> None:
        super().__init__()
        self.upsample_mode = upsample_mode
        if upsample_mode == "transpose":
            self.upsample = _conv_transpose_nd(spatial_dims)(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            )
            self.conv = None
        else:
            self.upsample = None
            self.conv = _conv_nd(spatial_dims)(in_channels, out_channels, kernel_size=3, padding=1)
        self.modulation = _modulation(cond_dim, out_channels, use_norm=use_norm)
        self.activation = _activation(activation)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        if self.upsample is not None:
            x = self.upsample(x)
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            if self.conv is None:
                raise RuntimeError("interpolate upsample block is missing its convolution.")
            x = self.conv(x)
        return self.activation(self.modulation(x, conditioning))


def _modulation(cond_dim: int, channels: int, *, use_norm: bool) -> nn.Module:
    if use_norm:
        return FiLMGroupNorm(cond_dim, channels)
    return FiLMLayer(cond_dim, channels)


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
    spatial_shape = tuple(int(dim) for dim in x.shape[-spatial_dims:])
    if any(dim <= 0 for dim in spatial_shape):
        raise ValueError(f"Spatial dimensions must be positive, got {spatial_shape}.")
    if downsample_factor <= 1:
        return
    bad_dims = [dim for dim in spatial_shape if dim % downsample_factor != 0]
    if bad_dims:
        raise ValueError(
            "ConditionalCNNFieldTranslator inputs must have spatial dimensions divisible by "
            f"{downsample_factor}; got {spatial_shape}."
        )


def _normalize_channels(channels: Sequence[int]) -> tuple[int, ...]:
    normalized = tuple(_positive_int(value, "hidden_channels") for value in channels)
    if not normalized:
        raise ValueError("hidden_channels must contain at least one channel count.")
    return normalized


def _validate_spatial_dims(spatial_dims: int) -> SpatialDims:
    if spatial_dims not in (2, 3):
        raise ValueError(f"spatial_dims must be 2 or 3, got {spatial_dims}.")
    return spatial_dims  # type: ignore[return-value]


def _validate_upsample_mode(upsample_mode: str) -> UpsampleMode:
    if upsample_mode not in ("transpose", "interpolate"):
        raise ValueError(
            "upsample_mode must be 'transpose' or 'interpolate', "
            f"got {upsample_mode!r}."
        )
    return upsample_mode  # type: ignore[return-value]


def _positive_int(value: int, name: str) -> int:
    integer = int(value)
    if integer <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")
    return integer


def _conv_nd(spatial_dims: int) -> type[nn.Conv2d] | type[nn.Conv3d]:
    return nn.Conv3d if spatial_dims == 3 else nn.Conv2d


def _conv_transpose_nd(
    spatial_dims: int,
) -> type[nn.ConvTranspose2d] | type[nn.ConvTranspose3d]:
    return nn.ConvTranspose3d if spatial_dims == 3 else nn.ConvTranspose2d


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
