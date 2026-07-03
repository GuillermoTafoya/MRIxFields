"""Conditional U-Net image translator baseline.

This deterministic baseline preserves high-resolution anatomy through U-Net skip
connections while conditioning decoder blocks on source-target domain metadata.
It is not a diffusion model, Schrodinger bridge, adversarial model, or VAE.
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
UpsampleMode = Literal["interpolate", "transpose"]
SkipMode = Literal["gated", "concat", "none"]


class ConditionalUNetFieldTranslator(BaseTranslator):
    """Domain-conditioned U-Net baseline for MRI field/sequence translation."""

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
        skip_mode: SkipMode = "gated",
        final_activation: str | None = None,
        pad_to_multiple: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = _positive_int(in_channels, "in_channels")
        self.out_channels = _positive_int(out_channels, "out_channels")
        self.latent_channels = _positive_int(latent_channels, "latent_channels")
        self.cond_dim = _positive_int(cond_dim, "cond_dim")
        self.spatial_dims = _validate_spatial_dims(spatial_dims)
        self.upsample_mode = _validate_upsample_mode(upsample_mode)
        self.skip_mode = _validate_skip_mode(skip_mode)
        self.pad_to_multiple = bool(pad_to_multiple)
        channels = _normalize_channels(hidden_channels)
        self.downsample_factor = 2 ** len(channels)

        self.domain_embedding = DomainEmbedding(cond_dim=self.cond_dim)

        encoder_blocks: list[nn.Module] = []
        downsample_blocks: list[nn.Module] = []
        current_channels = self.in_channels
        for next_channels in channels:
            encoder_blocks.append(
                _ConditionedDoubleConv(
                    current_channels,
                    next_channels,
                    cond_dim=self.cond_dim,
                    spatial_dims=self.spatial_dims,
                    activation=activation,
                    use_norm=use_norm,
                )
            )
            downsample_blocks.append(
                _conv_nd(self.spatial_dims)(
                    next_channels,
                    next_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                )
            )
            current_channels = next_channels
        self.encoder_blocks = nn.ModuleList(encoder_blocks)
        self.downsample_blocks = nn.ModuleList(downsample_blocks)
        self.bottleneck = _ConditionedDoubleConv(
            current_channels,
            self.latent_channels,
            cond_dim=self.cond_dim,
            spatial_dims=self.spatial_dims,
            activation=activation,
            use_norm=use_norm,
        )

        decoder_blocks: list[nn.Module] = []
        current_channels = self.latent_channels
        for skip_channels in reversed(channels):
            decoder_blocks.append(
                _UNetDecoderBlock(
                    current_channels,
                    skip_channels,
                    cond_dim=self.cond_dim,
                    spatial_dims=self.spatial_dims,
                    activation=activation,
                    use_norm=use_norm,
                    upsample_mode=self.upsample_mode,
                    skip_mode=self.skip_mode,
                )
            )
            current_channels = skip_channels
        self.decoder_blocks = nn.ModuleList(decoder_blocks)
        self.output_projection = _conv_nd(self.spatial_dims)(
            current_channels,
            self.out_channels,
            kernel_size=1,
        )
        self.final_activation = _activation(final_activation) if final_activation else None

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
        )
        original_spatial_shape = tuple(int(dim) for dim in x.shape[-self.spatial_dims :])
        if self.pad_to_multiple:
            x = _pad_to_multiple(
                x,
                spatial_dims=self.spatial_dims,
                multiple=self.downsample_factor,
            )
        else:
            _validate_divisible(
                original_spatial_shape,
                downsample_factor=self.downsample_factor,
                model_name=type(self).__name__,
            )

        conditioning = self.domain_embedding(
            source_domain,
            target_domain,
            batch_size=int(x.shape[0]),
            device=x.device,
            dtype=x.dtype,
        )

        skips: list[torch.Tensor] = []
        h = x
        for encoder_block, downsample_block in zip(self.encoder_blocks, self.downsample_blocks):
            h = encoder_block(h, conditioning)
            skips.append(h)
            h = downsample_block(h)
        h = self.bottleneck(h, conditioning)

        for decoder_block, skip in zip(self.decoder_blocks, reversed(skips)):
            h = decoder_block(h, skip, conditioning)
        output = self.output_projection(h)
        if self.final_activation is not None:
            output = self.final_activation(output)
        return _crop_to_spatial_shape(output, original_spatial_shape, self.spatial_dims)


class _UNetDecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        *,
        cond_dim: int,
        spatial_dims: SpatialDims,
        activation: str,
        use_norm: bool,
        upsample_mode: UpsampleMode,
        skip_mode: SkipMode,
    ) -> None:
        super().__init__()
        self.skip_mode = skip_mode
        if upsample_mode == "transpose":
            self.upsample = _conv_transpose_nd(spatial_dims)(
                in_channels,
                skip_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            )
            self.post_upsample_conv = None
        else:
            self.upsample = None
            self.post_upsample_conv = _conv_nd(spatial_dims)(
                in_channels,
                skip_channels,
                kernel_size=3,
                padding=1,
            )
        self.skip_gate = (
            _ChannelwiseSkipGate(skip_channels, cond_dim, spatial_dims)
            if skip_mode == "gated"
            else None
        )
        merge_channels = skip_channels if skip_mode == "none" else 2 * skip_channels
        self.conv_block = _ConditionedDoubleConv(
            merge_channels,
            skip_channels,
            cond_dim=cond_dim,
            spatial_dims=spatial_dims,
            activation=activation,
            use_norm=use_norm,
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> torch.Tensor:
        if self.upsample is not None:
            x = self.upsample(x)
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            if self.post_upsample_conv is None:
                raise RuntimeError("interpolate decoder block is missing its convolution.")
            x = self.post_upsample_conv(x)

        if self.skip_mode != "none":
            skip = _match_spatial(skip, x.shape[2:])
            if self.skip_gate is not None:
                skip = self.skip_gate(skip, conditioning)
            x = torch.cat([x, skip], dim=1)
        return self.conv_block(x, conditioning)


class _ConditionedDoubleConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        cond_dim: int,
        spatial_dims: SpatialDims,
        activation: str,
        use_norm: bool,
    ) -> None:
        super().__init__()
        conv = _conv_nd(spatial_dims)
        self.conv1 = conv(in_channels, out_channels, kernel_size=3, padding=1)
        self.modulation1 = _modulation(cond_dim, out_channels, use_norm=use_norm)
        self.activation1 = _activation(activation)
        self.conv2 = conv(out_channels, out_channels, kernel_size=3, padding=1)
        self.modulation2 = _modulation(cond_dim, out_channels, use_norm=use_norm)
        self.activation2 = _activation(activation)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        x = self.activation1(self.modulation1(self.conv1(x), conditioning))
        return self.activation2(self.modulation2(self.conv2(x), conditioning))


class _ChannelwiseSkipGate(nn.Module):
    def __init__(self, channels: int, cond_dim: int, spatial_dims: SpatialDims) -> None:
        super().__init__()
        self.channels = _positive_int(channels, "channels")
        self.spatial_dims = spatial_dims
        self.projection = nn.Linear(cond_dim, self.channels)

    def forward(self, skip: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        if conditioning.ndim != 2 or int(conditioning.shape[0]) != int(skip.shape[0]):
            raise ValueError(
                "Skip gate conditioning must have shape (batch, cond_dim) with the same "
                f"batch size as skip features; got {tuple(conditioning.shape)} and "
                f"{tuple(skip.shape)}."
            )
        gate = torch.sigmoid(self.projection(conditioning))
        gate_shape = (skip.shape[0], self.channels, *([1] * self.spatial_dims))
        return skip * gate.reshape(gate_shape)


def _modulation(cond_dim: int, channels: int, *, use_norm: bool) -> nn.Module:
    if use_norm:
        return FiLMGroupNorm(cond_dim, channels)
    return FiLMLayer(cond_dim, channels)


def _pad_to_multiple(x: torch.Tensor, *, spatial_dims: int, multiple: int) -> torch.Tensor:
    if multiple <= 1:
        return x
    spatial_shape = tuple(int(dim) for dim in x.shape[-spatial_dims:])
    pad_by_dim = [(multiple - (dim % multiple)) % multiple for dim in spatial_shape]
    if not any(pad_by_dim):
        return x
    pad: list[int] = []
    for amount in reversed(pad_by_dim):
        pad.extend([0, amount])
    return F.pad(x, pad, mode="constant", value=0.0)


def _crop_to_spatial_shape(
    x: torch.Tensor,
    spatial_shape: tuple[int, ...],
    spatial_dims: int,
) -> torch.Tensor:
    slices = [slice(None), slice(None)]
    for dim_size in spatial_shape:
        slices.append(slice(0, dim_size))
    return x[tuple(slices[: spatial_dims + 2])]


def _match_spatial(x: torch.Tensor, target_shape: torch.Size | tuple[int, ...]) -> torch.Tensor:
    target = tuple(int(dim) for dim in target_shape)
    current = tuple(int(dim) for dim in x.shape[2:])
    if current == target:
        return x

    slices: list[slice] = [slice(None), slice(None)]
    cropped_shape: list[int] = []
    for current_dim, target_dim in zip(current, target):
        if current_dim > target_dim:
            start = (current_dim - target_dim) // 2
            slices.append(slice(start, start + target_dim))
            cropped_shape.append(target_dim)
        else:
            slices.append(slice(None))
            cropped_shape.append(current_dim)
    x = x[tuple(slices)]

    pad: list[int] = []
    for current_dim, target_dim in reversed(list(zip(cropped_shape, target))):
        deficit = max(target_dim - current_dim, 0)
        left = deficit // 2
        right = deficit - left
        pad.extend([left, right])
    if any(pad):
        x = F.pad(x, pad, mode="constant", value=0.0)
    return x


def _validate_input_tensor(x: torch.Tensor, *, spatial_dims: int, channels: int) -> None:
    expected_dims = spatial_dims + 2
    if x.ndim != expected_dims:
        raise ValueError(f"Expected {expected_dims}D input tensor, got shape {tuple(x.shape)}.")
    if int(x.shape[1]) != channels:
        raise ValueError(f"Expected {channels} input channels, got {int(x.shape[1])}.")
    spatial_shape = tuple(int(dim) for dim in x.shape[-spatial_dims:])
    if any(dim <= 0 for dim in spatial_shape):
        raise ValueError(f"Spatial dimensions must be positive, got {spatial_shape}.")


def _validate_divisible(
    spatial_shape: tuple[int, ...],
    *,
    downsample_factor: int,
    model_name: str,
) -> None:
    bad_dims = [dim for dim in spatial_shape if dim % downsample_factor != 0]
    if bad_dims:
        raise ValueError(
            f"{model_name} inputs must have spatial dimensions divisible by "
            f"{downsample_factor} when pad_to_multiple=False; got {spatial_shape}."
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
    if upsample_mode not in ("interpolate", "transpose"):
        raise ValueError(
            "upsample_mode must be 'interpolate' or 'transpose', "
            f"got {upsample_mode!r}."
        )
    return upsample_mode  # type: ignore[return-value]


def _validate_skip_mode(skip_mode: str) -> SkipMode:
    if skip_mode not in ("gated", "concat", "none"):
        raise ValueError(f"skip_mode must be 'gated', 'concat', or 'none', got {skip_mode!r}.")
    return skip_mode  # type: ignore[return-value]


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
