"""Small FiLM-conditioned denoising network operating on the VAE's latent grid.

Supports 2D and 3D latents (spatial_dims=2 or 3, matching KLVAEEncoder/Decoder — see
that module's docstring for why 3D support exists here at all). At most 1-2
downsample/upsample levels — sized for a small latent grid, not a deep multi-scale
U-Net (that would be scope creep at this resolution; the scale-down vs. the DDM paper
is in T and spatial depth, not in channel width — `base_channels` should stay
comparable to `latent_channels`). Timestep and field-strength conditioning embeddings
are SUMMED (matching the paper's "t added with g" combination) before being fed into
each block's FiLM layer — this is where we deviate from the paper, which injects
additively into the residual stream directly; we feed the summed signal through FiLM's
learned scale+shift instead, since FiLM is already implemented/tested in this project
(and already generalized to 4D/5D tensors) and additive is FiLM's degenerate case with
scale fixed at 0.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
from torch import nn

from fieldbridge.data.domains import Domain
from fieldbridge.models.diffusion.field_conditioner import FieldStrengthConditioner
from fieldbridge.models.diffusion.timestep_embedding import sinusoidal_timestep_embedding
from fieldbridge.models.film import FiLMLayer

DomainBatch = Domain | Sequence[Domain]
SpatialDims = Literal[2, 3]


class ConditionedResidualBlock(nn.Module):
    """conv -> norm -> activation -> FiLM(conditioning) -> conv -> norm -> activation, + residual."""

    def __init__(
        self,
        *,
        channels: int,
        conditioning_dim: int,
        spatial_dims: SpatialDims = 2,
        activation: str = "silu",
        use_norm: bool = True,
    ) -> None:
        super().__init__()
        conv = _conv_nd(spatial_dims)
        self.conv1 = conv(channels, channels, kernel_size=3, stride=1, padding=1)
        self.norm1 = _norm(channels) if use_norm else nn.Identity()
        self.act1 = _activation(activation)
        self.film = FiLMLayer(conditioning_dim, channels)
        self.conv2 = conv(channels, channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = _norm(channels) if use_norm else nn.Identity()
        self.act2 = _activation(activation)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        h = self.act1(self.norm1(self.conv1(x)))
        h = self.film(h, conditioning)
        h = self.act2(self.norm2(self.conv2(h)))
        return x + h


class DenoisingUNet(nn.Module):
    """Predicts the noise added to `z_t` at timestep `t`, conditioned on field strength."""

    def __init__(
        self,
        *,
        latent_channels: int = 128,
        base_channels: int = 128,
        spatial_dims: SpatialDims = 2,
        num_levels: int = 1,
        num_blocks_per_level: int = 2,
        timestep_embedding_dim: int = 64,
        field_conditioning_dim: int = 32,
        activation: str = "silu",
        use_norm: bool = True,
    ) -> None:
        super().__init__()
        if num_levels not in (1, 2):
            raise ValueError(f"num_levels must be 1 or 2 (small latent, not a deep U-Net), got {num_levels}.")
        self.latent_channels = int(latent_channels)
        self.spatial_dims = _validate_spatial_dims(spatial_dims)
        self.num_levels = num_levels
        self.timestep_embedding_dim = int(timestep_embedding_dim)

        conditioning_dim = max(timestep_embedding_dim, field_conditioning_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.timestep_embedding_dim, conditioning_dim),
            nn.SiLU(),
            nn.Linear(conditioning_dim, conditioning_dim),
        )
        self.field_conditioner = FieldStrengthConditioner(
            conditioning_dim=conditioning_dim, hidden_dim=field_conditioning_dim
        )

        conv = _conv_nd(self.spatial_dims)
        self.stem = conv(self.latent_channels, base_channels, kernel_size=3, stride=1, padding=1)
        self.blocks_level0 = nn.ModuleList(
            [
                ConditionedResidualBlock(
                    channels=base_channels,
                    conditioning_dim=conditioning_dim,
                    spatial_dims=self.spatial_dims,
                    activation=activation,
                    use_norm=use_norm,
                )
                for _ in range(num_blocks_per_level)
            ]
        )

        if num_levels == 2:
            self.downsample = conv(base_channels, base_channels, kernel_size=3, stride=2, padding=1)
            self.blocks_level1 = nn.ModuleList(
                [
                    ConditionedResidualBlock(
                        channels=base_channels,
                        conditioning_dim=conditioning_dim,
                        spatial_dims=self.spatial_dims,
                        activation=activation,
                        use_norm=use_norm,
                    )
                    for _ in range(num_blocks_per_level)
                ]
            )
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                conv(base_channels, base_channels, kernel_size=3, stride=1, padding=1),
            )
            self.skip_combine = conv(base_channels * 2, base_channels, kernel_size=1)

        self.out_conv = conv(base_channels, self.latent_channels, kernel_size=1)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, domain: DomainBatch) -> torch.Tensor:
        expected_dims = self.spatial_dims + 2
        if z_t.ndim != expected_dims:
            raise ValueError(f"Expected a {expected_dims}D latent tensor, got shape {tuple(z_t.shape)}.")
        if int(z_t.shape[1]) != self.latent_channels:
            raise ValueError(f"Expected {self.latent_channels} latent channels, got {int(z_t.shape[1])}.")
        if self.num_levels == 2:
            spatial = tuple(int(dim) for dim in z_t.shape[-self.spatial_dims :])
            if any(dim % 2 != 0 for dim in spatial):
                raise ValueError(f"num_levels=2 requires even spatial dims, got {spatial}.")

        t_emb = self.time_mlp(sinusoidal_timestep_embedding(t, embedding_dim=self.timestep_embedding_dim))
        field_emb = self.field_conditioner(domain, batch_size=z_t.shape[0], device=z_t.device)
        conditioning = t_emb + field_emb

        h = self.stem(z_t)
        for block in self.blocks_level0:
            h = block(h, conditioning)

        if self.num_levels == 2:
            skip = h
            h = self.downsample(h)
            for block in self.blocks_level1:
                h = block(h, conditioning)
            h = self.upsample(h)
            h = self.skip_combine(torch.cat([h, skip], dim=1))

        return self.out_conv(h)


def _validate_spatial_dims(spatial_dims: int) -> SpatialDims:
    if spatial_dims not in (2, 3):
        raise ValueError(f"spatial_dims must be 2 or 3, got {spatial_dims}.")
    return spatial_dims  # type: ignore[return-value]


def _conv_nd(spatial_dims: SpatialDims) -> type[nn.Conv2d] | type[nn.Conv3d]:
    return nn.Conv3d if spatial_dims == 3 else nn.Conv2d


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
