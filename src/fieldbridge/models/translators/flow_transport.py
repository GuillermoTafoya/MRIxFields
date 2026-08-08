"""Time-conditioned latent transport network v_theta(z_t, t, c_source, c_target).

A single shared network for the Etapa-2 ablation ladder (OT-CFM -> Schrodinger bridge):
the coupling and the bridge are training-time choices, this network is the same. It reuses
the FiLM-conditioned U-Net body of :class:`ConditionalUNetFieldTranslator` (domain
conditioning via :class:`DomainEmbedding`, NEVER a per-field/contrast router) and folds a
sinusoidal embedding of the continuous flow time ``t in [0, 1]`` into that conditioning, so
the same block modulation carries both the (source, target) domain pair and the time.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
from torch import nn

from fieldbridge.data.domains import Domain
from fieldbridge.models.diffusion.timestep_embedding import sinusoidal_timestep_embedding
from fieldbridge.models.translators.conditional_unet import (
    ConditionalUNetFieldTranslator,
    _crop_to_spatial_shape,
    _pad_to_multiple,
    _validate_input_tensor,
)

DomainBatch = Domain | Sequence[Domain]
SpatialDims = Literal[2, 3]


class FlowMatchingLatentTranslator(ConditionalUNetFieldTranslator):
    """U-Net velocity field over latents, conditioned on (source, target, time).

    ``in_channels`` and ``out_channels`` both equal the VAE latent channel count (the
    network maps a latent to a latent-shaped velocity). ``bottleneck_channels`` is the
    U-Net bottleneck width (the parent's ``latent_channels``), renamed here to avoid
    colliding with the *data* latent channels.
    """

    def __init__(
        self,
        *,
        latent_channels: int = 4,
        hidden_channels: Sequence[int] = (64, 128),
        bottleneck_channels: int = 256,
        cond_dim: int = 128,
        time_embed_dim: int = 128,
        time_scale: float = 1000.0,
        spatial_dims: SpatialDims = 3,
        activation: str = "silu",
        use_norm: bool = True,
        upsample_mode: str = "interpolate",
        skip_mode: str = "concat",
        pad_to_multiple: bool = True,
        zero_init_output: bool = False,
    ) -> None:
        super().__init__(
            in_channels=latent_channels,
            out_channels=latent_channels,
            hidden_channels=hidden_channels,
            latent_channels=bottleneck_channels,
            cond_dim=cond_dim,
            spatial_dims=spatial_dims,
            activation=activation,
            use_norm=use_norm,
            upsample_mode=upsample_mode,
            skip_mode=skip_mode,
            final_activation=None,  # velocity is unbounded
            pad_to_multiple=pad_to_multiple,
        )
        self.data_latent_channels = latent_channels
        self.time_embed_dim = int(time_embed_dim)
        self.time_scale = float(time_scale)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_embed_dim, self.cond_dim),
            nn.SiLU(),
            nn.Linear(self.cond_dim, self.cond_dim),
        )
        # Fuse the domain-pair conditioning and the time conditioning into one cond vector
        # that every FiLM block consumes (same interface as the parent U-Net).
        self.cond_combine = nn.Sequential(
            nn.Linear(2 * self.cond_dim, self.cond_dim),
            nn.SiLU(),
            nn.Linear(self.cond_dim, self.cond_dim),
        )
        self.zero_init_output = bool(zero_init_output)
        if self.zero_init_output:
            nn.init.zeros_(self.output_projection.weight)
            if self.output_projection.bias is not None:
                nn.init.zeros_(self.output_projection.bias)

    def _time_conditioning(
        self, t: torch.Tensor | float, *, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        if isinstance(t, torch.Tensor):
            t_vec = t.reshape(-1).to(device=device, dtype=torch.float32)
            if t_vec.numel() == 1:
                t_vec = t_vec.expand(batch_size)
            elif t_vec.numel() != batch_size:
                raise ValueError(
                    f"time t has {t_vec.numel()} entries but batch size is {batch_size}."
                )
        else:
            t_vec = torch.full((batch_size,), float(t), device=device, dtype=torch.float32)
        embedding = sinusoidal_timestep_embedding(
            t_vec * self.time_scale, embedding_dim=self.time_embed_dim
        )
        return self.time_mlp(embedding.to(dtype))

    def forward(
        self,
        z: torch.Tensor,
        source_domain: DomainBatch,
        target_domain: DomainBatch,
        t: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        if t is None:
            raise ValueError("FlowMatchingLatentTranslator requires the flow time t.")
        _validate_input_tensor(z, spatial_dims=self.spatial_dims, channels=self.in_channels)
        batch_size = int(z.shape[0])
        original_spatial_shape = tuple(int(dim) for dim in z.shape[-self.spatial_dims :])
        if self.pad_to_multiple:
            z = _pad_to_multiple(z, spatial_dims=self.spatial_dims, multiple=self.downsample_factor)

        domain_cond = self.domain_embedding(
            source_domain,
            target_domain,
            batch_size=batch_size,
            device=z.device,
            dtype=z.dtype,
        )
        time_cond = self._time_conditioning(
            t, batch_size=batch_size, device=z.device, dtype=z.dtype
        )
        conditioning = self.cond_combine(torch.cat([domain_cond, time_cond], dim=-1))

        skips: list[torch.Tensor] = []
        h = z
        for encoder_block, downsample_block in zip(self.encoder_blocks, self.downsample_blocks):
            h = encoder_block(h, conditioning)
            skips.append(h)
            h = downsample_block(h)
        h = self.bottleneck(h, conditioning)
        for decoder_block, skip in zip(self.decoder_blocks, reversed(skips)):
            h = decoder_block(h, skip, conditioning)
        velocity = self.output_projection(h)
        return _crop_to_spatial_shape(velocity, original_spatial_shape, self.spatial_dims)


__all__ = ["FlowMatchingLatentTranslator"]
