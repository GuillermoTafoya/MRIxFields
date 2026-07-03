"""Sinusoidal diffusion-timestep embedding (DDPM/Transformer convention)."""

from __future__ import annotations

import math

import torch


def sinusoidal_timestep_embedding(
    timesteps: torch.Tensor, *, embedding_dim: int, max_period: float = 10_000.0
) -> torch.Tensor:
    """Standard sin/cos position embedding, no learnable parameters at this layer.

    `timesteps`: (B,) integer or float tensor of diffusion steps.
    Returns: (B, embedding_dim) tensor, finite. The learnable projection from this fixed
    embedding to the denoising network's internal hidden size lives in the network
    itself (e.g. a small MLP), not here.
    """

    if embedding_dim <= 0:
        raise ValueError(f"embedding_dim must be positive, got {embedding_dim}.")

    half_dim = embedding_dim // 2
    device = timesteps.device
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half_dim, dtype=torch.float32, device=device) / max(half_dim, 1)
    )
    args = timesteps.float().unsqueeze(-1) * freqs.unsqueeze(0)
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if embedding_dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
