"""Feature-wise linear modulation (FiLM) conditioning layer."""

from __future__ import annotations

import torch
from torch import nn


class FiLMLayer(nn.Module):
    """Modulate a 2D feature map with per-channel scale/shift from a conditioning vector."""

    def __init__(self, conditioning_dim: int, num_channels: int) -> None:
        super().__init__()
        self.num_channels = int(num_channels)
        self.projection = nn.Linear(conditioning_dim, 2 * self.num_channels)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        scale, shift = self.projection(conditioning).chunk(2, dim=-1)
        scale = scale[..., None, None]
        shift = shift[..., None, None]
        return x * (1 + scale) + shift
