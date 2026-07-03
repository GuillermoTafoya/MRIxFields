"""Feature-wise linear modulation (FiLM) conditioning layers."""

from __future__ import annotations

import torch
from torch import nn


class FiLMLayer(nn.Module):
    """Modulate a 2D or 3D feature map with per-channel scale/shift."""

    def __init__(self, conditioning_dim: int, num_channels: int) -> None:
        super().__init__()
        self.conditioning_dim = _positive_int(conditioning_dim, "conditioning_dim")
        self.num_channels = int(num_channels)
        if self.num_channels <= 0:
            raise ValueError(f"num_channels must be positive, got {num_channels}.")
        self.projection = nn.Linear(self.conditioning_dim, 2 * self.num_channels)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        _validate_feature_conditioning(
            x,
            conditioning,
            num_channels=self.num_channels,
            conditioning_dim=self.conditioning_dim,
        )
        scale, shift = self.projection(conditioning).chunk(2, dim=-1)
        broadcast_shape = (x.shape[0], self.num_channels, *([1] * (x.ndim - 2)))
        scale = scale.reshape(broadcast_shape)
        shift = shift.reshape(broadcast_shape)
        return x * (1 + scale) + shift


class FiLMGroupNorm(nn.Module):
    """GroupNorm without affine parameters, followed by FiLM scale/shift."""

    def __init__(
        self,
        conditioning_dim: int,
        num_channels: int,
        *,
        max_groups: int = 8,
    ) -> None:
        super().__init__()
        self.conditioning_dim = _positive_int(conditioning_dim, "conditioning_dim")
        self.num_channels = _positive_int(num_channels, "num_channels")
        groups = _valid_group_count(self.num_channels, max_groups=max_groups)
        self.norm = nn.GroupNorm(groups, self.num_channels, affine=False)
        self.projection = nn.Linear(self.conditioning_dim, 2 * self.num_channels)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        _validate_feature_conditioning(
            x,
            conditioning,
            num_channels=self.num_channels,
            conditioning_dim=self.conditioning_dim,
        )
        scale, shift = self.projection(conditioning).chunk(2, dim=-1)
        broadcast_shape = (x.shape[0], self.num_channels, *([1] * (x.ndim - 2)))
        scale = scale.reshape(broadcast_shape)
        shift = shift.reshape(broadcast_shape)
        return self.norm(x) * (1 + scale) + shift


def _validate_feature_conditioning(
    x: torch.Tensor,
    conditioning: torch.Tensor,
    *,
    num_channels: int,
    conditioning_dim: int,
) -> None:
    if x.ndim not in (4, 5):
        raise ValueError(f"Expected a 4D or 5D feature tensor, got shape {tuple(x.shape)}.")
    if int(x.shape[1]) != num_channels:
        raise ValueError(f"Expected {num_channels} channels, got {int(x.shape[1])}.")
    if conditioning.ndim != 2:
        raise ValueError(
            "Expected conditioning tensor with shape (batch, conditioning_dim), "
            f"got {tuple(conditioning.shape)}."
        )
    if int(conditioning.shape[0]) != int(x.shape[0]):
        raise ValueError(
            "Conditioning batch size must match feature batch size; "
            f"got {int(conditioning.shape[0])} and {int(x.shape[0])}."
        )
    if int(conditioning.shape[1]) != conditioning_dim:
        raise ValueError(
            f"Expected conditioning_dim={conditioning_dim}, got {int(conditioning.shape[1])}."
        )


def _valid_group_count(channels: int, *, max_groups: int) -> int:
    groups = min(_positive_int(max_groups, "max_groups"), channels)
    while channels % groups != 0:
        groups -= 1
    return groups


def _positive_int(value: int, name: str) -> int:
    integer = int(value)
    if integer <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")
    return integer
