"""Identity-initialized conditional residual U-Net translator."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from fieldbridge.data.preprocessing import ModelRange
from fieldbridge.models.translators.base import BaseTranslator
from fieldbridge.models.translators.conditional_unet import (
    ConditionalUNetFieldTranslator,
    DomainBatch,
    SkipMode,
    SpatialDims,
    UpsampleMode,
)


class ConditionalResidualUNetFieldTranslator(BaseTranslator):
    """Predict a bounded residual while starting exactly at the input image."""

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
        pad_to_multiple: bool = True,
        model_range: ModelRange = "minus_one_one",
    ) -> None:
        super().__init__()
        if int(in_channels) != int(out_channels):
            raise ValueError(
                "ConditionalResidualUNetFieldTranslator requires matching input and "
                f"output channels for identity initialization; got {in_channels} and "
                f"{out_channels}."
            )
        self.model_range = _validate_model_range(model_range)
        self.backbone = ConditionalUNetFieldTranslator(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            latent_channels=latent_channels,
            cond_dim=cond_dim,
            spatial_dims=spatial_dims,
            activation=activation,
            use_norm=use_norm,
            upsample_mode=upsample_mode,
            skip_mode=skip_mode,
            final_activation=None,
            pad_to_multiple=pad_to_multiple,
        )
        nn.init.zeros_(self.backbone.output_projection.weight)
        if self.backbone.output_projection.bias is not None:
            nn.init.zeros_(self.backbone.output_projection.bias)

    def forward(
        self,
        x: torch.Tensor,
        source_domain: DomainBatch,
        target_domain: DomainBatch,
        t: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        residual = self.backbone(x, source_domain, target_domain, t=t)
        lower, upper = _model_range_bounds(self.model_range)
        return torch.clamp(x + residual, min=lower, max=upper)


def _validate_model_range(model_range: str) -> ModelRange:
    if model_range not in ("zero_one", "minus_one_one"):
        raise ValueError(
            "model_range must be 'zero_one' or 'minus_one_one', "
            f"got {model_range!r}."
        )
    return model_range  # type: ignore[return-value]


def _model_range_bounds(model_range: ModelRange) -> tuple[float, float]:
    if model_range == "zero_one":
        return 0.0, 1.0
    return -1.0, 1.0
