"""Basic tensor metrics for smoke-level evaluation."""

from __future__ import annotations

import torch


def mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((prediction - target) ** 2)


def mae(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(prediction - target))


def psnr(prediction: torch.Tensor, target: torch.Tensor, *, data_range: float = 1.0) -> torch.Tensor:
    error = mse(prediction, target).clamp_min(torch.finfo(prediction.dtype).eps)
    return 20 * torch.log10(torch.tensor(data_range, dtype=prediction.dtype, device=prediction.device)) - (
        10 * torch.log10(error)
    )

