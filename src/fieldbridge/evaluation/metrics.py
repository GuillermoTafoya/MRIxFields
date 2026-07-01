"""Tensor metrics, including the three official MRIxFields Task 3 metrics: nRMSE, SSIM, LPIPS."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((prediction - target) ** 2)


def mae(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(prediction - target))


def psnr(prediction: torch.Tensor, target: torch.Tensor, *, data_range: float = 1.0) -> torch.Tensor:
    error = mse(prediction, target).clamp_min(torch.finfo(prediction.dtype).eps)
    return 20 * torch.log10(torch.tensor(data_range, dtype=prediction.dtype, device=prediction.device)) - (
        10 * torch.log10(error)
    )


def nrmse(prediction: torch.Tensor, target: torch.Tensor, *, data_range: float = 1.0) -> torch.Tensor:
    """RMSE normalized by the intensity range (official MRIxFields Task 3 metric)."""

    return torch.sqrt(mse(prediction, target)) / data_range


def ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    data_range: float = 1.0,
    window_size: int = 7,
) -> torch.Tensor:
    """2D structural similarity (official MRIxFields Task 3 metric), uniform-window."""

    if prediction.ndim != 4:
        raise ValueError("ssim expects (B, C, H, W) tensors — this project is 2D-only.")

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    pad = window_size // 2

    def local_mean(x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(x, kernel_size=window_size, stride=1, padding=pad)

    mu_p = local_mean(prediction)
    mu_t = local_mean(target)
    mu_p_sq, mu_t_sq, mu_pt = mu_p**2, mu_t**2, mu_p * mu_t

    sigma_p_sq = local_mean(prediction**2) - mu_p_sq
    sigma_t_sq = local_mean(target**2) - mu_t_sq
    sigma_pt = local_mean(prediction * target) - mu_pt

    numerator = (2 * mu_pt + c1) * (2 * sigma_pt + c2)
    denominator = (mu_p_sq + mu_t_sq + c1) * (sigma_p_sq + sigma_t_sq + c2)
    return (numerator / denominator).mean()


def lpips_metric(
    prediction: torch.Tensor, target: torch.Tensor, *, net: torch.nn.Module | None = None
) -> torch.Tensor:
    """Perceptual distance (official MRIxFields Task 3 metric). Requires the optional `lpips` package."""

    from fieldbridge.training.losses import lpips_loss

    with torch.no_grad():
        return lpips_loss(prediction, target, net=net)

