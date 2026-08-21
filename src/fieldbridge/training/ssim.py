"""Autocast-safe bounded SSIM similarities for differentiable training losses."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def stable_training_ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    data_range: float = 1.0,
    window_size: int = 7,
) -> torch.Tensor:
    """Stable 2D training similarity in the documented range ``[-1, 1]``."""

    if prediction.ndim != 4:
        raise ValueError(
            "stable_training_ssim expects (B, C, H, W) tensors."
        )
    return _stable_training_ssim(
        prediction,
        target,
        data_range=data_range,
        window_size=window_size,
        spatial_dims=2,
    )


def stable_training_ssim3d(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    data_range: float = 1.0,
    window_size: int = 7,
) -> torch.Tensor:
    """Stable 3D training similarity in the documented range ``[-1, 1]``."""

    if prediction.ndim != 5:
        raise ValueError(
            "stable_training_ssim3d expects (B, C, D, H, W) tensors."
        )
    return _stable_training_ssim(
        prediction,
        target,
        data_range=data_range,
        window_size=window_size,
        spatial_dims=3,
    )


def _stable_training_ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    data_range: float,
    window_size: int,
    spatial_dims: int,
) -> torch.Tensor:
    """Uniform-window SSIM with float32 moments and covariance projection.

    The pre-v3 ``E[x²] - E[x]²`` implementation could execute in bf16 autocast.
    Cancellation then allowed negative variances and similarities above one. Moments
    now run in float32, variances are projected nonnegative, and covariance is projected
    onto its Cauchy-Schwarz bound before the bounded luminance and structure factors are
    formed.
    """

    _validate_same_shape(prediction, target)
    if not bool(torch.isfinite(prediction).all()) or not bool(
        torch.isfinite(target).all()
    ):
        raise ValueError("Training SSIM inputs must contain only finite values.")
    if not data_range > 0:
        raise ValueError("Training SSIM data_range must be positive.")
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError(
            "Training SSIM window_size must be a positive odd integer."
        )

    pool = F.avg_pool3d if spatial_dims == 3 else F.avg_pool2d
    pad = window_size // 2
    padding = tuple(
        value for _ in range(spatial_dims) for value in (pad, pad)
    )
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        pred = prediction.float()
        tgt = target.float()

        def local_mean(x: torch.Tensor) -> torch.Tensor:
            return pool(
                F.pad(x, padding, mode="replicate"),
                kernel_size=window_size,
                stride=1,
            )

        mu_pred = local_mean(pred)
        mu_tgt = local_mean(tgt)
        var_pred = (
            local_mean(pred.square()) - mu_pred.square()
        ).clamp_min(0.0)
        var_tgt = (
            local_mean(tgt.square()) - mu_tgt.square()
        ).clamp_min(0.0)
        covariance = local_mean(pred * tgt) - mu_pred * mu_tgt
        # The projection bound is a forward-only numerical constraint. Differentiating
        # through sqrt(var_pred * var_tgt) is singular at the exact-zero local
        # variances that are required by the MRI background contract: sqrt'(0) is
        # unbounded even though the projected covariance and final SSIM are finite.
        # Detaching the bound preserves the exact forward projection and documented
        # [-1, 1] semantics while gradients continue through covariance, luminance,
        # variance, and the structure denominator. At the projection boundary this is
        # the standard stop-gradient treatment of a feasibility bound, rather than an
        # epsilon that would relax the Cauchy-Schwarz constraint.
        covariance_limit = (var_pred * var_tgt).detach().sqrt()
        covariance = torch.maximum(
            torch.minimum(covariance, covariance_limit), -covariance_limit
        )
        c1 = float(0.01 * data_range) ** 2
        c2 = float(0.03 * data_range) ** 2
        luminance = (2.0 * mu_pred * mu_tgt + c1) / (
            mu_pred.square() + mu_tgt.square() + c1
        )
        structure = (2.0 * covariance + c2) / (
            var_pred + var_tgt + c2
        )
        similarity = (luminance * structure).mean().clamp(-1.0, 1.0)

    if not bool(torch.isfinite(similarity)):
        raise ValueError("Training SSIM produced a non-finite value.")
    return similarity


def _validate_same_shape(
    prediction: torch.Tensor, target: torch.Tensor
) -> None:
    if prediction.shape != target.shape:
        raise ValueError(
            "prediction and target must have the same shape; "
            f"got {tuple(prediction.shape)} and {tuple(target.shape)}."
        )
