"""Tensor metrics, including the three official MRIxFields Task 3 metrics: nRMSE, SSIM, LPIPS."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((prediction - target) ** 2)


def mae(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(prediction - target))


def masked_mae(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean absolute error inside an optional binary mask."""

    _validate_same_shape(prediction, target)
    if mask is None:
        return mae(prediction, target)
    prepared_mask = _prepare_mask(mask, prediction)
    denominator = _positive_mask_sum(prepared_mask, "masked_mae")
    return (torch.abs(prediction - target) * prepared_mask).sum() / denominator


def masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean squared error inside an optional binary mask."""

    _validate_same_shape(prediction, target)
    if mask is None:
        return mse(prediction, target)
    prepared_mask = _prepare_mask(mask, prediction)
    denominator = _positive_mask_sum(prepared_mask, "masked_mse")
    return ((prediction - target).pow(2) * prepared_mask).sum() / denominator


def psnr(prediction: torch.Tensor, target: torch.Tensor, *, data_range: float = 1.0) -> torch.Tensor:
    error = mse(prediction, target).clamp_min(torch.finfo(prediction.dtype).eps)
    return 20 * torch.log10(torch.tensor(data_range, dtype=prediction.dtype, device=prediction.device)) - (
        10 * torch.log10(error)
    )


def masked_psnr(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    data_range: float = 1.0,
) -> torch.Tensor:
    error = masked_mse(prediction, target, mask).clamp_min(torch.finfo(prediction.dtype).eps)
    value_range = torch.tensor(data_range, dtype=prediction.dtype, device=prediction.device)
    return 20 * torch.log10(value_range) - 10 * torch.log10(error)


def normalized_cross_correlation(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Pearson-style normalized correlation for sanity checks."""

    _validate_same_shape(prediction, target)
    if mask is None:
        pred = prediction.reshape(-1)
        tgt = target.reshape(-1)
    else:
        prepared_mask = _prepare_mask(mask, prediction).bool()
        if not bool(prepared_mask.any().detach().cpu().item()):
            raise ValueError("normalized_cross_correlation mask must select at least one element.")
        pred = prediction[prepared_mask]
        tgt = target[prepared_mask]
    pred = pred - pred.mean()
    tgt = tgt - tgt.mean()
    denominator = pred.pow(2).mean().sqrt() * tgt.pow(2).mean().sqrt()
    return (pred * tgt).mean() / denominator.clamp_min(eps)


def outside_mask_mean_abs(prediction: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean absolute prediction outside a foreground mask."""

    prepared_mask = _prepare_mask(mask, prediction)
    outside = (1.0 - prepared_mask).clamp_min(0.0)
    if not bool(torch.any(outside > 0).detach().cpu().item()):
        return prediction.sum() * 0.0
    return (prediction.abs() * outside).sum() / outside.sum()


def gradient_mae(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean absolute error between finite spatial gradients."""

    _validate_same_shape(prediction, target)
    if prediction.ndim < 3:
        raise ValueError("gradient_mae expects batch, channel, and spatial dimensions.")
    prepared_mask = _prepare_mask(mask, prediction) if mask is not None else None
    losses: list[torch.Tensor] = []
    for dim in range(2, prediction.ndim):
        if int(prediction.shape[dim]) < 2:
            continue
        pred_diff = prediction.diff(dim=dim)
        target_diff = target.diff(dim=dim)
        diff_mask = _gradient_mask(prepared_mask, dim) if prepared_mask is not None else None
        losses.append(masked_mae(pred_diff, target_diff, diff_mask))
    if not losses:
        return prediction.sum() * 0.0
    return torch.stack(losses).mean()


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
    """Stable 2D structural similarity in the documented mathematical range [-1, 1]."""

    if prediction.ndim != 4:
        raise ValueError("ssim expects (B, C, H, W) tensors — this project is 2D-only.")

    return _stable_ssim(
        prediction, target, data_range=data_range, window_size=window_size, spatial_dims=2
    )


def ssim3d(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    data_range: float = 1.0,
    window_size: int = 7,
) -> torch.Tensor:
    """Stable volumetric SSIM in [-1, 1], the 3D analogue of :func:`ssim`.

    Not the official 2D Task 3 metric — used as a training-time loss term for
    spatial_dims=3 volumes, where the 2D `ssim` (avg_pool2d, 4D-only) can't apply.
    """

    if prediction.ndim != 5:
        raise ValueError("ssim3d expects (B, C, D, H, W) tensors.")

    return _stable_ssim(
        prediction, target, data_range=data_range, window_size=window_size, spatial_dims=3
    )


def _stable_ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    data_range: float,
    window_size: int,
    spatial_dims: int,
) -> torch.Tensor:
    """Uniform-window SSIM with autocast-safe moments and covariance projection.

    The former ``E[x²] - E[x]²`` implementation ran in bf16 autocast. Cancellation
    could make variances negative, break the covariance bound, and produce SSIM > 1
    (therefore a negative training loss). Moments now run in float32, variances are
    projected to nonnegative values, and covariance is projected onto the
    Cauchy-Schwarz bound before forming the two bounded SSIM factors.
    """

    _validate_same_shape(prediction, target)
    if not bool(torch.isfinite(prediction).all()) or not bool(torch.isfinite(target).all()):
        raise ValueError("SSIM inputs must contain only finite values.")
    if not data_range > 0:
        raise ValueError("SSIM data_range must be positive.")
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("SSIM window_size must be a positive odd integer.")
    pool = F.avg_pool3d if spatial_dims == 3 else F.avg_pool2d
    pad_fn = F.pad
    pad = window_size // 2
    padding = tuple(value for _ in range(spatial_dims) for value in (pad, pad))
    device_type = prediction.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        p = prediction.float()
        t = target.float()

        def local_mean(x: torch.Tensor) -> torch.Tensor:
            return pool(
                pad_fn(x, padding, mode="replicate"),
                kernel_size=window_size,
                stride=1,
            )

        mu_p = local_mean(p)
        mu_t = local_mean(t)
        var_p = (local_mean(p.square()) - mu_p.square()).clamp_min(0.0)
        var_t = (local_mean(t.square()) - mu_t.square()).clamp_min(0.0)
        covariance = local_mean(p * t) - mu_p * mu_t
        covariance_limit = (var_p * var_t).sqrt()
        covariance = torch.maximum(
            torch.minimum(covariance, covariance_limit), -covariance_limit
        )
        c1 = float(0.01 * data_range) ** 2
        c2 = float(0.03 * data_range) ** 2
        luminance = (2.0 * mu_p * mu_t + c1) / (
            mu_p.square() + mu_t.square() + c1
        )
        structure = (2.0 * covariance + c2) / (var_p + var_t + c2)
        similarity = (luminance * structure).mean()
        # Only projects final floating-point roundoff; the moment corrections above fix
        # the invalid covariance/denominator that caused the observed out-of-range value.
        similarity = similarity.clamp(-1.0, 1.0)
    if not bool(torch.isfinite(similarity)):
        raise ValueError("SSIM produced a non-finite value.")
    return similarity


def lpips_metric(
    prediction: torch.Tensor, target: torch.Tensor, *, net: torch.nn.Module | None = None
) -> torch.Tensor:
    """Perceptual distance (official MRIxFields Task 3 metric). Requires the optional `lpips` package."""

    from fieldbridge.training.losses import lpips_loss

    with torch.no_grad():
        return lpips_loss(prediction, target, net=net)


def _validate_same_shape(prediction: torch.Tensor, target: torch.Tensor) -> None:
    if prediction.shape != target.shape:
        raise ValueError(
            "prediction and target must have the same shape; "
            f"got {tuple(prediction.shape)} and {tuple(target.shape)}."
        )


def _prepare_mask(mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    prepared = mask.to(device=reference.device, dtype=reference.dtype)
    if prepared.ndim == reference.ndim - 1 and int(prepared.shape[0]) == int(reference.shape[0]):
        prepared = prepared.unsqueeze(1)
    try:
        prepared = torch.broadcast_to(prepared, reference.shape)
    except RuntimeError as exc:
        raise ValueError(
            f"mask with shape {tuple(mask.shape)} cannot broadcast to {tuple(reference.shape)}."
        ) from exc
    if not torch.isfinite(prepared).all():
        raise ValueError("mask must contain only finite values.")
    return prepared


def _positive_mask_sum(mask: torch.Tensor, metric_name: str) -> torch.Tensor:
    denominator = mask.sum()
    if not bool((denominator > 0).detach().cpu().item()):
        raise ValueError(f"{metric_name} mask must select at least one element.")
    return denominator


def _gradient_mask(mask: torch.Tensor | None, dim: int) -> torch.Tensor | None:
    if mask is None:
        return None
    before = mask.narrow(dim, 0, int(mask.shape[dim]) - 1)
    after = mask.narrow(dim, 1, int(mask.shape[dim]) - 1)
    return before * after

